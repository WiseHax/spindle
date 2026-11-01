//! Streams transformer layers from the model file with explicit reads instead of page faults.
//!
//! When the model is bigger than the RAM we can use, the first layers are pinned in memory
//! and layers `first..n` are streamed: each one is read with a single large `ReadFile` into
//! one of two private buffers. While layer `l` computes from one buffer, a background thread
//! fills the other with layer `l + 1` (or wraps to `first` for the next token).
//!
//! Why not just let mmap page them in: a cap smaller than the model makes Windows recycle our
//! pages, and every 4 KiB page then costs a fault on every token (millions per reply, most of
//! the time on a 0.6B model even when the data is still in the standby cache). One read per
//! layer replaces thousands of faults, and the read overlaps with compute.

use std::{
    fs::File,
    io::{Error, ErrorKind, Result},
    os::windows::fs::FileExt,
    sync::mpsc::{channel, Receiver, Sender},
    thread::{self, JoinHandle},
};

pub struct LayerStreamer {
    /// `(file offset, length)` of every layer; only `first..` are streamed.
    spans: Vec<(usize, usize)>,
    pub first: usize,
    req_tx: Option<Sender<(usize, Vec<u8>)>>,
    done_rx: Receiver<(usize, Result<Vec<u8>>)>,
    handle: Option<JoinHandle<()>>,
    spare: Option<Vec<u8>>,
    /// Bytes delivered so far, for stats.
    pub bytes_read: u64,
}

fn read_exact_at(file: &File, buf: &mut [u8], mut off: u64) -> Result<()> {
    let mut done = 0;
    while done < buf.len() {
        let n = file.seek_read(&mut buf[done..], off)?;
        if n == 0 {
            return Err(Error::new(ErrorKind::UnexpectedEof, "model file ended inside a layer"));
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}

fn alloc(len: usize) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|_| Error::new(ErrorKind::OutOfMemory, "layer buffer allocation failed"))?;
    v.resize(len, 0);
    Ok(v)
}

impl LayerStreamer {
    /// Starts the reader thread and immediately queues layer `first`.
    pub fn new(path: &str, spans: Vec<(usize, usize)>, first: usize) -> Result<Self> {
        assert!(first < spans.len());
        let max = spans[first..].iter().map(|s| s.1).max().unwrap_or(0);
        let (a, b) = (alloc(max)?, alloc(max)?);
        let file = File::open(path)?;
        let (req_tx, req_rx) = channel::<(usize, Vec<u8>)>();
        let (done_tx, done_rx) = channel();
        let thread_spans = spans.clone();
        let handle = thread::Builder::new().name("layer-stream".into()).spawn(move || {
            for (l, mut buf) in req_rx {
                let (off, len) = thread_spans[l];
                buf.resize(len, 0);
                // One large read per layer. Splitting it into 4 parallel reads was measured
                // and made no difference: at this point the drive itself is the limit.
                let res = read_exact_at(&file, &mut buf, off as u64).map(|_| buf);
                if done_tx.send((l, res)).is_err() {
                    break;
                }
            }
        })?;
        req_tx.send((first, a)).map_err(|_| Error::other("layer reader stopped"))?;
        Ok(Self { spans, first, req_tx: Some(req_tx), done_rx, handle: Some(handle), spare: Some(b), bytes_read: 0 })
    }

    pub fn max_buffer(&self) -> usize {
        self.spans[self.first..].iter().map(|s| s.1).max().unwrap_or(0)
    }

    fn request(&self, l: usize, buf: Vec<u8>) -> Result<()> {
        self.req_tx.as_ref().unwrap().send((l, buf)).map_err(|_| Error::other("layer reader stopped"))
    }

    /// Waits for layer `l`, queues the layer after it into the spare buffer, and returns
    /// `l`'s bytes. Give the buffer back with `give_back` once the layer is done.
    pub fn take(&mut self, l: usize) -> Result<Vec<u8>> {
        let buf = loop {
            let (got, res) = self.done_rx.recv().map_err(|_| Error::other("layer reader stopped"))?;
            let buf = res?;
            if got == l {
                break buf;
            }
            // Out of the expected order (shouldn't happen: layers always run first..n).
            self.request(l, buf)?;
        };
        let next = if l + 1 < self.spans.len() { l + 1 } else { self.first };
        let spare = self.spare.take().expect("give_back the previous layer's buffer before taking the next");
        self.request(next, spare)?;
        self.bytes_read += buf.len() as u64;
        Ok(buf)
    }

    pub fn give_back(&mut self, buf: Vec<u8>) {
        self.spare = Some(buf);
    }
}

impl Drop for LayerStreamer {
    fn drop(&mut self) {
        drop(self.req_tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivers_layers_in_cycles_with_the_right_bytes() {
        let path = std::env::temp_dir().join("spindle_layer_stream_test.bin");
        let data: Vec<u8> = (0..10_000u32).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let spans = vec![(0, 1000), (1000, 3000), (4000, 2500), (6500, 3500)];
        let mut s = LayerStreamer::new(path.to_str().unwrap(), spans.clone(), 1).unwrap();
        for _token in 0..3 {
            for l in 1..4 {
                let buf = s.take(l).unwrap();
                let (off, len) = spans[l];
                assert_eq!(&buf[..], &data[off..off + len], "layer {l}");
                s.give_back(buf);
            }
        }
        drop(s);
        let _ = std::fs::remove_file(path);
    }
}
