//! Hardware bandwidth probe: `spindle <file> --probe`.
//!
//! Decode speed is bounded by `bandwidth / bytes read per token`, so every design decision
//! (what stays in RAM, what streams from the SSD, how big a read should be) needs real numbers
//! from *this* machine. This measures:
//! - RAM read bandwidth (1 thread and all-thread), the ceiling for models that fit in RAM.
//! - NVMe unbuffered reads (bypassing the page cache): sequential large blocks, and random
//!   reads at expert-like (1 MiB) and page-like (4 KiB) sizes, at queue depth 1 and 8.
//!
//! RAM use is capped to a quarter of currently free memory so the probe itself never hurts
//! the laptop.

use std::{
    alloc::{alloc, dealloc, Layout},
    arch::x86_64::*,
    hint::black_box,
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use windows::{
    core::HSTRING,
    Win32::{
        Foundation::{CloseHandle, GENERIC_READ, HANDLE},
        Storage::FileSystem::{
            CreateFileW, ReadFile, SetFilePointerEx, FILE_BEGIN,
            FILE_FLAG_NO_BUFFERING, FILE_SHARE_READ, OPEN_EXISTING,
        },
    },
};

use crate::mmap_file::available_ram_bytes;

const MIB: f64 = 1024.0 * 1024.0;
const GB: f64 = 1e9;
/// Unbuffered I/O needs sector-aligned buffers, offsets and sizes; 4 KiB covers every NVMe.
const ALIGN: usize = 4096;

/// A page-aligned heap buffer (unbuffered `ReadFile` rejects unaligned memory).
struct AlignedBuf {
    ptr: *mut u8,
    layout: Layout,
}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len, ALIGN).unwrap();
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "probe: out of memory");
        Self { ptr, layout }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

pub struct ProbeResult {
    pub ram_1t_gbs: f64,
    pub ram_mt_gbs: f64,
    pub ssd_seq_gbs: f64,
}

pub fn run(path: &str) -> std::io::Result<ProbeResult> {
    let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    println!("=== spindle bandwidth probe ===");
    println!("[probe] free RAM: {:.0} MiB, logical CPUs: {threads}", available_ram_bytes() as f64 / MIB);

    // ---- RAM ----
    let ram_bytes = ((available_ram_bytes() / 4) as usize).clamp(64 << 20, 512 << 20) & !(ALIGN - 1);
    println!("\n[ram] streaming reads over a {:.0} MiB buffer (far larger than L3)", ram_bytes as f64 / MIB);
    let ram_1t = ram_read_gbs(ram_bytes, 1);
    println!("[ram]  1 thread : {ram_1t:6.1} GB/s");
    let mut ram_mt = ram_1t;
    for &t in &[4usize, threads] {
        let bw = ram_read_gbs(ram_bytes, t);
        println!("[ram] {t:2} threads: {bw:6.1} GB/s");
        ram_mt = ram_mt.max(bw);
    }

    // ---- SSD ----
    let file_len = file_len(path)?;
    println!(
        "\n[ssd] unbuffered reads (page cache bypassed) from '{path}' ({:.0} MiB)",
        file_len as f64 / MIB
    );
    if file_len < (64 << 20) {
        println!("[ssd] file too small for a meaningful test (need >= 64 MiB); skipping");
        return Ok(ProbeResult { ram_1t_gbs: ram_1t, ram_mt_gbs: ram_mt, ssd_seq_gbs: 0.0 });
    }
    let window = Duration::from_millis(1500);
    let seq = ssd_read(path, file_len, 4 << 20, 1, false, window)?;
    println!("[ssd] sequential 4 MiB  QD1 : {:6.2} GB/s", seq.gbs);
    let seq8 = ssd_read(path, file_len, 4 << 20, 8, false, window)?;
    println!("[ssd] sequential 4 MiB  QD8 : {:6.2} GB/s", seq8.gbs);
    for &(size, label) in &[(1usize << 20, "1 MiB "), (4096, "4 KiB ")] {
        for &qd in &[1usize, 8] {
            let r = ssd_read(path, file_len, size, qd, true, window)?;
            println!(
                "[ssd] random {label} QD{qd} : {:6.2} GB/s  ({:7.0} IOPS, {:6.1} us avg latency)",
                r.gbs, r.iops, r.avg_latency_us
            );
        }
    }
    let ssd_best = seq.gbs.max(seq8.gbs);

    // ---- What it means ----
    println!("\n=== Speed ceilings (tok/s <= bandwidth / bytes read per token) ===");
    println!("{:>26} | {:>10} | {:>10}", "active weights per token", "from RAM", "from SSD");
    for &gb in &[0.1f64, 0.4, 1.0, 2.0, 4.0] {
        println!("{:>23.1} GB | {:>10.1} | {:>10.2}", gb, ram_mt / gb, ssd_best / gb);
    }
    if path.ends_with(".gguf") {
        let gb = file_len as f64 / GB;
        println!(
            "\n[model] this file is {gb:.2} GB: dense decode ceiling ~{:.0} tok/s from RAM, ~{:.1} tok/s if streamed from SSD",
            ram_mt / gb,
            ssd_best / gb
        );
    }
    Ok(ProbeResult { ram_1t_gbs: ram_1t, ram_mt_gbs: ram_mt, ssd_seq_gbs: ssd_best })
}

/// Best-of-3 aggregate read bandwidth over `bytes`, split evenly across `n_threads`.
fn ram_read_gbs(bytes: usize, n_threads: usize) -> f64 {
    let buf = AlignedBuf::new(bytes);
    // Touch every page so we measure RAM, not page faults.
    unsafe { std::ptr::write_bytes(buf.ptr, 1, bytes) };
    let addr = buf.ptr as usize;
    let chunk = (bytes / n_threads) & !(ALIGN - 1);
    let mut best = 0.0f64;
    for _ in 0..3 {
        let t0 = Instant::now();
        thread::scope(|s| {
            for t in 0..n_threads {
                s.spawn(move || unsafe { black_box(sum_avx2((addr + t * chunk) as *const u8, chunk)) });
            }
        });
        let secs = t0.elapsed().as_secs_f64();
        best = best.max((chunk * n_threads) as f64 / secs / GB);
    }
    best
}

#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(p: *const u8, len: usize) -> u64 {
    // 4 independent accumulators keep enough loads in flight to saturate the memory bus.
    let (mut a, mut b, mut c, mut d) =
        (_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256());
    let mut i = 0;
    while i + 128 <= len {
        a = _mm256_add_epi64(a, _mm256_load_si256(p.add(i) as *const __m256i));
        b = _mm256_add_epi64(b, _mm256_load_si256(p.add(i + 32) as *const __m256i));
        c = _mm256_add_epi64(c, _mm256_load_si256(p.add(i + 64) as *const __m256i));
        d = _mm256_add_epi64(d, _mm256_load_si256(p.add(i + 96) as *const __m256i));
        i += 128;
    }
    let s = _mm256_add_epi64(_mm256_add_epi64(a, b), _mm256_add_epi64(c, d));
    let mut out = [0u64; 4];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, s);
    out.iter().sum()
}

fn file_len(path: &str) -> std::io::Result<u64> {
    Ok(std::fs::metadata(path)?.len())
}

struct SsdResult {
    gbs: f64,
    iops: f64,
    avg_latency_us: f64,
}

/// Reads `block`-sized aligned chunks for `window` using `qd` threads, each with its own
/// unbuffered handle (one outstanding synchronous read per thread = queue depth `qd`).
fn ssd_read(
    path: &str,
    file_len: u64,
    block: usize,
    qd: usize,
    random: bool,
    window: Duration,
) -> std::io::Result<SsdResult> {
    let n_blocks = (file_len / block as u64).max(1);
    let total_bytes = AtomicU64::new(0);
    let total_ops = AtomicU64::new(0);
    let total_lat_ns = AtomicU64::new(0);
    let t0 = Instant::now();
    let err: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::new(None);

    thread::scope(|s| {
        for t in 0..qd {
            let (total_bytes, total_ops, total_lat_ns, err) = (&total_bytes, &total_ops, &total_lat_ns, &err);
            s.spawn(move || unsafe {
                let h = match CreateFileW(
                    &HSTRING::from(path),
                    GENERIC_READ.0,
                    FILE_SHARE_READ,
                    None,
                    OPEN_EXISTING,
                    FILE_FLAG_NO_BUFFERING,
                    HANDLE::default(),
                ) {
                    Ok(h) => h,
                    Err(e) => {
                        *err.lock().unwrap() = Some(e.into());
                        return;
                    }
                };
                let buf = AlignedBuf::new(block);
                let slice = std::slice::from_raw_parts_mut(buf.ptr, block);
                // xorshift for random offsets; sequential threads interleave stripes.
                let mut rng = 0x9E3779B97F4A7C15u64 ^ ((t as u64 + 1) * 0xBF58476D1CE4E5B9);
                let mut next_seq = t as u64;
                while t0.elapsed() < window {
                    let idx = if random {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        rng % n_blocks
                    } else {
                        let i = next_seq % n_blocks;
                        next_seq += qd as u64;
                        i
                    };
                    let off = (idx * block as u64) as i64;
                    let r0 = Instant::now();
                    let mut got = 0u32;
                    let ok = SetFilePointerEx(h, off, None, FILE_BEGIN).is_ok()
                        && ReadFile(h, Some(slice), Some(&mut got), None).is_ok();
                    if !ok {
                        *err.lock().unwrap() = Some(std::io::Error::last_os_error());
                        break;
                    }
                    total_lat_ns.fetch_add(r0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    total_bytes.fetch_add(got as u64, Ordering::Relaxed);
                    total_ops.fetch_add(1, Ordering::Relaxed);
                }
                let _ = CloseHandle(h);
                drop(buf);
            });
        }
    });
    if let Some(e) = err.into_inner().unwrap() {
        return Err(e);
    }
    let secs = t0.elapsed().as_secs_f64();
    let ops = total_ops.load(Ordering::Relaxed).max(1) as f64;
    Ok(SsdResult {
        gbs: total_bytes.load(Ordering::Relaxed) as f64 / secs / GB,
        iops: ops / secs,
        avg_latency_us: total_lat_ns.load(Ordering::Relaxed) as f64 / ops / 1000.0,
    })
}

