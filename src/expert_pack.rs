//! Repacked expert layout for streaming MoE weights from disk: `<model>.spindle.pack`.
//!
//! In standard GGUF files, an MoE layer stores its expert weights as three separate 3D
//! tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`), each contiguous across all
//! experts. Fetching one expert `e` requires 3 separate disk reads at different file offsets.
//! For 8 active experts across 24 layers, that is 576 non-contiguous reads per token!
//!
//! This module repacks each expert's `(gate, up, down)` into one contiguous, 4096-byte-aligned
//! slab:
//! ```text
//! [SPNDLPK1 Header (4096 B)]
//! [Table of (layer, expert) offsets (4096-aligned)]
//! [Layer 0, Expert 0: gate + up + down + padding to 4096 B]
//! [Layer 0, Expert 1: gate + up + down + padding to 4096 B]
//! ...
//! ```
//! A single unbuffered `ReadFile` with `FILE_FLAG_NO_BUFFERING` can then fetch the entire
//! expert directly into a 4K-aligned memory buffer at near-sequential NVMe bandwidth (~2.8 GB/s).

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

pub const PACK_MAGIC: &[u8; 8] = b"SPNDLPK1";
pub const PACK_ALIGN: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct PackHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub header_bytes: u32,
    pub source_file_len: u64,
    pub n_layers: u32,
    pub n_expert: u32,
    pub gate_bytes: u64,
    pub up_bytes: u64,
    pub down_bytes: u64,
    pub expert_payload_bytes: u64,
    pub expert_slot_bytes: u64,
}

impl PackHeader {
    pub fn to_bytes(&self) -> [u8; 80] {
        let mut buf = [0u8; 80];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.header_bytes.to_le_bytes());
        buf[16..24].copy_from_slice(&self.source_file_len.to_le_bytes());
        buf[24..28].copy_from_slice(&self.n_layers.to_le_bytes());
        buf[28..32].copy_from_slice(&self.n_expert.to_le_bytes());
        buf[32..40].copy_from_slice(&self.gate_bytes.to_le_bytes());
        buf[40..48].copy_from_slice(&self.up_bytes.to_le_bytes());
        buf[48..56].copy_from_slice(&self.down_bytes.to_le_bytes());
        buf[56..64].copy_from_slice(&self.expert_payload_bytes.to_le_bytes());
        buf[64..72].copy_from_slice(&self.expert_slot_bytes.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8]) -> std::io::Result<Self> {
        if buf.len() < 80 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "pack header too short"));
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&buf[0..8]);
        if &magic != PACK_MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid pack magic"));
        }
        Ok(Self {
            magic,
            version: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            header_bytes: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
            source_file_len: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            n_layers: u32::from_le_bytes(buf[24..28].try_into().unwrap()),
            n_expert: u32::from_le_bytes(buf[28..32].try_into().unwrap()),
            gate_bytes: u64::from_le_bytes(buf[32..40].try_into().unwrap()),
            up_bytes: u64::from_le_bytes(buf[40..48].try_into().unwrap()),
            down_bytes: u64::from_le_bytes(buf[48..56].try_into().unwrap()),
            expert_payload_bytes: u64::from_le_bytes(buf[56..64].try_into().unwrap()),
            expert_slot_bytes: u64::from_le_bytes(buf[64..72].try_into().unwrap()),
        })
    }
}

/// Metadata and index for an open `.spindle.pack` file.
pub struct PackIndex {
    pub hdr: PackHeader,
    /// `offsets[layer * n_expert + expert]` gives the 4096-aligned file offset in the pack.
    pub offsets: Vec<u64>,
}

impl PackIndex {
    pub fn offset(&self, layer: usize, expert: usize) -> u64 {
        let idx = layer * (self.hdr.n_expert as usize) + expert;
        self.offsets[idx]
    }

    pub fn read_from_file(file: &mut File) -> std::io::Result<Self> {
        file.seek(SeekFrom::Start(0))?;
        let mut hdr_buf = [0u8; 80];
        file.read_exact(&mut hdr_buf)?;
        let hdr = PackHeader::from_bytes(&hdr_buf)?;

        let total_experts = (hdr.n_layers as usize) * (hdr.n_expert as usize);
        let table_bytes = total_experts * 8;
        file.seek(SeekFrom::Start(hdr.header_bytes as u64))?;
        let mut table_buf = vec![0u8; table_bytes];
        file.read_exact(&mut table_buf)?;

        let mut offsets = Vec::with_capacity(total_experts);
        for i in 0..total_experts {
            let off = u64::from_le_bytes(table_buf[i * 8..(i + 1) * 8].try_into().unwrap());
            offsets.push(off);
        }

        Ok(Self { hdr, offsets })
    }
}

/// Standard pack sidecar path: `<model.gguf>.spindle.pack`.
pub fn pack_path_for_model(model_path: &str) -> String {
    format!("{model_path}.spindle.pack")
}

/// Checks whether an existing pack sidecar file matches the source model's length and format.
pub fn is_pack_valid(model_path: &str, pack_path: &str) -> bool {
    let Ok(model_meta) = std::fs::metadata(model_path) else { return false };
    let Ok(mut f) = File::open(pack_path) else { return false };
    let Ok(idx) = PackIndex::read_from_file(&mut f) else { return false };
    idx.hdr.source_file_len == model_meta.len()
}

/// Repacks expert weights from a memory-mapped model into a `.spindle.pack` file.
///
/// Progress callback `P(layer, n_layers, bytes_written, total_bytes)` is called per layer.
pub fn repack_experts<P: FnMut(usize, usize, u64, u64)>(
    model_bytes: &[u8],
    source_file_len: u64,
    pack_dest_path: &str,
    n_layers: usize,
    n_expert: usize,
    expert_spans_fn: &dyn Fn(usize, usize) -> [(usize, usize); 3],
    mut progress: P,
) -> std::io::Result<PackIndex> {
    if n_layers == 0 || n_expert == 0 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "zero layers or experts"));
    }

    // Header records expert (0, 0)'s sizes. Quant mixes like Q4_K_M store some layers'
    // tensors in a different type (e.g. ffn_down as Q6_K), so expert sizes can vary per
    // layer. Every slot is sized for the largest expert; the forward pass takes the real
    // per-layer offsets inside a slot from the GGUF tensor shapes, not from this header.
    let [gate_span, up_span, down_span] = expert_spans_fn(0, 0);
    let gate_bytes = gate_span.1 as u64;
    let up_bytes = up_span.1 as u64;
    let down_bytes = down_span.1 as u64;
    let mut payload_bytes = 0u64;
    for l in 0..n_layers {
        for e in 0..n_expert {
            let s = expert_spans_fn(l, e);
            payload_bytes = payload_bytes.max((s[0].1 + s[1].1 + s[2].1) as u64);
        }
    }
    let slot_bytes = ((payload_bytes as usize + PACK_ALIGN - 1) & !(PACK_ALIGN - 1)) as u64;

    let total_experts = n_layers * n_expert;
    let table_bytes = total_experts * 8;
    // Align header and table to PACK_ALIGN
    let header_bytes = PACK_ALIGN as u32;
    let data_start = ((header_bytes as usize + table_bytes + PACK_ALIGN - 1) & !(PACK_ALIGN - 1)) as u64;

    let hdr = PackHeader {
        magic: *PACK_MAGIC,
        version: 1,
        header_bytes,
        source_file_len,
        n_layers: n_layers as u32,
        n_expert: n_expert as u32,
        gate_bytes,
        up_bytes,
        down_bytes,
        expert_payload_bytes: payload_bytes,
        expert_slot_bytes: slot_bytes,
    };

    let mut offsets = Vec::with_capacity(total_experts);
    for i in 0..total_experts {
        offsets.push(data_start + (i as u64) * slot_bytes);
    }

    let tmp_path = format!("{pack_dest_path}.tmp");
    let out_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp_path)?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, out_file);

    // 1. Write header (padded to PACK_ALIGN)
    let hdr_bytes = hdr.to_bytes();
    writer.write_all(&hdr_bytes)?;
    let pad_hdr = header_bytes as usize - hdr_bytes.len();
    writer.write_all(&vec![0u8; pad_hdr])?;

    // 2. Write offset table (padded to data_start)
    for &off in &offsets {
        writer.write_all(&off.to_le_bytes())?;
    }
    let pad_table = (data_start as usize) - (header_bytes as usize + table_bytes);
    if pad_table > 0 {
        writer.write_all(&vec![0u8; pad_table])?;
    }

    // 3. Write each expert slab: [gate, up, down] + padding to slot_bytes
    let total_pack_bytes = data_start + (total_experts as u64) * slot_bytes;
    let pad_zeros = vec![0u8; slot_bytes as usize];

    let mut written_bytes = data_start;
    for l in 0..n_layers {
        for e in 0..n_expert {
            let [g_span, u_span, d_span] = expert_spans_fn(l, e);
            let pad_slot = slot_bytes as usize - (g_span.1 + u_span.1 + d_span.1);
            if g_span.0 + g_span.1 > model_bytes.len()
                || u_span.0 + u_span.1 > model_bytes.len()
                || d_span.0 + d_span.1 > model_bytes.len()
            {
                let _ = std::fs::remove_file(&tmp_path);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("expert ({l}, {e}) spans exceed model file size"),
                ));
            }
            writer.write_all(&model_bytes[g_span.0..g_span.0 + g_span.1])?;
            writer.write_all(&model_bytes[u_span.0..u_span.0 + u_span.1])?;
            writer.write_all(&model_bytes[d_span.0..d_span.0 + d_span.1])?;
            if pad_slot > 0 {
                writer.write_all(&pad_zeros[..pad_slot])?;
            }
            written_bytes += slot_bytes;
        }
        progress(l + 1, n_layers, written_bytes, total_pack_bytes);
    }

    writer.flush()?;
    drop(writer);

    // Atomically rename tmp to destination
    if Path::new(pack_dest_path).exists() {
        let _ = std::fs::remove_file(pack_dest_path);
    }
    std::fs::rename(&tmp_path, pack_dest_path)?;

    Ok(PackIndex { hdr, offsets })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_header_round_trip() {
        let hdr = PackHeader {
            magic: *PACK_MAGIC,
            version: 1,
            header_bytes: 4096,
            source_file_len: 1422239776,
            n_layers: 24,
            n_expert: 32,
            gate_bytes: 524288,
            up_bytes: 524288,
            down_bytes: 524288,
            expert_payload_bytes: 1572864,
            expert_slot_bytes: 1572864,
        };
        let b = hdr.to_bytes();
        let parsed = PackHeader::from_bytes(&b).unwrap();
        assert_eq!(hdr, parsed);
    }

    #[test]
    fn repack_and_index_synthetic() {
        let n_layers = 2;
        let n_expert = 4;
        let gate_len = 128;
        let up_len = 128;
        let down_len = 256;
        let payload = gate_len + up_len + down_len;

        // Create synthetic model buffer
        let total_size = 64 * 1024;
        let mut model_bytes = vec![0u8; total_size];
        for (i, b) in model_bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        let expert_spans = move |l: usize, e: usize| -> [(usize, usize); 3] {
            let base = (l * 4 + e) * 1024;
            [
                (base, gate_len),
                (base + gate_len, up_len),
                (base + gate_len + up_len, down_len),
            ]
        };

        let temp_pack = std::env::temp_dir().join("test_repack.spindle.pack");
        let pack_path_str = temp_pack.to_str().unwrap();

        let idx = repack_experts(
            &model_bytes,
            model_bytes.len() as u64,
            pack_path_str,
            n_layers,
            n_expert,
            &expert_spans,
            |_l, _nl, _cur, _tot| {},
        )
        .unwrap();

        assert_eq!(idx.hdr.n_layers, 2);
        assert_eq!(idx.hdr.n_expert, 4);
        assert_eq!(idx.hdr.expert_payload_bytes, payload as u64);
        assert_eq!(idx.hdr.expert_slot_bytes % PACK_ALIGN as u64, 0);

        // Verify with PackIndex reading back from file
        let mut f = File::open(pack_path_str).unwrap();
        let read_idx = PackIndex::read_from_file(&mut f).unwrap();
        assert_eq!(read_idx.hdr, idx.hdr);
        assert_eq!(read_idx.offsets, idx.offsets);

        // Check content of layer 1, expert 2
        let off = read_idx.offset(1, 2);
        f.seek(SeekFrom::Start(off)).unwrap();
        let mut slot = vec![0u8; read_idx.hdr.expert_slot_bytes as usize];
        f.read_exact(&mut slot).unwrap();

        let spans = expert_spans(1, 2);
        assert_eq!(&slot[0..gate_len], &model_bytes[spans[0].0..spans[0].0 + gate_len]);
        assert_eq!(&slot[gate_len..gate_len + up_len], &model_bytes[spans[1].0..spans[1].0 + up_len]);
        assert_eq!(
            &slot[gate_len + up_len..payload],
            &model_bytes[spans[2].0..spans[2].0 + down_len]
        );

        drop(f);
        let _ = std::fs::remove_file(temp_pack);
    }

    /// Q4_K_M-style mix: layer 1's down tensor is larger than layer 0's. Packing must
    /// succeed, size every slot for the largest expert, and keep each expert's bytes intact.
    #[test]
    fn repack_mixed_expert_sizes_per_layer() {
        let (n_layers, n_expert) = (2, 3);
        let model_bytes: Vec<u8> = (0..64 * 1024).map(|i| (i % 251) as u8).collect();
        let down_len = |l: usize| if l == 0 { 256 } else { 5000 };
        let spans = move |l: usize, e: usize| -> [(usize, usize); 3] {
            let base = (l * n_expert + e) * 8192;
            [(base, 128), (base + 128, 128), (base + 256, down_len(l))]
        };
        let temp_pack = std::env::temp_dir().join("test_repack_mixed.spindle.pack");
        let p = temp_pack.to_str().unwrap();
        let idx = repack_experts(&model_bytes, model_bytes.len() as u64, p, n_layers, n_expert, &spans, |_, _, _, _| {}).unwrap();
        assert_eq!(idx.hdr.expert_payload_bytes, 256 + 5000);
        assert_eq!(idx.hdr.expert_slot_bytes, 8192);
        assert_eq!(std::fs::metadata(p).unwrap().len(), idx.offsets[0] + 6 * 8192);

        let mut f = File::open(p).unwrap();
        for l in 0..n_layers {
            for e in 0..n_expert {
                let s = spans(l, e);
                let len = s[0].1 + s[1].1 + s[2].1;
                let mut buf = vec![0u8; len];
                f.seek(SeekFrom::Start(idx.offset(l, e))).unwrap();
                f.read_exact(&mut buf).unwrap();
                assert_eq!(&buf[..], &model_bytes[s[0].0..s[0].0 + len], "expert ({l}, {e})");
            }
        }
        drop(f);
        let _ = std::fs::remove_file(temp_pack);
    }
}

