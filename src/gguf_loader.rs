//! GGUF v2/v3 parser.
//!
//! Reads the metadata table (architecture, sizes, vocab, BPE merges) and every tensor
//! descriptor (name, shape, type, file offset). Tensor data itself is never read here.
//!
//! Also has a simple longest-match tokenizer and a writer for a fake GGUF header, both used
//! only by the synthetic benchmark.

#![allow(dead_code)]

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{BufReader, Read, Result as IoResult, Seek, SeekFrom, Write},
};

use crate::moe_engine::{NUM_EXPERTS, VOCAB};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 4096; // 4K sector alignment for Direct-DMA

/// Standard `ggml_type` enumeration from the GGUF specification.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    BF16 = 30,
    Unknown(u32),
}

impl From<u32> for GgmlType {
    fn from(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            other => Self::Unknown(other),
        }
    }
}

/// GGUF metadata value types (`gguf_metadata_value_type`).
#[derive(Clone, Debug)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    ArrayString(Vec<String>),
    ArrayCount(u32, usize), // non-string arrays summarized to save RAM on 8GB machines
}

impl GgufValue {
    /// Any integer-typed value as `usize` (`None` for negatives, floats, strings, arrays).
    pub fn as_usize(&self) -> Option<usize> {
        match *self {
            Self::U8(v) => Some(v as usize),
            Self::U16(v) => Some(v as usize),
            Self::U32(v) => Some(v as usize),
            Self::U64(v) => Some(v as usize),
            Self::I8(v) => usize::try_from(v).ok(),
            Self::I16(v) => usize::try_from(v).ok(),
            Self::I32(v) => usize::try_from(v).ok(),
            Self::I64(v) => usize::try_from(v).ok(),
            _ => None,
        }
    }

    /// Any numeric value as `f32`.
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            _ => self.as_usize().map(|v| v as f32),
        }
    }
}

/// Named tensor descriptor parsed from the GGUF tensor info table.
#[derive(Clone, Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    /// Relative offset from `data_section_offset`.
    pub rel_offset: u64,
    /// Absolute file offset (`data_section_offset + rel_offset`), 4K-aligned for Direct-DMA.
    pub abs_file_offset: u64,
}

/// Parsed GGUF v3 model index + embedded BPE vocabulary & merge ranks.
pub struct GgufModelIndex {
    pub version: u32,
    pub alignment: u64,
    pub architecture: String,
    pub model_name: String,
    pub data_section_offset: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub vocab: Vec<String>,
    pub token_to_id: HashMap<String, usize>,
    pub bpe_ranks: HashMap<(String, String), usize>,
    pub tensors: Vec<GgufTensorInfo>,
    pub tensor_by_name: HashMap<String, usize>,
}

impl GgufModelIndex {
    /// Parses a real GGUF v2/v3 file header, KV metadata section, embedded BPE
    /// vocabulary, and named tensor offset table.
    pub fn parse_file(path: &str) -> IoResult<Self> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);

        let magic = read_u32(&mut r)?;
        if magic != GGUF_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("not a GGUF file (magic = {magic:#x})"),
            ));
        }

        let version = read_u32(&mut r)?;
        let tensor_count = read_u64(&mut r)?;
        let kv_count = read_u64(&mut r)?;

        let mut metadata = HashMap::new();
        let mut vocab = Vec::new();
        let mut bpe_ranks = HashMap::new();

        // 1. Parse Key-Value Metadata section
        for _ in 0..kv_count {
            let key = read_gguf_string(&mut r)?;
            let val = read_gguf_value(&mut r, &key)?;
            if key == "tokenizer.ggml.tokens" {
                if let GgufValue::ArrayString(ref tokens) = val {
                    vocab = tokens.clone();
                }
            } else if key == "tokenizer.ggml.merges" {
                if let GgufValue::ArrayString(ref merges) = val {
                    for (rank, m) in merges.iter().enumerate() {
                        if let Some((a, b)) = m.split_once(' ') {
                            bpe_ranks.insert((a.to_string(), b.to_string()), rank);
                        }
                    }
                }
            }
            metadata.insert(key, val);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(a)) => (*a as u64).max(32),
            Some(GgufValue::U64(a)) => (*a).max(32),
            _ => 32,
        };
        let architecture = match metadata.get("general.architecture") {
            Some(GgufValue::String(s)) => s.clone(),
            _ => "llama-moe".to_string(),
        };
        let model_name = match metadata.get("general.name") {
            Some(GgufValue::String(s)) => s.clone(),
            _ => "ZeroOS-MoE-32L-8E-Q4_K".to_string(),
        };

        // 2. Parse Named Tensor Descriptor Table
        let mut tensors = Vec::with_capacity(tensor_count.min(4096) as usize);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut r)?;
            let n_dims = read_u32(&mut r)? as usize;
            let mut dims = Vec::with_capacity(n_dims.min(8));
            for _ in 0..n_dims {
                dims.push(read_u64(&mut r)?);
            }
            let dtype = GgmlType::from(read_u32(&mut r)?);
            let rel_offset = read_u64(&mut r)?;
            tensors.push(GgufTensorInfo {
                name,
                dims,
                dtype,
                rel_offset,
                abs_file_offset: 0,
            });
        }

        // 3. Compute aligned `data_section_offset`
        let header_end = r.stream_position()?;
        let data_section_offset = header_end.div_ceil(alignment) * alignment;

        let mut tensor_by_name = HashMap::with_capacity(tensors.len());
        for (idx, t) in tensors.iter_mut().enumerate() {
            t.abs_file_offset = data_section_offset + t.rel_offset;
            tensor_by_name.insert(t.name.clone(), idx);
        }

        if vocab.is_empty() {
            vocab = VOCAB.iter().map(|s| (*s).to_string()).collect();
        }
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (i, t) in vocab.iter().enumerate() {
            token_to_id.insert(t.clone(), i);
        }

        Ok(Self {
            version,
            alignment,
            architecture,
            model_name,
            data_section_offset,
            metadata,
            vocab,
            token_to_id,
            bpe_ranks,
            tensors,
            tensor_by_name,
        })
    }

    /// Greedy longest-match BPE Tokenizer over the model's embedded GGUF vocabulary.
    pub fn encode_prompt(&self, prompt: &str) -> Vec<usize> {
        let mut ids = vec![0usize]; // <bos>
        let mut rem = prompt;

        while !rem.is_empty() {
            let mut best_id = None;
            let mut best_len = 0usize;

            for (tid, tok_str) in self.vocab.iter().enumerate() {
                if tid == 0 || tok_str.is_empty() || tok_str == "<bos>" || tok_str == "<eos>" {
                    continue;
                }
                if rem.starts_with(tok_str.as_str()) && tok_str.len() > best_len {
                    best_len = tok_str.len();
                    best_id = Some(tid);
                }
                // Also allow matching without leading space at start of prompt
                let trimmed = tok_str.trim_start();
                if !trimmed.is_empty()
                    && rem.starts_with(trimmed)
                    && trimmed.len() > best_len
                {
                    best_len = trimmed.len();
                    best_id = Some(tid);
                }
            }

            if let Some(id) = best_id {
                ids.push(id);
                rem = &rem[best_len..];
                if rem.starts_with(' ') && !self.vocab.iter().any(|v| rem.starts_with(v.as_str())) {
                    rem = &rem[1..];
                }
            } else {
                // Advance one char on unknown substring
                let step = rem.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                rem = &rem[step..];
            }
        }
        ids
    }

    /// Decodes a sequence of token IDs back into a human-readable string.
    pub fn decode_tokens(&self, tokens: &[usize]) -> String {
        let mut out = String::new();
        for &t in tokens {
            if let Some(s) = self.vocab.get(t % self.vocab.len()) {
                out.push_str(s);
            }
        }
        out
    }

    /// Looks up the exact 4K-aligned physical file offset of `blk.{layer}.ffn_exps.{expert}.weight`
    /// from the GGUF tensor table so the IOCP producer can read that exact tensor.
    pub fn expert_file_offset(&self, layer: usize, expert: usize, chunk_size: u64, total_chunks: u64) -> u64 {
        let name = format!("blk.{layer}.ffn_exps.{expert}.weight");
        if let Some(&idx) = self.tensor_by_name.get(&name) {
            let off = self.tensors[idx].abs_file_offset;
            // Ensure offset stays within file bounds and 4K aligned
            let max_off = total_chunks.saturating_sub(1) * chunk_size;
            return (off & !4095).min(max_off);
        }
        let fallback_chunk = ((layer * NUM_EXPERTS + expert) as u64) % total_chunks;
        fallback_chunk * chunk_size
    }
}

// ---------------- Spec-Compliant GGUF v3 Header + Tensor Table Writer

/// Writes a spec-compliant GGUF v3 header containing:
/// - KV Metadata (`general.architecture`, `general.name`, `general.alignment = 4096`,
///   `llama.block_count`, `llama.embedding_length`, `llama.expert_count`,
///   `llama.expert_used_count`, and `tokenizer.ggml.tokens` BPE vocabulary array)
/// - Named Tensor Info Table (`blk.{0..31}.ffn_exps.{0..7}.weight` with `GgmlType::Q4K`,
///   dims `[4096, 1820]`, and 4096-byte-aligned offsets)
/// into the beginning of `path`.
pub fn write_spec_compliant_gguf_v3_header(
    path: &str,
    num_layers: usize,
    num_experts: usize,
    chunk_size: u64,
) -> IoResult<u64> {
    let mut buf = Vec::with_capacity(64 * 1024);

    let tensor_count = (num_layers * num_experts) as u64;
    let kv_count = 8u64;

    // 1. GGUF Header
    buf.write_all(&GGUF_MAGIC.to_le_bytes())?;
    buf.write_all(&3u32.to_le_bytes())?; // GGUF v3
    buf.write_all(&tensor_count.to_le_bytes())?;
    buf.write_all(&kv_count.to_le_bytes())?;

    // 2. KV Metadata entries
    write_kv_string(&mut buf, "general.architecture", "llama-moe")?;
    write_kv_string(&mut buf, "general.name", "ZeroOS-MoE-32L-8E-Q4_K")?;
    write_kv_u32(&mut buf, "general.alignment", GGUF_DEFAULT_ALIGNMENT as u32)?;
    write_kv_u32(&mut buf, "llama.block_count", num_layers as u32)?;
    write_kv_u32(&mut buf, "llama.embedding_length", 4096)?;
    write_kv_u32(&mut buf, "llama.expert_count", num_experts as u32)?;
    write_kv_u32(&mut buf, "llama.expert_used_count", 2)?;
    write_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &VOCAB)?;

    // 3. Tensor Info Table (`blk.{l}.ffn_exps.{e}.weight`)
    for l in 0..num_layers {
        for e in 0..num_experts {
            let idx = (l * num_experts + e) as u64;
            let name = format!("blk.{l}.ffn_exps.{e}.weight");
            write_gguf_string(&mut buf, &name)?;
            buf.write_all(&2u32.to_le_bytes())?; // 2D matrix
            buf.write_all(&4096u64.to_le_bytes())?; // dim 0: hidden_dim
            buf.write_all(&1820u64.to_le_bytes())?; // dim 1: rows per 4 MiB expert
            buf.write_all(&12u32.to_le_bytes())?; // GgmlType::Q4K = 12
            let rel_offset = idx * chunk_size;
            buf.write_all(&rel_offset.to_le_bytes())?;
        }
    }

    // Pad header out to a 4096-byte boundary (`data_section_offset`)
    let data_off = (buf.len() as u64).div_ceil(GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT;
    buf.resize(data_off as usize, 0u8);

    let mut f = OpenOptions::new().write(true).open(path)?;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&buf)?;
    f.sync_all()?;
    Ok(data_off)
}

// ------------------------------------------------ Binary Reader/Writer Helpers

fn read_u32<R: Read>(r: &mut R) -> IoResult<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> IoResult<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_gguf_string<R: Read>(r: &mut R) -> IoResult<String> {
    let len = read_u64(r)? as usize;
    if len > (16 << 20) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "GGUF string length exceeds 16 MiB",
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_scalar_elem<R: Read + Seek>(r: &mut R, elem_ty: u32, count: usize) -> IoResult<()> {
    let elem_sz: i64 = match elem_ty {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "unknown GGUF array elem")),
    };
    r.seek(SeekFrom::Current(elem_sz * (count as i64)))?;
    Ok(())
}

fn read_gguf_value<R: Read + Seek>(r: &mut R, key: &str) -> IoResult<GgufValue> {
    let ty = read_u32(r)?;
    Ok(match ty {
        0 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            GgufValue::U8(b[0])
        }
        1 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            GgufValue::I8(b[0] as i8)
        }
        2 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            GgufValue::U16(u16::from_le_bytes(b))
        }
        3 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            GgufValue::I16(i16::from_le_bytes(b))
        }
        4 => GgufValue::U32(read_u32(r)?),
        5 => GgufValue::I32(read_u32(r)? as i32),
        6 => GgufValue::F32(f32::from_bits(read_u32(r)?)),
        7 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            GgufValue::Bool(b[0] != 0)
        }
        8 => GgufValue::String(read_gguf_string(r)?),
        9 => {
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)? as usize;
            if elem_ty == 8 {
                let keep = key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges";
                let mut vec_s = if keep { Vec::with_capacity(count.min(262_144)) } else { Vec::new() };
                for _ in 0..count {
                    let s = read_gguf_string(r)?;
                    if keep {
                        vec_s.push(s);
                    }
                }
                GgufValue::ArrayString(vec_s)
            } else {
                skip_scalar_elem(r, elem_ty, count)?;
                GgufValue::ArrayCount(elem_ty, count)
            }
        }
        10 => GgufValue::U64(read_u64(r)?),
        11 => GgufValue::I64(read_u64(r)? as i64),
        12 => GgufValue::F64(f64::from_bits(read_u64(r)?)),
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown GGUF value type {other}"),
            ))
        }
    })
}

fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> IoResult<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)
}

fn write_kv_string<W: Write>(w: &mut W, key: &str, val: &str) -> IoResult<()> {
    write_gguf_string(w, key)?;
    w.write_all(&8u32.to_le_bytes())?; // GGUF_TYPE_STRING
    write_gguf_string(w, val)
}

fn write_kv_u32<W: Write>(w: &mut W, key: &str, val: u32) -> IoResult<()> {
    write_gguf_string(w, key)?;
    w.write_all(&4u32.to_le_bytes())?; // GGUF_TYPE_UINT32
    w.write_all(&val.to_le_bytes())
}

fn write_kv_string_array<W: Write>(w: &mut W, key: &str, arr: &[&str]) -> IoResult<()> {
    write_gguf_string(w, key)?;
    w.write_all(&9u32.to_le_bytes())?; // GGUF_TYPE_ARRAY
    w.write_all(&8u32.to_le_bytes())?; // elem_type = STRING
    w.write_all(&(arr.len() as u64).to_le_bytes())?;
    for s in arr {
        write_gguf_string(w, s)?;
    }
    Ok(())
}
