//! The inference engine: Llama/Qwen forward pass (dense or mixture-of-experts), BPE
//! tokenizer, sampling and multi-turn chat.
//!
//! Tested end to end on SmolLM2 135M/360M. Also handles the Qwen2/Qwen3 attention variants
//! (QKV bias, per-head Q/K RMSNorm, NEOX RoPE, head_dim != dim/heads) and MoE FFNs with a
//! softmax top-k router and optional shared expert (Qwen2-MoE, Qwen3-MoE, Mixtral-style
//! `ffn_*_exps` tensors). Sizes come from the GGUF metadata, weights are used straight from
//! the memory-mapped file, and matvecs are split across P-cores.

use std::{
    arch::x86_64::*,
    collections::HashMap,
    io::{Error, ErrorKind, Write},
    time::Instant,
};

use crate::{
    gguf_loader::{GgmlType, GgufModelIndex, GgufValue},
    mmap_file::MappedFile,
    quant::{dequant_row, dot_f32_avx2, matvec_rows_avx2, row_bytes, Mat},
    worker_pool::WorkerPool,
};

fn invalid(msg: String) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

/// Hyperparameters read from GGUF metadata (`{arch}.embedding_length`, ...).
#[derive(Clone, Copy, Debug)]
pub struct ModelConfig {
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// `n_heads * head_dim`. Equal to `dim` for Llama, larger for e.g. Qwen3.
    pub q_dim: usize,
    pub kv_dim: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub ctx_len: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// NEOX-style RoPE rotates pairs `(i, i + head_dim/2)` instead of `(2i, 2i + 1)`.
    pub rope_neox: bool,
    pub n_expert: usize,
    pub n_expert_used: usize,
    /// Renormalize the selected experts' router weights to sum to 1.
    pub expert_norm: bool,
}

impl ModelConfig {
    pub fn from_gguf(idx: &GgufModelIndex) -> std::io::Result<Self> {
        let arch = idx.architecture.as_str();
        let get = |key: &str| idx.metadata.get(&format!("{arch}.{key}"));
        let num = |key: &str| get(key).and_then(|v| v.as_usize());
        let req = |key: &str| num(key).ok_or_else(|| invalid(format!("GGUF metadata missing `{arch}.{key}`")));

        let rope_neox = match arch {
            "llama" => false,
            "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" => true,
            other => return Err(invalid(format!("unsupported architecture `{other}`"))),
        };
        if num("expert_gating_func").is_some_and(|f| f != 1) {
            return Err(invalid("only softmax expert gating is supported".into()));
        }

        let dim = req("embedding_length")?;
        let n_heads = req("attention.head_count")?;
        let n_kv_heads = num("attention.head_count_kv").unwrap_or(n_heads);
        let head_dim = num("attention.key_length").unwrap_or(dim / n_heads);
        let n_expert = num("expert_count").unwrap_or(0);
        let cfg = Self {
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            q_dim: n_heads * head_dim,
            kv_dim: n_kv_heads * head_dim,
            n_layers: req("block_count")?,
            vocab: idx.vocab.len(),
            ctx_len: num("context_length").unwrap_or(2048),
            rope_theta: get("rope.freq_base").and_then(|v| v.as_f32()).unwrap_or(10000.0),
            rms_eps: get("attention.layer_norm_rms_epsilon").and_then(|v| v.as_f32()).unwrap_or(1e-5),
            rope_neox,
            n_expert,
            n_expert_used: num("expert_used_count").unwrap_or(0).min(n_expert),
            // Qwen2-MoE doesn't renormalize; Qwen3-MoE and Mixtral do.
            expert_norm: match get("expert_weights_norm") {
                Some(GgufValue::Bool(b)) => *b,
                _ => arch != "qwen2moe",
            },
        };
        // The AVX2 attention loops step 8 floats at a time.
        if head_dim % 8 != 0 || n_heads % n_kv_heads != 0 {
            return Err(invalid(format!("unsupported head layout: {cfg:?}")));
        }
        Ok(cfg)
    }
}

// ---------------------------------------------------------------- per-layer weights

/// One SwiGLU feed-forward block: `down(silu(gate x) * up x)`.
#[derive(Clone, Copy, Debug)]
struct Ffn {
    gate: Mat,
    up: Mat,
    down: Mat,
}

impl Ffn {
    fn expert(self, e: usize) -> Self {
        Self { gate: self.gate.expert(e), up: self.up.expert(e), down: self.down.expert(e) }
    }

    fn bytes(&self) -> usize {
        [self.gate, self.up, self.down]
            .iter()
            .map(|m| m.rows * row_bytes(m.dtype, m.cols).unwrap_or(0))
            .sum()
    }
}

#[derive(Clone, Copy, Debug)]
struct Moe {
    /// `[dim, n_expert]`: one logit per expert.
    router: Mat,
    /// Expert tensors viewed as expert 0; `experts.expert(e)` is expert `e`.
    experts: Ffn,
    n_expert: usize,
    n_used: usize,
    norm: bool,
    /// Always-on expert (Qwen2-MoE), optionally scaled by `sigmoid(shared_gate · x)`.
    shared: Option<Ffn>,
    shared_gate: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
enum FfnKind {
    Dense(Ffn),
    Moe(Moe),
}

/// Byte offsets (for F32 vectors) and matrix views for one transformer layer, resolved once
/// at load so the per-token path does no name lookups.
#[derive(Clone, Copy, Debug)]
struct Layer {
    attn_norm: usize,
    wq: Mat,
    wk: Mat,
    wv: Mat,
    wo: Mat,
    q_bias: Option<usize>,
    k_bias: Option<usize>,
    v_bias: Option<usize>,
    q_norm: Option<usize>,
    k_norm: Option<usize>,
    ffn_norm: usize,
    ffn: FfnKind,
}

fn resolve_layers(idx: &GgufModelIndex, c: &ModelConfig) -> std::io::Result<Vec<Layer>> {
    let find = |name: &str| idx.tensor_by_name.get(name).map(|&i| &idx.tensors[i]);
    let mat = |name: &str| find(name).map(Mat::of).ok_or_else(|| invalid(format!("missing tensor {name}")));
    // Norms and biases are small F32 vectors used in place.
    let vec_f32 = |name: &str| -> std::io::Result<Option<usize>> {
        match find(name) {
            None => Ok(None),
            Some(t) if t.dtype == GgmlType::F32 => Ok(Some(t.abs_file_offset as usize)),
            Some(t) => Err(invalid(format!("{name}: expected F32, got {:?}", t.dtype))),
        }
    };
    let need = |name: &str| vec_f32(name)?.ok_or_else(|| invalid(format!("missing tensor {name}")));

    (0..c.n_layers)
        .map(|l| {
            let n = |s: &str| format!("blk.{l}.{s}");
            let ffn = if find(&n("ffn_gate_inp.weight")).is_some() {
                let experts = Ffn {
                    gate: mat(&n("ffn_gate_exps.weight"))?,
                    up: mat(&n("ffn_up_exps.weight"))?,
                    down: mat(&n("ffn_down_exps.weight"))?,
                };
                let shared = match find(&n("ffn_gate_shexp.weight")) {
                    None => None,
                    Some(_) => Some(Ffn {
                        gate: mat(&n("ffn_gate_shexp.weight"))?,
                        up: mat(&n("ffn_up_shexp.weight"))?,
                        down: mat(&n("ffn_down_shexp.weight"))?,
                    }),
                };
                if c.n_expert == 0 || c.n_expert_used == 0 {
                    return Err(invalid("MoE tensors present but expert_count/expert_used_count missing".into()));
                }
                FfnKind::Moe(Moe {
                    router: mat(&n("ffn_gate_inp.weight"))?,
                    experts,
                    n_expert: c.n_expert,
                    n_used: c.n_expert_used,
                    norm: c.expert_norm,
                    shared,
                    shared_gate: vec_f32(&n("ffn_gate_inp_shexp.weight"))?,
                })
            } else if find(&n("ffn_gate.0.weight")).is_some() {
                return Err(invalid("old per-expert Mixtral layout (ffn_gate.N) is not supported; re-convert the GGUF".into()));
            } else {
                FfnKind::Dense(Ffn {
                    gate: mat(&n("ffn_gate.weight"))?,
                    up: mat(&n("ffn_up.weight"))?,
                    down: mat(&n("ffn_down.weight"))?,
                })
            };
            Ok(Layer {
                attn_norm: need(&n("attn_norm.weight"))?,
                wq: mat(&n("attn_q.weight"))?,
                wk: mat(&n("attn_k.weight"))?,
                wv: mat(&n("attn_v.weight"))?,
                wo: mat(&n("attn_output.weight"))?,
                q_bias: vec_f32(&n("attn_q.bias"))?,
                k_bias: vec_f32(&n("attn_k.bias"))?,
                v_bias: vec_f32(&n("attn_v.bias"))?,
                q_norm: vec_f32(&n("attn_q_norm.weight"))?,
                k_norm: vec_f32(&n("attn_k_norm.weight"))?,
                ffn_norm: need(&n("ffn_norm.weight"))?,
                ffn,
            })
        })
        .collect()
}

// ---------------------------------------------------------------- math helpers

/// Below this many weights a matvec runs on one thread (sync cost > math).
const PAR_MIN_WEIGHTS: usize = 64 * 1024;

/// `out = W * x`, split by rows across `pool` (if any).
unsafe fn matvec(pool: Option<&WorkerPool>, base: *const u8, m: &Mat, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(out.len(), m.rows);
    let n_rows = m.rows;
    let pool = match pool {
        Some(p) if p.n_threads() > 1 && n_rows * m.cols >= PAR_MIN_WEIGHTS => p,
        _ => return matvec_rows_avx2(base, m, x, out, 0),
    };
    // Raw pointers aren't `Sync`; pass addresses. Each thread writes a disjoint row range.
    let base_addr = base as usize;
    let out_addr = out.as_mut_ptr() as usize;
    pool.run(&|t, n| {
        let r0 = n_rows * t / n;
        let r1 = n_rows * (t + 1) / n;
        let dst = std::slice::from_raw_parts_mut((out_addr as *mut f32).add(r0), r1 - r0);
        matvec_rows_avx2(base_addr as *const u8, m, x, dst, r0);
    });
}

unsafe fn f32_at<'a>(base: *const u8, offset: usize, len: usize) -> &'a [f32] {
    std::slice::from_raw_parts(base.add(offset) as *const f32, len)
}

fn rmsnorm(x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

/// Per-head RMSNorm in place (Qwen3's Q/K norm). `w` has `head_dim` entries.
fn rmsnorm_heads(v: &mut [f32], w: &[f32], head_dim: usize, eps: f32) {
    let mut tmp = vec![0.0f32; head_dim];
    for head in v.chunks_mut(head_dim) {
        rmsnorm(head, w, &mut tmp, eps);
        head.copy_from_slice(&tmp);
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU block. `out` has `ffn.down.rows` entries.
unsafe fn ffn_forward(pool: Option<&WorkerPool>, base: *const u8, f: &Ffn, x: &[f32], out: &mut [f32]) {
    let n_ff = f.gate.rows;
    let mut g = vec![0.0f32; n_ff];
    let mut u = vec![0.0f32; n_ff];
    matvec(pool, base, &f.gate, x, &mut g);
    matvec(pool, base, &f.up, x, &mut u);
    for i in 0..n_ff {
        g[i] = silu(g[i]) * u[i];
    }
    matvec(pool, base, &f.down, &g, out);
}

/// Running counters for the expert traffic of the current chat turn.
#[derive(Default)]
pub struct MoeStats {
    pub tokens: usize,
    pub expert_bytes: usize,
    /// One flag per `(layer, expert)`: was it used this turn?
    pub touched: Vec<bool>,
}

/// Mixture-of-experts FFN: softmax router, top-`n_used` experts weighted by their router
/// probability (renormalized if `norm`), plus the optional shared expert.
unsafe fn moe_forward(
    pool: Option<&WorkerPool>,
    base: *const u8,
    m: &Moe,
    x: &[f32],
    out: &mut [f32],
    stats: Option<(&mut MoeStats, usize)>,
) {
    let mut probs = vec![0.0f32; m.n_expert];
    matvec(pool, base, &m.router, x, &mut probs);
    let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    probs.iter_mut().for_each(|p| *p = (*p - max).exp());
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);

    let mut chosen: Vec<usize> = (0..m.n_expert).collect();
    chosen.select_nth_unstable_by(m.n_used - 1, |&a, &b| probs[b].total_cmp(&probs[a]));
    chosen.truncate(m.n_used);
    let wsum: f32 = chosen.iter().map(|&e| probs[e]).sum();

    out.fill(0.0);
    let mut tmp = vec![0.0f32; out.len()];
    let mut stats = stats;
    for &e in &chosen {
        let w = if m.norm { probs[e] / wsum } else { probs[e] };
        let f = m.experts.expert(e);
        ffn_forward(pool, base, &f, x, &mut tmp);
        for i in 0..out.len() {
            out[i] += w * tmp[i];
        }
        if let Some((s, layer)) = stats.as_mut() {
            s.expert_bytes += f.bytes();
            s.touched[*layer * m.n_expert + e] = true;
        }
    }
    if let Some(sh) = &m.shared {
        ffn_forward(pool, base, sh, x, &mut tmp);
        let g = match m.shared_gate {
            Some(off) => {
                let w = f32_at(base, off, x.len());
                let z: f32 = w.iter().zip(x).map(|(a, b)| a * b).sum();
                1.0 / (1.0 + (-z).exp())
            }
            None => 1.0,
        };
        for i in 0..out.len() {
            out[i] += g * tmp[i];
        }
    }
}

/// Rotary position embedding with precomputed inverse frequencies.
/// Rotations compose, so `rotate(v, a)` then `rotate(v, b)` equals `rotate(v, a + b)`;
/// the KV-cache shift relies on that.
struct Rope {
    inv_freq: Vec<f32>,
    neox: bool,
}

impl Rope {
    fn new(head_dim: usize, theta: f32, neox: bool) -> Self {
        let inv_freq = (0..head_dim / 2).map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32)).collect();
        Self { inv_freq, neox }
    }

    fn rotate(&self, v: &mut [f32], head_dim: usize, pos: f32) {
        let half = head_dim / 2;
        for head in v.chunks_mut(head_dim) {
            for (i, &f) in self.inv_freq.iter().enumerate() {
                let (s, c) = (pos * f).sin_cos();
                let (a, b) = if self.neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
                let (x0, x1) = (head[a], head[b]);
                head[a] = x0 * c - x1 * s;
                head[b] = x0 * s + x1 * c;
            }
        }
    }
}

// ---------------------------------------------------------------- byte-level BPE

/// GPT-2's byte <-> printable-char table: printable bytes map to themselves, the rest to
/// chars from U+0100 up (space -> `Ġ`, newline -> `Ċ`). Vocab strings are in this alphabet.
fn byte_to_char_table() -> [char; 256] {
    let mut keep: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut chars: Vec<u32> = keep.clone();
    let mut extra = 0;
    for b in 0..256u32 {
        if !keep.contains(&b) {
            keep.push(b);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (b, c) in keep.iter().zip(&chars) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

/// Decoding settings. `temperature <= 0` means greedy (always the most likely token).
#[derive(Clone, Copy, Debug)]
pub struct SamplerConfig {
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens (0 = no limit).
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probabilities sum to `top_p` (1.0 = off).
    pub top_p: f32,
    /// Drop tokens less likely than `min_p * p(best)` (0.0 = off). Adapts to the model's
    /// confidence: strict when one token dominates, loose when many are plausible.
    pub min_p: f32,
    /// Divides positive / multiplies negative logits of tokens already in this reply.
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            repeat_penalty: 1.1,
            seed: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x5EED),
        }
    }
}

/// Small, fast, seedable PRNG (SplitMix64). Not cryptographic, and doesn't need to be.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct SmolLm2Engine {
    pub cfg: ModelConfig,
    pub idx: GgufModelIndex,
    /// The GGUF file, memory-mapped read-only (weights are used in place, never copied).
    pub raw_bytes: MappedFile,
    layers: Vec<Layer>,
    rope: Rope,
    embed: Mat,
    lm_head: Mat,
    output_norm: usize,
    pub k_cache: Vec<f32>, // [n_layers * max_seq * kv_dim]
    pub v_cache: Vec<f32>,
    pub max_seq: usize,
    pub cur_pos: usize,
    /// Number of leading KV positions (the system prompt) that are never evicted by the
    /// sliding-window shift. 0 means the system prompt has not been fed yet.
    pub n_keep: usize,
    /// Optional P-core worker pool for row-parallel matvecs (`None` = single-threaded).
    pool: Option<WorkerPool>,
    pub sampler: SamplerConfig,
    rng: SplitMix64,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    im_start: usize,
    im_end: usize,
    /// Tokens that end a reply (`<|im_end|>`, the GGUF EOS, `<|endoftext|>`).
    stop_ids: Vec<usize>,
    system_prompt: &'static str,
    pub moe_stats: MoeStats,
}

/// SmolLM2's own system prompt. Giving the assistant a name stops the tiny model from
/// adopting the user's name ("Hi, I'm Wise" -> "My name is Wise").
const SMOLLM_SYSTEM_PROMPT: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

impl SmolLm2Engine {
    pub fn load(path: &str) -> std::io::Result<Self> {
        Self::load_with_index(path, GgufModelIndex::parse_file(path)?)
    }

    /// Like `load`, reusing an already-parsed index (the vocab + merge tables are large).
    pub fn load_with_index(path: &str, idx: GgufModelIndex) -> std::io::Result<Self> {
        let c = ModelConfig::from_gguf(&idx)?;
        let layers = resolve_layers(&idx, &c)?;
        let tensor = |name: &str| idx.tensor_by_name.get(name).map(|&i| &idx.tensors[i]);
        let embed = Mat::of(tensor("token_embd.weight").ok_or_else(|| invalid("missing token_embd.weight".into()))?);
        let lm_head = tensor("output.weight").map(Mat::of).unwrap_or(embed);
        let output_norm = tensor("output_norm.weight")
            .ok_or_else(|| invalid("missing output_norm.weight".into()))?
            .abs_file_offset as usize;

        let token = |s: &str| idx.token_to_id.get(s).copied();
        let (im_start, im_end) = match (token("<|im_start|>"), token("<|im_end|>")) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err(invalid("model has no ChatML tokens (<|im_start|>/<|im_end|>)".into())),
        };
        let mut stop_ids = vec![im_end];
        let eos = idx.metadata.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_usize());
        for id in [eos, token("<|endoftext|>")].into_iter().flatten() {
            if !stop_ids.contains(&id) {
                stop_ids.push(id);
            }
        }
        let system_prompt = if idx.model_name.to_lowercase().contains("smollm") {
            SMOLLM_SYSTEM_PROMPT
        } else {
            DEFAULT_SYSTEM_PROMPT
        };

        let raw_bytes = MappedFile::open(path)?;
        let prefetched = raw_bytes.prefetch_if_fits();
        println!(
            "[model:mmap] {:.1} MiB mapped ({})",
            raw_bytes.len() as f64 / (1024.0 * 1024.0),
            if prefetched { "prefetching into page cache" } else { "paged on demand: larger than half of free RAM" }
        );

        // 1024 positions keeps the f32 KV-cache small on 8 GB RAM (135M: ~45 MiB, 1.7B: ~384 MiB).
        let max_seq = std::env::var("CHAT_CTX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024)
            .clamp(64, c.ctx_len);
        let kv_mib = (2 * c.n_layers * max_seq * c.kv_dim * 4) as f64 / (1024.0 * 1024.0);
        println!(
            "[model:config] {} | dim={} layers={} heads={}/{} kv (head_dim {}) vocab={} rope_theta={}{} | KV-cache {max_seq} pos = {kv_mib:.1} MiB",
            idx.architecture,
            c.dim,
            c.n_layers,
            c.n_heads,
            c.n_kv_heads,
            c.head_dim,
            c.vocab,
            c.rope_theta,
            if c.rope_neox { " (neox)" } else { "" }
        );
        if c.n_expert > 0 {
            let moe_layers = layers.iter().filter(|l| matches!(l.ffn, FfnKind::Moe(_))).count();
            println!(
                "[model:moe] {moe_layers} MoE layers, {} of {} experts per token{}",
                c.n_expert_used,
                c.n_expert,
                if c.expert_norm { "" } else { " (weights not renormalized)" }
            );
        }

        let byte_enc = byte_to_char_table();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, &ch)| (ch, b as u8)).collect();
        Ok(Self {
            rope: Rope::new(c.head_dim, c.rope_theta, c.rope_neox),
            cfg: c,
            idx,
            raw_bytes,
            layers,
            embed,
            lm_head,
            output_norm,
            k_cache: vec![0.0f32; c.n_layers * max_seq * c.kv_dim],
            v_cache: vec![0.0f32; c.n_layers * max_seq * c.kv_dim],
            max_seq,
            cur_pos: 0,
            n_keep: 0,
            pool: None,
            sampler: SamplerConfig::default(),
            rng: SplitMix64(SamplerConfig::default().seed),
            byte_enc,
            byte_dec,
            im_start,
            im_end,
            stop_ids,
            system_prompt,
            moe_stats: MoeStats { touched: vec![false; c.n_layers * c.n_expert], ..Default::default() },
        })
    }

    /// Splits matvecs across `n_threads` (the caller is thread 0). `pin(i)` runs on each
    /// helper `i in 1..n_threads` first, e.g. to lock it to a P-core.
    pub fn enable_threads<P: Fn(usize) + Send + Sync + 'static>(&mut self, n_threads: usize, pin: P) {
        self.pool = (n_threads > 1).then(|| WorkerPool::new(n_threads, pin));
        // Wait for every helper to finish pinning so its log lines don't interleave with the reply.
        if let Some(p) = &self.pool {
            p.run(&|_, _| {});
        }
    }

    /// Replaces the decoding settings and reseeds the RNG (same seed => same replies).
    pub fn set_sampler(&mut self, cfg: SamplerConfig) {
        self.sampler = cfg;
        self.rng = SplitMix64(cfg.seed);
    }

    /// Forgets the whole conversation (system prompt is re-fed on the next turn).
    pub fn reset_kv(&mut self) {
        self.cur_pos = 0;
        self.n_keep = 0;
    }

    pub fn is_moe(&self) -> bool {
        self.cfg.n_expert > 0
    }

    /// `(MiB of expert weights read per token, experts touched, total experts)` for the
    /// turn so far, then resets the counters.
    pub fn take_moe_stats(&mut self) -> Option<(f64, usize, usize)> {
        if !self.is_moe() {
            return None;
        }
        let s = &mut self.moe_stats;
        let per_tok = s.expert_bytes as f64 / s.tokens.max(1) as f64 / (1024.0 * 1024.0);
        let touched = s.touched.iter().filter(|&&t| t).count();
        let total = s.touched.len();
        s.tokens = 0;
        s.expert_bytes = 0;
        s.touched.fill(false);
        Some((per_tok, touched, total))
    }

    /// Makes room for `needed` more positions by evicting the oldest non-system tokens.
    /// Kept tokens slide down by `n_discard` positions; since K was cached *after* RoPE,
    /// each kept K vector is re-rotated by `-n_discard` so its position encoding stays
    /// consistent.
    pub fn ensure_kv_room(&mut self, needed: usize) {
        let c = self.cfg;
        if self.cur_pos + needed <= self.max_seq {
            return;
        }
        let n_ctx = self.cur_pos - self.n_keep;
        // Evict at least what we need, rounded up to half the window to amortize the shift.
        let want = (self.cur_pos + needed - self.max_seq).max(n_ctx / 2);
        let n_discard = want.min(n_ctx);
        let n_move = n_ctx - n_discard;

        let mut k_tmp = vec![0.0f32; c.kv_dim];
        for l in 0..c.n_layers {
            let layer = l * self.max_seq;
            for i in 0..n_move {
                let src = (layer + self.n_keep + n_discard + i) * c.kv_dim;
                let dst = (layer + self.n_keep + i) * c.kv_dim;
                k_tmp.copy_from_slice(&self.k_cache[src..src + c.kv_dim]);
                self.rope.rotate(&mut k_tmp, c.head_dim, -(n_discard as f32));
                self.k_cache[dst..dst + c.kv_dim].copy_from_slice(&k_tmp);
                self.v_cache.copy_within(src..src + c.kv_dim, dst);
            }
        }
        self.cur_pos -= n_discard;
        eprintln!("\n[kv:shift] evicted {n_discard} old tokens (kept {} system + {n_move} recent)", self.n_keep);
    }

    /// Feeds `ids` into the KV-cache starting at `cur_pos` and returns the last hidden state.
    unsafe fn feed(&mut self, ids: &[usize]) -> Vec<f32> {
        let mut last_hidden = vec![0.0f32; self.cfg.dim];
        for &tid in ids {
            self.ensure_kv_room(1);
            last_hidden = self.forward_hidden(tid, self.cur_pos);
            self.cur_pos += 1;
        }
        last_hidden
    }

    /// Runs one token at position `pos` through every layer and returns the final-normed
    /// hidden state (`dim` floats).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn forward_hidden(&mut self, token_id: usize, pos: usize) -> Vec<f32> {
        let c = self.cfg;
        let base = self.raw_bytes.as_ptr();
        let pool = self.pool.as_ref();

        let rb = row_bytes(self.embed.dtype, c.dim).expect("unsupported embedding type");
        let row = std::slice::from_raw_parts(base.add(self.embed.offset + token_id * rb), rb);
        let mut x = dequant_row(self.embed.dtype, row, c.dim);

        let mut x_norm = vec![0.0f32; c.dim];
        let mut q = vec![0.0f32; c.q_dim];
        let mut k = vec![0.0f32; c.kv_dim];
        let mut v = vec![0.0f32; c.kv_dim];
        let mut attn_out = vec![0.0f32; c.q_dim];
        let mut proj_out = vec![0.0f32; c.dim];
        let add = |dst: &mut [f32], off: Option<usize>| {
            if let Some(off) = off {
                let bias = f32_at(base, off, dst.len());
                for (d, b) in dst.iter_mut().zip(bias) {
                    *d += b;
                }
            }
        };
        if self.is_moe() {
            self.moe_stats.tokens += 1;
        }

        for l in 0..c.n_layers {
            let ly = self.layers[l];

            // Attention: norm, Q/K/V (+ bias), optional per-head Q/K norm, RoPE.
            rmsnorm(&x, f32_at(base, ly.attn_norm, c.dim), &mut x_norm, c.rms_eps);
            matvec(pool, base, &ly.wq, &x_norm, &mut q);
            matvec(pool, base, &ly.wk, &x_norm, &mut k);
            matvec(pool, base, &ly.wv, &x_norm, &mut v);
            add(&mut q, ly.q_bias);
            add(&mut k, ly.k_bias);
            add(&mut v, ly.v_bias);
            if let Some(off) = ly.q_norm {
                rmsnorm_heads(&mut q, f32_at(base, off, c.head_dim), c.head_dim, c.rms_eps);
            }
            if let Some(off) = ly.k_norm {
                rmsnorm_heads(&mut k, f32_at(base, off, c.head_dim), c.head_dim, c.rms_eps);
            }
            self.rope.rotate(&mut q, c.head_dim, pos as f32);
            self.rope.rotate(&mut k, c.head_dim, pos as f32);

            let kv_off = (l * self.max_seq + pos) * c.kv_dim;
            self.k_cache[kv_off..kv_off + c.kv_dim].copy_from_slice(&k);
            self.v_cache[kv_off..kv_off + c.kv_dim].copy_from_slice(&v);

            // Grouped-query attention with an online softmax over positions 0..=pos.
            let scale = 1.0f32 / (c.head_dim as f32).sqrt();
            attn_out.fill(0.0);
            for h in 0..c.n_heads {
                let kvh = h / (c.n_heads / c.n_kv_heads);
                let q_head = &q[h * c.head_dim..(h + 1) * c.head_dim];
                let out_head = &mut attn_out[h * c.head_dim..(h + 1) * c.head_dim];

                let mut m_prev = f32::NEG_INFINITY;
                let mut l_prev = 0.0f32;
                for t in 0..=pos {
                    let t_off = (l * self.max_seq + t) * c.kv_dim + kvh * c.head_dim;
                    let k_head = &self.k_cache[t_off..t_off + c.head_dim];
                    let v_head = &self.v_cache[t_off..t_off + c.head_dim];

                    let score = dot_f32_avx2(q_head.as_ptr(), k_head.as_ptr(), c.head_dim) * scale;
                    let m_new = m_prev.max(score);
                    let alpha = (m_prev - m_new).exp();
                    let p = (score - m_new).exp();
                    l_prev = l_prev * alpha + p;
                    m_prev = m_new;

                    // out = out * alpha + p * v
                    let (va, vp) = (_mm256_set1_ps(alpha), _mm256_set1_ps(p));
                    for d in (0..c.head_dim).step_by(8) {
                        let o = out_head.as_mut_ptr().add(d);
                        let ov = _mm256_mul_ps(_mm256_loadu_ps(o), va);
                        _mm256_storeu_ps(o, _mm256_fmadd_ps(vp, _mm256_loadu_ps(v_head.as_ptr().add(d)), ov));
                    }
                }
                let inv_l = 1.0f32 / l_prev.max(1e-9);
                for d in 0..c.head_dim {
                    out_head[d] *= inv_l;
                }
            }

            matvec(pool, base, &ly.wo, &attn_out, &mut proj_out);
            for i in 0..c.dim {
                x[i] += proj_out[i];
            }

            // Feed-forward: dense SwiGLU or mixture of experts.
            rmsnorm(&x, f32_at(base, ly.ffn_norm, c.dim), &mut x_norm, c.rms_eps);
            match &ly.ffn {
                FfnKind::Dense(f) => ffn_forward(pool, base, f, &x_norm, &mut proj_out),
                FfnKind::Moe(m) => moe_forward(pool, base, m, &x_norm, &mut proj_out, Some((&mut self.moe_stats, l))),
            }
            for i in 0..c.dim {
                x[i] += proj_out[i];
            }
        }

        rmsnorm(&x, f32_at(base, self.output_norm, c.dim), &mut x_norm, c.rms_eps);
        x_norm
    }

    /// Computes LM-head logits for `hidden` and picks the next token using `self.sampler`:
    /// repetition penalty, temperature, top-k, min-p, top-p, then a seeded random draw
    /// (or plain argmax when `temperature <= 0`). If `forbid_stop` is set, stop tokens are
    /// masked so the reply can't be empty.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn sample_next_token(&mut self, hidden: &[f32], recent: &[usize], forbid_stop: bool) -> usize {
        let c = self.cfg;
        let mut logits = vec![0.0f32; c.vocab];
        matvec(self.pool.as_ref(), self.raw_bytes.as_ptr(), &self.lm_head, hidden, &mut logits);

        // O(1) membership instead of `recent.contains()` per vocab entry.
        let mut penalized = vec![false; c.vocab];
        for &t in recent {
            penalized[t] = true;
        }
        let sc = self.sampler;
        for (tid, logit) in logits.iter_mut().enumerate() {
            if penalized[tid] {
                *logit = if *logit > 0.0 { *logit / sc.repeat_penalty } else { *logit * sc.repeat_penalty };
            }
        }
        if forbid_stop {
            for &id in &self.stop_ids {
                logits[id] = f32::NEG_INFINITY;
            }
        }

        let argmax = || {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        if sc.temperature <= 0.0 {
            return argmax();
        }

        // Candidates: top-k by logit (partial select, then sort just those k).
        let mut cand: Vec<usize> = (0..c.vocab).filter(|&t| logits[t].is_finite()).collect();
        let k = if sc.top_k == 0 { cand.len() } else { sc.top_k.min(cand.len()) };
        if k == 0 {
            return argmax();
        }
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
            cand.truncate(k);
        }
        cand.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

        // Softmax with temperature (subtract max for numerical stability).
        let max = logits[cand[0]];
        let mut probs: Vec<f32> = cand.iter().map(|&t| ((logits[t] - max) / sc.temperature).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);

        // min-p, then top-p (nucleus). Both keep at least the best token.
        let floor = sc.min_p * probs[0];
        let mut keep = probs.iter().take_while(|&&p| p >= floor).count().max(1);
        let mut cum = 0.0f32;
        for (i, &p) in probs[..keep].iter().enumerate() {
            cum += p;
            if cum >= sc.top_p {
                keep = i + 1;
                break;
            }
        }

        // Sample from the renormalized survivors.
        let total: f32 = probs[..keep].iter().sum();
        let mut r = self.rng.next_f32() * total;
        for i in 0..keep {
            r -= probs[i];
            if r <= 0.0 {
                return cand[i];
            }
        }
        cand[keep - 1]
    }

    /// `<|im_start|>system\n{prompt}<|im_end|>\n`
    fn encode_system(&self) -> Vec<usize> {
        let mut ids = vec![self.im_start];
        ids.extend(self.encode_bpe_segment(&format!("system\n{}", self.system_prompt)));
        ids.push(self.im_end);
        ids.extend(self.encode_bpe_segment("\n"));
        ids
    }

    /// `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
    fn encode_user_turn(&self, user_prompt: &str) -> Vec<usize> {
        let mut ids = vec![self.im_start];
        ids.extend(self.encode_bpe_segment(&format!("user\n{}", user_prompt.trim())));
        ids.push(self.im_end);
        ids.extend(self.encode_bpe_segment("\n"));
        ids.push(self.im_start);
        ids.extend(self.encode_bpe_segment("assistant\n"));
        ids
    }

    /// Full ChatML prompt for a single message (system + user + assistant header).
    pub fn encode_chatml(&self, user_prompt: &str) -> Vec<usize> {
        let mut ids = self.encode_system();
        ids.extend(self.encode_user_turn(user_prompt));
        ids
    }

    fn encode_bpe_segment(&self, text: &str) -> Vec<usize> {
        // Split text into GPT-2 pre-tokenization words (preserving leading space on words,
        // keeping newlines separate, and stripping '\r').
        let mut clean = String::with_capacity(text.len());
        for ch in text.chars() {
            if ch != '\r' {
                clean.push(ch);
            }
        }

        // Pre-split into chunks: each newline is its own chunk; spaces attach to the following word
        let mut chunks: Vec<String> = Vec::new();
        let mut cur = String::new();
        for ch in clean.chars() {
            if ch == '\n' {
                if !cur.is_empty() {
                    chunks.push(std::mem::take(&mut cur));
                }
                chunks.push("\u{010A}".to_string());
            } else if ch == ' ' {
                if !cur.is_empty() && !cur.ends_with('\u{0120}') {
                    chunks.push(std::mem::take(&mut cur));
                }
                cur.push('\u{0120}');
            } else {
                // Separate punctuation from alphanumeric runs just like GPT-2 regex
                let is_punct = ch.is_ascii_punctuation();
                let prev_is_punct = cur
                    .chars()
                    .last()
                    .map(|p| p != '\u{0120}' && p.is_ascii_punctuation())
                    .unwrap_or(false);
                let prev_has_alnum = cur
                    .chars()
                    .last()
                    .map(|p| p != '\u{0120}' && !p.is_ascii_punctuation())
                    .unwrap_or(false);
                if (is_punct && prev_has_alnum) || (!is_punct && prev_is_punct) {
                    chunks.push(std::mem::take(&mut cur));
                }
                // Byte-level BPE: every UTF-8 byte maps to one vocab-alphabet char.
                let mut buf = [0u8; 4];
                for &b in ch.encode_utf8(&mut buf).as_bytes() {
                    cur.push(self.byte_enc[b as usize]);
                }
            }
        }
        if !cur.is_empty() {
            chunks.push(cur);
        }

        let mut out = Vec::new();
        for word in chunks {
            // Fast path: if the whole pre-tokenized word is already a single vocab token, emit it.
            if let Some(&tid) = self.idx.token_to_id.get(&word) {
                out.push(tid);
                continue;
            }

            // Otherwise run canonical BPE merge loop using `bpe_ranks`
            let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            while parts.len() >= 2 {
                let mut best_pair_idx = None;
                let mut best_rank = usize::MAX;
                for i in 0..(parts.len() - 1) {
                    if let Some(&rank) = self
                        .idx
                        .bpe_ranks
                        .get(&(parts[i].clone(), parts[i + 1].clone()))
                    {
                        if rank < best_rank {
                            best_rank = rank;
                            best_pair_idx = Some(i);
                        }
                    }
                }
                if let Some(i) = best_pair_idx {
                    let merged = format!("{}{}", parts[i], parts[i + 1]);
                    parts[i] = merged;
                    parts.remove(i + 1);
                } else {
                    break;
                }
            }

            for p in parts {
                if let Some(&tid) = self.idx.token_to_id.get(&p) {
                    out.push(tid);
                }
            }
        }
        out
    }

    /// Raw bytes of a token (vocab strings use the GPT-2 byte alphabet). A multi-byte UTF-8
    /// character can be split across tokens, so callers join bytes before decoding.
    pub fn token_bytes(&self, tid: usize) -> Vec<u8> {
        let raw = self.idx.vocab.get(tid).map(String::as_str).unwrap_or("");
        let mut out = Vec::with_capacity(raw.len());
        for ch in raw.chars() {
            match self.byte_dec.get(&ch) {
                Some(&b) => out.push(b),
                None => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        out
    }

    /// Generates a conversational reply to `user_prompt`, printing each token live.
    /// The KV-cache persists across calls, so earlier turns stay in context; only the new
    /// user turn is prefilled. Call `reset_kv()` to start a fresh conversation.
    pub unsafe fn generate_reply(&mut self, user_prompt: &str, max_new_tokens: usize) -> (String, f64, usize) {
        print!("\n[{}]: ", self.idx.model_name);
        let _ = std::io::stdout().flush();

        let t0 = Instant::now();
        if self.n_keep == 0 {
            let sys = self.encode_system();
            self.feed(&sys);
            self.n_keep = self.cur_pos;
        }
        let mut turn_ids = self.encode_user_turn(user_prompt);
        // A single turn can never be larger than the non-system window (leave room to reply).
        let max_turn = (self.max_seq - self.n_keep) / 2;
        if turn_ids.len() > max_turn {
            turn_ids.drain(..turn_ids.len() - max_turn);
        }
        let mut last_hidden = self.feed(&turn_ids);

        let mut recent = Vec::with_capacity(max_new_tokens);
        let mut reply = String::new();
        // Bytes of a UTF-8 character that isn't complete yet.
        let mut pending: Vec<u8> = Vec::new();
        let mut generated = 0usize;

        for step in 0..max_new_tokens {
            // Block stop tokens for the first 2 tokens so the reply is never empty.
            let next_id = self.sample_next_token(&last_hidden, &recent, step < 2);
            if self.stop_ids.contains(&next_id) {
                break;
            }
            pending.extend(self.token_bytes(next_id));
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                // Incomplete trailing char: wait for more bytes. Invalid bytes: flush lossily.
                Err(e) if e.error_len().is_none() => e.valid_up_to(),
                Err(_) => pending.len(),
            };
            let piece = String::from_utf8_lossy(&pending[..valid]).into_owned();
            pending.drain(..valid);
            print!("{piece}");
            let _ = std::io::stdout().flush();
            reply.push_str(&piece);
            recent.push(next_id);
            generated += 1;

            last_hidden = self.feed(&[next_id]);
        }
        if !pending.is_empty() {
            let piece = String::from_utf8_lossy(&pending).into_owned();
            print!("{piece}");
            reply.push_str(&piece);
        }
        // Close the assistant turn in the cache (`<|im_end|>\n`) so the next user turn
        // follows proper ChatML framing, whether we stopped on EOS or hit max_new_tokens.
        let mut closing = vec![self.im_end];
        closing.extend(self.encode_bpe_segment("\n"));
        self.feed(&closing);
        println!();
        let secs = t0.elapsed().as_secs_f64();
        (reply, secs, turn_ids.len() + generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smollm2_bpe_chatml_and_rust_prompts() {
        let path = "smollm2-135m-q4_k_m.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let mut eng = SmolLm2Engine::load(path).expect("load smollm2-135m-q4_k_m.gguf");
        for prompt in ["What is Rust Programming Language", "what is rust", "hi"] {
            let ids = eng.encode_chatml(prompt);
            let pieces: Vec<String> = ids.iter().map(|&id| format!("{id}:{:?}", eng.idx.vocab[id])).collect();
            println!("\nPrompt: {prompt:?}\nTokens: {}", pieces.join(", "));
            unsafe {
                let (reply, secs, toks) = eng.generate_reply(prompt, 48);
                println!("[test:stats] {toks} toks in {secs:.2}s");
                assert!(!reply.trim().is_empty(), "Expected non-empty reply for prompt {prompt:?}");
            }
        }
    }

    /// Non-ASCII text must survive encode -> decode (it used to be dropped silently).
    #[test]
    fn non_ascii_round_trips_through_tokenizer() {
        let path = "smollm2-135m-q4_k_m.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let eng = SmolLm2Engine::load(path).expect("load");
        for text in ["Kumusta, Señor! Magandang umaga.", "naïve café 日本語 🦀", "tabs\tand\nnewlines"] {
            let ids = eng.encode_bpe_segment(text);
            let bytes: Vec<u8> = ids.iter().flat_map(|&id| eng.token_bytes(id)).collect();
            assert_eq!(String::from_utf8(bytes).unwrap(), text);
        }
    }

    #[test]
    fn byte_table_is_a_bijection() {
        let t = byte_to_char_table();
        let uniq: std::collections::HashSet<char> = t.iter().copied().collect();
        assert_eq!(uniq.len(), 256);
        assert_eq!(t[b' ' as usize], '\u{0120}');
        assert_eq!(t[b'\n' as usize], '\u{010A}');
        assert_eq!(t[b'A' as usize], 'A');
    }

    /// The textbook definitions, written independently of `Rope`.
    fn rope_reference(v: &[f32], head_dim: usize, pos: f32, theta: f32, neox: bool) -> Vec<f32> {
        let mut out = v.to_vec();
        let half = head_dim / 2;
        for h in 0..v.len() / head_dim {
            for i in 0..half {
                let ang = pos * theta.powf(-(2.0 * i as f32) / head_dim as f32);
                let (a, b) = if neox { (h * head_dim + i, h * head_dim + i + half) } else { (h * head_dim + 2 * i, h * head_dim + 2 * i + 1) };
                out[a] = v[a] * ang.cos() - v[b] * ang.sin();
                out[b] = v[a] * ang.sin() + v[b] * ang.cos();
            }
        }
        out
    }

    #[test]
    fn rope_matches_reference_and_composes() {
        let head_dim = 64;
        let v: Vec<f32> = (0..3 * head_dim).map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0).collect();
        for neox in [false, true] {
            let rope = Rope::new(head_dim, 100000.0, neox);
            let mut a = v.clone();
            rope.rotate(&mut a, head_dim, 17.0);
            for (x, y) in a.iter().zip(rope_reference(&v, head_dim, 17.0, 100000.0, neox)) {
                assert!((x - y).abs() < 1e-4, "neox={neox}: {x} vs {y}");
            }
            // Shift used by the KV-cache: rotate(p) then rotate(-d) == rotate(p - d).
            rope.rotate(&mut a, head_dim, -5.0);
            let mut b = v.clone();
            rope.rotate(&mut b, head_dim, 12.0);
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-4, "neox={neox} shift: {x} vs {y}");
            }
        }
    }

    /// Builds a small all-F32 MoE layer in memory and checks `moe_forward` against a plain
    /// loop-by-loop implementation of the same math.
    #[test]
    fn moe_forward_matches_reference() {
        let (dim, ff, n_exp, k, sh_ff) = (32usize, 24usize, 6usize, 2usize, 16usize);
        let mut buf: Vec<f32> = Vec::new();
        let mut seed = 12345u64;
        let mut push = |n: usize, buf: &mut Vec<f32>| -> usize {
            let off = buf.len() * 4;
            for _ in 0..n {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                buf.push(((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5);
            }
            off
        };
        let mat = |offset: usize, cols: usize, rows: usize| Mat { dtype: GgmlType::F32, offset, cols, rows };
        let router = mat(push(dim * n_exp, &mut buf), dim, n_exp);
        let gate = mat(push(dim * ff * n_exp, &mut buf), dim, ff);
        let up = mat(push(dim * ff * n_exp, &mut buf), dim, ff);
        let down = mat(push(ff * dim * n_exp, &mut buf), ff, dim);
        let sh = Ffn {
            gate: mat(push(dim * sh_ff, &mut buf), dim, sh_ff),
            up: mat(push(dim * sh_ff, &mut buf), dim, sh_ff),
            down: mat(push(sh_ff * dim, &mut buf), sh_ff, dim),
        };
        let sh_gate = push(dim, &mut buf);
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let base = buf.as_ptr() as *const u8;
        let w = |m: &Mat, e: usize, r: usize, c: usize| buf[m.offset / 4 + (e * m.rows + r) * m.cols + c];

        for (norm, shared) in [(true, false), (false, true)] {
            let moe = Moe {
                router,
                experts: Ffn { gate, up, down },
                n_expert: n_exp,
                n_used: k,
                norm,
                shared: shared.then_some(sh),
                shared_gate: shared.then_some(sh_gate),
            };
            let mut got = vec![0.0f32; dim];
            unsafe { moe_forward(None, base, &moe, &x, &mut got, None) };

            // Reference.
            let logits: Vec<f32> = (0..n_exp).map(|e| (0..dim).map(|c| w(&router, 0, e, c) * x[c]).sum()).collect();
            let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
            let ex: Vec<f32> = logits.iter().map(|l| (l - mx).exp()).collect();
            let s: f32 = ex.iter().sum();
            let probs: Vec<f32> = ex.iter().map(|e| e / s).collect();
            let mut order: Vec<usize> = (0..n_exp).collect();
            order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
            let top = &order[..k];
            let tsum: f32 = top.iter().map(|&e| probs[e]).sum();
            let ffn_ref = |f: &Ffn, e: usize| -> Vec<f32> {
                let n = f.gate.rows;
                let h: Vec<f32> = (0..n)
                    .map(|r| {
                        let g: f32 = (0..dim).map(|c| w(&f.gate, e, r, c) * x[c]).sum();
                        let u: f32 = (0..dim).map(|c| w(&f.up, e, r, c) * x[c]).sum();
                        g / (1.0 + (-g).exp()) * u
                    })
                    .collect();
                (0..dim).map(|r| (0..n).map(|c| w(&f.down, e, r, c) * h[c]).sum()).collect()
            };
            let mut want = vec![0.0f32; dim];
            for &e in top {
                let wt = if norm { probs[e] / tsum } else { probs[e] };
                for (o, y) in want.iter_mut().zip(ffn_ref(&Ffn { gate, up, down }, e)) {
                    *o += wt * y;
                }
            }
            if shared {
                let z: f32 = (0..dim).map(|c| buf[sh_gate / 4 + c] * x[c]).sum();
                let g = 1.0 / (1.0 + (-z).exp());
                for (o, y) in want.iter_mut().zip(ffn_ref(&sh, 0)) {
                    *o += g * y;
                }
            }
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-4, "norm={norm} shared={shared}: {a} vs {b}");
            }
        }
    }
}
