//! The inference engine: quantized kernels, Llama forward pass, BPE tokenizer, sampling
//! and multi-turn chat.
//!
//! Works with Llama-family GGUF models (SmolLM2 135M/360M tested) whose tensors are `F32`,
//! `Q8_0`, `Q5_0`, `Q4_K` or `Q6_K`. Sizes come from the GGUF metadata, weights are used
//! straight from the memory-mapped file, and matvecs are split across P-cores.

use std::{
    arch::x86_64::*,
    io::Write,
    time::Instant,
};

use crate::{
    gguf_loader::{GgufModelIndex, GgufTensorInfo},
    mmap_file::MappedFile,
    quant::{dequant_row, dot_f32_avx2, matvec_rows_avx2, row_bytes},
    worker_pool::WorkerPool,
};

/// Llama-family hyperparameters, read from GGUF metadata (`{arch}.embedding_length`, ...)
/// so the same engine runs SmolLM2-135M / 360M / 1.7B.
#[derive(Clone, Copy, Debug)]
pub struct ModelConfig {
    pub dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub ctx_len: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

impl ModelConfig {
    pub fn from_gguf(idx: &GgufModelIndex) -> std::io::Result<Self> {
        let arch = &idx.architecture;
        let get = |key: &str| idx.metadata.get(&format!("{arch}.{key}"));
        let req = |key: &str| {
            get(key).and_then(|v| v.as_usize()).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("GGUF metadata missing `{arch}.{key}`"),
                )
            })
        };
        let dim = req("embedding_length")?;
        let n_heads = req("attention.head_count")?;
        let n_kv_heads = get("attention.head_count_kv").and_then(|v| v.as_usize()).unwrap_or(n_heads);
        let head_dim = dim / n_heads;
        let cfg = Self {
            dim,
            ffn_dim: req("feed_forward_length")?,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim: n_kv_heads * head_dim,
            n_layers: req("block_count")?,
            vocab: idx.vocab.len(),
            ctx_len: get("context_length").and_then(|v| v.as_usize()).unwrap_or(2048),
            rope_theta: get("rope.freq_base").and_then(|v| v.as_f32()).unwrap_or(10000.0),
            rms_eps: get("attention.layer_norm_rms_epsilon").and_then(|v| v.as_f32()).unwrap_or(1e-5),
        };
        // The AVX2 attention loops step 8 floats at a time.
        if head_dim % 8 != 0 || n_heads % n_kv_heads != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported head layout: {cfg:?}"),
            ));
        }
        Ok(cfg)
    }
}

/// Below this many weights a matvec runs on one thread (sync cost > math).
const PAR_MIN_WEIGHTS: usize = 64 * 1024;

/// `out = W * x`, split by rows across `pool` (if any).
unsafe fn matvec_gguf(
    pool: Option<&WorkerPool>,
    base_ptr: *const u8,
    info: &GgufTensorInfo,
    x: &[f32],
    out: &mut [f32],
) {
    let n_rows = info.dims[1] as usize;
    debug_assert_eq!(out.len(), n_rows);
    let pool = match pool {
        Some(p) if p.n_threads() > 1 && n_rows * (info.dims[0] as usize) >= PAR_MIN_WEIGHTS => p,
        _ => return matvec_rows_avx2(base_ptr, info, x, out, 0),
    };
    // Raw pointers aren't `Sync`; pass addresses. Each thread writes a disjoint row range.
    let base_addr = base_ptr as usize;
    let out_addr = out.as_mut_ptr() as usize;
    pool.run(&|t, n| {
        let r0 = n_rows * t / n;
        let r1 = n_rows * (t + 1) / n;
        let dst = std::slice::from_raw_parts_mut((out_addr as *mut f32).add(r0), r1 - r0);
        matvec_rows_avx2(base_addr as *const u8, info, x, dst, r0);
    });
}

// -------------------------------------- SmolLM2 Transformer Forward Pass

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
    pub k_cache: Vec<f32>, // [c.n_layers * max_seq * c.kv_dim]
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
}

/// SmolLM2's official default system prompt. Giving the assistant its own name stops the
/// tiny model from adopting the user's name ("Hi, I'm Wise" -> "My name is Wise").
const SYSTEM_PROMPT: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

impl SmolLm2Engine {
    pub fn load(path: &str) -> std::io::Result<Self> {
        Self::load_with_index(path, GgufModelIndex::parse_file(path)?)
    }

    /// Like `load`, reusing an already-parsed index (the vocab + merge tables are large).
    pub fn load_with_index(path: &str, idx: GgufModelIndex) -> std::io::Result<Self> {
        let c = ModelConfig::from_gguf(&idx)?;
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
            "[model:config] dim={} ffn={} layers={} heads={}/{} kv (head_dim {}) vocab={} rope_theta={} | KV-cache {max_seq} pos = {kv_mib:.1} MiB",
            c.dim, c.ffn_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, c.vocab, c.rope_theta
        );
        Ok(Self {
            cfg: c,
            idx,
            raw_bytes,
            k_cache: vec![0.0f32; c.n_layers * max_seq * c.kv_dim],
            v_cache: vec![0.0f32; c.n_layers * max_seq * c.kv_dim],
            max_seq,
            cur_pos: 0,
            n_keep: 0,
            pool: None,
            sampler: SamplerConfig::default(),
            rng: SplitMix64(SamplerConfig::default().seed),
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

    /// Makes room for `needed` more positions by evicting the oldest non-system tokens.
    /// Kept tokens slide down by `n_discard` positions; since K was cached *after* RoPE,
    /// each kept K vector is re-rotated by `-n_discard` so its position encoding stays
    /// consistent (RoPE rotations compose: R(p - d) = R(-d) * R(p)).
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
                rope_shift(&mut k_tmp, c.n_kv_heads, c.head_dim, -(n_discard as f32), c.rope_theta);
                self.k_cache[dst..dst + c.kv_dim].copy_from_slice(&k_tmp);
                self.v_cache.copy_within(src..src + c.kv_dim, dst);
            }
        }
        self.cur_pos -= n_discard;
        eprintln!("\n[kv:shift] evicted {n_discard} old tokens (kept {} system + {n_move} recent)", self.n_keep);
    }

    /// Feeds `ids` into the KV-cache starting at `cur_pos` and returns the last hidden state.
    unsafe fn feed(&mut self, ids: &[usize]) -> Vec<f32> {
        let c = self.cfg;
        let mut last_hidden = vec![0.0f32; c.dim];
        for &tid in ids {
            self.ensure_kv_room(1);
            last_hidden = self.forward_hidden(tid, self.cur_pos);
            self.cur_pos += 1;
        }
        last_hidden
    }

    fn tensor(&self, name: &str) -> &GgufTensorInfo {
        let &i = self
            .idx
            .tensor_by_name
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        &self.idx.tensors[i]
    }

    /// Runs a single token `token_id` at sequence position `pos` through all `n_layers`
    /// Transformer layers and returns the final-normed hidden state (`dim` floats).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn forward_hidden(&mut self, token_id: usize, pos: usize) -> Vec<f32> {
        let c = self.cfg;
        let base = self.raw_bytes.as_ptr();

        // 1. Dequantize `token_embd.weight` row `token_id`
        let emb_info = self.tensor("token_embd.weight");
        let rb = row_bytes(emb_info.dtype, c.dim).expect("unsupported embedding type");
        let row = std::slice::from_raw_parts(base.add(emb_info.abs_file_offset as usize + token_id * rb), rb);
        let mut x = dequant_row(emb_info.dtype, row, c.dim);

        let mut x_norm = vec![0.0f32; c.dim];
        let mut q = vec![0.0f32; c.dim];
        let mut k = vec![0.0f32; c.kv_dim];
        let mut v = vec![0.0f32; c.kv_dim];
        let mut attn_out = vec![0.0f32; c.dim];
        let mut proj_out = vec![0.0f32; c.dim];
        let mut gate = vec![0.0f32; c.ffn_dim];
        let mut up = vec![0.0f32; c.ffn_dim];

        for l in 0..c.n_layers {
            // A. Attention RMSNorm
            let an_info = self.tensor(&format!("blk.{l}.attn_norm.weight"));
            let an_w = std::slice::from_raw_parts(
                base.add(an_info.abs_file_offset as usize) as *const f32,
                c.dim,
            );
            rmsnorm_slice(&x, an_w, &mut x_norm, c.rms_eps);

            // B. Q, K, V Projections
            let wq = self.tensor(&format!("blk.{l}.attn_q.weight"));
            let wk = self.tensor(&format!("blk.{l}.attn_k.weight"));
            let wv = self.tensor(&format!("blk.{l}.attn_v.weight"));
            matvec_gguf(self.pool.as_ref(), base, wq, &x_norm, &mut q);
            matvec_gguf(self.pool.as_ref(), base, wk, &x_norm, &mut k);
            matvec_gguf(self.pool.as_ref(), base, wv, &x_norm, &mut v);

            // C. Llama-style Rotary Positional Embeddings (RoPE, theta = 100,000.0)
            apply_rope_llama(&mut q, c.n_heads, c.head_dim, pos, c.rope_theta);
            apply_rope_llama(&mut k, c.n_kv_heads, c.head_dim, pos, c.rope_theta);

            // D. Store K, V in KV-Cache for layer `l`, position `pos`
            let kv_off = (l * self.max_seq + pos) * c.kv_dim;
            self.k_cache[kv_off..kv_off + c.kv_dim].copy_from_slice(&k);
            self.v_cache[kv_off..kv_off + c.kv_dim].copy_from_slice(&v);

            // E. Grouped-Query Attention (9 Query Heads : 3 KV Heads = 3:1 GQA)
            let scale = 1.0f32 / (c.head_dim as f32).sqrt();
            attn_out.fill(0.0);
            for h in 0..c.n_heads {
                let kvh = h / (c.n_heads / c.n_kv_heads);
                let q_head = &q[h * c.head_dim..(h + 1) * c.head_dim];
                let out_head = &mut attn_out[h * c.head_dim..(h + 1) * c.head_dim];

                // Online softmax over `0..=pos`
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

            // F. Output Projection `attn_output.weight` + Residual
            let wo = self.tensor(&format!("blk.{l}.attn_output.weight"));
            matvec_gguf(self.pool.as_ref(), base, wo, &attn_out, &mut proj_out);
            for i in 0..c.dim {
                x[i] += proj_out[i];
            }

            // G. FFN RMSNorm
            let fn_info = self.tensor(&format!("blk.{l}.ffn_norm.weight"));
            let fn_w = std::slice::from_raw_parts(
                base.add(fn_info.abs_file_offset as usize) as *const f32,
                c.dim,
            );
            rmsnorm_slice(&x, fn_w, &mut x_norm, c.rms_eps);

            // H. SwiGLU FFN (`ffn_gate`, `ffn_up`, `ffn_down`) + Residual
            let w_gate = self.tensor(&format!("blk.{l}.ffn_gate.weight"));
            let w_up = self.tensor(&format!("blk.{l}.ffn_up.weight"));
            let w_down = self.tensor(&format!("blk.{l}.ffn_down.weight"));
            matvec_gguf(self.pool.as_ref(), base, w_gate, &x_norm, &mut gate);
            matvec_gguf(self.pool.as_ref(), base, w_up, &x_norm, &mut up);

            for i in 0..c.ffn_dim {
                let g = gate[i];
                let silu = g / (1.0 + (-g).exp());
                gate[i] = silu * up[i];
            }

            matvec_gguf(self.pool.as_ref(), base, w_down, &gate, &mut proj_out);
            for i in 0..c.dim {
                x[i] += proj_out[i];
            }
        }

        // Final `output_norm.weight`
        let on_info = self.tensor("output_norm.weight");
        let on_w = std::slice::from_raw_parts(
            base.add(on_info.abs_file_offset as usize) as *const f32,
            c.dim,
        );
        rmsnorm_slice(&x, on_w, &mut x_norm, c.rms_eps);
        x_norm
    }

    /// Computes LM-head logits for `hidden` and picks the next token using `self.sampler`:
    /// repetition penalty → temperature → top-k → min-p → top-p → seeded random draw
    /// (or plain argmax when `temperature <= 0`).
    /// If `forbid_eos` is true (on the first generated token), `<|im_end|>` and `<|endoftext|>`
    /// are masked out so the assistant never returns an empty reply.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn sample_next_token(
        &mut self,
        hidden: &[f32],
        recent: &[usize],
        forbid_eos: bool,
        im_end_id: usize,
    ) -> usize {
        let c = self.cfg;
        let base = self.raw_bytes.as_ptr();
        // Separate `output.weight` if the model has one, else the tied `token_embd.weight`
        // (`[dim cols, vocab rows]`), so either way it's a plain matvec.
        let head = if self.idx.tensor_by_name.contains_key("output.weight") {
            self.tensor("output.weight")
        } else {
            self.tensor("token_embd.weight")
        };
        let mut logits = vec![0.0f32; c.vocab];
        matvec_gguf(self.pool.as_ref(), base, head, hidden, &mut logits);

        // O(1) membership instead of `recent.contains()` per vocab entry.
        let mut penalized = vec![false; c.vocab];
        for &t in recent {
            penalized[t] = true;
        }
        let sc = self.sampler;
        for (tid, logit) in logits.iter_mut().enumerate() {
            if forbid_eos && (tid == 0 || tid == 1 || tid == im_end_id) {
                *logit = f32::NEG_INFINITY;
            } else if penalized[tid] {
                *logit = if *logit > 0.0 { *logit / sc.repeat_penalty } else { *logit * sc.repeat_penalty };
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

    /// Encodes `text` using the exact GPT-2 / Llama-3 BPE merge table (`tokenizer.ggml.merges`)
    /// from the GGUF metadata, plus special token framing (`<|im_start|>`, `<|im_end|>`).
    pub fn encode_chatml(&self, user_prompt: &str) -> Vec<usize> {
        // <|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n
        let mut ids = self.encode_system();
        ids.extend(self.encode_user_turn(user_prompt));
        ids
    }

    /// `<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n`
    fn encode_system(&self) -> Vec<usize> {
        let im_start = *self.idx.token_to_id.get("<|im_start|>").unwrap_or(&1);
        let im_end = *self.idx.token_to_id.get("<|im_end|>").unwrap_or(&2);
        let mut ids = vec![im_start];
        ids.extend(self.encode_bpe_segment(&format!("system\n{SYSTEM_PROMPT}")));
        ids.push(im_end);
        ids.extend(self.encode_bpe_segment("\n"));
        ids
    }

    /// `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
    fn encode_user_turn(&self, user_prompt: &str) -> Vec<usize> {
        let im_start = *self.idx.token_to_id.get("<|im_start|>").unwrap_or(&1);
        let im_end = *self.idx.token_to_id.get("<|im_end|>").unwrap_or(&2);
        let mut ids = vec![im_start];
        ids.extend(self.encode_bpe_segment(&format!("user\n{}", user_prompt.trim())));
        ids.push(im_end);
        ids.extend(self.encode_bpe_segment("\n"));
        ids.push(im_start);
        ids.extend(self.encode_bpe_segment("assistant\n"));
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
                cur.push(ch);
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

    pub fn decode_token(&self, tid: usize) -> String {
        let raw = self
            .idx
            .vocab
            .get(tid)
            .map(String::as_str)
            .unwrap_or("");
        let mut out = String::with_capacity(raw.len());
        for ch in raw.chars() {
            match ch {
                '\u{0120}' => out.push(' '),
                '\u{010A}' => out.push('\n'),
                other => out.push(other),
            }
        }
        out
    }

    /// Generates a conversational reply to `user_prompt`, printing each token live.
    /// The KV-cache persists across calls, so earlier turns stay in context; only the new
    /// user turn is prefilled. Call `reset_kv()` to start a fresh conversation.
    pub unsafe fn generate_reply(&mut self, user_prompt: &str, max_new_tokens: usize) -> (String, f64, usize) {
        let im_end = self
            .idx
            .vocab
            .iter()
            .position(|s| s == "<|im_end|>")
            .unwrap_or(2);

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
        let mut generated = 0usize;

        for step in 0..max_new_tokens {
            // Block EOS for the first 2 tokens so the reply is never empty.
            let next_id = self.sample_next_token(&last_hidden, &recent, step < 2, im_end);
            if next_id == im_end || next_id == 0 {
                break;
            }
            let piece = self.decode_token(next_id);
            print!("{piece}");
            let _ = std::io::stdout().flush();
            reply.push_str(&piece);
            recent.push(next_id);
            generated += 1;

            last_hidden = self.feed(&[next_id]);
        }
        // Close the assistant turn in the cache (`<|im_end|>\n`) so the next user turn
        // follows proper ChatML framing, whether we stopped on EOS or hit max_new_tokens.
        let mut closing = vec![im_end];
        closing.extend(self.encode_bpe_segment("\n"));
        self.feed(&closing);
        println!();
        let secs = t0.elapsed().as_secs_f64();
        (reply, secs, turn_ids.len() + generated)
    }
}

fn rmsnorm_slice(x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0f32 / (ss / (x.len() as f32) + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

/// Standard Llama / SmolLM2 RoPE rotation (`dim = 64`, `freq_base = 100000.0`).
/// Note: GGUF `llama` models use adjacent-pair rotation `(2*d, 2*d+1)`.
fn apply_rope_llama(vec: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let p = pos as f32;
    for h in 0..n_heads {
        let base = h * head_dim;
        for d in (0..head_dim).step_by(2) {
            let freq = 1.0f32 / theta.powf((d as f32) / (head_dim as f32));
            let angle = p * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let v0 = vec[base + d];
            let v1 = vec[base + d + 1];
            vec[base + d] = v0 * cos_a - v1 * sin_a;
            vec[base + d + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

/// Rotates already-RoPE'd vectors by an extra `delta` positions (negative = shift back).
fn rope_shift(vec: &mut [f32], n_heads: usize, head_dim: usize, delta: f32, theta: f32) {
    for h in 0..n_heads {
        let base = h * head_dim;
        for d in (0..head_dim).step_by(2) {
            let freq = 1.0f32 / theta.powf((d as f32) / (head_dim as f32));
            let (sin_a, cos_a) = (delta * freq).sin_cos();
            let v0 = vec[base + d];
            let v1 = vec[base + d + 1];
            vec[base + d] = v0 * cos_a - v1 * sin_a;
            vec[base + d + 1] = v0 * sin_a + v1 * cos_a;
        }
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
        for prompt in [
            "What is Rust Programming Language",
            "what is rust",
            "hi",
        ] {
            let ids = eng.encode_chatml(prompt);
            let pieces: Vec<String> = ids
                .iter()
                .map(|&id| format!("{id}:{:?}", eng.idx.vocab[id]))
                .collect();
            println!("\nPrompt: {prompt:?}\nTokens: {}", pieces.join(", "));
            unsafe {
                let (reply, secs, toks) = eng.generate_reply(prompt, 48);
                println!("[test:stats] {toks} toks in {secs:.2}s");
                assert!(
                    !reply.trim().is_empty(),
                    "Expected non-empty reply for prompt {prompt:?}"
                );
            }
        }
    }
}
