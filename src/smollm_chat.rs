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
    io::{Error, ErrorKind},
    time::Instant,
};

use crate::{
    gguf_loader::{GgmlType, GgufModelIndex, GgufValue},
    mmap_file::{MappedFile, Prefetcher},
    quant::{dequant_row, dot_f16_row_avx2, f16_to_f32_slice, f32_to_f16_slice, matvec_rows_avx2, row_bytes, Mat},
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
    /// Granite multipliers (1.0 elsewhere): input embeddings are scaled by `embedding_scale`,
    /// attention and FFN outputs by `residual_scale` before the residual add, and logits are
    /// divided by `logit_scale`. `attn_scale` replaces `1/sqrt(head_dim)` when set.
    pub embedding_scale: f32,
    pub residual_scale: f32,
    pub logit_scale: f32,
    pub attn_scale: Option<f32>,
}

impl ModelConfig {
    pub fn from_gguf(idx: &GgufModelIndex) -> std::io::Result<Self> {
        let arch = idx.architecture.as_str();
        let get = |key: &str| idx.metadata.get(&format!("{arch}.{key}"));
        let num = |key: &str| get(key).and_then(|v| v.as_usize());
        let req = |key: &str| num(key).ok_or_else(|| invalid(format!("GGUF metadata missing `{arch}.{key}`")));

        let rope_neox = match arch {
            "llama" | "granite" | "granitemoe" => false,
            "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" | "olmoe" => true,
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
            // Qwen2-MoE and OLMoE don't renormalize; Qwen3-MoE, Mixtral and Granite do.
            expert_norm: match get("expert_weights_norm") {
                Some(GgufValue::Bool(b)) => *b,
                _ => !matches!(arch, "qwen2moe" | "olmoe"),
            },
            embedding_scale: get("embedding_scale").and_then(|v| v.as_f32()).unwrap_or(1.0),
            residual_scale: get("residual_scale").and_then(|v| v.as_f32()).unwrap_or(1.0),
            logit_scale: get("logit_scale").and_then(|v| v.as_f32()).unwrap_or(1.0),
            attn_scale: get("attention.scale").and_then(|v| v.as_f32()),
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

/// `(file offset, byte length)` of a matrix, for prefetch hints.
fn span(m: &Mat) -> (usize, usize) {
    (m.offset, m.rows * row_bytes(m.dtype, m.cols).unwrap_or(0))
}

impl Ffn {
    fn spans(&self) -> [(usize, usize); 3] {
        [span(&self.gate), span(&self.up), span(&self.down)]
    }

    fn expert(self, e: usize) -> Self {
        Self { gate: self.gate.expert(e), up: self.up.expert(e), down: self.down.expert(e) }
    }

    fn bytes(&self) -> usize {
        self.spans().iter().map(|s| s.1).sum()
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
    /// Relative probability cutoff for dynamic MoE expert pruning (0.0 = disabled).
    moe_prune: f32,
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
    /// Q/K norm weights cover the whole projection (OLMoE) instead of one head (Qwen3).
    qk_norm_full: bool,
    ffn_norm: usize,
    ffn: FfnKind,
}

fn resolve_layers(idx: &GgufModelIndex, c: &ModelConfig, layer_range: std::ops::Range<usize>) -> std::io::Result<Vec<Layer>> {
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

    layer_range
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
                    moe_prune: 0.0,
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
                qk_norm_full: find(&n("attn_q_norm.weight")).is_some_and(|t| t.dims[0] as usize == c.q_dim && c.q_dim != c.head_dim),
                ffn_norm: need(&n("ffn_norm.weight"))?,
                ffn,
            })
        })
        .collect()
}

// ---------------------------------------------------------------- math helpers

/// Below this many weights a matvec runs on one thread (sync cost > math).
/// With zero-syscall atomic spin workers, parallelizing down to 16K weights yields near-linear speedups.
const PAR_MIN_WEIGHTS: usize = 16 * 1024;

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

/// `base` may be a streamed layer buffer shifted back by the layer's file offset, so the
/// offset is applied with `wrapping_add` (the result is always inside the buffer).
unsafe fn f32_at<'a>(base: *const u8, offset: usize, len: usize) -> &'a [f32] {
    std::slice::from_raw_parts(base.wrapping_add(offset) as *const f32, len)
}

unsafe fn add_bias(base: *const u8, dst: &mut [f32], off: Option<usize>) {
    if let Some(off) = off {
        let bias = f32_at(base, off, dst.len());
        for (d, b) in dst.iter_mut().zip(bias) {
            *d += b;
        }
    }
}

/// File span `(start, end)` of everything one layer reads, and the bytes actually used.
fn layer_extent(ly: &Layer, c: &ModelConfig) -> (usize, usize, usize) {
    let mut parts = vec![
        (ly.attn_norm, c.dim * 4),
        (ly.ffn_norm, c.dim * 4),
        span(&ly.wq),
        span(&ly.wk),
        span(&ly.wv),
        span(&ly.wo),
    ];
    for (off, len) in [(ly.q_bias, c.q_dim), (ly.k_bias, c.kv_dim), (ly.v_bias, c.kv_dim)] {
        parts.extend(off.map(|o| (o, len * 4)));
    }
    let (qn, kn) = if ly.qk_norm_full { (c.q_dim, c.kv_dim) } else { (c.head_dim, c.head_dim) };
    for (off, len) in [(ly.q_norm, qn), (ly.k_norm, kn)] {
        parts.extend(off.map(|o| (o, len * 4)));
    }
    match &ly.ffn {
        FfnKind::Dense(f) => parts.extend(f.spans()),
        FfnKind::Moe(m) => {
            parts.push(span(&m.router));
            // All experts: a whole MoE layer is far too big to stream as one block.
            for e in 0..m.n_expert {
                parts.extend(m.experts.expert(e).spans());
            }
        }
    }
    let start = parts.iter().map(|p| p.0).min().unwrap_or(0);
    let end = parts.iter().map(|p| p.0 + p.1).max().unwrap_or(0);
    (start, end, parts.iter().map(|p| p.1).sum())
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

/// Detailed timing breakdown of MoE operations for a turn.
#[derive(Default, Clone, Copy, Debug)]
pub struct MoeTiming {
    pub router_us: u64,
    pub cache_lookup_us: u64,
    pub disk_read_us: u64,
    pub expert_matvec_us: u64,
}

/// Running counters for the expert traffic of the current chat turn.
#[derive(Default)]
pub struct MoeStats {
    pub tokens: usize,
    pub expert_bytes: usize,
    /// One flag per `(layer, expert)`: was it used this turn?
    pub touched: Vec<bool>,
    pub timing: MoeTiming,
}

/// Multi-token Mixture-of-Experts FFN: runs the router for every token in `xs`, takes the
/// deduplicated union of chosen experts across the batch so shared experts are fetched from
/// NVMe / `ExpertCache` only once per layer, and overlaps P-core hit matvecs with E-core reads.
unsafe fn moe_forward_batch(
    pool: Option<&WorkerPool>,
    base: *const u8,
    m: &Moe,
    xs: &[Vec<f32>],
    outs: &mut [Vec<f32>],
    mut stats: Option<(&mut MoeStats, usize)>,
    prefetch: Option<&Prefetcher>,
    mut cache: Option<&mut crate::expert_cache::ExpertCache>,
) {
    let bsz = xs.len();
    if bsz == 0 {
        return;
    }
    if bsz == 1 {
        let mut tmp = vec![0.0f32; outs[0].len()];
        moe_forward(pool, base, m, &xs[0], &mut outs[0], stats, prefetch, cache, &mut tmp);
        return;
    }

    let t_router = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
    let mut probs_batch = vec![vec![0.0f32; m.n_expert]; bsz];
    let mut chosen_batch = Vec::with_capacity(bsz);
    let mut wsum_batch = vec![0.0f32; bsz];
    let mut unique_experts = Vec::new();

    for b in 0..bsz {
        let probs = &mut probs_batch[b];
        matvec(pool, base, &m.router, &xs[b], probs);
        let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        probs.iter_mut().for_each(|p| *p = (*p - max).exp());
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);

        let mut chosen: Vec<usize> = (0..m.n_expert).collect();
        chosen.select_nth_unstable_by(m.n_used - 1, |&a, &b| probs[b].total_cmp(&probs[a]));
        chosen.truncate(m.n_used);
        chosen.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

        let mut num_keep = m.n_used;
        if m.moe_prune > 0.0 && m.n_used > 2 {
            let max_p = probs[chosen[0]];
            let min_keep = 2.max(m.n_used / 2);
            let cutoff = max_p * m.moe_prune;
            let mut k = min_keep;
            while k < m.n_used && probs[chosen[k]] >= cutoff {
                k += 1;
            }
            num_keep = k;
        }
        chosen.truncate(num_keep);
        wsum_batch[b] = chosen.iter().map(|&e| probs[e]).sum();
        for &e in &chosen {
            if !unique_experts.contains(&e) {
                unique_experts.push(e);
            }
        }
        chosen_batch.push(chosen);
    }
    let router_us = t_router.map_or(0, |t| t.elapsed().as_micros() as u64);

    if let (Some(pf), None) = (prefetch, cache.as_ref()) {
        pf.hint(unique_experts.iter().flat_map(|&e| m.experts.expert(e).spans()).collect());
    }

    let layer = stats.as_ref().map(|s| s.1).unwrap_or(0);
    let (lookup_us, pending_misses) = if let Some(c) = cache.as_mut() {
        c.begin_layer(layer, &unique_experts)
    } else {
        (0, Vec::new())
    };

    if let (Some(pf), Some(c)) = (prefetch, cache.as_ref()) {
        let bypassed: Vec<(usize, usize)> = unique_experts
            .iter()
            .copied()
            .filter(|&e| c.get_expert_ptr(layer, e).is_none() && !pending_misses.iter().any(|&(me, _)| me == e))
            .flat_map(|e| m.experts.expert(e).spans())
            .collect();
        if !bypassed.is_empty() {
            pf.hint(bypassed);
        }
    }

    for out in outs.iter_mut() {
        out.fill(0.0);
    }
    let dim = outs[0].len();
    let run_expert_for_token = |e: usize, c_ptr: Option<*const u8>, x: &[f32], dst: &mut [f32]| unsafe {
        let f = m.experts.expert(e);
        let [g_span, u_span, _d_span] = f.spans();
        if let Some(c_ptr) = c_ptr {
            let packed_f = Ffn {
                gate: Mat { offset: 0, ..f.gate },
                up: Mat { offset: g_span.1, ..f.up },
                down: Mat { offset: g_span.1 + u_span.1, ..f.down },
            };
            ffn_forward(pool, c_ptr, &packed_f, x, dst);
        } else {
            ffn_forward(pool, base, &f, x, dst);
        }
    };

    let mut expert_outs = vec![vec![vec![0.0f32; dim]; m.n_used]; bsz];
    let mut computed = vec![vec![false; m.n_used]; bsz];

    let (disk_us, matvec_us) = if !pending_misses.is_empty() {
        let t_hit = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
        for b in 0..bsz {
            for (idx, &e) in chosen_batch[b].iter().enumerate() {
                if let Some(c_ptr) = cache.as_ref().and_then(|c| c.get_expert_ptr(layer, e)) {
                    computed[b][idx] = true;
                    run_expert_for_token(e, Some(c_ptr), &xs[b], &mut expert_outs[b][idx]);
                }
            }
        }
        let hit_us = t_hit.map_or(0, |t| t.elapsed().as_micros() as u64);

        let mut disk_us = 0u64;
        let mut miss_us = 0u64;
        for (miss_idx, &(e, _)) in pending_misses.iter().enumerate() {
            disk_us += cache.as_mut().map_or(0, |c| c.wait_one_miss(layer, &pending_misses, miss_idx));
            let c_ptr = cache.as_ref().and_then(|c| c.get_expert_ptr(layer, e));
            let t_m = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
            for b in 0..bsz {
                if let Some(idx) = chosen_batch[b].iter().position(|&ce| ce == e) {
                    if !computed[b][idx] {
                        computed[b][idx] = true;
                        run_expert_for_token(e, c_ptr, &xs[b], &mut expert_outs[b][idx]);
                    }
                }
            }
            miss_us += t_m.map_or(0, |t| t.elapsed().as_micros() as u64);
        }
        let t_rem = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
        for b in 0..bsz {
            for (idx, &e) in chosen_batch[b].iter().enumerate() {
                if !computed[b][idx] {
                    let c_ptr = cache.as_ref().and_then(|c| c.get_expert_ptr(layer, e));
                    run_expert_for_token(e, c_ptr, &xs[b], &mut expert_outs[b][idx]);
                }
            }
        }
        miss_us += t_rem.map_or(0, |t| t.elapsed().as_micros() as u64);
        (disk_us, hit_us + miss_us)
    } else {
        let t_matvec = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
        for b in 0..bsz {
            for (idx, &e) in chosen_batch[b].iter().enumerate() {
                let c_ptr = cache.as_ref().and_then(|c| c.get_expert_ptr(layer, e));
                run_expert_for_token(e, c_ptr, &xs[b], &mut expert_outs[b][idx]);
            }
        }
        (0, t_matvec.map_or(0, |t| t.elapsed().as_micros() as u64))
    };

    let t_acc = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
    for b in 0..bsz {
        for (idx, &e) in chosen_batch[b].iter().enumerate() {
            let w = if m.norm { probs_batch[b][e] / wsum_batch[b] } else { probs_batch[b][e] };
            let tmp = &expert_outs[b][idx];
            for i in 0..dim {
                outs[b][i] += w * tmp[i];
            }
            if let Some((s, _)) = stats.as_mut() {
                s.expert_bytes += m.experts.expert(e).bytes();
                s.touched[layer * m.n_expert + e] = true;
            }
        }
    }

    if let Some(sh) = &m.shared {
        let mut tmp = vec![0.0f32; dim];
        for b in 0..bsz {
            ffn_forward(pool, base, sh, &xs[b], &mut tmp);
            let g = match m.shared_gate {
                Some(off) => {
                    let w = f32_at(base, off, xs[b].len());
                    let z: f32 = w.iter().zip(&xs[b]).map(|(a, x)| a * x).sum();
                    1.0 / (1.0 + (-z).exp())
                }
                None => 1.0,
            };
            for i in 0..dim {
                outs[b][i] += g * tmp[i];
            }
        }
    }
    let sh_us = t_acc.map_or(0, |t| t.elapsed().as_micros() as u64);

    if let Some((s, _)) = stats {
        s.timing.router_us += router_us;
        s.timing.cache_lookup_us += lookup_us;
        s.timing.disk_read_us += disk_us;
        s.timing.expert_matvec_us += matvec_us + sh_us;
    }
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
    prefetch: Option<&Prefetcher>,
    mut cache: Option<&mut crate::expert_cache::ExpertCache>,
    tmp: &mut [f32],
) {
    let t_router = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
    let mut stack_probs = [0.0f32; 256];
    let mut heap_probs;
    let probs: &mut [f32] = if m.n_expert <= 256 {
        &mut stack_probs[..m.n_expert]
    } else {
        heap_probs = vec![0.0f32; m.n_expert];
        &mut heap_probs
    };

    matvec(pool, base, &m.router, x, probs);
    let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    probs.iter_mut().for_each(|p| *p = (*p - max).exp());
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);

    let mut stack_chosen = [0usize; 256];
    let mut heap_chosen;
    let chosen_slice: &mut [usize] = if m.n_expert <= 256 {
        for i in 0..m.n_expert {
            stack_chosen[i] = i;
        }
        &mut stack_chosen[..m.n_expert]
    } else {
        heap_chosen = (0..m.n_expert).collect::<Vec<usize>>();
        &mut heap_chosen
    };
    chosen_slice.select_nth_unstable_by(m.n_used - 1, |&a, &b| probs[b].total_cmp(&probs[a]));
    let top_chosen = &mut chosen_slice[..m.n_used];
    top_chosen.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

    let mut num_keep = m.n_used;
    if m.moe_prune > 0.0 && m.n_used > 2 {
        let max_p = probs[top_chosen[0]];
        let min_keep = 2.max(m.n_used / 2);
        let cutoff = max_p * m.moe_prune;
        let mut k = min_keep;
        while k < m.n_used && probs[top_chosen[k]] >= cutoff {
            k += 1;
        }
        num_keep = k;
    }
    let chosen = &top_chosen[..num_keep];
    let wsum: f32 = chosen.iter().map(|&e| probs[e]).sum();
    let router_us = t_router.map_or(0, |t| t.elapsed().as_micros() as u64);

    // Start reading every chosen expert now, so experts 2..k load while expert 1 computes.
    // Not needed when the expert cache serves them: that would read each expert twice.
    if let (Some(pf), None) = (prefetch, cache.as_ref()) {
        pf.hint(chosen.iter().flat_map(|&e| m.experts.expert(e).spans()).collect());
    }

    let layer = stats.as_ref().map(|s| s.1).unwrap_or(0);
    let (lookup_us, pending_misses) = if let Some(c) = cache.as_mut() {
        c.begin_layer(layer, chosen)
    } else {
        (0, Vec::new())
    };

    // If W-TinyLFU bypassed any cold outliers (not in cache, not being read into a slot),
    // prefetch their mapped spans while the P-cores compute the cached hits.
    if let (Some(pf), Some(c)) = (prefetch, cache.as_ref()) {
        let bypassed: Vec<(usize, usize)> = chosen
            .iter()
            .copied()
            .filter(|&e| c.get_expert_ptr(layer, e).is_none() && !pending_misses.iter().any(|&(me, _)| me == e))
            .flat_map(|e| m.experts.expert(e).spans())
            .collect();
        if !bypassed.is_empty() {
            pf.hint(bypassed);
        }
    }

    out.fill(0.0);
    let mut stats = stats;
    let run_expert = |e: usize, c_ptr: Option<*const u8>, dst: &mut [f32]| unsafe {
        let f = m.experts.expert(e);
        let [g_span, u_span, _d_span] = f.spans();
        if let Some(c_ptr) = c_ptr {
            let packed_f = Ffn {
                gate: Mat { offset: 0, ..f.gate },
                up: Mat { offset: g_span.1, ..f.up },
                down: Mat { offset: g_span.1 + u_span.1, ..f.down },
            };
            ffn_forward(pool, c_ptr, &packed_f, x, dst);
        } else {
            ffn_forward(pool, base, &f, x, dst);
        }
    };

    let (disk_us, matvec_us) = if !pending_misses.is_empty() {
        // Overlap P-core matvecs of already-cached (hit) experts with E-core disk reads of
        // missed experts, and pipeline each missed expert as soon as its segments finish.
        let mut expert_outs = vec![vec![0.0f32; out.len()]; chosen.len()];
        let mut computed = vec![false; chosen.len()];
        let t_hit = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
        for (idx, &e) in chosen.iter().enumerate() {
            if let Some(c_ptr) = cache.as_ref().and_then(|c| c.get_expert_ptr(layer, e)) {
                computed[idx] = true;
                run_expert(e, Some(c_ptr), &mut expert_outs[idx]);
            }
        }
        let hit_us = t_hit.map_or(0, |t| t.elapsed().as_micros() as u64);

        let mut disk_us = 0u64;
        let mut miss_us = 0u64;
        for (miss_idx, &(e, _)) in pending_misses.iter().enumerate() {
            disk_us += cache.as_mut().map_or(0, |c| c.wait_one_miss(layer, &pending_misses, miss_idx));
            if let Some(idx) = chosen.iter().position(|&ce| ce == e) {
                let t_m = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
                let c_ptr = cache.as_ref().and_then(|c| c.get_expert_ptr(layer, e));
                computed[idx] = true;
                run_expert(e, c_ptr, &mut expert_outs[idx]);
                miss_us += t_m.map_or(0, |t| t.elapsed().as_micros() as u64);
            }
        }
        let t_rem = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
        for (idx, &e) in chosen.iter().enumerate() {
            if !computed[idx] {
                let c_ptr = cache.as_ref().and_then(|c| c.get_expert_ptr(layer, e));
                run_expert(e, c_ptr, &mut expert_outs[idx]);
            }
        }
        for (idx, &e) in chosen.iter().enumerate() {
            let w = if m.norm { probs[e] / wsum } else { probs[e] };
            let tmp = &expert_outs[idx];
            for i in 0..out.len() {
                out[i] += w * tmp[i];
            }
            if let Some((s, _)) = stats.as_mut() {
                s.expert_bytes += m.experts.expert(e).bytes();
                s.touched[layer * m.n_expert + e] = true;
            }
        }
        miss_us += t_rem.map_or(0, |t| t.elapsed().as_micros() as u64);
        (disk_us, hit_us + miss_us)
    } else {
        let t_matvec = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
        for &e in chosen {
            let w = if m.norm { probs[e] / wsum } else { probs[e] };
            let c_ptr = cache.as_ref().and_then(|c| c.get_expert_ptr(layer, e));
            run_expert(e, c_ptr, tmp);
            for i in 0..out.len() {
                out[i] += w * tmp[i];
            }
            if let Some((s, _)) = stats.as_mut() {
                s.expert_bytes += m.experts.expert(e).bytes();
                s.touched[layer * m.n_expert + e] = true;
            }
        }
        (0, t_matvec.map_or(0, |t| t.elapsed().as_micros() as u64))
    };

    let t_sh = if crate::ui::verbose() { Some(std::time::Instant::now()) } else { None };
    if let Some(sh) = &m.shared {
        ffn_forward(pool, base, sh, x, tmp);
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
    let matvec_us = matvec_us + t_sh.map_or(0, |t| t.elapsed().as_micros() as u64);

    if let Some((s, _)) = stats.as_mut() {
        s.timing.router_us += router_us;
        s.timing.cache_lookup_us += lookup_us;
        s.timing.disk_read_us += disk_us;
        s.timing.expert_matvec_us += matvec_us;
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
    /// Background prefetch hints, only when the model doesn't fit in RAM. Declared before
    /// `raw_bytes` so it is dropped (and its thread joined) before the file is unmapped.
    prefetcher: Option<Prefetcher>,
    /// Reads layers `first..` into private buffers when they don't fit in RAM.
    streamer: Option<crate::layer_stream::LayerStreamer>,
    /// Explicit in-RAM expert cache for streaming MoE weights from disk.
    pub expert_cache: Option<crate::expert_cache::ExpertCache>,
    /// `(file offset, length)` per layer, if every layer is one contiguous dense block.
    layer_spans: Option<Vec<(usize, usize)>>,
    path: String,
    /// The GGUF file, memory-mapped read-only (weights are used in place, never copied).
    pub raw_bytes: MappedFile,
    layers: Vec<Layer>,
    pub layer_range: std::ops::Range<usize>,
    rope: Rope,
    embed: Mat,
    lm_head: Mat,
    output_norm: usize,
    /// K and V for every layer and position, stored as f16 (`[n_layers * max_seq * kv_dim]`).
    /// Half the RAM of f32; when the model streams from disk, that RAM pins more layers.
    pub k_cache: Vec<u16>,
    pub v_cache: Vec<u16>,
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
    format: ChatFormat,
    /// Whether raw prompts start with BOS (`tokenizer.ggml.add_bos_token`).
    bos: Option<usize>,
    /// Tokens that end a reply (`<|im_end|>`, the GGUF EOS, `<|endoftext|>`).
    stop_ids: Vec<usize>,
    /// `<think>` / `</think>` token ids for reasoning models (Qwen3).
    think_ids: Option<(usize, usize)>,
    /// Counts KV-cache shifts, so a reply knows whether its positions moved under it.
    kv_shifts: u64,
    /// Dynamic system prompt for the assistant.
    pub system_prompt: String,
    /// Speculative draft token batch length (default: 4).
    pub draft_len: usize,
    /// Whether agentic tools (file ops, web search, shell) are enabled.
    pub enable_tools: bool,
    pub moe_stats: MoeStats,
    /// Optional small draft engine for speculative decoding (`--draft <path>`).
    pub draft_engine: Option<Box<SmolLm2Engine>>,
    /// Speculative decoding stats `(accepted_draft_tokens, total_drafted_tokens, verify_batches)` for the turn.
    pub spec_stats: (usize, usize, usize),
    /// Optional cluster network client for distributed inference across multiple machines.
    pub cluster_client: Option<crate::cluster_net::ClusterClient>,
    /// Saved KV-cache branches for zero-cost hypothesis rollout and rewinding in RAM.
    pub kv_branches: HashMap<String, (usize, usize, Option<usize>, Option<usize>)>,
    /// Cluster execution timing breakdown per token.
    pub cluster_timing: ClusterTiming,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct ClusterTiming {
    pub worker_us: u64,
    pub local_us: u64,
    pub steps: u64,
}

/// How a conversation is framed in tokens. Built once at load from the model's special
/// tokens, so ChatML (SmolLM2, Qwen), Granite and Tulu-style (OLMoE) models share one code path.
#[derive(Clone, Debug, Default)]
struct ChatFormat {
    system_open: Vec<usize>,
    system_close: Vec<usize>,
    user_open: Vec<usize>,
    user_close: Vec<usize>,
    assistant_open: Vec<usize>,
    /// Appended after a reply; its first token is also what the model emits to end a turn.
    assistant_close: Vec<usize>,
}

/// SmolLM2's own system prompt. Giving the assistant a name stops the tiny model from
/// adopting the user's name ("Hi, I'm Wise" -> "My name is Wise").
pub const SMOLLM_SYSTEM_PROMPT: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
pub const GRANITE_SYSTEM_PROMPT: &str = "You are Granite, developed by IBM. You are a helpful AI assistant.";
pub const QWEN_SYSTEM_PROMPT: &str = "You are Qwen, created by Alibaba. You are a helpful AI assistant. Always provide direct, helpful answers without internal monologue or self-narration.";

impl SmolLm2Engine {
    pub fn load(path: &str) -> std::io::Result<Self> {
        let idx = GgufModelIndex::parse_file(path)?;
        let c = ModelConfig::from_gguf(&idx)?;
        Self::load_with_index_sharded(path, idx, 0..c.n_layers)
    }

    /// Like `load`, reusing an already-parsed index (the vocab + merge tables are large).
    pub fn load_with_index(path: &str, idx: GgufModelIndex) -> std::io::Result<Self> {
        let c = ModelConfig::from_gguf(&idx)?;
        Self::load_with_index_sharded(path, idx, 0..c.n_layers)
    }

    /// Loads only a slice of layers (`layer_range.start..layer_range.end`) for distributed pipeline parallelism.
    pub fn load_sharded(path: &str, layer_range: std::ops::Range<usize>) -> std::io::Result<Self> {
        Self::load_with_index_sharded(path, GgufModelIndex::parse_file(path)?, layer_range)
    }

    /// Loads a specific layer range reusing an already-parsed GGUF index.
    pub fn load_with_index_sharded(
        path: &str,
        idx: GgufModelIndex,
        layer_range: std::ops::Range<usize>,
    ) -> std::io::Result<Self> {
        let c = ModelConfig::from_gguf(&idx)?;
        if layer_range.start >= layer_range.end || layer_range.end > c.n_layers {
            return Err(invalid(format!(
                "invalid layer range {:?}: model has {} layers",
                layer_range, c.n_layers
            )));
        }
        let n_local_layers = layer_range.len();
        let layers = resolve_layers(&idx, &c, layer_range.clone())?;
        let tensor = |name: &str| idx.tensor_by_name.get(name).map(|&i| &idx.tensors[i]);
        let embed = Mat::of(tensor("token_embd.weight").ok_or_else(|| invalid("missing token_embd.weight".into()))?);
        let lm_head = tensor("output.weight").map(Mat::of).unwrap_or(embed);
        let output_norm = tensor("output_norm.weight")
            .ok_or_else(|| invalid("missing output_norm.weight".into()))?
            .abs_file_offset as usize;

        let token = |s: &str| idx.token_to_id.get(s).copied();
        let meta_id = |k: &str| idx.metadata.get(k).and_then(|v| v.as_usize());
        let eos = meta_id("tokenizer.ggml.eos_token_id");
        let bos = match idx.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(GgufValue::Bool(true)) => meta_id("tokenizer.ggml.bos_token_id"),
            _ => None,
        };
        let mut stop_ids = Vec::new();
        for id in [eos, token("<|endoftext|>"), token("<|im_end|>"), token("<|end_of_text|>")].into_iter().flatten() {
            if !stop_ids.contains(&id) {
                stop_ids.push(id);
            }
        }
        let think_ids = token("<think>").zip(token("</think>"));
        let name = idx.model_name.to_lowercase();
        let system_prompt = if let Ok(custom) = std::env::var("SPINDLE_SYSTEM_PROMPT") {
            custom
        } else if name.contains("smollm") {
            SMOLLM_SYSTEM_PROMPT.to_string()
        } else if name.contains("granite") {
            GRANITE_SYSTEM_PROMPT.to_string()
        } else if name.contains("qwen") {
            QWEN_SYSTEM_PROMPT.to_string()
        } else {
            DEFAULT_SYSTEM_PROMPT.to_string()
        };
        let draft_len = std::env::var("SPINDLE_DRAFT_LEN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4)
            .clamp(1, 16);

        let raw_bytes = MappedFile::open(path)?;
        // A partial download still parses (the header is at the start), but tensors near the
        // end would point past the file and fault on first use. Check every tensor fits.
        let file_len = raw_bytes.len() as u64;
        let mut needed = 0u64;
        for t in &idx.tensors {
            let rows: u64 = t.dims.iter().skip(1).product();
            if let Some(rb) = row_bytes(t.dtype, t.dims[0] as usize) {
                needed = needed.max(t.abs_file_offset + rb as u64 * rows);
            }
        }
        if needed > file_len {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "the file is incomplete: it is {} MiB but its tensors need {} MiB. The download was probably cut off; download it again.",
                    file_len >> 20,
                    needed >> 20
                ),
            ));
        }
        // `SPINDLE_FORCE_STREAM=1` treats any model as too big for RAM, to test that path.
        let force_stream = std::env::var("SPINDLE_FORCE_STREAM").is_ok_and(|v| v == "1");
        let sharded_ratio = n_local_layers as f64 / c.n_layers as f64;
        let sharded_needed = (file_len as f64 * sharded_ratio) as u64;
        let avail_ram = crate::mmap_file::available_ram_bytes();
        let fits_in_ram = sharded_needed <= avail_ram.saturating_sub(128 * 1024 * 1024);
        let prefetched = if force_stream {
            false
        } else if fits_in_ram {
            let mut ranges = Vec::new();
            for ly in &layers {
                let (start, len, _) = layer_extent(ly, &c);
                ranges.push((start, len));
            }
            if layer_range.end == c.n_layers {
                ranges.push((lm_head.offset, lm_head.rows * row_bytes(lm_head.dtype, lm_head.cols).unwrap_or(0)));
                ranges.push((output_norm, c.dim * 4));
            }
            raw_bytes.prefetch_ranges(&ranges);
            true
        } else {
            raw_bytes.prefetch_if_fits()
        };
        crate::vlog!(
            "[model:mmap] {:.1} MiB mapped (sharded {:.1} MiB: {})",
            raw_bytes.len() as f64 / (1024.0 * 1024.0),
            sharded_needed as f64 / (1024.0 * 1024.0),
            if prefetched { "prefetching into page cache" } else { "paged on demand: larger than available RAM" }
        );

        // Preflight: the KV-cache is the one big private allocation. Size it to what's free
        // (a quarter of available RAM at most) instead of letting a failed allocation abort
        // the process. 2048 positions is the default.
        let wanted = std::env::var("CHAT_CTX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2048)
            .clamp(64, c.ctx_len);
        let per_pos = 2 * n_local_layers * c.kv_dim * 2;
        let scratch = c.vocab * 8 + 64 * 1024 * 1024;
        let avail = crate::mmap_file::available_ram_bytes() as usize;
        let max_seq = wanted.min((avail / 2).saturating_sub(scratch) / per_pos.max(1));
        if max_seq < 64 {
            return Err(Error::new(
                ErrorKind::OutOfMemory,
                format!(
                    "not enough free RAM for this model: {} MiB free, need about {} MiB. Close some apps and retry.",
                    avail >> 20,
                    ((64 * per_pos + scratch) * 2) >> 20
                ),
            ));
        }
        if max_seq < wanted {
            crate::ui::note(&format!("only {} MiB of RAM free: context limited to {max_seq} tokens", avail >> 20));
        }
        let kv_mib = (max_seq * per_pos) as f64 / (1024.0 * 1024.0);
        let alloc = |n: usize| -> std::io::Result<Vec<u16>> {
            let mut v = Vec::new();
            v.try_reserve_exact(n).map_err(|_| Error::new(ErrorKind::OutOfMemory, "KV-cache allocation failed"))?;
            v.resize(n, 0);
            Ok(v)
        };
        let k_cache = alloc(n_local_layers * max_seq * c.kv_dim)?;
        let v_cache = alloc(n_local_layers * max_seq * c.kv_dim)?;
        crate::vlog!(
            "[model:config] {} | dim={} layers={}/{} (range {:?}) heads={}/{} kv (head_dim {}) vocab={} rope_theta={}{} | KV-cache {max_seq} pos = {kv_mib:.1} MiB",
            idx.architecture,
            c.dim,
            n_local_layers,
            c.n_layers,
            layer_range,
            c.n_heads,
            c.n_kv_heads,
            c.head_dim,
            c.vocab,
            c.rope_theta,
            if c.rope_neox { " (neox)" } else { "" }
        );
        if c.n_expert > 0 {
            let moe_layers = layers.iter().filter(|l| matches!(l.ffn, FfnKind::Moe(_))).count();
            crate::vlog!(
                "[model:moe] {moe_layers} MoE layers, {} of {} experts per token{}",
                c.n_expert_used,
                c.n_expert,
                if c.expert_norm { "" } else { " (weights not renormalized)" }
            );
        }

        let byte_enc = byte_to_char_table();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, &ch)| (ch, b as u8)).collect();
        // SAFETY: field order guarantees the prefetcher is dropped before `raw_bytes`.
        let prefetcher = (!prefetched).then(|| unsafe { Prefetcher::new(&raw_bytes) });
        // Streaming whole layers needs each layer's tensors stored back to back.
        let layer_spans = {
            let ext: Vec<_> = layers.iter().map(|ly| layer_extent(ly, &c)).collect();
            let dense = layers.iter().all(|ly| matches!(ly.ffn, FfnKind::Dense(_)));
            let tight = ext.iter().all(|&(s, e, used)| e - s <= used + used / 20 + 4096);
            (dense && tight).then(|| ext.iter().map(|&(s, e, _)| (s, e - s)).collect())
        };
        let mut engine = Self {
            prefetcher,
            streamer: None,
            expert_cache: None,
            layer_spans,
            path: path.to_string(),
            rope: Rope::new(c.head_dim, c.rope_theta, c.rope_neox),
            cfg: c,
            idx,
            raw_bytes,
            layers,
            layer_range,
            embed,
            lm_head,
            output_norm,
            k_cache,
            v_cache,
            max_seq,
            cur_pos: 0,
            n_keep: 0,
            pool: None,
            sampler: SamplerConfig::default(),
            rng: SplitMix64(SamplerConfig::default().seed),
            byte_enc,
            byte_dec,
            format: ChatFormat::default(),
            bos,
            stop_ids,
            think_ids,
            kv_shifts: 0,
            system_prompt,
            draft_len,
            enable_tools: false,
            moe_stats: MoeStats { touched: vec![false; c.n_layers * c.n_expert], ..Default::default() },
            draft_engine: None,
            spec_stats: (0, 0, 0),
            cluster_client: None,
            kv_branches: HashMap::new(),
            cluster_timing: ClusterTiming::default(),
        };
        engine.format = engine.detect_format()?;
        // The token that closes an assistant turn always ends a reply.
        let end = engine.format.assistant_close[0];
        if !engine.stop_ids.contains(&end) {
            engine.stop_ids.insert(0, end);
        }
        if let Ok(v) = std::env::var("SPINDLE_MOE_PRUNE") {
            if let Ok(p) = v.parse::<f32>() {
                engine.set_moe_prune(p);
            }
        }
        Ok(engine)
    }

    /// Picks the chat framing from the special tokens the model has.
    fn detect_format(&self) -> std::io::Result<ChatFormat> {
        let token = |s: &str| self.idx.token_to_id.get(s).copied();
        let nl = self.encode_bpe_segment("\n");
        let text = |s: &str| self.encode_bpe_segment(s);
        let cat = |parts: &[&[usize]]| parts.concat();
        if let (Some(start), Some(end)) = (token("<|im_start|>"), token("<|im_end|>")) {
            // <|im_start|>role\n ... <|im_end|>\n
            let open = |role: &str| cat(&[&[start], &text(&format!("{role}\n"))]);
            let close = cat(&[&[end], &nl]);
            return Ok(ChatFormat {
                system_open: open("system"),
                system_close: close.clone(),
                user_open: open("user"),
                user_close: close.clone(),
                assistant_open: open("assistant"),
                assistant_close: close,
            });
        }
        if let (Some(start), Some(role_end), Some(end)) =
            (token("<|start_of_role|>"), token("<|end_of_role|>"), token("<|end_of_text|>"))
        {
            // <|start_of_role|>role<|end_of_role|> ... <|end_of_text|>\n
            let open = |role: &str| cat(&[&[start], &text(role), &[role_end]]);
            let close = cat(&[&[end], &nl]);
            return Ok(ChatFormat {
                system_open: open("system"),
                system_close: close.clone(),
                user_open: open("user"),
                user_close: close.clone(),
                assistant_open: open("assistant"),
                assistant_close: close,
            });
        }
        let template = match self.idx.metadata.get("tokenizer.chat_template") {
            Some(GgufValue::String(t)) => t.as_str(),
            _ => "",
        };
        if let (true, Some(eos)) = (template.contains("<|user|>"), self.stop_ids.first().copied()) {
            // Tulu: <|role|>\n ... \n, assistant turns end with EOS.
            let tag = |role: &str| {
                let t = format!("<|{role}|>");
                let mut ids = token(&t).map_or_else(|| text(&t), |id| vec![id]);
                ids.extend(&nl);
                ids
            };
            return Ok(ChatFormat {
                system_open: tag("system"),
                system_close: nl.clone(),
                user_open: tag("user"),
                user_close: nl.clone(),
                assistant_open: tag("assistant"),
                assistant_close: cat(&[&[eos], &nl]),
            });
        }
        Err(invalid("unknown chat format: no ChatML, Granite or Tulu markers in this model".into()))
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
        if let Some(client) = self.cluster_client.as_mut() {
            let _ = client.reset_kv();
        }
        self.cur_pos = 0;
        self.n_keep = 0;
        if let Some(draft) = self.draft_engine.as_mut() {
            draft.cur_pos = 0;
            draft.n_keep = 0;
        }
    }

    /// Saves a named snapshot of the current KV-cache position for zero-cost branching.
    pub fn save_kv_branch(&mut self, name: &str) -> usize {
        let draft_pos = self.draft_engine.as_ref().map(|d| d.cur_pos);
        let draft_keep = self.draft_engine.as_ref().map(|d| d.n_keep);
        self.kv_branches.insert(name.to_string(), (self.cur_pos, self.n_keep, draft_pos, draft_keep));
        self.cur_pos
    }

    /// Rewinds KV-cache to a previously saved branch in 0 ms without re-evaluating prompt tokens.
    pub fn restore_kv_branch(&mut self, name: &str) -> Result<usize, String> {
        let (pos, keep, draft_pos, draft_keep) = self.kv_branches.get(name).copied()
            .ok_or_else(|| {
                let avail: Vec<String> = self.kv_branches.keys().cloned().collect();
                format!("KV branch '{}' not found. Available branches: {:?}", name, avail)
            })?;
        self.cur_pos = pos;
        self.n_keep = keep;
        if let Some(draft) = self.draft_engine.as_mut() {
            if let Some(dp) = draft_pos {
                draft.cur_pos = dp;
            }
            if let Some(dk) = draft_keep {
                draft.n_keep = dk;
            }
        }
        Ok(pos)
    }

    /// Lists all active KV-cache branch checkpoints.
    pub fn list_kv_branches(&self) -> Vec<(String, usize)> {
        let mut list: Vec<(String, usize)> = self.kv_branches.iter().map(|(k, v)| (k.clone(), v.0)).collect();
        list.sort_by(|a, b| a.0.cmp(&b.0));
        list
    }

    /// Attaches a small draft engine for speculative decoding (`--draft <path>`).
    pub fn enable_draft(&mut self, mut draft: SmolLm2Engine) {
        let max_seq = self.max_seq.min(draft.max_seq);
        self.max_seq = max_seq;
        draft.max_seq = max_seq;
        draft.system_prompt = self.system_prompt.clone();
        self.draft_engine = Some(Box::new(draft));
    }

    /// Sets or updates the assistant's system prompt dynamically.
    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt = prompt.to_string();
        if let Some(draft) = self.draft_engine.as_mut() {
            draft.system_prompt = self.system_prompt.clone();
        }
    }

    /// Feeds raw text into the KV-cache.
    pub unsafe fn feed_text(&mut self, text: &str) {
        let ids = self.encode_bpe_segment(text);
        self.feed(&ids);
        if let Some(draft) = self.draft_engine.as_mut() {
            draft.feed(&ids);
        }
    }

    /// Configures the speculative draft speculation batch length (1..16).
    pub fn set_draft_len(&mut self, len: usize) {
        self.draft_len = len.clamp(1, 16);
    }

    /// Enables or disables agentic tool usage (file ops, web search, shell execution).
    pub fn set_tools_enabled(&mut self, enabled: bool) {
        self.enable_tools = enabled;
        let base_prompt = if let Ok(custom) = std::env::var("SPINDLE_SYSTEM_PROMPT") {
            custom
        } else {
            let name = self.idx.model_name.to_lowercase();
            if name.contains("smollm") {
                SMOLLM_SYSTEM_PROMPT.to_string()
            } else if name.contains("granite") {
                GRANITE_SYSTEM_PROMPT.to_string()
            } else if name.contains("qwen") {
                QWEN_SYSTEM_PROMPT.to_string()
            } else {
                DEFAULT_SYSTEM_PROMPT.to_string()
            }
        };
        if enabled {
            let mut prompt = format!("{}\n{}", base_prompt, crate::tools::TOOLS_SYSTEM_PROMPT);
            if let Some(spindle_md) = crate::tools::load_spindle_md() {
                prompt.push_str("\n# Project Instructions (SPINDLE.md)\n");
                prompt.push_str(&spindle_md);
                prompt.push('\n');
            }
            self.system_prompt = prompt;
        } else {
            self.system_prompt = base_prompt;
        }
        if let Some(draft) = self.draft_engine.as_mut() {
            draft.system_prompt = self.system_prompt.clone();
        }
    }

    /// Gets the current dynamic MoE pruning cutoff threshold.
    pub fn moe_prune(&self) -> f32 {
        for layer in &self.layers {
            if let FfnKind::Moe(m) = &layer.ffn {
                return m.moe_prune;
            }
        }
        0.0
    }

    /// Sets the dynamic MoE expert pruning threshold (e.g. 0.03 = prune experts with prob < 3% of max).
    pub fn set_moe_prune(&mut self, threshold: f32) {
        for layer in &mut self.layers {
            if let FfnKind::Moe(m) = &mut layer.ffn {
                m.moe_prune = threshold;
            }
        }
        if let Some(c) = self.cluster_client.as_mut() {
            let _ = c.set_moe_prune(threshold);
        }
    }

    /// Takes the average cluster timing breakdown (worker_ms, local_ms) for the completed turn.
    pub fn take_cluster_timing(&mut self) -> Option<(f64, f64)> {
        if self.cluster_timing.steps == 0 {
            return None;
        }
        let w_avg_ms = (self.cluster_timing.worker_us as f64 / self.cluster_timing.steps as f64) / 1000.0;
        let l_avg_ms = (self.cluster_timing.local_us as f64 / self.cluster_timing.steps as f64) / 1000.0;
        self.cluster_timing = ClusterTiming::default();
        Some((w_avg_ms, l_avg_ms))
    }

    /// Locks up to `budget` bytes of weights into RAM and returns how many were locked.
    ///
    /// Why: every token reads the layers in the same order, 0..n, then starts over. The
    /// working set behaves roughly like an LRU cache, and LRU on a cyclic scan bigger than the
    /// cache evicts exactly the page that is needed next, so nearly every access misses.
    /// Pinning a fixed part of the model turns that part into guaranteed hits and leaves only
    /// the rest cycling. Every pinned byte saves one page-in per token, so which bytes are
    /// pinned matters less than how many; dense weights that every token reads come first,
    /// MoE experts (each used by only some tokens) are never pinned.
    pub fn pin_hot_weights(&self, budget: usize) -> usize {
        let c = self.cfg;
        let mut spans = vec![span(&self.lm_head), (self.output_norm, c.dim * 4)];
        for ly in &self.layers {
            spans.extend([(ly.attn_norm, c.dim * 4), (ly.ffn_norm, c.dim * 4)]);
            spans.extend([span(&ly.wq), span(&ly.wk), span(&ly.wv), span(&ly.wo)]);
            match &ly.ffn {
                FfnKind::Dense(f) => spans.extend(f.spans()),
                FfnKind::Moe(m) => {
                    spans.push(span(&m.router));
                    if let Some(sh) = &m.shared {
                        spans.extend(sh.spans());
                    }
                }
            }
        }
        let mut chosen = Vec::new();
        let mut total = 0;
        for s in spans {
            if chosen.iter().any(|&(off, _)| off == s.0) {
                continue; // tied embedding / LM head
            }
            if total + s.1 > budget {
                continue; // too big for what's left; smaller tensors later may still fit
            }
            total += s.1;
            chosen.push(s);
        }
        if total == 0 || !crate::memory_guard::allow_locking((total + 64 * 1024 * 1024) as u64) {
            return 0;
        }
        self.raw_bytes.lock_ranges(&chosen)
    }

    /// Decides what lives in RAM for a model that doesn't fit, given `budget` bytes for
    /// weights. The LM head (read in full every token) is pinned first, then whole layers
    /// from layer 0 while they fit. The remaining layers are streamed with one read each
    /// through two buffers (which come out of the budget). Models whose layers aren't
    /// contiguous (MoE) fall back to `pin_hot_weights`. Returns `(pinned bytes, first
    /// streamed layer)`.
    ///
    /// Streaming is opt-in (`stream_layers`): measured on Qwen3-0.6B it matched mmap paging
    /// under a tight cap (both ~7-8 tok/s, both drive-limited) while cutting page faults
    /// ~100x, but it is planned once at startup and can't grow into RAM that frees up later,
    /// so by default (pin + mmap) wins whenever memory loosens.
    pub fn plan_residency(&mut self, budget: usize, stream_layers: bool) -> (usize, Option<usize>) {
        let spans = match (&self.layer_spans, stream_layers) {
            (Some(s), true) => s.clone(),
            _ => return (self.pin_hot_weights(budget), None),
        };
        let n = spans.len();
        let max_span = spans.iter().map(|s| s.1).max().unwrap_or(0);
        // Two stream buffers, unless everything ends up pinned anyway.
        let mut left = budget.saturating_sub(2 * max_span);
        let mut chosen = Vec::new();
        let head = (span(&self.lm_head), (self.output_norm, self.cfg.dim * 4));
        if head.0 .1 + head.1 .1 <= left {
            left -= head.0 .1 + head.1 .1;
            chosen.extend([head.0, head.1]);
        }
        let mut first = 0;
        while first < n && spans[first].1 <= left {
            left -= spans[first].1;
            chosen.push(spans[first]);
            first += 1;
        }
        let want = chosen.iter().map(|s| s.1).sum::<usize>();
        let pinned = if want > 0 && crate::memory_guard::allow_locking((want + 64 * 1024 * 1024) as u64) {
            self.raw_bytes.lock_ranges(&chosen)
        } else {
            0
        };
        if first == n {
            return (pinned, None);
        }
        match crate::layer_stream::LayerStreamer::new(&self.path, spans, first) {
            Ok(s) => {
                self.streamer = Some(s);
                (pinned, Some(first))
            }
            Err(e) => {
                crate::ui::note(&format!("layer streaming unavailable ({e}), paging layers instead"));
                (pinned, None)
            }
        }
    }

    /// Bytes of model weights read from the file by the layer streamer so far.
    pub fn streamed_bytes(&self) -> u64 {
        self.streamer.as_ref().map_or(0, |s| s.bytes_read)
    }

    /// True when the model was too big to prefetch, so weights page in as they're used.
    pub fn is_streaming(&self) -> bool {
        self.prefetcher.is_some()
    }

    /// Private memory this engine keeps resident: KV-cache plus per-token scratch.
    pub fn private_bytes_estimate(&self) -> u64 {
        let buffers = self.streamer.as_ref().map_or(0, |s| 2 * s.max_buffer())
            + self.expert_cache.as_ref().map_or(0, |c| c.bytes());
        ((self.k_cache.len() + self.v_cache.len()) * 2 + self.cfg.vocab * 8 + 64 * 1024 * 1024 + buffers) as u64
    }

    pub fn is_moe(&self) -> bool {
        self.cfg.n_expert > 0
    }

    /// `(MiB of expert weights read per token, experts touched, total experts, timing)` for the
    /// turn so far, then resets the counters.
    pub fn take_moe_stats(&mut self) -> Option<(f64, usize, usize, MoeTiming)> {
        if !self.is_moe() {
            return None;
        }
        let s = &mut self.moe_stats;
        let per_tok = s.expert_bytes as f64 / s.tokens.max(1) as f64 / (1024.0 * 1024.0);
        let touched = s.touched.iter().filter(|&&t| t).count();
        let total = s.touched.len();
        let timing = s.timing;
        s.tokens = 0;
        s.expert_bytes = 0;
        s.touched.fill(false);
        s.timing = MoeTiming::default();
        Some((per_tok, touched, total, timing))
    }

    /// Repacks all MoE expert weights into a single contiguous 4096-aligned sidecar file
    /// `<model>.spindle.pack`. Returns the written `PackIndex`.
    pub fn repack_moe_experts<P: FnMut(usize, usize, u64, u64)>(&self, progress: P) -> std::io::Result<crate::expert_pack::PackIndex> {
        if !self.is_moe() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "model has no MoE layers"));
        }
        let pack_path = crate::expert_pack::pack_path_for_model(&self.path);
        let expert_spans = |l: usize, e: usize| -> [(usize, usize); 3] {
            match &self.layers[l].ffn {
                FfnKind::Moe(m) => m.experts.expert(e).spans(),
                _ => panic!("layer {l} is not an MoE layer"),
            }
        };
        crate::expert_pack::repack_experts(
            &self.raw_bytes,
            self.raw_bytes.len() as u64,
            &pack_path,
            self.cfg.n_layers,
            self.cfg.n_expert,
            &expert_spans,
            progress,
        )
    }

    /// Opens an in-RAM expert cache with the specified budget. Uses `<model>.spindle.pack`
    /// when a valid one exists, otherwise reads experts straight from the GGUF, which needs
    /// no extra disk space (a pack duplicates every expert).
    pub fn enable_expert_cache(&mut self, budget_bytes: usize) -> std::io::Result<()> {
        if !self.is_moe() {
            return Ok(());
        }
        let n_expert = self.cfg.n_expert;
        // Expert sizes can differ per layer (mixed quants such as Q4_K_M), so each expert
        // carries its own spans. Non-MoE layers get empty entries and are never requested.
        let segments: Vec<crate::expert_cache::ExpertSegments> = self
            .layers
            .iter()
            .flat_map(|l| {
                (0..n_expert).map(move |e| match &l.ffn {
                    FfnKind::Moe(m) => m.experts.expert(e).spans().map(|(off, len)| (off as u64, len)),
                    _ => [(0, 0); 3],
                })
            })
            .collect();
        let max_payload = segments.iter().map(|s| s.iter().map(|x| x.1).sum::<usize>()).max().unwrap_or(0) as u64;

        let pack_path = crate::expert_pack::pack_path_for_model(&self.path);
        let (cache, source) = if crate::expert_pack::is_pack_valid(&self.path, &pack_path) {
            let h = std::fs::File::open(&pack_path)
                .and_then(|mut f| crate::expert_pack::PackIndex::read_from_file(&mut f))?
                .hdr;
            let first = segments.iter().find(|s| s[0].1 > 0);
            let shape_ok = h.n_layers as usize == self.cfg.n_layers
                && h.n_expert as usize == n_expert
                && max_payload <= h.expert_slot_bytes
                && first.is_some_and(|s| (s[0].1 as u64, s[1].1 as u64, s[2].1 as u64) == (h.gate_bytes, h.up_bytes, h.down_bytes));
            if !shape_ok {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{pack_path} doesn't match this model's experts; delete it and run --pack-experts again"),
                ));
            }
            (crate::expert_cache::ExpertCache::open(&pack_path, budget_bytes)?, "pack")
        } else {
            let c = crate::expert_cache::ExpertCache::open_gguf(&self.path, segments, self.cfg.n_layers, n_expert, budget_bytes)?;
            (c, "gguf")
        };
        crate::vlog!(
            "[expert:cache] enabled in-RAM cache from {source}: {} expert slots ({:.1} MiB budget), slot size {:.2} MiB",
            cache.capacity(),
            (cache.capacity() * cache.slot_bytes) as f64 / (1024.0 * 1024.0),
            cache.slot_bytes as f64 / (1024.0 * 1024.0)
        );
        self.expert_cache = Some(cache);
        Ok(())
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
        for l in 0..self.layers.len() {
            let layer = l * self.max_seq;
            for i in 0..n_move {
                let src = (layer + self.n_keep + n_discard + i) * c.kv_dim;
                let dst = (layer + self.n_keep + i) * c.kv_dim;
                // SAFETY: main checks for AVX2/F16C at startup; kv_dim is a multiple of 8.
                unsafe {
                    f16_to_f32_slice(&self.k_cache[src..src + c.kv_dim], &mut k_tmp);
                    self.rope.rotate(&mut k_tmp, c.head_dim, -(n_discard as f32));
                    f32_to_f16_slice(&k_tmp, &mut self.k_cache[dst..dst + c.kv_dim]);
                }
                self.v_cache.copy_within(src..src + c.kv_dim, dst);
            }
        }
        self.cur_pos -= n_discard;
        self.kv_shifts += 1;
        crate::vlog!("[kv:shift] evicted {n_discard} old tokens (kept {} system + {n_move} recent)", self.n_keep);
    }

    /// Feeds `ids` into the KV-cache starting at `cur_pos` and returns the last hidden state.
    /// Processes up to 32 tokens per layer-synchronous batch so a multi-token prompt scans
    /// the model's layers once per chunk instead of once per token.
    unsafe fn feed(&mut self, ids: &[usize]) -> Vec<f32> {
        let mut last_hidden = vec![0.0f32; self.cfg.dim];
        let mut client = self.cluster_client.take();
        if let Some(c) = client.as_mut() {
            for chunk in ids.chunks(32) {
                self.ensure_kv_room(chunk.len());
                let worker_hiddens = c
                    .prefill(chunk, self.cur_pos)
                    .expect("cluster prefill failed");
                let mut hiddens =
                    self.forward_batch_slice(None, Some(&worker_hiddens), self.cur_pos);
                self.cur_pos += chunk.len();
                if let Some(h) = hiddens.pop() {
                    last_hidden = h;
                }
            }
            self.cluster_client = client;
            return last_hidden;
        }
        for chunk in ids.chunks(32) {
            self.ensure_kv_room(chunk.len());
            if chunk.len() == 1 {
                last_hidden = self.forward_hidden(chunk[0], self.cur_pos);
                self.cur_pos += 1;
            } else {
                let mut hiddens = self.forward_batch(chunk, self.cur_pos);
                self.cur_pos += chunk.len();
                if let Some(h) = hiddens.pop() {
                    last_hidden = h;
                }
            }
        }
        last_hidden
    }

    /// Evaluates one token at `pos`, coordinating with the cluster worker if connected.
    #[inline]
    pub unsafe fn step_token(&mut self, token_id: usize, pos: usize) -> Vec<f32> {
        if self.cluster_client.is_some() {
            let mut client = self.cluster_client.take().unwrap();
            let t_w = std::time::Instant::now();
            let worker_act = client.step(token_id, pos).expect("cluster worker step failed");
            let w_us = t_w.elapsed().as_micros() as u64;
            self.cluster_client = Some(client);

            let t_l = std::time::Instant::now();
            let act = self.forward_slice(None, Some(&worker_act), pos);
            let l_us = t_l.elapsed().as_micros() as u64;

            self.cluster_timing.worker_us += w_us;
            self.cluster_timing.local_us += l_us;
            self.cluster_timing.steps += 1;
            act
        } else {
            self.forward_hidden(token_id, pos)
        }
    }

    /// Runs a batch of tokens `token_ids` starting at `start_pos` synchronously layer-by-layer
    /// (`l in 0..n_layers` on the outer loop, `b in 0..token_ids.len()` on the inner loop) and
    /// returns the hidden state for each token in `token_ids`.
    #[inline]
    pub unsafe fn forward_batch(&mut self, token_ids: &[usize], start_pos: usize) -> Vec<Vec<f32>> {
        self.forward_batch_slice(Some(token_ids), None, start_pos)
    }

    /// Evaluates this engine's assigned layer partition (`self.layer_range`) for a batch of inputs.
    /// - If `token_ids` is Some, loads initial embeddings (Stage 0).
    /// - If `in_hiddens` is Some, uses the provided activation vectors from the upstream stage (Stage 1).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn forward_batch_slice(
        &mut self,
        token_ids: Option<&[usize]>,
        in_hiddens: Option<&[Vec<f32>]>,
        start_pos: usize,
    ) -> Vec<Vec<f32>> {
        let bsz = token_ids.map_or_else(|| in_hiddens.map_or(0, |h| h.len()), |t| t.len());
        if bsz == 0 {
            return Vec::new();
        }
        if bsz == 1 {
            let tid = token_ids.map(|t| t[0]);
            let in_h = in_hiddens.map(|h| h[0].as_slice());
            return vec![self.forward_slice(tid, in_h, start_pos)];
        }
        let c = self.cfg;
        let base = self.raw_bytes.as_ptr();
        let pool = self.pool.as_ref();

        let mut xs: Vec<Vec<f32>> = match (token_ids, in_hiddens) {
            (Some(ids), _) => {
                let rb = row_bytes(self.embed.dtype, c.dim).expect("unsupported embedding type");
                ids.iter()
                    .map(|&tid| {
                        let row = std::slice::from_raw_parts(base.add(self.embed.offset + tid * rb), rb);
                        let mut x = dequant_row(self.embed.dtype, row, c.dim);
                        if c.embedding_scale != 1.0 {
                            x.iter_mut().for_each(|v| *v *= c.embedding_scale);
                        }
                        x
                    })
                    .collect()
            }
            (None, Some(hiddens)) => hiddens.to_vec(),
            (None, None) => return Vec::new(),
        };

        let mut x_norms = vec![vec![0.0f32; c.dim]; bsz];
        let mut proj_outs = vec![vec![0.0f32; c.dim]; bsz];
        let mut q = vec![0.0f32; c.q_dim];
        let mut k = vec![0.0f32; c.kv_dim];
        let mut v = vec![0.0f32; c.kv_dim];
        let mut attn_out = vec![0.0f32; c.q_dim];
        if self.is_moe() {
            self.moe_stats.tokens += bsz;
        }

        let hint = self.prefetcher.as_ref();
        let first_streamed = self.streamer.as_ref().map_or(usize::MAX, |s| s.first);
        for (local_l, &ly) in self.layers.iter().enumerate() {
            let global_l = self.layer_range.start + local_l;
            let streamed = match self.streamer.as_mut() {
                Some(st) if global_l >= st.first => Some(st.take(global_l).expect("reading a layer from the model file failed")),
                _ => None,
            };
            let lbase = match (&streamed, &self.layer_spans) {
                (Some(buf), Some(spans)) => buf.as_ptr().wrapping_sub(spans[global_l].0),
                _ => base,
            };
            if let (Some(pf), Some(next)) = (hint.filter(|_| global_l + 1 < first_streamed), self.layers.get(local_l + 1)) {
                let mut r = vec![span(&next.wq), span(&next.wk), span(&next.wv), span(&next.wo)];
                match &next.ffn {
                    FfnKind::Dense(f) => r.extend(f.spans()),
                    FfnKind::Moe(m) => r.push(span(&m.router)),
                }
                pf.hint(r);
            }

            let scale = c.attn_scale.unwrap_or(1.0 / (c.head_dim as f32).sqrt());
            for b in 0..bsz {
                let pos = start_pos + b;
                rmsnorm(&xs[b], f32_at(lbase, ly.attn_norm, c.dim), &mut x_norms[b], c.rms_eps);
                matvec(pool, lbase, &ly.wq, &x_norms[b], &mut q);
                matvec(pool, lbase, &ly.wk, &x_norms[b], &mut k);
                matvec(pool, lbase, &ly.wv, &x_norms[b], &mut v);
                add_bias(lbase, &mut q, ly.q_bias);
                add_bias(lbase, &mut k, ly.k_bias);
                add_bias(lbase, &mut v, ly.v_bias);
                if let Some(off) = ly.q_norm {
                    let n = if ly.qk_norm_full { c.q_dim } else { c.head_dim };
                    rmsnorm_heads(&mut q, f32_at(lbase, off, n), n, c.rms_eps);
                }
                if let Some(off) = ly.k_norm {
                    let n = if ly.qk_norm_full { c.kv_dim } else { c.head_dim };
                    rmsnorm_heads(&mut k, f32_at(lbase, off, n), n, c.rms_eps);
                }
                self.rope.rotate(&mut q, c.head_dim, pos as f32);
                self.rope.rotate(&mut k, c.head_dim, pos as f32);

                let kv_off = (local_l * self.max_seq + pos) * c.kv_dim;
                f32_to_f16_slice(&k, &mut self.k_cache[kv_off..kv_off + c.kv_dim]);
                f32_to_f16_slice(&v, &mut self.v_cache[kv_off..kv_off + c.kv_dim]);

                let n_heads = c.n_heads;
                let head_dim = c.head_dim;
                let kv_dim = c.kv_dim;
                let heads_per_kv = c.n_heads / c.n_kv_heads;
                let n_vec = head_dim / 8;
                let max_seq = self.max_seq;
                let q_addr = q.as_ptr() as usize;
                let k_addr = self.k_cache.as_ptr() as usize;
                let v_addr = self.v_cache.as_ptr() as usize;
                let out_addr = attn_out.as_mut_ptr() as usize;

                let compute_head_fn = |h: usize| unsafe {
                    let kvh = h / heads_per_kv;
                    let q_head = (q_addr as *const f32).add(h * head_dim);
                    let out_head = (out_addr as *mut f32).add(h * head_dim);

                    let mut m_prev = f32::NEG_INFINITY;
                    let mut l_prev = 0.0f32;
                    let mut acc_v = [_mm256_setzero_ps(); 16];

                    for t in 0..=pos {
                        let t_off = (local_l * max_seq + t) * kv_dim + kvh * head_dim;
                        let k_head = (k_addr as *const u16).add(t_off);
                        let v_head = (v_addr as *const u16).add(t_off);

                        let score = dot_f16_row_avx2(k_head as *const u8, q_head, head_dim) * scale;
                        let m_new = m_prev.max(score);
                        let alpha = (m_prev - m_new).exp();
                        let p = (score - m_new).exp();
                        l_prev = l_prev * alpha + p;
                        m_prev = m_new;

                        let (va, vp) = (_mm256_set1_ps(alpha), _mm256_set1_ps(p));
                        for d in 0..n_vec {
                            let ov = _mm256_mul_ps(acc_v[d], va);
                            let vv = _mm256_cvtph_ps(_mm_loadu_si128(v_head.add(d * 8) as *const __m128i));
                            acc_v[d] = _mm256_fmadd_ps(vp, vv, ov);
                        }
                    }
                    let inv_l = _mm256_set1_ps(1.0f32 / l_prev.max(1e-9));
                    for d in 0..n_vec {
                        _mm256_storeu_ps(out_head.add(d * 8), _mm256_mul_ps(acc_v[d], inv_l));
                    }
                };

                if let Some(p) = pool {
                    if p.n_threads() > 1 && n_heads >= p.n_threads() {
                        p.run(&|t, n| {
                            let h0 = n_heads * t / n;
                            let h1 = n_heads * (t + 1) / n;
                            for h in h0..h1 {
                                compute_head_fn(h);
                            }
                        });
                    } else {
                        for h in 0..n_heads {
                            compute_head_fn(h);
                        }
                    }
                } else {
                    for h in 0..n_heads {
                        compute_head_fn(h);
                    }
                }

                matvec(pool, lbase, &ly.wo, &attn_out, &mut proj_outs[b]);
                for i in 0..c.dim {
                    xs[b][i] += c.residual_scale * proj_outs[b][i];
                }
                rmsnorm(&xs[b], f32_at(lbase, ly.ffn_norm, c.dim), &mut x_norms[b], c.rms_eps);
            }

            match &ly.ffn {
                FfnKind::Dense(f) => {
                    for b in 0..bsz {
                        ffn_forward(pool, lbase, f, &x_norms[b], &mut proj_outs[b]);
                    }
                }
                FfnKind::Moe(m) => moe_forward_batch(
                    pool,
                    lbase,
                    m,
                    &x_norms,
                    &mut proj_outs,
                    Some((&mut self.moe_stats, global_l)),
                    hint,
                    self.expert_cache.as_mut(),
                ),
            }
            for b in 0..bsz {
                for i in 0..c.dim {
                    xs[b][i] += c.residual_scale * proj_outs[b][i];
                }
            }
            if let (Some(buf), Some(st)) = (streamed, self.streamer.as_mut()) {
                st.give_back(buf);
            }
        }

        if self.layer_range.end == self.cfg.n_layers {
            for b in 0..bsz {
                rmsnorm(&xs[b], f32_at(base, self.output_norm, c.dim), &mut x_norms[b], c.rms_eps);
            }
            x_norms
        } else {
            xs
        }
    }

    /// Evaluates this engine's assigned layer partition (`self.layer_range`).
    /// - If `token_id` is Some, initializes `x` from the embedding table (Stage 0 / Worker).
    /// - If `in_hidden` is Some, uses the provided activation vector from the previous stage.
    /// - Evaluates `self.layers`, storing local keys/values in `k_cache`/`v_cache`.
    /// - If `self.layer_range.end == self.cfg.n_layers`, applies `output_norm` and returns normalized hidden state.
    /// - Otherwise, returns the un-normalized residual hidden state ready for the next node.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn forward_slice(
        &mut self,
        token_id: Option<usize>,
        in_hidden: Option<&[f32]>,
        pos: usize,
    ) -> Vec<f32> {
        let c = self.cfg;
        let base = self.raw_bytes.as_ptr();
        let pool = self.pool.as_ref();

        let mut x = if let Some(tid) = token_id {
            let rb = row_bytes(self.embed.dtype, c.dim).expect("unsupported embedding type");
            let row = std::slice::from_raw_parts(base.add(self.embed.offset + tid * rb), rb);
            let mut val = dequant_row(self.embed.dtype, row, c.dim);
            if c.embedding_scale != 1.0 {
                val.iter_mut().for_each(|v| *v *= c.embedding_scale);
            }
            val
        } else if let Some(h) = in_hidden {
            h.to_vec()
        } else {
            panic!("forward_slice requires token_id or in_hidden");
        };

        let mut x_norm = vec![0.0f32; c.dim];
        let mut q = vec![0.0f32; c.q_dim];
        let mut k = vec![0.0f32; c.kv_dim];
        let mut v = vec![0.0f32; c.kv_dim];
        let mut attn_out = vec![0.0f32; c.q_dim];
        let mut proj_out = vec![0.0f32; c.dim];
        let mut moe_tmp = vec![0.0f32; c.dim];
        if self.is_moe() {
            self.moe_stats.tokens += 1;
        }

        let hint = self.prefetcher.as_ref();
        let first_streamed = self.streamer.as_ref().map_or(usize::MAX, |s| s.first);
        for (local_l, &ly) in self.layers.iter().enumerate() {
            let global_l = self.layer_range.start + local_l;
            let streamed = match self.streamer.as_mut() {
                Some(st) if global_l >= st.first => Some(st.take(global_l).expect("reading a layer from the model file failed")),
                _ => None,
            };
            let lbase = match (&streamed, &self.layer_spans) {
                (Some(buf), Some(spans)) => buf.as_ptr().wrapping_sub(spans[global_l].0),
                _ => base,
            };
            if let (Some(pf), Some(next)) = (hint.filter(|_| global_l + 1 < first_streamed), self.layers.get(local_l + 1)) {
                let mut r = vec![span(&next.wq), span(&next.wk), span(&next.wv), span(&next.wo)];
                match &next.ffn {
                    FfnKind::Dense(f) => r.extend(f.spans()),
                    FfnKind::Moe(m) => r.push(span(&m.router)),
                }
                pf.hint(r);
            }

            // Attention: norm, Q/K/V (+ bias), optional per-head Q/K norm, RoPE.
            rmsnorm(&x, f32_at(lbase, ly.attn_norm, c.dim), &mut x_norm, c.rms_eps);
            matvec(pool, lbase, &ly.wq, &x_norm, &mut q);
            matvec(pool, lbase, &ly.wk, &x_norm, &mut k);
            matvec(pool, lbase, &ly.wv, &x_norm, &mut v);
            add_bias(lbase, &mut q, ly.q_bias);
            add_bias(lbase, &mut k, ly.k_bias);
            add_bias(lbase, &mut v, ly.v_bias);
            if let Some(off) = ly.q_norm {
                let n = if ly.qk_norm_full { c.q_dim } else { c.head_dim };
                rmsnorm_heads(&mut q, f32_at(lbase, off, n), n, c.rms_eps);
            }
            if let Some(off) = ly.k_norm {
                let n = if ly.qk_norm_full { c.kv_dim } else { c.head_dim };
                rmsnorm_heads(&mut k, f32_at(lbase, off, n), n, c.rms_eps);
            }
            self.rope.rotate(&mut q, c.head_dim, pos as f32);
            self.rope.rotate(&mut k, c.head_dim, pos as f32);

            let kv_off = (local_l * self.max_seq + pos) * c.kv_dim;
            f32_to_f16_slice(&k, &mut self.k_cache[kv_off..kv_off + c.kv_dim]);
            f32_to_f16_slice(&v, &mut self.v_cache[kv_off..kv_off + c.kv_dim]);

            // Grouped-query attention with an online softmax over positions 0..=pos.
            let scale = c.attn_scale.unwrap_or(1.0 / (c.head_dim as f32).sqrt());
            let n_heads = c.n_heads;
            let head_dim = c.head_dim;
            let kv_dim = c.kv_dim;
            let heads_per_kv = c.n_heads / c.n_kv_heads;
            let n_vec = head_dim / 8;
            let max_seq = self.max_seq;
            let q_addr = q.as_ptr() as usize;
            let k_addr = self.k_cache.as_ptr() as usize;
            let v_addr = self.v_cache.as_ptr() as usize;
            let out_addr = attn_out.as_mut_ptr() as usize;

            let compute_head_fn = |h: usize| unsafe {
                let kvh = h / heads_per_kv;
                let q_head = (q_addr as *const f32).add(h * head_dim);
                let out_head = (out_addr as *mut f32).add(h * head_dim);

                let mut m_prev = f32::NEG_INFINITY;
                let mut l_prev = 0.0f32;
                let mut acc_v = [_mm256_setzero_ps(); 16];

                for t in 0..=pos {
                    let t_off = (local_l * max_seq + t) * kv_dim + kvh * head_dim;
                    let k_head = (k_addr as *const u16).add(t_off);
                    let v_head = (v_addr as *const u16).add(t_off);

                    let score = dot_f16_row_avx2(k_head as *const u8, q_head, head_dim) * scale;
                    let m_new = m_prev.max(score);
                    let alpha = (m_prev - m_new).exp();
                    let p = (score - m_new).exp();
                    l_prev = l_prev * alpha + p;
                    m_prev = m_new;

                    let (va, vp) = (_mm256_set1_ps(alpha), _mm256_set1_ps(p));
                    for d in 0..n_vec {
                        let ov = _mm256_mul_ps(acc_v[d], va);
                        let vv = _mm256_cvtph_ps(_mm_loadu_si128(v_head.add(d * 8) as *const __m128i));
                        acc_v[d] = _mm256_fmadd_ps(vp, vv, ov);
                    }
                }
                let inv_l = _mm256_set1_ps(1.0f32 / l_prev.max(1e-9));
                for d in 0..n_vec {
                    _mm256_storeu_ps(out_head.add(d * 8), _mm256_mul_ps(acc_v[d], inv_l));
                }
            };

            if let Some(p) = pool {
                if p.n_threads() > 1 && n_heads >= p.n_threads() {
                    p.run(&|t, n| {
                        let h0 = n_heads * t / n;
                        let h1 = n_heads * (t + 1) / n;
                        for h in h0..h1 {
                            compute_head_fn(h);
                        }
                    });
                } else {
                    for h in 0..n_heads {
                        compute_head_fn(h);
                    }
                }
            } else {
                for h in 0..n_heads {
                    compute_head_fn(h);
                }
            }

            matvec(pool, lbase, &ly.wo, &attn_out, &mut proj_out);
            for i in 0..c.dim {
                x[i] += c.residual_scale * proj_out[i];
            }

            // Feed-forward: dense SwiGLU or mixture of experts.
            rmsnorm(&x, f32_at(lbase, ly.ffn_norm, c.dim), &mut x_norm, c.rms_eps);
            match &ly.ffn {
                FfnKind::Dense(f) => ffn_forward(pool, lbase, f, &x_norm, &mut proj_out),
                FfnKind::Moe(m) => moe_forward(
                    pool,
                    lbase,
                    m,
                    &x_norm,
                    &mut proj_out,
                    Some((&mut self.moe_stats, global_l)),
                    hint,
                    self.expert_cache.as_mut(),
                    &mut moe_tmp,
                ),
            }
            for i in 0..c.dim {
                x[i] += c.residual_scale * proj_out[i];
            }
            if let (Some(buf), Some(st)) = (streamed, self.streamer.as_mut()) {
                st.give_back(buf);
            }
        }

        if self.layer_range.end == self.cfg.n_layers {
            rmsnorm(&x, f32_at(base, self.output_norm, c.dim), &mut x_norm, c.rms_eps);
            x_norm
        } else {
            x
        }
    }

    /// Evaluates all assigned layers for `token_id` at `pos`.
    #[inline]
    pub unsafe fn forward_hidden(&mut self, token_id: usize, pos: usize) -> Vec<f32> {
        self.forward_slice(Some(token_id), None, pos)
    }

    /// Computes LM-head logits for `hidden` and picks the next token using `self.sampler`:
    /// repetition penalty, temperature, top-k, min-p, top-p, then a seeded random draw
    /// (or plain argmax when `temperature <= 0`). If `forbid_stop` is set, stop tokens are
    /// masked so the reply can't be empty. If `forbid_think` is set, opening `<think>` is masked.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn sample_next_token(
        &mut self,
        hidden: &[f32],
        recent: &[usize],
        forbid_stop: bool,
        forbid_think: bool,
    ) -> usize {
        let c = self.cfg;
        let mut logits = vec![0.0f32; c.vocab];
        matvec(self.pool.as_ref(), self.raw_bytes.as_ptr(), &self.lm_head, hidden, &mut logits);
        if c.logit_scale != 1.0 {
            logits.iter_mut().for_each(|l| *l /= c.logit_scale);
        }

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
        if forbid_think {
            if let Some((open, _)) = self.think_ids {
                logits[open] = f32::NEG_INFINITY;
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

    /// The system message, framed for this model.
    fn encode_system(&self) -> Vec<usize> {
        let f = &self.format;
        if self.system_prompt.trim().is_empty() {
            return Vec::new();
        }
        [&f.system_open[..], &self.encode_bpe_segment(&self.system_prompt), &f.system_close].concat()
    }

    /// Pre-warms and feeds the system prompt into the KV-cache so initial turns respond instantly.
    pub unsafe fn prefill_system(&mut self) {
        if self.n_keep == 0 {
            let sys = self.encode_system();
            if !sys.is_empty() {
                self.feed(&sys);
                if let Some(draft) = self.draft_engine.as_mut() {
                    draft.feed(&sys);
                    draft.n_keep = self.cur_pos;
                }
                self.n_keep = self.cur_pos;
            }
        }
    }

    /// A user message followed by the opening of the assistant's reply.
    fn encode_user_turn(&self, user_prompt: &str, allow_think: bool) -> Vec<usize> {
        let f = &self.format;
        let mut ids = [&f.user_open[..], &self.encode_bpe_segment(user_prompt.trim()), &f.user_close, &f.assistant_open].concat();
        if !allow_think {
            if let Some((open, close)) = self.think_ids {
                // Official Qwen/R1 fast mode: inject a closed think block (<think>\n\n</think>\n\n)
                // so the model recognizes reasoning is already done and immediately begins the direct answer.
                ids.push(open);
                ids.extend(self.encode_bpe_segment("\n\n"));
                ids.push(close);
                ids.extend(self.encode_bpe_segment("\n\n"));
            }
        }
        ids
    }

    /// Full prompt for a single message (system + user + assistant header).
    pub fn encode_chatml(&self, user_prompt: &str) -> Vec<usize> {
        let mut ids = self.encode_system();
        ids.extend(self.encode_user_turn(user_prompt, false));
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

    /// Plain text completion with no chat framing, for comparing against llama.cpp: encodes
    /// `prompt` (with BOS if the model wants one), then generates up to `n` tokens.
    pub unsafe fn complete_raw(&mut self, prompt: &str, n: usize, on_text: &mut dyn FnMut(&str)) -> usize {
        self.reset_kv();
        let mut ids: Vec<usize> = self.bos.into_iter().collect();
        ids.extend(self.encode_bpe_segment(prompt));
        crate::vlog!("[raw] prompt tokens: {ids:?}");
        let mut hidden = self.feed(&ids);
        let mut recent = Vec::new();
        let mut pending: Vec<u8> = Vec::new();
        for _ in 0..n {
            if crate::ui::verbose() {
                // The two best candidates and their logit gap: a tiny gap means another engine
                // can legitimately pick the other one because of rounding.
                let mut logits = vec![0.0f32; self.cfg.vocab];
                matvec(self.pool.as_ref(), self.raw_bytes.as_ptr(), &self.lm_head, &hidden, &mut logits);
                let mut idx: Vec<usize> = (0..logits.len()).collect();
                idx.select_nth_unstable_by(1, |&a, &b| logits[b].total_cmp(&logits[a]));
                let (a, b) = if logits[idx[0]] >= logits[idx[1]] { (idx[0], idx[1]) } else { (idx[1], idx[0]) };
                let s = |t: usize| String::from_utf8_lossy(&self.token_bytes(t)).into_owned();
                crate::vlog!(
                    "[raw] step {}: {:?} {:.3} vs {:?} {:.3} (gap {:.3})",
                    recent.len(),
                    s(a),
                    logits[a] / self.cfg.logit_scale,
                    s(b),
                    logits[b] / self.cfg.logit_scale,
                    (logits[a] - logits[b]) / self.cfg.logit_scale
                );
            }
            let next = self.sample_next_token(&hidden, &recent, false, false);
            if self.stop_ids.contains(&next) {
                break;
            }
            pending.extend(self.token_bytes(next));
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) if e.error_len().is_none() => e.valid_up_to(),
                Err(_) => pending.len(),
            };
            on_text(&String::from_utf8_lossy(&pending[..valid]));
            pending.drain(..valid);
            recent.push(next);
            hidden = self.feed(&[next]);
        }
        on_text(&String::from_utf8_lossy(&pending));
        recent.len()
    }

    /// Generates a conversational reply to `user_prompt`, passing text to `on_text` as it is
    /// produced (whole UTF-8 characters only). The KV-cache persists across calls, so earlier
    /// turns stay in context; only the new user turn is prefilled. Call `reset_kv()` to start
    /// a fresh conversation. Returns `(reply, seconds, tokens processed)`.
    pub unsafe fn generate_reply(
        &mut self,
        user_prompt: &str,
        max_new_tokens: usize,
        allow_think: bool,
        on_text: &mut dyn FnMut(&str),
    ) -> (String, f64, usize) {
        let t0 = Instant::now();
        self.spec_stats = (0, 0, 0);
        let mut draft_engine = self.draft_engine.take();
        if self.n_keep == 0 {
            let sys = self.encode_system();
            self.feed(&sys);
            if let Some(draft) = draft_engine.as_mut() {
                draft.feed(&sys);
                draft.n_keep = self.n_keep;
            }
            self.n_keep = self.cur_pos;
        }
        let mut turn_ids = self.encode_user_turn(user_prompt, allow_think);
        // A single turn can never be larger than the non-system window (leave room to reply).
        let max_turn = (self.max_seq - self.n_keep) / 2;
        if turn_ids.len() > max_turn {
            turn_ids.drain(..turn_ids.len() - max_turn);
        }
        let mut last_hidden = self.feed(&turn_ids);
        if let Some(draft) = draft_engine.as_mut() {
            draft.feed(&turn_ids);
        }
        let reply_start = self.cur_pos;
        let shifts_before = self.kv_shifts;

        let mut recent = Vec::with_capacity(max_new_tokens);
        // Reasoning models: are we inside <think>, and where does the answer start in `recent`?
        let mut in_think = false;
        let mut answer_from: Option<usize> = None;
        let mut reply = String::new();
        // Bytes of a UTF-8 character that isn't complete yet.
        let mut pending: Vec<u8> = Vec::new();
        let mut generated = 0usize;
        let forbid_think = !allow_think;

        let should_stop = |gen: usize, recent_len: usize, ans_from: Option<usize>, _thinking: bool| -> bool {
            if let Some(from) = ans_from {
                recent_len.saturating_sub(from) >= max_new_tokens
            } else {
                gen >= max_new_tokens
            }
        };

        while !should_stop(generated, recent.len(), answer_from, in_think) {
            let penalty_window = &recent[answer_from.unwrap_or(0).min(recent.len())..];
            let next_id = self.sample_next_token(&last_hidden, penalty_window, generated < 2 || in_think, forbid_think);
            if self.stop_ids.contains(&next_id) {
                break;
            }
            match self.think_ids {
                Some((open, _)) if next_id == open => in_think = true,
                Some((_, close)) if next_id == close => {
                    in_think = false;
                    answer_from = Some(recent.len() + 1);
                }
                _ => {}
            }
            pending.extend(self.token_bytes(next_id));
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) if e.error_len().is_none() => e.valid_up_to(),
                Err(_) => pending.len(),
            };
            let piece = String::from_utf8_lossy(&pending[..valid]).into_owned();
            pending.drain(..valid);
            on_text(&piece);
            reply.push_str(&piece);
            recent.push(next_id);
            generated += 1;
            if should_stop(generated, recent.len(), answer_from, in_think) {
                break;
            }

            if let Some(draft) = draft_engine.as_mut() {
                // 1. Advance draft with the verified token
                draft.ensure_kv_room(1);
                let mut draft_hidden = draft.forward_hidden(next_id, draft.cur_pos);
                draft.cur_pos += 1;

                // 2. Draft up to K tokens using the small fast model (configurable via self.draft_len)
                let max_draft = self.draft_len.min(max_new_tokens.saturating_sub(recent.len().saturating_sub(answer_from.unwrap_or(0))));
                let mut draft_tokens = Vec::with_capacity(max_draft);
                for _ in 0..max_draft {
                    draft.ensure_kv_room(1);
                    let d = draft.sample_next_token(&draft_hidden, &[], false, forbid_think);
                    if draft.stop_ids.contains(&d) {
                        break;
                    }
                    draft_tokens.push(d);
                    draft_hidden = draft.forward_hidden(d, draft.cur_pos);
                    draft.cur_pos += 1;
                }

                if draft_tokens.is_empty() {
                    self.ensure_kv_room(1);
                    last_hidden = self.step_token(next_id, self.cur_pos);
                    self.cur_pos += 1;
                    continue;
                }

                // 3. Batch verify on target in ONE layer-synchronous scan!
                // Target evaluates: [next_id, draft_tokens[0], ..., draft_tokens[M-1]]
                let m = draft_tokens.len();
                let mut verify_batch = Vec::with_capacity(m + 1);
                verify_batch.push(next_id);
                verify_batch.extend_from_slice(&draft_tokens);

                self.ensure_kv_room(verify_batch.len());
                let target_start_pos = self.cur_pos;
                let hiddens = if self.cluster_client.is_some() {
                    let mut client = self.cluster_client.take().unwrap();
                    let w_hiddens = client.prefill(&verify_batch, target_start_pos).expect("cluster prefill failed");
                    let h = self.forward_batch_slice(None, Some(&w_hiddens), target_start_pos);
                    self.cluster_client = Some(client);
                    h
                } else {
                    self.forward_batch(&verify_batch, target_start_pos)
                };
                self.cur_pos += verify_batch.len();

                // 4. Verify draft tokens sequentially against target logits
                let mut all_accepted = true;
                let mut stopped = false;
                for (i, &d_tok) in draft_tokens.iter().enumerate() {
                    self.spec_stats.1 += 1;
                    let target_tok = self.sample_next_token(&hiddens[i], &recent[answer_from.unwrap_or(0).min(recent.len())..], in_think, forbid_think);
                    if target_tok == d_tok {
                        self.spec_stats.0 += 1;
                        if self.stop_ids.contains(&d_tok) {
                            let keep_pos = target_start_pos + 1 + i;
                            self.cur_pos = keep_pos;
                            draft.cur_pos = keep_pos;
                            stopped = true;
                            break;
                        }
                        match self.think_ids {
                            Some((open, _)) if d_tok == open => in_think = true,
                            Some((_, close)) if d_tok == close => {
                                in_think = false;
                                answer_from = Some(recent.len() + 1);
                            }
                            _ => {}
                        }
                        pending.extend(self.token_bytes(d_tok));
                        let valid = match std::str::from_utf8(&pending) {
                            Ok(s) => s.len(),
                            Err(e) if e.error_len().is_none() => e.valid_up_to(),
                            Err(_) => pending.len(),
                        };
                        let piece = String::from_utf8_lossy(&pending[..valid]).into_owned();
                        pending.drain(..valid);
                        on_text(&piece);
                        reply.push_str(&piece);
                        recent.push(d_tok);
                        generated += 1;
                        if should_stop(generated, recent.len(), answer_from, in_think) {
                            let keep_pos = target_start_pos + 1 + i;
                            self.cur_pos = keep_pos;
                            draft.cur_pos = keep_pos;
                            stopped = true;
                            break;
                        }
                    } else {
                        all_accepted = false;
                        if self.stop_ids.contains(&target_tok) {
                            let keep_pos = target_start_pos + 1 + i;
                            self.cur_pos = keep_pos;
                            draft.cur_pos = keep_pos;
                            stopped = true;
                            break;
                        }
                        match self.think_ids {
                            Some((open, _)) if target_tok == open => in_think = true,
                            Some((_, close)) if target_tok == close => {
                                in_think = false;
                                answer_from = Some(recent.len() + 1);
                            }
                            _ => {}
                        }
                        pending.extend(self.token_bytes(target_tok));
                        let valid = match std::str::from_utf8(&pending) {
                            Ok(s) => s.len(),
                            Err(e) if e.error_len().is_none() => e.valid_up_to(),
                            Err(_) => pending.len(),
                        };
                        let piece = String::from_utf8_lossy(&pending[..valid]).into_owned();
                        pending.drain(..valid);
                        on_text(&piece);
                        reply.push_str(&piece);
                        recent.push(target_tok);
                        generated += 1;

                        let mismatch_pos = target_start_pos + 1 + i;
                        self.cur_pos = mismatch_pos;
                        last_hidden = self.step_token(target_tok, self.cur_pos);
                        self.cur_pos += 1;

                        draft.cur_pos = mismatch_pos;
                        draft.forward_hidden(target_tok, draft.cur_pos);
                        draft.cur_pos += 1;
                        break;
                    }
                }
                self.spec_stats.2 += 1;

                if stopped {
                    break;
                }

                if all_accepted && !should_stop(generated, recent.len(), answer_from, in_think) {
                    let bonus_tok = self.sample_next_token(&hiddens[m], &recent[answer_from.unwrap_or(0).min(recent.len())..], in_think, forbid_think);
                    if self.stop_ids.contains(&bonus_tok) {
                        break;
                    }
                    match self.think_ids {
                        Some((open, _)) if bonus_tok == open => in_think = true,
                        Some((_, close)) if bonus_tok == close => {
                            in_think = false;
                            answer_from = Some(recent.len() + 1);
                        }
                        _ => {}
                    }
                    pending.extend(self.token_bytes(bonus_tok));
                    let valid = match std::str::from_utf8(&pending) {
                        Ok(s) => s.len(),
                        Err(e) if e.error_len().is_none() => e.valid_up_to(),
                        Err(_) => pending.len(),
                    };
                    let piece = String::from_utf8_lossy(&pending[..valid]).into_owned();
                    pending.drain(..valid);
                    on_text(&piece);
                    reply.push_str(&piece);
                    recent.push(bonus_tok);
                    generated += 1;

                    self.ensure_kv_room(1);
                    last_hidden = self.step_token(bonus_tok, self.cur_pos);
                    self.cur_pos += 1;

                    draft.ensure_kv_room(1);
                    draft.forward_hidden(bonus_tok, draft.cur_pos);
                    draft.cur_pos += 1;
                }
            } else {
                self.ensure_kv_room(1);
                last_hidden = self.step_token(next_id, self.cur_pos);
                self.cur_pos += 1;
            }
        }
        if !pending.is_empty() {
            let piece = String::from_utf8_lossy(&pending).into_owned();
            on_text(&piece);
            reply.push_str(&piece);
        }
        // Qwen3's chat template keeps only the final answer of earlier turns, not their
        // reasoning. Rewind the cache to the start of this reply and re-feed just the answer,
        // which also saves context (the reasoning is usually the longer part). Skipped if the
        // cache shifted during the reply, since `reply_start` no longer points at it.
        if let Some(from) = answer_from {
            if self.kv_shifts == shifts_before {
                let answer: Vec<usize> = recent[from.min(recent.len())..]
                    .iter()
                    .copied()
                    .skip_while(|&t| self.idx.vocab[t].chars().all(|c| c == '\u{010A}'))
                    .collect();
                self.cur_pos = reply_start;
                self.feed(&answer);
                if let Some(draft) = draft_engine.as_mut() {
                    draft.cur_pos = reply_start;
                    draft.feed(&answer);
                }
            }
        }
        // Close the assistant turn in the cache (e.g. `<|im_end|>\n`) so the next user turn
        // follows the model's framing, whether we stopped on EOS or hit max_new_tokens.
        let closing = self.format.assistant_close.clone();
        self.feed(&closing);
        if let Some(draft) = draft_engine.as_mut() {
            draft.feed(&closing);
        }
        self.draft_engine = draft_engine;
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
                let (reply, secs, toks) = eng.generate_reply(prompt, 48, false, &mut |_| {});
                println!("[test:stats] {toks} toks in {secs:.2}s");
                assert!(!reply.trim().is_empty(), "Expected non-empty reply for prompt {prompt:?}");
            }
        }
    }

    #[test]
    fn test_qwen3_0_6b_draft_load_and_generate() {
        let path = "qwen3-0.6b-q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let mut eng = SmolLm2Engine::load(path).expect("load qwen3-0.6b-q8_0.gguf");
        unsafe {
            let (reply, secs, toks) = eng.generate_reply("Hi", 8, false, &mut |_| {});
            println!("[qwen3-0.6b] {toks} tokens in {secs:.3}s => {:.1} tok/s | reply: {reply:?}", toks as f64 / secs);
        }
    }

    #[test]
    fn test_speculative_decoding_smollm2() {
        let path = "smollm2-135m-q4_k_m.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let mut target = SmolLm2Engine::load(path).expect("load target");
        let draft = SmolLm2Engine::load(path).expect("load draft");
        target.enable_draft(draft);
        unsafe {
            let (reply, secs, toks) = target.generate_reply("Hi", 16, false, &mut |_| {});
            println!("[speculative:test] {toks} tokens in {secs:.3}s | stats: {:?} | reply: {reply:?}", target.spec_stats);
            assert!(target.spec_stats.0 > 0, "Expected accepted draft tokens");
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
        for text in ["Kumusta, Señor! Magandang umaga.", "naïve café 日本語 ∑", "tabs\tand\nnewlines"] {
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
                moe_prune: 0.0,
            };
            let mut got = vec![0.0f32; dim];
            let mut tmp = vec![0.0f32; dim];
            unsafe { moe_forward(None, base, &moe, &x, &mut got, None, None, None, &mut tmp) };

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

    #[test]
    fn test_dynamic_moe_pruning() {
        let (dim, ff, n_exp, k) = (16usize, 16usize, 8usize, 4usize);
        let mut buf: Vec<f32> = Vec::new();
        let mut seed = 42u64;
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
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.5).sin()).collect();
        let base = buf.as_ptr() as *const u8;

        let moe_full = Moe {
            router,
            experts: Ffn { gate, up, down },
            n_expert: n_exp,
            n_used: k,
            norm: true,
            shared: None,
            shared_gate: None,
            moe_prune: 0.0,
        };
        let moe_pruned = Moe {
            router,
            experts: Ffn { gate, up, down },
            n_expert: n_exp,
            n_used: k,
            norm: true,
            shared: None,
            shared_gate: None,
            moe_prune: 0.10,
        };

        let mut out_full = vec![0.0f32; dim];
        let mut out_pruned = vec![0.0f32; dim];
        let mut tmp = vec![0.0f32; dim];

        unsafe {
            moe_forward(None, base, &moe_full, &x, &mut out_full, None, None, None, &mut tmp);
            moe_forward(None, base, &moe_pruned, &x, &mut out_pruned, None, None, None, &mut tmp);
        }

        assert!(out_full.iter().all(|v| v.is_finite()));
        assert!(out_pruned.iter().all(|v| v.is_finite()));
        println!("[test:moe_prune] Out full: {:.4?}, Out pruned: {:.4?}", &out_full[..4], &out_pruned[..4]);
    }

    #[test]
    fn test_layer_sharding_matches_full_model() {
        let path = "smollm2-135m-q4_k_m.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let mut full = SmolLm2Engine::load(path).expect("load full");
        let n_layers = full.cfg.n_layers;
        let mid = n_layers / 2;

        let mut stage0 = SmolLm2Engine::load_sharded(path, 0..mid).expect("load stage 0");
        let mut stage1 = SmolLm2Engine::load_sharded(path, mid..n_layers).expect("load stage 1");

        assert_eq!(stage0.layers.len(), mid);
        assert_eq!(stage1.layers.len(), n_layers - mid);

        let prompt = "Rust is a systems programming language";
        let tokens = full.encode_chatml(prompt);
        assert!(!tokens.is_empty());

        for (pos, &tok) in tokens.iter().enumerate() {
            unsafe {
                let full_out = full.forward_hidden(tok, pos);

                let stage0_out = stage0.forward_slice(Some(tok), None, pos);
                let stage1_out = stage1.forward_slice(None, Some(&stage0_out), pos);

                assert_eq!(full_out.len(), stage1_out.len());
                for (i, (&f, &s)) in full_out.iter().zip(&stage1_out).enumerate() {
                    let diff = (f - s).abs();
                    assert!(
                        diff < 1e-3,
                        "Mismatch at pos {pos}, dim {i}: full={f}, sharded={s}, diff={diff}"
                    );
                }
            }
        }
        println!("[test] Sharded layer split (0..{mid} + {mid}..{n_layers}) matched full model across {} tokens!", tokens.len());

        // Also verify batched forward_batch_slice equivalence reusing the loaded engines
        unsafe {
            full.reset_kv();
            stage0.reset_kv();
            stage1.reset_kv();

            let full_outs = full.forward_batch(&tokens, 0);
            let s0_outs = stage0.forward_batch_slice(Some(&tokens), None, 0);
            let s1_outs = stage1.forward_batch_slice(None, Some(&s0_outs), 0);

            assert_eq!(full_outs.len(), s1_outs.len());
            for b in 0..tokens.len() {
                for i in 0..full.cfg.dim {
                    let diff = (full_outs[b][i] - s1_outs[b][i]).abs();
                    assert!(diff < 1e-3, "Batch mismatch at token {b}, dim {i}: full={}, sharded={}, diff={diff}", full_outs[b][i], s1_outs[b][i]);
                }
            }
            println!("[test] Batched sharded prefill matched full model across {} tokens!", tokens.len());
        }
    }
}
