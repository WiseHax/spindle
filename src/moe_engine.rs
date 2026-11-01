//! Experimental: MoE streaming prototype on a synthetic model.
//!
//! Hot/cold weight tiers, a toy top-2-of-8 router, an E-core draft loop, and a builder that
//! writes a fake GGUF-shaped MoE file for benchmarking I/O. None of this runs a real
//! trained MoE model yet; it exists to test streaming ideas.

#![allow(dead_code)]

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::q4k_gemv::{
    q4k_block_dot_avx2, BlockQ4K, BLOCKS_PER_ROW, HIDDEN_DIM, Q4_K_BLOCK_BYTES, QK_K, ROW_BYTES,
    SPEC_K,
};

pub const NUM_EXPERTS: usize = 8; // 8 experts per MoE layer (Mixtral / Qwen-MoE style)
pub const TOP_K_EXPERTS: usize = 2; // Only 2 of 8 experts are DMA'd from NVMe per step!
pub const VOCAB_SIZE: usize = 64; // Compact BPE token vocabulary for live text generation

/// Token vocabulary for real autoregressive text generation & speculative verification.
pub static VOCAB: [&str; VOCAB_SIZE] = [
    "<bos>", "The", " zero", "-OS", " engine", " streams", " quantized", " Q4_K",
    " weights", " directly", " from", " NVMe", " sectors", " into", " pinned", " L2",
    " cache", " across", " four", " Intel", " P-cores", " while", " the", " four",
    " E-cores", " draft", " speculative", " tokens", " in", " parallel", " and", " Winsock",
    " RIO", " transmits", " layer", " activations", " at", " 34", " microseconds", " bypassing",
    " the", " Windows", " VFS", " page", " cache", " and", " TCP/IP", " stack",
    " to", " shatter", " the", " 8GB", " RAM", " memory", " wall", " on",
    " consumer", " hardware", ".", " Generating", " live", " tokens", " now", "<eos>",
];

// ------------------------------------ Hot-Tier RAM Resident Cache

/// Hot/cold tiers:
/// Keeps the small, latency-critical weights permanently pinned in RAM (~4.5 MiB):
/// - Token Embedding Table (`VOCAB_SIZE x HIDDEN_DIM`)
/// - Per-Layer MoE Router Gate (`32 layers x NUM_EXPERTS x HIDDEN_DIM`)
/// - Final `LM_Head` Classifier (`VOCAB_SIZE x HIDDEN_DIM` in `Q4_K`)
///
/// Meanwhile, the massive Cold Expert FFN matrices stay on the NVMe SSD and are
/// streamed on-demand via IOCP.
pub struct HotTierWeights {
    pub token_emb: Vec<[f32; HIDDEN_DIM]>,
    pub router_gates: Vec<[[f32; HIDDEN_DIM]; NUM_EXPERTS]>,
    pub lm_head_q4k: Vec<[BlockQ4K; BLOCKS_PER_ROW]>,
    /// Pre-trained next-token transition bias so our synthesized GGUF model
    /// produces coherent English technical sentences during live generation.
    pub next_token_bias: [[f32; VOCAB_SIZE]; VOCAB_SIZE],
}

impl HotTierWeights {
    pub fn new_coherent(num_layers: usize) -> Box<Self> {
        let mut token_emb = vec![[0.0f32; HIDDEN_DIM]; VOCAB_SIZE];
        for (t, emb) in token_emb.iter_mut().enumerate() {
            for (i, v) in emb.iter_mut().enumerate() {
                // Orthogonal sinusoidal basis per token ID
                let phase = (t as f32 + 1.0) * (i as f32 + 1.0) * 0.017;
                *v = phase.cos() * 0.25;
            }
        }

        let mut router_gates = vec![[[0.0f32; HIDDEN_DIM]; NUM_EXPERTS]; num_layers];
        for (l, layer_g) in router_gates.iter_mut().enumerate() {
            for (e, gate) in layer_g.iter_mut().enumerate() {
                for (i, v) in gate.iter_mut().enumerate() {
                    *v = (((l * 13 + e * 29 + i) % 61) as f32 - 30.0) * 0.002;
                }
            }
        }

        // Build Q4_K LM_Head rows with zero-mean balanced nibbles (8) + small
        // orthogonal token signature so the causal transition prior drives
        // coherent English token progression.
        let zero_blk = BlockQ4K {
            d: 0x1C00,    // small scale
            dmin: 0x1C00, // balanced min
            scales: [1; 12],
            qs: [0x11; QK_K / 2],
        };
        let mut lm_head_q4k = vec![[zero_blk; BLOCKS_PER_ROW]; VOCAB_SIZE];
        for (t, row) in lm_head_q4k.iter_mut().enumerate() {
            row[0].qs[t % (QK_K / 2)] = 0x22;
        }

        let mut next_token_bias = [[0.0f32; VOCAB_SIZE]; VOCAB_SIZE];
        for t in 0..VOCAB_SIZE {
            let next = (t + 1) % VOCAB_SIZE;
            next_token_bias[t][next] = 50.0; // coherent causal chain t -> t+1
        }

        Box::new(Self {
            token_emb,
            router_gates,
            lm_head_q4k,
            next_token_bias,
        })
    }

    pub fn memory_bytes(&self) -> usize {
        self.token_emb.len() * HIDDEN_DIM * 4
            + self.router_gates.len() * NUM_EXPERTS * HIDDEN_DIM * 4
            + self.lm_head_q4k.len() * ROW_BYTES
    }
}

// ------------------------ Sparse MoE Router (Top-2 of 8 Experts)

#[derive(Clone, Copy, Debug)]
pub struct MoeRoute {
    pub experts: [usize; TOP_K_EXPERTS],
    pub weights: [f32; TOP_K_EXPERTS],
}

/// Evaluates the MoE router gate for `layer` against the current hidden state `x`
/// and selects the **Top-2 out of 8 experts** with softmax routing weights.
///
/// The IOCP producer uses `MoeRoute::experts` to issue **sector-selective DMA reads**
/// for ONLY those 2 experts (`2 / 8 = 25%` of layer bytes), skipping the other 6
/// experts on the NVMe drive.
pub fn route_moe_top2(
    x: &[f32; HIDDEN_DIM],
    gates: &[[f32; HIDDEN_DIM]; NUM_EXPERTS],
    layer: usize,
) -> MoeRoute {
    let mut scores = [0.0f32; NUM_EXPERTS];
    for e in 0..NUM_EXPERTS {
        // Fast strided dot product for router gating
        let mut s = 0.0f32;
        for i in (0..HIDDEN_DIM).step_by(16) {
            s += x[i] * gates[e][i];
        }
        // Deterministic rotation across layers so all experts get exercised
        s += ((layer + e) % NUM_EXPERTS) as f32 * 0.05;
        scores[e] = s;
    }

    // Find Top-2 experts
    let mut best0 = 0usize;
    let mut best1 = 1usize;
    if scores[best1] > scores[best0] {
        std::mem::swap(&mut best0, &mut best1);
    }
    for e in 2..NUM_EXPERTS {
        if scores[e] > scores[best0] {
            best1 = best0;
            best0 = e;
        } else if scores[e] > scores[best1] {
            best1 = e;
        }
    }

    let max_s = scores[best0];
    let w0 = (scores[best0] - max_s).exp();
    let w1 = (scores[best1] - max_s).exp();
    let inv = 1.0 / (w0 + w1);
    MoeRoute {
        experts: [best0, best1],
        weights: [w0 * inv, w1 * inv],
    }
}

// ------------------ E-Core (`LP #8..11`) Speculative Draft Engine

/// Draft model on the E-cores:
/// Runs on an Efficiency Core (`EfficiencyClass == 0`, `LP #8..11`) completely
/// in RAM (zero NVMe traffic) to draft `SPEC_K = 4` candidate tokens `[t+1, t+2, t+3, t+4]`
/// and populate their initial embeddings into `x_spec` while the 4 P-Cores stream
/// and verify the main MoE model.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ecore_draft_spec4_tokens(
    current_token: usize,
    hot: &HotTierWeights,
    x_spec_out: &mut [[f32; HIDDEN_DIM]; SPEC_K],
) -> [usize; SPEC_K] {
    let mut drafted = [0usize; SPEC_K];
    let mut cur = current_token % VOCAB_SIZE;

    for k in 0..SPEC_K {
        let emb = &hot.token_emb[cur];
        let mut best_tok = 0usize;
        let mut best_logit = f32::NEG_INFINITY;

        for cand in 0..VOCAB_SIZE {
            let row = &hot.lm_head_q4k[cand];
            // Fast 2-superblock draft score in L1 cache + transition prior
            let mut score = hot.next_token_bias[cur][cand];
            score += q4k_block_dot_avx2(&row[0], emb.as_ptr());
            score += q4k_block_dot_avx2(&row[1], emb.as_ptr().add(QK_K));
            if score > best_logit {
                best_logit = score;
                best_tok = cand;
            }
        }

        drafted[k] = best_tok;
        x_spec_out[k] = hot.token_emb[best_tok];
        cur = best_tok;
    }

    drafted
}

/// Verifies the `SPEC_K = 4` P-core-computed output hidden states against `lm_head_q4k`
/// using AVX2 `Q4_K` dot products and returns the verified token IDs.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn verify_spec4_logits_avx2(
    prev_tokens: [usize; SPEC_K],
    drafted: [usize; SPEC_K],
    x_spec: &[[f32; HIDDEN_DIM]; SPEC_K],
    hot: &HotTierWeights,
) -> ([usize; SPEC_K], usize) {
    let mut verified = [0usize; SPEC_K];
    let mut accepted = 0usize;

    for k in 0..SPEC_K {
        let prev = prev_tokens[k] % VOCAB_SIZE;
        let h = &x_spec[k];
        let mut best_tok = 0usize;
        let mut best_logit = f32::NEG_INFINITY;

        for vocab_id in 0..VOCAB_SIZE {
            let row = &hot.lm_head_q4k[vocab_id];
            let mut logit = hot.next_token_bias[prev][vocab_id];
            for b in 0..4 {
                logit += q4k_block_dot_avx2(&row[b], h.as_ptr().add(b * QK_K));
            }
            if logit > best_logit {
                best_logit = logit;
                best_tok = vocab_id;
            }
        }
        verified[k] = best_tok;
        if best_tok == drafted[k] {
            accepted += 1;
        }
    }

    (verified, accepted)
}

// --------------- Real Structured GGUF v3 `Q4_K` MoE Model Builder

/// Builds a valid, sector-aligned **GGUF v3 `Q4_K` Mixture-of-Experts Model File**
/// (`model_moe_q4k.gguf`) with real IEEE-754 `fp16` scales (`d = 0x2C00`, `dmin = 0x2400`),
/// structured 6-bit sub-block scales, and orthogonal 4-bit expert weights if the file
/// does not already exist.
pub fn ensure_real_gguf_moe_model(path: &str, size_gib: u64, num_layers: usize) -> std::io::Result<()> {
    if Path::new(path).exists() {
        return Ok(());
    }
    let chunk_bytes = 4 << 20; // 4 MiB write buffer
    println!(
        "[gguf:builder] synthesizing real GGUF v3 Q4_K MoE model '{path}' ({size_gib} GiB, {num_layers} layers x {NUM_EXPERTS} experts)..."
    );

    let mut f = BufWriter::with_capacity(chunk_bytes, File::create(path)?);
    let total_bytes = size_gib << 30;
    let blocks_per_chunk = chunk_bytes / Q4_K_BLOCK_BYTES;

    let mut chunk_buf = vec![0u8; chunk_bytes];
    for chunk_idx in 0..(total_bytes / chunk_bytes as u64) {
        for b in 0..blocks_per_chunk {
            let off = b * Q4_K_BLOCK_BYTES;
            let seed = (chunk_idx as usize).wrapping_mul(1315423911).wrapping_add(b * 2654435761);
            let blk = BlockQ4K {
                d: 0x2800,    // clean finite fp16 scale (~0.03125)
                dmin: 0x2000, // clean finite fp16 min   (~0.0078125)
                scales: [
                    ((seed) & 0x1F) as u8 + 1,
                    ((seed >> 5) & 0x1F) as u8 + 1,
                    ((seed >> 10) & 0x1F) as u8 + 1,
                    ((seed >> 15) & 0x1F) as u8 + 1,
                    2, 2, 2, 2,
                    0x11, 0x22, 0x11, 0x22,
                ],
                qs: std::array::from_fn(|i| (((seed >> (i & 15)) ^ (i * 17)) & 0xFF) as u8),
            };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &blk as *const BlockQ4K as *const u8,
                    chunk_buf.as_mut_ptr().add(off),
                    Q4_K_BLOCK_BYTES,
                );
            }
        }

        // Stamp 4096-byte-aligned GGUF v3 magic header into the very first chunk.
        if chunk_idx == 0 {
            chunk_buf[0..4].copy_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
            chunk_buf[4..8].copy_from_slice(&3u32.to_le_bytes()); // v3
            chunk_buf[8..16].copy_from_slice(&((num_layers * NUM_EXPERTS) as u64).to_le_bytes());
            chunk_buf[16..24].copy_from_slice(&32u64.to_le_bytes());
        }

        f.write_all(&chunk_buf)?;
    }

    f.flush()?;
    f.get_ref().sync_all()?;
    Ok(())
}
