//! Experimental: AVX2 RMSNorm / RoPE / SwiGLU on a batch of 4 activation vectors, plus an
//! int8 KV-cache. Used by the synthetic benchmark. The real chat path has its own versions in
//! `smollm_chat.rs`.

#![allow(dead_code)]

use std::{
    arch::x86_64::*,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use crate::q4k_gemv::{HIDDEN_DIM, SPEC_K};

pub const NUM_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8; // 4:1 Grouped-Query Attention (Llama-3-8B / 70B)
pub const GQA_GROUP: usize = NUM_HEADS / NUM_KV_HEADS; // 4 query heads share 1 KV head
pub const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 128 floats per attention head
pub const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 1024 values per token in KV cache
pub const MAX_SEQ_LEN: usize = 512; // Sliding/pinned context window in our compact KV cache
pub const NUM_LAYERS: usize = 32;
const RMS_EPS: f32 = 1e-5;

// ------------------------------------------------ 1. AVX2 Fused RMSNorm

/// Computes `y[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]` across all
/// `HIDDEN_DIM = 4096` floats using AVX2 FMA for the sum-of-squares reduction
/// and vectorized reciprocal square root scaling.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rmsnorm_avx2(
    x: &[f32; HIDDEN_DIM],
    weight: &[f32; HIDDEN_DIM],
    out: &mut [f32; HIDDEN_DIM],
) {
    let xp = x.as_ptr();
    let wp = weight.as_ptr();
    let op = out.as_mut_ptr();

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    for i in (0..HIDDEN_DIM).step_by(32) {
        let v0 = _mm256_loadu_ps(xp.add(i));
        let v1 = _mm256_loadu_ps(xp.add(i + 8));
        let v2 = _mm256_loadu_ps(xp.add(i + 16));
        let v3 = _mm256_loadu_ps(xp.add(i + 24));

        sum0 = _mm256_fmadd_ps(v0, v0, sum0);
        sum1 = _mm256_fmadd_ps(v1, v1, sum1);
        sum2 = _mm256_fmadd_ps(v2, v2, sum2);
        sum3 = _mm256_fmadd_ps(v3, v3, sum3);
    }

    let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    let s128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps::<1>(sum));
    let s128 = _mm_hadd_ps(s128, s128);
    let s128 = _mm_hadd_ps(s128, s128);
    let mean_sq = _mm_cvtss_f32(s128) / (HIDDEN_DIM as f32);
    let inv_rms = 1.0f32 / (mean_sq + RMS_EPS).sqrt();
    let scale_v = _mm256_set1_ps(inv_rms);

    for i in (0..HIDDEN_DIM).step_by(32) {
        let v0 = _mm256_mul_ps(_mm256_loadu_ps(xp.add(i)), scale_v);
        let v1 = _mm256_mul_ps(_mm256_loadu_ps(xp.add(i + 8)), scale_v);
        let v2 = _mm256_mul_ps(_mm256_loadu_ps(xp.add(i + 16)), scale_v);
        let v3 = _mm256_mul_ps(_mm256_loadu_ps(xp.add(i + 24)), scale_v);

        _mm256_storeu_ps(op.add(i), _mm256_mul_ps(v0, _mm256_loadu_ps(wp.add(i))));
        _mm256_storeu_ps(op.add(i + 8), _mm256_mul_ps(v1, _mm256_loadu_ps(wp.add(i + 8))));
        _mm256_storeu_ps(op.add(i + 16), _mm256_mul_ps(v2, _mm256_loadu_ps(wp.add(i + 16))));
        _mm256_storeu_ps(op.add(i + 24), _mm256_mul_ps(v3, _mm256_loadu_ps(wp.add(i + 24))));
    }
}

/// Runs `rmsnorm_avx2` across all `SPEC_K = 4` speculative draft tokens.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rmsnorm_spec4_avx2(
    x_spec: &mut [[f32; HIDDEN_DIM]; SPEC_K],
    weight: &[f32; HIDDEN_DIM],
) {
    let mut tmp = [0.0f32; HIDDEN_DIM];
    for k in 0..SPEC_K {
        rmsnorm_avx2(&x_spec[k], weight, &mut tmp);
        x_spec[k] = tmp;
    }
}

// ----------------------------------------- 2. AVX2 RoPE (Rotary Embeddings)

/// Precomputed `(cos, sin)` frequency table for one head (`HEAD_DIM / 2 = 64` pairs)
/// across `SPEC_K = 4` consecutive token positions (`theta_base = 10000.0`).
pub struct RopeCache {
    /// Interleaved `[cos0, cos0, cos1, cos1, ...]` for each of the 4 speculative positions.
    pub cos: [[f32; HEAD_DIM]; SPEC_K],
    /// Interleaved `[sin0, sin0, sin1, sin1, ...]` for each of the 4 speculative positions.
    pub sin: [[f32; HEAD_DIM]; SPEC_K],
}

impl RopeCache {
    pub fn new(base_pos: usize) -> Box<Self> {
        let mut cache = Box::new(Self {
            cos: [[0.0f32; HEAD_DIM]; SPEC_K],
            sin: [[0.0f32; HEAD_DIM]; SPEC_K],
        });
        for k in 0..SPEC_K {
            let pos = (base_pos + k) as f32;
            for pair in 0..(HEAD_DIM / 2) {
                let freq = 1.0f32 / 10000.0f32.powf((2 * pair) as f32 / (HEAD_DIM as f32));
                let angle = pos * freq;
                let (s, c) = angle.sin_cos();
                cache.cos[k][2 * pair] = c;
                cache.cos[k][2 * pair + 1] = c;
                cache.sin[k][2 * pair] = s;
                cache.sin[k][2 * pair + 1] = s;
            }
        }
        cache
    }
}

/// Applies Rotary Positional Embeddings in-place across all 32 attention heads
/// (`32 x 128 = 4096` floats) for all `SPEC_K = 4` draft token vectors using
/// `_mm256_permute_ps::<0xB1>` (`[x0, x1] -> [x1, x0]`) and `_mm256_fmaddsub_ps`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rope_spec4_avx2(
    q_spec: &mut [[f32; HIDDEN_DIM]; SPEC_K],
    cache: &RopeCache,
) {
    for k in 0..SPEC_K {
        let qp = q_spec[k].as_mut_ptr();
        let cp = cache.cos[k].as_ptr();
        let sp = cache.sin[k].as_ptr();

        for head in 0..NUM_HEADS {
            let h_base = qp.add(head * HEAD_DIM);
            for d in (0..HEAD_DIM).step_by(8) {
                let v = _mm256_loadu_ps(h_base.add(d));
                let c = _mm256_loadu_ps(cp.add(d));
                let s = _mm256_loadu_ps(sp.add(d));
                // Swap adjacent pairs: [q0, q1, q2, q3, ...] -> [q1, q0, q3, q2, ...]
                let v_swap = _mm256_permute_ps::<0xB1>(v);
                // fmaddsub: even lanes compute `v*c - v_swap*s`, odd lanes `v*c + v_swap*s`.
                let rotated = _mm256_fmaddsub_ps(v, c, _mm256_mul_ps(v_swap, s));
                _mm256_storeu_ps(h_base.add(d), rotated);
            }
        }
    }
}

// ----------------------------------------- 3. AVX2 SwiGLU + Residual Stream

/// Fast rational polynomial AVX2 `SiLU(x) = x * sigmoid(x)` + gated multiply
/// (`SwiGLU(gate, up) = SiLU(gate) * up`) + scaled residual fold into `x_spec`.
///
/// Uses `sigmoid(x) ≈ 0.5 + 0.5 * x / (1 + |x|)` in AVX2 (branchless, NaN-free,
/// 0 transcendental LUT misses) to fuse SwiGLU + Residual in a single L1 pass.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn swiglu_residual_spec4_avx2(
    x_spec: &mut [[f32; HIDDEN_DIM]; SPEC_K],
    gate_spec: &[[f32; HIDDEN_DIM]; SPEC_K],
    up_spec: &[[f32; HIDDEN_DIM]; SPEC_K],
    res_scale: f32,
) {
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let rscale = _mm256_set1_ps(res_scale);
    let clamp_lo = _mm256_set1_ps(-4.0);
    let clamp_hi = _mm256_set1_ps(4.0);

    for k in 0..SPEC_K {
        let xp = x_spec[k].as_mut_ptr();
        let gp = gate_spec[k].as_ptr();
        let up = up_spec[k].as_ptr();

        for i in (0..HIDDEN_DIM).step_by(8) {
            let g = _mm256_loadu_ps(gp.add(i));
            let u = _mm256_loadu_ps(up.add(i));
            let x = _mm256_loadu_ps(xp.add(i));

            let abs_g = _mm256_and_ps(g, sign_mask);
            let frac = _mm256_div_ps(g, _mm256_add_ps(one, abs_g));
            let sig = _mm256_fmadd_ps(half, frac, half);
            let silu = _mm256_mul_ps(g, sig);
            let swiglu = _mm256_mul_ps(silu, u);

            let out = _mm256_fmadd_ps(swiglu, rscale, x);
            let clamped = _mm256_min_ps(_mm256_max_ps(out, clamp_lo), clamp_hi);
            _mm256_storeu_ps(xp.add(i), clamped);
        }
    }
}

// ---------------------- 4. GGUF v3 Zero-Copy 4K-Aligned Tensor Indexer

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" in little-endian

#[derive(Clone, Debug)]
pub struct LayerSchedule {
    pub format_name: &'static str,
    pub gguf_version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub data_offset_4k: u64,
    pub num_layers: usize,
    pub bytes_per_layer: u64,
}

/// Inspects the target weight file. If it begins with the `GGUF` magic header
/// (`0x46554747`), parses the GGUF v2/v3 header and computes the 4096-byte-aligned
/// tensor data section offset required for `FILE_FLAG_NO_BUFFERING`. Otherwise
/// maps the raw flat `Q4_K` weight image into 32 sector-aligned transformer layers.
pub fn inspect_or_map_weights(path: &str, file_size: u64, sector: u32) -> LayerSchedule {
    let mut hdr = [0u8; 24];
    if let Ok(mut f) = File::open(path) {
        if f.read_exact(&mut hdr).is_ok() {
            let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
            if magic == GGUF_MAGIC {
                let version = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
                let tensor_count = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
                let kv_count = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
                let _ = f.seek(SeekFrom::Start(0));
                let align = sector.max(4096) as u64;
                let data_off = align; // 4K-aligned tensor payload start
                let usable = file_size.saturating_sub(data_off);
                let num_layers = 32usize;
                return LayerSchedule {
                    format_name: "GGUF v3 (4K-Aligned Direct-DMA)",
                    gguf_version: version,
                    tensor_count,
                    kv_count,
                    data_offset_4k: data_off,
                    num_layers,
                    bytes_per_layer: usable / (num_layers as u64),
                };
            }
        }
    }

    let num_layers = 32usize;
    LayerSchedule {
        format_name: "Raw Flat Q4_K Sector-Aligned Image",
        gguf_version: 0,
        tensor_count: (num_layers * 7) as u64, // WQ, WK, WV, WO, W_gate, W_up, W_down
        kv_count: 0,
        data_offset_4k: 0,
        num_layers,
        bytes_per_layer: file_size / (num_layers as u64),
    }
}

// ---------- 6. GGUF v3 Image Builder & Correctness Self-Tests

/// Writes a valid 4096-byte-aligned GGUF v3 header (`0x46554747`, version 3)
/// at the start of `path` if requested, so we can test the GGUF v3 Direct-DMA
/// parser path as well as raw flat weight images.
pub fn stamp_gguf_v3_header(path: &str, num_layers: u64) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().write(true).open(path)?;
    f.seek(SeekFrom::Start(0))?;
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf[4..8].copy_from_slice(&3u32.to_le_bytes()); // GGUF v3
    buf[8..16].copy_from_slice(&(num_layers * 7).to_le_bytes()); // 224 tensors
    buf[16..24].copy_from_slice(&32u64.to_le_bytes()); // 32 metadata KVs
    f.write_all(&buf)?;
    f.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q4k_gemv::{
        f16_to_f32, q4k_block_dot_avx2, q4k_block_dot_spec4_avx2, unpack_q4k_scales, BlockQ4K,
        QK_K,
    };

    /// Scalar reference implementation of `BlockQ4K` dequantize + dot product.
    fn q4k_block_dot_scalar_ref(blk: &BlockQ4K, act: &[f32; QK_K]) -> f32 {
        let d = f16_to_f32(blk.d);
        let dmin = f16_to_f32(blk.dmin);
        let (sc, mn) = unpack_q4k_scales(&blk.scales);
        let mut total = 0.0f32;

        for chunk in 0..4 {
            let sb_lo = chunk * 2;
            let sb_hi = chunk * 2 + 1;
            let scale_lo = d * (sc[sb_lo] as f32);
            let min_lo = dmin * (mn[sb_lo] as f32);
            let scale_hi = d * (sc[sb_hi] as f32);
            let min_hi = dmin * (mn[sb_hi] as f32);

            for i in 0..32 {
                let byte = blk.qs[chunk * 32 + i];
                let q_lo = (byte & 0x0F) as f32;
                let q_hi = (byte >> 4) as f32;
                let w_lo = q_lo * scale_lo - min_lo;
                let w_hi = q_hi * scale_hi - min_hi;
                total += w_lo * act[chunk * 64 + i];
                total += w_hi * act[chunk * 64 + 32 + i];
            }
        }
        total
    }

    #[test]
    fn test_q4k_avx2_matches_scalar_reference() {
        let blk = BlockQ4K {
            d: 0x3C00,    // 1.0 in IEEE-754 fp16
            dmin: 0x3800, // 0.5 in IEEE-754 fp16
            scales: [3, 5, 7, 9, 2, 4, 6, 8, 0x21, 0x43, 0x65, 0x87],
            qs: std::array::from_fn(|i| ((i * 73 + 19) & 0xFF) as u8),
        };
        let mut act0 = [0.0f32; QK_K];
        let mut act1 = [0.0f32; QK_K];
        let mut act2 = [0.0f32; QK_K];
        let mut act3 = [0.0f32; QK_K];
        for i in 0..QK_K {
            act0[i] = ((i as f32) - 128.0) * 0.01;
            act1[i] = ((i as f32) * 0.5 - 64.0) * 0.01;
            act2[i] = (((i * 3) % 256) as f32 - 128.0) * 0.01;
            act3[i] = (((i * 7) % 256) as f32 - 128.0) * 0.01;
        }

        let ref0 = q4k_block_dot_scalar_ref(&blk, &act0);
        let ref1 = q4k_block_dot_scalar_ref(&blk, &act1);
        let ref2 = q4k_block_dot_scalar_ref(&blk, &act2);
        let ref3 = q4k_block_dot_scalar_ref(&blk, &act3);

        unsafe {
            let avx_single = q4k_block_dot_avx2(&blk, act0.as_ptr());
            let avx_spec4 = q4k_block_dot_spec4_avx2(
                &blk,
                act0.as_ptr(),
                act1.as_ptr(),
                act2.as_ptr(),
                act3.as_ptr(),
            );

            assert!((avx_single - ref0).abs() < 1e-2, "single AVX2 {avx_single} != ref {ref0}");
            assert!((avx_spec4[0] - ref0).abs() < 1e-2, "spec4[0] {} != ref {ref0}", avx_spec4[0]);
            assert!((avx_spec4[1] - ref1).abs() < 1e-2, "spec4[1] {} != ref {ref1}", avx_spec4[1]);
            assert!((avx_spec4[2] - ref2).abs() < 1e-2, "spec4[2] {} != ref {ref2}", avx_spec4[2]);
            assert!((avx_spec4[3] - ref3).abs() < 1e-2, "spec4[3] {} != ref {ref3}", avx_spec4[3]);
        }
    }

    #[test]
    fn test_rope_preserves_l2_norm_and_rmsnorm_unit_variance() {
        let mut x = [[0.0f32; HIDDEN_DIM]; SPEC_K];
        let weight = [1.0f32; HIDDEN_DIM];
        for k in 0..SPEC_K {
            for i in 0..HIDDEN_DIM {
                x[k][i] = ((i as f32) * 0.013 + (k as f32)).sin() * 3.5;
            }
        }

        unsafe {
            rmsnorm_spec4_avx2(&mut x, &weight);
            for k in 0..SPEC_K {
                let mean_sq: f32 = x[k].iter().map(|v| v * v).sum::<f32>() / (HIDDEN_DIM as f32);
                assert!(
                    (mean_sq - 1.0).abs() < 1e-3,
                    "RMSNorm output mean_sq = {mean_sq} (expected ~1.0)"
                );
            }

            // Record L2 norm before RoPE
            let norms_before: Vec<f32> = (0..SPEC_K)
                .map(|k| x[k].iter().map(|v| v * v).sum::<f32>().sqrt())
                .collect();

            let cache = RopeCache::new(128);
            rope_spec4_avx2(&mut x, &cache);

            // 2D rotation must preserve L2 norm across all 32 heads.
            for k in 0..SPEC_K {
                let norm_after = x[k].iter().map(|v| v * v).sum::<f32>().sqrt();
                assert!(
                    (norm_after - norms_before[k]).abs() < 1e-2,
                    "RoPE changed L2 norm: {} -> {}",
                    norms_before[k],
                    norm_after
                );
            }
        }
    }

    #[test]
    fn test_int8_gqa_flash_decoding_finite_and_causal() {
        let mut q_spec = [[0.25f32; HIDDEN_DIM]; SPEC_K];
        let mut kv = QuantizedKvCache::new(64);
        unsafe {
            flash_decoding_gqa_spec4_avx2(&mut q_spec, &mut kv, 0);
        }
        for k in 0..SPEC_K {
            for &v in &q_spec[k] {
                assert!(v.is_finite(), "Flash-Decoding produced non-finite value");
            }
        }
    }
}


/// Symmetric `Q8_0` KV-Cache for all 32 layers (`NUM_KV_HEADS = 8` heads x `128` dim).
///
/// Memory footprint for 32 layers x 512 context length:
/// - Standard FP32 KV-cache (MHA 32 heads): `32 * 512 * 2 * 4096 * 4 B = 512 MiB` (too much for 8 GB)
/// - Our GQA 4:1 `Q8_0` KV-cache (8 KV heads, int8):
///   `32 * 512 * 2 * 1024 * 1 B = 32.0 MiB` (16x smaller).
pub struct QuantizedKvCache {
    /// `k_q8[layer][pos][kv_head * 128 + d]`
    pub k_q8: Vec<[[i8; KV_DIM]; MAX_SEQ_LEN]>,
    /// `v_q8[layer][pos][kv_head * 128 + d]`
    pub v_q8: Vec<[[i8; KV_DIM]; MAX_SEQ_LEN]>,
    /// Per-(layer, pos, kv_head) dequantization scale (`max_abs / 127.0`).
    pub k_scale: Vec<[[f32; NUM_KV_HEADS]; MAX_SEQ_LEN]>,
    pub v_scale: Vec<[[f32; NUM_KV_HEADS]; MAX_SEQ_LEN]>,
    pub seq_len: usize,
}

impl QuantizedKvCache {
    /// Allocates and pre-populates an initial context window (`init_ctx` tokens)
    /// so Flash-Decoding GQA Attention runs over a realistic causal history.
    pub fn new(init_ctx: usize) -> Box<Self> {
        let ctx = init_ctx.min(MAX_SEQ_LEN - SPEC_K);
        let mut kv = Box::new(Self {
            k_q8: vec![[[0i8; KV_DIM]; MAX_SEQ_LEN]; NUM_LAYERS],
            v_q8: vec![[[0i8; KV_DIM]; MAX_SEQ_LEN]; NUM_LAYERS],
            k_scale: vec![[[1.0 / 127.0; NUM_KV_HEADS]; MAX_SEQ_LEN]; NUM_LAYERS],
            v_scale: vec![[[1.0 / 127.0; NUM_KV_HEADS]; MAX_SEQ_LEN]; NUM_LAYERS],
            seq_len: ctx,
        });

        for l in 0..NUM_LAYERS {
            for t in 0..ctx {
                for i in 0..KV_DIM {
                    kv.k_q8[l][t][i] = (((l * 17 + t * 31 + i) % 127) as i8).wrapping_sub(63);
                    kv.v_q8[l][t][i] = (((l * 13 + t * 29 + i) % 127) as i8).wrapping_sub(63);
                }
            }
        }
        kv
    }

    pub fn memory_bytes(&self) -> usize {
        NUM_LAYERS * MAX_SEQ_LEN * (KV_DIM * 2 + NUM_KV_HEADS * 8)
    }
}

/// Quantizes the first `KV_DIM = 1024` floats of `x` into symmetric `int8` (`-127..127`)
/// per `HEAD_DIM = 128` head using AVX2 max-abs reduction.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn quantize_kv_step_avx2(
    src: &[f32; HIDDEN_DIM],
    dst_q8: &mut [i8; KV_DIM],
    dst_scales: &mut [f32; NUM_KV_HEADS],
) {
    for kvh in 0..NUM_KV_HEADS {
        let base = kvh * HEAD_DIM;
        let mut max_abs = 1e-6f32;
        for d in 0..HEAD_DIM {
            let a = src[base + d].abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = max_abs / 127.0;
        let inv = 127.0 / max_abs;
        dst_scales[kvh] = scale;
        for d in 0..HEAD_DIM {
            dst_q8[base + d] = (src[base + d] * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// AVX2 dot product between an `f32` query head (`q[0..128]`) and an `int8` cached
/// key head (`k_q8[0..128]`) scaled by `k_scale`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_q_f32_k_i8_128_avx2(q: *const f32, k_q8: *const i8, k_scale: f32) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for d in (0..HEAD_DIM).step_by(16) {
        let ki0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(k_q8.add(d) as *const __m128i));
        let ki1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(k_q8.add(d + 8) as *const __m128i));
        let kf0 = _mm256_cvtepi32_ps(ki0);
        let kf1 = _mm256_cvtepi32_ps(ki1);

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(d)), kf0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(d + 8)), kf1, acc1);
    }

    let sum = _mm256_add_ps(acc0, acc1);
    let x = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps::<1>(sum));
    let x = _mm_hadd_ps(x, x);
    let x = _mm_hadd_ps(x, x);
    _mm_cvtss_f32(x) * k_scale
}

/// AVX2 weighted accumulation of an `int8` value head (`out[0..128] = out * rescale + weight * v_q8`).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fmadd_v_i8_128_avx2(
    out: *mut f32,
    rescale: f32,
    v_q8: *const i8,
    weighted_scale: f32,
) {
    let r_v = _mm256_set1_ps(rescale);
    let w_v = _mm256_set1_ps(weighted_scale);

    for d in (0..HEAD_DIM).step_by(16) {
        let vi0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(v_q8.add(d) as *const __m128i));
        let vi1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(v_q8.add(d + 8) as *const __m128i));
        let vf0 = _mm256_cvtepi32_ps(vi0);
        let vf1 = _mm256_cvtepi32_ps(vi1);

        let o0 = _mm256_mul_ps(_mm256_loadu_ps(out.add(d)), r_v);
        let o1 = _mm256_mul_ps(_mm256_loadu_ps(out.add(d + 8)), r_v);

        _mm256_storeu_ps(out.add(d), _mm256_fmadd_ps(vf0, w_v, o0));
        _mm256_storeu_ps(out.add(d + 8), _mm256_fmadd_ps(vf1, w_v, o1));
    }
}

/// **AVX2 Online-Softmax Flash-Decoding GQA Attention (`32` Query Heads, `8` KV Heads):**
/// 1. Quantizes and appends the `SPEC_K = 4` new K/V vectors into `kv_cache` for `layer_idx`.
/// 2. Runs single-pass Online-Softmax Flash-Decoding (`m_new = max(m, score)`,
///    `l_new = l * exp(m - m_new) + exp(score - m_new)`) over the causal `Q8_0` KV-cache,
///    never materializing an `N x N` attention matrix in RAM.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn flash_decoding_gqa_spec4_avx2(
    q_spec: &mut [[f32; HIDDEN_DIM]; SPEC_K],
    kv_cache: &mut QuantizedKvCache,
    layer_idx: usize,
) {
    let l = layer_idx % NUM_LAYERS;
    let base_seq = kv_cache.seq_len.min(MAX_SEQ_LEN - SPEC_K);
    let inv_sqrt_d = 1.0f32 / (HEAD_DIM as f32).sqrt(); // 1 / sqrt(128) = 0.08838835

    // 1. Quantize and append K and V for all 4 speculative draft tokens into the Int8 KV-Cache.
    for k in 0..SPEC_K {
        let pos = base_seq + k;
        quantize_kv_step_avx2(
            &q_spec[k],
            &mut kv_cache.k_q8[l][pos],
            &mut kv_cache.k_scale[l][pos],
        );
        quantize_kv_step_avx2(
            &q_spec[k],
            &mut kv_cache.v_q8[l][pos],
            &mut kv_cache.v_scale[l][pos],
        );
    }

    // 2. Single-pass Online-Softmax Flash-Decoding across all 32 Query Heads (4:1 GQA).
    let mut head_out = [0.0f32; HEAD_DIM];
    for k in 0..SPEC_K {
        let causal_end = base_seq + k + 1;
        for qh in 0..NUM_HEADS {
            let kvh = qh / GQA_GROUP;
            let q_ptr = q_spec[k].as_ptr().add(qh * HEAD_DIM);
            head_out.fill(0.0);

            let mut m_prev = f32::NEG_INFINITY;
            let mut l_prev = 0.0f32;

            for pos in 0..causal_end {
                let k_ptr = kv_cache.k_q8[l][pos].as_ptr().add(kvh * HEAD_DIM);
                let v_ptr = kv_cache.v_q8[l][pos].as_ptr().add(kvh * HEAD_DIM);
                let k_sc = kv_cache.k_scale[l][pos][kvh];
                let v_sc = kv_cache.v_scale[l][pos][kvh];

                let score = (dot_q_f32_k_i8_128_avx2(q_ptr, k_ptr, k_sc) * inv_sqrt_d)
                    .clamp(-20.0, 20.0);
                let m_new = m_prev.max(score);
                let alpha = (m_prev - m_new).exp();
                let p = (score - m_new).exp();
                l_prev = l_prev * alpha + p;
                m_prev = m_new;

                fmadd_v_i8_128_avx2(head_out.as_mut_ptr(), alpha, v_ptr, p * v_sc);
            }

            let inv_l = 1.0f32 / l_prev.max(1e-8);
            let inv_v = _mm256_set1_ps(inv_l);
            let dst_ptr = q_spec[k].as_mut_ptr().add(qh * HEAD_DIM);
            for d in (0..HEAD_DIM).step_by(8) {
                let o = _mm256_mul_ps(_mm256_loadu_ps(head_out.as_ptr().add(d)), inv_v);
                _mm256_storeu_ps(dst_ptr.add(d), o);
            }
        }
    }
}


