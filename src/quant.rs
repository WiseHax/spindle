//! Quantized weight formats: block layouts, AVX2 dot-product kernels, row matvec dispatch,
//! and a scalar reference dequantizer the kernels are tested against.
//!
//! Supported: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K`,
//! `Q6_K`. Layouts follow ggml.

use std::arch::x86_64::*;

use crate::{
    gguf_loader::{GgmlType, GgufTensorInfo},
    q4k_gemv::BlockQ4K,
};

/// `Q8_0` block: 34 bytes per 32 weights (`d: f16` + `qs: [i8; 32]`).
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ8_0 {
    d: u16,
    qs: [i8; 32],
}

/// `Q5_0` block: 22 bytes per 32 weights (`d: f16` + `qh: [u8; 4]` + `qs: [u8; 16]`).
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ5_0 {
    d: u16,
    qh: [u8; 4],
    qs: [u8; 16],
}

/// `Q6_K` super-block: 210 bytes per 256 weights (`ql: [u8; 128]`, `qh: [u8; 64]`, `scales: [i8; 16]`, `d: f16`).
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ6K {
    ql: [u8; 128],
    qh: [u8; 64],
    scales: [i8; 16],
    d: u16,
}


#[inline]
#[target_feature(enable = "f16c")]
unsafe fn f16_to_f32_fast(h: u16) -> f32 {
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)))
}

/// AVX2 dot product of one `Q8_0` row (`n_cols` weights, multiple of 32) against `x[0..n_cols]`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q8_0_row_avx2(row_ptr: *const u8, x: *const f32, n_cols: usize) -> f32 {
    let n_blocks = n_cols / 32;
    let mut acc = _mm256_setzero_ps();

    for b in 0..n_blocks {
        let blk = &*(row_ptr.add(b * 34) as *const BlockQ8_0);
        let qp = blk.qs.as_ptr();
        let xp = x.add(b * 32);

        // Sum q*x for the block first, then scale once by `d`.
        let mut s = _mm256_setzero_ps();
        for sub in (0..32).step_by(8) {
            let qi = _mm256_cvtepi8_epi32(_mm_loadl_epi64(qp.add(sub) as *const __m128i));
            s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(qi), _mm256_loadu_ps(xp.add(sub)), s);
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk.d)), s, acc);
    }

    hsum_avx2(acc)
}

/// Interleaved 2-row AVX2 dot product for `Q8_0`.
/// Keeps activations hot in L1, loads `x` once for two rows, and dual-issues FMAs across Port 0 & 1.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q8_0_2rows_avx2(
    row0_ptr: *const u8,
    row1_ptr: *const u8,
    x: *const f32,
    n_cols: usize,
) -> (f32, f32) {
    let n_blocks = n_cols / 32;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for b in 0..n_blocks {
        let blk0 = &*(row0_ptr.add(b * 34) as *const BlockQ8_0);
        let blk1 = &*(row1_ptr.add(b * 34) as *const BlockQ8_0);
        let qp0 = blk0.qs.as_ptr();
        let qp1 = blk1.qs.as_ptr();
        let xp = x.add(b * 32);

        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        for sub in (0..32).step_by(8) {
            let x_vec = _mm256_loadu_ps(xp.add(sub));
            let qi0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qp0.add(sub) as *const __m128i)));
            let qi1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qp1.add(sub) as *const __m128i)));
            s0 = _mm256_fmadd_ps(qi0, x_vec, s0);
            s1 = _mm256_fmadd_ps(qi1, x_vec, s1);
        }
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk0.d)), s0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk1.d)), s1, acc1);
    }

    (hsum_avx2(acc0), hsum_avx2(acc1))
}

/// Converts 16 signed bytes to 2x8 f32 and accumulates `bytes * x[0..16]` into `acc`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fma_i8x16(v: __m128i, x: *const f32, acc: __m256) -> __m256 {
    let a = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
    let b = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(v)));
    _mm256_fmadd_ps(b, _mm256_loadu_ps(x.add(8)), _mm256_fmadd_ps(a, _mm256_loadu_ps(x), acc))
}

/// AVX2 dot product of one `Q5_0` row (`n_cols` weights, multiple of 32) against `x[0..n_cols]`.
/// In `ggml` `block_q5_0`:
/// - Low 4 bits of `qs[j]` (`j = 0..16`) + bit `j` of `qh` -> weight `j` (`0..16`) minus 16
/// - High 4 bits of `qs[j]` (`j = 0..16`) + bit `j+16` of `qh` -> weight `j+16` (`16..32`) minus 16
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q5_0_row_avx2(row_ptr: *const u8, x: *const f32, n_cols: usize) -> f32 {
    let n_blocks = n_cols / 32;
    let mut acc = _mm256_setzero_ps();

    let m4 = _mm_set1_epi8(0x0F);
    // Expands the 32 `qh` bits to 32 bytes (0xFF where the bit is set), as in ggml.
    let shuf = _mm256_set_epi64x(
        0x0303030303030303,
        0x0202020202020202,
        0x0101010101010101,
        0x0000000000000000,
    );
    let bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe_u64 as i64);
    let all_ones = _mm256_set1_epi64x(-1);
    let hi_bit = _mm256_set1_epi8(0x10);
    let bias = _mm256_set1_epi8(16);

    for b in 0..n_blocks {
        let blk = &*(row_ptr.add(b * 22) as *const BlockQ5_0);
        let qh = u32::from_le_bytes(blk.qh);
        let xp = x.add(b * 32);

        let qs = _mm_loadu_si128(blk.qs.as_ptr() as *const __m128i);
        let lo = _mm_and_si128(qs, m4);
        let hi = _mm_and_si128(_mm_srli_epi16::<4>(qs), m4);
        let nib = _mm256_set_m128i(hi, lo);

        let bits = _mm256_shuffle_epi8(_mm256_set1_epi32(qh as i32), shuf);
        let bits = _mm256_cmpeq_epi8(_mm256_or_si256(bits, bit_mask), all_ones);
        let q = _mm256_sub_epi8(_mm256_or_si256(nib, _mm256_and_si256(bits, hi_bit)), bias);

        let mut s = fma_i8x16(_mm256_castsi256_si128(q), xp, _mm256_setzero_ps());
        s = fma_i8x16(_mm256_extracti128_si256::<1>(q), xp.add(16), s);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk.d)), s, acc);
    }

    hsum_avx2(acc)
}

/// Precomputes scalar sums of each 32-float subblock of `x`.
/// Used by `Q4_K` and `Q5_K` to eliminate redundant activation additions across rows.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn compute_subblock_sums_32(x: *const f32, n_cols: usize, out_sums: *mut f32) {
    let n_sub = n_cols / 32;
    for sb in 0..n_sub {
        let p = x.add(sb * 32);
        let s0 = _mm256_loadu_ps(p);
        let s1 = _mm256_loadu_ps(p.add(8));
        let s2 = _mm256_loadu_ps(p.add(16));
        let s3 = _mm256_loadu_ps(p.add(24));
        let sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        *out_sums.add(sb) = hsum_avx2(sum);
    }
}

/// Accumulates 16 unsigned 4-bit values (one per byte) times `x[0..16]` into `sqx`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fma_u8x16(v: __m128i, x: *const f32, sqx: &mut __m256) {
    let x0 = _mm256_loadu_ps(x);
    let x1 = _mm256_loadu_ps(x.add(8));
    let a = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v));
    let b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v)));
    *sqx = _mm256_fmadd_ps(b, x1, _mm256_fmadd_ps(a, x0, *sqx));
}

/// Accumulates 16 unsigned 4-bit values for two independent rows times shared `x[0..16]` into `sqx0` and `sqx1`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fma_u8x16_pair(
    v0: __m128i,
    v1: __m128i,
    x: *const f32,
    sqx0: &mut __m256,
    sqx1: &mut __m256,
) {
    let x0 = _mm256_loadu_ps(x);
    let x1 = _mm256_loadu_ps(x.add(8));

    let a0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v0));
    let b0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v0)));
    *sqx0 = _mm256_fmadd_ps(b0, x1, _mm256_fmadd_ps(a0, x0, *sqx0));

    let a1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v1));
    let b1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v1)));
    *sqx1 = _mm256_fmadd_ps(b1, x1, _mm256_fmadd_ps(a1, x0, *sqx1));
}

/// Single-row AVX2 dot product for `Q4_K` using precomputed activation subblock sums.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q4_k_row_avx2_presum(
    row_ptr: *const u8,
    x: *const f32,
    act_sums: *const f32,
    n_cols: usize,
) -> f32 {
    let n_blocks = n_cols / 256;
    let m4 = _mm_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    let mut total_min = 0.0f32;

    for b in 0..n_blocks {
        let blk = &*(row_ptr.add(b * 144) as *const BlockQ4K);
        let d = f16_to_f32_fast(blk.d);
        let dmin = f16_to_f32_fast(blk.dmin);
        let (sc, mn) = crate::q4k_gemv::unpack_q4k_scales(&blk.scales);
        let xb = x.add(b * 256);

        let sb_base = b * 8;
        let mut min_term = 0.0f32;
        for k in 0..8 {
            min_term += mn[k] as f32 * *act_sums.add(sb_base + k);
        }
        total_min += min_term * dmin;

        for chunk in 0..4 {
            let qp = blk.qs.as_ptr().add(chunk * 32);
            let q0 = _mm_loadu_si128(qp as *const __m128i);
            let q1 = _mm_loadu_si128(qp.add(16) as *const __m128i);
            let a1 = xb.add(chunk * 64);
            let a2 = xb.add(chunk * 64 + 32);

            let mut sqx0 = _mm256_setzero_ps();
            fma_u8x16(_mm_and_si128(q0, m4), a1, &mut sqx0);
            fma_u8x16(_mm_and_si128(q1, m4), a1.add(16), &mut sqx0);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc[chunk * 2] as f32), sqx0, acc);

            let mut sqx1 = _mm256_setzero_ps();
            fma_u8x16(_mm_and_si128(_mm_srli_epi16::<4>(q0), m4), a2, &mut sqx1);
            fma_u8x16(_mm_and_si128(_mm_srli_epi16::<4>(q1), m4), a2.add(16), &mut sqx1);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc[chunk * 2 + 1] as f32), sqx1, acc);
        }
    }

    hsum_avx2(acc) - total_min
}

/// Interleaved 2-row AVX2 dot product for `Q4_K` with precomputed activation sums.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q4_k_2rows_avx2_presum(
    row0_ptr: *const u8,
    row1_ptr: *const u8,
    x: *const f32,
    act_sums: *const f32,
    n_cols: usize,
) -> (f32, f32) {
    let n_blocks = n_cols / 256;
    let m4 = _mm_set1_epi8(0x0F);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut total_min0 = 0.0f32;
    let mut total_min1 = 0.0f32;

    for b in 0..n_blocks {
        let blk0 = &*(row0_ptr.add(b * 144) as *const BlockQ4K);
        let blk1 = &*(row1_ptr.add(b * 144) as *const BlockQ4K);

        let d0 = f16_to_f32_fast(blk0.d);
        let dmin0 = f16_to_f32_fast(blk0.dmin);
        let (sc0, mn0) = crate::q4k_gemv::unpack_q4k_scales(&blk0.scales);

        let d1 = f16_to_f32_fast(blk1.d);
        let dmin1 = f16_to_f32_fast(blk1.dmin);
        let (sc1, mn1) = crate::q4k_gemv::unpack_q4k_scales(&blk1.scales);

        let xb = x.add(b * 256);
        let sb_base = b * 8;

        let mut min0 = 0.0f32;
        let mut min1 = 0.0f32;
        for k in 0..8 {
            let s = *act_sums.add(sb_base + k);
            min0 += mn0[k] as f32 * s;
            min1 += mn1[k] as f32 * s;
        }
        total_min0 += min0 * dmin0;
        total_min1 += min1 * dmin1;

        for chunk in 0..4 {
            let qp0 = blk0.qs.as_ptr().add(chunk * 32);
            let qp1 = blk1.qs.as_ptr().add(chunk * 32);

            let q0_0 = _mm_loadu_si128(qp0 as *const __m128i);
            let q0_1 = _mm_loadu_si128(qp0.add(16) as *const __m128i);

            let q1_0 = _mm_loadu_si128(qp1 as *const __m128i);
            let q1_1 = _mm_loadu_si128(qp1.add(16) as *const __m128i);

            let a1 = xb.add(chunk * 64);
            let a2 = xb.add(chunk * 64 + 32);

            let mut sqx0 = _mm256_setzero_ps();
            let mut sqx1 = _mm256_setzero_ps();
            fma_u8x16_pair(_mm_and_si128(q0_0, m4), _mm_and_si128(q1_0, m4), a1, &mut sqx0, &mut sqx1);
            fma_u8x16_pair(_mm_and_si128(q0_1, m4), _mm_and_si128(q1_1, m4), a1.add(16), &mut sqx0, &mut sqx1);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0 * sc0[chunk * 2] as f32), sqx0, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1 * sc1[chunk * 2] as f32), sqx1, acc1);

            let mut sqx0 = _mm256_setzero_ps();
            let mut sqx1 = _mm256_setzero_ps();
            fma_u8x16_pair(
                _mm_and_si128(_mm_srli_epi16::<4>(q0_0), m4),
                _mm_and_si128(_mm_srli_epi16::<4>(q1_0), m4),
                a2,
                &mut sqx0,
                &mut sqx1,
            );
            fma_u8x16_pair(
                _mm_and_si128(_mm_srli_epi16::<4>(q0_1), m4),
                _mm_and_si128(_mm_srli_epi16::<4>(q1_1), m4),
                a2.add(16),
                &mut sqx0,
                &mut sqx1,
            );
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0 * sc0[chunk * 2 + 1] as f32), sqx0, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1 * sc1[chunk * 2 + 1] as f32), sqx1, acc1);
        }
    }

    (hsum_avx2(acc0) - total_min0, hsum_avx2(acc1) - total_min1)
}

/// Fallback single-row dot product for `Q4_K` (computes activation sums on the fly).
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q4_k_row_avx2(row_ptr: *const u8, x: *const f32, n_cols: usize) -> f32 {
    let mut sums = [0.0f32; 512];
    compute_subblock_sums_32(x, n_cols, sums.as_mut_ptr());
    dot_q4_k_row_avx2_presum(row_ptr, x, sums.as_ptr(), n_cols)
}

/// Dot product of one `Q6_K` row (`n_cols` weights, multiple of 256) against `x[0..n_cols]`.
/// Decodes 8 weights at a time in 32-bit lanes (same bit layout as ggml's reference loop).
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q6_k_row_avx2(row_ptr: *const u8, x: *const f32, n_cols: usize) -> f32 {
    let n_blocks = n_cols / 256;
    let mut total = _mm256_setzero_ps();
    let m4 = _mm256_set1_epi32(0x0F);
    let m3 = _mm256_set1_epi32(3);
    let off = _mm256_set1_epi32(32);

    for b in 0..n_blocks {
        let blk = &*(row_ptr.add(b * 210) as *const BlockQ6K);
        let mut yp = x.add(b * 256);
        let mut ql = blk.ql.as_ptr();
        let mut qh = blk.qh.as_ptr();
        let mut sc = blk.scales.as_ptr();
        let mut acc = _mm256_setzero_ps();

        for _ in (0..256).step_by(128) {
            for l in (0..32).step_by(8) {
                let is = l / 16;
                let load8 = |p: *const u8| _mm256_cvtepu8_epi32(_mm_loadl_epi64(p as *const __m128i));
                let ql0 = load8(ql.add(l));
                let ql32 = load8(ql.add(l + 32));
                let h = load8(qh.add(l));

                let q1 = _mm256_sub_epi32(
                    _mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_slli_epi32::<4>(_mm256_and_si256(h, m3))),
                    off,
                );
                let q2 = _mm256_sub_epi32(
                    _mm256_or_si256(
                        _mm256_and_si256(ql32, m4),
                        _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<2>(h), m3)),
                    ),
                    off,
                );
                let q3 = _mm256_sub_epi32(
                    _mm256_or_si256(
                        _mm256_srli_epi32::<4>(ql0),
                        _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<4>(h), m3)),
                    ),
                    off,
                );
                let q4 = _mm256_sub_epi32(
                    _mm256_or_si256(
                        _mm256_srli_epi32::<4>(ql32),
                        _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<6>(h), m3)),
                    ),
                    off,
                );

                let term = |q: __m256i, s: i8, y: *const f32, acc: __m256| {
                    let w = _mm256_mul_ps(_mm256_cvtepi32_ps(q), _mm256_set1_ps(s as f32));
                    _mm256_fmadd_ps(w, _mm256_loadu_ps(y), acc)
                };
                acc = term(q1, *sc.add(is), yp.add(l), acc);
                acc = term(q2, *sc.add(is + 2), yp.add(l + 32), acc);
                acc = term(q3, *sc.add(is + 4), yp.add(l + 64), acc);
                acc = term(q4, *sc.add(is + 6), yp.add(l + 96), acc);
            }
            yp = yp.add(128);
            ql = ql.add(64);
            qh = qh.add(32);
            sc = sc.add(8);
        }
        total = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk.d)), acc, total);
    }

    hsum_avx2(total)
}

/// Interleaved 2-row AVX2 dot product for `Q6_K`.
/// Loads activation `x` once for two rows, interleaving FMAs across Port 0 & 1.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q6_k_2rows_avx2(
    row0_ptr: *const u8,
    row1_ptr: *const u8,
    x: *const f32,
    n_cols: usize,
) -> (f32, f32) {
    let n_blocks = n_cols / 256;
    let mut total0 = _mm256_setzero_ps();
    let mut total1 = _mm256_setzero_ps();
    let m4 = _mm256_set1_epi32(0x0F);
    let m3 = _mm256_set1_epi32(3);
    let off = _mm256_set1_epi32(32);

    for b in 0..n_blocks {
        let blk0 = &*(row0_ptr.add(b * 210) as *const BlockQ6K);
        let blk1 = &*(row1_ptr.add(b * 210) as *const BlockQ6K);
        let mut yp = x.add(b * 256);
        let mut ql0 = blk0.ql.as_ptr();
        let mut qh0 = blk0.qh.as_ptr();
        let mut sc0 = blk0.scales.as_ptr();
        let mut ql1 = blk1.ql.as_ptr();
        let mut qh1 = blk1.qh.as_ptr();
        let mut sc1 = blk1.scales.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        for _ in (0..256).step_by(128) {
            for l in (0..32).step_by(8) {
                let is = l / 16;
                let load8 = |p: *const u8| _mm256_cvtepu8_epi32(_mm_loadl_epi64(p as *const __m128i));

                let y0 = _mm256_loadu_ps(yp.add(l));
                let y1 = _mm256_loadu_ps(yp.add(l + 32));
                let y2 = _mm256_loadu_ps(yp.add(l + 64));
                let y3 = _mm256_loadu_ps(yp.add(l + 96));

                // Row 0
                let ql0_0 = load8(ql0.add(l));
                let ql32_0 = load8(ql0.add(l + 32));
                let h0 = load8(qh0.add(l));

                let q1_0 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(ql0_0, m4), _mm256_slli_epi32::<4>(_mm256_and_si256(h0, m3))), off);
                let q2_0 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(ql32_0, m4), _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<2>(h0), m3))), off);
                let q3_0 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32::<4>(ql0_0), _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<4>(h0), m3))), off);
                let q4_0 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32::<4>(ql32_0), _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<6>(h0), m3))), off);

                let w1_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q1_0), _mm256_set1_ps(*sc0.add(is) as f32));
                let w2_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q2_0), _mm256_set1_ps(*sc0.add(is + 2) as f32));
                let w3_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q3_0), _mm256_set1_ps(*sc0.add(is + 4) as f32));
                let w4_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q4_0), _mm256_set1_ps(*sc0.add(is + 6) as f32));

                acc0 = _mm256_fmadd_ps(w1_0, y0, acc0);
                acc0 = _mm256_fmadd_ps(w2_0, y1, acc0);
                acc0 = _mm256_fmadd_ps(w3_0, y2, acc0);
                acc0 = _mm256_fmadd_ps(w4_0, y3, acc0);

                // Row 1
                let ql0_1 = load8(ql1.add(l));
                let ql32_1 = load8(ql1.add(l + 32));
                let h1 = load8(qh1.add(l));

                let q1_1 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(ql0_1, m4), _mm256_slli_epi32::<4>(_mm256_and_si256(h1, m3))), off);
                let q2_1 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(ql32_1, m4), _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<2>(h1), m3))), off);
                let q3_1 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32::<4>(ql0_1), _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<4>(h1), m3))), off);
                let q4_1 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32::<4>(ql32_1), _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<6>(h1), m3))), off);

                let w1_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q1_1), _mm256_set1_ps(*sc1.add(is) as f32));
                let w2_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q2_1), _mm256_set1_ps(*sc1.add(is + 2) as f32));
                let w3_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q3_1), _mm256_set1_ps(*sc1.add(is + 4) as f32));
                let w4_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q4_1), _mm256_set1_ps(*sc1.add(is + 6) as f32));

                acc1 = _mm256_fmadd_ps(w1_1, y0, acc1);
                acc1 = _mm256_fmadd_ps(w2_1, y1, acc1);
                acc1 = _mm256_fmadd_ps(w3_1, y2, acc1);
                acc1 = _mm256_fmadd_ps(w4_1, y3, acc1);
            }
            yp = yp.add(128);
            ql0 = ql0.add(64);
            qh0 = qh0.add(32);
            sc0 = sc0.add(8);
            ql1 = ql1.add(64);
            qh1 = qh1.add(32);
            sc1 = sc1.add(8);
        }
        total0 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk0.d)), acc0, total0);
        total1 = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk1.d)), acc1, total1);
    }

    (hsum_avx2(total0), hsum_avx2(total1))
}

/// Plain f32 dot product.
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let main = n & !7;
    for i in (0..main).step_by(8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc);
    }
    let mut tail = 0.0;
    for i in main..n {
        tail += *a.add(i) * *b.add(i);
    }
    hsum_avx2(acc) + tail
}

#[inline(always)]
pub(crate) fn f16_to_f32_exact(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let x = _mm_hadd_ps(x, x);
    let x = _mm_hadd_ps(x, x);
    _mm_cvtss_f32(x)
}

/// A 2D weight matrix inside the mapped file: `rows` rows of `cols` weights of `dtype`,
/// starting `offset` bytes into the file. For a 3D expert tensor (`[cols, rows, n_expert]`)
/// `expert(e)` gives the view of one expert's slab.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mat {
    pub dtype: GgmlType,
    pub offset: usize,
    pub cols: usize,
    pub rows: usize,
}

impl Mat {
    pub fn of(info: &GgufTensorInfo) -> Self {
        Self {
            dtype: info.dtype,
            offset: info.abs_file_offset as usize,
            cols: info.dims[0] as usize,
            rows: info.dims.get(1).copied().unwrap_or(1) as usize,
        }
    }

    pub fn expert(self, e: usize) -> Self {
        let rb = row_bytes(self.dtype, self.cols).expect("unsupported expert tensor type");
        Self { offset: self.offset + e * self.rows * rb, ..self }
    }
}

/// Computes rows `r0..r0 + out.len()` of `W * x` into `out`.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn matvec_rows_avx2(base_ptr: *const u8, m: &Mat, x: &[f32], out: &mut [f32], r0: usize) {
    let n_cols = m.cols;
    debug_assert_eq!(x.len(), n_cols);

    let tensor_ptr = base_ptr.wrapping_add(m.offset);
    let xp = x.as_ptr();
    let n_rows = out.len();

    macro_rules! rows_1x {
        ($row_bytes:expr, $dot:ident) => {
            for (i, dst) in out.iter_mut().enumerate() {
                *dst = $dot(tensor_ptr.add((r0 + i) * $row_bytes), xp, n_cols);
            }
        };
    }

    macro_rules! rows_2x {
        ($row_bytes:expr, $dot_pair:ident, $dot_single:ident) => {{
            let mut i = 0;
            while i + 1 < n_rows {
                let p0 = tensor_ptr.add((r0 + i) * $row_bytes);
                let p1 = tensor_ptr.add((r0 + i + 1) * $row_bytes);
                let (v0, v1) = $dot_pair(p0, p1, xp, n_cols);
                out[i] = v0;
                out[i + 1] = v1;
                i += 2;
            }
            if i < n_rows {
                let p0 = tensor_ptr.add((r0 + i) * $row_bytes);
                out[i] = $dot_single(p0, xp, n_cols);
            }
        }};
    }

    match m.dtype {
        GgmlType::Q8_0 => rows_2x!((n_cols / 32) * 34, dot_q8_0_2rows_avx2, dot_q8_0_row_avx2),
        GgmlType::Q4_0 => rows_1x!((n_cols / 32) * 18, dot_q4_0_row_avx2),
        GgmlType::Q4_1 => rows_1x!((n_cols / 32) * 20, dot_q4_1_row_avx2),
        GgmlType::Q5_0 => rows_1x!((n_cols / 32) * 22, dot_q5_0_row_avx2),
        GgmlType::Q5_1 => rows_1x!((n_cols / 32) * 24, dot_q5_1_row_avx2),
        GgmlType::Q4K => {
            let n_sub = n_cols / 32;
            let mut act_sums_stack = [0.0f32; 512];
            let mut act_sums_heap = Vec::new();
            let act_sums_ptr = if n_sub <= 512 {
                compute_subblock_sums_32(xp, n_cols, act_sums_stack.as_mut_ptr());
                act_sums_stack.as_ptr()
            } else {
                act_sums_heap.resize(n_sub, 0.0f32);
                compute_subblock_sums_32(xp, n_cols, act_sums_heap.as_mut_ptr());
                act_sums_heap.as_ptr()
            };

            let row_bytes = (n_cols / 256) * 144;
            let mut i = 0;
            while i + 1 < n_rows {
                let p0 = tensor_ptr.add((r0 + i) * row_bytes);
                let p1 = tensor_ptr.add((r0 + i + 1) * row_bytes);
                let (v0, v1) = dot_q4_k_2rows_avx2_presum(p0, p1, xp, act_sums_ptr, n_cols);
                out[i] = v0;
                out[i + 1] = v1;
                i += 2;
            }
            if i < n_rows {
                let p0 = tensor_ptr.add((r0 + i) * row_bytes);
                out[i] = dot_q4_k_row_avx2_presum(p0, xp, act_sums_ptr, n_cols);
            }
        }
        GgmlType::Q5K => {
            let n_sub = n_cols / 32;
            let mut act_sums_stack = [0.0f32; 512];
            let mut act_sums_heap = Vec::new();
            let act_sums_ptr = if n_sub <= 512 {
                compute_subblock_sums_32(xp, n_cols, act_sums_stack.as_mut_ptr());
                act_sums_stack.as_ptr()
            } else {
                act_sums_heap.resize(n_sub, 0.0f32);
                compute_subblock_sums_32(xp, n_cols, act_sums_heap.as_mut_ptr());
                act_sums_heap.as_ptr()
            };

            let row_bytes = (n_cols / 256) * 176;
            let mut i = 0;
            while i + 1 < n_rows {
                let p0 = tensor_ptr.add((r0 + i) * row_bytes);
                let p1 = tensor_ptr.add((r0 + i + 1) * row_bytes);
                let (v0, v1) = dot_q5_k_2rows_avx2_presum(p0, p1, xp, act_sums_ptr, n_cols);
                out[i] = v0;
                out[i + 1] = v1;
                i += 2;
            }
            if i < n_rows {
                let p0 = tensor_ptr.add((r0 + i) * row_bytes);
                out[i] = dot_q5_k_row_avx2_presum(p0, xp, act_sums_ptr, n_cols);
            }
        }
        GgmlType::Q6K => rows_2x!((n_cols / 256) * 210, dot_q6_k_2rows_avx2, dot_q6_k_row_avx2),
        GgmlType::F16 => rows_2x!(n_cols * 2, dot_f16_2rows_avx2, dot_f16_row_avx2),
        GgmlType::BF16 => rows_1x!(n_cols * 2, dot_bf16_row_avx2),
        GgmlType::F32 => {
            let mut i = 0;
            while i + 1 < n_rows {
                let p0 = tensor_ptr.add((r0 + i) * n_cols * 4) as *const f32;
                let p1 = tensor_ptr.add((r0 + i + 1) * n_cols * 4) as *const f32;
                let (v0, v1) = dot_f32_2rows_avx2(p0, p1, xp, n_cols);
                out[i] = v0;
                out[i + 1] = v1;
                i += 2;
            }
            if i < n_rows {
                let p0 = tensor_ptr.add((r0 + i) * n_cols * 4) as *const f32;
                out[i] = dot_f32_avx2(p0, xp, n_cols);
            }
        }
        other => panic!("unsupported GGUF tensor dtype: {other:?}"),
    }
}

// ---------------------------------------------------------------- more formats

/// `Q4_0`: 18 bytes per 32 weights. Weight = (nibble - 8) * d.
#[repr(C)]
struct BlockQ4_0 {
    d: u16,
    qs: [u8; 16],
}

/// `Q4_1`: 20 bytes per 32 weights. Weight = nibble * d + m.
#[repr(C)]
struct BlockQ4_1 {
    d: u16,
    m: u16,
    qs: [u8; 16],
}

/// `Q5_1`: 24 bytes per 32 weights. Weight = (nibble | high bit << 4) * d + m.
#[repr(C)]
struct BlockQ5_1 {
    d: u16,
    m: u16,
    qh: [u8; 4],
    qs: [u8; 16],
}

/// `Q5_K`: 176 bytes per 256 weights. Same 6-bit scale/min packing as `Q4_K`, plus one extra
/// bit per weight in `qh`. Each 64-weight chunk `c` takes its high bits from bit `2c` (low
/// nibbles) and `2c + 1` (high nibbles) of `qh[l]`.
#[repr(C)]
struct BlockQ5K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qh: [u8; 32],
    qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);
const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);
const _: () = assert!(std::mem::size_of::<BlockQ5K>() == 176);

/// 16 packed bytes -> 32 bytes holding one nibble each, in weight order: low nibbles of
/// `qs[0..16]` are weights 0..16, high nibbles are weights 16..32.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn nibbles32(qs: *const u8) -> __m256i {
    let v = _mm_loadu_si128(qs as *const __m128i);
    let m4 = _mm_set1_epi8(0x0F);
    _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16::<4>(v), m4), _mm_and_si128(v, m4))
}

/// Byte `j` of the result is 0xFF if bit `j` of `bits` is set (same trick as ggml).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn bits_to_bytes(bits: u32) -> __m256i {
    let shuf = _mm256_set_epi64x(
        0x0303030303030303,
        0x0202020202020202,
        0x0101010101010101,
        0x0000000000000000,
    );
    let spread = _mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), shuf);
    let probe = _mm256_or_si256(spread, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe_u64 as i64));
    _mm256_cmpeq_epi8(probe, _mm256_set1_epi64x(-1))
}

/// Signed bytes times `x[0..32]`, added to `acc`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fma_i8x32(q: __m256i, x: *const f32, acc: __m256) -> __m256 {
    let acc = fma_i8x16(_mm256_castsi256_si128(q), x, acc);
    fma_i8x16(_mm256_extracti128_si256::<1>(q), x.add(16), acc)
}

/// Unsigned bytes times `x[0..32]` into `sqx`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fma_u8x32(q: __m256i, x: *const f32, sqx: &mut __m256) {
    fma_u8x16(_mm256_castsi256_si128(q), x, sqx);
    fma_u8x16(_mm256_extracti128_si256::<1>(q), x.add(16), sqx);
}

/// Unsigned bytes times `x[0..32]` for two rows into `sqx0` and `sqx1`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fma_u8x32_pair(
    q0: __m256i,
    q1: __m256i,
    x: *const f32,
    sqx0: &mut __m256,
    sqx1: &mut __m256,
) {
    fma_u8x16_pair(
        _mm256_castsi256_si128(q0),
        _mm256_castsi256_si128(q1),
        x,
        sqx0,
        sqx1,
    );
    fma_u8x16_pair(
        _mm256_extracti128_si256::<1>(q0),
        _mm256_extracti128_si256::<1>(q1),
        x.add(16),
        sqx0,
        sqx1,
    );
}

#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q4_0_row_avx2(row_ptr: *const u8, x: *const f32, n_cols: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let eight = _mm256_set1_epi8(8);
    for b in 0..n_cols / 32 {
        let blk = &*(row_ptr.add(b * 18) as *const BlockQ4_0);
        let q = _mm256_sub_epi8(nibbles32(blk.qs.as_ptr()), eight);
        let s = fma_i8x32(q, x.add(b * 32), _mm256_setzero_ps());
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk.d)), s, acc);
    }
    hsum_avx2(acc)
}

/// `sum((q*d + m) * x) = d*sum(q*x) + m*sum(x)`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q4_1_row_avx2(row_ptr: *const u8, x: *const f32, n_cols: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for b in 0..n_cols / 32 {
        let blk = &*(row_ptr.add(b * 20) as *const BlockQ4_1);
        let mut sqx = _mm256_setzero_ps();
        fma_u8x32(nibbles32(blk.qs.as_ptr()), x.add(b * 32), &mut sqx);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk.d)), sqx, acc);

        let xp = x.add(b * 32);
        let s0 = _mm256_loadu_ps(xp);
        let s1 = _mm256_loadu_ps(xp.add(8));
        let s2 = _mm256_loadu_ps(xp.add(16));
        let s3 = _mm256_loadu_ps(xp.add(24));
        let sx = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk.m)), sx, acc);
    }
    hsum_avx2(acc)
}

#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q5_1_row_avx2(row_ptr: *const u8, x: *const f32, n_cols: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let hi_bit = _mm256_set1_epi8(0x10);
    for b in 0..n_cols / 32 {
        let blk = &*(row_ptr.add(b * 24) as *const BlockQ5_1);
        let high = _mm256_and_si256(bits_to_bytes(u32::from_le_bytes(blk.qh)), hi_bit);
        let q = _mm256_or_si256(nibbles32(blk.qs.as_ptr()), high);
        let mut sqx = _mm256_setzero_ps();
        fma_u8x32(q, x.add(b * 32), &mut sqx);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk.d)), sqx, acc);

        let xp = x.add(b * 32);
        let s0 = _mm256_loadu_ps(xp);
        let s1 = _mm256_loadu_ps(xp.add(8));
        let s2 = _mm256_loadu_ps(xp.add(16));
        let s3 = _mm256_loadu_ps(xp.add(24));
        let sx = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16_to_f32_fast(blk.m)), sx, acc);
    }
    hsum_avx2(acc)
}

/// Single-row AVX2 dot product for `Q5_K` using precomputed activation sums.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q5_k_row_avx2_presum(
    row_ptr: *const u8,
    x: *const f32,
    act_sums: *const f32,
    n_cols: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut total_min = 0.0f32;
    let m4 = _mm256_set1_epi8(0x0F);
    let hi_bit = _mm256_set1_epi8(0x10);

    for b in 0..n_cols / 256 {
        let blk = &*(row_ptr.add(b * 176) as *const BlockQ5K);
        let d = f16_to_f32_fast(blk.d);
        let dmin = f16_to_f32_fast(blk.dmin);
        let (sc, mn) = crate::q4k_gemv::unpack_q4k_scales(&blk.scales);
        let qh = _mm256_loadu_si256(blk.qh.as_ptr() as *const __m256i);
        let xb = x.add(b * 256);

        let sb_base = b * 8;
        let mut min_term = 0.0f32;
        for k in 0..8 {
            min_term += mn[k] as f32 * *act_sums.add(sb_base + k);
        }
        total_min += min_term * dmin;

        for c in 0..4 {
            let q = _mm256_loadu_si256(blk.qs.as_ptr().add(c * 32) as *const __m256i);
            let halves = [
                (_mm256_and_si256(q, m4), 1u8 << (2 * c)),
                (_mm256_and_si256(_mm256_srli_epi16::<4>(q), m4), 2u8 << (2 * c)),
            ];
            for (h, (nib, bit)) in halves.into_iter().enumerate() {
                let bit = _mm256_set1_epi8(bit as i8);
                let set = _mm256_cmpeq_epi8(_mm256_and_si256(qh, bit), bit);
                let q5 = _mm256_or_si256(nib, _mm256_and_si256(set, hi_bit));
                let mut sqx = _mm256_setzero_ps();
                fma_u8x32(q5, xb.add(c * 64 + h * 32), &mut sqx);
                let j = 2 * c + h;
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc[j] as f32), sqx, acc);
            }
        }
    }
    hsum_avx2(acc) - total_min
}

/// Interleaved 2-row AVX2 dot product for `Q5_K` with precomputed activation sums.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q5_k_2rows_avx2_presum(
    row0_ptr: *const u8,
    row1_ptr: *const u8,
    x: *const f32,
    act_sums: *const f32,
    n_cols: usize,
) -> (f32, f32) {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut total_min0 = 0.0f32;
    let mut total_min1 = 0.0f32;
    let m4 = _mm256_set1_epi8(0x0F);
    let hi_bit = _mm256_set1_epi8(0x10);

    for b in 0..n_cols / 256 {
        let blk0 = &*(row0_ptr.add(b * 176) as *const BlockQ5K);
        let blk1 = &*(row1_ptr.add(b * 176) as *const BlockQ5K);

        let d0 = f16_to_f32_fast(blk0.d);
        let dmin0 = f16_to_f32_fast(blk0.dmin);
        let (sc0, mn0) = crate::q4k_gemv::unpack_q4k_scales(&blk0.scales);

        let d1 = f16_to_f32_fast(blk1.d);
        let dmin1 = f16_to_f32_fast(blk1.dmin);
        let (sc1, mn1) = crate::q4k_gemv::unpack_q4k_scales(&blk1.scales);

        let qh0 = _mm256_loadu_si256(blk0.qh.as_ptr() as *const __m256i);
        let qh1 = _mm256_loadu_si256(blk1.qh.as_ptr() as *const __m256i);
        let xb = x.add(b * 256);
        let sb_base = b * 8;

        let mut min0 = 0.0f32;
        let mut min1 = 0.0f32;
        for k in 0..8 {
            let s = *act_sums.add(sb_base + k);
            min0 += mn0[k] as f32 * s;
            min1 += mn1[k] as f32 * s;
        }
        total_min0 += min0 * dmin0;
        total_min1 += min1 * dmin1;

        for c in 0..4 {
            let q0 = _mm256_loadu_si256(blk0.qs.as_ptr().add(c * 32) as *const __m256i);
            let q1 = _mm256_loadu_si256(blk1.qs.as_ptr().add(c * 32) as *const __m256i);

            let halves = [
                (
                    _mm256_and_si256(q0, m4),
                    _mm256_and_si256(q1, m4),
                    1u8 << (2 * c),
                ),
                (
                    _mm256_and_si256(_mm256_srli_epi16::<4>(q0), m4),
                    _mm256_and_si256(_mm256_srli_epi16::<4>(q1), m4),
                    2u8 << (2 * c),
                ),
            ];

            for (h, (nib0, nib1, bit)) in halves.into_iter().enumerate() {
                let bit_vec = _mm256_set1_epi8(bit as i8);
                let set0 = _mm256_cmpeq_epi8(_mm256_and_si256(qh0, bit_vec), bit_vec);
                let set1 = _mm256_cmpeq_epi8(_mm256_and_si256(qh1, bit_vec), bit_vec);

                let q5_0 = _mm256_or_si256(nib0, _mm256_and_si256(set0, hi_bit));
                let q5_1 = _mm256_or_si256(nib1, _mm256_and_si256(set1, hi_bit));

                let xp_sub = xb.add(c * 64 + h * 32);
                let mut sqx0 = _mm256_setzero_ps();
                let mut sqx1 = _mm256_setzero_ps();

                fma_u8x32_pair(q5_0, q5_1, xp_sub, &mut sqx0, &mut sqx1);

                let j = 2 * c + h;
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0 * sc0[j] as f32), sqx0, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1 * sc1[j] as f32), sqx1, acc1);
            }
        }
    }

    (hsum_avx2(acc0) - total_min0, hsum_avx2(acc1) - total_min1)
}

/// Fallback single-row dot product for `Q5_K` (computes activation sums on the fly).
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_q5_k_row_avx2(row_ptr: *const u8, x: *const f32, n_cols: usize) -> f32 {
    let mut sums = [0.0f32; 512];
    compute_subblock_sums_32(x, n_cols, sums.as_mut_ptr());
    dot_q5_k_row_avx2_presum(row_ptr, x, sums.as_ptr(), n_cols)
}

/// Needs F16C on top of AVX2 (every AVX2 Intel/AMD CPU has it; `main` checks anyway).
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
pub(crate) unsafe fn dot_f16_row_avx2(row_ptr: *const u8, x: *const f32, n_cols: usize) -> f32 {
    let w = row_ptr as *const u16;
    let mut acc = _mm256_setzero_ps();
    let main = n_cols & !7;
    for i in (0..main).step_by(8) {
        let wf = _mm256_cvtph_ps(_mm_loadu_si128(w.add(i) as *const __m128i));
        acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x.add(i)), acc);
    }
    let mut tail = 0.0;
    for i in main..n_cols {
        tail += f16_to_f32_fast(w.add(i).read_unaligned()) * *x.add(i);
    }
    hsum_avx2(acc) + tail
}

/// Interleaved 2-row AVX2 dot product for `F16`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_f16_2rows_avx2(
    row0_ptr: *const u8,
    row1_ptr: *const u8,
    x: *const f32,
    n_cols: usize,
) -> (f32, f32) {
    let w0 = row0_ptr as *const u16;
    let w1 = row1_ptr as *const u16;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let main = n_cols & !7;
    for i in (0..main).step_by(8) {
        let xv = _mm256_loadu_ps(x.add(i));
        let wf0 = _mm256_cvtph_ps(_mm_loadu_si128(w0.add(i) as *const __m128i));
        let wf1 = _mm256_cvtph_ps(_mm_loadu_si128(w1.add(i) as *const __m128i));
        acc0 = _mm256_fmadd_ps(wf0, xv, acc0);
        acc1 = _mm256_fmadd_ps(wf1, xv, acc1);
    }
    let mut tail0 = 0.0;
    let mut tail1 = 0.0;
    for i in main..n_cols {
        let xv = *x.add(i);
        tail0 += f16_to_f32_fast(w0.add(i).read_unaligned()) * xv;
        tail1 += f16_to_f32_fast(w1.add(i).read_unaligned()) * xv;
    }
    (hsum_avx2(acc0) + tail0, hsum_avx2(acc1) + tail1)
}

/// Interleaved 2-row AVX2 dot product for `F32`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_f32_2rows_avx2(
    a0: *const f32,
    a1: *const f32,
    b: *const f32,
    n: usize,
) -> (f32, f32) {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let main = n & !7;
    for i in (0..main).step_by(8) {
        let bv = _mm256_loadu_ps(b.add(i));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0.add(i)), bv, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1.add(i)), bv, acc1);
    }
    let mut tail0 = 0.0;
    let mut tail1 = 0.0;
    for i in main..n {
        let bv = *b.add(i);
        tail0 += *a0.add(i) * bv;
        tail1 += *a1.add(i) * bv;
    }
    (hsum_avx2(acc0) + tail0, hsum_avx2(acc1) + tail1)
}

/// bf16 is the top half of an f32, so widening is a 16-bit shift.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_bf16_row_avx2(row_ptr: *const u8, x: *const f32, n_cols: usize) -> f32 {
    let w = row_ptr as *const u16;
    let mut acc = _mm256_setzero_ps();
    let main = n_cols & !7;
    for i in (0..main).step_by(8) {
        let wide = _mm256_cvtepu16_epi32(_mm_loadu_si128(w.add(i) as *const __m128i));
        let wf = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(wide));
        acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x.add(i)), acc);
    }
    let mut tail = 0.0;
    for i in main..n_cols {
        tail += f32::from_bits((w.add(i).read_unaligned() as u32) << 16) * *x.add(i);
    }
    hsum_avx2(acc) + tail
}

/// f32 -> f16 for the KV-cache. `src.len()` must be a multiple of 8.
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    debug_assert!(src.len() % 8 == 0 && dst.len() == src.len());
    for i in (0..src.len()).step_by(8) {
        let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(src.as_ptr().add(i)));
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
    }
}

/// f16 -> f32. `src.len()` must be a multiple of 8.
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn f16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    debug_assert!(src.len() % 8 == 0 && dst.len() == src.len());
    for i in (0..src.len()).step_by(8) {
        let f = _mm256_cvtph_ps(_mm_loadu_si128(src.as_ptr().add(i) as *const __m128i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), f);
    }
}

// ---------------------------------------------------------------- reference path

/// Bytes per row of `n_cols` weights, or `None` for unsupported types / bad widths.
pub(crate) fn row_bytes(dtype: GgmlType, n_cols: usize) -> Option<usize> {
    let (block, bytes) = match dtype {
        GgmlType::F32 => (1, 4),
        GgmlType::F16 | GgmlType::BF16 => (1, 2),
        GgmlType::Q4_0 => (32, 18),
        GgmlType::Q4_1 => (32, 20),
        GgmlType::Q5_0 => (32, 22),
        GgmlType::Q5_1 => (32, 24),
        GgmlType::Q8_0 => (32, 34),
        GgmlType::Q4K => (256, 144),
        GgmlType::Q5K => (256, 176),
        GgmlType::Q6K => (256, 210),
        _ => return None,
    };
    (n_cols % block == 0).then_some(n_cols / block * bytes)
}

fn rd16(b: &[u8], off: usize) -> f32 {
    f16_to_f32_exact(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// Plain scalar dequantization of one row, written straight from the ggml format
/// definitions. Slow but obviously correct: the SIMD kernels are tested against it, and it
/// is also used to look up single embedding rows.
pub(crate) fn dequant_row(dtype: GgmlType, row: &[u8], n_cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n_cols];
    match dtype {
        GgmlType::F32 => {
            for (i, v) in y.iter_mut().enumerate() {
                *v = f32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (i, v) in y.iter_mut().enumerate() {
                *v = rd16(row, i * 2);
            }
        }
        GgmlType::BF16 => {
            for (i, v) in y.iter_mut().enumerate() {
                *v = f32::from_bits((u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]) as u32) << 16);
            }
        }
        GgmlType::Q8_0 => {
            for (b, out) in y.chunks_mut(32).enumerate() {
                let blk = &row[b * 34..];
                let d = rd16(blk, 0);
                for (j, v) in out.iter_mut().enumerate() {
                    *v = (blk[2 + j] as i8) as f32 * d;
                }
            }
        }
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 => {
            let (size, has_m, has_h) = match dtype {
                GgmlType::Q4_0 => (18, false, false),
                GgmlType::Q4_1 => (20, true, false),
                GgmlType::Q5_0 => (22, false, true),
                _ => (24, true, true),
            };
            for (b, out) in y.chunks_mut(32).enumerate() {
                let blk = &row[b * size..];
                let d = rd16(blk, 0);
                let m = if has_m { rd16(blk, 2) } else { 0.0 };
                let mut p = if has_m { 4 } else { 2 };
                let qh = if has_h {
                    let v = u32::from_le_bytes(blk[p..p + 4].try_into().unwrap());
                    p += 4;
                    v
                } else {
                    0
                };
                let qs = &blk[p..p + 16];
                for j in 0..16 {
                    let mut lo = (qs[j] & 0x0F) as i32;
                    let mut hi = (qs[j] >> 4) as i32;
                    if has_h {
                        lo |= (((qh >> j) & 1) << 4) as i32;
                        hi |= (((qh >> (j + 16)) & 1) << 4) as i32;
                    }
                    // Symmetric formats are centered; "_1" formats use the min `m` instead.
                    let center = if has_m { 0 } else if has_h { 16 } else { 8 };
                    out[j] = (lo - center) as f32 * d + m;
                    out[j + 16] = (hi - center) as f32 * d + m;
                }
            }
        }
        GgmlType::Q4K | GgmlType::Q5K => {
            let size = if dtype == GgmlType::Q4K { 144 } else { 176 };
            for (b, out) in y.chunks_mut(256).enumerate() {
                let blk = &row[b * size..];
                let d = rd16(blk, 0);
                let dmin = rd16(blk, 2);
                let (sc, mn) = crate::q4k_gemv::unpack_q4k_scales(blk[4..16].try_into().unwrap());
                let (qh, qs) = if size == 176 { (&blk[16..48], &blk[48..176]) } else { (&blk[0..0], &blk[16..144]) };
                for c in 0..4 {
                    for l in 0..32 {
                        let q = qs[c * 32 + l];
                        let (mut lo, mut hi) = ((q & 0x0F) as u32, (q >> 4) as u32);
                        if size == 176 {
                            lo |= ((qh[l] >> (2 * c)) as u32 & 1) << 4;
                            hi |= ((qh[l] >> (2 * c + 1)) as u32 & 1) << 4;
                        }
                        out[c * 64 + l] = d * sc[2 * c] as f32 * lo as f32 - dmin * mn[2 * c] as f32;
                        out[c * 64 + 32 + l] = d * sc[2 * c + 1] as f32 * hi as f32 - dmin * mn[2 * c + 1] as f32;
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for (b, out) in y.chunks_mut(256).enumerate() {
                let blk = &row[b * 210..];
                let (ql, qh) = (&blk[0..128], &blk[128..192]);
                let sc = |i: usize| blk[192 + i] as i8 as f32;
                let d = rd16(blk, 208);
                for half in 0..2 {
                    let (ql, qh, y) = (&ql[half * 64..], &qh[half * 32..], &mut out[half * 128..]);
                    let s = half * 8;
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        y[l] = d * sc(s + is) * q1 as f32;
                        y[l + 32] = d * sc(s + is + 2) * q2 as f32;
                        y[l + 64] = d * sc(s + is + 4) * q3 as f32;
                        y[l + 96] = d * sc(s + is + 6) * q4 as f32;
                    }
                }
            }
        }
        other => panic!("dequant_row: unsupported dtype {other:?}"),
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)
        }
    }

    /// f32 -> f16 for normal-range values (enough for test scales).
    fn f16(v: f32) -> [u8; 2] {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = (((b >> 23) & 0xff) as i32 - 127 + 15) as u16;
        let mant = ((b >> 13) & 0x3ff) as u16;
        (sign | (exp << 10) | mant).to_le_bytes()
    }

    /// Random row bytes for `dtype`, with every scale field set to a sane finite value.
    fn random_row(dtype: GgmlType, n: usize, rng: &mut Rng) -> Vec<u8> {
        let len = row_bytes(dtype, n).unwrap();
        let mut row: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        let (block_bytes, scale_offsets): (usize, &[usize]) = match dtype {
            GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q5_0 => (row_bytes(dtype, 32).unwrap(), &[0]),
            GgmlType::Q4_1 | GgmlType::Q5_1 => (row_bytes(dtype, 32).unwrap(), &[0, 2]),
            GgmlType::Q4K | GgmlType::Q5K => (row_bytes(dtype, 256).unwrap(), &[0, 2]),
            GgmlType::Q6K => (210, &[208]),
            GgmlType::F16 => (2, &[0]),
            GgmlType::BF16 => {
                for c in row.chunks_mut(2) {
                    c.copy_from_slice(&((rng.f(-1.0, 1.0).to_bits() >> 16) as u16).to_le_bytes());
                }
                return row;
            }
            GgmlType::F32 => {
                for c in row.chunks_mut(4) {
                    c.copy_from_slice(&rng.f(-1.0, 1.0).to_le_bytes());
                }
                return row;
            }
            _ => unreachable!(),
        };
        for blk in row.chunks_mut(block_bytes) {
            for &off in scale_offsets {
                let v = if dtype == GgmlType::F16 { rng.f(-1.0, 1.0) } else { rng.f(0.001, 0.05) };
                blk[off..off + 2].copy_from_slice(&f16(v));
            }
        }
        row
    }

    /// Runs the real `matvec_rows_avx2` dispatch on a multi-row matrix and checks every row
    /// against `dequant_row` + an f64 dot product.
    #[test]
    fn simd_kernels_match_scalar_reference() {
        assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
        let mut rng = Rng(0x2545F4914F6CDD1D);
        let types = [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
        ];
        for dtype in types {
            for n in [256usize, 768] {
                let rows = 5;
                let mut mat = Vec::new();
                for _ in 0..rows {
                    mat.extend(random_row(dtype, n, &mut rng));
                }
                let x: Vec<f32> = (0..n).map(|_| rng.f(-1.0, 1.0)).collect();
                let m = Mat { dtype, offset: 0, cols: n, rows };
                // Rows 1..5 through the r0 offset path.
                let mut out = vec![0.0f32; rows - 1];
                unsafe { matvec_rows_avx2(mat.as_ptr(), &m, &x, &mut out, 1) };

                let rb = row_bytes(dtype, n).unwrap();
                for (i, &got) in out.iter().enumerate() {
                    let r = i + 1;
                    let w = dequant_row(dtype, &mat[r * rb..(r + 1) * rb], n);
                    let want: f64 = w.iter().zip(&x).map(|(&a, &b)| a as f64 * b as f64).sum();
                    let scale: f64 = w.iter().zip(&x).map(|(&a, &b)| (a as f64 * b as f64).abs()).sum();
                    let err = (got as f64 - want).abs();
                    assert!(
                        err <= 1e-4 * scale + 1e-6,
                        "{dtype:?} n={n} row={r}: simd {got} vs reference {want} (err {err:e}, scale {scale:e})"
                    );
                }
            }
        }
    }
}
