//! Multi-vector matvec ("skinny GEMM"): `outs[b] = W * xs[b]` for a small batch of vectors.
//!
//! Single-token decode reads every weight once per token. When several tokens go through the
//! same matrix at once (speculative verify, prefill chunks), this kernel decodes each quant
//! block ONCE into an L1 buffer and multiplies it against up to `GROUP` activation vectors,
//! so weight bandwidth and dequant work are shared by the whole batch.
//!
//! Formats with a fused block decoder: `Q4_K`, `Q6_K`, `Q8_0`, `F32`. Anything else falls back
//! to one `matvec_rows_avx2` per vector (correct, just not shared).

use std::arch::x86_64::*;

use crate::{
    gguf_loader::GgmlType,
    quant::{matvec_rows_avx2, Mat},
};

/// Vectors handled per pass (8 YMM accumulators + 1 weight + 1 activation register).
pub(crate) const GROUP: usize = 8;

const T_Q4K: u8 = 0;
const T_Q6K: u8 = 1;
const T_Q8_0: u8 = 2;
const T_F32: u8 = 3;

#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn h2f(p: *const u8) -> f32 {
    let h = u16::from_le_bytes([*p, *p.add(1)]);
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum(v: __m256) -> f32 {
    let x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    let x = _mm_add_ss(x, _mm_shuffle_ps::<0x55>(x, x));
    _mm_cvtss_f32(x)
}

/// 16 nibble bytes -> `s * q - o` for 16 weights at `dst`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn store_u8x16(v: __m128i, s: __m256, o: __m256, dst: *mut f32) {
    let a = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v));
    let b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v)));
    _mm256_storeu_ps(dst, _mm256_fmsub_ps(s, a, o));
    _mm256_storeu_ps(dst.add(8), _mm256_fmsub_ps(s, b, o));
}

/// One `Q4_K` super-block (144 bytes) -> 256 f32. Same layout as `dequant_row`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn deq_q4k(blk: *const u8, w: *mut f32) {
    let d = h2f(blk);
    let dmin = h2f(blk.add(2));
    let (sc, mn) = crate::q4k_gemv::unpack_q4k_scales(&*(blk.add(4) as *const [u8; 12]));
    let qs = blk.add(16);
    let m4 = _mm_set1_epi8(0x0F);
    for c in 0..4 {
        let q0 = _mm_loadu_si128(qs.add(c * 32) as *const __m128i);
        let q1 = _mm_loadu_si128(qs.add(c * 32 + 16) as *const __m128i);
        let (s_lo, o_lo) = (_mm256_set1_ps(d * sc[2 * c] as f32), _mm256_set1_ps(dmin * mn[2 * c] as f32));
        let (s_hi, o_hi) = (_mm256_set1_ps(d * sc[2 * c + 1] as f32), _mm256_set1_ps(dmin * mn[2 * c + 1] as f32));
        let y = w.add(c * 64);
        store_u8x16(_mm_and_si128(q0, m4), s_lo, o_lo, y);
        store_u8x16(_mm_and_si128(q1, m4), s_lo, o_lo, y.add(16));
        store_u8x16(_mm_and_si128(_mm_srli_epi16::<4>(q0), m4), s_hi, o_hi, y.add(32));
        store_u8x16(_mm_and_si128(_mm_srli_epi16::<4>(q1), m4), s_hi, o_hi, y.add(48));
    }
}

/// One `Q6_K` super-block (210 bytes) -> 256 f32. Same layout as `dequant_row`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn deq_q6k(blk: *const u8, w: *mut f32) {
    let d = h2f(blk.add(208));
    let m4 = _mm256_set1_epi32(0x0F);
    let m3 = _mm256_set1_epi32(3);
    let k32 = _mm256_set1_epi32(32);
    for half in 0..2 {
        let ql = blk.add(half * 64);
        let qh = blk.add(128 + half * 32);
        let sc = blk.add(192 + half * 8) as *const i8;
        let y = w.add(half * 128);
        for l0 in (0..32).step_by(8) {
            let is = l0 / 16;
            let a = _mm256_cvtepu8_epi32(_mm_loadl_epi64(ql.add(l0) as *const __m128i));
            let b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(ql.add(32 + l0) as *const __m128i));
            let h = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qh.add(l0) as *const __m128i));
            let h0 = _mm256_slli_epi32::<4>(_mm256_and_si256(h, m3));
            let h2 = _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<2>(h), m3));
            let h4 = _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<4>(h), m3));
            let h6 = _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<6>(h), m3));
            let q1 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(a, m4), h0), k32);
            let q2 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(b, m4), h2), k32);
            let q3 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32::<4>(a), h4), k32);
            let q4 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32::<4>(b), h6), k32);
            let s0 = _mm256_set1_ps(d * *sc.add(is) as f32);
            let s2 = _mm256_set1_ps(d * *sc.add(is + 2) as f32);
            let s4 = _mm256_set1_ps(d * *sc.add(is + 4) as f32);
            let s6 = _mm256_set1_ps(d * *sc.add(is + 6) as f32);
            _mm256_storeu_ps(y.add(l0), _mm256_mul_ps(s0, _mm256_cvtepi32_ps(q1)));
            _mm256_storeu_ps(y.add(32 + l0), _mm256_mul_ps(s2, _mm256_cvtepi32_ps(q2)));
            _mm256_storeu_ps(y.add(64 + l0), _mm256_mul_ps(s4, _mm256_cvtepi32_ps(q3)));
            _mm256_storeu_ps(y.add(96 + l0), _mm256_mul_ps(s6, _mm256_cvtepi32_ps(q4)));
        }
    }
}

/// One `Q8_0` block (34 bytes) -> 32 f32.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn deq_q8_0(blk: *const u8, w: *mut f32) {
    let d = _mm256_set1_ps(h2f(blk));
    let qs = blk.add(2);
    for i in 0..4 {
        let q = _mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(i * 8) as *const __m128i));
        _mm256_storeu_ps(w.add(i * 8), _mm256_mul_ps(d, _mm256_cvtepi32_ps(q)));
    }
}

/// Rows `r0..r1` of `W * xs[b]` for exactly `B` vectors, writing `outs[b][r]`.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn rows_multi<const B: usize, const T: u8>(
    tensor: *const u8,
    n_cols: usize,
    xs: &[*const f32],
    outs: &[*mut f32],
    r0: usize,
    r1: usize,
) {
    let (bs, bbytes) = match T {
        T_Q4K => (256usize, 144usize),
        T_Q6K => (256, 210),
        T_Q8_0 => (32, 34),
        _ => (8, 32),
    };
    let rb = (n_cols / bs) * bbytes;
    let mut wbuf = [0.0f32; 256];
    let x: [*const f32; B] = std::array::from_fn(|b| xs[b]);
    for r in r0..r1 {
        let row = tensor.add(r * rb);
        // Two accumulator sets (even/odd 8-wide chunk) so small B still has enough
        // independent FMA chains to hide latency.
        let mut acc0 = [_mm256_setzero_ps(); B];
        let mut acc1 = [_mm256_setzero_ps(); B];
        if T == T_F32 {
            let wp = row as *const f32;
            let mut j = 0;
            while j + 16 <= n_cols {
                let w0 = _mm256_loadu_ps(wp.add(j));
                let w1 = _mm256_loadu_ps(wp.add(j + 8));
                for b in 0..B {
                    acc0[b] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x[b].add(j)), acc0[b]);
                    acc1[b] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x[b].add(j + 8)), acc1[b]);
                }
                j += 16;
            }
            while j < n_cols {
                let w0 = _mm256_loadu_ps(wp.add(j));
                for b in 0..B {
                    acc0[b] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x[b].add(j)), acc0[b]);
                }
                j += 8;
            }
        } else if T == T_Q8_0 {
            // Fused: integer weights times x per block, then one scale FMA per block and vector
            // (no f32 round trip through `wbuf`; Q8_0 is too cheap to decode for that to pay).
            for blk in 0..n_cols / 32 {
                let bp = row.add(blk * 34);
                let d = _mm256_set1_ps(h2f(bp));
                let qs = bp.add(2);
                let xo = blk * 32;
                let mut s = [_mm256_setzero_ps(); B];
                for i in 0..4 {
                    let w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(i * 8) as *const __m128i)));
                    for b in 0..B {
                        s[b] = _mm256_fmadd_ps(w, _mm256_loadu_ps(x[b].add(xo + i * 8)), s[b]);
                    }
                }
                for b in 0..B {
                    if blk & 1 == 0 {
                        acc0[b] = _mm256_fmadd_ps(d, s[b], acc0[b]);
                    } else {
                        acc1[b] = _mm256_fmadd_ps(d, s[b], acc1[b]);
                    }
                }
            }
        } else {
            let wb = wbuf.as_mut_ptr();
            for blk in 0..n_cols / bs {
                let bp = row.add(blk * bbytes);
                match T {
                    T_Q4K => deq_q4k(bp, wb),
                    T_Q6K => deq_q6k(bp, wb),
                    _ => deq_q8_0(bp, wb),
                }
                let xo = blk * bs;
                let mut j = 0;
                while j < bs {
                    let w0 = _mm256_loadu_ps(wb.add(j));
                    let w1 = _mm256_loadu_ps(wb.add(j + 8));
                    for b in 0..B {
                        acc0[b] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x[b].add(xo + j)), acc0[b]);
                        acc1[b] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x[b].add(xo + j + 8)), acc1[b]);
                    }
                    j += 16;
                }
            }
        }
        for b in 0..B {
            *outs[b].add(r) = hsum(_mm256_add_ps(acc0[b], acc1[b]));
        }
    }
}

/// Does `dtype` have a shared-decode kernel here? (Otherwise batching gives no weight reuse.)
pub(crate) fn has_multi_kernel(dtype: GgmlType, n_cols: usize) -> bool {
    match dtype {
        GgmlType::Q4K | GgmlType::Q6K => n_cols % 256 == 0,
        GgmlType::Q8_0 => n_cols % 32 == 0,
        GgmlType::F32 => n_cols % 8 == 0,
        _ => false,
    }
}

/// Rows `r0..r1` of `outs[b] = W * xs[b]` for every `b`. `xs[b]` has `m.cols` floats and
/// `outs[b]` points at the START of a full `m.rows` output (row `r` is written to `outs[b][r]`).
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn matmul_rows_avx2(base_ptr: *const u8, m: &Mat, xs: &[*const f32], outs: &[*mut f32], r0: usize, r1: usize) {
    debug_assert_eq!(xs.len(), outs.len());
    if r1 <= r0 || xs.is_empty() {
        return;
    }
    let tensor = base_ptr.wrapping_add(m.offset);
    let n_cols = m.cols;
    if !has_multi_kernel(m.dtype, n_cols) || xs.len() == 1 {
        for (x, o) in xs.iter().zip(outs) {
            let xsl = std::slice::from_raw_parts(*x, n_cols);
            let osl = std::slice::from_raw_parts_mut(o.add(r0), r1 - r0);
            matvec_rows_avx2(base_ptr, m, xsl, osl, r0);
        }
        return;
    }
    macro_rules! go {
        ($t:expr, $xs:expr, $os:expr) => {
            match $xs.len() {
                1 => rows_multi::<1, { $t }>(tensor, n_cols, $xs, $os, r0, r1),
                2 => rows_multi::<2, { $t }>(tensor, n_cols, $xs, $os, r0, r1),
                3 => rows_multi::<3, { $t }>(tensor, n_cols, $xs, $os, r0, r1),
                4 => rows_multi::<4, { $t }>(tensor, n_cols, $xs, $os, r0, r1),
                5 => rows_multi::<5, { $t }>(tensor, n_cols, $xs, $os, r0, r1),
                6 => rows_multi::<6, { $t }>(tensor, n_cols, $xs, $os, r0, r1),
                7 => rows_multi::<7, { $t }>(tensor, n_cols, $xs, $os, r0, r1),
                _ => rows_multi::<8, { $t }>(tensor, n_cols, $xs, $os, r0, r1),
            }
        };
    }
    let mut g = 0;
    while g < xs.len() {
        let e = (g + GROUP).min(xs.len());
        let (gx, go_) = (&xs[g..e], &outs[g..e]);
        match m.dtype {
            GgmlType::Q4K => go!(T_Q4K, gx, go_),
            GgmlType::Q6K => go!(T_Q6K, gx, go_),
            GgmlType::Q8_0 => go!(T_Q8_0, gx, go_),
            _ => go!(T_F32, gx, go_),
        }
        g = e;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{dequant_row, row_bytes};

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

    fn f16(v: f32) -> [u8; 2] {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = (((b >> 23) & 0xff) as i32 - 127 + 15) as u16;
        let mant = ((b >> 13) & 0x3ff) as u16;
        (sign | (exp << 10) | mant).to_le_bytes()
    }

    fn random_matrix(dtype: GgmlType, n: usize, rows: usize, rng: &mut Rng) -> Vec<u8> {
        let rb = row_bytes(dtype, n).unwrap();
        let mut m: Vec<u8> = (0..rb * rows).map(|_| rng.next() as u8).collect();
        let (bb, offs): (usize, &[usize]) = match dtype {
            GgmlType::Q8_0 => (34, &[0]),
            GgmlType::Q4K => (144, &[0, 2]),
            GgmlType::Q6K => (210, &[208]),
            GgmlType::F32 => {
                for c in m.chunks_mut(4) {
                    c.copy_from_slice(&rng.f(-1.0, 1.0).to_le_bytes());
                }
                return m;
            }
            _ => unreachable!(),
        };
        for blk in m.chunks_mut(bb) {
            for &o in offs {
                blk[o..o + 2].copy_from_slice(&f16(rng.f(0.001, 0.05)));
            }
        }
        m
    }

    /// Throughput check (run with `cargo test --release bench_multi -- --ignored --nocapture`):
    /// cost of B vectors through the shared-decode kernel vs B separate single-vector matvecs.
    #[test]
    #[ignore]
    fn bench_multi_vs_serial() {
        let mut rng = Rng(7);
        for (dtype, n, rows, name) in [
            (GgmlType::Q4K, 2048usize, 768usize * 16, "Q4_K expert gate/up (2048 cols)"),
            (GgmlType::Q6K, 768, 2048 * 16, "Q6_K expert down (768 cols)"),
            (GgmlType::Q6K, 2048, 16384, "Q6_K LM-head slice (2048 cols)"),
            (GgmlType::Q8_0, 2048, 8192, "Q8_0 (2048 cols)"),
        ] {
            let mat = random_matrix(dtype, n, rows, &mut rng);
            let m = Mat { dtype, offset: 0, cols: n, rows };
            println!("{name}: {rows} rows, {:.1} MB", mat.len() as f64 / 1e6);
            let mut base_us = 0.0;
            for bsz in [1usize, 2, 3, 4, 5, 8] {
                let xs: Vec<Vec<f32>> = (0..bsz).map(|_| (0..n).map(|_| rng.f(-1.0, 1.0)).collect()).collect();
                let mut outs = vec![vec![0.0f32; rows]; bsz];
                let xp: Vec<*const f32> = xs.iter().map(|v| v.as_ptr()).collect();
                let op: Vec<*mut f32> = outs.iter_mut().map(|v| v.as_mut_ptr()).collect();
                let iters = 20;
                let t = std::time::Instant::now();
                for _ in 0..iters {
                    for b in 0..bsz {
                        unsafe { matvec_rows_avx2(mat.as_ptr(), &m, &xs[b], &mut outs[b], 0) };
                    }
                }
                let serial = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
                let t = std::time::Instant::now();
                for _ in 0..iters {
                    unsafe { matmul_rows_avx2(mat.as_ptr(), &m, &xp, &op, 0, rows) };
                }
                let multi = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
                if bsz == 1 {
                    base_us = serial;
                }
                println!(
                    "  B={bsz}: serial {serial:8.0} us | multi {multi:8.0} us | multi cost = {:.2}x of one token",
                    multi / base_us
                );
            }
        }
    }

    /// Every batch size 1..=11 (crosses the 8-vector group boundary) and every row tail must
    /// match the f64 scalar reference built from `dequant_row`.
    #[test]
    fn multi_matvec_matches_scalar_reference() {
        assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for dtype in [GgmlType::Q4K, GgmlType::Q6K, GgmlType::Q8_0, GgmlType::F32, GgmlType::Q5K] {
            for n in [256usize, 768, 2048] {
                let rows = 7;
                let mat = if dtype == GgmlType::Q5K {
                    // No fused kernel: exercises the per-vector fallback path.
                    let rb = row_bytes(dtype, n).unwrap();
                    let mut m: Vec<u8> = (0..rb * rows).map(|_| rng.next() as u8).collect();
                    for blk in m.chunks_mut(176) {
                        blk[0..2].copy_from_slice(&f16(rng.f(0.001, 0.05)));
                        blk[2..4].copy_from_slice(&f16(rng.f(0.001, 0.05)));
                    }
                    m
                } else {
                    random_matrix(dtype, n, rows, &mut rng)
                };
                let m = Mat { dtype, offset: 0, cols: n, rows };
                let rb = row_bytes(dtype, n).unwrap();
                let wrows: Vec<Vec<f32>> = (0..rows).map(|r| dequant_row(dtype, &mat[r * rb..(r + 1) * rb], n)).collect();
                for bsz in 1..=11usize {
                    let xs: Vec<Vec<f32>> = (0..bsz).map(|_| (0..n).map(|_| rng.f(-1.0, 1.0)).collect()).collect();
                    let mut outs = vec![vec![f32::NAN; rows]; bsz];
                    let xp: Vec<*const f32> = xs.iter().map(|v| v.as_ptr()).collect();
                    let op: Vec<*mut f32> = outs.iter_mut().map(|v| v.as_mut_ptr()).collect();
                    // Split rows like the worker pool does: [0..3) and [3..7).
                    unsafe {
                        matmul_rows_avx2(mat.as_ptr(), &m, &xp, &op, 0, 3);
                        matmul_rows_avx2(mat.as_ptr(), &m, &xp, &op, 3, rows);
                    }
                    for b in 0..bsz {
                        for r in 0..rows {
                            let want: f64 = wrows[r].iter().zip(&xs[b]).map(|(&a, &x)| a as f64 * x as f64).sum();
                            let scale: f64 = wrows[r].iter().zip(&xs[b]).map(|(&a, &x)| (a as f64 * x as f64).abs()).sum();
                            let got = outs[b][r] as f64;
                            assert!(
                                (got - want).abs() <= 1e-4 * scale + 1e-6,
                                "{dtype:?} n={n} bsz={bsz} b={b} r={r}: got {got} want {want}"
                            );
                        }
                    }
                }
            }
        }
    }
}
