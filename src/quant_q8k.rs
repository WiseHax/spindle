//! Integer matvec for K-quant weights: the activation vector is quantized to 8 bits once
//! (`Q8_K`: per 256 values one f32 scale, 256 int8 and 16 sums of 16), then each weight block
//! is multiplied in integers (`vpmaddubsw` u8 × i8 → i16, `vpmaddwd` → i32) with one float
//! multiply per block. Weight nibbles are never converted to float.
//!
//! Formats: `Q4_K` and `Q6_K` (everything else uses the float kernels in `quant`).
//! The result differs from the float path by the activation rounding (8 bits per value with
//! a per-256 scale), the same scheme llama.cpp's CPU backend uses for K-quants. It can be
//! turned off with `--float-kernels` or `SPINDLE_FLOAT_KERNELS=1`.

use std::arch::x86_64::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::{gguf_loader::GgmlType, quant::Mat};

pub(crate) const QK: usize = 256;

/// 0 = not decided yet (read `SPINDLE_FLOAT_KERNELS` on first use), 1 = on, 2 = off.
static ENABLED: AtomicU8 = AtomicU8::new(0);

pub(crate) fn set_enabled(on: bool) {
    ENABLED.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

thread_local! {
    /// Per-thread override (tests): `Some(false)` forces the float kernels for matvecs this
    /// thread dispatches. The dispatching thread alone decides, so pool workers follow it.
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_thread_override(on: Option<bool>) {
    OVERRIDE.with(|o| o.set(on));
}

#[inline]
pub(crate) fn enabled() -> bool {
    if let Some(on) = OVERRIDE.with(|o| o.get()) {
        return on;
    }
    match ENABLED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var_os("SPINDLE_FLOAT_KERNELS").map_or(true, |v| v.is_empty() || v == "0");
            set_enabled(on);
            on
        }
    }
}

/// Whether `m` runs on the integer kernels.
#[inline]
pub(crate) fn supports(m: &Mat) -> bool {
    enabled() && matches!(m.dtype, GgmlType::Q4K | GgmlType::Q6K) && m.cols % QK == 0 && m.cols > 0
}

/// One activation vector in `Q8_K` form.
#[derive(Default, Clone)]
pub(crate) struct Q8K {
    pub d: Vec<f32>,
    pub qs: Vec<i8>,
    pub bsums: Vec<i16>,
}

impl Q8K {
    pub fn len(&self) -> usize {
        self.qs.len()
    }

    /// Quantizes `x` (length a multiple of 256) into `self`, reusing its buffers.
    pub fn quantize(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len() % QK, 0);
        let nb = x.len() / QK;
        self.d.resize(nb, 0.0);
        self.qs.resize(x.len(), 0);
        self.bsums.resize(nb * 16, 0);
        unsafe { quantize_avx2(x, &mut self.d, &mut self.qs, &mut self.bsums) };
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_avx2(x: &[f32], d: &mut [f32], qs: &mut [i8], bsums: &mut [i16]) {
    let sign = _mm256_set1_ps(-0.0);
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (b, xb) in x.chunks_exact(QK).enumerate() {
        let xp = xb.as_ptr();
        let mut amax = _mm256_setzero_ps();
        for j in (0..QK).step_by(8) {
            amax = _mm256_max_ps(amax, _mm256_andnot_ps(sign, _mm256_loadu_ps(xp.add(j))));
        }
        let m = {
            let v = _mm_max_ps(_mm256_castps256_ps128(amax), _mm256_extractf128_ps::<1>(amax));
            let v = _mm_max_ps(v, _mm_movehl_ps(v, v));
            let v = _mm_max_ss(v, _mm_shuffle_ps::<0x55>(v, v));
            _mm_cvtss_f32(v)
        };
        let out = qs.as_mut_ptr().add(b * QK);
        if !(m > 0.0) || !m.is_finite() {
            d[b] = 0.0;
            std::ptr::write_bytes(out, 0, QK);
            bsums[b * 16..b * 16 + 16].fill(0);
            continue;
        }
        let iscale = _mm256_set1_ps(127.0 / m);
        d[b] = m / 127.0;
        for j in (0..QK).step_by(32) {
            let q = |o: usize| _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xp.add(j + o)), iscale));
            let (a, bq, c, e) = (q(0), q(8), q(16), q(24));
            let ab = _mm256_packs_epi32(a, bq);
            let ce = _mm256_packs_epi32(c, e);
            let v = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(ab, ce), perm);
            _mm256_storeu_si256(out.add(j) as *mut __m256i, v);
        }
        let q = std::slice::from_raw_parts(out, QK);
        for (g, s) in bsums[b * 16..b * 16 + 16].iter_mut().enumerate() {
            *s = q[g * 16..g * 16 + 16].iter().map(|&v| v as i16).sum();
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<Vec<Q8K>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` with `n` reusable `Q8K` buffers of this thread.
pub(crate) fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Q8K]) -> R) -> R {
    SCRATCH.with(|s| match s.try_borrow_mut() {
        Ok(mut v) => {
            if v.len() < n {
                v.resize_with(n, Q8K::default);
            }
            f(&mut v[..n])
        }
        // Re-entered on this thread (not expected): use fresh buffers.
        Err(_) => {
            let mut v = vec![Q8K::default(); n];
            f(&mut v)
        }
    })
}

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

const T_Q4K: u8 = 0;
const T_Q6K: u8 = 1;

/// Rows `r0..r1` of `W · x_b` for `B` quantized vectors; row `r` of vector `b` goes to
/// `outs[b].add(r)`.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn rows<const B: usize, const T: u8>(tensor: *const u8, n_cols: usize, xq: &[&Q8K], outs: &[*mut f32], r0: usize, r1: usize) {
    let nb = n_cols / QK;
    let bbytes = if T == T_Q4K { 144 } else { 210 };
    let rb = nb * bbytes;
    let m4 = _mm256_set1_epi8(0x0F);
    let m3 = _mm256_set1_epi8(3);
    let xd: [*const f32; B] = std::array::from_fn(|b| xq[b].d.as_ptr());
    let xs: [*const i8; B] = std::array::from_fn(|b| xq[b].qs.as_ptr());
    let xb: [*const i16; B] = std::array::from_fn(|b| xq[b].bsums.as_ptr());
    for r in r0..r1 {
        let row = tensor.add(r * rb);
        let mut acc = [_mm256_setzero_ps(); B];
        let mut corr = [0.0f32; B];
        for i in 0..nb {
            let blk = row.add(i * bbytes);
            // Decode the block's 256 weights once into 8 byte vectors and 8 scale vectors.
            let mut w = [_mm256_setzero_si256(); 8];
            let mut sv = [_mm256_setzero_si256(); 8];
            let d;
            if T == T_Q4K {
                d = h2f(blk);
                let dmin = h2f(blk.add(2));
                let (sc, mn) = crate::q4k_gemv::unpack_q4k_scales(&*(blk.add(4) as *const [u8; 12]));
                let qs = blk.add(16);
                for c in 0..4 {
                    let v = _mm256_loadu_si256(qs.add(c * 32) as *const __m256i);
                    w[2 * c] = _mm256_and_si256(v, m4);
                    w[2 * c + 1] = _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4);
                    sv[2 * c] = _mm256_set1_epi16(sc[2 * c] as i16);
                    sv[2 * c + 1] = _mm256_set1_epi16(sc[2 * c + 1] as i16);
                }
                // Mins: dmin · Σ_j m_j · (sum of x over sub-block j), per vector.
                for b in 0..B {
                    let bs = xb[b].add(i * 16);
                    let mut s = 0i32;
                    for j in 0..8 {
                        s += mn[j] as i32 * (*bs.add(2 * j) as i32 + *bs.add(2 * j + 1) as i32);
                    }
                    corr[b] += dmin * *xd[b].add(i) * s as f32;
                }
            } else {
                d = h2f(blk.add(208));
                let sc = blk.add(192) as *const i8;
                for h in 0..2 {
                    let ql = blk.add(h * 64);
                    let qh = _mm256_loadu_si256(blk.add(128 + h * 32) as *const __m256i);
                    let l0 = _mm256_loadu_si256(ql as *const __m256i);
                    let l1 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
                    let hi = |s: __m256i| _mm256_slli_epi16::<4>(_mm256_and_si256(s, m3));
                    w[4 * h] = _mm256_or_si256(_mm256_and_si256(l0, m4), hi(qh));
                    w[4 * h + 1] = _mm256_or_si256(_mm256_and_si256(l1, m4), hi(_mm256_srli_epi16::<2>(qh)));
                    w[4 * h + 2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(l0), m4), hi(_mm256_srli_epi16::<4>(qh)));
                    w[4 * h + 3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(l1), m4), hi(_mm256_srli_epi16::<6>(qh)));
                    for k in 0..4 {
                        let s_lo = *sc.add(8 * h + 2 * k) as i16;
                        let s_hi = *sc.add(8 * h + 2 * k + 1) as i16;
                        sv[4 * h + k] = _mm256_setr_m128i(_mm_set1_epi16(s_lo), _mm_set1_epi16(s_hi));
                    }
                }
                // Weights are q - 32: subtract 32 · Σ_s scale_s · (sum of x over group s).
                for b in 0..B {
                    let bs = xb[b].add(i * 16);
                    let mut s = 0i32;
                    for g in 0..16 {
                        s += *sc.add(g) as i32 * *bs.add(g) as i32;
                    }
                    corr[b] += d * *xd[b].add(i) * (32 * s) as f32;
                }
            }
            for b in 0..B {
                let y = xs[b].add(i * QK);
                let mut sumi = _mm256_setzero_si256();
                for k in 0..8 {
                    let yv = _mm256_loadu_si256(y.add(k * 32) as *const __m256i);
                    let p = _mm256_maddubs_epi16(w[k], yv);
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, sv[k]));
                }
                acc[b] = _mm256_fmadd_ps(_mm256_set1_ps(d * *xd[b].add(i)), _mm256_cvtepi32_ps(sumi), acc[b]);
            }
        }
        for b in 0..B {
            *outs[b].add(r) = hsum(acc[b]) - corr[b];
        }
    }
}

/// Rows `r0..r1` of `outs[b] = W · xq[b]`. `outs[b]` points at the start of a full output
/// (row `r` is written to `outs[b][r]`). `m` must satisfy `supports`.
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn matmul_rows(base: *const u8, m: &Mat, xq: &[&Q8K], outs: &[*mut f32], r0: usize, r1: usize) {
    debug_assert_eq!(xq.len(), outs.len());
    if r1 <= r0 || xq.is_empty() {
        return;
    }
    let tensor = base.wrapping_add(m.offset);
    let n = m.cols;
    macro_rules! go {
        ($t:expr, $x:expr, $o:expr) => {
            match $x.len() {
                1 => rows::<1, { $t }>(tensor, n, $x, $o, r0, r1),
                2 => rows::<2, { $t }>(tensor, n, $x, $o, r0, r1),
                3 => rows::<3, { $t }>(tensor, n, $x, $o, r0, r1),
                4 => rows::<4, { $t }>(tensor, n, $x, $o, r0, r1),
                5 => rows::<5, { $t }>(tensor, n, $x, $o, r0, r1),
                6 => rows::<6, { $t }>(tensor, n, $x, $o, r0, r1),
                7 => rows::<7, { $t }>(tensor, n, $x, $o, r0, r1),
                _ => rows::<8, { $t }>(tensor, n, $x, $o, r0, r1),
            }
        };
    }
    let mut g = 0;
    while g < xq.len() {
        let e = (g + 8).min(xq.len());
        let (gx, go_) = (&xq[g..e], &outs[g..e]);
        match m.dtype {
            GgmlType::Q4K => go!(T_Q4K, gx, go_),
            _ => go!(T_Q6K, gx, go_),
        }
        g = e;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{dequant_row, matvec_rows_avx2, row_bytes};

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
            GgmlType::Q4K => (144, &[0, 2]),
            _ => (210, &[208]),
        };
        for blk in m.chunks_mut(bb) {
            for &o in offs {
                blk[o..o + 2].copy_from_slice(&f16(rng.f(0.001, 0.05)));
            }
        }
        m
    }

    fn dequant_q8k(q: &Q8K) -> Vec<f32> {
        q.qs.iter().enumerate().map(|(i, &v)| q.d[i / QK] * v as f32).collect()
    }

    #[test]
    fn quantize_round_trip_and_sums() {
        let mut rng = Rng(11);
        let x: Vec<f32> = (0..768).map(|_| rng.f(-3.0, 3.0)).collect();
        let mut q = Q8K::default();
        q.quantize(&x);
        let back = dequant_q8k(&q);
        for (b, chunk) in x.chunks(QK).enumerate() {
            let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (j, &v) in chunk.iter().enumerate() {
                assert!((back[b * QK + j] - v).abs() <= amax / 127.0 * 0.5 + 1e-6);
            }
        }
        for g in 0..q.bsums.len() {
            let s: i32 = q.qs[g * 16..g * 16 + 16].iter().map(|&v| v as i32).sum();
            assert_eq!(q.bsums[g] as i32, s);
        }
        // All zeros.
        q.quantize(&[0.0; 256]);
        assert!(q.qs.iter().all(|&v| v == 0) && q.d[0] == 0.0);
    }

    /// The integer kernel must equal the exact dot of the dequantized weights with the
    /// dequantized Q8_K vector (it is the same math, only reordered), for every batch size
    /// and a split row range.
    #[test]
    fn int_kernel_matches_reference() {
        assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
        let mut rng = Rng(0x51ED);
        for dtype in [GgmlType::Q4K, GgmlType::Q6K] {
            for n in [256usize, 768, 2048] {
                let rows = 7;
                let mat = random_matrix(dtype, n, rows, &mut rng);
                let m = Mat { dtype, offset: 0, cols: n, rows };
                let rb = row_bytes(dtype, n).unwrap();
                let wrows: Vec<Vec<f32>> = (0..rows).map(|r| dequant_row(dtype, &mat[r * rb..(r + 1) * rb], n)).collect();
                for bsz in 1..=11usize {
                    let xs: Vec<Vec<f32>> = (0..bsz).map(|_| (0..n).map(|_| rng.f(-2.0, 2.0)).collect()).collect();
                    let qs: Vec<Q8K> = xs
                        .iter()
                        .map(|x| {
                            let mut q = Q8K::default();
                            q.quantize(x);
                            q
                        })
                        .collect();
                    let qr: Vec<&Q8K> = qs.iter().collect();
                    let mut outs = vec![vec![f32::NAN; rows]; bsz];
                    let op: Vec<*mut f32> = outs.iter_mut().map(|v| v.as_mut_ptr()).collect();
                    unsafe {
                        matmul_rows(mat.as_ptr(), &m, &qr, &op, 0, 3);
                        matmul_rows(mat.as_ptr(), &m, &qr, &op, 3, rows);
                    }
                    for b in 0..bsz {
                        let xq = dequant_q8k(&qs[b]);
                        for r in 0..rows {
                            let want: f64 = wrows[r].iter().zip(&xq).map(|(&a, &x)| a as f64 * x as f64).sum();
                            let scale: f64 = wrows[r].iter().zip(&xq).map(|(&a, &x)| (a as f64 * x as f64).abs()).sum();
                            let got = outs[b][r] as f64;
                            assert!((got - want).abs() <= 1e-4 * scale + 1e-6, "{dtype:?} n={n} bsz={bsz} b={b} r={r}: got {got} want {want}");
                            // And close to the float kernel on the unrounded x.
                            let exact: f64 = wrows[r].iter().zip(&xs[b]).map(|(&a, &x)| a as f64 * x as f64).sum();
                            assert!((got - exact).abs() <= 0.02 * scale + 1e-4, "{dtype:?} rounding too large: {got} vs {exact}");
                        }
                    }
                }
            }
        }
    }

    /// Speculative verify relies on this: a vector gives bit-identical results whether it goes
    /// through the kernel alone or inside a batch.
    #[test]
    fn int_kernel_batch_is_bitwise() {
        let mut rng = Rng(77);
        for dtype in [GgmlType::Q4K, GgmlType::Q6K] {
            let (n, rows) = (768usize, 5usize);
            let mat = random_matrix(dtype, n, rows, &mut rng);
            let m = Mat { dtype, offset: 0, cols: n, rows };
            let qs: Vec<Q8K> = (0..11)
                .map(|_| {
                    let x: Vec<f32> = (0..n).map(|_| rng.f(-2.0, 2.0)).collect();
                    let mut q = Q8K::default();
                    q.quantize(&x);
                    q
                })
                .collect();
            let qr: Vec<&Q8K> = qs.iter().collect();
            let mut batch = vec![vec![0.0f32; rows]; qs.len()];
            let op: Vec<*mut f32> = batch.iter_mut().map(|v| v.as_mut_ptr()).collect();
            unsafe { matmul_rows(mat.as_ptr(), &m, &qr, &op, 0, rows) };
            for (b, q) in qs.iter().enumerate() {
                let mut one = vec![0.0f32; rows];
                unsafe { matmul_rows(mat.as_ptr(), &m, &[q], &[one.as_mut_ptr()], 0, rows) };
                for r in 0..rows {
                    assert_eq!(one[r].to_bits(), batch[b][r].to_bits(), "{dtype:?} b={b} r={r}");
                }
            }
        }
    }

    /// `cargo test --release bench_int -- --ignored --nocapture`: float vs integer kernels,
    /// one thread, on Qwen3-30B-A3B expert shapes.
    #[test]
    #[ignore]
    fn bench_int_vs_float() {
        let mut rng = Rng(3);
        for (dtype, n, rows, name) in [
            (GgmlType::Q4K, 2048usize, 768usize * 16, "Q4_K expert gate/up (2048 cols)"),
            (GgmlType::Q6K, 768, 2048 * 16, "Q6_K expert down (768 cols)"),
            (GgmlType::Q4K, 768, 2048 * 16, "Q4_K expert down (768 cols)"),
            (GgmlType::Q6K, 2048, 16384, "Q6_K LM-head slice (2048 cols)"),
        ] {
            let mat = random_matrix(dtype, n, rows, &mut rng);
            let m = Mat { dtype, offset: 0, cols: n, rows };
            println!("{name}: {rows} rows, {:.1} MB", mat.len() as f64 / 1e6);
            for bsz in [1usize, 2, 4, 8] {
                let xs: Vec<Vec<f32>> = (0..bsz).map(|_| (0..n).map(|_| rng.f(-1.0, 1.0)).collect()).collect();
                let mut outs = vec![vec![0.0f32; rows]; bsz];
                let xp: Vec<*const f32> = xs.iter().map(|v| v.as_ptr()).collect();
                let op: Vec<*mut f32> = outs.iter_mut().map(|v| v.as_mut_ptr()).collect();
                let iters = 30;
                let t = std::time::Instant::now();
                for _ in 0..iters {
                    if bsz == 1 {
                        unsafe { matvec_rows_avx2(mat.as_ptr(), &m, &xs[0], &mut outs[0], 0) };
                    } else {
                        unsafe { crate::quant_multi::matmul_rows_avx2(mat.as_ptr(), &m, &xp, &op, 0, rows) };
                    }
                }
                let float_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
                let mut qs = vec![Q8K::default(); bsz];
                let t = std::time::Instant::now();
                for _ in 0..iters {
                    for (q, x) in qs.iter_mut().zip(&xs) {
                        q.quantize(x);
                    }
                    let qr: Vec<&Q8K> = qs.iter().collect();
                    unsafe { matmul_rows(mat.as_ptr(), &m, &qr, &op, 0, rows) };
                }
                let int_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
                println!("  B={bsz}: float {float_us:8.0} us | int {int_us:8.0} us | {:.2}x faster", float_us / int_us);
            }
        }
    }
}
