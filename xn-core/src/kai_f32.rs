//! Arm KleidiAI's SME2 kernels for f32 matmuls, behind the `kai` feature.
//!
//! The arithmetic is f32 multiply-adds like the NEON path's, so the results differ from it
//! only by the order of the sums. Both operands are packed into KleidiAI's layout, into
//! buffers kept per thread. Used when [`crate::quantized::kai::active`] holds. With
//! `accelerate` on, Accelerate takes the f32 matmuls it supports, and only the layouts it
//! hands back to the generic gemm reach this.
//!
//! A model parameter is packed once, not on every call: an operand inside the span the caller
//! declared with `gemm::packed_cache::with_constant` (a linear layer's weight, a convolution's
//! kernel; see [`crate::cpu_backend::parameter`]) keeps its packed copy in that cache, as
//! xn-gemm's packed panels do, under the same budget, `GEMM_PACKED_LHS_CACHE_MB`. Packing moves,
//! not what is computed, so results are bit for bit those of packing per call.
//!
//! A one-row matmul stays on the NEON path: it reads each weight once, so packing the weight
//! first would cost as much as the matmul itself.

use std::cell::RefCell;
use std::ffi::c_void;

/// What a KleidiAI panel is keyed on in `gemm::packed_cache`, in the place of xn-gemm's `kc`.
/// That is a k-chunk and never this large, so a panel packed here never matches one xn-gemm
/// packed from the same operand, nor a packed lhs a packed rhs.
const LHS_PANEL: usize = usize::MAX;
const RHS_PANEL: usize = usize::MAX - 1;
/// A cache line, which is more than the kernels need.
const PANEL_ALIGN: usize = 64;

unsafe extern "C" {
    fn kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
    fn kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
    fn kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
    fn kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() -> usize;
    fn kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
        m: usize,
        n: usize,
        k: usize,
        lhs_packed: *const c_void,
        rhs_packed: *const c_void,
        dst: *mut c_void,
        dst_stride_row: usize,
        dst_stride_col: usize,
        clamp_min: f32,
        clamp_max: f32,
    );
    fn kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(
        m: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
    ) -> usize;
    fn kai_run_lhs_pack_f32p2vlx1_f32_sme(
        m: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
        m_idx_start: usize,
        lhs: *const c_void,
        lhs_stride_row: usize,
        lhs_packed: *mut c_void,
    );
    fn kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
        n: usize,
        k: usize,
    ) -> usize;
    fn kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
        num_groups: usize,
        n: usize,
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
        rhs_stride_row: usize,
        rhs: *const c_void,
        bias: *const c_void,
        scale: *const c_void,
        rhs_packed: *mut c_void,
        extra_bytes: usize,
        params: *const c_void,
    );
    fn kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(
        n: usize,
        k: usize,
    ) -> usize;
    fn kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(
        num_groups: usize,
        n: usize,
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
        rhs_stride: usize,
        rhs: *const c_void,
        bias: *const c_void,
        scale: *const c_void,
        rhs_packed: *mut c_void,
        extra_bytes: usize,
        params: *const c_void,
    );
}

/// How the rhs is laid out: `k` rows of `n`, or `n` rows of `k`, each with a row stride.
#[derive(Clone, Copy)]
enum Rhs {
    KxN(usize),
    NxK(usize),
}

/// How the lhs is laid out: `m` rows of `k`, or `k` rows of `m`, each with a row stride. The
/// packing routine takes the first, so the second is transposed into it first.
#[derive(Clone, Copy)]
enum Lhs {
    MxK(usize),
    KxM(usize),
}

/// Packed operands, reused across calls. Each buffer keeps the size of the largest matmul its
/// thread has run, for the life of the thread.
#[derive(Default)]
struct Scratch {
    lhs: Vec<f32>,
    lhs_t: Vec<f32>,
    rhs: Vec<f32>,
    bias: Vec<f32>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

fn grow(buf: &mut Vec<f32>, bytes: usize) -> *mut c_void {
    let len = bytes.div_ceil(4);
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
    buf.as_mut_ptr().cast()
}

/// `dst[b] = lhs[b] @ rhs[b]` with the same stride conventions as `gemm_`, element strides
/// given as `(column, row)`. Returns `false`, having done nothing, for a layout or size the
/// kernels do not take.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    dst: &mut [f32],
    lhs: &[f32],
    rhs: &[f32],
    (m, n, k): (usize, usize, usize),
    batch: usize,
    (lhs_b_stride, rhs_b_stride): (usize, usize),
    (dst_cs, dst_rs): (usize, usize),
    (lhs_cs, lhs_rs): (usize, usize),
    (rhs_cs, rhs_rs): (usize, usize),
) -> bool {
    if m < 2 || n == 0 || k == 0 || batch == 0 || !crate::quantized::kai::active() {
        return false;
    }
    if dst_cs != 1 {
        return false;
    }
    let lhs_layout = if lhs_cs == 1 {
        Lhs::MxK(lhs_rs)
    } else if lhs_rs == 1 {
        Lhs::KxM(lhs_cs)
    } else {
        return false;
    };
    let layout = if rhs_cs == 1 {
        Rhs::KxN(rhs_rs)
    } else if rhs_rs == 1 {
        Rhs::NxK(rhs_cs)
    } else {
        return false;
    };
    // Every element the kernels read or write has to be inside the slices.
    let lhs_end = (batch - 1) * lhs_b_stride
        + match lhs_layout {
            Lhs::MxK(s) => (m - 1) * s + k,
            Lhs::KxM(s) => (k - 1) * s + m,
        };
    let rhs_end = (batch - 1) * rhs_b_stride
        + match layout {
            Rhs::KxN(s) => (k - 1) * s + n,
            Rhs::NxK(s) => (n - 1) * s + k,
        };
    let dst_end = (batch - 1) * m * n + (m - 1) * dst_rs + n;
    if lhs.len() < lhs_end || rhs.len() < rhs_end || dst.len() < dst_end || dst_rs < n {
        return false;
    }
    // Batches of `dst` are `m * n` apart, so padded rows would run into the next batch.
    if batch > 1 && dst_rs != n {
        return false;
    }

    SCRATCH.with_borrow_mut(|s| {
        // SAFETY: the getters have no preconditions; the packers write at most the sizes their
        // own queries return, into buffers grown to that size or cached panels allocated at
        // it; and every operand offset was checked against its slice above. That a cached
        // operand's bytes do not change is what the caller vouched for with `with_constant`.
        unsafe {
            let mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            let nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            let kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            let sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            let f = std::mem::size_of::<f32>();
            // The packed rhs carries a bias per column, which is zero here.
            if s.bias.len() < n {
                s.bias.resize(n, 0.0);
            }
            let bias = s.bias.as_ptr().cast();
            let lhs_size = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, kr, sr);
            let rhs_size = match layout {
                Rhs::KxN(_) => {
                    kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(n, k)
                }
                Rhs::NxK(_) => {
                    kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(n, k)
                }
            };
            // A cached panel for an operand that is the same in every batch entry, if the
            // caller declared it constant and has used it before; `None` packs it per call.
            // The lhs is `m` by `k` and the rhs, a weight's usual side, is taken as `n` by `k`.
            let lhs_panel = (batch == 1 || lhs_b_stride == 0)
                .then(|| {
                    gemm::packed_cache::get(
                        lhs.as_ptr(),
                        m,
                        k,
                        lhs_rs as isize,
                        lhs_cs as isize,
                        LHS_PANEL,
                        mr,
                        lhs_size,
                        PANEL_ALIGN,
                    )
                })
                .flatten();
            let rhs_panel = (batch == 1 || rhs_b_stride == 0)
                .then(|| {
                    gemm::packed_cache::get(
                        rhs.as_ptr(),
                        n,
                        k,
                        rhs_cs as isize,
                        rhs_rs as isize,
                        RHS_PANEL,
                        nr,
                        rhs_size,
                        PANEL_ALIGN,
                    )
                })
                .flatten();
            let lhs_packed = match &lhs_panel {
                Some(p) => p.ptr.cast(),
                None => grow(&mut s.lhs, lhs_size),
            };
            let rhs_packed = match &rhs_panel {
                Some(p) => p.ptr.cast(),
                None => grow(&mut s.rhs, rhs_size),
            };
            // A panel filled by an earlier call is used as is.
            let lhs_ready = lhs_panel.as_ref().is_some_and(|p| p.filled);
            let rhs_ready = rhs_panel.as_ref().is_some_and(|p| p.filled);
            let pack_rhs = |rhs: *const f32| match layout {
                Rhs::KxN(st) => kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
                    1,
                    n,
                    k,
                    nr,
                    kr,
                    sr,
                    st * f,
                    rhs.cast(),
                    bias,
                    std::ptr::null(),
                    rhs_packed,
                    0,
                    std::ptr::null(),
                ),
                Rhs::NxK(st) => kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(
                    1,
                    n,
                    k,
                    nr,
                    kr,
                    sr,
                    st * f,
                    rhs.cast(),
                    bias,
                    std::ptr::null(),
                    rhs_packed,
                    0,
                    std::ptr::null(),
                ),
            };
            for b in 0..batch {
                if !rhs_ready && (b == 0 || rhs_b_stride != 0) {
                    pack_rhs(rhs.as_ptr().add(b * rhs_b_stride));
                    // Only an operand shared by every batch entry has a panel, and it is
                    // packed here, at the first.
                    if let Some(p) = &rhs_panel {
                        p.mark_filled();
                    }
                }
                if !lhs_ready && (b == 0 || lhs_b_stride != 0) {
                    let src = &lhs[b * lhs_b_stride..];
                    let (ptr, stride) = match lhs_layout {
                        Lhs::MxK(st) => (src.as_ptr(), st),
                        Lhs::KxM(st) => {
                            s.lhs_t.resize(m * k, 0.0);
                            for (p, col) in src.chunks(st).take(k).enumerate() {
                                for (i, &x) in col[..m].iter().enumerate() {
                                    s.lhs_t[i * k + p] = x;
                                }
                            }
                            (s.lhs_t.as_ptr(), k)
                        }
                    };
                    kai_run_lhs_pack_f32p2vlx1_f32_sme(
                        m,
                        k,
                        mr,
                        kr,
                        sr,
                        0,
                        ptr.cast(),
                        stride * f,
                        lhs_packed,
                    );
                    if let Some(p) = &lhs_panel {
                        p.mark_filled();
                    }
                }
                kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
                    m,
                    n,
                    k,
                    lhs_packed,
                    rhs_packed,
                    dst.as_mut_ptr().add(b * m * n).cast(),
                    dst_rs * f,
                    f,
                    // The kernel clamps with `fclamp`, which keeps the number when one side is
                    // NaN. NaN bounds therefore clamp nothing, and a NaN result stays NaN;
                    // infinite bounds would turn it into -inf.
                    f32::NAN,
                    f32::NAN,
                );
            }
        }
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(seed: u64, len: usize) -> Vec<f32> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// `lhs[(i, p)]` at `i * ls.1 + p * ls.0`, `rhs[(p, j)]` at `p * rs.1 + j * rs.0`.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        lhs: &[f32],
        rhs: &[f32],
        (m, n, k): (usize, usize, usize),
        batch: usize,
        (lb, rb): (usize, usize),
        dst_rs: usize,
        (lhs_cs, lhs_rs): (usize, usize),
        (rhs_cs, rhs_rs): (usize, usize),
    ) -> Vec<f32> {
        let mut dst = vec![0f32; (batch - 1) * m * n + (m - 1) * dst_rs + n];
        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    dst[b * m * n + i * dst_rs + j] = (0..k)
                        .map(|p| {
                            let r = rhs[b * rb + p * rhs_rs + j * rhs_cs] as f64;
                            lhs[b * lb + i * lhs_rs + p * lhs_cs] as f64 * r
                        })
                        .sum::<f64>() as f32;
                }
            }
        }
        dst
    }

    /// Both layouts of each input, partial tiles in `m` and `n`, padded lhs and dst rows, and
    /// batches with a shared or a per-batch lhs and rhs.
    #[test]
    fn matches_the_reference() {
        // (m, n, k, batch, which input is per batch, kxm lhs, kxn rhs, lhs pad, dst pad)
        let (lhs_b, rhs_b, both) = ((true, false), (false, true), (true, true));
        let cases = [
            (2, 3, 1, 1, both, false, true, 0, 0),
            (16, 2048, 512, 1, both, false, false, 0, 0),
            (17, 70, 33, 1, both, false, true, 5, 3),
            (96, 128, 64, 1, both, false, false, 0, 0),
            (480, 64, 384, 1, both, false, true, 0, 0),
            (16, 266, 64, 3, both, false, true, 0, 0),
            (16, 64, 266, 3, both, false, false, 0, 0),
            (5, 40, 24, 4, lhs_b, false, false, 2, 0),
            (16, 3072, 512, 1, both, true, true, 0, 0),
            (19, 130, 40, 2, both, true, true, 3, 0),
            (64, 96, 72, 3, rhs_b, false, true, 0, 0),
            (7, 33, 18, 2, rhs_b, true, false, 1, 0),
        ];
        for (m, n, k, batch, (lhs_per, rhs_per), kxm, kxn, lhs_pad, dst_pad) in cases {
            let dst_rs = n + dst_pad;
            let (lhs_cs, lhs_rs) = if kxm { (m + lhs_pad, 1) } else { (1, k + lhs_pad) };
            let (rhs_cs, rhs_rs) = if kxn { (1, n) } else { (k, 1) };
            let lhs_len = if kxm { k * (m + lhs_pad) } else { m * (k + lhs_pad) };
            let lb = if lhs_per { lhs_len } else { 0 };
            let rb = if rhs_per { n * k } else { 0 };
            let lhs =
                uniform(0xA076_1D64 + (m * n) as u64, lhs_len * if lhs_per { batch } else { 1 });
            let rhs =
                uniform(0xE703_7ED1 + (n * k) as u64, n * k * if rhs_per { batch } else { 1 });
            let want = reference(
                &lhs,
                &rhs,
                (m, n, k),
                batch,
                (lb, rb),
                dst_rs,
                (lhs_cs, lhs_rs),
                (rhs_cs, rhs_rs),
            );
            let mut got = vec![f32::NAN; want.len()];
            let ran = gemm(
                &mut got,
                &lhs,
                &rhs,
                (m, n, k),
                batch,
                (lb, rb),
                (1, dst_rs),
                (lhs_cs, lhs_rs),
                (rhs_cs, rhs_rs),
            );
            if !crate::quantized::kai::active() {
                assert!(!ran);
                continue;
            }
            assert!(ran, "m={m} n={n} k={k}");
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                let (b, r) = (idx / (m * n), idx % (m * n));
                if r / dst_rs >= m || r % dst_rs >= n {
                    continue;
                }
                let tol = 1e-5 * (k as f32).sqrt().max(1.0);
                assert!((g - w).abs() <= tol, "m={m} n={n} k={k} b={b} at {idx}: {g} vs {w}");
            }
        }
    }

    /// A NaN in a row of the lhs gives NaN in that row of the output, as on the NEON path,
    /// and leaves the other rows alone.
    #[test]
    fn keeps_nan() {
        let (m, n, k) = (4, 40, 8);
        let mut lhs = vec![1f32; m * k];
        lhs[0] = f32::NAN;
        let rhs = vec![1f32; k * n];
        let mut dst = vec![0f32; m * n];
        if !gemm(&mut dst, &lhs, &rhs, (m, n, k), 1, (0, 0), (1, n), (1, k), (1, n)) {
            return;
        }
        assert!(dst[..n].iter().all(|x| x.is_nan()), "{:?}", &dst[..n]);
        assert!(dst[n..].iter().all(|&x| x == k as f32));
    }

    /// An operand declared constant is packed once and then served from the cache, on either
    /// side, in either layout, and shared over a batch, with the same bits as packing it per
    /// call. An undeclared one is never cached.
    #[test]
    fn a_declared_operand_is_packed_once() {
        use gemm::packed_cache;
        if !crate::quantized::kai::active() {
            return;
        }
        // (m, n, k, batch, the lhs is the constant, kxm lhs, kxn rhs)
        let cases = [
            // A linear layer: `x @ W^T`, the weight as an `n` by `k` rhs.
            (16, 96, 64, 1, false, false, false),
            // A transposed convolution: a kxm lhs per batch entry, the kernel a shared kxn rhs.
            (6, 40, 24, 3, false, true, true),
            // A convolution: the kernel as a shared lhs, its windows a kxn rhs per batch entry.
            (32, 20, 48, 2, true, false, true),
            // A kxm lhs as the constant.
            (9, 33, 17, 1, true, true, false),
        ];
        for (m, n, k, batch, constant_lhs, kxm, kxn) in cases {
            let (lhs_cs, lhs_rs) = if kxm { (m, 1) } else { (1, k) };
            let (rhs_cs, rhs_rs) = if kxn { (1, n) } else { (k, 1) };
            // The constant is the same for every batch entry, the other operand is not.
            let (lb, rb) = if constant_lhs { (0, n * k) } else { (m * k, 0) };
            let lhs = uniform(0x5EED + m as u64, m * k * if constant_lhs { 1 } else { batch });
            let rhs = uniform(0xF00D + n as u64, n * k * if constant_lhs { batch } else { 1 });
            let run = || {
                let mut dst = vec![f32::NAN; batch * m * n];
                let layouts = ((1, n), (lhs_cs, lhs_rs), (rhs_cs, rhs_rs));
                let (d, l, r) = layouts;
                assert!(gemm(&mut dst, &lhs, &rhs, (m, n, k), batch, (lb, rb), d, l, r));
                dst.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            };
            let panels = packed_cache::stats().1;
            let per_call = run();
            assert_eq!(run(), per_call);
            if !packed_cache::enabled() {
                assert_eq!(packed_cache::stats().1, panels, "an undeclared operand was cached");
            }
            let constant = if constant_lhs {
                packed_cache::Constant::new(lhs.as_ptr(), lhs.len())
            } else {
                packed_cache::Constant::new(rhs.as_ptr(), rhs.len())
            };
            // The first call only notes the operand, the second packs it into the cache, and
            // later ones use that copy.
            packed_cache::with_constant(constant, || {
                for call in 0..4 {
                    assert_eq!(run(), per_call, "m={m} n={n} k={k}, call {call}");
                }
            });
            if packed_cache::budget_bytes() > 0 {
                let held = packed_cache::stats().1;
                assert_eq!(held, panels + 1, "m={m} n={n} k={k}: one panel for the constant");
            }
        }
        packed_cache::clear();
    }

    #[test]
    fn declines_what_it_does_not_take() {
        let (lhs, rhs, mut dst) = (vec![1f32; 64], vec![1f32; 64], vec![0f32; 64]);
        let run =
            |dst: &mut [f32], m, d, l, r| gemm(dst, &lhs, &rhs, (m, 4, 4), 1, (0, 0), d, l, r);
        // One row, a strided lhs column, an rhs strided both ways, a strided dst column.
        assert!(!run(&mut dst, 1, (1, 4), (1, 4), (1, 4)));
        assert!(!run(&mut dst, 4, (1, 4), (2, 8), (1, 4)));
        assert!(!run(&mut dst, 4, (1, 4), (1, 4), (2, 8)));
        assert!(!run(&mut dst, 4, (2, 8), (1, 4), (1, 4)));
        // A dst too short for the rows.
        assert!(!run(&mut dst[..10], 4, (1, 4), (1, 4), (1, 4)));
        // Padded dst rows in a batch.
        assert!(!gemm(&mut dst, &lhs, &rhs, (4, 4, 4), 2, (16, 0), (1, 5), (1, 4), (1, 4)));
        assert!(!dst.iter().any(|&x| x != 0.0));
    }
}
