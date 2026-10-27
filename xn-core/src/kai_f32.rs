//! Arm KleidiAI's SME2 kernels for f32 matmuls, behind the `kai` feature.
//!
//! The arithmetic is f32 multiply-adds like the NEON path's, so the results differ from it
//! only by the order of the sums. Used when [`crate::quantized::kai::active`] holds. With
//! `accelerate` on, Accelerate takes the f32 matmuls it supports, and only the layouts it
//! hands back to the generic gemm reach this.
//!
//! A matmul of two or more rows packs both operands into KleidiAI's layout on every call, into
//! buffers kept per thread. A one-row matmul reads each weight once, so packing the weight on
//! every call would cost as much as the matmul itself. It takes KleidiAI's one-row kernel only
//! when its rhs is a parameter the caller declared with `gemm::packed_cache::with_constant`, as
//! `nn::Linear` does: the packed copy is then kept, shared by every thread, and reused. Any
//! other one-row matmul stays on the NEON path.

use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, PoisonError};

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
    fn kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla() -> usize;
    fn kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla() -> usize;
    fn kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla() -> usize;
    fn kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla(
        m: usize,
        n: usize,
        k: usize,
        lhs: *const c_void,
        lhs_stride: usize,
        rhs_packed: *const c_void,
        dst: *mut c_void,
        dst_stride_row: usize,
        dst_stride_col: usize,
        clamp_min: f32,
        clamp_max: f32,
    );
}

/// How the rhs is laid out: `k` rows of `n`, or `n` rows of `k`, each with a row stride.
#[derive(Clone, Copy, PartialEq, Eq)]
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

/// A parameter packed for the one-row kernel. Its bytes count as held until the last user drops
/// it.
struct Packed(Vec<f32>);

impl Drop for Packed {
    fn drop(&mut self) {
        HELD.fetch_sub(self.0.len() * 4, Relaxed);
    }
}

/// A cached [`Packed`]. The address, shape and layout say which parameter it is; a sample of
/// its contents tells apart one freed and declared again at the same address.
struct Entry {
    key: (usize, usize, usize, Rhs),
    fingerprint: [u32; 8],
    last_use: u64,
    packed: Arc<Packed>,
}

/// The parameters packed for the one-row kernel, shared by every thread, since a model's steps
/// may each run on a new one. When the budget is spent, the least recently used go first.
struct Cache {
    entries: Vec<Entry>,
    clock: u64,
}

static PACKED: Mutex<Cache> = Mutex::new(Cache { entries: Vec::new(), clock: 0 });

/// Bytes of packed parameters held. They count against the budget of `gemm`'s own packed cache,
/// as a separate tally.
static HELD: AtomicUsize = AtomicUsize::new(0);

fn grow(buf: &mut Vec<f32>, bytes: usize) -> *mut c_void {
    let len = bytes.div_ceil(4);
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
    buf.as_mut_ptr().cast()
}

/// The bytes [`pack_rhs`] writes.
fn packed_rhs_size(layout: Rhs, n: usize, k: usize) -> usize {
    // SAFETY: the size queries have no preconditions.
    unsafe {
        match layout {
            Rhs::KxN(_) => kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(n, k),
            Rhs::NxK(_) => kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(n, k),
        }
    }
}

/// Packs the `rhs` that `layout` describes, with the `n` biases at `bias`, into `out`. Both f32
/// kernels read this layout.
///
/// # Safety
///
/// `rhs` holds the operand `layout` describes, `bias` holds `n` floats, and `out` holds
/// [`packed_rhs_size`] bytes.
unsafe fn pack_rhs(
    layout: Rhs,
    (n, k): (usize, usize),
    (nr, kr, sr): (usize, usize, usize),
    rhs: *const f32,
    bias: *const c_void,
    out: *mut c_void,
) {
    let f = std::mem::size_of::<f32>();
    // SAFETY: as the caller promises.
    unsafe {
        match layout {
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
                out,
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
                out,
                0,
                std::ptr::null(),
            ),
        }
    }
}

/// Eight elements spread over `x`, as bits.
fn fingerprint(x: &[f32]) -> [u32; 8] {
    let last = x.len() - 1;
    std::array::from_fn(|i| x[i * last / 7].to_bits())
}

/// Claims `bytes` against the budget, or fails having claimed nothing.
fn reserve(bytes: usize) -> bool {
    let budget = gemm::packed_cache::budget_bytes();
    HELD.fetch_update(Relaxed, Relaxed, |held| held.checked_add(bytes).filter(|&t| t <= budget))
        .is_ok()
}

/// The packed copy of the parameter `rhs`, packing it if it is not cached. `None` when it cannot
/// fit in the budget even with every other entry evicted.
fn packed(rhs: &[f32], layout: Rhs, (n, k): (usize, usize)) -> Option<Arc<Packed>> {
    let key = (rhs.as_ptr() as usize, n, k, layout);
    let print = fingerprint(rhs);
    let mut cache = PACKED.lock().unwrap_or_else(PoisonError::into_inner);
    cache.clock += 1;
    let clock = cache.clock;
    if let Some(e) = cache.entries.iter_mut().find(|e| e.key == key && e.fingerprint == print) {
        e.last_use = clock;
        return Some(Arc::clone(&e.packed));
    }
    // An old copy at this address holds other contents now.
    cache.entries.retain(|e| e.key != key);
    let bytes = packed_rhs_size(layout, n, k).div_ceil(4) * 4;
    while cache.entries.len() >= gemm::packed_cache::MAX_ENTRIES || !reserve(bytes) {
        let oldest = (0..cache.entries.len()).min_by_key(|&i| cache.entries[i].last_use)?;
        cache.entries.swap_remove(oldest);
    }
    let mut data = vec![0f32; bytes / 4];
    let bias = vec![0f32; n];
    // SAFETY: the getters have no preconditions, `rhs` holds the operand `layout` describes, and
    // `data` holds the size the packer's query returns.
    unsafe {
        let nr = kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla();
        let kr = kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla();
        let sr = kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla();
        let (bias, out) = (bias.as_ptr().cast(), data.as_mut_ptr().cast());
        pack_rhs(layout, (n, k), (nr, kr, sr), rhs.as_ptr(), bias, out);
    }
    let packed = Arc::new(Packed(data));
    cache.entries.push(Entry { key, fingerprint: print, last_use: clock, packed: packed.clone() });
    Some(packed)
}

/// `dst[b] = lhs[b] @ rhs`, one row per batch, when `rhs` is the whole parameter declared on
/// this thread and every batch shares it: packed on first use, then reused. Returns `false`,
/// having done nothing, for any other rhs, or when the budget is spent. The caller has checked
/// every offset against the slices, and `lhs` rows are contiguous.
fn one_row(
    dst: &mut [f32],
    lhs: &[f32],
    rhs: &[f32],
    (n, k): (usize, usize),
    batch: usize,
    (lhs_b_stride, rhs_b_stride): (usize, usize),
    layout: Rhs,
) -> bool {
    let declared = gemm::packed_cache::Constant::new(rhs.as_ptr(), rhs.len());
    if (batch > 1 && rhs_b_stride != 0) || gemm::packed_cache::constant() != declared {
        return false;
    }
    let Some(packed) = packed(rhs, layout, (n, k)) else {
        return false;
    };
    let f = std::mem::size_of::<f32>();
    for b in 0..batch {
        // SAFETY: the caller checked every operand offset against its slice, and `packed` holds
        // `rhs` packed for this kernel.
        unsafe {
            kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla(
                1,
                n,
                k,
                lhs.as_ptr().add(b * lhs_b_stride).cast(),
                k * f,
                packed.0.as_ptr().cast(),
                dst.as_mut_ptr().add(b * n).cast(),
                n * f,
                f,
                // NaN bounds clamp nothing; see the same arguments in `gemm`.
                f32::NAN,
                f32::NAN,
            )
        };
    }
    true
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
    if m == 0 || n == 0 || k == 0 || batch == 0 || !crate::quantized::kai::active() {
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
    if m == 1 {
        let strides = (lhs_b_stride, rhs_b_stride);
        let contiguous = matches!(lhs_layout, Lhs::MxK(_));
        return contiguous && one_row(dst, lhs, &rhs[..rhs_end], (n, k), batch, strides, layout);
    }

    SCRATCH.with_borrow_mut(|s| {
        // SAFETY: the getters have no preconditions; the packers write at most the sizes their
        // own queries return, into buffers grown to that size; and every operand offset was
        // checked against its slice above.
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
            let lhs_packed = grow(&mut s.lhs, lhs_size);
            let rhs_packed = grow(&mut s.rhs, packed_rhs_size(layout, n, k));
            for b in 0..batch {
                if b == 0 || rhs_b_stride != 0 {
                    let rhs = rhs.as_ptr().add(b * rhs_b_stride);
                    pack_rhs(layout, (n, k), (nr, kr, sr), rhs, bias, rhs_packed);
                }
                if b == 0 || lhs_b_stride != 0 {
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

    /// Runs `gemm` for one row per batch against a shared rhs, under `declared` if given.
    fn one_row_run(
        lhs: &[f32],
        rhs: &[f32],
        (n, k): (usize, usize),
        batch: usize,
        kxn: bool,
        declared: Option<gemm::packed_cache::Constant>,
    ) -> Option<Vec<f32>> {
        let rhs_strides = if kxn { (1, n) } else { (k, 1) };
        let mut got = vec![f32::NAN; batch * n];
        let mut run =
            || gemm(&mut got, lhs, rhs, (1, n, k), batch, (k, 0), (1, n), (1, k), rhs_strides);
        let ran = match declared {
            Some(range) => gemm::packed_cache::with_constant(range, run),
            None => run(),
        };
        ran.then_some(got)
    }

    fn assert_close(got: &[f32], want: &[f32], k: usize, what: &str) {
        let tol = 1e-5 * (k as f32).sqrt().max(1.0);
        for (idx, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol, "{what} at {idx}: {g} vs {w}");
        }
    }

    /// One row times a declared parameter takes the one-row kernel, for either layout of the
    /// rhs, partial column tiles, and batches sharing it, on the first call and on a later one
    /// that reuses the packed copy. Undeclared, or declared in part, it stays on the NEON path.
    #[test]
    fn one_row_takes_a_declared_parameter() {
        for (n, k, batch, kxn) in [
            (1, 1, 1, false),
            (37, 19, 1, true),
            (512, 512, 1, false),
            (1536, 512, 3, false),
            (130, 40, 2, true),
        ] {
            let what = format!("n={n} k={k} batch={batch} kxn={kxn}");
            let lhs = uniform(0x2545_F491 + n as u64, batch * k);
            let rhs = uniform(0x9E37_79B9 + k as u64, n * k);
            let rhs_strides = if kxn { (1, n) } else { (k, 1) };
            let want = reference(&lhs, &rhs, (1, n, k), batch, (k, 0), n, (1, k), rhs_strides);
            assert!(one_row_run(&lhs, &rhs, (n, k), batch, kxn, None).is_none(), "{what}");
            let part = gemm::packed_cache::Constant::new(rhs.as_ptr(), rhs.len() - 1);
            assert!(one_row_run(&lhs, &rhs, (n, k), batch, kxn, Some(part)).is_none(), "{what}");
            let whole = gemm::packed_cache::Constant::new(rhs.as_ptr(), rhs.len());
            for _ in 0..2 {
                let got = one_row_run(&lhs, &rhs, (n, k), batch, kxn, Some(whole));
                if !crate::quantized::kai::active() {
                    assert!(got.is_none());
                    break;
                }
                assert_close(&got.expect("ran"), &want, k, &what);
            }
        }
    }

    /// New contents at the address of a parameter already packed are packed again rather than
    /// read from the old copy.
    #[test]
    fn one_row_repacks_changed_contents() {
        let (n, k) = (96, 64);
        let lhs = uniform(7, k);
        let mut rhs = uniform(11, n * k);
        for seed in [11, 13] {
            rhs.copy_from_slice(&uniform(seed, n * k));
            let want = reference(&lhs, &rhs, (1, n, k), 1, (0, 0), n, (1, k), (k, 1));
            let whole = gemm::packed_cache::Constant::new(rhs.as_ptr(), rhs.len());
            match one_row_run(&lhs, &rhs, (n, k), 1, false, Some(whole)) {
                Some(got) => assert_close(&got, &want, k, &format!("seed={seed}")),
                None => assert!(!crate::quantized::kai::active()),
            }
        }
    }

    /// The packed copy of `rhs` in the cache, by address.
    fn cached(rhs: &[f32]) -> Option<usize> {
        let cache = PACKED.lock().unwrap_or_else(PoisonError::into_inner);
        let e = cache.entries.iter().find(|e| e.key.0 == rhs.as_ptr() as usize)?;
        Some(Arc::as_ptr(&e.packed) as usize)
    }

    /// A parameter packed on one thread is reused on the next, as when every request runs its
    /// steps on a new thread.
    #[test]
    fn one_row_shares_the_packed_copy_across_threads() {
        let (n, k) = (80, 48);
        let lhs = uniform(17, k);
        let rhs = uniform(19, n * k);
        let want = reference(&lhs, &rhs, (1, n, k), 1, (0, 0), n, (1, k), (k, 1));
        let whole = gemm::packed_cache::Constant::new(rhs.as_ptr(), rhs.len());
        let mut copies = vec![];
        for _ in 0..2 {
            let got = std::thread::scope(|s| {
                s.spawn(|| one_row_run(&lhs, &rhs, (n, k), 1, false, Some(whole))).join().unwrap()
            });
            let Some(got) = got else {
                assert!(!crate::quantized::kai::active());
                return;
            };
            assert_close(&got, &want, k, "across threads");
            copies.push(cached(&rhs).expect("cached"));
        }
        assert_eq!(copies[0], copies[1], "the second thread reused the first one's copy");
    }

    /// A NaN in the row gives NaN in every output, as on the NEON path.
    #[test]
    fn one_row_keeps_nan() {
        let (n, k) = (40, 8);
        let mut lhs = vec![1f32; k];
        lhs[3] = f32::NAN;
        let rhs = vec![1f32; n * k];
        let whole = gemm::packed_cache::Constant::new(rhs.as_ptr(), rhs.len());
        if let Some(got) = one_row_run(&lhs, &rhs, (n, k), 1, false, Some(whole)) {
            assert!(got.iter().all(|x| x.is_nan()), "{got:?}");
        }
    }

    #[test]
    fn declines_what_it_does_not_take() {
        let (lhs, rhs, mut dst) = (vec![1f32; 64], vec![1f32; 64], vec![0f32; 64]);
        let run =
            |dst: &mut [f32], m, d, l, r| gemm(dst, &lhs, &rhs, (m, 4, 4), 1, (0, 0), d, l, r);
        // One row with no declared parameter, a strided lhs column, an rhs strided both ways,
        // a strided dst column.
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
