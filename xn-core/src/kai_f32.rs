//! Arm KleidiAI's SME2 kernels for f32 matmuls, behind the `kai` feature.
//!
//! The arithmetic is f32 multiply-adds like the NEON path's, so the results differ from it
//! only by the order of the sums. Operands are packed into KleidiAI's layout, into buffers kept
//! per thread, apart from the parameters below. Used when [`crate::quantized::kai::active`]
//! holds. With `accelerate` on, Accelerate takes the f32 matmuls it supports, except a declared
//! parameter times a smaller operand, which comes here first; the layouts Accelerate hands back
//! to the generic gemm reach this too.
//!
//! A model parameter is packed once, not on every call: an operand that is exactly the span the
//! caller declared with `gemm::packed_cache::with_constant` (a linear layer's weight, a
//! convolution's kernel; see [`crate::cpu_backend::parameter`]) is packed the first time it is
//! seen and kept. The copies are shared by every thread, since a caller may run each request on
//! a new one, and count against the budget of `gemm`'s own packed cache,
//! `GEMM_PACKED_LHS_CACHE_MB` (0 turns this off), as a separate tally; past it, the least
//! recently used go first. Packing moves, not what is computed, so results are bit for bit
//! those of packing per call.
//!
//! A one-row matmul stays on the NEON path: it reads each weight once, so packing the weight
//! first would cost as much as the matmul itself.

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

/// Which operand a [`Panel`] was packed as: the two packers lay the same matrix out differently.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Lhs,
    Rhs,
}

/// A parameter packed for the kernel. Its bytes count as held until the last user drops it.
struct Panel(Vec<f32>);

impl Drop for Panel {
    fn drop(&mut self) {
        HELD.fetch_sub(self.0.len() * std::mem::size_of::<f32>(), Relaxed);
    }
}

/// What a panel's layout depends on: the operand's address, side, shape and strides.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    ptr: usize,
    side: Side,
    rows: usize,
    k: usize,
    rs: usize,
    cs: usize,
}

/// A cached [`Panel`], with a sample of the operand's contents that tells apart one freed and
/// declared again at the same address.
struct Entry {
    key: Key,
    fingerprint: [u32; 8],
    last_use: u64,
    panel: Arc<Panel>,
}

/// The parameters packed so far, shared by every thread. When the budget is spent, the least
/// recently used go first.
struct Cache {
    entries: Vec<Entry>,
    clock: u64,
}

static PANELS: Mutex<Cache> = Mutex::new(Cache { entries: Vec::new(), clock: 0 });

/// Bytes of packed parameters held. They count against the budget of `gemm`'s own packed
/// cache, as a separate tally.
static HELD: AtomicUsize = AtomicUsize::new(0);

/// Claims `bytes` against the budget, or fails having claimed nothing.
fn reserve(bytes: usize) -> bool {
    let budget = gemm::packed_cache::budget_bytes();
    HELD.fetch_update(Relaxed, Relaxed, |held| held.checked_add(bytes).filter(|&t| t <= budget))
        .is_ok()
}

/// Whether the operand spanning `span` is exactly the parameter declared on this thread.
fn declared(span: &[f32]) -> bool {
    !span.is_empty()
        && gemm::packed_cache::constant()
            == gemm::packed_cache::Constant::new(span.as_ptr(), span.len())
}

/// Eight elements spread over `x`, as bits.
fn fingerprint(x: &[f32]) -> [u32; 8] {
    let last = x.len() - 1;
    std::array::from_fn(|i| x[i * last / 7].to_bits())
}

/// The packed copy of the parameter spanning `span`, calling `pack` to fill a new one of
/// `bytes` if none is cached. `None` when it cannot fit in the budget even with every other
/// entry evicted. Packing runs under the lock, so two threads never pack the same parameter.
fn panel(
    span: &[f32],
    key: Key,
    bytes: usize,
    pack: impl FnOnce(*mut c_void),
) -> Option<Arc<Panel>> {
    let print = fingerprint(span);
    let mut cache = PANELS.lock().unwrap_or_else(PoisonError::into_inner);
    cache.clock += 1;
    let clock = cache.clock;
    if let Some(e) = cache.entries.iter_mut().find(|e| e.key == key && e.fingerprint == print) {
        e.last_use = clock;
        return Some(Arc::clone(&e.panel));
    }
    // An old copy at this address holds other contents now.
    cache.entries.retain(|e| e.key != key);
    let len = bytes.div_ceil(std::mem::size_of::<f32>());
    let bytes = len * std::mem::size_of::<f32>();
    while cache.entries.len() >= gemm::packed_cache::MAX_ENTRIES || !reserve(bytes) {
        let oldest = (0..cache.entries.len()).min_by_key(|&i| cache.entries[i].last_use)?;
        cache.entries.swap_remove(oldest);
    }
    // Built before packing, so the claim above is given back if the packer unwinds.
    let mut panel = Panel(vec![0f32; len]);
    pack(panel.0.as_mut_ptr().cast());
    let panel = Arc::new(panel);
    cache.entries.push(Entry { key, fingerprint: print, last_use: clock, panel: panel.clone() });
    Some(panel)
}

/// The panels cached for an operand at `span`'s address with its current contents. Other tests
/// may leave panels for operands since freed, at the same address but with other contents.
#[cfg(test)]
fn panels_for(span: &[f32]) -> usize {
    let (ptr, print) = (span.as_ptr() as usize, fingerprint(span));
    let cache = PANELS.lock().unwrap_or_else(PoisonError::into_inner);
    cache.entries.iter().filter(|e| e.key.ptr == ptr && e.fingerprint == print).count()
}

/// `dst[b] = lhs[b] @ rhs[b]` with the same stride conventions as `gemm_`, element strides
/// given as `(column, row)`. Returns `false`, having done nothing, for a layout or size the
/// kernels do not take, or, with `only_declared`, unless an operand is a declared parameter it
/// keeps packed and the other is the smaller.
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
    only_declared: bool,
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
    let lhs_one = match lhs_layout {
        Lhs::MxK(s) => (m - 1) * s + k,
        Lhs::KxM(s) => (k - 1) * s + m,
    };
    let rhs_one = match layout {
        Rhs::KxN(s) => (k - 1) * s + n,
        Rhs::NxK(s) => (n - 1) * s + k,
    };
    let lhs_end = (batch - 1) * lhs_b_stride + lhs_one;
    let rhs_end = (batch - 1) * rhs_b_stride + rhs_one;
    let dst_end = (batch - 1) * m * n + (m - 1) * dst_rs + n;
    if lhs.len() < lhs_end || rhs.len() < rhs_end || dst.len() < dst_end || dst_rs < n {
        return false;
    }
    // Batches of `dst` are `m * n` apart, so padded rows would run into the next batch.
    if batch > 1 && dst_rs != n {
        return false;
    }
    // An operand shared by every batch entry that is the declared parameter, if any.
    let shared_lhs = (batch == 1 || lhs_b_stride == 0) && declared(&lhs[..lhs_one]);
    let shared_rhs = (batch == 1 || rhs_b_stride == 0) && declared(&rhs[..rhs_one]);
    // Only where packing the other operand per call costs less than the declared one would:
    // a weight times a few rows, not a kernel slid over a long signal, where packing the
    // windows outweighs what keeping the kernel packed saves.
    let worth =
        (shared_lhs && batch * rhs_one < lhs_one) || (shared_rhs && batch * lhs_one < rhs_one);
    if only_declared && !worth {
        return false;
    }

    SCRATCH.with_borrow_mut(|s| {
        // SAFETY: the getters have no preconditions; the packers write at most the sizes their
        // own queries return, into buffers grown to that size or panels allocated at it; and
        // every operand offset was checked against its slice above. That a cached operand's
        // bytes do not change is what the caller vouched for with `with_constant`.
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
            let pack_rhs = |rhs: *const f32, out: *mut c_void| match layout {
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
            };
            // A kxm lhs is transposed into `lhs_t` first, since the packer takes mxk.
            let pack_lhs = |src: &[f32], lhs_t: &mut Vec<f32>, out: *mut c_void| {
                let (ptr, stride) = match lhs_layout {
                    Lhs::MxK(st) => (src.as_ptr(), st),
                    Lhs::KxM(st) => {
                        lhs_t.resize(m * k, 0.0);
                        for (p, col) in src.chunks(st).take(k).enumerate() {
                            for (i, &x) in col[..m].iter().enumerate() {
                                lhs_t[i * k + p] = x;
                            }
                        }
                        (lhs_t.as_ptr(), k)
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
                    out,
                );
            };
            // The declared parameter's panel, packed on its first use and then kept; `None`
            // packs it per call, as any other operand.
            let lhs_t = &mut s.lhs_t;
            let lhs_panel = shared_lhs
                .then(|| {
                    let key = Key {
                        ptr: lhs.as_ptr() as usize,
                        side: Side::Lhs,
                        rows: m,
                        k,
                        rs: lhs_rs,
                        cs: lhs_cs,
                    };
                    panel(&lhs[..lhs_one], key, lhs_size, |out| pack_lhs(lhs, lhs_t, out))
                })
                .flatten();
            let rhs_panel = shared_rhs
                .then(|| {
                    let key = Key {
                        ptr: rhs.as_ptr() as usize,
                        side: Side::Rhs,
                        rows: n,
                        k,
                        rs: rhs_rs,
                        cs: rhs_cs,
                    };
                    panel(&rhs[..rhs_one], key, rhs_size, |out| pack_rhs(rhs.as_ptr(), out))
                })
                .flatten();
            let lhs_packed = match &lhs_panel {
                Some(p) => p.0.as_ptr().cast(),
                None => grow(&mut s.lhs, lhs_size).cast_const(),
            };
            let rhs_packed = match &rhs_panel {
                Some(p) => p.0.as_ptr().cast(),
                None => grow(&mut s.rhs, rhs_size).cast_const(),
            };
            for b in 0..batch {
                if rhs_panel.is_none() && (b == 0 || rhs_b_stride != 0) {
                    pack_rhs(rhs.as_ptr().add(b * rhs_b_stride), rhs_packed.cast_mut());
                }
                if lhs_panel.is_none() && (b == 0 || lhs_b_stride != 0) {
                    pack_lhs(&lhs[b * lhs_b_stride..], &mut s.lhs_t, lhs_packed.cast_mut());
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
                false,
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
        if !gemm(&mut dst, &lhs, &rhs, (m, n, k), 1, (0, 0), (1, n), (1, k), (1, n), false) {
            return;
        }
        assert!(dst[..n].iter().all(|x| x.is_nan()), "{:?}", &dst[..n]);
        assert!(dst[n..].iter().all(|&x| x == k as f32));
    }

    /// An operand declared constant is packed on its first use and then served from the cache,
    /// on either side, in either layout, shared over a batch and by other threads, with the
    /// same bits as packing it per call. An undeclared one is never cached, and new contents at
    /// a cached address are packed again.
    #[test]
    fn a_declared_operand_is_packed_once() {
        use gemm::packed_cache;
        if !crate::quantized::kai::active() || packed_cache::budget_bytes() == 0 {
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
            let what = format!("m={m} n={n} k={k}");
            let (lhs_cs, lhs_rs) = if kxm { (m, 1) } else { (1, k) };
            let (rhs_cs, rhs_rs) = if kxn { (1, n) } else { (k, 1) };
            // The constant is the same for every batch entry, the other operand is not.
            let (lb, rb) = if constant_lhs { (0, n * k) } else { (m * k, 0) };
            let mut lhs = uniform(0x5EED + m as u64, m * k * if constant_lhs { 1 } else { batch });
            let mut rhs = uniform(0xF00D + n as u64, n * k * if constant_lhs { batch } else { 1 });
            let run = |lhs: &[f32], rhs: &[f32]| {
                let mut dst = vec![f32::NAN; batch * m * n];
                let layouts = ((1, n), (lhs_cs, lhs_rs), (rhs_cs, rhs_rs));
                let (d, l, r) = layouts;
                assert!(gemm(&mut dst, lhs, rhs, (m, n, k), batch, (lb, rb), d, l, r, false));
                dst.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            };
            let constant = if constant_lhs {
                packed_cache::Constant::new(lhs.as_ptr(), lhs.len())
            } else {
                packed_cache::Constant::new(rhs.as_ptr(), rhs.len())
            };
            let held = |lhs: &[f32], rhs: &[f32]| panels_for(if constant_lhs { lhs } else { rhs });
            let per_call = run(&lhs, &rhs);
            assert_eq!(run(&lhs, &rhs), per_call, "{what}");
            assert_eq!(held(&lhs, &rhs), 0, "{what}: an undeclared operand was cached");
            packed_cache::with_constant(constant, || {
                for call in 0..3 {
                    assert_eq!(run(&lhs, &rhs), per_call, "{what}, call {call}");
                    assert_eq!(held(&lhs, &rhs), 1, "{what}, call {call}: one panel");
                }
            });
            // Another thread reads the same panel.
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    packed_cache::with_constant(constant, || {
                        assert_eq!(run(&lhs, &rhs), per_call, "{what}, another thread");
                    })
                });
            });
            assert_eq!(held(&lhs, &rhs), 1, "{what}: shared by the other thread");
            // New contents at the same address, as when a parameter is freed and another
            // allocated there, are packed again.
            let changed = if constant_lhs { &mut lhs } else { &mut rhs };
            changed.iter_mut().for_each(|x| *x = 1.0 - *x);
            let per_call = run(&lhs, &rhs);
            packed_cache::with_constant(constant, || {
                assert_eq!(run(&lhs, &rhs), per_call, "{what}, new contents");
            });
            assert_eq!(held(&lhs, &rhs), 1, "{what}: packed again");
        }
    }

    /// With `only_declared`, it takes a declared parameter times a smaller operand, and leaves
    /// an undeclared product, or one whose other operand is the larger, to the caller.
    #[test]
    fn only_declared_takes_a_parameter_times_a_smaller_operand() {
        use gemm::packed_cache::{Constant, with_constant};
        // `x @ W^T`: `m` rows of `x` against an `n` by `k` weight.
        let (n, k) = (64, 32);
        let weight = uniform(0xBEEF, n * k);
        let declared = Constant::new(weight.as_ptr(), weight.len());
        for (m, takes) in [(4, true), (n - 1, true), (n, false), (3 * n, false)] {
            let x = uniform(0xCAFE + m as u64, m * k);
            let mut dst = vec![0f32; m * n];
            let mut run =
                || gemm(&mut dst, &x, &weight, (m, n, k), 1, (0, 0), (1, n), (1, k), (k, 1), true);
            assert!(!run(), "m={m}: undeclared");
            let ran = with_constant(declared, run);
            assert_eq!(ran, takes && crate::quantized::kai::active(), "m={m}");
        }
    }

    #[test]
    fn declines_what_it_does_not_take() {
        let (lhs, rhs, mut dst) = (vec![1f32; 64], vec![1f32; 64], vec![0f32; 64]);
        let run = |dst: &mut [f32], m, d, l, r| {
            gemm(dst, &lhs, &rhs, (m, 4, 4), 1, (0, 0), d, l, r, false)
        };
        // One row, a strided lhs column, an rhs strided both ways, a strided dst column.
        assert!(!run(&mut dst, 1, (1, 4), (1, 4), (1, 4)));
        assert!(!run(&mut dst, 4, (1, 4), (2, 8), (1, 4)));
        assert!(!run(&mut dst, 4, (1, 4), (1, 4), (2, 8)));
        assert!(!run(&mut dst, 4, (2, 8), (1, 4), (1, 4)));
        // A dst too short for the rows.
        assert!(!run(&mut dst[..10], 4, (1, 4), (1, 4), (1, 4)));
        // Padded dst rows in a batch.
        assert!(!gemm(&mut dst, &lhs, &rhs, (4, 4, 4), 2, (16, 0), (1, 5), (1, 4), (1, 4), false));
        assert!(!dst.iter().any(|&x| x != 0.0));
    }
}
