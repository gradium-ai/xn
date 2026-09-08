//! An f32 matmul path backed by XNNPACK's `fully_connected` operator.
//!
//! The `gemm` crate packs its operands on every call. For batch-1 decode that is most of the
//! cost: the weight panel is packed afresh for each frame and then used for only `m` rows of
//! output, and on this workload `m` is 16 (Mimi expands one latent into 16 timesteps). Perf
//! put `pack_lhs` at 13.4% of total runtime, and the gemm crate reaches only 13-21 GFLOP/s at
//! `m = 16` against 26-31 at `m = 128`.
//!
//! XNNPACK packs the weights inside `xnn_create_*` instead, so a cached operator pays for
//! packing once and every later call is pure arithmetic. Measured on this workload's shapes at
//! `m = 16`, that is 27-31 GFLOP/s -- what the gemm crate only reaches once `m` is 128.
//!
//! # Applicability
//!
//! `fully_connected` computes `out[m, n] = in[m, k] * W[n, k]^T`, which is exactly xn's
//! `matmul_t`, and the `[n, k]` weight layout it wants is the one xn already passes for
//! `matmul_t` and for `conv1d`'s im2col gemm. [`try_gemm_f32`] recognises that layout and
//! declines everything else, so the gemm crate stays the fallback.
//!
//! # Caching, and what it assumes
//!
//! Operators are cached per thread, because `xnn_setup_*` writes the input and output pointers
//! into the operator and so one operator cannot serve two threads at once. The key is the
//! weight's address, its shape, and a fingerprint of a few of its values.
//!
//! That makes an assumption worth stating: the weight behind a given address does not change
//! while the process runs. It holds for inference, where weights are loaded once and read for
//! the life of the model, and the fingerprint catches the case where an address is freed and
//! reused for different content. It would *not* hold if a weight were mutated in place, so this
//! is not a path to enable for training.

use std::cell::RefCell;
use std::collections::HashMap;
use std::os::raw::{c_float, c_uint, c_void};

type XnnOperator = *mut c_void;
type XnnStatus = c_uint;
const XNN_STATUS_SUCCESS: XnnStatus = 0;

unsafe extern "C" {
    fn xnn_initialize(allocator: *const c_void) -> XnnStatus;
    fn xnn_create_fully_connected_nc_f32(
        input_channels: usize,
        output_channels: usize,
        input_stride: usize,
        output_stride: usize,
        kernel: *const c_float,
        bias: *const c_float,
        output_min: c_float,
        output_max: c_float,
        flags: u32,
        weights_cache: *mut c_void,
        op_out: *mut XnnOperator,
    ) -> XnnStatus;
    fn xnn_reshape_fully_connected_nc_f32(
        op: XnnOperator,
        batch_size: usize,
        threadpool: *mut c_void,
    ) -> XnnStatus;
    fn xnn_setup_fully_connected_nc_f32(
        op: XnnOperator,
        input: *const c_float,
        output: *mut c_float,
    ) -> XnnStatus;
    fn xnn_run_operator(op: XnnOperator, threadpool: *mut c_void) -> XnnStatus;
}

/// Successful XNNPACK gemms, and ones declined back to the gemm crate. `XN_XNNPACK_STATS=1`
/// reports each distinct shape the first time it is seen, which is how you check that this
/// path is carrying the traffic you think it is rather than silently declining all of it.
static TAKEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static DECLINED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn stats() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XN_XNNPACK_STATS").as_deref() == Ok("1"))
}

/// Gemms taken by XNNPACK and gemms handed back to the gemm crate, since process start.
pub fn counters() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (TAKEN.load(Relaxed), DECLINED.load(Relaxed))
}

fn note(taken: bool, m: usize, n: usize, k: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    if taken { &TAKEN } else { &DECLINED }.fetch_add(1, Relaxed);
    if !stats() {
        return;
    }
    static SEEN: std::sync::Mutex<Vec<(bool, usize, usize, usize)>> =
        std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(taken, m, n, k)) {
        seen.push((taken, m, n, k));
        eprintln!(
            "xnnpack: {} m={m} n={n} k={k}",
            if taken { "taken  " } else { "declined" }
        );
    }
}

/// `XN_XNNPACK=0` forces the gemm-crate path, for A/B measurement.
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if matches!(std::env::var("XN_XNNPACK").as_deref(), Ok("0")) {
            return false;
        }
        // SAFETY: no arguments, and XNNPACK documents this as idempotent.
        unsafe { xnn_initialize(std::ptr::null()) == XNN_STATUS_SUCCESS }
    })
}

/// How the caller stores the weight. `fully_connected` wants `[n, k]`; a `[k, n]` weight is
/// transposed once when the operator is built, which the cache then amortises away.
///
/// Both appear in xn: `matmul_t` and `conv1d`'s im2col gemm pass `[n, k]`, while
/// `conv_transpose1d` passes `[k, n]`. Declining the latter would leave the decoder's
/// single most expensive gemm on the gemm crate.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
enum Layout {
    Nk,
    Kn,
}

/// Identifies the packed weight an operator holds. `n0`/`n1` bound the output-channel slice,
/// so a striped gemm caches one operator per stripe rather than one per whole weight.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct Key {
    ptr: usize,
    k: usize,
    n0: usize,
    n1: usize,
    out_stride: usize,
    layout: Layout,
    fingerprint: u64,
}

/// A few values spread through the weight, so that an address reused for different content is
/// not mistaken for a cache hit. Constant work regardless of weight size.
fn fingerprint(w: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ (w.len() as u64);
    let step = (w.len() / 8).max(1);
    let mut i = 0;
    while i < w.len() {
        h ^= w[i].to_bits() as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
        i += step;
    }
    h
}

struct Op {
    handle: XnnOperator,
    /// `xnn_reshape` is only needed when the row count changes.
    batch: usize,
}

// SAFETY: an `Op` never leaves the thread that created it -- the cache is thread-local and
// hands out only `&mut` borrows -- so the operator is never set up or run concurrently.
thread_local! {
    static CACHE: RefCell<HashMap<Key, Op>> = RefCell::new(HashMap::new());
}

/// Runs `dst[n0..n1] = lhs * weight[n0..n1]^T` for one stripe of output channels.
///
/// Returns false if XNNPACK declines the shape, leaving `dst` untouched.
#[allow(clippy::too_many_arguments)]
fn run_stripe(
    dst: &mut [f32],
    lhs: &[f32],
    weight: &[f32],
    m: usize,
    n: usize,
    k: usize,
    n0: usize,
    n1: usize,
    layout: Layout,
) -> bool {
    let oc = n1 - n0;
    // Fingerprint the whole weight rather than the stripe, so the cost and the result are the
    // same whichever layout it is in.
    let key = Key {
        ptr: weight.as_ptr() as usize,
        k,
        n0,
        n1,
        out_stride: n,
        layout,
        fingerprint: fingerprint(weight),
    };
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let op = match cache.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                // For [n, k] a stripe of output channels is already a contiguous row range.
                // For [k, n] it is a column range, so gather it into [oc, k] first. Either
                // way `create` copies it into its own packed form, so this is a one-off.
                let owned: Vec<f32>;
                let kernel: &[f32] = match layout {
                    Layout::Nk => &weight[n0 * k..n1 * k],
                    Layout::Kn => {
                        owned = (n0..n1)
                            .flat_map(|j| (0..k).map(move |p| weight[p * n + j]))
                            .collect();
                        &owned
                    }
                };
                let mut handle: XnnOperator = std::ptr::null_mut();
                // SAFETY: `kernel` is `oc * k` f32 as promised by `input_channels`/
                // `output_channels`; a null bias and null weights cache are both accepted.
                let st = unsafe {
                    xnn_create_fully_connected_nc_f32(
                        k,
                        oc,
                        k,
                        n,
                        kernel.as_ptr(),
                        std::ptr::null(),
                        f32::NEG_INFINITY,
                        f32::INFINITY,
                        0,
                        std::ptr::null_mut(),
                        &mut handle,
                    )
                };
                if st != XNN_STATUS_SUCCESS || handle.is_null() {
                    return false;
                }
                // The weights are packed inside `create`, so the kernel need not outlive this.
                e.insert(Op { handle, batch: 0 })
            }
        };
        // SAFETY: `handle` came from a successful create above and is owned by this thread.
        unsafe {
            if op.batch != m {
                if xnn_reshape_fully_connected_nc_f32(op.handle, m, std::ptr::null_mut())
                    != XNN_STATUS_SUCCESS
                {
                    return false;
                }
                op.batch = m;
            }
            // `dst` is the stripe's base, i.e. row 0 column n0; rows advance by `out_stride`.
            if xnn_setup_fully_connected_nc_f32(op.handle, lhs.as_ptr(), dst.as_mut_ptr())
                != XNN_STATUS_SUCCESS
            {
                return false;
            }
            xnn_run_operator(op.handle, std::ptr::null_mut()) == XNN_STATUS_SUCCESS
        }
    })
}

/// Which weight layout the rhs strides describe, if either.
fn rhs_layout(n: usize, k: usize, rhs_cs: usize, rhs_rs: usize) -> Option<Layout> {
    // A single row or column is both layouts at once; either reading is correct.
    if (rhs_cs, rhs_rs) == (k, 1) {
        Some(Layout::Nk)
    } else if (rhs_cs, rhs_rs) == (1, n) {
        Some(Layout::Kn)
    } else {
        None
    }
}

/// Attempts `dst = lhs * rhs` on XNNPACK, returning false if the operands are not in a
/// layout `fully_connected` can be given.
///
/// Required: `lhs` row-major `[m, k]`, `dst` row-major `[m, n]`, a single batch, and an `rhs`
/// that is either `[n, k]` (`rhs_cs == k, rhs_rs == 1`) or `[k, n]` (`rhs_cs == 1,
/// rhs_rs == n`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gemm_f32(
    dst: &mut [f32],
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    n: usize,
    k: usize,
    (dst_cs, dst_rs): (usize, usize),
    (lhs_cs, lhs_rs): (usize, usize),
    (rhs_cs, rhs_rs): (usize, usize),
) -> bool {
    if !enabled() {
        return false;
    }
    if m == 0
        || n == 0
        || k == 0
        || (dst_cs, dst_rs) != (1, n)
        || (lhs_cs, lhs_rs) != (1, k)
        || rhs_layout(n, k, rhs_cs, rhs_rs).is_none()
        || lhs.len() < m * k
        || rhs.len() < n * k
        || dst.len() < m * n
    {
        note(false, m, n, k);
        return false;
    }

    // Stripe by output channel on xn's own pool, matching what `gemm_` does, so this path
    // scales with threads without XNNPACK reaching for a second pool of its own. The split is
    // static rather than work-stealing so that a given stripe stays on a given thread and each
    // thread caches only its own slice of the packed weight.
    let nth = crate::threadpool::size();
    let stripes = if nth > 1 && n >= nth * 4 && m * n * k >= 1 << 14 { nth } else { 1 };
    // Checked non-None by the guard above.
    let layout = match rhs_layout(n, k, rhs_cs, rhs_rs) {
        Some(l) => l,
        None => return false,
    };
    if stripes == 1 {
        let ok = run_stripe(dst, lhs, rhs, m, n, k, 0, n, layout);
        note(ok, m, n, k);
        return ok;
    }

    let ok = std::sync::atomic::AtomicBool::new(true);
    let dst_ptr = dst.as_mut_ptr() as usize;
    crate::threadpool::dispatch(|ith, nth| {
        let per = n.div_ceil(nth);
        let n0 = (ith * per).min(n);
        let n1 = (n0 + per).min(n);
        if n0 == n1 {
            return;
        }
        // SAFETY: stripes own disjoint column ranges of every row, and `dst` outlives the
        // dispatch. The slice starts at row 0 column n0 and spans to the end of `dst`.
        let stripe = unsafe {
            std::slice::from_raw_parts_mut((dst_ptr as *mut f32).add(n0), m * n - n0)
        };
        if !run_stripe(stripe, lhs, rhs, m, n, k, n0, n1, layout) {
            ok.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    // A stripe that declined leaves part of `dst` unwritten, so the caller must redo the whole
    // product on the fallback. Shapes do not vary per stripe, so this is all-or-nothing in
    // practice.
    let ok = ok.load(std::sync::atomic::Ordering::Relaxed);
    note(ok, m, n, k);
    ok
}
