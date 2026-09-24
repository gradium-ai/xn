//! WebGPU compute backend (via the `wgpu` crate).
//!
//! This backend runs on any GPU that `wgpu` can drive (Vulkan, Metal, DX12 or
//! GL under the hood) and is the portable counterpart to the native CUDA /
//! Metal / Vulkan backends. It mirrors the CUDA backend's op set but targets
//! the WebGPU compute model, so it makes a few deliberate simplifications:
//!
//!   * Compute is always done in `f32`. Only `f32` tensor storage takes the GPU
//!     path; `f16`/`bf16`/`i64`/`u8` storage falls back to host loops (WebGPU /
//!     WGSL has no `bf16` type and gates `f16` behind an optional feature, so a
//!     single portable `f32` path keeps the backend dependency-free). This is
//!     exactly how the Vulkan backend behaves on a device without 16-bit
//!     support.
//!   * Kernels are WGSL compute shaders (see `webgpu-kernels/`), compiled
//!     lazily on first use and cached. Parameters are passed as push constants
//!     (the `PUSH_CONSTANTS` native feature), matching the Vulkan/Metal layout.
//!
//! Unlike the native GPU backends, WebGPU storage buffers cannot be persistently
//! host-mapped, so uploads go through `Queue::write_buffer` and readbacks copy
//! into a `MAP_READ` staging buffer. Host fallbacks therefore read their inputs
//! back, compute on the CPU and write the result out.
//!
//! Synchronization model: dispatches and buffer copies are recorded into a
//! single command encoder and only submitted when the batch is flushed (on host
//! readback / `synchronize` / before a host fallback). Consecutive dispatches
//! share one compute pass; they execute in order and WebGPU requires each to
//! observe its predecessors' writes. A buffer copy cannot sit inside a pass, so
//! recording one closes it first. The flush waits via `Device::poll`.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use crate::{BinaryOp, DType, Result, UnaryOp, WithDType, WithDTypeF};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

pub mod quantization;

fn wgpuerr<E: std::fmt::Debug>(context: &str) -> impl Fn(E) -> crate::Error + '_ {
    move |e| crate::Error::msg(format!("webgpu: {context}: {e:?}"))
}

/// WGSL source for a kernel and its storage-buffer binding count. The dispatch
/// names carry a dtype suffix (always `_f32` on the GPU path) for parity with
/// the other backends; it is stripped here since only the `f32` variant exists.
/// `None` for an unknown kernel so a wrong dispatch fails loudly.
fn kernel_src(name: &str) -> Option<(&'static str, u32)> {
    let base = name.strip_suffix("_f32").unwrap_or(name);
    let def = match base {
        "fill" => (include_str!("../../webgpu-kernels/fill.wgsl"), 1),
        "unary" => (include_str!("../../webgpu-kernels/unary.wgsl"), 2),
        "binary" => (include_str!("../../webgpu-kernels/binary.wgsl"), 3),
        "scale_add" => (include_str!("../../webgpu-kernels/scale_add.wgsl"), 2),
        "broadcast" => (include_str!("../../webgpu-kernels/broadcast.wgsl"), 4),
        "softmax" => (include_str!("../../webgpu-kernels/softmax.wgsl"), 2),
        "rmsnorm" => (include_str!("../../webgpu-kernels/rmsnorm.wgsl"), 3),
        "layernorm" => (include_str!("../../webgpu-kernels/layernorm.wgsl"), 4),
        "rope" => (include_str!("../../webgpu-kernels/rope.wgsl"), 4),
        "rope_i" => (include_str!("../../webgpu-kernels/rope_i.wgsl"), 4),
        "reduce" => (include_str!("../../webgpu-kernels/reduce.wgsl"), 2),
        "reduce_arg" => (include_str!("../../webgpu-kernels/reduce_arg.wgsl"), 2),
        "transpose" => (include_str!("../../webgpu-kernels/transpose.wgsl"), 2),
        "copy2d" => (include_str!("../../webgpu-kernels/copy2d.wgsl"), 2),
        "copy_strided" => (include_str!("../../webgpu-kernels/copy_strided.wgsl"), 3),
        "index_select" => (include_str!("../../webgpu-kernels/index_select.wgsl"), 3),
        "causality_mask" => (include_str!("../../webgpu-kernels/causality_mask.wgsl"), 1),
        "scatter_set" => (include_str!("../../webgpu-kernels/scatter_set.wgsl"), 3),
        "gemm_tiled" => (GEMM_SRC, 3),
        // rhs is bound twice: scalar + a vec4 view for the aligned fast path.
        "gemv" => (GEMV_SRC, 4),
        // q8_0 weights: dst, lhs, quants, scales. Named without a dtype
        // suffix -- the activation is always f32 and the weight always q8_0.
        "gemv_q8" => (GEMV_Q8_SRC, 4),
        "gemm_q8" => (GEMM_Q8_SRC, 4),
        "gemm_q8_tiled" => (GEMM_Q8_TILED_SRC, 4),
        "conv1d" => (include_str!("../../webgpu-kernels/conv1d.wgsl"), 3),
        "conv_transpose1d" => (include_str!("../../webgpu-kernels/conv_transpose1d.wgsl"), 3),
        "im2col1d" => (include_str!("../../webgpu-kernels/im2col1d.wgsl"), 2),
        "col2im1d" => (include_str!("../../webgpu-kernels/col2im1d.wgsl"), 2),
        _ => return None,
    };
    Some(def)
}

/// What `map_async` reports back through the readback channel.
type MapResult = std::result::Result<(), wgpu::BufferAsyncError>;

const MAX_BINDINGS: usize = 4;
/// Binding index the kernels read their parameters from. Above every storage
/// binding any kernel declares, so it never collides with one.
const PARAMS_BINDING: u32 = 8;
/// Bytes reserved per dispatch in the parameter ring. Must be a multiple of the
/// adapter's `min_uniform_buffer_offset_alignment`, checked at device creation.
const PARAMS_SLOT_SIZE: u64 = 256;
/// Dispatches a batch can record before the ring must be flushed.
const PARAMS_RING_SLOTS: u64 = 4096;
const WORKGROUP_SIZE: u32 = 256;
/// The two matmul kernels, kept as named constants because the dispatch
/// geometry below is parsed back out of them.
const GEMM_SRC: &str = include_str!("../../webgpu-kernels/gemm_tiled.wgsl");
const GEMV_SRC: &str = include_str!("../../webgpu-kernels/gemv.wgsl");
const GEMV_Q8_SRC: &str = include_str!("../../webgpu-kernels/gemv_q8.wgsl");
const GEMM_Q8_SRC: &str = include_str!("../../webgpu-kernels/gemm_q8.wgsl");
const GEMM_Q8_TILED_SRC: &str = include_str!("../../webgpu-kernels/gemm_q8_tiled.wgsl");

/// How many times `pat` occurs in `src`. Const context only.
const fn count(src: &str, pat: &str) -> u32 {
    let (s, p) = (src.as_bytes(), pat.as_bytes());
    let (mut i, mut n) = (0, 0);
    while i + p.len() <= s.len() {
        let mut j = 0;
        while j < p.len() && s[i + j] == p[j] {
            j += 1;
        }
        if j == p.len() {
            n += 1;
        }
        i += 1;
    }
    n
}

/// Whether `pat` occurs in `src`. Const context only.
const fn contains(src: &str, pat: &str) -> bool {
    count(src, pat) > 0
}

/// Parses the decimal number written right after the sole occurrence of `pat`.
///
/// Used in const context only, to pin the constants below to the WGSL that
/// actually defines them. Absence, a non-numeric tail, and a *second*
/// occurrence are all compile-time panics rather than fallbacks. The last of
/// those matters as much as the first: taking the first match would let a
/// commented-out or stale earlier declaration -- `// const TILE: u32 = 16u`
/// above the real one -- quietly decide the dispatch geometry, which is the
/// silent failure this whole mechanism exists to prevent.
const fn u32_after(src: &str, pat: &str) -> u32 {
    // One pass, counting matches and remembering the first: this runs at
    // compile time for every constant below, and rescanning each shader once
    // per assertion is enough const-eval work to trip `long_running_const_eval`.
    let (s, p) = (src.as_bytes(), pat.as_bytes());
    let (mut i, mut n, mut at) = (0, 0, 0);
    while i + p.len() <= s.len() {
        let mut j = 0;
        while j < p.len() && s[i + j] == p[j] {
            j += 1;
        }
        if j == p.len() {
            if n == 0 {
                at = i;
            }
            n += 1;
        }
        i += 1;
    }
    assert!(
        n != 0,
        "WGSL: pattern not found -- the shader no longer declares what Rust reads from it"
    );
    assert!(
        n == 1,
        "WGSL: pattern occurs more than once -- which one sets the constant is ambiguous"
    );
    let mut k = at + p.len();
    let (mut v, mut digits) = (0u32, 0u32);
    while k < s.len() && s[k].is_ascii_digit() {
        v = v * 10 + (s[k] - b'0') as u32;
        k += 1;
        digits += 1;
    }
    assert!(digits > 0, "WGSL: pattern is not followed by a number");
    v
}

/// GEMM output-tile edge: the grid is `ceil(n/TILE) x ceil(m/TILE) x batch`.
/// Read out of the shader rather than restated, because the two drifting apart
/// has no loud failure mode -- a `TILE` larger here than there under-dispatches
/// and leaves the tail of `dst` holding whatever the buffer pool last wrote
/// into it, with no error from wgpu or anywhere else.
const TILE: u32 = u32_after(GEMM_SRC, "const TILE: u32 = ");
/// Output columns one GEMV workgroup produces: grid is `ceil(n/GEMV_TN)`. Same
/// reasoning as `TILE`.
const GEMV_TN: u32 = u32_after(GEMV_SRC, "const TN: u32 = ");
/// Dispatch geometry of the three q8_0 kernels, read out of the shaders for
/// the same reason as `TILE`: the grids are `ceil(n/GEMV_Q8_TN)`,
/// `ceil(n/GEMM_Q8_TN) x ceil(m/GEMM_Q8_MR)` and
/// `ceil(n/GEMM_Q8_TILE) x ceil(m/GEMM_Q8_TILE)`.
const GEMV_Q8_TN: u32 = u32_after(GEMV_Q8_SRC, "const TN: u32 = ");
const GEMM_Q8_TN: u32 = u32_after(GEMM_Q8_SRC, "const TN: u32 = ");
const GEMM_Q8_MR: u32 = u32_after(GEMM_Q8_SRC, "const MR: u32 = ");
const GEMM_Q8_TILE: u32 = u32_after(GEMM_Q8_TILED_SRC, "const TILE: u32 = ");

/// The sizes each kernel hardcodes, checked against the constants they were
/// derived from. Both kernels fully unroll their inner tile into named scalars
/// (`c00..c33`, `acc0..acc3`) and stage through fixed-size workgroup arrays, so
/// a change to one constant has to be carried by hand into several literals.
/// This turns "carried it everywhere" into a compile error rather than wrong
/// numbers at runtime.
const _: () = {
    // gemm_tiled.wgsl: a TILE x TILE output tile over a 64-thread (8x8)
    // workgroup, each thread owning an RT x RT patch of it.
    let kstep = u32_after(GEMM_SRC, "const KSTEP: u32 = ");
    let tpb = u32_after(GEMM_SRC, "const TPB: u32 = ");
    let rt = u32_after(GEMM_SRC, "const RT: u32 = ");
    assert!(tpb == 64, "gemm_tiled.wgsl: TPB must be the thread count of the 8x8 @workgroup_size");
    // Matched literally, so a reformat trips this rather than going unnoticed.
    assert!(
        contains(GEMM_SRC, "@workgroup_size(8, 8, 1)"),
        "gemm_tiled.wgsl: @workgroup_size must be spelled exactly `@workgroup_size(8, 8, 1)`"
    );
    assert!(rt == 4, "gemm_tiled.wgsl: the c00..c33 accumulators are unrolled for RT == 4");
    assert!(
        TILE == 8 * rt,
        "gemm_tiled.wgsl: TILE must be 8 threads x the RT-wide patch each one owns"
    );
    // Both operand tiles are staged `stage / TPB` elements per thread, and both
    // staging loops are written `for (var s = 0u; s < 4u; ...)`.
    let stage = TILE * kstep;
    assert!(
        stage == u32_after(GEMM_SRC, "var<workgroup> at: array<f32, "),
        "gemm_tiled.wgsl: `at` must hold TILE*KSTEP elements"
    );
    assert!(
        stage == u32_after(GEMM_SRC, "var<workgroup> bt: array<f32, "),
        "gemm_tiled.wgsl: `bt` must hold KSTEP*TILE elements"
    );
    assert!(
        stage == 4 * tpb,
        "gemm_tiled.wgsl: the staging loops are hardcoded to 4 elements per thread"
    );

    // gemv.wgsl: TPB threads each accumulating TN columns, reduced through one
    // shared array in log2(TPB) halving steps.
    let gtpb = u32_after(GEMV_SRC, "const TPB: u32 = ");
    // Checked before the `== 64` below, which would otherwise make it dead: this
    // is the constraint that survives a change of workgroup size, that one is
    // only today's value.
    assert!(
        gtpb.is_power_of_two(),
        "gemv.wgsl: the reduction halves the active thread count each step"
    );
    assert!(gtpb == 64, "gemv.wgsl: TPB must be the thread count of the @workgroup_size");
    // Matched literally, so a reformat trips this rather than going unnoticed.
    assert!(
        contains(GEMV_SRC, "@workgroup_size(64)"),
        "gemv.wgsl: @workgroup_size must be spelled exactly `@workgroup_size(64)`"
    );
    assert!(
        GEMV_TN == 4,
        "gemv.wgsl: the acc0..acc3 accumulators and the vec4 rhs view are unrolled for TN == 4"
    );
    assert!(
        gtpb * GEMV_TN == u32_after(GEMV_SRC, "var<workgroup> sh: array<f32, "),
        "gemv.wgsl: `sh` must hold one slot per (thread, column)"
    );

    // gemv_q8.wgsl: the f32 gemv's shape over a q8_0 weight -- TPB threads
    // each accumulating TN columns, reduced in log2(TPB) halving steps.
    let vtpb = u32_after(GEMV_Q8_SRC, "const TPB: u32 = ");
    assert!(
        vtpb.is_power_of_two(),
        "gemv_q8.wgsl: the reduction halves the active thread count each step"
    );
    assert!(vtpb == 64, "gemv_q8.wgsl: TPB must be the thread count of the @workgroup_size");
    // Matched literally, so a reformat trips this rather than going unnoticed.
    assert!(
        contains(GEMV_Q8_SRC, "@workgroup_size(64)"),
        "gemv_q8.wgsl: @workgroup_size must be spelled exactly `@workgroup_size(64)`"
    );
    assert!(GEMV_Q8_TN == 4, "gemv_q8.wgsl: the acc0..acc3 accumulators are unrolled for TN == 4");
    assert!(
        vtpb * GEMV_Q8_TN == u32_after(GEMV_Q8_SRC, "var<workgroup> sh: array<f32, "),
        "gemv_q8.wgsl: `sh` must hold one slot per (thread, column)"
    );

    // gemm_q8.wgsl: the same reduction widened to MR rows, so one pass over
    // the weight stream serves MR * TN outputs.
    let qtpb = u32_after(GEMM_Q8_SRC, "const TPB: u32 = ");
    let qacc = u32_after(GEMM_Q8_SRC, "const ACC: u32 = ");
    assert!(
        qtpb.is_power_of_two(),
        "gemm_q8.wgsl: the reduction halves the active thread count each step"
    );
    assert!(qtpb == 64, "gemm_q8.wgsl: TPB must be the thread count of the @workgroup_size");
    // Matched literally, so a reformat trips this rather than going unnoticed.
    assert!(
        contains(GEMM_Q8_SRC, "@workgroup_size(64)"),
        "gemm_q8.wgsl: @workgroup_size must be spelled exactly `@workgroup_size(64)`"
    );
    assert!(
        GEMM_Q8_MR == 4 && GEMM_Q8_TN == 4,
        "gemm_q8.wgsl: the a00..a33 accumulators are unrolled for MR == 4, TN == 4"
    );
    assert!(
        qacc == GEMM_Q8_MR * GEMM_Q8_TN,
        "gemm_q8.wgsl: ACC must be the MR x TN accumulator count"
    );
    assert!(
        qtpb * qacc == u32_after(GEMM_Q8_SRC, "var<workgroup> sh: array<f32, "),
        "gemm_q8.wgsl: `sh` must hold one slot per (thread, accumulator)"
    );

    // gemm_q8_tiled.wgsl: gemm_tiled's staged TILE x TILE shape, dequantizing
    // the weight into the `bt` stage so it is read once regardless of m.
    let ttpb = u32_after(GEMM_Q8_TILED_SRC, "const TPB: u32 = ");
    let tkstep = u32_after(GEMM_Q8_TILED_SRC, "const KSTEP: u32 = ");
    let trt = u32_after(GEMM_Q8_TILED_SRC, "const RT: u32 = ");
    assert!(
        ttpb == 64,
        "gemm_q8_tiled.wgsl: TPB must be the thread count of the 8x8 @workgroup_size"
    );
    // Matched literally, so a reformat trips this rather than going unnoticed.
    assert!(
        contains(GEMM_Q8_TILED_SRC, "@workgroup_size(8, 8, 1)"),
        "gemm_q8_tiled.wgsl: @workgroup_size must be spelled exactly `@workgroup_size(8, 8, 1)`"
    );
    assert!(trt == 4, "gemm_q8_tiled.wgsl: the c00..c33 accumulators are unrolled for RT == 4");
    assert!(
        GEMM_Q8_TILE == 8 * trt,
        "gemm_q8_tiled.wgsl: TILE must be 8 threads x the RT-wide patch each one owns"
    );
    let tstage = GEMM_Q8_TILE * tkstep;
    assert!(
        tstage == u32_after(GEMM_Q8_TILED_SRC, "var<workgroup> at: array<f32, "),
        "gemm_q8_tiled.wgsl: `at` must hold TILE*KSTEP elements"
    );
    assert!(
        tstage == u32_after(GEMM_Q8_TILED_SRC, "var<workgroup> bt: array<f32, "),
        "gemm_q8_tiled.wgsl: `bt` must hold KSTEP*TILE elements"
    );
};
/// Busy-poll budget for `Device::wait_for_queue` (see there), overridable with
/// `XN_WEBGPU_SPIN_US`; `0` blocks immediately.
const DEFAULT_SPIN_BUDGET_US: u64 = 2_000;

/// Little-endian push-constant byte builder. The WGSL kernels declare their
/// push constants as a struct of `u32`/`f32` fields, which have the same
/// tightly-packed 4-byte layout.
#[derive(Default)]
struct Pc {
    bytes: Vec<u8>,
}

impl Pc {
    fn new() -> Self {
        Self { bytes: Vec::with_capacity(64) }
    }
    fn u32(mut self, v: u32) -> Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn f32(mut self, v: f32) -> Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn usize(self, v: usize) -> Self {
        self.u32(v as u32)
    }
}

struct CachedPipeline {
    pipeline: wgpu::ComputePipeline,
    bindings: u32,
    /// Stable index, used to skip a redundant `set_pipeline` when consecutive
    /// dispatches in the same pass use the same kernel.
    idx: usize,
}

/// A wgpu buffer plus the size class it was allocated at.
///
/// The class travels with the buffer so `Storage` does not have to keep its own
/// copy to return it to the pool on drop. Derefs to `wgpu::Buffer`, so call
/// sites read the same as they did when this was a bare buffer.
struct Buf {
    buffer: wgpu::Buffer,
    class: u64,
}

impl std::ops::Deref for Buf {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Buf {
    /// Another handle to the same GPU buffer and size class.
    fn dup(&self) -> Buf {
        Buf { buffer: self.buffer.clone(), class: self.class }
    }
}

/// Recycling pool for buffer allocations, keyed by size class. Decoding
/// allocates hundreds of intermediate tensors per token, so freed buffers are
/// returned here (after the batch referencing them has completed on the GPU)
/// and reused instead of being re-created.
#[derive(Default)]
struct BufferPool {
    free: HashMap<u64, Vec<Buf>>,
    hits: u64,
    misses: u64,
}

/// Round a byte size up to its allocation class: the next power of two below
/// 1 MiB (256 B minimum), else a 1/16 subdivision of the enclosing power of two
/// (max ~12.5% waste). Buffers are created with the class size so any same-class
/// request can reuse them.
fn size_class(bytes: usize) -> u64 {
    let bytes = bytes.max(4) as u64;
    let np2 = bytes.next_power_of_two();
    if np2 <= (1 << 20) { np2.max(256) } else { bytes.div_ceil(np2 / 16) * (np2 / 16) }
}

/// Profiling counters (enabled via `XN_WEBGPU_PROFILE=1`). Because a whole
/// forward pass records into one encoder and flushes exactly once (at the
/// logits readback), the per-op timeline is: record ops (GPU idle) -> submit +
/// wait for the queue to drain (GPU busy) -> readback. Timing those phases
/// separately shows whether wall-clock is spent building commands on the CPU
/// (`record_ns`), waiting on the GPU (`submit_wait_ns`), or in host readback.
#[derive(Default)]
struct ProfStats {
    dispatches: u64,
    copies: u64,
    submits: u64,
    readbacks: u64,
    /// CPU time building dispatches/copies into the encoder (GPU idle).
    record_ns: u128,
    /// CPU time in submit + draining the queue. The GPU is busy here, but the
    /// CPU is not idle: `wait_for_queue` spins for the first `spin_budget`.
    submit_wait_ns: u128,
    /// CPU time in the readback staging copy + map. Excludes the flush that
    /// carries the copy, which is counted as a submit in `submit_wait_ns`.
    readback_ns: u128,
    /// Per-kernel dispatch counts.
    per_kernel: HashMap<String, u64>,
}

/// Command-recording state, guarded by a mutex. Dispatches/copies are recorded
/// into `encoder` and only submitted on flush.
struct OpCtx {
    /// The compute pass dispatches record into, held open across consecutive
    /// dispatches. Must be dropped (via `end_pass`) before the encoder is
    /// touched again or finished; `forget_lifetime` has erased the borrow that
    /// would otherwise make the compiler enforce that. The `Drop` impl below
    /// holds the invariant on the implicit path, so this does not depend on
    /// staying declared ahead of `encoder`.
    pass: Option<wgpu::ComputePass<'static>>,
    encoder: Option<wgpu::CommandEncoder>,
    /// `CachedPipeline::idx` of the pipeline currently bound in `pass`.
    last_pipeline: usize,
    /// Whether `encoder` holds recorded, unsubmitted commands.
    open: bool,
    /// Buffers (dropped tensors + scratch) to recycle into the pool on the next
    /// flush, once the batch referencing them has finished executing.
    free_bufs: Vec<Buf>,
    /// Kernel parameters for every dispatch in this batch, one
    /// `PARAMS_SLOT_SIZE` slot each, uploaded to `params_ring` in one write at
    /// flush. A browser has no push constants, so this is how parameters travel.
    params: Vec<u8>,
}

impl Drop for OpCtx {
    /// End the pass while the encoder is still alive.
    ///
    /// Every explicit path already calls `end_pass`, but a device dropped with
    /// a batch still pending (tensors built and discarded with no readback)
    /// reaches neither. Field declaration order alone would cover it, and
    /// nothing but a comment would hold that order in place -- reordering the
    /// two fields, or matching them to the struct literal in `Device::new`,
    /// compiles and passes every test. `Drop::drop` runs before any field is
    /// dropped, so this makes the ordering irrelevant.
    ///
    /// Sound here because no field is ever moved out of `OpCtx`: `flush_locked`
    /// uses `Option::take` and `Vec::drain`, which a `Drop` impl permits.
    fn drop(&mut self) {
        self.pass = None;
    }
}

pub struct DeviceInner {
    device: wgpu::Device,
    /// Kernel parameters for the batch in flight, one `PARAMS_SLOT_SIZE` slot
    /// per dispatch, addressed by a dynamic offset.
    params_ring: wgpu::Buffer,
    queue: wgpu::Queue,
    // bind_group_layouts[n] / pipeline_layouts[n] describe `n` storage bindings.
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    pipeline_layouts: Vec<wgpu::PipelineLayout>,
    pipelines: Mutex<HashMap<String, CachedPipeline>>,
    pool: Mutex<BufferPool>,
    ctx: Mutex<OpCtx>,
    device_name: String,
    profile: bool,
    /// Busy-poll budget for `wait_for_queue`; zero means block immediately.
    spin_budget: std::time::Duration,
    pstats: Mutex<ProfStats>,
}

#[derive(Clone)]
pub struct Device(Arc<DeviceInner>);

impl std::ops::Deref for Device {
    type Target = DeviceInner;
    fn deref(&self) -> &DeviceInner {
        &self.0
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebGpuDevice").field("name", &self.device_name).finish()
    }
}

fn device_type_score(t: wgpu::DeviceType) -> u32 {
    match t {
        wgpu::DeviceType::DiscreteGpu => 4,
        wgpu::DeviceType::IntegratedGpu => 3,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 1,
        wgpu::DeviceType::Other => 0,
    }
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        pollster::block_on(Self::new_async(ordinal))
    }

    async fn new_async(ordinal: usize) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        // Rank adapters by preference (discrete > integrated > cpu). `ordinal`
        // selects among the ranked list. `XN_WEBGPU_DEVICE` overrides it with a
        // raw enumeration index.
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        if adapters.is_empty() {
            crate::bail!("webgpu: no adapters found (is a GPU driver installed?)");
        }
        let mut ranked: Vec<(u32, usize)> = adapters
            .iter()
            .enumerate()
            .map(|(i, a)| (device_type_score(a.get_info().device_type), i))
            .collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let idx = match std::env::var("XN_WEBGPU_DEVICE").ok().and_then(|v| v.parse::<usize>().ok())
        {
            Some(i) if i < adapters.len() => i,
            _ => ranked.get(ordinal).map(|r| r.1).unwrap_or(ranked[0].1),
        };
        let adapter = &adapters[idx];
        let info = adapter.get_info();
        let device_name = format!("{} ({:?})", info.name, info.backend);

        // No feature or limit beyond what the adapter already reports: kernel
        // parameters travel in a uniform, and a browser grants none of wgpu's
        // native-only features.
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("xn-webgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(wgpuerr("request_device"))?;

        // A storage-buffer bind group layout + pipeline layout for each binding
        // count. Every binding is a read_write storage buffer (info/ids buffers
        // are declared read_write in WGSL too), so a single layout per count
        // serves every kernel with that many bindings.
        let mut bind_group_layouts = Vec::with_capacity(MAX_BINDINGS + 1);
        let mut pipeline_layouts = Vec::with_capacity(MAX_BINDINGS + 1);
        for n in 0..=MAX_BINDINGS {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n)
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .chain(std::iter::once(wgpu::BindGroupLayoutEntry {
                    binding: PARAMS_BINDING,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // One ring buffer, windowed to this dispatch's slot.
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(PARAMS_SLOT_SIZE),
                    },
                    count: None,
                }))
                .collect();
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("xn-bgl-{n}")),
                entries: &entries,
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("xn-pl-{n}")),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
            bind_group_layouts.push(bgl);
            pipeline_layouts.push(pl);
        }

        // A dynamic offset must be a multiple of this, and the slot size is what
        // every offset is a multiple of. 256 satisfies every adapter seen so far;
        // failing here beats miscomputing offsets on one that wants more.
        let uniform_align = device.limits().min_uniform_buffer_offset_alignment as u64;
        if !PARAMS_SLOT_SIZE.is_multiple_of(uniform_align) {
            crate::bail!(
                "webgpu: parameter slot size {PARAMS_SLOT_SIZE} is not a multiple of this \
                 device's uniform offset alignment ({uniform_align})"
            );
        }
        let params_ring = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xn-params"),
            size: PARAMS_SLOT_SIZE * PARAMS_RING_SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let profile = std::env::var("XN_WEBGPU_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0");
        let spin_us = std::env::var("XN_WEBGPU_SPIN_US")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_SPIN_BUDGET_US);
        let spin_budget = std::time::Duration::from_micros(spin_us);

        let inner = DeviceInner {
            device,
            params_ring,
            queue,
            bind_group_layouts,
            pipeline_layouts,
            pipelines: Mutex::new(HashMap::new()),
            pool: Mutex::new(BufferPool::default()),
            ctx: Mutex::new(OpCtx {
                params: Vec::new(),
                pass: None,
                encoder: None,
                last_pipeline: usize::MAX,
                open: false,
                free_bufs: Vec::new(),
            }),
            device_name,
            profile,
            spin_budget,
            pstats: Mutex::new(ProfStats::default()),
        };
        Ok(Self(Arc::new(inner)))
    }

    /// WebGPU compute is f32-only, so 16-bit storage is never taken on the GPU.
    /// Kept for API parity with the Vulkan/Metal backends.
    pub fn supports_f16(&self) -> bool {
        false
    }
    pub fn supports_bf16(&self) -> bool {
        false
    }

    /// Allocate a buffer of at least `size_bytes`, reusing a pooled buffer of
    /// the same size class when one is available.
    fn alloc_buffer(&self, size_bytes: usize) -> Buf {
        let class = size_class(size_bytes);
        {
            let mut pool = self.pool.lock().unwrap();
            if let Some(b) = pool.free.get_mut(&class).and_then(|v| v.pop()) {
                pool.hits += 1;
                return b;
            }
            pool.misses += 1;
        }
        // Created at full class size so any same-class request can reuse it.
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xn-storage"),
            size: class,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Buf { buffer, class }
    }

    fn get_pipeline(&self, name: &str) -> Result<(wgpu::ComputePipeline, u32, usize)> {
        {
            let pipelines = self.pipelines.lock().unwrap();
            if let Some(p) = pipelines.get(name) {
                return Ok((p.pipeline.clone(), p.bindings, p.idx));
            }
        }
        let (src, bindings) = kernel_src(name)
            .ok_or_else(|| crate::Error::msg(format!("webgpu: unknown kernel {name}")))?;
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(src)),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&self.pipeline_layouts[bindings as usize]),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let mut pipelines = self.pipelines.lock().unwrap();
        let idx = pipelines.len();
        let entry =
            pipelines.entry(name.to_string()).or_insert(CachedPipeline { pipeline, bindings, idx });
        Ok((entry.pipeline.clone(), entry.bindings, entry.idx))
    }

    /// Record a single dispatch of `kernel` (1D workgroup count).
    fn dispatch(&self, kernel: &str, buffers: &[&Buf], push: &Pc, groups_x: u32) -> Result<()> {
        self.dispatch_nd(kernel, buffers, push, (groups_x, 1, 1))
    }

    /// Record a dispatch of `kernel` with an explicit 3D workgroup count into
    /// the current batch (deferred; submitted on the next flush).
    fn dispatch_nd(
        &self,
        kernel: &str,
        buffers: &[&Buf],
        push: &Pc,
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let (gx, gy, gz) = groups;
        if gx == 0 || gy == 0 || gz == 0 {
            return Ok(());
        }
        let t0 = self.profile.then(std::time::Instant::now);
        let (pipeline, bindings, pidx) = self.get_pipeline(kernel)?;
        assert_eq!(bindings as usize, buffers.len(), "kernel {kernel} binding count mismatch");
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .chain(std::iter::once(wgpu::BindGroupEntry {
                binding: PARAMS_BINDING,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.params_ring,
                    offset: 0,
                    size: std::num::NonZeroU64::new(PARAMS_SLOT_SIZE),
                }),
            }))
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel),
            layout: &self.bind_group_layouts[bindings as usize],
            entries: &entries,
        });
        if push.bytes.len() as u64 > PARAMS_SLOT_SIZE {
            crate::bail!(
                "webgpu: kernel {kernel} has {} bytes of parameters, slot is {PARAMS_SLOT_SIZE}",
                push.bytes.len()
            );
        }
        let mut ctx = self.ctx.lock().unwrap();
        // The ring is uploaded in one write at flush, so a full ring means
        // flushing now rather than growing it.
        if ctx.params.len() as u64 + PARAMS_SLOT_SIZE > PARAMS_SLOT_SIZE * PARAMS_RING_SLOTS {
            self.flush_locked(&mut ctx)?;
        }
        let params_offset = ctx.params.len() as u32;
        ctx.params.extend_from_slice(&push.bytes);
        ctx.params.resize(params_offset as usize + PARAMS_SLOT_SIZE as usize, 0);
        self.ensure_pass(&mut ctx);
        let switch_pipeline = ctx.last_pipeline != pidx;
        ctx.last_pipeline = pidx;
        let cpass = ctx.pass.as_mut().unwrap();
        if switch_pipeline {
            cpass.set_pipeline(&pipeline);
        }
        cpass.set_bind_group(0, &bind_group, &[params_offset]);
        cpass.dispatch_workgroups(gx, gy, gz);
        drop(ctx);
        if let Some(t0) = t0 {
            let mut p = self.pstats.lock().unwrap();
            p.dispatches += 1;
            p.record_ns += t0.elapsed().as_nanos();
            *p.per_kernel.entry(kernel.to_string()).or_insert(0) += 1;
        }
        Ok(())
    }

    /// Record a buffer-to-buffer copy of `bytes` into the current batch.
    fn record_copy(&self, dst: &wgpu::Buffer, src: &wgpu::Buffer, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let t0 = self.profile.then(std::time::Instant::now);
        // Copies are size-aligned to 4 bytes; buffers are class-sized (>= 256,
        // multiple of 256) so rounding up never overruns the allocation.
        let bytes = round4(bytes) as u64;
        let mut ctx = self.ctx.lock().unwrap();
        // A copy cannot be recorded inside a compute pass.
        Self::end_pass(&mut ctx);
        self.begin_if_needed(&mut ctx);
        ctx.encoder.as_mut().unwrap().copy_buffer_to_buffer(src, 0, dst, 0, bytes);
        drop(ctx);
        if let Some(t0) = t0 {
            let mut p = self.pstats.lock().unwrap();
            p.copies += 1;
            p.record_ns += t0.elapsed().as_nanos();
        }
    }

    /// Open the compute pass if there is not one already.
    ///
    /// One pass per operator cost 38% of wall-clock in command recording with
    /// the GPU idle (on Metal a WebGPU pass is a fresh
    /// `MTLComputeCommandEncoder`). The Metal backend keeps one serial encoder
    /// open per batch and Vulkan records into one command buffer; this is the
    /// WebGPU equivalent, and ordering still holds because dispatches within a
    /// pass run in order and must observe their predecessors' writes.
    fn ensure_pass(&self, ctx: &mut OpCtx) {
        self.begin_if_needed(ctx);
        if ctx.pass.is_none() {
            let enc = ctx.encoder.as_mut().unwrap();
            let pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("xn"),
                timestamp_writes: None,
            });
            // `forget_lifetime` lets the pass outlive the borrow of `encoder`;
            // `end_pass` runs before every copy and before `finish`.
            ctx.pass = Some(pass.forget_lifetime());
            ctx.last_pipeline = usize::MAX;
        }
    }

    /// Close the open compute pass, if any. Required before recording anything
    /// else into the encoder (buffer copies) or finishing it.
    fn end_pass(ctx: &mut OpCtx) {
        ctx.pass = None;
        ctx.last_pipeline = usize::MAX;
    }

    fn begin_if_needed(&self, ctx: &mut OpCtx) {
        if ctx.encoder.is_none() {
            ctx.encoder = Some(
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("xn") }),
            );
        }
        ctx.open = true;
    }

    /// Submit any pending recorded commands and wait for completion. Safe to
    /// call when nothing is pending.
    fn flush(&self) -> Result<()> {
        let mut ctx = self.ctx.lock().unwrap();
        self.flush_locked(&mut ctx)
    }

    fn flush_locked(&self, ctx: &mut OpCtx) -> Result<()> {
        let had_work = ctx.open;
        let t0 = (self.profile && had_work).then(std::time::Instant::now);
        // The pass borrows the encoder; it has to go before `finish`.
        Self::end_pass(ctx);
        // `write_buffer` applies at the head of the submission, so one upload
        // here covers every dispatch recorded in this batch.
        if !ctx.params.is_empty() {
            self.queue.write_buffer(&self.params_ring, 0, &ctx.params);
        }
        if ctx.open {
            let enc = ctx.encoder.take().unwrap();
            self.queue.submit(Some(enc.finish()));
            ctx.open = false;
        }
        ctx.params.clear();
        // Drive the queue to completion so host reads and buffer recycling are
        // safe. See `wait_for_queue` for how the wait is split.
        self.wait_for_queue()?;
        if let Some(t0) = t0 {
            let mut p = self.pstats.lock().unwrap();
            p.submits += 1;
            p.submit_wait_ns += t0.elapsed().as_nanos();
        }
        if !ctx.free_bufs.is_empty() {
            let mut pool = self.pool.lock().unwrap();
            for b in ctx.free_bufs.drain(..) {
                pool.free.entry(b.class).or_default().push(b);
            }
        }
        Ok(())
    }

    /// Schedule a buffer to be recycled into the pool on the next flush. Called
    /// from `Storage::drop`; the current (unsubmitted) batch may still reference
    /// the buffer, so recycling waits until the next flush completes.
    fn defer_free(&self, buf: Buf) {
        self.ctx.lock().unwrap().free_bufs.push(buf);
    }

    /// Block until every submitted command has finished: busy-poll for
    /// `spin_budget`, then hand the core back and block.
    ///
    /// `poll(Wait)` is a real kernel wait on Vulkan (`vkWaitSemaphores`) and
    /// DX12 (`SetEventOnCompletion`), but not on Metal, where wgpu-hal polls
    /// `MTLCommandBuffer::status()` with a `thread::sleep(1ms)` between checks
    /// (`wgpu-hal/src/metal/device.rs`). There every wait costs a sleep quantum
    /// however briefly the GPU was busy -- which is why a 1 f32 and a 24,000
    /// f32 readback both measured ~1.3 ms, and why a decode frame ended up made
    /// of sleep quanta rather than of GPU work.
    ///
    /// Spinning is not free either: each `poll(Poll)` re-enters wgpu-core's
    /// whole `maintain` path (several locks plus a driver `get_fence_value`),
    /// with `ctx` held when the caller is `flush_locked`. So the budget is
    /// short -- enough for the sub-millisecond decode waits the win comes from,
    /// while prefill and software adapters, whose "GPU" work runs on the very
    /// cores being spun on, fall through to the blocking wait. Same trade as
    /// `crate::threadpool`'s spin-then-park.
    ///
    /// The two phases differ slightly: `is_queue_empty()` is stricter than
    /// `poll(Wait)`, which targets the last submission index as of the call, so
    /// a spinning thread also waits out other threads' concurrent submissions.
    /// Both cover the caller's own work; the budget bounds the difference.
    fn wait_for_queue(&self) -> Result<()> {
        let deadline = std::time::Instant::now() + self.spin_budget;
        while std::time::Instant::now() < deadline {
            let status = self.device.poll(wgpu::PollType::Poll).map_err(wgpuerr("poll"))?;
            if status.is_queue_empty() {
                return Ok(());
            }
            std::hint::spin_loop();
        }
        self.device.poll(wgpu::PollType::Wait).map_err(wgpuerr("poll"))?;
        Ok(())
    }

    /// Block until a pending `map_async` has reported back. A poll that reports
    /// the queue empty has already fired the map callback (wgpu-core collects
    /// the mapping closures before computing `queue_empty`), so draining the
    /// queue is sufficient; the blocking wait relies on the same ordering.
    fn wait_for_map(&self, rx: &std::sync::mpsc::Receiver<MapResult>) -> Result<()> {
        self.wait_for_queue()?;
        rx.recv().map_err(wgpuerr("map recv"))?.map_err(wgpuerr("map_async"))
    }

    /// Read `len` elements of `T` back from a GPU buffer into a host `Vec`.
    ///
    /// The staging copy goes into the pending batch rather than a submit of its
    /// own, so a readback costs one GPU wait rather than two: the dispatches
    /// producing `buf` are in the same encoder, and `write_buffer` uploads
    /// apply at the head of the submission, so both are ordered before it.
    fn read_buffer<T: WithDType>(&self, buf: &wgpu::Buffer, len: usize) -> Result<Vec<T>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let t0 = self.profile.then(std::time::Instant::now);
        let bytes = len * T::BYTE_SIZE;
        let padded = round4(bytes) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xn-readback"),
            size: padded,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut readback_ns = 0u128;
        {
            let mut ctx = self.ctx.lock().unwrap();
            // A copy cannot be recorded inside a compute pass.
            Self::end_pass(&mut ctx);
            self.begin_if_needed(&mut ctx);
            ctx.encoder.as_mut().unwrap().copy_buffer_to_buffer(buf, 0, &staging, 0, padded);
            if let Some(t0) = t0 {
                readback_ns += t0.elapsed().as_nanos();
            }
            // Counts as a submit, not as readback time.
            self.flush_locked(&mut ctx)?;
        }
        let t1 = self.profile.then(std::time::Instant::now);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait_for_map(&rx)?;
        let mapped = slice.get_mapped_range();
        let mut out = Vec::<T>::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(mapped.as_ptr(), out.as_mut_ptr() as *mut u8, bytes);
            out.set_len(len);
        }
        drop(mapped);
        staging.unmap();
        if let Some(t1) = t1 {
            readback_ns += t1.elapsed().as_nanos();
            let mut p = self.pstats.lock().unwrap();
            p.readbacks += 1;
            p.readback_ns += readback_ns;
        }
        Ok(out)
    }

    /// Upload a `u32` array (kernel `info` dims/strides scratch) into a buffer.
    fn write_buffer_u32(&self, buf: &wgpu::Buffer, data: &[u32]) {
        if data.is_empty() {
            return;
        }
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        self.queue.write_buffer(buf, 0, src);
    }

    /// Upload host data into a GPU buffer. The write is applied at the next
    /// queue submission, ahead of any command recorded after this call.
    fn write_buffer_data<T: WithDType>(&self, buf: &wgpu::Buffer, data: &[T]) {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return;
        }
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
        if bytes.is_multiple_of(4) {
            self.queue.write_buffer(buf, 0, src);
        } else {
            let mut padded = src.to_vec();
            padded.resize(round4(bytes), 0);
            self.queue.write_buffer(buf, 0, &padded);
        }
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        if !self.profile {
            return;
        }
        let p = self.pstats.lock().unwrap();
        let ms = |ns: u128| ns as f64 / 1e6;
        let record = ms(p.record_ns);
        let wait = ms(p.submit_wait_ns);
        let readback = ms(p.readback_ns);
        let total = record + wait + readback;
        eprintln!("\n=== xn webgpu profile: {} ===", self.device_name);
        eprintln!(
            "{:>10} dispatches, {:>6} copies, {:>5} submits, {:>5} readbacks",
            p.dispatches, p.copies, p.submits, p.readbacks
        );
        if total > 0.0 {
            eprintln!("CPU wall-clock split across the three phases (serial):");
            eprintln!(
                "  record   (build cmds, GPU idle) : {:>9.1} ms  ({:>4.1}%)",
                record,
                100.0 * record / total
            );
            eprintln!(
                "  submit+wait (GPU busy)           : {:>9.1} ms  ({:>4.1}%)",
                wait,
                100.0 * wait / total
            );
            eprintln!(
                "  readback (staging copy + map)    : {:>9.1} ms  ({:>4.1}%)",
                readback,
                100.0 * readback / total
            );
        }
        if p.submits > 0 {
            eprintln!(
                "per submit: {:.1} dispatches, {:.3} ms wait",
                p.dispatches as f64 / p.submits as f64,
                wait / p.submits as f64
            );
        }
        let mut rows: Vec<_> = p.per_kernel.iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(*r.1));
        let kernels: Vec<String> = rows.iter().take(8).map(|(k, c)| format!("{k}:{c}")).collect();
        if !kernels.is_empty() {
            eprintln!("top kernels (count): {}", kernels.join(", "));
        }
        let pool = self.pool.lock().unwrap();
        let allocs = pool.hits + pool.misses;
        if allocs > 0 {
            eprintln!(
                "buffer pool: {} hits / {} allocs ({:.1}% reuse)",
                pool.hits,
                allocs,
                100.0 * pool.hits as f64 / allocs as f64,
            );
        }
    }
}

/// WebGPU tensor storage: an `f32`-capable storage buffer. Host access goes
/// through readback/upload rather than a mapped pointer.
pub struct Storage<T: WithDType> {
    buffer: Buf,
    len: usize,
    device: Device,
    _t: PhantomData<T>,
}

impl<T: WithDType> Storage<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn to_host(&self) -> Result<Vec<T>> {
        self.device.read_buffer::<T>(&self.buffer, self.len)
    }
}

impl<T: WithDType> Drop for Storage<T> {
    fn drop(&mut self) {
        // The current (unsubmitted) batch may still reference this buffer, so
        // defer recycling it until the next flush completes on the GPU.
        self.device.defer_free(self.buffer.dup());
    }
}

/// Round a byte count up to the WebGPU 4-byte copy/write alignment.
fn round4(bytes: usize) -> usize {
    bytes.div_ceil(4) * 4
}

fn check_f32<T: WithDType>(op: &str) -> Result<()> {
    if T::DTYPE != DType::F32 {
        crate::bail!("webgpu: {op} only supports f32, got {:?}", T::DTYPE);
    }
    Ok(())
}

/// Convert a float-typed scalar to `f32`. Only called on the GPU path, where
/// `T` has already been restricted to f32.
fn scalar_to_f32<T: WithDType>(v: T) -> f32 {
    match T::DTYPE {
        DType::F32 => unsafe { *(&v as *const T as *const f32) },
        DType::F16 => unsafe { (*(&v as *const T as *const half::f16)).to_f32() },
        DType::BF16 => unsafe { (*(&v as *const T as *const half::bf16)).to_f32() },
        d => unreachable!("scalar_to_f32 on non-float dtype {d:?}"),
    }
}

/// Shader dtype suffix for a float storage type. Only `f32` runs on the GPU;
/// `f16`/`bf16` error (callers with a host fallback use `float_suffix`).
fn dtype_suffix<T: WithDType>(op: &str) -> Result<&'static str> {
    match T::DTYPE {
        DType::F32 => Ok("f32"),
        d => crate::bail!("webgpu: {op} only supports f32 on the GPU, got {d:?}"),
    }
}

/// `Some("f32")` for f32 storage (GPU path), `None` otherwise (host fallback).
fn float_suffix<T: WithDType>() -> Option<&'static str> {
    match T::DTYPE {
        DType::F32 => Some("f32"),
        _ => None,
    }
}

fn bin_apply<T: WithDType>(op: BinaryOp, a: T, b: T) -> T {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Maximum => {
            if a > b {
                a
            } else {
                b
            }
        }
        BinaryOp::Minimum => {
            if a < b {
                a
            } else {
                b
            }
        }
    }
}

fn unary_op_code(op: UnaryOp) -> (u32, f32) {
    match op {
        UnaryOp::Cos => (0, 0.0),
        UnaryOp::Sin => (1, 0.0),
        UnaryOp::Exp => (2, 0.0),
        UnaryOp::Log => (3, 0.0),
        UnaryOp::Neg => (4, 0.0),
        UnaryOp::Sqr => (5, 0.0),
        UnaryOp::Sqrt => (6, 0.0),
        UnaryOp::Rsqrt => (7, 0.0),
        UnaryOp::Abs => (8, 0.0),
        UnaryOp::GeluErf => (9, 0.0),
        UnaryOp::Elu { alpha } => (10, alpha),
        UnaryOp::Relu => (11, 0.0),
        UnaryOp::Silu => (12, 0.0),
        UnaryOp::Tanh => (13, 0.0),
        UnaryOp::Sigmoid => (14, 0.0),
    }
}

fn binary_op_code(op: BinaryOp) -> u32 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Maximum => 4,
        BinaryOp::Minimum => 5,
    }
}

fn div_ceil(n: usize, d: u32) -> u32 {
    (n as u32).div_ceil(d)
}

include!("backend_impl.rs");
