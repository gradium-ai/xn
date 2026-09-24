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
//!     lazily on first use and cached. Parameters travel in a uniform buffer
//!     addressed by a dynamic offset -- WebGPU has no push constants -- staged
//!     host-side and uploaded once per batch.
//!
//! Unlike the native GPU backends, WebGPU storage buffers cannot be persistently
//! host-mapped, so uploads go through `Queue::write_buffer` and readbacks copy
//! into a `MAP_READ` staging buffer. Host fallbacks therefore read their inputs
//! back, compute on the CPU and write the result out.
//!
//! Synchronization model: dispatches and buffer copies are recorded into a
//! single command encoder and only submitted when the batch is flushed (on host
//! readback / `synchronize` / before a host fallback). Each op runs in its own
//! compute pass, so WebGPU's automatic cross-pass hazard tracking orders reads
//! after prior writes. The flush waits for GPU completion via `Device::poll`.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use crate::{BinaryOp, DType, Result, Tensor, UnaryOp, WithDType, WithDTypeF};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

fn wgpuerr<E: std::fmt::Debug>(context: &str) -> impl Fn(E) -> crate::Error + '_ {
    move |e| crate::Error::msg(format!("webgpu: {context}: {e:?}"))
}

/// WGSL preamble that binds the storage scalar type. WGSL has no preprocessor,
/// so a kernel is specialised by prepending this to its source -- the same thing
/// the Vulkan backend does by including `dtype.glsl`.
///
/// Kernels declare storage as `array<S>` / `array<S4>` and convert explicitly at
/// the boundary: `f32(..)` on load, `S(..)` on store. Both are no-ops when `S`
/// is `f32`, so one source serves every dtype, and all arithmetic runs in f32
/// regardless of how the tensor is stored.
fn dtype_preamble(suffix: &str) -> Option<&'static str> {
    match suffix {
        "f32" => Some(concat!(
            "alias S = f32;\n",
            "alias S4 = vec4<f32>;\n",
            // Exactly -f32::MAX. Browsers evaluate the literal as an abstract
            // float and range-check it strictly, so the shortest round-trip form
            // Rust prints for f32::MIN (-3.4028235e38) is rejected as out of
            // range even though it rounds back to f32::MIN. Naga accepts it,
            // which is how this reached a browser as a bug in every f32 kernel.
            "const S_NEG_BIG: S = -3.4028234663852886e38;\n",
        )),
        _ => None,
    }
}

/// The WGSL scalar type for a dtype suffix, or `None` if WGSL has no such type.
fn wgsl_scalar(suffix: &str) -> Option<&'static str> {
    match suffix {
        "f32" => Some("f32"),
        _ => None,
    }
}

/// Preamble for `cast`, which is the one kernel reading and writing different
/// storage types: `SRC` for the source, `S` for the destination. `enable f16`
/// must appear once and before anything else, so it is emitted if either side
/// is f16.
fn cast_preamble(src: &str, dst: &str) -> Option<String> {
    let (src_ty, dst_ty) = (wgsl_scalar(src)?, wgsl_scalar(dst)?);
    let enable = if src_ty == "f16" || dst_ty == "f16" { "enable f16;\n" } else { "" };
    Some(format!("{enable}alias SRC = {src_ty};\nalias S = {dst_ty};\n"))
}

/// Every kernel base name. Kept beside `kernel_src` so the two stay in step;
/// used by [`all_shader_sources`] to hand the whole set to a validator.
pub const KERNEL_NAMES: &[&str] = &[
    "fill",
    "unary",
    "unary_inplace",
    "binary",
    "binary_inplace",
    "scale_add",
    "broadcast",
    "softmax",
    "rmsnorm",
    "layernorm",
    "rope",
    "rope_i",
    "reduce",
    "reduce_arg",
    "transpose",
    "copy2d",
    "copy_strided",
    "index_select",
    "causality_mask",
    "scatter_set",
    "gemm_tiled",
    "gemv",
    "gemv_tpc",
    "conv1d",
    "conv_transpose1d",
    "im2col1d",
    "col2im1d",
    "qgemv_q8",
    "qgemm_q8",
];

/// Composed WGSL for every kernel in every dtype, as `(dispatch name, source)`.
///
/// The backend compiles kernels lazily, and a browser's WGSL implementation is
/// stricter than Naga's, so this exists to hand the whole set to one for
/// validation without having to execute every op first. Cast kernels take two
/// dtypes and so are named separately.
pub fn all_shader_sources() -> Vec<(String, String)> {
    let dtypes = ["f32"];
    let mut out = Vec::new();
    for name in KERNEL_NAMES {
        for dt in dtypes {
            let Some(def) = kernel_src(name) else { continue };
            let Some(preamble) = dtype_preamble(dt) else { continue };
            let ops = if def.needs_ops { OPS_SRC } else { "" };
            out.push((format!("{name}_{dt}"), format!("{preamble}{ops}{}", def.src)));
        }
    }
    for src in dtypes {
        for dst in dtypes {
            let def = kernel_src("cast").expect("cast kernel");
            let preamble = cast_preamble(src, dst).expect("cast preamble");
            out.push((format!("cast_{src}_{dst}"), format!("{preamble}{}", def.src)));
        }
    }
    out
}

/// Shared elementwise op bodies, prepended for kernels that ask for them. WGSL
/// has no include mechanism, so snippets are composed here.
const OPS_SRC: &str = include_str!("../../webgpu-kernels/ops.wgsl");

/// A kernel's source and how it binds.
struct KernelDef {
    src: &'static str,
    /// Storage bindings, always at indices `0..bindings`.
    bindings: u32,
    /// Bit `i` set means binding `i` is written. Every other binding is declared
    /// read-only in WGSL and in the layout, which is what lets a kernel bind one
    /// buffer to several slots: WebGPU permits aliasing only when no aliased
    /// binding is writable (gemv binds its weights twice, scalar and vec4).
    writable: u32,
    /// Whether `ops.wgsl` is needed.
    needs_ops: bool,
}

/// WGSL source for a kernel by base name (no dtype suffix). `None` for an unknown
/// kernel so a wrong dispatch fails loudly rather than silently doing nothing.
fn kernel_src(base: &str) -> Option<KernelDef> {
    // (source, bindings, index of the written binding, needs ops.wgsl)
    let def: (&'static str, u32, u32, bool) = match base {
        "fill" => (include_str!("../../webgpu-kernels/fill.wgsl"), 1, 0, false),
        "unary" => (include_str!("../../webgpu-kernels/unary.wgsl"), 2, 1, true),
        "unary_inplace" => (include_str!("../../webgpu-kernels/unary_inplace.wgsl"), 1, 0, true),
        "binary" => (include_str!("../../webgpu-kernels/binary.wgsl"), 3, 2, true),
        "binary_inplace" => (include_str!("../../webgpu-kernels/binary_inplace.wgsl"), 2, 0, true),
        "scale_add" => (include_str!("../../webgpu-kernels/scale_add.wgsl"), 2, 1, false),
        "broadcast" => (include_str!("../../webgpu-kernels/broadcast.wgsl"), 4, 2, true),
        "softmax" => (include_str!("../../webgpu-kernels/softmax.wgsl"), 2, 1, false),
        "rmsnorm" => (include_str!("../../webgpu-kernels/rmsnorm.wgsl"), 3, 1, false),
        "layernorm" => (include_str!("../../webgpu-kernels/layernorm.wgsl"), 4, 1, false),
        "rope" => (include_str!("../../webgpu-kernels/rope.wgsl"), 4, 3, false),
        "rope_i" => (include_str!("../../webgpu-kernels/rope_i.wgsl"), 4, 3, false),
        "reduce" => (include_str!("../../webgpu-kernels/reduce.wgsl"), 2, 1, false),
        "reduce_arg" => (include_str!("../../webgpu-kernels/reduce_arg.wgsl"), 2, 1, false),
        "transpose" => (include_str!("../../webgpu-kernels/transpose.wgsl"), 2, 1, false),
        "copy2d" => (include_str!("../../webgpu-kernels/copy2d.wgsl"), 2, 1, false),
        "copy_strided" => (include_str!("../../webgpu-kernels/copy_strided.wgsl"), 3, 1, false),
        "index_select" => (include_str!("../../webgpu-kernels/index_select.wgsl"), 3, 1, false),
        "causality_mask" => (include_str!("../../webgpu-kernels/causality_mask.wgsl"), 1, 0, false),
        "scatter_set" => (include_str!("../../webgpu-kernels/scatter_set.wgsl"), 3, 0, false),
        "gemm_tiled" => (include_str!("../../webgpu-kernels/gemm_tiled.wgsl"), 3, 0, false),
        "gemv" => (include_str!("../../webgpu-kernels/gemv.wgsl"), 4, 0, false),
        "gemv_tpc" => (include_str!("../../webgpu-kernels/gemv_tpc.wgsl"), 4, 0, false),
        "conv1d" => (include_str!("../../webgpu-kernels/conv1d.wgsl"), 3, 0, false),
        "conv_transpose1d" => {
            (include_str!("../../webgpu-kernels/conv_transpose1d.wgsl"), 3, 0, false)
        }
        "im2col1d" => (include_str!("../../webgpu-kernels/im2col1d.wgsl"), 2, 0, false),
        "col2im1d" => (include_str!("../../webgpu-kernels/col2im1d.wgsl"), 2, 0, false),
        "cast" => (include_str!("../../webgpu-kernels/cast.wgsl"), 2, 0, false),
        "qgemv_q8" => (include_str!("../../webgpu-kernels/qgemv_q8.wgsl"), 5, 0, false),
        "qgemm_q8" => (include_str!("../../webgpu-kernels/qgemm_q8.wgsl"), 5, 0, false),
        _ => return None,
    };
    let (src, bindings, dst_binding, needs_ops) = def;
    Some(KernelDef { src, bindings, writable: 1u32 << dst_binding, needs_ops })
}

/// What `map_async` reports back through the readback channel. Only the
/// blocking reads use it, and those are native-only.
#[cfg(not(target_arch = "wasm32"))]
type MapResult = std::result::Result<(), wgpu::BufferAsyncError>;

/// Busy-poll budget for `Device::wait_for_queue` (see there), overridable with
/// `XN_WEBGPU_SPIN_US`; `0` blocks immediately. Native only: a browser can
/// neither spin nor block, so it awaits instead.
#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_SPIN_BUDGET_US: u64 = 2_000;

/// Largest storage-binding count any kernel uses (`qgemv_q8`). The device
/// allows far more; this only sizes the layout table.
/// Most storage bindings any kernel declares (`qgemv_q8`). Browsers cap
/// `maxStorageBuffersPerShaderStage` far lower than native adapters do -- Chrome
/// on Metal reports 10 against native wgpu's 31 -- so this is checked rather
/// than assumed.
const MAX_BINDINGS: u32 = 5;
/// Binding index for the kernel-parameter uniform, the same in every kernel so
/// storage bindings never have to be renumbered around it.
const PARAMS_BINDING: u32 = 8;
/// Bytes reserved per dispatch in the parameter ring. Must be at least the
/// largest `Params` struct (gemm's 14 u32 = 56 B) and a multiple of the device's
/// `min_uniform_buffer_offset_alignment`, which is 256 on every adapter wgpu
/// reports; asserted against the real limit at device creation.
const PARAMS_SLOT_SIZE: u64 = 256;
/// Dispatches worth of parameters held before a flush is forced. A decode frame
/// records a few hundred, so this only bounds pathological batches.
const PARAMS_RING_SLOTS: u64 = 4096;
const WORKGROUP_SIZE: u32 = 256;
/// GEMM tile size; must match `TILE` / the `@workgroup_size` in gemm_tiled.wgsl.
const TILE: u32 = 16;
/// Columns per workgroup in gemv_tpc.wgsl; must match its `@workgroup_size`.
const GEMV_TPC_COLS: u32 = 64;
/// Reduction length at which the cooperative gemv overtakes the thread-per-column
/// one. Below this a thread can walk a whole weight row without losing locality;
/// above it, 64 threads walking 64 rows at once thrash. Measured crossover on
/// Apple M5 sits between k = 768 and k = 1024.
const GEMV_TPC_MAX_K: usize = 1024;

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

/// Resolves when a wgpu completion callback fires.
///
/// Hand-rolled rather than pulling in a futures channel crate for this one use.
/// The waker is stored so a browser's event loop can drive the future; on native
/// the callback runs inside `poll`, so the future is usually already ready.
#[derive(Default)]
struct SignalState {
    done: bool,
    waker: Option<std::task::Waker>,
}

struct Signal(Arc<Mutex<SignalState>>);

impl Signal {
    /// Returns the future and the callback that completes it.
    fn new() -> (Self, impl FnOnce() + Send + 'static) {
        let state = Arc::new(Mutex::new(SignalState::default()));
        let fired = state.clone();
        let fire = move || {
            let mut st = fired.lock().unwrap();
            st.done = true;
            if let Some(w) = st.waker.take() {
                w.wake();
            }
        };
        (Signal(state), fire)
    }
}

impl std::future::Future for Signal {
    type Output = ();
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        let mut st = self.0.lock().unwrap();
        if st.done {
            std::task::Poll::Ready(())
        } else {
            st.waker = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }
}

struct CachedPipeline {
    pipeline: wgpu::ComputePipeline,
    bindings: u32,
    /// Built from the kernel's writable mask, so it is per-kernel rather than
    /// shared per binding count.
    bgl: wgpu::BindGroupLayout,
}

/// A pooled buffer plus its size class.
struct PooledBuf {
    buffer: wgpu::Buffer,
    class: u64,
}

/// Recycling pool for buffer allocations, keyed by size class. Decoding
/// allocates hundreds of intermediate tensors per token, so freed buffers are
/// returned here (after the batch referencing them has completed on the GPU)
/// and reused instead of being re-created.
#[derive(Default)]
struct BufferPool {
    free: HashMap<u64, Vec<PooledBuf>>,
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
    /// Compute passes opened. Ideally far below `dispatches`.
    passes: u64,
    copies: u64,
    submits: u64,
    readbacks: u64,
    /// CPU time building dispatches/copies into the encoder (GPU idle).
    record_ns: u128,
    /// CPU time in submit + draining the queue. The GPU is busy here, but the
    /// CPU is not idle: `wait_for_queue` spins for the first `spin_budget`.
    submit_wait_ns: u128,
    /// Of `submit_wait_ns`: ending the compute pass. wgpu buffers a pass's
    /// commands and only replays them into the platform encoder when the pass
    /// ends, so this is CPU encoding time, not GPU time.
    pass_end_ns: u128,
    /// Of `submit_wait_ns`: `finish()` + `queue.submit()`.
    submit_ns: u128,
    /// Of `submit_wait_ns`: `poll(Wait)` alone -- the only genuinely
    /// GPU-blocked portion.
    poll_ns: u128,
    /// CPU time in the readback staging copy + map (excludes the inner flush).
    readback_ns: u128,
    /// Per-kernel dispatch counts.
    per_kernel: HashMap<String, u64>,
    /// Readbacks bucketed by element count. Each one forces a flush, so the
    /// small buckets are the interesting ones: they drain the pipeline to move
    /// a handful of bytes.
    readback_elems: HashMap<usize, u64>,
    /// Per-kernel total workgroups. Divided by the dispatch count this gives the
    /// average launch size, which separates a kernel that is called constantly
    /// on tiny data from one that moves real bytes.
    per_kernel_groups: HashMap<String, u64>,
}

/// Command-recording state, guarded by a mutex. Dispatches/copies are recorded
/// into `encoder` and only submitted on flush.
struct OpCtx {
    encoder: Option<wgpu::CommandEncoder>,
    /// Compute pass held open across consecutive dispatches. WebGPU orders the
    /// dispatches within a pass and makes each one's writes visible to the
    /// next, so a whole run of ops can share a single pass. Ending one per op
    /// instead costs a full pipeline drain at every boundary -- on Metal each
    /// pass is its own `MTLComputeCommandEncoder`. Must be dropped (which ends
    /// the pass) before the encoder is used for a copy or finished.
    pass: Option<wgpu::ComputePass<'static>>,
    /// Whether `encoder` holds recorded, unsubmitted commands.
    open: bool,
    /// Buffers (dropped tensors + scratch) to recycle into the pool on the next
    /// flush, once the batch referencing them has finished executing.
    free_bufs: Vec<PooledBuf>,
    /// Kernel parameters for every dispatch in this batch, one `PARAMS_SLOT_SIZE`
    /// slot each, uploaded in a single write just before the batch is submitted.
    ///
    /// WebGPU has no push constants, so parameters travel in a uniform buffer
    /// addressed by a dynamic offset. Staging them host-side and writing once per
    /// batch keeps that to one `write_buffer` rather than one per dispatch.
    params: Vec<u8>,
}

pub struct DeviceInner {
    device: wgpu::Device,
    /// Ring of kernel-parameter slots, bound with a dynamic offset.
    params_ring: wgpu::Buffer,
    /// Idle `MAP_READ` staging buffers by size class. Creating one per readback
    /// dominated `readback_ns`; a readback only borrows it between `map_async`
    /// and `unmap`, so they recycle cleanly.
    staging: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
    queue: wgpu::Queue,
    pipelines: Mutex<HashMap<String, CachedPipeline>>,
    pool: Mutex<BufferPool>,
    ctx: Mutex<OpCtx>,
    device_name: String,
    /// Whether the adapter advertises WGSL `shader-f16`.
    adapter_f16: bool,
    profile: bool,
    /// Busy-poll budget for `wait_for_queue`; zero means block immediately.
    /// Native only: a browser can neither spin nor block, it awaits.
    #[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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
    /// Blocks on device creation. Kept on wasm as a failing stub rather than
    /// removed, so the synchronous `Runner` entry points still compile there and
    /// the reason surfaces as a message rather than a missing symbol.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(ordinal: usize) -> Result<Self> {
        pollster::block_on(Self::new_async(ordinal))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new(_ordinal: usize) -> Result<Self> {
        crate::bail!("webgpu: a browser cannot block on device creation; use Device::new_async")
    }

    pub async fn new_async(ordinal: usize) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        // A browser exposes no adapter enumeration -- `navigator.gpu` hands back
        // one adapter for a set of options -- so wasm asks for the high-performance
        // one and ignores `ordinal`.
        #[cfg(target_arch = "wasm32")]
        let adapter = {
            let _ = ordinal;
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .map_err(wgpuerr("request_adapter"))?
        };

        // Rank adapters by preference (discrete > integrated > cpu). `ordinal`
        // selects among the ranked list. `XN_WEBGPU_DEVICE` overrides it with a
        // raw enumeration index.
        #[cfg(not(target_arch = "wasm32"))]
        let adapter = {
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
            let idx = match std::env::var("XN_WEBGPU_DEVICE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
            {
                Some(i) if i < adapters.len() => i,
                _ => ranked.get(ordinal).map(|r| r.1).unwrap_or(ranked[0].1),
            };
            adapters[idx].clone()
        };
        let info = adapter.get_info();
        // Browsers mask the adapter name, so fall back to what they do report
        // rather than rendering an empty pair of parentheses.
        let device_name = if info.name.is_empty() {
            format!("{:?} {:?}", info.device_type, info.backend)
        } else {
            format!("{} ({:?})", info.name, info.backend)
        };
        // What the adapter offers, which is not yet what the backend uses: the
        // kernels are f32-only, so this is currently reporting-only. It is the
        // gate an f16 compute path would have to check.
        let adapter_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);

        // Kernel parameters travel in a uniform buffer, not push constants, so
        // the backend asks for no non-standard features beyond optional f16 and
        // stays inside what a browser exposes.
        let limits = adapter.limits();
        let uniform_align = u64::from(limits.min_uniform_buffer_offset_alignment);
        if !PARAMS_SLOT_SIZE.is_multiple_of(uniform_align) {
            crate::bail!(
                "webgpu: parameter slot size {PARAMS_SLOT_SIZE} is not a multiple of this \
                 adapter's min_uniform_buffer_offset_alignment ({uniform_align})"
            );
        }
        // No optional features are requested. A browser grants none of the native
        // ones, and this backend computes in f32, so there is nothing to ask for.
        let features = wgpu::Features::empty();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("xn-webgpu"),
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(wgpuerr("request_device (push-constant support required)"))?;

        let profile = std::env::var("XN_WEBGPU_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0");
        #[cfg(not(target_arch = "wasm32"))]
        let spin_budget = {
            let us = std::env::var("XN_WEBGPU_SPIN_US")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_SPIN_BUDGET_US);
            std::time::Duration::from_micros(us)
        };

        let params_ring = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xn-params"),
            size: PARAMS_SLOT_SIZE * PARAMS_RING_SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let inner = DeviceInner {
            device,
            params_ring,
            queue,
            pipelines: Mutex::new(HashMap::new()),
            pool: Mutex::new(BufferPool::default()),
            staging: Mutex::new(HashMap::new()),
            ctx: Mutex::new(OpCtx {
                encoder: None,
                pass: None,
                open: false,
                free_bufs: Vec::new(),
                params: Vec::new(),
            }),
            device_name,
            adapter_f16,
            profile,
            #[cfg(not(target_arch = "wasm32"))]
            spin_budget,
            pstats: Mutex::new(ProfStats::default()),
        };
        Ok(Self(Arc::new(inner)))
    }

    /// Whether the adapter advertises WGSL `shader-f16`. Reported for callers
    /// that probe the device; this backend computes in f32 either way, and
    /// 16-bit storage takes the host fallback.
    pub fn supports_f16(&self) -> bool {
        self.adapter_f16
    }

    /// Whether the underlying adapter advertises WGSL `shader-f16`.
    pub fn adapter_supports_f16(&self) -> bool {
        self.adapter_f16
    }

    /// WGSL has no `bf16` type, so bf16 storage always falls back to the host.
    pub fn supports_bf16(&self) -> bool {
        false
    }

    /// Allocate a buffer of at least `size_bytes`, reusing a pooled buffer of
    /// the same size class when one is available.
    fn alloc_buffer(&self, size_bytes: usize) -> wgpu::Buffer {
        let class = size_class(size_bytes);
        {
            let mut pool = self.pool.lock().unwrap();
            if let Some(b) = pool.free.get_mut(&class).and_then(|v| v.pop()) {
                pool.hits += 1;
                return b.buffer;
            }
            pool.misses += 1;
        }
        // Created at full class size so any same-class request can reuse it.
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xn-storage"),
            size: class,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[allow(clippy::type_complexity)]
    fn get_pipeline(
        &self,
        name: &str,
    ) -> Result<(wgpu::ComputePipeline, u32, wgpu::BindGroupLayout)> {
        {
            let pipelines = self.pipelines.lock().unwrap();
            if let Some(p) = pipelines.get(name) {
                return Ok((p.pipeline.clone(), p.bindings, p.bgl.clone()));
            }
        }
        // Dispatch names are `<base>_<dtype>`, or `cast_<src>_<dst>`.
        let (base, suffix) = name.rsplit_once('_').ok_or_else(|| {
            crate::Error::msg(format!("webgpu: kernel {name} has no dtype suffix"))
        })?;
        let (base, preamble) = match base.rsplit_once('_') {
            Some(("cast", src_suffix)) => (
                "cast",
                cast_preamble(src_suffix, suffix).ok_or_else(|| {
                    crate::Error::msg(format!("webgpu: no WGSL cast {src_suffix} -> {suffix}"))
                })?,
            ),
            _ => (
                base,
                dtype_preamble(suffix)
                    .ok_or_else(|| {
                        crate::Error::msg(format!("webgpu: no WGSL preamble for {suffix}"))
                    })?
                    .to_string(),
            ),
        };
        let def = kernel_src(base)
            .ok_or_else(|| crate::Error::msg(format!("webgpu: unknown kernel {name}")))?;
        let bindings = def.bindings;
        let ops = if def.needs_ops { OPS_SRC } else { "" };
        let src = format!("{preamble}{ops}{}", def.src);
        let bgl = self.bind_group_layout(name, bindings, def.writable);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(name),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let mut pipelines = self.pipelines.lock().unwrap();
        let entry =
            pipelines.entry(name.to_string()).or_insert(CachedPipeline { pipeline, bindings, bgl });
        Ok((entry.pipeline.clone(), entry.bindings, entry.bgl.clone()))
    }

    /// Bind group layout for a kernel: `bindings` storage buffers, of which only
    /// those set in `writable` are read_write, plus the parameter uniform.
    ///
    /// Declaring the rest read-only is what makes the backend WebGPU-legal:
    /// aliasing one buffer across several bindings is permitted only when none of
    /// them is writable, and several kernels do exactly that (gemv binds its
    /// weights as both scalars and vec4s).
    fn bind_group_layout(
        &self,
        label: &str,
        bindings: u32,
        writable: u32,
    ) -> wgpu::BindGroupLayout {
        debug_assert!(
            bindings <= MAX_BINDINGS,
            "kernel {label} declares {bindings} storage bindings, over the {MAX_BINDINGS} \
             this backend is documented to stay within"
        );
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..bindings)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: writable & (1u32 << i) == 0 },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: PARAMS_BINDING,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        });
        self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        })
    }

    /// Record a single dispatch of `kernel` (1D workgroup count).
    fn dispatch(
        &self,
        kernel: &str,
        buffers: &[&wgpu::Buffer],
        push: &Pc,
        groups_x: u32,
    ) -> Result<()> {
        self.dispatch_nd(kernel, buffers, push, (groups_x, 1, 1))
    }

    /// Record a dispatch of `kernel` with an explicit 3D workgroup count into
    /// the current batch (deferred; submitted on the next flush).
    fn dispatch_nd(
        &self,
        kernel: &str,
        buffers: &[&wgpu::Buffer],
        push: &Pc,
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let (gx, gy, gz) = groups;
        if gx == 0 || gy == 0 || gz == 0 {
            return Ok(());
        }
        let t0 = self.profile.then(std::time::Instant::now);
        let (pipeline, bindings, bgl) = self.get_pipeline(kernel)?;
        assert_eq!(bindings as usize, buffers.len(), "kernel {kernel} binding count mismatch");
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: PARAMS_BINDING,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.params_ring,
                offset: 0,
                size: Some(std::num::NonZeroU64::new(PARAMS_SLOT_SIZE).unwrap()),
            }),
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel),
            layout: &bgl,
            entries: &entries,
        });
        let mut ctx = self.ctx.lock().unwrap();
        // The ring is written in one go at flush, so a full ring means flushing
        // now rather than growing it.
        if ctx.params.len() as u64 + PARAMS_SLOT_SIZE > PARAMS_SLOT_SIZE * PARAMS_RING_SLOTS {
            self.flush_locked(&mut ctx)?;
        }
        let params_offset = ctx.params.len() as u32;
        if push.bytes.len() as u64 > PARAMS_SLOT_SIZE {
            crate::bail!(
                "webgpu: kernel {kernel} has {} bytes of parameters, slot is {PARAMS_SLOT_SIZE}",
                push.bytes.len()
            );
        }
        ctx.params.extend_from_slice(&push.bytes);
        ctx.params.resize(params_offset as usize + PARAMS_SLOT_SIZE as usize, 0);
        self.begin_if_needed(&mut ctx);
        let opened = ctx.pass.is_none();
        if opened {
            let enc = ctx.encoder.as_mut().unwrap();
            // `forget_lifetime` lets the pass outlive this call and sit in
            // `ctx` next to the encoder that owns it; `end_pass` drops it
            // before the encoder is touched again.
            let cpass = enc
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("xn"),
                    timestamp_writes: None,
                })
                .forget_lifetime();
            ctx.pass = Some(cpass);
        }
        {
            let cpass = ctx.pass.as_mut().unwrap();
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[params_offset]);
            cpass.dispatch_workgroups(gx, gy, gz);
        }
        drop(ctx);
        if let Some(t0) = t0 {
            let mut p = self.pstats.lock().unwrap();
            p.dispatches += 1;
            p.passes += u64::from(opened);
            p.record_ns += t0.elapsed().as_nanos();
            *p.per_kernel.entry(kernel.to_string()).or_insert(0) += 1;
            *p.per_kernel_groups.entry(kernel.to_string()).or_insert(0) +=
                u64::from(gx) * u64::from(gy) * u64::from(gz);
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
        self.begin_if_needed(&mut ctx);
        ctx.pass = None;
        ctx.encoder.as_mut().unwrap().copy_buffer_to_buffer(src, 0, dst, 0, bytes);
        drop(ctx);
        if let Some(t0) = t0 {
            let mut p = self.pstats.lock().unwrap();
            p.copies += 1;
            p.record_ns += t0.elapsed().as_nanos();
        }
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
        let (mut pass_end_ns, mut submit_ns) = (0u128, 0u128);
        if ctx.open {
            if !ctx.params.is_empty() {
                self.queue.write_buffer(&self.params_ring, 0, &ctx.params);
            }
            let t = self.profile.then(std::time::Instant::now);
            ctx.pass = None;
            if let Some(t) = t {
                pass_end_ns = t.elapsed().as_nanos();
            }
            let t = self.profile.then(std::time::Instant::now);
            let enc = ctx.encoder.take().unwrap();
            self.queue.submit(Some(enc.finish()));
            if let Some(t) = t {
                submit_ns = t.elapsed().as_nanos();
            }
            ctx.open = false;
            ctx.params.clear();
        }
        // Drive the queue to completion so host reads and buffer recycling are
        // safe.
        let t = (self.profile && had_work).then(std::time::Instant::now);
        self.poll_blocking()?;
        let poll_ns = t.map_or(0, |t| t.elapsed().as_nanos());
        // Recycling a buffer the GPU has not finished with would corrupt it, so
        // it happens only where completion is actually known. On wasm nothing can
        // block, so this sync flush leaves the buffers queued and `flush_async`
        // recycles them after awaiting.
        #[cfg(not(target_arch = "wasm32"))]
        Self::recycle(&self.pool, ctx);
        if let Some(t0) = t0 {
            let mut p = self.pstats.lock().unwrap();
            p.submits += 1;
            p.submit_wait_ns += t0.elapsed().as_nanos();
            p.pass_end_ns += pass_end_ns;
            p.submit_ns += submit_ns;
            p.poll_ns += poll_ns;
        }
        Ok(())
    }

    /// Move buffers freed by this batch back into the pool. Only sound once the
    /// batch that referenced them has completed.
    fn recycle(pool: &Mutex<BufferPool>, ctx: &mut OpCtx) {
        if ctx.free_bufs.is_empty() {
            return;
        }
        let mut pool = pool.lock().unwrap();
        for b in ctx.free_bufs.drain(..) {
            pool.free.entry(b.class).or_default().push(b);
        }
    }

    /// Block until the queue has drained. Native only: a browser has no way to
    /// block, so wasm builds must reach a completion point by awaiting instead
    /// (see `flush_async`).
    #[cfg(not(target_arch = "wasm32"))]
    fn poll_blocking(&self) -> Result<()> {
        self.wait_for_queue()
    }

    /// On wasm this only services callbacks that are already due; it cannot wait.
    /// Any code path that needs completion has to await `flush_async`.
    #[cfg(target_arch = "wasm32")]
    fn poll_blocking(&self) -> Result<()> {
        let _ = self.device.poll(wgpu::PollType::Poll);
        Ok(())
    }

    /// Submit any pending work and await its completion. The browser-safe
    /// counterpart to `flush`.
    pub async fn flush_async(&self) -> Result<()> {
        {
            let mut ctx = self.ctx.lock().unwrap();
            if ctx.open {
                if !ctx.params.is_empty() {
                    self.queue.write_buffer(&self.params_ring, 0, &ctx.params);
                }
                ctx.pass = None;
                let enc = ctx.encoder.take().expect("open batch has an encoder");
                self.queue.submit(Some(enc.finish()));
                ctx.open = false;
                ctx.params.clear();
            }
        }
        let (signal, fire) = Signal::new();
        self.queue.on_submitted_work_done(fire);
        // Native needs a poll to service the callback; in a browser the event
        // loop does it while this future is pending.
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::Wait).map_err(wgpuerr("poll (flush_async)"))?;
        signal.await;
        // Completion is known now, so queued buffers can go back in the pool.
        Self::recycle(&self.pool, &mut self.ctx.lock().unwrap());
        Ok(())
    }

    /// Schedule a buffer to be recycled into the pool on the next flush. Called
    /// from `Storage::drop`; the current (unsubmitted) batch may still reference
    /// the buffer, so recycling waits until the next flush completes.
    fn defer_free(&self, buf: PooledBuf) {
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
    #[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(not(target_arch = "wasm32"))]
    fn wait_for_map(&self, rx: &std::sync::mpsc::Receiver<MapResult>) -> Result<()> {
        self.wait_for_queue()?;
        rx.recv().map_err(wgpuerr("map recv"))?.map_err(wgpuerr("map_async"))
    }

    /// Read `len` elements of `T` back from a GPU buffer into a host `Vec`.
    ///
    /// The staging copy is recorded into the batch that is already pending and
    /// flushed with it, so a readback costs one GPU round trip. Submitting the
    /// copy separately -- flush, then a second encoder and a second
    /// `poll(Wait)` -- measured ~1.3 ms per readback on Apple M5 even for a
    /// handful of bytes, because the second wait is a fresh submission rather
    /// than work already in flight.
    #[cfg(not(target_arch = "wasm32"))]
    fn read_buffer<T: WithDType>(&self, buf: &wgpu::Buffer, len: usize) -> Result<Vec<T>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        // Timed in two segments around the flush, so the flush stays attributed
        // to `submit_wait_ns` alone and the three reported phases stay disjoint.
        // Debug aid: XN_WEBGPU_TRACE_READBACK=<len> prints one backtrace for the
        // first readback of that element count, to locate an unexpected sync.
        if std::env::var("XN_WEBGPU_TRACE_READBACK").is_ok_and(|v| v.parse::<usize>() == Ok(len)) {
            {
                static ONCE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = ONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 3 {
                    eprintln!(
                        "=== readback #{n} of {len} x {:?} ===\n{}",
                        T::DTYPE,
                        std::backtrace::Backtrace::force_capture()
                    );
                }
            }
        }
        let t0 = self.profile.then(std::time::Instant::now);
        let bytes = len * T::BYTE_SIZE;
        let padded = round4(bytes) as u64;
        let class = size_class(padded as usize);
        let staging = self.take_staging(class);
        let mut own_ns = t0.map_or(0, |t| t.elapsed().as_nanos());
        {
            // Append the copy to the pending batch and submit once. Any open
            // compute pass has to end first: the encoder cannot record a copy
            // while a pass borrowed from it is still live.
            let mut ctx = self.ctx.lock().unwrap();
            self.begin_if_needed(&mut ctx);
            ctx.pass = None;
            ctx.encoder.as_mut().unwrap().copy_buffer_to_buffer(buf, 0, &staging, 0, padded);
            self.flush_locked(&mut ctx)?;
        }
        let t1 = self.profile.then(std::time::Instant::now);

        let slice = staging.slice(..padded);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // The copy has already completed above, so this only runs the map
        // callback rather than waiting on a new submission.
        self.wait_for_map(&rx)?;
        let out = copy_mapped::<T>(&slice, len, bytes);
        staging.unmap();
        self.staging.lock().unwrap().entry(class).or_default().push(staging);
        if let Some(t1) = t1 {
            own_ns += t1.elapsed().as_nanos();
        }
        if self.profile {
            let mut p = self.pstats.lock().unwrap();
            p.readbacks += 1;
            p.readback_ns += own_ns;
            *p.readback_elems.entry(len).or_insert(0) += 1;
        }
        Ok(out)
    }

    /// In a browser the map callback is delivered by the event loop, so blocking
    /// on it here would deadlock rather than merely be slow. Every other op in
    /// the backend only records into the batch, so a browser caller can drive the
    /// whole op set through the synchronous `Backend` trait and await only
    /// [`Device::read_buffer_async`] where it wants values back.
    #[cfg(target_arch = "wasm32")]
    fn read_buffer<T: WithDType>(&self, _buf: &wgpu::Buffer, len: usize) -> Result<Vec<T>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        crate::bail!(
            "webgpu: a blocking readback of {len} {:?} would deadlock in a browser; \
             use Device::read_buffer_async",
            T::DTYPE
        )
    }

    /// Read `len` elements of `T` back without blocking.
    ///
    /// The browser-safe counterpart to `read_buffer`: same single-round-trip
    /// staging copy appended to the pending batch, but the completion is awaited
    /// rather than polled for. This is the only op in the backend that has to be
    /// async -- everything else merely records into the batch -- so a browser
    /// caller can drive the whole op set synchronously and await only where it
    /// actually wants values back.
    pub async fn read_buffer_async<T: WithDType>(
        &self,
        buf: &wgpu::Buffer,
        len: usize,
    ) -> Result<Vec<T>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let bytes = len * T::BYTE_SIZE;
        let padded = round4(bytes) as u64;
        let class = size_class(padded as usize);
        let staging = self.take_staging(class);
        {
            let mut ctx = self.ctx.lock().unwrap();
            self.begin_if_needed(&mut ctx);
            ctx.pass = None;
            ctx.encoder.as_mut().unwrap().copy_buffer_to_buffer(buf, 0, &staging, 0, padded);
        }
        self.flush_async().await?;

        let slice = staging.slice(..padded);
        let (signal, fire) = Signal::new();
        let status = Arc::new(Mutex::new(None));
        let st = status.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            *st.lock().unwrap() = Some(r);
            fire();
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::Wait).map_err(wgpuerr("poll (readback async)"))?;
        signal.await;
        match status.lock().unwrap().take() {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(wgpuerr("map_async")(e)),
            None => crate::bail!("webgpu: readback completed without a status"),
        }
        let out = copy_mapped::<T>(&slice, len, bytes);
        staging.unmap();
        self.staging.lock().unwrap().entry(class).or_default().push(staging);
        Ok(out)
    }

    /// A tensor's contents, read back without blocking.
    ///
    /// The browser-safe counterpart to `Tensor::to_vec`. Ops only record into the
    /// batch, so a browser caller runs them through the ordinary synchronous
    /// `Backend` trait and awaits this only where it wants values back.
    pub async fn tensor_to_vec<T: WithDType>(&self, t: &Tensor<T, Self>) -> Result<Vec<T>> {
        let len = t.shape().elem_count();
        let buffer = {
            let storage = t.storage()?;
            storage.buffer.clone()
        };
        self.read_buffer_async::<T>(&buffer, len).await
    }

    /// An idle `MAP_READ` staging buffer of this size class, or a new one.
    fn take_staging(&self, class: u64) -> wgpu::Buffer {
        if let Some(b) = self.staging.lock().unwrap().get_mut(&class).and_then(|v| v.pop()) {
            return b;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xn-readback"),
            size: class,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
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
        eprintln!(
            "{:>10} compute passes ({:.1} dispatches per pass)",
            p.passes,
            p.dispatches as f64 / (p.passes.max(1)) as f64
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
                "      of which: pass-end (cpu encode) {:>8.1} ms | submit {:>7.1} ms | poll (gpu) {:>8.1} ms",
                ms(p.pass_end_ns),
                ms(p.submit_ns),
                ms(p.poll_ns)
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
        let mut rb: Vec<_> = p.readback_elems.iter().collect();
        rb.sort_by_key(|r| std::cmp::Reverse(*r.1));
        if !rb.is_empty() {
            let list: Vec<String> =
                rb.iter().take(8).map(|(n, c)| format!("{n} elems x{c}")).collect();
            eprintln!("readbacks by size (each forces a flush): {}", list.join(", "));
        }
        let mut rows: Vec<_> = p.per_kernel.iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(*r.1));
        if !rows.is_empty() {
            eprintln!(
                "\n{:<22} {:>9} {:>12} {:>10}  share of dispatches",
                "kernel", "dispatches", "workgroups", "wg/disp"
            );
            let total: u64 = p.per_kernel.values().sum();
            for (k, c) in rows.iter() {
                let g = p.per_kernel_groups.get(*k).copied().unwrap_or(0);
                eprintln!(
                    "{:<22} {:>9} {:>12} {:>10.1}  {:>5.1}%",
                    k,
                    c,
                    g,
                    g as f64 / **c as f64,
                    100.0 * **c as f64 / total as f64
                );
            }
        }
        let pool = self.pool.lock().unwrap();
        let retained: u64 = pool.free.iter().map(|(c, v)| c * v.len() as u64).sum();
        let buffers: usize = pool.free.values().map(|v| v.len()).sum();
        eprintln!(
            "buffer pool retains {:.1} MB in {} idle buffers across {} size classes",
            retained as f64 / (1 << 20) as f64,
            buffers,
            pool.free.len(),
        );
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
    buffer: wgpu::Buffer,
    len: usize,
    /// Allocation size class; used to return the buffer to the pool on drop.
    class: u64,
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
        self.device.defer_free(PooledBuf { buffer: self.buffer.clone(), class: self.class });
    }
}

/// Copy `len` elements of `T` out of a mapped buffer range.
fn copy_mapped<T: WithDType>(slice: &wgpu::BufferSlice<'_>, len: usize, bytes: usize) -> Vec<T> {
    let mapped = slice.get_mapped_range();
    let mut out = Vec::<T>::with_capacity(len);
    unsafe {
        std::ptr::copy_nonoverlapping(mapped.as_ptr(), out.as_mut_ptr() as *mut u8, bytes);
        out.set_len(len);
    }
    out
}

/// Round a byte count up to the WebGPU 4-byte copy/write alignment.
fn round4(bytes: usize) -> usize {
    bytes.div_ceil(4) * 4
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

impl DeviceInner {
    /// Shader dtype suffix for `T`, or `None` when this device cannot compute on
    /// it and the caller must fall back to the host. f16 depends on the adapter
    /// advertising `shader-f16`; bf16 has no WGSL type at all.
    fn float_suffix<T: WithDType>(&self) -> Option<&'static str> {
        match T::DTYPE {
            DType::F32 => Some("f32"),
            _ => None,
        }
    }

    /// Like [`Self::float_suffix`] but for ops with no host fallback, so an
    /// unsupported dtype is an error rather than a slow path.
    fn dtype_suffix<T: WithDType>(&self, op: &str) -> Result<&'static str> {
        self.float_suffix::<T>().ok_or_else(|| {
            crate::Error::msg(format!(
                "webgpu: {op} cannot run on {:?}; this backend computes in f32",
                T::DTYPE
            ))
        })
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

pub mod quantization;
