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
        "gemm_tiled" => (include_str!("../../webgpu-kernels/gemm_tiled.wgsl"), 3),
        // rhs is bound twice: scalar + a vec4 view for the aligned fast path.
        "gemv" => (include_str!("../../webgpu-kernels/gemv.wgsl"), 4),
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
const PUSH_CONSTANT_SIZE: u32 = 128;
const WORKGROUP_SIZE: u32 = 256;
/// GEMM tile size; must match `TILE` / the `@workgroup_size` in gemm_tiled.wgsl.
const TILE: u32 = 16;
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
    encoder: Option<wgpu::CommandEncoder>,
    /// The compute pass dispatches record into, held open across consecutive
    /// dispatches. Must be dropped (via `end_pass`) before the encoder is
    /// touched again or finished.
    pass: Option<wgpu::ComputePass<'static>>,
    /// `CachedPipeline::idx` of the pipeline currently bound in `pass`.
    last_pipeline: usize,
    /// Whether `encoder` holds recorded, unsubmitted commands.
    open: bool,
    /// Buffers (dropped tensors + scratch) to recycle into the pool on the next
    /// flush, once the batch referencing them has finished executing.
    free_bufs: Vec<Buf>,
}

pub struct DeviceInner {
    device: wgpu::Device,
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

        // Push constants (native feature) carry kernel parameters; f32 storage
        // buffers hold tensor data. Request a limit that fits the largest push
        // block (gemm: 14 u32 = 56 B) with headroom.
        let limits =
            wgpu::Limits { max_push_constant_size: PUSH_CONSTANT_SIZE, ..adapter.limits() };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("xn-webgpu"),
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(wgpuerr("request_device (push-constant support required)"))?;

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
                .collect();
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("xn-bgl-{n}")),
                entries: &entries,
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("xn-pl-{n}")),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..PUSH_CONSTANT_SIZE,
                }],
            });
            bind_group_layouts.push(bgl);
            pipeline_layouts.push(pl);
        }

        let profile = std::env::var("XN_WEBGPU_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0");
        let spin_us = std::env::var("XN_WEBGPU_SPIN_US")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_SPIN_BUDGET_US);
        let spin_budget = std::time::Duration::from_micros(spin_us);

        let inner = DeviceInner {
            device,
            queue,
            bind_group_layouts,
            pipeline_layouts,
            pipelines: Mutex::new(HashMap::new()),
            pool: Mutex::new(BufferPool::default()),
            ctx: Mutex::new(OpCtx {
                encoder: None,
                pass: None,
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
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel),
            layout: &self.bind_group_layouts[bindings as usize],
            entries: &entries,
        });
        let mut ctx = self.ctx.lock().unwrap();
        self.ensure_pass(&mut ctx);
        let switch_pipeline = ctx.last_pipeline != pidx;
        ctx.last_pipeline = pidx;
        let cpass = ctx.pass.as_mut().unwrap();
        if switch_pipeline {
            cpass.set_pipeline(&pipeline);
        }
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_push_constants(0, &push.bytes);
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
        if ctx.open {
            let enc = ctx.encoder.take().unwrap();
            self.queue.submit(Some(enc.finish()));
            ctx.open = false;
        }
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
