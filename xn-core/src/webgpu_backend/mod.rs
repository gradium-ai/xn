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
//! readback / `synchronize` / before a host fallback). Each op runs in its own
//! compute pass, so WebGPU's automatic cross-pass hazard tracking orders reads
//! after prior writes. The flush waits for GPU completion via `Device::poll`.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use crate::{BinaryOp, DType, Result, UnaryOp, WithDType, WithDTypeF};
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
            "const S_NEG_BIG: S = -3.4028235e38;\n",
        )),
        "f16" => Some(concat!(
            "enable f16;\n",
            "alias S = f16;\n",
            "alias S4 = vec4<f16>;\n",
            // f16 has no -inf literal. Its most negative finite value plays the
            // same role in masking: exp(x - max) underflows to zero either way.
            "const S_NEG_BIG: S = -65504.0h;\n",
        )),
        _ => None,
    }
}

/// The WGSL scalar type for a dtype suffix, or `None` if WGSL has no such type.
fn wgsl_scalar(suffix: &str) -> Option<&'static str> {
    match suffix {
        "f32" => Some("f32"),
        "f16" => Some("f16"),
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

/// WGSL source for a kernel and its storage-buffer binding count, by base name
/// (no dtype suffix). `None` for an unknown kernel so a wrong dispatch fails
/// loudly rather than silently doing nothing.
fn kernel_src(base: &str) -> Option<(&'static str, u32)> {
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
        "gemv_tpc" => (include_str!("../../webgpu-kernels/gemv_tpc.wgsl"), 4),
        "conv1d" => (include_str!("../../webgpu-kernels/conv1d.wgsl"), 3),
        "conv_transpose1d" => (include_str!("../../webgpu-kernels/conv_transpose1d.wgsl"), 3),
        "im2col1d" => (include_str!("../../webgpu-kernels/im2col1d.wgsl"), 2),
        "col2im1d" => (include_str!("../../webgpu-kernels/col2im1d.wgsl"), 2),
        "cast" => (include_str!("../../webgpu-kernels/cast.wgsl"), 2),
        // dst, lhs (vec4 view), packed quants, scales, bias.
        "qgemv_q8" => (include_str!("../../webgpu-kernels/qgemv_q8.wgsl"), 5),
        "qgemm_q8" => (include_str!("../../webgpu-kernels/qgemm_q8.wgsl"), 5),
        _ => return None,
    };
    Some(def)
}

/// Largest storage-binding count any kernel uses (`qgemv_q8`). The device
/// allows far more; this only sizes the layout table.
const MAX_BINDINGS: usize = 5;
const PUSH_CONSTANT_SIZE: u32 = 128;
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

struct CachedPipeline {
    pipeline: wgpu::ComputePipeline,
    bindings: u32,
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
/// poll-wait (GPU busy) -> readback. Timing those phases separately shows
/// whether wall-clock is spent building commands on the CPU (`record_ns`),
/// blocked waiting on the GPU (`submit_wait_ns`), or in host readback.
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
    /// CPU time in submit + `poll(Wait)` (blocked on GPU execution).
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
}

pub struct DeviceInner {
    device: wgpu::Device,
    /// Idle `MAP_READ` staging buffers by size class. Creating one per readback
    /// dominated `readback_ns`; a readback only borrows it between `map_async`
    /// and `unmap`, so they recycle cleanly.
    staging: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
    queue: wgpu::Queue,
    // bind_group_layouts[n] / pipeline_layouts[n] describe `n` storage bindings.
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    pipeline_layouts: Vec<wgpu::PipelineLayout>,
    pipelines: Mutex<HashMap<String, CachedPipeline>>,
    pool: Mutex<BufferPool>,
    ctx: Mutex<OpCtx>,
    device_name: String,
    /// Whether the adapter advertises WGSL `shader-f16`.
    adapter_f16: bool,
    profile: bool,
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
        // What the adapter offers, which is not yet what the backend uses: the
        // kernels are f32-only, so this is currently reporting-only. It is the
        // gate an f16 compute path would have to check.
        let adapter_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);

        // Push constants (native feature) carry kernel parameters; f32 storage
        // buffers hold tensor data. Request a limit that fits the largest push
        // block (gemm: 14 u32 = 56 B) with headroom.
        let limits =
            wgpu::Limits { max_push_constant_size: PUSH_CONSTANT_SIZE, ..adapter.limits() };
        // f16 compute is opt-in per adapter. When it is missing the backend stays
        // f32-only and 16-bit storage falls back to the host, as before.
        let mut features = wgpu::Features::PUSH_CONSTANTS;
        if adapter_f16 {
            features |= wgpu::Features::SHADER_F16;
        }
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

        let inner = DeviceInner {
            device,
            queue,
            bind_group_layouts,
            pipeline_layouts,
            pipelines: Mutex::new(HashMap::new()),
            pool: Mutex::new(BufferPool::default()),
            staging: Mutex::new(HashMap::new()),
            ctx: Mutex::new(OpCtx {
                encoder: None,
                pass: None,
                open: false,
                free_bufs: Vec::new(),
            }),
            device_name,
            adapter_f16,
            profile,
            pstats: Mutex::new(ProfStats::default()),
        };
        Ok(Self(Arc::new(inner)))
    }

    /// Whether f16 tensors compute on the GPU. True when the adapter advertises
    /// WGSL `shader-f16`; otherwise 16-bit storage falls back to host loops.
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

    fn get_pipeline(&self, name: &str) -> Result<(wgpu::ComputePipeline, u32)> {
        {
            let pipelines = self.pipelines.lock().unwrap();
            if let Some(p) = pipelines.get(name) {
                return Ok((p.pipeline.clone(), p.bindings));
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
        let (body, bindings) = kernel_src(base)
            .ok_or_else(|| crate::Error::msg(format!("webgpu: unknown kernel {name}")))?;
        let src = format!("{preamble}{body}");
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
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
        let entry =
            pipelines.entry(name.to_string()).or_insert(CachedPipeline { pipeline, bindings });
        Ok((entry.pipeline.clone(), entry.bindings))
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
        let (pipeline, bindings) = self.get_pipeline(kernel)?;
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
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_push_constants(0, &push.bytes);
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
        }
        // Drive the queue to completion so host reads and buffer recycling are
        // safe. `poll(Wait)` blocks until all submitted work has finished.
        let t = (self.profile && had_work).then(std::time::Instant::now);
        self.device.poll(wgpu::PollType::Wait).map_err(wgpuerr("poll"))?;
        let poll_ns = t.map_or(0, |t| t.elapsed().as_nanos());
        if let Some(t0) = t0 {
            let mut p = self.pstats.lock().unwrap();
            p.submits += 1;
            p.submit_wait_ns += t0.elapsed().as_nanos();
            p.pass_end_ns += pass_end_ns;
            p.submit_ns += submit_ns;
            p.poll_ns += poll_ns;
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
    fn defer_free(&self, buf: PooledBuf) {
        self.ctx.lock().unwrap().free_bufs.push(buf);
    }

    /// Read `len` elements of `T` back from a GPU buffer into a host `Vec`.
    ///
    /// The staging copy is recorded into the batch that is already pending and
    /// flushed with it, so a readback costs one GPU round trip. Submitting the
    /// copy separately -- flush, then a second encoder and a second
    /// `poll(Wait)` -- measured ~1.3 ms per readback on Apple M5 even for a
    /// handful of bytes, because the second wait is a fresh submission rather
    /// than work already in flight.
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
        let staging = {
            let mut pool = self.staging.lock().unwrap();
            pool.get_mut(&class).and_then(|v| v.pop())
        }
        .unwrap_or_else(|| {
            // Class-sized, like the storage pool, so any same-class readback
            // can reuse it.
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("xn-readback"),
                size: class,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
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
        self.device.poll(wgpu::PollType::Wait).map_err(wgpuerr("poll (readback)"))?;
        rx.recv().map_err(wgpuerr("map recv"))?.map_err(wgpuerr("map_async"))?;
        let mapped = slice.get_mapped_range();
        let mut out = Vec::<T>::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(mapped.as_ptr(), out.as_mut_ptr() as *mut u8, bytes);
            out.set_len(len);
        }
        drop(mapped);
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
                "  submit+wait (blocked on GPU)     : {:>9.1} ms  ({:>4.1}%)",
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
            DType::F16 if self.adapter_f16 => Some("f16"),
            _ => None,
        }
    }

    /// Like [`Self::float_suffix`] but for ops with no host fallback, so an
    /// unsupported dtype is an error rather than a slow path.
    fn dtype_suffix<T: WithDType>(&self, op: &str) -> Result<&'static str> {
        self.float_suffix::<T>().ok_or_else(|| {
            crate::Error::msg(format!(
                "webgpu: {op} cannot run on {:?} on this device (f16 compute: {})",
                T::DTYPE,
                self.adapter_f16
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
