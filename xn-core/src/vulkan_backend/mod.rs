//! Vulkan compute backend.
//!
//! This backend targets integrated GPUs (in particular AMD APUs) where the GPU
//! shares system memory with the CPU. It allocates all tensor storage from a
//! memory type that is simultaneously `DEVICE_LOCAL`, `HOST_VISIBLE` and
//! `HOST_COHERENT` (the "BAR"/unified type exposed by APUs), and keeps every
//! buffer persistently mapped. Uploads, readbacks and fills are therefore plain
//! `memcpy`s with no staging buffers.
//!
//! Compute kernels are GLSL compute shaders compiled to SPIR-V at build time
//! (see `build.rs` / `vulkan-kernels/`). They currently operate on `f32`, which
//! is the compute dtype used by the Vulkan inference path. Data-movement ops
//! (copy, fill, dtype conversion, and the layout/indexing ops for non-`f32`
//! element types) run on the host over the mapped memory.
//!
//! Synchronization model: each compute dispatch is submitted and waited on
//! before returning. Because storage is host-coherent and no GPU work is ever
//! left in flight, host accesses to mapped memory are always consistent without
//! explicit barriers. This is simple and correct; batching dispatches is a
//! future optimization.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use crate::{BinaryOp, DType, Result, UnaryOp, WithDType, WithDTypeF};
use ash::vk;
use std::collections::HashMap;
use std::ffi::CStr;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

#[allow(dead_code)]
mod shaders {
    include!(concat!(env!("OUT_DIR"), "/vulkan_shaders.rs"));
}

pub mod quantization;

fn vkerr<E: std::fmt::Debug>(context: &str) -> impl Fn(E) -> crate::Error + '_ {
    move |e| crate::Error::msg(format!("vulkan: {context}: {e:?}"))
}

/// Definition of a compute kernel given a dtype-suffixed name such as
/// `"unary_f16"`: returns its SPIR-V (for the requested dtype) and its number
/// of storage-buffer bindings (all bound at set 0, bindings `0..bindings`).
/// Unsupported (kernel, dtype) combinations return `None` so that a wrong
/// dispatch fails loudly instead of silently running the wrong variant.
fn kernel_def(name: &str) -> Option<(&'static [u8], u32)> {
    use shaders::*;
    // Cast kernels are named by (src, dst) dtype pair rather than one dtype.
    if let Some(pair) = name.strip_prefix("cast_") {
        let bytes: &'static [u8] = match pair {
            "f32_f16" => CAST_F32_F16,
            "f16_f32" => CAST_F16_F32,
            "f32_bf16" => CAST_F32_BF16,
            "bf16_f32" => CAST_BF16_F32,
            "f16_bf16" => CAST_F16_BF16,
            "bf16_f16" => CAST_BF16_F16,
            "i64_f32" => CAST_I64_F32,
            _ => return None,
        };
        return Some((bytes, 2));
    }
    // q8_0 weight kernels: dst, lhs, quants, scales. Named without a dtype
    // suffix -- the activation is always f32 and the weight always q8_0.
    // The `_r<MR>` family is a row-block kernel built once per block height.
    match name {
        "gemv_q8" => return Some((GEMV_Q8_F32, 4)),
        "gemm_q8" => return Some((GEMM_Q8_F32, 4)),
        "gemv_q8_sg" => return Some((GEMV_Q8_SG_F32, 4)),
        "gemm_q8_sg_r1" => return Some((GEMM_Q8_SG_R1, 4)),
        "gemm_q8_sg_r2" => return Some((GEMM_Q8_SG_R2, 4)),
        "gemm_q8_sg_r4" => return Some((GEMM_Q8_SG_R4, 4)),
        "gemm_q8_sg_r8" => return Some((GEMM_Q8_SG_R8, 4)),
        "gemm_q8_sg_r16" => return Some((GEMM_Q8_SG_R16, 4)),
        "dequant_q8" => return Some((DEQUANT_Q8_F32, 3)),
        // f32 GEMMs: dst, lhs, rhs.
        "gemm_tiled64" => return Some((GEMM_TILED64_F32, 3)),
        "gemm_tiled32" => return Some((GEMM_TILED64_TM32, 3)),
        "gemm_tiled16" => return Some((GEMM_TILED16_F32, 3)),
        "ksplit_reduce" => return Some((KSPLIT_REDUCE_F32, 2)),
        "gemm_nt_sg_r1" => return Some((GEMM_NT_SG_R1, 3)),
        "gemm_nt_sg_r2" => return Some((GEMM_NT_SG_R2, 3)),
        "gemm_nt_sg_r4" => return Some((GEMM_NT_SG_R4, 3)),
        "gemm_nt_sg_r8" => return Some((GEMM_NT_SG_R8, 3)),
        "gemm_nt_sg_r16" => return Some((GEMM_NT_SG_R16, 3)),
        "gemm_nn_rows_r1" => return Some((GEMM_NN_ROWS_R1, 3)),
        "gemm_nn_rows_r2" => return Some((GEMM_NN_ROWS_R2, 3)),
        "gemm_nn_rows_r4" => return Some((GEMM_NN_ROWS_R4, 3)),
        "gemm_nn_rows_r8" => return Some((GEMM_NN_ROWS_R8, 3)),
        "gemm_nn_rows_r16" => return Some((GEMM_NN_ROWS_R16, 3)),
        "gemm_nn_rows_r1s" => return Some((GEMM_NN_ROWS_R1S, 3)),
        "gemm_nn_rows_r2s" => return Some((GEMM_NN_ROWS_R2S, 3)),
        "gemm_nn_rows_r4s" => return Some((GEMM_NN_ROWS_R4S, 3)),
        "gemm_nn_rows_r8s" => return Some((GEMM_NN_ROWS_R8S, 3)),
        "gemm_nn_rows_r16s" => return Some((GEMM_NN_ROWS_R16S, 3)),
        _ => {}
    }
    let (base, dt) = name.rsplit_once('_')?;
    // Pure data-movement kernels also exist as an i64 (uvec2) variant.
    let i64b: Option<&'static [u8]> = match base {
        "copy2d" => Some(COPY2D_I64),
        "copy_strided" => Some(COPY_STRIDED_I64),
        "transpose" => Some(TRANSPOSE_I64),
        "index_select" => Some(INDEX_SELECT_I64),
        "scatter_set" => Some(SCATTER_SET_I64),
        _ => None,
    };
    type Def = (&'static [u8], Option<&'static [u8]>, Option<&'static [u8]>, u32);
    // (f32 spirv, f16 spirv, bf16 spirv, binding count)
    let (f32b, f16b, bf16b, bindings): Def = match base {
        "fill" => (FILL_F32, Some(FILL_F16), Some(FILL_BF16), 1),
        "unary" => (UNARY_F32, Some(UNARY_F16), Some(UNARY_BF16), 2),
        "binary" => (BINARY_F32, Some(BINARY_F16), Some(BINARY_BF16), 3),
        "scale_add" => (SCALE_ADD_F32, Some(SCALE_ADD_F16), Some(SCALE_ADD_BF16), 2),
        "broadcast" => (BROADCAST_F32, Some(BROADCAST_F16), Some(BROADCAST_BF16), 4),
        "softmax" => (SOFTMAX_F32, Some(SOFTMAX_F16), Some(SOFTMAX_BF16), 2),
        "rmsnorm" => (RMSNORM_F32, Some(RMSNORM_F16), Some(RMSNORM_BF16), 3),
        "layernorm" => (LAYERNORM_F32, Some(LAYERNORM_F16), Some(LAYERNORM_BF16), 4),
        "rope" => (ROPE_F32, Some(ROPE_F16), Some(ROPE_BF16), 4),
        "rope_i" => (ROPE_I_F32, Some(ROPE_I_F16), Some(ROPE_I_BF16), 4),
        "reduce" => (REDUCE_F32, Some(REDUCE_F16), Some(REDUCE_BF16), 2),
        "reduce_arg" => (REDUCE_ARG_F32, Some(REDUCE_ARG_F16), Some(REDUCE_ARG_BF16), 2),
        "transpose" => (TRANSPOSE_F32, Some(TRANSPOSE_F16), Some(TRANSPOSE_BF16), 2),
        "copy2d" => (COPY2D_F32, Some(COPY2D_F16), Some(COPY2D_BF16), 2),
        "copy_strided" => (COPY_STRIDED_F32, Some(COPY_STRIDED_F16), Some(COPY_STRIDED_BF16), 3),
        "index_select" => (INDEX_SELECT_F32, Some(INDEX_SELECT_F16), Some(INDEX_SELECT_BF16), 3),
        "causality_mask" => {
            (CAUSALITY_MASK_F32, Some(CAUSALITY_MASK_F16), Some(CAUSALITY_MASK_BF16), 1)
        }
        "scatter_set" => (SCATTER_SET_F32, Some(SCATTER_SET_F16), Some(SCATTER_SET_BF16), 3),
        "gemm_tiled" => (GEMM_TILED_F32, Some(GEMM_TILED_F16), Some(GEMM_TILED_BF16), 3),
        "gemv" => (GEMV_F32, Some(GEMV_F16), Some(GEMV_BF16), 4),
        // conv shaders are f32-only; other dtypes must fail pipeline lookup.
        "conv1d" => (CONV1D_F32, None, None, 3),
        "conv_transpose1d" => (CONV_TRANSPOSE1D_F32, None, None, 3),
        "im2col1d" => (IM2COL1D_F32, None, None, 2),
        "col2im1d" => (COL2IM1D_F32, None, None, 2),
        _ => return None,
    };
    let bytes = match dt {
        "f16" => f16b?,
        "bf16" => bf16b?,
        "i64" => i64b?,
        _ => f32b,
    };
    Some((bytes, bindings))
}

const MAX_BINDINGS: usize = 4;

/// Picks the row-block variant of `prefix` for `rows` rows: the smallest power
/// of two that covers them, capped at `cap`, past which the grid walks the
/// rows in blocks of `cap`. Returns the kernel name and the block height.
pub(crate) fn row_block_kernel(prefix: &str, rows: usize, cap: usize) -> (String, u32) {
    let mr = rows.clamp(1, cap).next_power_of_two() as u32;
    (format!("{prefix}_r{mr}"), mr)
}
const PUSH_CONSTANT_SIZE: u32 = 128;
const WORKGROUP_SIZE: u32 = 256;

/// Little-endian push-constant byte builder.
#[derive(Default, Clone)]
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
    pipeline: vk::Pipeline,
    module: vk::ShaderModule,
    bindings: u32,
}

/// Command-recording resources, guarded by a mutex.
///
/// Dispatches are recorded into `command_buffer` and only submitted when the
/// batch is flushed (on host readback / `synchronize` / before host access to
/// mapped memory, or when the descriptor pool is about to overflow). This keeps
/// the GPU busy across many ops instead of paying a CPU↔GPU round-trip per op.
/// One unit of GPU work in flight: a command buffer with its fence,
/// descriptor pool and profiling queries, plus the buffers to recycle once it
/// has executed.
struct Batch {
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    descriptor_pool: vk::DescriptorPool,
    /// First timestamp query of this batch's slice of the query pool.
    query_base: u32,
    /// Submitted and not yet retired.
    in_flight: bool,
    /// Descriptor sets allocated while recording this batch.
    n_sets: u32,
    /// Buffers (dropped tensors + scratch) to recycle into the pool once this
    /// batch has finished executing: a buffer's last recorded use is in this
    /// batch or an earlier one, and the queue runs batches in order.
    free_bufs: Vec<PooledBuf>,
    /// Profiling: kernel name per recorded command.
    prof_names: Vec<String>,
    /// Profiling: timestamps written, including the batch's opening one.
    n_queries: u32,
    /// Buffers bound by the ops recorded since the last barrier. An op whose
    /// buffers are all new here cannot depend on those ops, so it needs no
    /// barrier; any overlap gets the conservative global one.
    touched: Vec<vk::Buffer>,
}

/// The recording state: a ring of batches. `cur` is the batch being recorded
/// (or the next to record into); the others may be in flight on the GPU.
///
/// Work is submitted every [`SUBMIT_AT`] dispatches without waiting, so the
/// GPU starts on a frame while the CPU is still recording the rest of it;
/// only a readback or host fallback waits for everything. Buffers a batch
/// released come back to the pool when that batch is retired, which happens
/// as soon as its fence is seen signalled at a later submit, or at a flush.
struct OpCtx {
    batches: Vec<Batch>,
    cur: usize,
    /// Whether `batches[cur]` has recorded, unsubmitted commands.
    open: bool,
}

/// Batches in the ring: how many can be in flight before recording blocks on
/// the oldest. A frame of decode is ~450 dispatches, so this is a frame.
const RING: usize = 8;
/// Dispatches recorded into a batch before it is submitted. Fewer means the
/// GPU starts sooner and more submits; each submit is a few microseconds of
/// CPU and a small gap on the GPU. Measured on Phonon's Mimi decoder (~180
/// dispatches a frame): 32 beat 16 and 64 by 3-5%.
const SUBMIT_AT: u32 = 32;
/// Dispatches before the first submit when nothing is in flight: the GPU is
/// idle, so getting it started matters more than amortizing the submit.
const FIRST_SUBMIT_AT: u32 = 8;
/// Descriptor sets a batch's pool holds: `SUBMIT_AT` dispatches plus slack for
/// the ops recorded after the threshold is crossed within one operator.
const MAX_SETS_PER_BATCH: u32 = SUBMIT_AT + 64;
/// Timestamps a batch may write when profiling: one per dispatch or copy,
/// plus the opening one. Copies do not allocate sets, so this is generous.
const BATCH_QUERIES: u32 = 2 * MAX_SETS_PER_BATCH + 8;

/// A buffer plus its memory and persistently-mapped pointer, as kept in the
/// recycling pool. The pointer is stored as `usize` so the struct stays
/// `Send`/`Sync` behind the pool mutex.
struct PooledBuf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: usize,
    class: u64,
}

/// Recycling pool for buffer allocations, keyed by size class.
/// `vkAllocateMemory` costs tens of microseconds and decoding allocates
/// hundreds of intermediate tensors per token, so freed buffers are returned
/// here (after the batch referencing them completes) and reused instead of
/// being destroyed.
#[derive(Default)]
struct BufferPool {
    free: HashMap<u64, Vec<PooledBuf>>,
    hits: u64,
    misses: u64,
}

/// Round a byte size up to its allocation class: the next power of two below
/// 1 MiB (256 B minimum), a 1/16 subdivision of the enclosing power of two
/// above it (max ~12.5% waste). Buffers are created with the class size so any
/// same-class request can reuse them.
fn size_class(bytes: usize) -> u64 {
    let bytes = bytes.max(4) as u64;
    let np2 = bytes.next_power_of_two();
    if np2 <= (1 << 20) { np2.max(256) } else { bytes.div_ceil(np2 / 16) * (np2 / 16) }
}
/// Timestamp query pool capacity (only used with `XN_VULKAN_PROFILE=1`).
const QUERY_CAP: u32 = RING as u32 * BATCH_QUERIES;

/// Accumulated profiling counters (enabled via `XN_VULKAN_PROFILE=1`).
/// GPU times come from timestamp queries written after every dispatch; the
/// batch serializes ops with global barriers, so consecutive timestamp deltas
/// are accurate per-op GPU durations.
#[derive(Default)]
struct ProfStats {
    /// kernel name -> (dispatch count, total gpu ns)
    per_kernel: HashMap<String, (u64, u128)>,
    gpu_ns: u128,
    dispatches: u64,
    flushes: u64,
    /// Batches submitted (each `SUBMIT_AT` dispatches or a flush).
    submits: u64,
    /// What triggered each flush (profiling only) — readbacks vs host
    /// fallbacks vs forced batch splits.
    flush_reasons: HashMap<&'static str, u64>,
    /// CPU time spent in submit + fence wait.
    wait_ns: u128,
}

pub struct DeviceInner {
    entry: ash::Entry,
    instance: ash::Instance,
    pdevice: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    command_pool: vk::CommandPool,
    // set_layouts[n] / pipeline_layouts[n] have `n` storage-buffer bindings.
    set_layouts: [vk::DescriptorSetLayout; MAX_BINDINGS + 1],
    pipeline_layouts: [vk::PipelineLayout; MAX_BINDINGS + 1],
    pipelines: Mutex<HashMap<String, CachedPipeline>>,
    supports_f16: bool,
    supports_bf16: bool,
    /// Subgroup size usable for `subgroupAdd` in compute shaders, 0 when
    /// subgroup arithmetic is unavailable (or disabled via env).
    subgroup_size: u32,
    /// `XN_VULKAN_GEMM`, a kernel name to force for every f32 GEMM it can
    /// serve (`tiled16`, `tiled32`, `tiled64`, `tiled`, `rowblock`, `generic`).
    /// For measuring; unset in normal use.
    gemm_force: Option<String>,
    /// Dispatches per batch before it is submitted (`SUBMIT_AT`, or
    /// `XN_VULKAN_SUBMIT_AT` for measuring).
    submit_at: u32,
    pool: Mutex<BufferPool>,
    /// Set when `XN_VULKAN_PROFILE=1` and the queue supports timestamps.
    profile_enabled: bool,
    /// Timestamp query pool (null unless profiling).
    query_pool: vk::QueryPool,
    /// Nanoseconds per timestamp tick.
    timestamp_period: f64,
    pstats: Mutex<ProfStats>,
    ctx: Mutex<OpCtx>,
    device_name: String,
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
        f.debug_struct("VulkanDevice").field("name", &self.device_name).finish()
    }
}

fn device_type_score(t: vk::PhysicalDeviceType) -> u32 {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 1,
        _ => 0,
    }
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| crate::Error::msg(format!("vulkan: failed to load loader: {e:?}")))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"xn")
            .api_version(vk::make_api_version(0, 1, 1, 0));
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(vkerr("create_instance"))?;

        let pdevices =
            unsafe { instance.enumerate_physical_devices() }.map_err(vkerr("enumerate_devices"))?;
        if pdevices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            crate::bail!("vulkan: no physical devices found (is a Vulkan driver installed?)");
        }

        // Rank devices by preference (discrete > integrated > cpu). `ordinal`
        // selects among the ranked list. An explicit `XN_VULKAN_DEVICE` env
        // var overrides the ordinal with a raw enumeration index.
        let mut ranked: Vec<(u32, usize, vk::PhysicalDevice)> = pdevices
            .iter()
            .enumerate()
            .map(|(i, &pd)| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                (device_type_score(props.device_type), i, pd)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        let pdevice = if let Ok(idx) =
            std::env::var("XN_VULKAN_DEVICE").unwrap_or_default().parse::<usize>()
        {
            pdevices.get(idx).copied().unwrap_or(ranked[0].2)
        } else {
            ranked.get(ordinal).map(|r| r.2).unwrap_or(ranked[0].2)
        };

        let props = unsafe { instance.get_physical_device_properties(pdevice) };
        let device_name =
            unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy().into_owned();

        // Pick a queue family that supports compute.
        let qfams = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
        let queue_family_index = qfams
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| crate::Error::msg("vulkan: no compute queue family"))?
            as u32;

        // Detect 16-bit support:
        //   f16 needs shaderFloat16 arithmetic + 16-bit SSBO storage;
        //   bf16 is emulated over uint16_t storage, needing shaderInt16 +
        //   16-bit SSBO storage (no extension: bf16 <-> f32 is bit shifting).
        let mut f16_int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f16_int8)
            .push_next(&mut storage16);
        unsafe { instance.get_physical_device_features2(pdevice, &mut features2) };
        // NB: read features2 before f16_int8/storage16 — it mutably borrows them.
        let shader_int16 = features2.features.shader_int16 != 0;
        let dev_exts =
            unsafe { instance.enumerate_device_extension_properties(pdevice) }.unwrap_or_default();
        let has_ext = |name: &CStr| {
            dev_exts.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == name)
        };
        let f16_ext_name = ash::khr::shader_float16_int8::NAME;
        let storage16_ok = storage16.storage_buffer16_bit_access != 0;
        let supports_f16 = f16_int8.shader_float16 != 0 && storage16_ok && has_ext(f16_ext_name);
        let supports_bf16 = shader_int16 && storage16_ok;

        // Subgroup arithmetic, for the reductions in the q8_0 GEMV. The
        // kernel packs `256 / subgroup_size` columns into a workgroup, so the
        // size has to divide the workgroup. `XN_VULKAN_SUBGROUP=0` forces the
        // shared-memory fallback, for comparison or for a driver that lies.
        let mut sg_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg_props);
        unsafe { instance.get_physical_device_properties2(pdevice, &mut props2) };
        let subgroup_forced_off = matches!(std::env::var("XN_VULKAN_SUBGROUP").as_deref(), Ok("0"));
        let subgroup_ok = sg_props.supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
            && sg_props.supported_operations.contains(vk::SubgroupFeatureFlags::ARITHMETIC)
            && sg_props.subgroup_size > 0
            && WORKGROUP_SIZE.is_multiple_of(sg_props.subgroup_size)
            && !subgroup_forced_off;
        let subgroup_size = if subgroup_ok { sg_props.subgroup_size } else { 0 };

        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);
        let queue_infos = [queue_info];

        // `VK_KHR_16bit_storage` is core in Vulkan 1.1 (enabled via the feature
        // struct); `shaderFloat16` still needs its extension string in 1.1.
        let ext_ptrs: Vec<*const std::ffi::c_char> =
            if supports_f16 { vec![f16_ext_name.as_ptr()] } else { vec![] };
        let core_features = vk::PhysicalDeviceFeatures::default().shader_int16(supports_bf16);
        let mut f16_enable =
            vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
        let mut s16_enable =
            vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true);
        let mut device_create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_features(&core_features)
            .enabled_extension_names(&ext_ptrs);
        if supports_f16 || supports_bf16 {
            // Both 16-bit dtypes need the 16-bit SSBO storage feature.
            device_create = device_create.push_next(&mut s16_enable);
        }
        if supports_f16 {
            device_create = device_create.push_next(&mut f16_enable);
        }
        let device = unsafe { instance.create_device(pdevice, &device_create, None) }
            .map_err(vkerr("create_device"))?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pdevice) };

        // Optional GPU profiling via timestamp queries (XN_VULKAN_PROFILE=1).
        let profile_requested =
            std::env::var("XN_VULKAN_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0");
        let profile_enabled =
            profile_requested && qfams[queue_family_index as usize].timestamp_valid_bits != 0;
        let timestamp_period = props.limits.timestamp_period as f64;
        let query_pool = if profile_enabled {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(QUERY_CAP);
            unsafe { device.create_query_pool(&info, None) }.map_err(vkerr("create_query_pool"))?
        } else {
            vk::QueryPool::null()
        };

        // Descriptor set + pipeline layouts for each supported binding count.
        let mut set_layouts = [vk::DescriptorSetLayout::null(); MAX_BINDINGS + 1];
        let mut pipeline_layouts = [vk::PipelineLayout::null(); MAX_BINDINGS + 1];
        for n in 1..=MAX_BINDINGS {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let set_layout = unsafe { device.create_descriptor_set_layout(&info, None) }
                .map_err(vkerr("create_descriptor_set_layout"))?;
            set_layouts[n] = set_layout;

            let pc_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(PUSH_CONSTANT_SIZE);
            let ranges = [pc_range];
            let sls = [set_layout];
            let info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&sls)
                .push_constant_ranges(&ranges);
            pipeline_layouts[n] = unsafe { device.create_pipeline_layout(&info, None) }
                .map_err(vkerr("create_pipeline_layout"))?;
        }

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(vkerr("create_command_pool"))?;

        let cb_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(RING as u32);
        let command_buffers = unsafe { device.allocate_command_buffers(&cb_info) }
            .map_err(vkerr("alloc_cmd_buffer"))?;
        let mut batches = Vec::with_capacity(RING);
        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(vkerr("create_fence"))?;
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(MAX_SETS_PER_BATCH * MAX_BINDINGS as u32)];
            let dp_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(MAX_SETS_PER_BATCH)
                .pool_sizes(&pool_sizes);
            let descriptor_pool = unsafe { device.create_descriptor_pool(&dp_info, None) }
                .map_err(vkerr("create_desc_pool"))?;
            batches.push(Batch {
                command_buffer,
                fence,
                descriptor_pool,
                query_base: i as u32 * BATCH_QUERIES,
                in_flight: false,
                n_sets: 0,
                free_bufs: Vec::new(),
                prof_names: Vec::new(),
                n_queries: 0,
                touched: Vec::new(),
            });
        }

        let inner = DeviceInner {
            entry,
            instance,
            pdevice,
            device,
            queue,
            queue_family_index,
            mem_props,
            command_pool,
            set_layouts,
            pipeline_layouts,
            pipelines: Mutex::new(HashMap::new()),
            supports_f16,
            supports_bf16,
            subgroup_size,
            gemm_force: std::env::var("XN_VULKAN_GEMM").ok().filter(|v| !v.is_empty()),
            submit_at: std::env::var("XN_VULKAN_SUBMIT_AT")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .map(|v| v.clamp(1, MAX_SETS_PER_BATCH - 8))
                .unwrap_or(SUBMIT_AT),
            pool: Mutex::new(BufferPool::default()),
            profile_enabled,
            query_pool,
            timestamp_period,
            pstats: Mutex::new(ProfStats::default()),
            ctx: Mutex::new(OpCtx { batches, cur: 0, open: false }),
            device_name,
        };
        let _ = inner.pdevice;
        let _ = inner.queue_family_index;
        let _ = &inner.entry;
        Ok(Self(Arc::new(inner)))
    }

    /// Whether this device supports f16 compute + storage (weights/activations
    /// can be stored as `half::f16`).
    pub fn supports_f16(&self) -> bool {
        self.supports_f16
    }

    /// Whether this device supports bf16 storage (emulated over uint16_t
    /// buffers with f32 compute; needs shaderInt16 + 16-bit SSBO storage).
    pub fn supports_bf16(&self) -> bool {
        self.supports_bf16
    }

    /// The device's compute subgroup size when subgroup arithmetic can be used,
    /// else 0.
    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }

    /// Find a memory type index within `type_bits` that has all of `flags`.
    fn find_memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            (type_bits & (1 << i)) != 0
                && self.mem_props.memory_types[i as usize].property_flags.contains(flags)
        })
    }

    /// Allocate a buffer of at least `size_bytes`, reusing a pooled buffer of
    /// the same size class when one is available. Returns the buffer, its
    /// memory, the persistently-mapped pointer, and the size class (needed to
    /// return the buffer to the pool on free).
    fn alloc_buffer(
        &self,
        size_bytes: usize,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8, u64)> {
        let class = size_class(size_bytes);
        {
            let mut pool = self.pool.lock().unwrap();
            if let Some(b) = pool.free.get_mut(&class).and_then(|v| v.pop()) {
                pool.hits += 1;
                return Ok((b.buffer, b.memory, b.ptr as *mut u8, class));
            }
            pool.misses += 1;
        }
        // Buffers are created with the full class size so that any same-class
        // request can reuse them.
        let size = class;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer =
            unsafe { self.device.create_buffer(&info, None) }.map_err(vkerr("create_buffer"))?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        // Prefer the unified APU type (device-local + host-visible + coherent),
        // fall back to any host-visible coherent type.
        let unified = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mem_type = self
            .find_memory_type(req.memory_type_bits, unified)
            .or_else(|| self.find_memory_type(req.memory_type_bits, host));
        let mem_type = match mem_type {
            Some(m) => m,
            None => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                crate::bail!("vulkan: no host-visible coherent memory type available");
            }
        };

        let alloc =
            vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mem_type);
        let memory = unsafe { self.device.allocate_memory(&alloc, None) }
            .map_err(vkerr("allocate_memory"))?;
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(vkerr("bind_buffer_memory"))?;
        let ptr = unsafe {
            self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(vkerr("map_memory"))? as *mut u8;
        Ok((buffer, memory, ptr, class))
    }

    fn get_pipeline(&self, name: &str) -> Result<(vk::Pipeline, vk::PipelineLayout, u32)> {
        let mut pipelines = self.pipelines.lock().unwrap();
        if let Some(p) = pipelines.get(name) {
            return Ok((p.pipeline, self.pipeline_layouts[p.bindings as usize], p.bindings));
        }
        let (spirv, bindings) = kernel_def(name)
            .ok_or_else(|| crate::Error::msg(format!("vulkan: unknown kernel {name}")))?;
        // SPIR-V is a little-endian stream of u32 words.
        let code: Vec<u32> =
            spirv.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let module_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let module = unsafe { self.device.create_shader_module(&module_info, None) }
            .map_err(vkerr("create_shader_module"))?;
        let entry = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layouts[bindings as usize]);
        let pipeline = unsafe {
            self.device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, e)| vkerr::<vk::Result>("create_compute_pipeline")(e))?[0];
        pipelines.insert(name.to_string(), CachedPipeline { pipeline, module, bindings });
        Ok((pipeline, self.pipeline_layouts[bindings as usize], bindings))
    }

    /// Record a single dispatch of `kernel` (1D workgroup count).
    fn dispatch(
        &self,
        kernel: &str,
        buffers: &[vk::Buffer],
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
        buffers: &[vk::Buffer],
        push: &Pc,
        groups: (u32, u32, u32),
    ) -> Result<()> {
        self.dispatch_labeled(kernel, None, buffers, push, groups)
    }

    /// [`Self::dispatch_nd`] with a profiler label in place of the kernel
    /// name, so the profile can split a kernel by problem shape. Callers pass
    /// `None` unless `profile_enabled` is set; the label is otherwise unused.
    fn dispatch_labeled(
        &self,
        kernel: &str,
        label: Option<String>,
        buffers: &[vk::Buffer],
        push: &Pc,
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let (gx, gy, gz) = groups;
        if gx == 0 || gy == 0 || gz == 0 {
            return Ok(());
        }
        let (pipeline, layout, bindings) = self.get_pipeline(kernel)?;
        assert_eq!(bindings as usize, buffers.len(), "kernel {kernel} binding count mismatch");
        let mut ctx = self.ctx.lock().unwrap();
        self.begin_if_needed(&mut ctx, buffers)?;
        let dev = &self.device;
        let cur = ctx.cur;
        let batch = &mut ctx.batches[cur];
        unsafe {
            let set_layouts = [self.set_layouts[bindings as usize]];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(batch.descriptor_pool)
                .set_layouts(&set_layouts);
            let set =
                dev.allocate_descriptor_sets(&alloc_info).map_err(vkerr("alloc_desc_set"))?[0];
            batch.n_sets += 1;

            let infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|&b| {
                    vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(vk::WHOLE_SIZE)
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = (0..buffers.len())
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[i]))
                })
                .collect();
            dev.update_descriptor_sets(&writes, &[]);

            let cb = batch.command_buffer;
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            dev.cmd_push_constants(cb, layout, vk::ShaderStageFlags::COMPUTE, 0, &push.bytes);
            dev.cmd_dispatch(cb, gx, gy, gz);
            if self.profile_enabled {
                dev.cmd_write_timestamp(
                    cb,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    batch.query_base + batch.n_queries,
                );
                batch.prof_names.push(label.unwrap_or_else(|| kernel.to_string()));
                batch.n_queries += 1;
            }
        }
        self.submit_if_full(&mut ctx)
    }

    /// Record a buffer-to-buffer copy of `bytes` into the current batch.
    fn record_copy(&self, dst: vk::Buffer, src: vk::Buffer, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut ctx = self.ctx.lock().unwrap();
        self.begin_if_needed(&mut ctx, &[dst, src])?;
        let cur = ctx.cur;
        let batch = &mut ctx.batches[cur];
        unsafe {
            let region = vk::BufferCopy::default().src_offset(0).dst_offset(0).size(bytes as u64);
            self.device.cmd_copy_buffer(batch.command_buffer, src, dst, &[region]);
            if self.profile_enabled {
                self.device.cmd_write_timestamp(
                    batch.command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    batch.query_base + batch.n_queries,
                );
                batch.prof_names.push("buffer_copy".to_string());
                batch.n_queries += 1;
            }
        }
        self.submit_if_full(&mut ctx)
    }

    /// Makes `ctx.cur` ready to record into, opening its command buffer if
    /// nothing is recorded yet (waiting for the batch previously submitted
    /// from that slot, if it is still in flight), and inserts the global
    /// memory barrier an op needs to observe the writes of everything before
    /// it -- unless nothing recorded since the last barrier touches the
    /// buffers this op binds, in which case it depends on none of it.
    fn begin_if_needed(&self, ctx: &mut OpCtx, buffers: &[vk::Buffer]) -> Result<()> {
        let dev = &self.device;
        let mut fresh = false;
        if !ctx.open {
            fresh = true;
            let cur = ctx.cur;
            if ctx.batches[cur].in_flight {
                self.wait_and_retire(&mut ctx.batches[cur])?;
            }
            let batch = &mut ctx.batches[cur];
            unsafe {
                dev.reset_command_buffer(
                    batch.command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(vkerr("reset_command_buffer"))?;
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                dev.begin_command_buffer(batch.command_buffer, &begin)
                    .map_err(vkerr("begin_command_buffer"))?;
                if self.profile_enabled {
                    dev.cmd_reset_query_pool(
                        batch.command_buffer,
                        self.query_pool,
                        batch.query_base,
                        BATCH_QUERIES,
                    );
                    dev.cmd_write_timestamp(
                        batch.command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        self.query_pool,
                        batch.query_base,
                    );
                    batch.n_queries = 1;
                }
            }
            ctx.open = true;
        }
        // A fresh batch runs behind everything submitted before it; within a
        // batch, an op only needs the barrier if it shares a buffer with an
        // op recorded since the last one.
        let cur = ctx.cur;
        let touched = &mut ctx.batches[cur].touched;
        let overlaps = touched.iter().any(|t| buffers.contains(t));
        if !fresh && !overlaps {
            touched.extend_from_slice(buffers);
            return Ok(());
        }
        touched.clear();
        touched.extend_from_slice(buffers);
        let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
        unsafe {
            dev.cmd_pipeline_barrier(
                ctx.batches[ctx.cur].command_buffer,
                stages,
                stages,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Submits the current batch once it holds `SUBMIT_AT` dispatches, or is
    /// about to run out of descriptor sets or profiling queries.
    fn submit_if_full(&self, ctx: &mut OpCtx) -> Result<()> {
        let idle = !ctx.batches.iter().any(|b| b.in_flight);
        let threshold = if idle { FIRST_SUBMIT_AT.min(self.submit_at) } else { self.submit_at };
        let batch = &ctx.batches[ctx.cur];
        if ctx.open
            && (batch.n_sets >= threshold
                || batch.n_sets + 8 >= MAX_SETS_PER_BATCH
                || batch.n_queries + 2 >= BATCH_QUERIES)
        {
            self.submit_locked(ctx)?;
        }
        Ok(())
    }

    /// Submits the current batch without waiting and moves recording on to
    /// the next slot. Batches whose fences have already signalled are
    /// retired on the way, so their buffers return to the pool promptly.
    fn submit_locked(&self, ctx: &mut OpCtx) -> Result<()> {
        if !ctx.open {
            return Ok(());
        }
        let dev = &self.device;
        let t0 = if self.profile_enabled { Some(std::time::Instant::now()) } else { None };
        let cur = ctx.cur;
        {
            let batch = &mut ctx.batches[cur];
            unsafe {
                dev.end_command_buffer(batch.command_buffer)
                    .map_err(vkerr("end_command_buffer"))?;
                let cbs = [batch.command_buffer];
                let submit = vk::SubmitInfo::default().command_buffers(&cbs);
                dev.queue_submit(self.queue, &[submit], batch.fence)
                    .map_err(vkerr("queue_submit"))?;
            }
            batch.in_flight = true;
        }
        ctx.open = false;
        ctx.cur = (cur + 1) % RING;
        for i in 0..RING {
            let b = &mut ctx.batches[i];
            if b.in_flight
                && unsafe { dev.get_fence_status(b.fence) }.map_err(vkerr("fence_status"))?
            {
                self.retire(b)?;
            }
        }
        if let Some(t0) = t0 {
            let mut stats = self.pstats.lock().unwrap();
            stats.submits += 1;
            stats.wait_ns += t0.elapsed().as_nanos();
        }
        Ok(())
    }

    /// Blocks until `batch` has executed, then retires it.
    fn wait_and_retire(&self, batch: &mut Batch) -> Result<()> {
        let t0 = if self.profile_enabled { Some(std::time::Instant::now()) } else { None };
        unsafe {
            self.device
                .wait_for_fences(&[batch.fence], true, u64::MAX)
                .map_err(vkerr("wait_for_fences"))?;
        }
        if let Some(t0) = t0 {
            self.pstats.lock().unwrap().wait_ns += t0.elapsed().as_nanos();
        }
        self.retire(batch)
    }

    /// Reclaims a batch whose fence has signalled: its fence and descriptor
    /// pool are reset, the buffers it released go back to the pool, and its
    /// profiling timestamps are read out.
    fn retire(&self, batch: &mut Batch) -> Result<()> {
        let dev = &self.device;
        unsafe {
            dev.reset_fences(&[batch.fence]).map_err(vkerr("reset_fences"))?;
            dev.reset_descriptor_pool(batch.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(vkerr("reset_descriptor_pool"))?;
        }
        if !batch.free_bufs.is_empty() {
            let mut pool = self.pool.lock().unwrap();
            for b in batch.free_bufs.drain(..) {
                pool.free.entry(b.class).or_default().push(b);
            }
        }
        if self.profile_enabled {
            let nq = batch.n_queries as usize;
            if nq >= 2 {
                let mut ts = vec![0u64; nq];
                unsafe {
                    dev.get_query_pool_results::<u64>(
                        self.query_pool,
                        batch.query_base,
                        &mut ts,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .map_err(vkerr("get_query_results"))?;
                }
                let mut stats = self.pstats.lock().unwrap();
                for (i, name) in batch.prof_names.iter().enumerate() {
                    let dt_ns =
                        (ts[i + 1].saturating_sub(ts[i]) as f64 * self.timestamp_period) as u128;
                    let e = stats.per_kernel.entry(name.clone()).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += dt_ns;
                    stats.gpu_ns += dt_ns;
                    stats.dispatches += 1;
                }
            }
            batch.prof_names.clear();
            batch.n_queries = 0;
        }
        batch.n_sets = 0;
        batch.touched.clear();
        batch.in_flight = false;
        Ok(())
    }

    /// Submit any pending recorded commands and wait for completion. Safe to
    /// call when nothing is pending. `reason` attributes the flush in the
    /// profiling report (only recorded when the flush actually submits work).
    fn flush(&self, reason: &'static str) -> Result<()> {
        let mut ctx = self.ctx.lock().unwrap();
        if self.profile_enabled && (ctx.open || ctx.batches.iter().any(|b| b.in_flight)) {
            *self.pstats.lock().unwrap().flush_reasons.entry(reason).or_insert(0) += 1;
        }
        self.flush_locked(&mut ctx)
    }

    /// Submits whatever is recorded and waits for every batch in flight, so
    /// the host may read or write mapped memory afterwards.
    fn flush_locked(&self, ctx: &mut OpCtx) -> Result<()> {
        let had_work = ctx.open || ctx.batches.iter().any(|b| b.in_flight);
        self.submit_locked(ctx)?;
        // Oldest first: the slot after `cur` was submitted longest ago.
        for i in 1..=RING {
            let idx = (ctx.cur + i) % RING;
            if ctx.batches[idx].in_flight {
                self.wait_and_retire(&mut ctx.batches[idx])?;
            }
        }
        // Buffers dropped while nothing was recording sit in the current
        // slot's list; the GPU is idle now, so they can go back too.
        let cur = ctx.cur;
        let batch = &mut ctx.batches[cur];
        if !batch.free_bufs.is_empty() {
            let mut pool = self.pool.lock().unwrap();
            for b in batch.free_bufs.drain(..) {
                pool.free.entry(b.class).or_default().push(b);
            }
        }
        if had_work && self.profile_enabled {
            self.pstats.lock().unwrap().flushes += 1;
        }
        Ok(())
    }
}

impl DeviceInner {
    /// Print accumulated per-kernel GPU times (profiling mode only).
    fn print_profile(&self) {
        let stats = self.pstats.lock().unwrap();
        if stats.dispatches == 0 {
            return;
        }
        let mut rows: Vec<_> = stats.per_kernel.iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1.1));
        eprintln!("\n=== xn vulkan profile: {} ===", self.device_name);
        let width = rows.iter().map(|r| r.0.len()).max().unwrap_or(22).max(22);
        eprintln!(
            "{:<width$} {:>9} {:>11} {:>9} {:>7}",
            "kernel", "count", "total ms", "avg us", "%gpu"
        );
        for (name, (cnt, ns)) in rows {
            eprintln!(
                "{:<width$} {:>9} {:>11.2} {:>9.1} {:>6.1}%",
                name,
                cnt,
                *ns as f64 / 1e6,
                *ns as f64 / 1e3 / *cnt as f64,
                100.0 * *ns as f64 / stats.gpu_ns as f64,
            );
        }
        eprintln!(
            "gpu total: {:.2} ms over {} dispatches in {} submits, {} flushes; cpu submit+wait: {:.2} ms",
            stats.gpu_ns as f64 / 1e6,
            stats.dispatches,
            stats.submits,
            stats.flushes,
            stats.wait_ns as f64 / 1e6,
        );
        if !stats.flush_reasons.is_empty() {
            let mut reasons: Vec<_> = stats.flush_reasons.iter().collect();
            reasons.sort_by_key(|r| std::cmp::Reverse(*r.1));
            let s: Vec<String> =
                reasons.iter().map(|(name, count)| format!("{name}: {count}")).collect();
            eprintln!("flush reasons: {}", s.join(", "));
        }
        let pool = self.pool.lock().unwrap();
        let total = pool.hits + pool.misses;
        if total > 0 {
            eprintln!(
                "buffer pool: {} hits / {} allocs ({:.1}% reuse)",
                pool.hits,
                total,
                100.0 * pool.hits as f64 / total as f64,
            );
        }
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        if self.profile_enabled {
            self.print_profile();
        }
        unsafe {
            let _ = self.device.device_wait_idle();
            if self.query_pool != vk::QueryPool::null() {
                self.device.destroy_query_pool(self.query_pool, None);
            }
            let mut pool = self.pool.lock().unwrap();
            for (_, bufs) in pool.free.drain() {
                for b in bufs {
                    self.device.destroy_buffer(b.buffer, None);
                    self.device.free_memory(b.memory, None);
                }
            }
            drop(pool);
            let pipelines = self.pipelines.lock().unwrap();
            for p in pipelines.values() {
                self.device.destroy_pipeline(p.pipeline, None);
                self.device.destroy_shader_module(p.module, None);
            }
            drop(pipelines);
            let mut ctx = self.ctx.lock().unwrap();
            for batch in ctx.batches.iter_mut() {
                for b in batch.free_bufs.drain(..) {
                    self.device.destroy_buffer(b.buffer, None);
                    self.device.free_memory(b.memory, None);
                }
                self.device.destroy_descriptor_pool(batch.descriptor_pool, None);
                self.device.destroy_fence(batch.fence, None);
            }
            drop(ctx);
            self.device.destroy_command_pool(self.command_pool, None);
            for n in 1..=MAX_BINDINGS {
                self.device.destroy_pipeline_layout(self.pipeline_layouts[n], None);
                self.device.destroy_descriptor_set_layout(self.set_layouts[n], None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Vulkan tensor storage: a persistently-mapped, host-coherent buffer.
pub struct Storage<T: WithDType> {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    len: usize,
    /// Allocation size class; used to return the buffer to the pool on drop.
    class: u64,
    device: Device,
    _t: PhantomData<T>,
}

// The mapped pointer is only accessed while holding a `&`/`&mut` to the
// storage; the device serializes GPU work. Safe to move across threads.
unsafe impl<T: WithDType> Send for Storage<T> {}
unsafe impl<T: WithDType> Sync for Storage<T> {}

impl<T: WithDType> Storage<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Host view of the mapped memory as `&[T]`.
    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) }
    }
}

impl<T: WithDType> Drop for Storage<T> {
    fn drop(&mut self) {
        // The current (unsubmitted) batch may still reference this buffer, so
        // defer recycling it until the next flush completes on the GPU.
        self.device.defer_free(PooledBuf {
            buffer: self.buffer,
            memory: self.memory,
            ptr: self.ptr as usize,
            class: self.class,
        });
    }
}

impl Device {
    /// Allocate a host-visible buffer holding `data` (e.g. `info` dims/strides
    /// arrays, or host-generated random values). Writing it host-side is safe
    /// without a flush because a freshly allocated buffer is never referenced
    /// by the pending batch (pool entries are only recycled after their batch
    /// completes). The caller must pass the returned `PooledBuf` to
    /// [`Self::defer_free`] *after* recording the command that uses it (see
    /// the `defer_free` invariant).
    fn scratch_from_slice<T: Copy>(&self, data: &[T]) -> Result<PooledBuf> {
        let bytes = std::mem::size_of_val(data);
        let (buffer, memory, ptr, class) = self.alloc_buffer(bytes)?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr, bytes);
        }
        Ok(PooledBuf { buffer, memory, ptr: ptr as usize, class })
    }

    /// Schedule a buffer to be recycled into the pool on the next flush.
    ///
    /// Invariant: this must only be called once *every* use of the buffer has
    /// been recorded into the command buffer. Recycling happens when a batch is
    /// flushed, and a flush may be forced in the middle of an op sequence (see
    /// `dispatch_nd`); a buffer deferred before its use is recorded would be
    /// recycled by such a flush and overwritten while the subsequently-recorded
    /// dispatch still references it.
    fn defer_free(&self, buf: PooledBuf) {
        let mut ctx = self.ctx.lock().unwrap();
        let cur = ctx.cur;
        ctx.batches[cur].free_bufs.push(buf);
    }
}

fn check_f32<T: WithDType>(op: &str) -> Result<()> {
    if T::DTYPE != DType::F32 {
        crate::bail!("vulkan: {op} only supports f32, got {:?}", T::DTYPE);
    }
    Ok(())
}

/// Shader dtype suffix ("f32"/"f16"/"bf16") for a float storage type; errors
/// on other dtypes or when the device lacks the required 16-bit support.
fn dtype_suffix<T: WithDType>(dev: &Device, op: &str) -> Result<&'static str> {
    match T::DTYPE {
        DType::F32 => Ok("f32"),
        DType::F16 if dev.supports_f16 => Ok("f16"),
        DType::F16 => crate::bail!("vulkan: {op}: device does not support f16"),
        DType::BF16 if dev.supports_bf16 => Ok("bf16"),
        DType::BF16 => crate::bail!("vulkan: {op}: device does not support bf16"),
        d => crate::bail!("vulkan: {op} supports f32/f16/bf16, got {d:?}"),
    }
}

/// Suffix for the GPU path of dtype-generic ops, or `None` to take the host
/// fallback (non-float dtypes, or 16-bit floats without device support).
fn float_suffix<T: WithDType>(dev: &Device) -> Option<&'static str> {
    match T::DTYPE {
        DType::F32 => Some("f32"),
        DType::F16 if dev.supports_f16 => Some("f16"),
        DType::BF16 if dev.supports_bf16 => Some("bf16"),
        _ => None,
    }
}

/// Suffix for pure data-movement ops (copy2d/copy_strided/transpose/
/// index_select/scatter_set), which additionally have an i64 (uvec2) shader
/// variant so kv-cache indices and token ids stay on the GPU path.
fn movement_suffix<T: WithDType>(dev: &Device) -> Option<&'static str> {
    match T::DTYPE {
        DType::I64 => Some("i64"),
        _ => float_suffix::<T>(dev),
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
