#![allow(clippy::too_many_arguments)]
mod cublaslt;
mod kernels;
pub mod q8;
pub mod quantization;

use crate::{BinaryOp, DType, Result, UnaryOp, WithDType, WithDTypeF};
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use std::sync::{Arc, Mutex};

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PTXModule {
    Arithmetic,
    Broadcast,
    Conv,
    Fattn,
    Fill,
    Fp8,
    Indexing,
    Layout,
    Q8,
    Reduce,
    Rope,
}

#[derive(Default)]
struct ModuleCache {
    arithmetic: Option<Arc<cudarc::driver::CudaModule>>,
    broadcast: Option<Arc<cudarc::driver::CudaModule>>,
    conv: Option<Arc<cudarc::driver::CudaModule>>,
    fattn: Option<Arc<cudarc::driver::CudaModule>>,
    fill: Option<Arc<cudarc::driver::CudaModule>>,
    fp8: Option<Arc<cudarc::driver::CudaModule>>,
    indexing: Option<Arc<cudarc::driver::CudaModule>>,
    layout: Option<Arc<cudarc::driver::CudaModule>>,
    q8: Option<Arc<cudarc::driver::CudaModule>>,
    reduce: Option<Arc<cudarc::driver::CudaModule>>,
    rope: Option<Arc<cudarc::driver::CudaModule>>,
}

/// Host-to-device upload mode, see [`Device::capture_record_begin`].
mod htod_mode {
    pub const PASSTHROUGH: u8 = 0;
    pub const RECORD: u8 = 1;
    pub const REPLAY: u8 = 2;
}

/// Device buffers holding recorded host uploads, keyed by element type name
/// and content bytes. Values are `CudaSlice<T>` boxed as `Any`.
type HtodCache =
    Mutex<std::collections::HashMap<(&'static str, Vec<u8>), Box<dyn std::any::Any + Send>>>;

pub struct DeviceInner {
    cuda: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: cudarc::cublas::CudaBlas,
    pub(crate) blas_lt: cublaslt::CudaBlasLT,
    curand: Mutex<CudaRng>,
    /// Cache for loaded PTX modules
    modules: Mutex<ModuleCache>,
    /// Host-to-device upload mode used by CUDA-graph capture, one of the
    /// [`htod_mode`] constants.
    htod_mode: std::sync::atomic::AtomicU8,
    /// Device-side copies of host uploads recorded in [`htod_mode::RECORD`].
    /// In [`htod_mode::REPLAY`] (i.e. while capturing) uploads are served
    /// from these buffers through device-to-device copies, which are legal
    /// inside a capture while a pageable host copy is not.
    htod_cache: HtodCache,
}

pub struct CudaEvent {
    event: cudarc::driver::CudaEvent,
    stream: Arc<CudaStream>,
}

impl CudaEvent {
    pub fn record(&self) -> Result<()> {
        self.event.record(&self.stream)?;
        Ok(())
    }

    pub fn is_complete(&self) -> bool {
        self.event.is_complete()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.event.synchronize()?;
        Ok(())
    }

    pub fn elapsed_ms(&self, other: &CudaEvent) -> Result<f32> {
        let ms = self.event.elapsed_ms(&other.event)?;
        Ok(ms)
    }
}

/// An instantiated CUDA graph, see [`Device::capture_begin`]. Replaying it
/// re-runs the captured kernels against the buffer addresses they were
/// captured with: refresh the input buffers in place, [`CaptureGraph::launch`],
/// then read the output buffers.
pub struct CaptureGraph(cudarc::driver::CudaGraph);

// SAFETY: `CudaGraph` is not auto-Send because it holds raw
// `CUgraph`/`CUgraphExec` handles, but those are plain context-scoped driver
// handles with no thread affinity; cudarc re-binds the CUDA context on every
// call. `CaptureGraph` is deliberately not `Sync`: `launch` must not be
// called concurrently from two threads.
unsafe impl Send for CaptureGraph {}

impl CaptureGraph {
    /// Enqueue a replay of the captured work on the stream it was captured
    /// from. Like any kernel launch this is asynchronous.
    pub fn launch(&self) -> Result<()> {
        self.0.launch()?;
        Ok(())
    }
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
        f.debug_struct("Device").field("ordinal", &self.cuda.ordinal()).finish()
    }
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let cuda = cudarc::driver::CudaContext::new(ordinal)?;
        // A proper (non-legacy) stream: the NULL default stream cannot be
        // captured into a CUDA graph. All work goes through this one stream,
        // so the synchronization semantics are unchanged.
        let stream = cuda.new_stream()?;
        // Creating a stream flips cudarc into multi-stream mode, which turns
        // on per-slice event waits; on our single stream they are pure
        // overhead, and events recorded before a CUDA-graph capture would
        // make the capture fail with CUDA_ERROR_STREAM_CAPTURE_ISOLATION.
        // SAFETY: all work runs on the single stream created above, so
        // stream-order already synchronizes every use.
        unsafe { cuda.disable_event_tracking() };
        let blas = cudarc::cublas::CudaBlas::new(stream.clone())?;
        let blas_lt = cublaslt::CudaBlasLT::new(stream.clone())?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone())?;
        Ok(Self(Arc::new(DeviceInner {
            cuda,
            stream,
            blas,
            blas_lt,
            modules: Mutex::new(Default::default()),
            curand: Mutex::new(CudaRng(curand)),
            htod_mode: std::sync::atomic::AtomicU8::new(0),
            htod_cache: Mutex::new(std::collections::HashMap::new()),
        })))
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Host-to-device upload that stays legal inside a CUDA-graph capture.
    ///
    /// In passthrough mode this is a plain [`CudaStream::clone_htod`]. In
    /// record mode (between [`Device::capture_record_begin`] and
    /// [`Device::capture_record_end`]) every upload also stores a persistent
    /// device-side copy keyed by its content. In replay mode (between
    /// [`Device::capture_begin`] and [`Device::capture_end`], i.e. while the
    /// stream is capturing) the upload is served without touching host
    /// memory: an in-capture allocation filled by a device-to-device copy
    /// from the recorded buffer; content that was not recorded beforehand is
    /// an error. This is only correct for uploads whose content is identical
    /// between the record pass and every replay of the captured graph
    /// (kernel shape/stride descriptors, position tables, masks, ...).
    pub(crate) fn cached_htod<T: cudarc::driver::DeviceRepr + Copy + Send + 'static>(
        &self,
        v: &[T],
    ) -> Result<CudaSlice<T>> {
        use std::sync::atomic::Ordering;
        let mode = self.htod_mode.load(Ordering::Relaxed);
        if mode == htod_mode::PASSTHROUGH {
            return Ok(self.stream.clone_htod(v)?);
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
        };
        let key = (std::any::type_name::<T>(), bytes.to_vec());
        let mut cache = self.htod_cache.lock().unwrap();
        if mode == htod_mode::RECORD && !cache.contains_key(&key) {
            let staged = self.stream.clone_htod(v)?;
            cache.insert(key.clone(), Box::new(staged));
        }
        let staged = match cache.get(&key) {
            None => crate::bail!(
                "htod upload of unrecorded content during CUDA graph capture ({} x{})",
                std::any::type_name::<T>(),
                v.len()
            ),
            Some(staged) => match staged.downcast_ref::<CudaSlice<T>>() {
                None => crate::bail!("htod cache type confusion"),
                Some(staged) => staged,
            },
        };
        // Both record and replay take this path so that the record pass
        // exercises exactly what the graph will capture.
        let mut out = unsafe { self.stream.alloc::<T>(v.len()) }?;
        self.stream.memcpy_dtod(staged, &mut out)?;
        Ok(out)
    }

    fn set_htod_mode(&self, from: u8, to: u8, op: &str) -> Result<()> {
        use std::sync::atomic::Ordering;
        match self.htod_mode.compare_exchange(from, to, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => Ok(()),
            Err(cur) => {
                crate::bail!("{op}: unexpected htod mode {cur} (record/capture calls unbalanced?)")
            }
        }
    }

    /// Start recording host uploads in the device-side cache, see
    /// [`Device::cached_htod`]. Run the exact workload to be captured once
    /// under record mode before capturing it with [`Device::capture_begin`];
    /// the recorded work executes normally.
    pub fn capture_record_begin(&self) -> Result<()> {
        self.set_htod_mode(htod_mode::PASSTHROUGH, htod_mode::RECORD, "capture_record_begin")
    }

    /// Stop recording host uploads.
    pub fn capture_record_end(&self) -> Result<()> {
        self.set_htod_mode(htod_mode::RECORD, htod_mode::PASSTHROUGH, "capture_record_end")
    }

    /// Begin capturing the device stream into a CUDA graph. The captured work
    /// is recorded, not executed. Requires a prior record pass over the same
    /// workload so that host uploads can be served from device buffers (see
    /// [`Device::cached_htod`]); an upload of unrecorded content fails, in
    /// which case the capture should be discarded with
    /// [`Device::capture_abort`].
    ///
    /// The usual pattern, with all graph inputs and outputs living in
    /// long-lived tensors that are refreshed in place:
    /// 1. `capture_record_begin()`; run the workload; `capture_record_end()`.
    /// 2. `capture_begin()`; run the workload again; `graph = capture_end()`.
    /// 3. Per step: refresh the inputs, `graph.launch()`, read the outputs.
    ///
    /// Tensors created before the capture must not be dropped inside it (a
    /// free of non-captured memory is invalid during capture), and tensors
    /// created inside must not escape it.
    pub fn capture_begin(&self) -> Result<()> {
        self.set_htod_mode(htod_mode::PASSTHROUGH, htod_mode::REPLAY, "capture_begin")?;
        let mode = cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;
        if let Err(err) = self.stream.begin_capture(mode) {
            self.htod_mode.store(htod_mode::PASSTHROUGH, std::sync::atomic::Ordering::Relaxed);
            Err(err)?
        }
        Ok(())
    }

    /// End the capture and instantiate the graph.
    pub fn capture_end(&self) -> Result<CaptureGraph> {
        self.htod_mode.store(htod_mode::PASSTHROUGH, std::sync::atomic::Ordering::Relaxed);
        let flags =
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        match self.stream.end_capture(flags)? {
            None => crate::bail!("CUDA graph capture ended without a graph"),
            Some(graph) => Ok(CaptureGraph(graph)),
        }
    }

    /// Abort an in-progress capture, discarding the partially captured graph.
    /// Errors from the discarded capture are ignored.
    pub fn capture_abort(&self) {
        self.htod_mode.store(htod_mode::PASSTHROUGH, std::sync::atomic::Ordering::Relaxed);
        let flags =
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let _ = self.stream.end_capture(flags);
    }

    /// Number of distinct host uploads recorded for capture replay.
    pub fn capture_cache_len(&self) -> usize {
        self.htod_cache.lock().unwrap().len()
    }

    /// Drop every recorded host upload. The device-to-device copies baked
    /// into a captured graph read from these buffers on every replay, so this
    /// must only be called once no [`CaptureGraph`] captured from this cache
    /// will be launched again.
    pub fn capture_cache_clear(&self) {
        self.htod_cache.lock().unwrap().clear();
    }

    pub fn event(&self) -> Result<CudaEvent> {
        let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
        let event = self.cuda.new_event(Some(flags))?;
        let event = CudaEvent { event, stream: self.stream.clone() };
        Ok(event)
    }

    fn get_or_load_module(&self, ptx: PTXModule) -> Result<Arc<cudarc::driver::CudaModule>> {
        let mut modules = self.modules.lock().unwrap();
        match ptx {
            PTXModule::Arithmetic => {
                if let Some(ref m) = modules.arithmetic {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(kernels::ARITHMETIC.into())?;
                modules.arithmetic = Some(m.clone());
                Ok(m)
            }
            PTXModule::Broadcast => {
                if let Some(ref m) = modules.broadcast {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(kernels::BROADCAST.into())?;
                modules.broadcast = Some(m.clone());
                Ok(m)
            }
            PTXModule::Conv => {
                if let Some(ref m) = modules.conv {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(kernels::CONV.into())?;
                modules.conv = Some(m.clone());
                Ok(m)
            }
            PTXModule::Fattn => {
                if let Some(ref m) = modules.fattn {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(kernels::FATTN.into())?;
                modules.fattn = Some(m.clone());
                Ok(m)
            }
            PTXModule::Fill => {
                if let Some(ref m) = modules.fill {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(kernels::FILL.into())?;
                modules.fill = Some(m.clone());
                Ok(m)
            }
            PTXModule::Fp8 => {
                if let Some(ref m) = modules.fp8 {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(kernels::FP8.into())?;
                modules.fp8 = Some(m.clone());
                Ok(m)
            }
            PTXModule::Indexing => {
                if let Some(ref m) = modules.indexing {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(kernels::INDEXING.into())?;
                modules.indexing = Some(m.clone());
                Ok(m)
            }
            PTXModule::Layout => {
                if let Some(ref m) = modules.layout {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(kernels::LAYOUT.into())?;
                modules.layout = Some(m.clone());
                Ok(m)
            }
            PTXModule::Reduce => {
                if let Some(ref m) = modules.reduce {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(kernels::REDUCE.into())?;
                modules.reduce = Some(m.clone());
                Ok(m)
            }
            PTXModule::Rope => {
                if let Some(ref m) = modules.rope {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(kernels::ROPE.into())?;
                modules.rope = Some(m.clone());
                Ok(m)
            }
            PTXModule::Q8 => {
                if let Some(ref m) = modules.q8 {
                    return Ok(m.clone());
                }
                let m = self.cuda.load_module(kernels::Q8.into())?;
                modules.q8 = Some(m.clone());
                Ok(m)
            }
        }
    }

    pub fn get_func(&self, name: &str, mdl: PTXModule) -> Result<CudaFunction> {
        let module = self.get_or_load_module(mdl).map_err(|e| e.context(format!("{mdl:?}")))?;
        let func = module
            .load_function(name)
            .map_err(|e| crate::Error::from(e).context(format!("{mdl:?} {name}")))?;
        Ok(func)
    }

    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
    }

    /// Returns the compute capability of the device as `(major, minor)`,
    /// e.g. `(9, 0)` for H100.
    pub fn compute_cap(&self) -> Result<(i32, i32)> {
        use cudarc::driver::sys::CUdevice_attribute::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        };
        let major = self.cuda.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = self.cuda.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok((major, minor))
    }

    /// When turned on, all cuda tensors **created after calling this function** will
    /// not track uses via cuda events.
    ///
    /// # Safety
    ///
    /// It is up to the user to ensure proper synchronization between multiple streams:
    /// - Ensure that no tensor is freed before a use on another stream is finished.
    /// - Ensure that a tensor is not used on another stream before allocation on the
    ///   allocating stream finishes.
    /// - Ensure that a tensor is not written two concurrently by multiple streams.
    pub unsafe fn disable_event_tracking(&self) {
        unsafe { self.cuda.disable_event_tracking() }
    }

    pub fn is_event_tracking(&self) -> bool {
        self.cuda.is_event_tracking()
    }
}

/// CUDA storage that holds both the device data and a reference to the device.
pub struct Storage<T: WithDType> {
    pub data: CudaSlice<T>,
    pub device: Device,
}

impl<T: WithDType> Storage<T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

fn kernel_name<T: WithDType>(base_name: &str) -> String {
    let dtype_str = match T::DTYPE {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        DType::I64 => "i64",
        DType::U8 => "u8",
    };
    format!("{base_name}_{dtype_str}")
}

fn gemm_config<T: num_traits::Zero + num_traits::One + std::fmt::Debug>(
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    lhs_b_stride: usize,
    rhs_b_stride: usize,
    (_dst_cs, dst_rs): (usize, usize),
    (lhs_m1, lhs_m2): (usize, usize),
    (rhs_m1, rhs_m2): (usize, usize),
) -> Result<StridedBatchedConfig<T>> {
    use cudarc::cublas::sys::cublasOperation_t;

    // Determine transposition and leading dimension for rhs (A in cuBLAS terms)
    let (lda, transa) = if rhs_m1 == 1 || n == 1 {
        (rhs_m2 as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if rhs_m2 == 1 || k == 1 {
        (rhs_m1 as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul rhs m:{m} n:{n} k:{k} strides:({rhs_m1}, {rhs_m2})")
    };

    // Determine transposition and leading dimension for lhs (B in cuBLAS terms)
    let (ldb, transb) = if lhs_m1 == 1 || k == 1 {
        (lhs_m2 as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if lhs_m2 == 1 || m == 1 {
        (lhs_m1 as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        crate::bail!("non-contiguous matmul lhs m:{m} n:{n} k:{k} strides:({lhs_m1}, {lhs_m2})")
    };

    // From the cublas documentation.
    // https://docs.nvidia.com/cuda/cublas/#cublasgemmstridedbatchedex
    // If m < 0 or n < 0 or k < 0, or
    // if transa and transb are not one of CUBLAS_OP_N, CUBLAS_OP_C, CUBLAS_OP_T, or
    // if lda < max(1, m) when transa == CUBLAS_OP_N and lda < max(1, k) otherwise, or
    // if ldb < max(1, k) when transb == CUBLAS_OP_N and ldb < max(1, n) otherwise, or
    // if ldc < max(1, m), or
    // if alpha or beta are NULL, or
    // if Atype or Btype or Ctype or algo or computeType is not supported
    let min_lda = if transa == cublasOperation_t::CUBLAS_OP_N {
        std::cmp::max(1, n as i32)
    } else {
        std::cmp::max(1, k as i32)
    };
    let min_ldb = if transb == cublasOperation_t::CUBLAS_OP_N {
        std::cmp::max(1, k as i32)
    } else {
        std::cmp::max(1, m as i32)
    };
    let lda = if lda < min_lda {
        if transa == cublasOperation_t::CUBLAS_OP_N && k == 1
            || transa == cublasOperation_t::CUBLAS_OP_T && m == 1
        {
            min_lda
        } else {
            crate::bail!("gemm: invalid lda {lda} for transa {transa:?} m:{m} n:{n} k:{k}")
        }
    } else {
        lda
    };
    let ldb = if ldb < min_ldb {
        if transb == cublasOperation_t::CUBLAS_OP_N && m == 1
            || transb == cublasOperation_t::CUBLAS_OP_T && k == 1
        {
            min_ldb
        } else {
            crate::bail!("gemm: invalid ldb {ldb} for transb {transb:?} m:{m} n:{n} k:{k}")
        }
    } else {
        ldb
    };

    let gemm = GemmConfig {
        alpha: T::one(),
        beta: T::zero(),
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: dst_rs as i32,
        transa,
        transb,
    };
    let cfg = StridedBatchedConfig {
        batch_size: lhs_b as i32,
        gemm,
        stride_a: rhs_b_stride as i64,
        stride_b: lhs_b_stride as i64,
        stride_c: (m * n) as i64,
    };
    Ok(cfg)
}

/// Implementation of GEMM using cuBLAS for f32.
fn gemm_f32(
    dst: &mut Storage<f32>,
    lhs: (&Storage<f32>, usize),
    rhs: (&Storage<f32>, usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    lhs_b_stride: usize,
    rhs_b_stride: usize,
    (_dst_cs, dst_rs): (usize, usize),
    (lhs_m1, lhs_m2): (usize, usize),
    (rhs_m1, rhs_m2): (usize, usize),
) -> Result<()> {
    let cfg = gemm_config(
        m,
        n,
        k,
        lhs_b,
        lhs_b_stride,
        rhs_b_stride,
        (_dst_cs, dst_rs),
        (lhs_m1, lhs_m2),
        (rhs_m1, rhs_m2),
    )?;

    let lhs = lhs.0.data.slice(lhs.1..);
    let rhs = rhs.0.data.slice(rhs.1..);
    unsafe { gemm_strided_batched_f32(&dst.device.blas, cfg, &rhs, &lhs, &mut dst.data)? }

    Ok(())
}

/// Implementation of GEMM using cuBLAS for f16.
fn gemm_f16(
    dst: &mut Storage<f16>,
    lhs: (&Storage<f16>, usize),
    rhs: (&Storage<f16>, usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    lhs_b_stride: usize,
    rhs_b_stride: usize,
    (_dst_cs, dst_rs): (usize, usize),
    (lhs_m1, lhs_m2): (usize, usize),
    (rhs_m1, rhs_m2): (usize, usize),
) -> Result<()> {
    let cfg = gemm_config(
        m,
        n,
        k,
        lhs_b,
        lhs_b_stride,
        rhs_b_stride,
        (_dst_cs, dst_rs),
        (lhs_m1, lhs_m2),
        (rhs_m1, rhs_m2),
    )?;
    let lhs = lhs.0.data.slice(lhs.1..);
    let rhs = rhs.0.data.slice(rhs.1..);
    unsafe { gemm_strided_batched_f16(&dst.device.blas, cfg, &rhs, &lhs, &mut dst.data)? }

    Ok(())
}

/// Implementation of GEMM using cuBLAS for bf16.
fn gemm_bf16(
    dst: &mut Storage<bf16>,
    lhs: (&Storage<bf16>, usize),
    rhs: (&Storage<bf16>, usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    lhs_b_stride: usize,
    rhs_b_stride: usize,
    (_dst_cs, dst_rs): (usize, usize),
    (lhs_m1, lhs_m2): (usize, usize),
    (rhs_m1, rhs_m2): (usize, usize),
) -> Result<()> {
    let cfg = gemm_config(
        m,
        n,
        k,
        lhs_b,
        lhs_b_stride,
        rhs_b_stride,
        (_dst_cs, dst_rs),
        (lhs_m1, lhs_m2),
        (rhs_m1, rhs_m2),
    )?;
    let lhs = lhs.0.data.slice(lhs.1..);
    let rhs = rhs.0.data.slice(rhs.1..);
    unsafe { gemm_strided_batched_bf16(&dst.device.blas, cfg, &rhs, &lhs, &mut dst.data)? }
    Ok(())
}

// Reduced precision settings
static MM_F16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_BF16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_F32_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

static CONV_F32_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Whether f32 convolutions may use tf32 tensor cores. Defaults to true,
/// matching PyTorch: convolutions tolerate the tf32 mantissa well and it is
/// what cuDNN-based stacks (including torch itself) run by default, while
/// f32 matmuls stay full precision unless
/// [`set_gemm_reduced_precision_f32`] is called.
pub fn conv_reduced_precision_f32() -> bool {
    CONV_F32_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_conv_reduced_precision_f32(b: bool) {
    CONV_F32_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

pub fn gemm_reduced_precision_f32() -> bool {
    MM_F32_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_gemm_reduced_precision_f32(b: bool) {
    MM_F32_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

pub fn gemm_reduced_precision_f16() -> bool {
    MM_F16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_gemm_reduced_precision_f16(b: bool) {
    MM_F16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

pub fn gemm_reduced_precision_bf16() -> bool {
    MM_BF16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_gemm_reduced_precision_bf16(b: bool) {
    MM_BF16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

pub fn flash_attn<T: WithDType>(
    dst: &mut Storage<T>,
    q: &Storage<T>,
    k: &Storage<T>,
    v: &Storage<T>,
    batch_size: usize,
    num_heads: usize,
    len_q: usize,
    len_kv: usize,
    head_dim: usize,
) -> Result<()> {
    const WARP_SIZE: usize = 32;
    const NUM_WARPS: usize = 4;
    const BLOCK_Q: usize = 64;
    const BLOCK_KV: usize = 64;

    if T::DTYPE != DType::BF16 {
        crate::bail!("flash_attn only supports bf16");
    }
    if !len_q.is_multiple_of(BLOCK_Q) && !len_kv.is_multiple_of(BLOCK_KV) {
        crate::bail!(
            "flash_attn requires len_q to be a multiple of {BLOCK_Q} or len_kv to be a multiple of {BLOCK_KV}"
        );
    }
    let func = dst.device.get_func("fattn_bf16", PTXModule::Fattn)?;
    if head_dim != 32 && head_dim != 64 && head_dim != 128 {
        crate::bail!("flash_attn only supports head_dim of 32, 64, or 128");
    }
    let bs = batch_size * num_heads;
    let cfg = LaunchConfig {
        grid_dim: ((bs * len_q.div_ceil(BLOCK_Q)) as u32, 1, 1),
        block_dim: ((NUM_WARPS * WARP_SIZE) as u32, 1, 1),
        shared_mem_bytes: (usize::max(BLOCK_Q, BLOCK_KV * 3) * head_dim) as u32 * 2,
    };
    let bs = bs as i32;
    let len_q = len_q as i32;
    let len_kv = len_kv as i32;
    let head_dim = head_dim as i32;
    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&q.data);
    launch_args.arg(&k.data);
    launch_args.arg(&v.data);
    launch_args.arg(&mut dst.data);
    launch_args.arg(&bs);
    launch_args.arg(&len_q);
    launch_args.arg(&len_kv);
    launch_args.arg(&head_dim);
    unsafe { launch_args.launch(cfg) }?;
    Ok(())
}

impl crate::Backend for Device {
    type Storage<T: WithDType> = Storage<T>;

    fn name(&self) -> String {
        format!("CUDA Device {}", self.cuda.ordinal())
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn storage_len<T: WithDType>(storage: &Self::Storage<T>) -> usize {
        storage.len()
    }

    unsafe fn alloc_uninit<T: WithDType>(len: usize, dev: &Self) -> Result<Self::Storage<T>> {
        let data = unsafe { dev.stream.alloc::<T>(len) }?;
        Ok(Storage { data, device: dev.clone() })
    }

    fn from_vec<T: WithDType>(v: Vec<T>, dev: &Self) -> Result<Self::Storage<T>> {
        let data = dev.cached_htod(&v)?;
        Ok(Storage { data, device: dev.clone() })
    }

    fn fill<T: WithDType>(dst: &mut Self::Storage<T>, elem: T, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let kname = kernel_name::<T>("fill");
        let func = dst.device.get_func(&kname, PTXModule::Fill)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&elem);
        launch_args.arg(&len);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn rand_uniform(dst: &mut Self::Storage<f32>, len: usize, lo: f32, up: f32) -> Result<()> {
        {
            let curand = dst.device.curand.lock().unwrap();
            let mut dst_slice = dst.data.slice_mut(..len);
            curand.0.fill_with_uniform(&mut dst_slice)?;
        }
        // Scale from [0, 1] to [lo, up]: v = v * (up - lo) + lo
        if lo != 0.0 || up != 1.0 {
            let range = up - lo;
            let func = dst.device.get_func("inplace_scale_add_f32", PTXModule::Arithmetic)?;
            let cfg = LaunchConfig::for_num_elems(len as u32);
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&len);
            launch_args.arg(&mut dst.data);
            launch_args.arg(&range);
            launch_args.arg(&lo);
            unsafe { launch_args.launch(cfg) }?;
        }
        Ok(())
    }

    fn randn(dst: &mut Self::Storage<f32>, len: usize, mean: f32, std: f32) -> Result<()> {
        let curand = dst.device.curand.lock().unwrap();
        let mut dst = dst.data.slice_mut(..len);
        curand.0.fill_with_normal(&mut dst, mean, std)?;
        Ok(())
    }

    fn copy<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()> {
        let src_slice = src.data.slice(..len);
        let mut dst_slice = dst.data.slice_mut(..len);
        dst.device.stream.memcpy_dtod(&src_slice, &mut dst_slice)?;
        Ok(())
    }

    fn to_dtype<T: WithDType, U: WithDType>(
        dst: &mut Self::Storage<U>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()> {
        let kname = format!("cast_{}_{}", T::DTYPE.cuda_name(), U::DTYPE.cuda_name());
        let func = dst.device.get_func(&kname, PTXModule::Arithmetic)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn data<T: WithDType>(src: &Self::Storage<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        let data = src.device.stream.clone_dtoh(&src.data.slice(..len))?;
        Ok(std::borrow::Cow::Owned(data))
    }

    fn inplace_unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        let (kname, alpha) = match op {
            UnaryOp::Cos => (kernel_name::<T>("inplace_cos"), None),
            UnaryOp::Sin => (kernel_name::<T>("inplace_sin"), None),
            UnaryOp::Exp => (kernel_name::<T>("inplace_exp"), None),
            UnaryOp::Log => (kernel_name::<T>("inplace_log"), None),
            UnaryOp::Neg => (kernel_name::<T>("inplace_neg"), None),
            UnaryOp::Sqr => (kernel_name::<T>("inplace_sqr"), None),
            UnaryOp::Sqrt => (kernel_name::<T>("inplace_sqrt"), None),
            UnaryOp::Rsqrt => (kernel_name::<T>("inplace_rsqrt"), None),
            UnaryOp::Abs => (kernel_name::<T>("inplace_abs"), None),
            UnaryOp::GeluErf => (kernel_name::<T>("inplace_gelu_erf"), None),
            UnaryOp::Elu { alpha } => (kernel_name::<T>("inplace_elu"), Some(alpha)),
            UnaryOp::Relu => (kernel_name::<T>("inplace_relu"), None),
            UnaryOp::Silu => (kernel_name::<T>("inplace_silu"), None),
            UnaryOp::Tanh => (kernel_name::<T>("inplace_tanh"), None),
            UnaryOp::Sigmoid => (kernel_name::<T>("inplace_sigmoid"), None),
        };
        let func = dst.device.get_func(&kname, PTXModule::Arithmetic)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&mut dst.data);
        if let Some(ref alpha) = alpha {
            launch_args.arg(alpha);
        }
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn bin_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        let kname = match op {
            BinaryOp::Add => kernel_name::<T>("assign_add"),
            BinaryOp::Sub => kernel_name::<T>("assign_sub"),
            BinaryOp::Mul => kernel_name::<T>("assign_mul"),
            BinaryOp::Div => kernel_name::<T>("assign_div"),
            BinaryOp::Maximum => kernel_name::<T>("assign_maximum"),
            BinaryOp::Minimum => kernel_name::<T>("assign_minimum"),
        };
        let func = dst.device.get_func(&kname, PTXModule::Arithmetic)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&s.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        let (kname, alpha) = match op {
            UnaryOp::Cos => (kernel_name::<T>("unary_cos"), None),
            UnaryOp::Sin => (kernel_name::<T>("unary_sin"), None),
            UnaryOp::Exp => (kernel_name::<T>("unary_exp"), None),
            UnaryOp::Log => (kernel_name::<T>("unary_log"), None),
            UnaryOp::Neg => (kernel_name::<T>("unary_neg"), None),
            UnaryOp::Sqr => (kernel_name::<T>("unary_sqr"), None),
            UnaryOp::Sqrt => (kernel_name::<T>("unary_sqrt"), None),
            UnaryOp::Rsqrt => (kernel_name::<T>("unary_rsqrt"), None),
            UnaryOp::Abs => (kernel_name::<T>("unary_abs"), None),
            UnaryOp::GeluErf => (kernel_name::<T>("unary_gelu_erf"), None),
            UnaryOp::Elu { alpha } => (kernel_name::<T>("unary_elu"), Some(alpha)),
            UnaryOp::Relu => (kernel_name::<T>("unary_relu"), None),
            UnaryOp::Silu => (kernel_name::<T>("unary_silu"), None),
            UnaryOp::Tanh => (kernel_name::<T>("unary_tanh"), None),
            UnaryOp::Sigmoid => (kernel_name::<T>("unary_sigmoid"), None),
        };
        let func = dst.device.get_func(&kname, PTXModule::Arithmetic)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        if let Some(ref alpha) = alpha {
            launch_args.arg(alpha);
        }
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        let kname = match op {
            BinaryOp::Add => kernel_name::<T>("binary_add"),
            BinaryOp::Sub => kernel_name::<T>("binary_sub"),
            BinaryOp::Mul => kernel_name::<T>("binary_mul"),
            BinaryOp::Div => kernel_name::<T>("binary_div"),
            BinaryOp::Maximum => kernel_name::<T>("binary_maximum"),
            BinaryOp::Minimum => kernel_name::<T>("binary_minimum"),
        };
        let func = dst.device.get_func(&kname, PTXModule::Arithmetic)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&lhs.data);
        launch_args.arg(&rhs.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn scale_add<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        scale: T,
        add: T,
        len: usize,
    ) -> Result<()> {
        let zero = T::zero();
        let one = T::one();
        if add == zero && scale == one {
            return Self::copy(dst, src, len);
        }
        let kname = kernel_name::<T>("scale_add");
        let func = dst.device.get_func(&kname, PTXModule::Arithmetic)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&len);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&scale);
        launch_args.arg(&add);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn transpose<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()> {
        let numel: usize = dims.iter().product();
        if dim1 == dim2 || dims.iter().filter(|v| **v != 1).count() <= 1 {
            // Simple copy when no real transpose needed
            let src_slice = src.data.slice(..numel);
            let mut dst_slice = dst.data.slice_mut(..numel);
            dst.device.stream.memcpy_dtod(&src_slice, &mut dst_slice)?;
        } else {
            let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
            let d_i: usize = dims[..dim1].iter().product();
            let d_j: usize = dims[dim1 + 1..dim2].iter().product();
            let d_k: usize = dims[(dim2 + 1)..].iter().product();
            let d1 = dims[dim1] as u32;
            let d2 = dims[dim2] as u32;
            let d_i = d_i as u32;
            let d_j = d_j as u32;
            let d_k = d_k as u32;

            let kname = kernel_name::<T>("transpose");
            let func = dst.device.get_func(&kname, PTXModule::Layout)?;
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&d1);
            launch_args.arg(&d2);
            launch_args.arg(&d_i);
            launch_args.arg(&d_j);
            launch_args.arg(&d_k);
            launch_args.arg(&src.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        }
        Ok(())
    }

    fn copy2d<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        d1: usize,
        d2: usize,
        dst_s: usize,
        src_s: usize,
        dst_o: usize,
        src_o: usize,
    ) -> Result<()> {
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let kname = kernel_name::<T>("copy2d");
        let func = dst.device.get_func(&kname, PTXModule::Fill)?;

        let d1 = d1 as u32;
        let d2 = d2 as u32;
        let src_s = src_s as u32;
        let dst_s = dst_s as u32;

        let cfg = LaunchConfig::for_num_elems(d1 * d2);
        let src_slice = src.data.slice(src_o..);
        let mut dst_slice = dst.data.slice_mut(dst_o..);

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src_slice);
        launch_args.arg(&mut dst_slice);
        launch_args.arg(&d1);
        launch_args.arg(&d2);
        launch_args.arg(&src_s);
        launch_args.arg(&dst_s);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn rope<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
        unbatched_rope: bool,
    ) -> Result<()> {
        let kname = kernel_name::<T>("rope");
        let func = dst.device.get_func(&kname, PTXModule::Rope)?;
        let bh = (b * h) as u32;
        let td = (t * d) as u32;
        let d_u32 = d as u32;
        let h_u32 = h as u32;
        // cos/sin per-batch stride: t * d/2 when batched (3D), 0 when unbatched (2D)
        let cs_stride_b = if unbatched_rope { (t * d / 2) as u32 } else { 0u32 };
        // The kernel processes bh * td / 2 elements (each thread handles 2 elements)
        let cfg = LaunchConfig::for_num_elems(bh * td / 2);

        // Slice cos/sin to start at the correct position (like CPU does)
        let cos_offset = pos * d / 2;
        let sin_offset = pos * d / 2;
        let cos_slice = cos.data.slice(cos_offset..);
        let sin_slice = sin.data.slice(sin_offset..);

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&cos_slice);
        launch_args.arg(&sin_slice);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&bh);
        launch_args.arg(&td);
        launch_args.arg(&d_u32);
        launch_args.arg(&h_u32);
        launch_args.arg(&cs_stride_b);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn rope_i<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
        unbatched_rope: bool,
    ) -> Result<()> {
        let kname = kernel_name::<T>("rope_i");
        let func = dst.device.get_func(&kname, PTXModule::Rope)?;
        let bh = (b * h) as u32;
        let td = (t * d) as u32;
        let h_u32 = h as u32;
        // cos/sin per-batch stride: t * d/2 when batched (3D), 0 when unbatched (2D)
        let cs_stride_b = if unbatched_rope { (t * d / 2) as u32 } else { 0u32 };
        // The kernel processes bh * td / 2 elements (each thread handles 2 elements)
        let cfg = LaunchConfig::for_num_elems(bh * td / 2);

        // Slice cos/sin to start at the correct position (like CPU does)
        let cos_offset = pos * d / 2;
        let sin_offset = pos * d / 2;
        let cos_slice = cos.data.slice(cos_offset..);
        let sin_slice = sin.data.slice(sin_offset..);

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&cos_slice);
        launch_args.arg(&sin_slice);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&bh);
        launch_args.arg(&td);
        launch_args.arg(&h_u32);
        launch_args.arg(&cs_stride_b);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn gemm<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: (&Self::Storage<T>, usize),
        rhs: (&Self::Storage<T>, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        lhs_b_stride: usize,
        rhs_b_stride: usize,
        dst_strides: (usize, usize),
        lhs_strides: (usize, usize),
        rhs_strides: (usize, usize),
    ) -> Result<()> {
        // Dispatch to type-specific GEMM implementations
        // We use pointer casting since we know the exact type from DTYPE
        match T::DTYPE {
            DType::F32 => {
                // SAFETY: T::DTYPE == F32 guarantees T is f32
                let dst = unsafe { &mut *(dst as *mut Storage<T> as *mut Storage<f32>) };
                let lhs_storage = unsafe { &*(lhs.0 as *const Storage<T> as *const Storage<f32>) };
                let rhs_storage = unsafe { &*(rhs.0 as *const Storage<T> as *const Storage<f32>) };
                gemm_f32(
                    dst,
                    (lhs_storage, lhs.1),
                    (rhs_storage, rhs.1),
                    m,
                    n,
                    k,
                    lhs_b,
                    lhs_b_stride,
                    rhs_b_stride,
                    dst_strides,
                    lhs_strides,
                    rhs_strides,
                )?
            }
            DType::F16 => {
                let dst = unsafe { &mut *(dst as *mut Storage<T> as *mut Storage<f16>) };
                let lhs_storage = unsafe { &*(lhs.0 as *const Storage<T> as *const Storage<f16>) };
                let rhs_storage = unsafe { &*(rhs.0 as *const Storage<T> as *const Storage<f16>) };
                gemm_f16(
                    dst,
                    (lhs_storage, lhs.1),
                    (rhs_storage, rhs.1),
                    m,
                    n,
                    k,
                    lhs_b,
                    lhs_b_stride,
                    rhs_b_stride,
                    dst_strides,
                    lhs_strides,
                    rhs_strides,
                )?
            }
            DType::BF16 => {
                let dst = unsafe { &mut *(dst as *mut Storage<T> as *mut Storage<bf16>) };
                let lhs_storage = unsafe { &*(lhs.0 as *const Storage<T> as *const Storage<bf16>) };
                let rhs_storage = unsafe { &*(rhs.0 as *const Storage<T> as *const Storage<bf16>) };
                gemm_bf16(
                    dst,
                    (lhs_storage, lhs.1),
                    (rhs_storage, rhs.1),
                    m,
                    n,
                    k,
                    lhs_b,
                    lhs_b_stride,
                    rhs_b_stride,
                    dst_strides,
                    lhs_strides,
                    rhs_strides,
                )?
            }
            _ => crate::bail!("GEMM not supported for dtype {:?}", T::DTYPE),
        }
        Ok(())
    }

    fn copy_strided<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        src_offset: usize,
        dims: &[usize],
        src_strides: &[usize],
    ) -> Result<()> {
        let numel: usize = dims.iter().product();
        if numel == 0 {
            return Ok(());
        }

        let n = dims.len();

        // Detect 2D strided pattern: innermost dim is contiguous (stride 1),
        // and all batch dims (before the last two) are contiguous relative to each other.
        // This covers e.g. dims [A, B] with strides [2*B, 1] — rows are contiguous but spaced.
        if n >= 2
            && src_strides[n - 1] == 1
            && (0..n.saturating_sub(2)).all(|i| src_strides[i] == dims[i + 1] * src_strides[i + 1])
        {
            let rows = dims[n - 2];
            let cols = dims[n - 1];
            let src_stride = src_strides[n - 2];
            let batch: usize = dims[..n - 2].iter().product::<usize>().max(1);

            const TILE: u32 = 32;
            const BLOCK_ROWS: u32 = 8;
            let cfg = LaunchConfig {
                grid_dim: (
                    (cols as u32).div_ceil(TILE),
                    (rows as u32).div_ceil(TILE),
                    batch as u32,
                ),
                block_dim: (TILE, BLOCK_ROWS, 1),
                shared_mem_bytes: 0,
            };

            let kname = kernel_name::<T>("copy_strided_2d");
            let func = dst.device.get_func(&kname, PTXModule::Layout)?;
            let src_offset_u32 = src_offset as u32;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&rows);
            launch_args.arg(&cols);
            launch_args.arg(&src_stride);
            launch_args.arg(&src_offset_u32);
            launch_args.arg(&src.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
            return Ok(());
        }

        let num_dims = n;
        let info: Vec<usize> = dims.iter().chain(src_strides.iter()).copied().collect();
        let info_dev = dst.device.cached_htod(&info)?;

        let kname = kernel_name::<T>("copy_strided");
        let func = dst.device.get_func(&kname, PTXModule::Layout)?;
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let num_dims_u32 = num_dims as u32;
        let src_offset_u32 = src_offset as u32;
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&numel);
        launch_args.arg(&num_dims_u32);
        launch_args.arg(&info_dev);
        launch_args.arg(&src_offset_u32);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn scatter_set<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &Self::Storage<i64>,
        dim: usize,
        dst_dims: &[usize],
        src_dims: &[usize],
    ) -> Result<()> {
        let right_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let src_dim_size = src_dims[dim];
        let dst_dim_size = dst_dims[dim];
        let numel: usize = src_dims.iter().product();

        let kname = kernel_name::<T>("scatter_set");
        let func = dst.device.get_func(&kname, PTXModule::Indexing)?;

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let numel_i32 = numel as i32;
        let right_size_i32 = right_size as i32;
        let src_dim_size_i32 = src_dim_size as i32;
        let dst_dim_size_i32 = dst_dim_size as i32;

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&src.data);
        launch_args.arg(&ids.data);
        launch_args.arg(&numel_i32);
        launch_args.arg(&right_size_i32);
        launch_args.arg(&src_dim_size_i32);
        launch_args.arg(&dst_dim_size_i32);
        unsafe { launch_args.launch(cfg) }?;

        Ok(())
    }

    fn index_select<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &Self::Storage<i64>,
        num_ids: usize,
        dim: usize,
        dims: &[usize],
    ) -> Result<()> {
        let left_size: usize = dims[..dim].iter().product();
        let right_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);

        let kname = kernel_name::<T>("is_i64");
        let func = dst.device.get_func(&kname, PTXModule::Indexing)?;

        const NUM_THREADS: u32 = 1024;
        let num_ids_u32 = num_ids as u32;
        let right_size_u32 = right_size as u32;
        let threads_x = u32::min(NUM_THREADS, num_ids_u32);
        let threads_y = u32::min(NUM_THREADS / threads_x, right_size_u32).max(1);
        let num_blocks_x = num_ids_u32.div_ceil(threads_x);
        let num_blocks_y = right_size_u32.div_ceil(threads_y);

        let cfg = LaunchConfig {
            block_dim: (threads_x, threads_y, 1),
            grid_dim: (num_blocks_x, num_blocks_y, 1),
            shared_mem_bytes: 0,
        };

        let num_ids_i32 = num_ids as i32;
        let right_size_i32 = right_size as i32;
        let src_dim_size = dims[dim];
        let src_dim_size_i32 = src_dim_size as i32;

        for left in 0..left_size {
            let src_offset = left * src_dim_size * right_size;
            let dst_offset = left * num_ids * right_size;
            let src_slice = src.data.slice(src_offset..);
            let mut dst_slice = dst.data.slice_mut(dst_offset..);

            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&num_ids_i32);
            launch_args.arg(&right_size_i32);
            launch_args.arg(&src_dim_size_i32);
            launch_args.arg(&ids.data);
            launch_args.arg(&src_slice);
            launch_args.arg(&mut dst_slice);
            unsafe { launch_args.launch(cfg) }?;
        }
        Ok(())
    }

    fn apply_causality_mask<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()> {
        let total = bh * t1 * t2;
        let kname = kernel_name::<T>("causality_mask");
        let func = dst.device.get_func(&kname, PTXModule::Indexing)?;

        let cfg = LaunchConfig::for_num_elems(total as u32);
        let bh = bh as u32;
        let t1 = t1 as u32;
        let t2 = t2 as u32;
        let offset = offset as u32;

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&bh);
        launch_args.arg(&t1);
        launch_args.arg(&t2);
        launch_args.arg(&offset);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn softmax<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()> {
        // dim_m1 is ncols (last dimension), d is nrows
        let ncols = dim_m1 as i32;
        let nrows = d as u32;

        let kname = kernel_name::<T>("softmax");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;

        // Kernel uses: row = blockDim.x*blockIdx.x + threadIdx.x, tid = threadIdx.y
        // One row per block, 32 threads per row for warp-based reduction
        let block_dim = (1, 32, 1);
        let grid_dim = (nrows, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&ncols);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn rms_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        // dim_m1 is ncols (last dimension), d is nrows
        let ncols = dim_m1 as i32;
        let nrows = d as i32;

        let kname = kernel_name::<T>("rmsnorm");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;
        let block_size = if ncols < 1024 { 32 } else { 1024 };
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&alpha.data);
        launch_args.arg(&ncols);
        launch_args.arg(&block_size);
        launch_args.arg(&eps);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn layer_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        weight: &Self::Storage<T>,
        bias: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
        remove_mean: bool,
    ) -> Result<()> {
        // dim_m1 is ncols (last dimension), d is nrows
        let ncols = dim_m1 as i32;
        let nrows = d as i32;
        let remove_mean: i32 = if remove_mean { 1 } else { 0 };

        let kname = kernel_name::<T>("layernorm");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;
        let block_size = if ncols < 1024 { 32 } else { 1024 };
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        launch_args.arg(&weight.data);
        launch_args.arg(&bias.data);
        launch_args.arg(&ncols);
        launch_args.arg(&block_size);
        launch_args.arg(&eps);
        launch_args.arg(&remove_mean);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn reduce_max<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        let src_numel = outer_size * dim_size * inner_size;
        let num_outputs = outer_size * inner_size;
        let el_to_sum_per_block = dim_size;

        // Set up dims and strides for strided access
        // Iteration shape: (outer_size, inner_size, dim_size)
        // Physical layout: (outer_size, dim_size, inner_size)
        let dims: [usize; 3] = [outer_size, inner_size, dim_size];
        let strides: [usize; 3] = [dim_size * inner_size, 1, inner_size];
        let info: Vec<usize> = dims.iter().chain(strides.iter()).copied().collect();
        let info_dev = dst.device.cached_htod(&info)?;

        let kname = kernel_name::<T>("fast_max");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;

        const BLOCK_SIZE: u32 = 1024;
        let block_dim = (BLOCK_SIZE, 1, 1);
        let grid_dim = (num_outputs as u32, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let num_dims: usize = 3;
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src_numel);
        launch_args.arg(&el_to_sum_per_block);
        launch_args.arg(&num_dims);
        launch_args.arg(&info_dev);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn reduce_min<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        let src_numel = outer_size * dim_size * inner_size;
        let num_outputs = outer_size * inner_size;
        let el_to_sum_per_block = dim_size;

        // Set up dims and strides for strided access
        // Iteration shape: (outer_size, inner_size, dim_size)
        // Physical layout: (outer_size, dim_size, inner_size)
        let dims: [usize; 3] = [outer_size, inner_size, dim_size];
        let strides: [usize; 3] = [dim_size * inner_size, 1, inner_size];
        let info: Vec<usize> = dims.iter().chain(strides.iter()).copied().collect();
        let info_dev = dst.device.cached_htod(&info)?;

        let kname = kernel_name::<T>("fast_min");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;

        const BLOCK_SIZE: u32 = 1024;
        let block_dim = (BLOCK_SIZE, 1, 1);
        let grid_dim = (num_outputs as u32, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let num_dims: usize = 3;
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src_numel);
        launch_args.arg(&el_to_sum_per_block);
        launch_args.arg(&num_dims);
        launch_args.arg(&info_dev);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn reduce_argmin<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        let src_numel = outer_size * dim_size * inner_size;
        let num_outputs = outer_size * inner_size;
        let el_to_sum_per_block = dim_size;

        // Set up dims and strides for strided access
        // Iteration shape: (outer_size, inner_size, dim_size)
        // Physical layout: (outer_size, dim_size, inner_size)
        let dims: [usize; 3] = [outer_size, inner_size, dim_size];
        let strides: [usize; 3] = [dim_size * inner_size, 1, inner_size];
        let info: Vec<usize> = dims.iter().chain(strides.iter()).copied().collect();
        let info_dev = dst.device.cached_htod(&info)?;

        let kname = kernel_name::<T>("fast_argmin");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;

        const BLOCK_SIZE: u32 = 1024;
        let block_dim = (BLOCK_SIZE, 1, 1);
        let grid_dim = (num_outputs as u32, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let num_dims: usize = 3;
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src_numel);
        launch_args.arg(&el_to_sum_per_block);
        launch_args.arg(&num_dims);
        launch_args.arg(&info_dev);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn reduce_argmax<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        let src_numel = outer_size * dim_size * inner_size;
        let num_outputs = outer_size * inner_size;
        let el_to_sum_per_block = dim_size;

        let dims: [usize; 3] = [outer_size, inner_size, dim_size];
        let strides: [usize; 3] = [dim_size * inner_size, 1, inner_size];
        let info: Vec<usize> = dims.iter().chain(strides.iter()).copied().collect();
        let info_dev = dst.device.cached_htod(&info)?;

        let kname = kernel_name::<T>("fast_argmax");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;

        const BLOCK_SIZE: u32 = 1024;
        let block_dim = (BLOCK_SIZE, 1, 1);
        let grid_dim = (num_outputs as u32, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let num_dims: usize = 3;
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src_numel);
        launch_args.arg(&el_to_sum_per_block);
        launch_args.arg(&num_dims);
        launch_args.arg(&info_dev);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn reduce_sum<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        let src_numel = outer_size * dim_size * inner_size;
        let num_outputs = outer_size * inner_size;
        let el_to_sum_per_block = dim_size;

        // Set up dims and strides for strided access
        // Iteration shape: (outer_size, inner_size, dim_size)
        // Physical layout: (outer_size, dim_size, inner_size)
        let dims: [usize; 3] = [outer_size, inner_size, dim_size];
        let strides: [usize; 3] = [dim_size * inner_size, 1, inner_size];
        let info: Vec<usize> = dims.iter().chain(strides.iter()).copied().collect();
        let info_dev = dst.device.cached_htod(&info)?;

        let kname = kernel_name::<T>("fast_sum");
        let func = dst.device.get_func(&kname, PTXModule::Reduce)?;

        const BLOCK_SIZE: u32 = 1024;
        let block_dim = (BLOCK_SIZE, 1, 1);
        let grid_dim = (num_outputs as u32, 1, 1);
        let cfg = LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 };

        let num_dims: usize = 3;
        let mut launch_args = dst.device.stream.launch_builder(&func);
        launch_args.arg(&src_numel);
        launch_args.arg(&el_to_sum_per_block);
        launch_args.arg(&num_dims);
        launch_args.arg(&info_dev);
        launch_args.arg(&src.data);
        launch_args.arg(&mut dst.data);
        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }

    fn broadcast_binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        op: BinaryOp,
    ) -> Result<()> {
        let numel: usize = dst_shape.iter().product();
        if numel == 0 {
            return Ok(());
        }

        let op_name = match op {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Maximum => "max",
            BinaryOp::Minimum => "min",
        };

        let lhs_no_zero = lhs_strides.iter().all(|&s| s > 0);
        let rhs_no_zero = rhs_strides.iter().all(|&s| s > 0);

        let cfg = LaunchConfig::for_num_elems(numel as u32);

        // Dispatch to optimized kernels based on stride patterns
        if lhs_no_zero && rhs_no_zero {
            // Both operands contiguous
            let kname = format!("broadcast_{}_contiguous_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else if lhs_no_zero && dst_shape.len() == 2 && rhs_strides == [0, 1] {
            // rhs broadcasts along first dim
            let dim1 = dst_shape[1];
            let kname = format!("broadcast_{}_rhs_row_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&dim1);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else if lhs_no_zero && dst_shape.len() == 2 && rhs_strides == [1, 0] {
            // rhs broadcasts along second dim
            let dim1 = dst_shape[1];
            let kname = format!("broadcast_{}_rhs_col_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&dim1);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else if rhs_no_zero && dst_shape.len() == 2 && lhs_strides == [0, 1] {
            // lhs broadcasts along first dim
            let dim1 = dst_shape[1];
            let kname = format!("broadcast_{}_lhs_row_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&dim1);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else if rhs_no_zero && dst_shape.len() == 2 && lhs_strides == [1, 0] {
            // lhs broadcasts along second dim
            let dim1 = dst_shape[1];
            let kname = format!("broadcast_{}_lhs_col_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&dim1);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else if lhs_no_zero
            && dst_shape.len() == 3
            && rhs_strides[2] == 1
            && rhs_strides[1] == 0
            && rhs_strides[0] == dst_shape[2]
        {
            // rhs broadcasts along middle dim of 3D, strides [dim2, 0, 1]
            let dim12 = dst_shape[1] * dst_shape[2];
            let dim2 = dst_shape[2];
            let kname = format!("broadcast_{}_rhs_3d_mid_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&dim12);
            launch_args.arg(&dim2);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else if rhs_no_zero
            && dst_shape.len() == 3
            && lhs_strides[2] == 1
            && lhs_strides[1] == 0
            && lhs_strides[0] == dst_shape[2]
        {
            // lhs broadcasts along middle dim of 3D, strides [dim2, 0, 1]
            let dim12 = dst_shape[1] * dst_shape[2];
            let dim2 = dst_shape[2];
            let kname = format!("broadcast_{}_lhs_3d_mid_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&dim12);
            launch_args.arg(&dim2);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        } else {
            // General strided case
            let num_dims = dst_shape.len();
            let info: Vec<usize> = dst_shape
                .iter()
                .chain(lhs_strides.iter())
                .chain(rhs_strides.iter())
                .copied()
                .collect();
            let info_dev = dst.device.cached_htod(&info)?;

            let kname = format!("broadcast_{}_strided_{}", op_name, T::DTYPE.cuda_name());
            let func = dst.device.get_func(&kname, PTXModule::Broadcast)?;
            let num_dims_u32 = num_dims as u32;
            let mut launch_args = dst.device.stream.launch_builder(&func);
            launch_args.arg(&numel);
            launch_args.arg(&num_dims_u32);
            launch_args.arg(&info_dev);
            launch_args.arg(&lhs.data);
            launch_args.arg(&rhs.data);
            launch_args.arg(&mut dst.data);
            unsafe { launch_args.launch(cfg) }?;
        }
        Ok(())
    }

    fn conv1d<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        kernel: &Self::Storage<T>,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        length: usize,
        out_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<()> {
        if groups == 1 {
            // IM2COL approach: transform conv1d into matrix multiplication
            // 1. Im2Col: transform input [B, C, L] -> [B, L_out, C * K]
            // 2. Matmul: [B, L_out, C*K] x [C*K, out_channels] -> [B, L_out, out_channels]
            // 3. Transpose result to [B, out_channels, L_out]
            conv1d_im2col(
                dst,
                src,
                kernel,
                batch,
                in_channels,
                out_channels,
                length,
                out_length,
                kernel_size,
                stride,
                padding,
                dilation,
            )
        } else {
            // Direct kernel for grouped convolutions
            conv1d_direct(
                dst,
                src,
                kernel,
                batch,
                in_channels,
                out_channels,
                length,
                out_length,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            )
        }
    }

    fn conv_transpose1d<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        kernel: &Self::Storage<T>,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        length: usize,
        out_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<()> {
        // COL2IM approach can be used when:
        // - groups == 1
        // - padding == 0
        // - output_padding == 0
        let can_use_col2im = groups == 1 && padding == 0 && output_padding == 0;

        if can_use_col2im {
            // COL2IM approach: matmul + col2im transformation
            conv_transpose1d_col2im(
                dst,
                src,
                kernel,
                batch,
                in_channels,
                out_channels,
                length,
                out_length,
                kernel_size,
                stride,
            )
        } else {
            // Direct kernel for grouped convolutions or with padding
            conv_transpose1d_direct(
                dst,
                src,
                kernel,
                batch,
                in_channels,
                out_channels,
                length,
                out_length,
                kernel_size,
                stride,
                padding,
                output_padding,
                groups,
            )
        }
    }
}

// ============================================================================
// Conv1d implementation using im2col + cuBLAS gemm
// ============================================================================

fn conv1d_im2col<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<()> {
    let k = in_channels * kernel_size;
    let col_numel = batch * out_length * k;

    // Step 1: Im2Col transformation
    // Allocate temporary buffer for col data [B, L_out, C_in * K]
    let mut col: CudaSlice<T> = unsafe { dst.device.stream.alloc(col_numel) }?;

    let kname = format!("im2col1d_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    const TILE: u32 = 32;
    const BLOCK_ROWS: u32 = 8;
    let cfg = LaunchConfig {
        grid_dim: ((k as u32).div_ceil(TILE), (out_length as u32).div_ceil(TILE), batch as u32),
        block_dim: (TILE, BLOCK_ROWS, 1),
        shared_mem_bytes: 0,
    };

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&in_channels);
    launch_args.arg(&length);
    launch_args.arg(&out_length);
    launch_args.arg(&kernel_size);
    launch_args.arg(&stride);
    launch_args.arg(&padding);
    launch_args.arg(&dilation);
    launch_args.arg(&src.data);
    launch_args.arg(&mut col);
    unsafe { launch_args.launch(cfg) }?;

    // Step 2: Matrix multiplication using cuBLAS
    // col: [B, L_out, K] where K = in_channels * kernel_size
    // kernel: [out_channels, K] (stored as [out_channels, in_channels, kernel_size])
    // result: [B, L_out, out_channels]
    //
    // We need to allocate a temporary buffer for the result since it's in a different layout
    let result_numel = batch * out_length * out_channels;
    let mut result: CudaSlice<T> = unsafe { dst.device.stream.alloc(result_numel) }?;

    // For each batch, perform: result[b] = col[b] @ kernel^T
    // col[b]: [L_out, K], kernel: [out_channels, K], result[b]: [L_out, out_channels]
    // Using cuBLAS: C = alpha * A * B + beta * C
    // We want: result = col @ kernel^T
    // cuBLAS uses column-major, so we compute: result^T = kernel @ col^T
    // Which gives us result in row-major as [L_out, out_channels]
    conv1d_gemm(&dst.device, &col, &kernel.data, &mut result, batch, out_length, out_channels, k)?;

    // Step 3: Transpose from [B, L_out, out_channels] to [B, out_channels, L_out]
    let kname = format!("transpose_blc_bcl_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    let cfg = LaunchConfig {
        grid_dim: (
            (out_channels as u32).div_ceil(TILE),
            (out_length as u32).div_ceil(TILE),
            batch as u32,
        ),
        block_dim: (TILE, BLOCK_ROWS, 1),
        shared_mem_bytes: 0,
    };

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&out_length);
    launch_args.arg(&out_channels);
    launch_args.arg(&result);
    launch_args.arg(&mut dst.data);
    unsafe { launch_args.launch(cfg) }?;

    Ok(())
}

/// Batched GEMM for conv1d: result[b] = col[b] @ kernel^T
/// col: [B, M, K], kernel: [N, K], result: [B, M, N]
fn conv1d_gemm<T: WithDTypeF>(
    device: &Device,
    col: &CudaSlice<T>,
    kernel: &CudaSlice<T>,
    result: &mut CudaSlice<T>,
    batch: usize,
    m: usize, // out_length
    n: usize, // out_channels
    k: usize, // in_channels * kernel_size
) -> Result<()> {
    match T::DTYPE {
        DType::F32 => {
            let col = unsafe { &*(col as *const CudaSlice<T> as *const CudaSlice<f32>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<f32>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<f32>) };
            conv1d_gemm_f32(device, col, kernel, result, batch, m, n, k)
        }
        DType::F16 => {
            let col = unsafe { &*(col as *const CudaSlice<T> as *const CudaSlice<f16>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<f16>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<f16>) };
            conv1d_gemm_f16(device, col, kernel, result, batch, m, n, k)
        }
        DType::BF16 => {
            let col = unsafe { &*(col as *const CudaSlice<T> as *const CudaSlice<bf16>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<bf16>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<bf16>) };
            conv1d_gemm_bf16(device, col, kernel, result, batch, m, n, k)
        }
        _ => crate::bail!("conv1d GEMM not supported for dtype {:?}", T::DTYPE),
    }
}

/// f32 gemm for the convolution paths, through `cublasGemmEx` so the compute
/// type can honor [`conv_reduced_precision_f32`] (the plain `cublasSgemm`
/// wrapper has no compute-type parameter).
unsafe fn conv_gemm_ex_f32(
    device: &Device,
    cfg: cudarc::cublas::GemmConfig<f32>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
    use cudarc::cublas::sys;
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let compute_type = if conv_reduced_precision_f32() {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
    } else {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    };
    let stream = device.stream.clone();
    let (a, _guard_a) = a.device_ptr(&stream);
    let (b, _guard_b) = b.device_ptr(&stream);
    let (c, _guard_c) = c.device_ptr_mut(&stream);
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *device.blas.handle(),
            cfg.transa,
            cfg.transb,
            cfg.m,
            cfg.n,
            cfg.k,
            &cfg.alpha as *const f32 as *const _,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.lda,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.ldb,
            &cfg.beta as *const f32 as *const _,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.ldc,
            compute_type,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
}

fn conv1d_gemm_f32(
    device: &Device,
    col: &CudaSlice<f32>,
    kernel: &CudaSlice<f32>,
    result: &mut CudaSlice<f32>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    // We compute: result = col @ kernel^T (row-major)
    // col[b]: [M, K] row-major, kernel: [N, K] row-major
    // result[b]: [M, N] row-major
    //
    // cuBLAS is column-major. Row-major [R, C] with stride C is seen as column-major [C, R]:
    // - kernel stored [N, K] row-major -> cuBLAS sees [K, N], lda = K
    // - col stored [M, K] row-major -> cuBLAS sees [K, M], ldb = K
    // - result stored [M, N] row-major -> cuBLAS sees [N, M], ldc = N
    //
    // We want: result = col @ kernel^T (row-major)
    // Taking transpose: result^T = kernel @ col^T
    // In cuBLAS terms for C = op(A) * op(B) where C is [N, M]:
    // - A = kernel, seen as [K, N], with transa=T gives [N, K]
    // - B = col, seen as [K, M], with transb=N gives [K, M]
    // - Result: [N, K] @ [K, M] = [N, M] ✓

    let gemm = GemmConfig {
        alpha: 1.0f32,
        beta: 0.0f32,
        m: n as i32,           // rows of C (N = out_channels)
        n: (batch * m) as i32, // cols of C (M = out_length)
        k: k as i32,           // inner dimension
        lda: k as i32,         // leading dim of A (row stride of kernel)
        ldb: k as i32,         // leading dim of B (row stride of col)
        ldc: n as i32,         // leading dim of C (row stride of result)
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };
    unsafe {
        conv_gemm_ex_f32(device, gemm, kernel, col, result)?;
    }
    Ok(())
}

fn conv1d_gemm_f16(
    device: &Device,
    col: &CudaSlice<f16>,
    kernel: &CudaSlice<f16>,
    result: &mut CudaSlice<f16>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    let gemm = GemmConfig {
        alpha: f16::ONE,
        beta: f16::ZERO,
        m: n as i32,
        n: (batch * m) as i32,
        k: k as i32,
        lda: k as i32,
        ldb: k as i32,
        ldc: n as i32,
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };
    unsafe {
        device.blas.gemm(gemm, kernel, col, result)?;
    }
    Ok(())
}

fn conv1d_gemm_bf16(
    device: &Device,
    col: &CudaSlice<bf16>,
    kernel: &CudaSlice<bf16>,
    result: &mut CudaSlice<bf16>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    let gemm = GemmConfig {
        alpha: bf16::ONE,
        beta: bf16::ZERO,
        m: n as i32,
        n: (batch * m) as i32,
        k: k as i32,
        lda: k as i32,
        ldb: k as i32,
        ldc: n as i32,
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };
    unsafe {
        device.blas.gemm(gemm, kernel, col, result)?;
    }
    Ok(())
}

// ============================================================================
// Conv1d direct implementation (fallback for grouped convolutions)
// ============================================================================

fn conv1d_direct<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<()> {
    let dst_numel = batch * out_channels * out_length;

    let kname = format!("conv1d_direct_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    let cfg = LaunchConfig::for_num_elems(dst_numel as u32);

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&dst_numel);
    launch_args.arg(&batch);
    launch_args.arg(&in_channels);
    launch_args.arg(&length);
    launch_args.arg(&out_channels);
    launch_args.arg(&out_length);
    launch_args.arg(&kernel_size);
    launch_args.arg(&stride);
    launch_args.arg(&padding);
    launch_args.arg(&dilation);
    launch_args.arg(&groups);
    launch_args.arg(&src.data);
    launch_args.arg(&kernel.data);
    launch_args.arg(&mut dst.data);
    unsafe { launch_args.launch(cfg) }?;

    Ok(())
}

// ============================================================================
// Conv transpose 1d implementation using col2im
// ============================================================================

fn conv_transpose1d_col2im<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,     // input length
    out_length: usize, // output length
    kernel_size: usize,
    stride: usize,
) -> Result<()> {
    // COL2IM approach:
    // 1. Transpose input from [B, C_in, L_in] to [B, L_in, C_in]
    // 2. Matmul: [B, L_in, C_in] @ [C_in, C_out * K] -> [B, L_in, C_out * K]
    // 3. Col2Im: [B, L_in, C_out, K] -> [B, C_out, L_out]

    let src_numel = batch * in_channels * length;
    let n = out_channels * kernel_size;
    let col_numel = batch * length * n;

    // Step 1: Transpose input from [B, C_in, L_in] to [B, L_in, C_in]
    let mut src_transposed: CudaSlice<T> = unsafe { dst.device.stream.alloc(src_numel) }?;

    let kname = format!("transpose_bcl_blc_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    const TILE: u32 = 32;
    const BLOCK_ROWS: u32 = 8;
    let cfg = LaunchConfig {
        grid_dim: (
            (length as u32).div_ceil(TILE),
            (in_channels as u32).div_ceil(TILE),
            batch as u32,
        ),
        block_dim: (TILE, BLOCK_ROWS, 1),
        shared_mem_bytes: 0,
    };

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&in_channels);
    launch_args.arg(&length);
    launch_args.arg(&src.data);
    launch_args.arg(&mut src_transposed);
    unsafe { launch_args.launch(cfg) }?;

    // Step 2: Matrix multiplication
    // src_transposed: [B, L_in, C_in]
    // kernel: [C_in, C_out, K] stored row-major, treat as [C_in, C_out * K]
    // result (col): [B, L_in, C_out * K]
    let mut col: CudaSlice<T> = unsafe { dst.device.stream.alloc(col_numel) }?;

    conv_transpose1d_gemm(
        &dst.device,
        &src_transposed,
        &kernel.data,
        &mut col,
        batch,
        length,
        n,
        in_channels,
    )?;

    // Step 3: Col2Im transformation
    // col: [B, L_in, C_out * K] = [B, L_in, C_out, K]
    // output: [B, C_out, L_out]
    let kname = format!("col2im1d_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    let cfg = LaunchConfig {
        grid_dim: (
            (out_length as u32).div_ceil(TILE),
            (out_channels as u32).div_ceil(TILE),
            batch as u32,
        ),
        block_dim: (TILE, BLOCK_ROWS, 1),
        shared_mem_bytes: 0,
    };

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&length);
    launch_args.arg(&out_channels);
    launch_args.arg(&out_length);
    launch_args.arg(&kernel_size);
    launch_args.arg(&stride);
    launch_args.arg(&col);
    launch_args.arg(&mut dst.data);
    unsafe { launch_args.launch(cfg) }?;

    Ok(())
}

/// Batched GEMM for conv_transpose1d: result[b] = src[b] @ kernel
/// src: [B, M, K], kernel: [K, N], result: [B, M, N]
fn conv_transpose1d_gemm<T: WithDTypeF>(
    device: &Device,
    src: &CudaSlice<T>,
    kernel: &CudaSlice<T>,
    result: &mut CudaSlice<T>,
    batch: usize,
    m: usize, // length (L_in)
    n: usize, // out_channels * kernel_size
    k: usize, // in_channels
) -> Result<()> {
    match T::DTYPE {
        DType::F32 => {
            let src = unsafe { &*(src as *const CudaSlice<T> as *const CudaSlice<f32>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<f32>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<f32>) };
            conv_transpose1d_gemm_f32(device, src, kernel, result, batch, m, n, k)
        }
        DType::F16 => {
            let src = unsafe { &*(src as *const CudaSlice<T> as *const CudaSlice<f16>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<f16>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<f16>) };
            conv_transpose1d_gemm_f16(device, src, kernel, result, batch, m, n, k)
        }
        DType::BF16 => {
            let src = unsafe { &*(src as *const CudaSlice<T> as *const CudaSlice<bf16>) };
            let kernel = unsafe { &*(kernel as *const CudaSlice<T> as *const CudaSlice<bf16>) };
            let result = unsafe { &mut *(result as *mut CudaSlice<T> as *mut CudaSlice<bf16>) };
            conv_transpose1d_gemm_bf16(device, src, kernel, result, batch, m, n, k)
        }
        _ => crate::bail!("conv_transpose1d GEMM not supported for dtype {:?}", T::DTYPE),
    }
}

fn conv_transpose1d_gemm_f32(
    device: &Device,
    src: &CudaSlice<f32>,
    kernel: &CudaSlice<f32>,
    result: &mut CudaSlice<f32>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    // We compute: result = src @ kernel (row-major)
    // src[b]: [M, K] row-major, kernel: [K, N] row-major
    // result[b]: [M, N] row-major
    //
    // cuBLAS is column-major. Row-major data appears transposed:
    // - src stored [M, K] -> cuBLAS sees [K, M]
    // - kernel stored [K, N] -> cuBLAS sees [N, K]
    // - result stored [M, N] -> cuBLAS sees [N, M]
    //
    // For row-major C = A @ B: C^T = B^T @ A^T in column-major
    // So: result^T[N, M] = kernel^T[N, K] @ src^T[K, M]
    //
    // cuBLAS sees kernel as [N, K] and src as [K, M], no transpose needed
    // Row-major [R, C] has row stride C, cuBLAS sees as col-major [C, R] with ld = C
    // - kernel [K, N] row-major -> cuBLAS [N, K], lda = N
    // - src [M, K] row-major -> cuBLAS [K, M], ldb = K
    // transa = N, transb = N

    let gemm = GemmConfig {
        alpha: 1.0f32,
        beta: 0.0f32,
        m: n as i32,           // rows of C = N
        n: (batch * m) as i32, // cols of C = M
        k: k as i32,           // inner dimension = K
        lda: n as i32,         // leading dim of kernel (row stride N)
        ldb: k as i32,         // leading dim of src (row stride K)
        ldc: n as i32,         // leading dim of result (row stride N)
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };
    unsafe {
        conv_gemm_ex_f32(device, gemm, kernel, src, result)?;
    }
    Ok(())
}

fn conv_transpose1d_gemm_f16(
    device: &Device,
    src: &CudaSlice<f16>,
    kernel: &CudaSlice<f16>,
    result: &mut CudaSlice<f16>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    let gemm = GemmConfig {
        alpha: f16::ONE,
        beta: f16::ZERO,
        m: n as i32,
        n: (batch * m) as i32,
        k: k as i32,
        lda: n as i32,
        ldb: k as i32,
        ldc: n as i32,
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };
    unsafe {
        device.blas.gemm(gemm, kernel, src, result)?;
    }
    Ok(())
}

fn conv_transpose1d_gemm_bf16(
    device: &Device,
    src: &CudaSlice<bf16>,
    kernel: &CudaSlice<bf16>,
    result: &mut CudaSlice<bf16>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t;

    // Row-major [R, C] has row stride C, cuBLAS sees as col-major [C, R] with ld = C
    // - kernel [K, N] row-major -> cuBLAS [N, K], lda = N
    // - src [M, K] row-major -> cuBLAS [K, M], ldb = K
    let gemm = GemmConfig {
        alpha: bf16::ONE,
        beta: bf16::ZERO,
        m: n as i32,
        n: (batch * m) as i32,
        k: k as i32,
        lda: n as i32,
        ldb: k as i32,
        ldc: n as i32,
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };
    unsafe {
        device.blas.gemm(gemm, kernel, src, result)?;
    }
    Ok(())
}

// ============================================================================
// Conv transpose 1d direct implementation (fallback)
// ============================================================================

fn conv_transpose1d_direct<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
) -> Result<()> {
    let dst_numel = batch * out_channels * out_length;

    let kname = format!("conv_transpose1d_direct_{}", T::DTYPE.cuda_name());
    let func = dst.device.get_func(&kname, PTXModule::Conv)?;
    let cfg = LaunchConfig::for_num_elems(dst_numel as u32);

    // Note: dilation is fixed to 1 for now (matching CPU backend behavior)
    let dilation: usize = 1;

    let mut launch_args = dst.device.stream.launch_builder(&func);
    launch_args.arg(&dst_numel);
    launch_args.arg(&batch);
    launch_args.arg(&in_channels);
    launch_args.arg(&length);
    launch_args.arg(&out_channels);
    launch_args.arg(&out_length);
    launch_args.arg(&kernel_size);
    launch_args.arg(&stride);
    launch_args.arg(&padding);
    launch_args.arg(&output_padding);
    launch_args.arg(&dilation);
    launch_args.arg(&groups);
    launch_args.arg(&src.data);
    launch_args.arg(&kernel.data);
    launch_args.arg(&mut dst.data);
    unsafe { launch_args.launch(cfg) }?;

    Ok(())
}

unsafe fn gemm_strided_batched_f32(
    cublas: &cudarc::cublas::CudaBlas,
    cfg: StridedBatchedConfig<f32>,
    a: &cudarc::driver::CudaView<f32>,
    b: &cudarc::driver::CudaView<f32>,
    c: &mut CudaSlice<f32>,
) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
    use cudarc::cublas::sys;
    use cudarc::driver::DevicePtrMut;

    let compute_type = if gemm_reduced_precision_f32() {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
    } else {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    };
    let alpha = &cfg.gemm.alpha as *const f32 as *const _;
    let beta = &cfg.gemm.beta as *const f32 as *const _;

    let stream = c.stream().clone();
    let (a, _guard_a) = a.device_ptr(&stream);
    let (b, _guard_b) = b.device_ptr(&stream);
    let (c, _guard_c) = c.device_ptr_mut(&stream);

    unsafe {
        cudarc::cublas::result::gemm_strided_batched_ex(
            *cublas.handle(),
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            alpha,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.gemm.lda,
            cfg.stride_a,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.gemm.ldb,
            cfg.stride_b,
            beta,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            compute_type,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
}

unsafe fn gemm_strided_batched_f16(
    cublas: &cudarc::cublas::CudaBlas,
    cfg: StridedBatchedConfig<f16>,
    a: &cudarc::driver::CudaView<f16>,
    b: &cudarc::driver::CudaView<f16>,
    c: &mut CudaSlice<f16>,
) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
    use cudarc::cublas::sys;
    use cudarc::driver::DevicePtrMut;

    let alpha = cfg.gemm.alpha;
    let beta = cfg.gemm.beta;
    let alpha_f32: f32 = cfg.gemm.alpha.to_f32();
    let beta_f32: f32 = cfg.gemm.beta.to_f32();
    let (compute_type, alpha, beta) = if gemm_reduced_precision_f16() {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
            (&alpha) as *const f16 as *const _,
            (&beta) as *const f16 as *const _,
        )
    } else {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    };

    let stream = c.stream().clone();
    let (a, _guard_a) = a.device_ptr(&stream);
    let (b, _guard_b) = b.device_ptr(&stream);
    let (c, _guard_c) = c.device_ptr_mut(&stream);
    unsafe {
        cudarc::cublas::result::gemm_strided_batched_ex(
            *cublas.handle(),
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            alpha,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.gemm.lda,
            cfg.stride_a,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.gemm.ldb,
            cfg.stride_b,
            beta,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            compute_type,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
}

unsafe fn gemm_strided_batched_bf16(
    cublas: &cudarc::cublas::CudaBlas,
    cfg: StridedBatchedConfig<bf16>,
    a: &cudarc::driver::CudaView<bf16>,
    b: &cudarc::driver::CudaView<bf16>,
    c: &mut CudaSlice<bf16>,
) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
    use cudarc::cublas::sys;
    use cudarc::driver::DevicePtrMut;

    let alpha_f32: f32 = cfg.gemm.alpha.to_f32();
    let beta_f32: f32 = cfg.gemm.beta.to_f32();
    // The type for alpha and beta depends on the computeType.
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex
    let (compute_type, alpha, beta) = if gemm_reduced_precision_bf16() {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    } else {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    };

    let stream = c.stream().clone();
    let (a, _guard_a) = a.device_ptr(&stream);
    let (b, _guard_b) = b.device_ptr(&stream);
    let (c, _guard_c) = c.device_ptr_mut(&stream);
    unsafe {
        cudarc::cublas::result::gemm_strided_batched_ex(
            *cublas.handle(),
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            alpha,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.gemm.lda,
            cfg.stride_a,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.gemm.ldb,
            cfg.stride_b,
            beta,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            compute_type,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
}
