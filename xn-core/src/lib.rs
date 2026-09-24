#[cfg(feature = "accelerate")]
mod accelerate;

pub mod backend;
pub mod convert;
pub mod cpu_backend;
pub mod display;
pub mod dtype;
pub mod error;
pub mod inplace_ops;
pub mod models;
pub mod nn;
pub mod ops;
pub mod quantized;
pub mod safetensors;
pub mod shape;
pub mod streaming;
pub mod tensor;
pub mod tensor_view;
pub mod threadpool;
pub mod utils;

pub use backend::{Backend, ThreadSafe};
pub use dtype::{DType, DTypeQ, WithDType, WithDTypeF};
pub use error::{Context, Error, Result};
pub use shape::{D, Dim, Shape};
pub use tensor::{Tensor, TypedTensor};
pub use tensor_view::{TensorOrView, TensorView};
pub use utils::{get_num_cpus, get_num_threads, set_num_threads};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuDevice;
pub type CpuTensor<T> = Tensor<T, CpuDevice>;

pub const CPU: CpuDevice = CpuDevice;

pub(crate) use inplace_ops::{BinaryOp, UnaryOp};

#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(feature = "cuda")]
pub use cuda_backend::Device as CudaDevice;

#[cfg(feature = "metal")]
pub mod metal_backend;
#[cfg(feature = "metal")]
pub use metal_backend::Device as MetalDevice;

#[cfg(feature = "vulkan")]
pub mod vulkan_backend;
#[cfg(feature = "vulkan")]
pub use vulkan_backend::Device as VulkanDevice;

#[cfg(feature = "webgpu")]
pub mod webgpu_backend;
#[cfg(feature = "webgpu")]
pub use webgpu_backend::Device as WebGpuDevice;

pub fn with_avx() -> bool {
    cfg!(target_feature = "avx")
}

pub fn with_neon() -> bool {
    cfg!(target_feature = "neon")
}

pub fn with_dotprod() -> bool {
    cfg!(target_feature = "dotprod")
}

pub fn with_simd128() -> bool {
    cfg!(target_feature = "simd128")
}

pub fn with_f16c() -> bool {
    cfg!(target_feature = "f16c")
}

pub trait Module {
    fn forward<T: WithDType, B: Backend>(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>>;
}

impl<M: Module> Module for Option<&M> {
    fn forward<T: WithDType, B: Backend>(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

pub trait ModuleT {
    type T: WithDTypeF;
    type B: Backend;

    fn forward(&self, xs: &Tensor<Self::T, Self::B>) -> Result<Tensor<Self::T, Self::B>>;
}

impl<M: ModuleT> ModuleT for Option<&M> {
    type T = M::T;
    type B = M::B;
    fn forward(&self, xs: &Tensor<Self::T, Self::B>) -> Result<Tensor<Self::T, Self::B>> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

pub trait BackendQ: Clone + 'static {
    type T: WithDTypeF;
    type B: Backend;
    type LinearQ: ModuleT<T = Self::T, B = Self::B> + crate::backend::ThreadSafe;

    fn from_linear(l: nn::Linear<Self::T, Self::B>) -> Result<Self::LinearQ>;

    fn linear_load<V: std::borrow::Borrow<nn::Path<Self::B>>>(
        vb: V,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self::LinearQ> {
        let l = nn::Linear::load(vb, in_features, out_features)?;
        Self::from_linear(l)
    }
}

#[derive(Clone)]
pub struct Unquantized<T: WithDTypeF, B: Backend> {
    _marker1: std::marker::PhantomData<(T, B)>,
}

impl<T: WithDTypeF, B: Backend> BackendQ for Unquantized<T, B> {
    type T = T;
    type B = B;
    type LinearQ = nn::Linear<T, B>;
    fn from_linear(l: nn::Linear<Self::T, Self::B>) -> Result<Self::LinearQ> {
        Ok(l)
    }
}

pub trait WithQ {
    type Output;
    fn run<Q: BackendQ>(self, dev: Q::B) -> Result<Self::Output>;
}

pub fn run_with_device<W: WithQ>(w: W, _cpu_only: bool, _device_id: usize) -> Result<W::Output> {
    #[cfg(feature = "cuda")]
    let res = if _cpu_only {
        w.run::<Unquantized<f32, _>>(CpuDevice)
    } else {
        let dev = cuda_backend::Device::new(_device_id)?;
        w.run::<Unquantized<half::bf16, _>>(dev)
    };
    #[cfg(all(feature = "vulkan", not(feature = "cuda")))]
    let res = if _cpu_only {
        w.run::<Unquantized<f32, _>>(CpuDevice)
    } else {
        let dev = vulkan_backend::Device::new(_device_id)?;
        w.run::<Unquantized<f32, _>>(dev)
    };
    #[cfg(all(feature = "metal", not(any(feature = "cuda", feature = "vulkan"))))]
    let res = if _cpu_only {
        w.run::<Unquantized<f32, _>>(CpuDevice)
    } else {
        let dev = metal_backend::Device::new(_device_id)?;
        w.run::<Unquantized<f32, _>>(dev)
    };
    #[cfg(all(
        feature = "webgpu",
        not(any(feature = "cuda", feature = "vulkan", feature = "metal"))
    ))]
    let res = if _cpu_only {
        w.run::<Unquantized<f32, _>>(CpuDevice)
    } else {
        let dev = webgpu_backend::Device::new(_device_id)?;
        w.run::<Unquantized<f32, _>>(dev)
    };
    #[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "metal", feature = "webgpu")))]
    let res = w.run::<Unquantized<f32, _>>(CpuDevice);
    res
}

pub struct Runner {
    cpu_only: bool,
    dtype: DTypeQ,
}

impl Runner {
    pub fn new() -> Self {
        Self { cpu_only: false, dtype: DTypeQ::BF16 }
    }

    pub fn cpu_only(mut self, cpu_only: bool) -> Self {
        self.cpu_only = cpu_only;
        self
    }

    pub fn dtype(mut self, dtype: DTypeQ) -> Self {
        self.dtype = dtype;
        self
    }

    /// Run on the CPU backend, using the quantized kernels when the dtype
    /// calls for them.
    #[cfg(not(feature = "cuda"))]
    fn run_cpu<W: WithQ>(&self, w: W) -> Result<W::Output> {
        match self.dtype {
            DTypeQ::Fp8 | DTypeQ::Fp8PerToken => {
                Err(Error::msg("FP8 quantization is not supported on CPU"))
            }
            DTypeQ::F32 => w.run::<Unquantized<f32, _>>(CpuDevice),
            DTypeQ::Q4_0 => w.run::<crate::quantized::Q40F32>(CpuDevice),
            DTypeQ::Q4_1 => w.run::<crate::quantized::Q41F32>(CpuDevice),
            DTypeQ::Q5_0 => w.run::<crate::quantized::Q50F32>(CpuDevice),
            DTypeQ::Q5_1 => w.run::<crate::quantized::Q51F32>(CpuDevice),
            DTypeQ::Q8_0 => w.run::<crate::quantized::Q80F32>(CpuDevice),
            DTypeQ::Q8_1 => w.run::<crate::quantized::Q81F32>(CpuDevice),
            DTypeQ::Q2K => w.run::<crate::quantized::Q2kF32>(CpuDevice),
            DTypeQ::Q3K => w.run::<crate::quantized::Q3kF32>(CpuDevice),
            DTypeQ::Q4K => w.run::<crate::quantized::Q4kF32>(CpuDevice),
            DTypeQ::Q5K => w.run::<crate::quantized::Q5kF32>(CpuDevice),
            DTypeQ::Q6K => w.run::<crate::quantized::Q6kF32>(CpuDevice),
            DTypeQ::Q8K => w.run::<crate::quantized::Q8kF32>(CpuDevice),
            DTypeQ::F16 | DTypeQ::BF16 => {
                Err(Error::msg(format!("{:?} is not yet supported on CPU", self.dtype)))
            }
        }
    }

    pub fn run<W: WithQ>(self, w: W, _device_id: usize) -> Result<W::Output> {
        #[cfg(feature = "cuda")]
        let res = if self.cpu_only {
            w.run::<Unquantized<f32, _>>(CpuDevice)
        } else {
            let dev = cuda_backend::Device::new(_device_id)?;
            match self.dtype {
                DTypeQ::Fp8 => w.run::<cuda_backend::quantization::Fp8ScalePerTensor>(dev),
                DTypeQ::Fp8PerToken => w.run::<cuda_backend::quantization::Fp8ScalePerToken>(dev),
                DTypeQ::F16 => w.run::<Unquantized<half::f16, _>>(dev),
                DTypeQ::BF16 => w.run::<Unquantized<half::bf16, _>>(dev),
                DTypeQ::F32 => w.run::<Unquantized<f32, _>>(dev),
                DTypeQ::Q4_0
                | DTypeQ::Q4_1
                | DTypeQ::Q5_0
                | DTypeQ::Q5_1
                | DTypeQ::Q8_0
                | DTypeQ::Q8_1
                | DTypeQ::Q2K
                | DTypeQ::Q3K
                | DTypeQ::Q4K
                | DTypeQ::Q5K
                | DTypeQ::Q6K
                | DTypeQ::Q8K => Err(Error::msg(format!(
                    "{:?} quantization is only supported on CPU",
                    self.dtype
                ))),
            }
        };
        #[cfg(all(feature = "vulkan", not(feature = "cuda")))]
        let res = if self.cpu_only {
            self.run_cpu(w)
        } else {
            // The Vulkan backend computes in f32, f16 or bf16 (device
            // permitting). Quantized formats stay on CPU.
            match self.dtype {
                DTypeQ::F32 => {
                    let dev = vulkan_backend::Device::new(_device_id)?;
                    w.run::<Unquantized<f32, _>>(dev)
                }
                DTypeQ::F16 => {
                    let dev = vulkan_backend::Device::new(_device_id)?;
                    if !dev.supports_f16() {
                        Err(Error::msg("vulkan device does not support f16"))
                    } else {
                        w.run::<Unquantized<half::f16, _>>(dev)
                    }
                }
                DTypeQ::BF16 => {
                    let dev = vulkan_backend::Device::new(_device_id)?;
                    if !dev.supports_bf16() {
                        Err(Error::msg("vulkan device does not support bf16"))
                    } else {
                        w.run::<Unquantized<half::bf16, _>>(dev)
                    }
                }
                _ => self.run_cpu(w),
            }
        };
        #[cfg(all(feature = "metal", not(any(feature = "cuda", feature = "vulkan"))))]
        let res = if self.cpu_only {
            self.run_cpu(w)
        } else {
            // The Metal backend computes in f32 with f16 or bf16 storage also
            // supported. Quantized formats stay on CPU.
            match self.dtype {
                DTypeQ::F32 => {
                    let dev = metal_backend::Device::new(_device_id)?;
                    w.run::<Unquantized<f32, _>>(dev)
                }
                DTypeQ::F16 => {
                    let dev = metal_backend::Device::new(_device_id)?;
                    w.run::<Unquantized<half::f16, _>>(dev)
                }
                DTypeQ::BF16 => {
                    let dev = metal_backend::Device::new(_device_id)?;
                    w.run::<Unquantized<half::bf16, _>>(dev)
                }
                _ => self.run_cpu(w),
            }
        };
        #[cfg(all(
            feature = "webgpu",
            not(any(feature = "cuda", feature = "vulkan", feature = "metal"))
        ))]
        let res = if self.cpu_only {
            self.run_cpu(w)
        } else {
            // The WebGPU backend computes in f32, with q8_0 weights
            // dequantized inside the matmul. Other formats stay on CPU.
            match self.dtype {
                DTypeQ::F32 => {
                    let dev = webgpu_backend::Device::new(_device_id)?;
                    w.run::<Unquantized<f32, _>>(dev)
                }
                DTypeQ::Q8_0 => {
                    let dev = webgpu_backend::Device::new(_device_id)?;
                    w.run::<webgpu_backend::quantization::Q8F32>(dev)
                }
                _ => self.run_cpu(w),
            }
        };
        #[cfg(not(any(
            feature = "cuda",
            feature = "vulkan",
            feature = "metal",
            feature = "webgpu"
        )))]
        let res = self.run_cpu(w);
        res
    }
}

impl std::default::Default for Runner {
    fn default() -> Self {
        Self::new()
    }
}
