//! Code for GGML and GGUF files
use crate::{Result, Shape};
use k_quants::*;
use std::borrow::Cow;

#[cfg(target_feature = "avx")]
#[allow(unsafe_op_in_unsafe_fn)]
pub mod avx;
pub mod ggml_file;
pub mod gguf_file;
pub mod k_quants;
#[cfg(all(feature = "kleidiai", target_arch = "aarch64"))]
pub mod kleidiai;
#[cfg(target_feature = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
pub mod neon;
pub mod repack;
#[cfg(target_feature = "simd128")]
#[allow(unsafe_op_in_unsafe_fn)]
pub mod simd128;
#[allow(unsafe_op_in_unsafe_fn)]
pub mod utils;
use half::{bf16, f16};

pub use k_quants::GgmlType;

pub struct QTensor {
    storage: QStorage,
    shape: Shape,
}

pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
}

impl QStorage {
    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &[f32]) -> Result<()> {
        match self {
            QStorage::Cpu(storage) => {
                storage.from_float(src)?;
            }
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> Result<Vec<f32>> {
        match self {
            QStorage::Cpu(storage) => storage.dequantize(elem_count),
        }
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => storage.raw_data(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    BF16,
}

impl GgmlDType {
    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::BF16 => 30,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
            Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
            Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
            Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
            Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
            Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
        }
    }
    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::BF16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<Vec<f32>>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    /// The tensor's bytes in its *canonical* on-disk layout.
    ///
    /// Storages that reorder the data for compute (see [`repack`]) must undo that here, so
    /// anything persisting the tensor writes the format its dtype claims.
    fn raw_data(&self) -> Result<Cow<'_, [u8]>> {
        let data =
            unsafe { std::slice::from_raw_parts(self.as_ptr(), self.storage_size_in_bytes()) };
        Ok(Cow::from(data))
    }
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]) -> Result<()>;
    fn size(&self) -> usize;
}

impl<T: k_quants::GgmlType + Send + Sync> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) -> Result<()> {
        T::from_float(xs, self)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<Vec<f32>> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys)?;
        Ok(ys)
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if !dims[dims.len() - 1].is_multiple_of(block_size) {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

impl QTensor {
    pub fn new<S: Into<Shape>>(storage: QStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self { storage, shape })
    }

    pub fn quantize_f32(src: &[f32], shape: &Shape, dtype: GgmlDType) -> Result<Self> {
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!("tensor size ({shape:?}) is not divisible by block size {}", block_size)
        }
        // Weights built in process get the same layout as weights read from a file: the
        // layout is keyed off the shape, not off where the blocks came from. Without this a
        // weight quantized by `QLinear::from_linear` takes a different numeric path than the
        // same weight loaded from GGUF.
        if dtype == GgmlDType::Q8_0 {
            // KleidiAI requantizes per row anyway; from f32 that is one rounding, not two.
            #[cfg(all(feature = "kleidiai", target_arch = "aarch64"))]
            if let Some(storage) = kleidiai::q8_0_storage_f32(src, shape.dims()) {
                return Ok(Self { storage, shape: shape.clone() });
            }
            let mut blocks = vec![BlockQ8_0::zeros(); elem_count / QK8_0];
            BlockQ8_0::from_float(src, &mut blocks)?;
            let storage = repack::q8_0_storage_owned(blocks, shape.dims());
            return Ok(Self { storage, shape: shape.clone() });
        }
        let mut storage = QStorage::Cpu(dtype.cpu_zeros(elem_count));
        storage.quantize(src)?;
        Ok(Self { storage, shape: shape.clone() })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dequantize(&self) -> Result<Vec<f32>> {
        self.storage.dequantize(self.shape.elem_count())
    }

    pub fn to_tensor(&self) -> Result<crate::Tensor<f32, crate::CpuDevice>> {
        let data = self.dequantize()?;
        crate::Tensor::from_vec(data, self.shape.clone(), &crate::CpuDevice)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }

    #[tracing::instrument(name = "qmatmul-t", skip_all)]
    pub fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        match &self.storage {
            QStorage::Cpu(storage) => storage.matmul_t(mkn, lhs, dst),
        }
    }
}

pub struct QLinear {
    weight: QTensor,
    bias: Option<crate::Tensor<f32, crate::CpuDevice>>,
}

impl crate::ModuleT for QLinear {
    type T = f32;
    type B = crate::CpuDevice;

    #[tracing::instrument(name = "qlinear-forward", skip_all)]
    fn forward(
        &self,
        xs: &crate::Tensor<Self::T, Self::B>,
    ) -> Result<crate::Tensor<Self::T, Self::B>> {
        use crate::error::Context;

        let weight = &self.weight;
        let bias = &self.bias;
        let (n, k) = self.weight.shape.dims2()?;
        let src_shape = xs.shape();

        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {src_shape:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().context("empty dst_shape")?;
        if last_k != k {
            crate::bail!("input tensor {src_shape:?} incompatible with {:?}", weight.shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let mut dst = vec![0.0f32; dst_shape.elem_count()];
        {
            let xs = xs.storage()?;
            self.weight.matmul_t((dst_shape.elem_count() / n, k, n), &xs, &mut dst)?;
        }

        let mut dst_t = crate::Tensor::from_vec(dst, dst_shape, &crate::CpuDevice)?;
        if let Some(bias) = bias {
            dst_t = dst_t.broadcast_add(bias)?;
        }
        Ok(dst_t)
    }
}

impl QLinear {
    pub fn new(weight: QTensor) -> Self {
        Self { weight, bias: None }
    }

    pub fn with_bias(mut self, bias: crate::Tensor<f32, crate::CpuDevice>) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn from_linear(
        linear: crate::nn::Linear<f32, crate::CpuDevice>,
        ggml_dtype: GgmlDType,
    ) -> Result<Self> {
        let weight = linear.weight();
        let src = weight.storage()?;
        let weight = QTensor::quantize_f32(&src, weight.shape(), ggml_dtype)?;
        Ok(Self { weight, bias: linear.bias().cloned() })
    }
}

macro_rules! backend_q_f32 {
    ($name:ident, $dtype:expr) => {
        #[derive(Clone)]
        pub struct $name;

        impl crate::BackendQ for $name {
            type T = f32;
            type B = crate::CpuDevice;
            type LinearQ = QLinear;

            fn from_linear(l: crate::nn::Linear<Self::T, Self::B>) -> Result<Self::LinearQ> {
                QLinear::from_linear(l, $dtype)
            }

            fn linear_load<V: std::borrow::Borrow<crate::nn::Path<Self::B>>>(
                vb: V,
                in_features: usize,
                out_features: usize,
            ) -> Result<Self::LinearQ> {
                if let Some(qt) = vb.borrow().qtensor("weight")? {
                    if qt.shape().dims() != [out_features, in_features] {
                        crate::bail!(
                            "quantized weight tensor has wrong shape {:?}, expected [{out_features}, {in_features}]",
                            qt.shape()
                        )
                    }
                    // TODO(laurent): maybe we should change the quants if they don't match Self?
                    return Ok(QLinear::new(qt));
                }
                let l = crate::nn::Linear::load(vb, in_features, out_features)?;
                Self::from_linear(l)
            }
        }
    };
}

backend_q_f32!(Q40F32, GgmlDType::Q4_0);
backend_q_f32!(Q41F32, GgmlDType::Q4_1);
backend_q_f32!(Q50F32, GgmlDType::Q5_0);
backend_q_f32!(Q51F32, GgmlDType::Q5_1);
backend_q_f32!(Q80F32, GgmlDType::Q8_0);
backend_q_f32!(Q81F32, GgmlDType::Q8_1);
backend_q_f32!(Q2kF32, GgmlDType::Q2K);
backend_q_f32!(Q3kF32, GgmlDType::Q3K);
backend_q_f32!(Q4kF32, GgmlDType::Q4K);
backend_q_f32!(Q5kF32, GgmlDType::Q5K);
backend_q_f32!(Q6kF32, GgmlDType::Q6K);
backend_q_f32!(Q8kF32, GgmlDType::Q8K);
