//! q8_0 weights on the Vulkan backend.
//!
//! The rest of the backend computes in f32 and the quantized formats live on
//! the CPU, so a linear layer here is the one place where 8-bit weights reach
//! the GPU. Decode is bound by the weight stream, so the whole win is in the
//! bytes: q8_0 spends 8.5 bits per weight against f32's 32, and the
//! dequantization rides along inside the matmul instead of materializing an
//! f32 copy.
//!
//! Weights come either straight out of a GGUF's q8_0 blocks or, for f32
//! checkpoints, through [`BlockQ8_0::from_float`], the same routine the CPU
//! path uses, so a layer produces the same numbers on both backends up to
//! summation order. What differs is the layout: ggml's 34-byte block (`f16`
//! scale, then 32 `i8`) is split on upload into a byte stream of quants and an
//! `f32` stream of scales, because 34 is not a multiple of 4 and the shaders
//! read the quants four at a time as `uint`. See `vulkan-kernels/gemv_q8.comp`.

use super::{Device, Pc, div_ceil};
use crate::quantized::k_quants::QK8_0;
use crate::quantized::{GgmlDType, QTensor, quantize_split_q8_0, split_q8_0};
use crate::{Backend, Result, Shape, Tensor};

use super::row_block_kernel;

/// Largest `m` the dequantize-fused row-block kernels take, walking `m` in
/// blocks of 16 and re-reading the weight once per block. Past it the weight
/// is dequantized once and multiplied by the backend's f32 GEMM, as the CUDA
/// backend does with cuBLAS.
const ROW_BLOCK_MAX: usize = 64;

/// A `(n, k)` weight matrix held on the GPU as q8_0.
pub struct Q8Tensor {
    /// `n * k` int8 quants, row `j` at `j * k`; each aligned group of four is
    /// one `uint` to the shaders.
    qs: Tensor<u8, Device>,
    /// `n * k/32` scales, row `j` at `j * k/32`.
    scales: Tensor<f32, Device>,
    shape: Shape,
}

impl Q8Tensor {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dims(shape: &Shape) -> Result<(usize, usize)> {
        let (n, k) = shape.dims2()?;
        if !k.is_multiple_of(QK8_0) {
            crate::bail!("vulkan q8_0: k = {k} is not a multiple of the {QK8_0}-value block")
        }
        Ok((n, k))
    }

    /// Quantize a row-major `(n, k)` f32 matrix and upload it. `k` must be a
    /// multiple of the 32-value q8_0 block.
    pub fn from_f32(dev: &Device, src: &[f32], shape: &Shape) -> Result<Self> {
        let (n, k) = Self::dims(shape)?;
        if src.len() != n * k {
            crate::bail!("vulkan q8_0: expected {} elements, got {}", n * k, src.len())
        }
        let (qs, scales) = quantize_split_q8_0(src)?;
        Self::upload(dev, qs, scales, shape)
    }

    /// Upload a quantized tensor as read from a GGUF. q8_0 blocks are uploaded
    /// as they are; any other format is dequantized and re-quantized to q8_0.
    pub fn from_qtensor(dev: &Device, qt: &QTensor) -> Result<Self> {
        let shape = qt.shape();
        match qt.dtype() {
            GgmlDType::Q8_0 => {
                Self::dims(shape)?;
                let (qs, scales) = split_q8_0(&qt.data()?)?;
                Self::upload(dev, qs, scales, shape)
            }
            _ => Self::from_f32(dev, &qt.dequantize()?, shape),
        }
    }

    fn upload(dev: &Device, qs: Vec<u8>, scales: Vec<f32>, shape: &Shape) -> Result<Self> {
        // Fresh buffers: nothing recorded can reference them yet, so the
        // host-side copy lands before any dispatch that reads them.
        let qs = Tensor::from_vec(qs, (shape.elem_count(),), dev)?;
        let scales = Tensor::from_vec(scales, (shape.elem_count() / QK8_0,), dev)?;
        Ok(Self { qs, scales, shape: shape.clone() })
    }

    /// The weight back in f32, `(n, k)`.
    pub fn dequantize(&self) -> Result<Tensor<f32, Device>> {
        let dev = self.qs.device();
        let out: Tensor<f32, Device> = unsafe { Tensor::alloc_uninit(self.shape.clone(), dev)? };
        let words = self.shape.elem_count() / 4;
        let push = Pc::new().usize(words);
        {
            let out_s = out.storage()?;
            let qs_s = self.qs.storage()?;
            let sc_s = self.scales.storage()?;
            let buffers = [out_s.buffer, qs_s.buffer, sc_s.buffer];
            dev.dispatch_nd("dequant_q8", &buffers, &push, (div_ceil(words, 256), 1, 1))?;
        }
        Ok(out)
    }

    /// `dst = lhs @ self^T`, with `lhs` of shape `(.., k)` and `dst` `(.., n)`.
    pub fn matmul_t(&self, lhs: &Tensor<f32, Device>) -> Result<Tensor<f32, Device>> {
        let (n, k) = self.shape.dims2()?;
        let dims = lhs.dims();
        if dims.is_empty() {
            crate::bail!("vulkan q8_0 matmul: input tensor is a scalar")
        }
        let last = dims[dims.len() - 1];
        if last != k {
            crate::bail!("vulkan q8_0 matmul: input {dims:?} does not match weight (n={n}, k={k})")
        }
        let m = lhs.shape().elem_count() / k;
        if m > ROW_BLOCK_MAX {
            let w = self.dequantize()?;
            return lhs.matmul_t(&w);
        }

        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(n);
        let dev = lhs.device();
        let out: Tensor<f32, Device> = unsafe { Tensor::alloc_uninit(Shape::from(out_dims), dev)? };
        if m == 0 {
            return Ok(out);
        }

        let push = Pc::new().usize(m).usize(n).usize(k);
        {
            let lhs_s = lhs.storage()?;
            let out_s = out.storage()?;
            let qs_s = self.qs.storage()?;
            let sc_s = self.scales.storage()?;
            let buffers = [out_s.buffer, lhs_s.buffer, qs_s.buffer, sc_s.buffer];
            let label =
                |kernel: &str| dev.profile_enabled.then(|| format!("{kernel} m{m} n{n} k{k}"));
            // One subgroup per output column where subgroup arithmetic
            // exists: gemv_q8_sg.comp at m == 1, else gemm_q8_sg.comp with
            // the smallest row block that covers m (capped at 16, above which
            // the grid walks m in blocks of 16 and the weight is re-read once
            // per block). Without subgroups, the shared-memory kernels.
            match super::WORKGROUP_SIZE.checked_div(dev.subgroup_size()) {
                Some(cols_per_wg) if m == 1 => {
                    let groups = (div_ceil(n, cols_per_wg), 1, 1);
                    dev.dispatch_labeled(
                        "gemv_q8_sg",
                        label("gemv_q8_sg"),
                        &buffers,
                        &push,
                        groups,
                    )?;
                }
                Some(cols_per_wg) => {
                    let (kernel, mr) = row_block_kernel("gemm_q8_sg", m, 16);
                    let groups = (div_ceil(n, cols_per_wg), div_ceil(m, mr), 1);
                    dev.dispatch_labeled(&kernel, label(&kernel), &buffers, &push, groups)?;
                }
                None if m == 1 => {
                    let groups = (div_ceil(n, 4), 1, 1);
                    dev.dispatch_labeled("gemv_q8", label("gemv_q8"), &buffers, &push, groups)?;
                }
                None => {
                    let groups = (div_ceil(n, 4), div_ceil(m, 4), 1);
                    dev.dispatch_labeled("gemm_q8", label("gemm_q8"), &buffers, &push, groups)?;
                }
            }
        }
        Ok(out)
    }
}

/// The weight of a [`Q8Linear`]: q8_0 when the layer is block-aligned or was
/// quantized in the file, f32 when the file kept it that way.
enum Weight {
    Q8(Q8Tensor),
    F32(Tensor<f32, Device>),
}

/// A linear layer with q8_0 weights, computed on the GPU.
pub struct Q8Linear {
    weight: Weight,
    bias: Option<Tensor<f32, Device>>,
}

impl Q8Linear {
    pub fn new(weight: Q8Tensor, bias: Option<Tensor<f32, Device>>) -> Self {
        Self { weight: Weight::Q8(weight), bias }
    }

    pub fn forward(&self, xs: &Tensor<f32, Device>) -> Result<Tensor<f32, Device>> {
        let out = match &self.weight {
            Weight::Q8(w) => w.matmul_t(xs)?,
            Weight::F32(w) => xs.matmul_t(w)?,
        };
        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}

impl crate::ModuleT for Q8Linear {
    type T = f32;
    type B = Device;
    fn forward(&self, xs: &Tensor<Self::T, Self::B>) -> Result<Tensor<Self::T, Self::B>> {
        self.forward(xs)
    }
}

/// `BackendQ` selecting q8_0 weights on the Vulkan backend.
///
/// From a GGUF, q8_0 tensors are uploaded block for block and tensors the file
/// left in f32 stay f32, which mirrors the CPU `Q80F32` path. From an f32
/// checkpoint, every linear loaded through this type is quantized once, here.
#[derive(Clone, Copy)]
pub struct Q8F32;

impl crate::BackendQ for Q8F32 {
    type T = f32;
    type B = Device;
    type LinearQ = Q8Linear;

    fn from_linear(l: crate::nn::Linear<f32, Device>) -> Result<Q8Linear> {
        let w = l.weight();
        let shape = w.shape().clone();
        let (_n, k) = shape.dims2()?;
        // A k that is not block-aligned cannot be q8_0; such layers stay f32.
        if !k.is_multiple_of(QK8_0) {
            return Ok(Q8Linear { weight: Weight::F32(w.clone()), bias: l.bias().cloned() });
        }
        let data = {
            let storage = w.storage()?;
            <Device as Backend>::data(&storage, shape.elem_count())?.into_owned()
        };
        let weight = Q8Tensor::from_f32(w.device(), &data, &shape)?;
        Ok(Q8Linear::new(weight, l.bias().cloned()))
    }

    fn linear_load<V: std::borrow::Borrow<crate::nn::Path<Device>>>(
        vb: V,
        in_features: usize,
        out_features: usize,
    ) -> Result<Q8Linear> {
        let vb = vb.borrow();
        if let Some(qt) = vb.qtensor("weight")? {
            if qt.shape().dims() != [out_features, in_features] {
                crate::bail!(
                    "quantized weight tensor has wrong shape {:?}, expected [{out_features}, {in_features}]",
                    qt.shape()
                )
            }
            let weight = match qt.dtype() {
                GgmlDType::F32 => Weight::F32(Tensor::from_vec(
                    qt.dequantize()?,
                    (out_features, in_features),
                    vb.device(),
                )?),
                _ => Weight::Q8(Q8Tensor::from_qtensor(vb.device(), &qt)?),
            };
            // GGUF linears carry no bias, as on the CPU path.
            return Ok(Q8Linear { weight, bias: None });
        }
        Self::from_linear(crate::nn::Linear::load(vb, in_features, out_features)?)
    }
}
