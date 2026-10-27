//! q8_0 weights on the CUDA backend.
//!
//! Decoding a token is bound by reading the weights, and q8_0 holds a weight in about a byte,
//! against two for bf16 and four for f32. A linear layer keeps its weight as q8_0 on the
//! device: matmuls of a few rows, as at decode, dequantize it inside the kernel, and larger ones
//! dequantize it into a dense temporary for cuBLAS.
//!
//! Weights are quantized with [`BlockQ8_0::from_float`], as on the CPU, or taken from a GGUF's
//! q8_0 blocks as they are. The 34-byte block (an f16 scale, then 32 `i8` quants) is split on
//! upload into a quant stream and a scale stream, so that the kernel reads the quants 16 bytes
//! at a time.

use super::{Device, PTXModule, kernel_name};
use crate::quantized::GgmlType;
use crate::quantized::k_quants::{BlockQ8_0, QK8_0};
use crate::quantized::{GgmlDType, QTensor};
use crate::{Result, Shape, Tensor, WithDTypeF};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use half::f16;

/// Rows of input the in-kernel matmul is built for: `matmul_q8`'s `_m1` to `_m16`
/// instantiations in `reduce.cu`. More rows go through a dense copy of the weight.
const KERNEL_MAX_ROWS: usize = 16;

/// A `(n, k)` weight matrix held on the device as q8_0.
pub struct Q8Tensor {
    /// `n * k` quants, row `j` at `j * k`.
    qs: CudaSlice<i8>,
    /// `n * k / 32` scales, row `j` at `j * k / 32`.
    scales: CudaSlice<f16>,
    shape: Shape,
    device: Device,
}

impl Q8Tensor {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Quantize a row-major `(n, k)` matrix. `k` must be a multiple of the 32-value block.
    pub fn from_f32(dev: &Device, src: &[f32], shape: &Shape) -> Result<Self> {
        let (n, k) = shape.dims2()?;
        if !k.is_multiple_of(QK8_0) {
            crate::bail!("cuda q8_0: k = {k} is not a multiple of the {QK8_0}-value block")
        }
        if src.len() != n * k {
            crate::bail!("cuda q8_0: expected {} elements, got {}", n * k, src.len())
        }
        let mut blocks = vec![BlockQ8_0::zeros(); n * k / QK8_0];
        BlockQ8_0::from_float(src, &mut blocks)?;
        let scales = blocks.iter().map(|b| b.d).collect::<Vec<_>>();
        let quants = blocks.iter().flat_map(|b| b.qs).collect::<Vec<_>>();
        Self::upload(dev, &quants, &scales, shape)
    }

    /// Bytes one q8_0 block occupies in a GGUF: an f16 scale then 32 `i8` quants.
    const BLOCK_BYTES: usize = 2 + QK8_0;

    /// Take q8_0 blocks straight from a GGUF.
    pub fn from_q8_0(qt: &QTensor, dev: &Device) -> Result<Self> {
        if qt.dtype() != GgmlDType::Q8_0 {
            crate::bail!("cuda q8_0: expected a q8_0 tensor, got {:?}", qt.dtype())
        }
        let shape = qt.shape().clone();
        let (n, k) = shape.dims2()?;
        if !k.is_multiple_of(QK8_0) {
            crate::bail!("cuda q8_0: k = {k} is not a multiple of the {QK8_0}-value block")
        }
        let n_blocks = n * k / QK8_0;
        let bytes = qt.data()?;
        let want = n_blocks * Self::BLOCK_BYTES;
        if bytes.len() != want {
            crate::bail!(
                "cuda q8_0: [{n}, {k}] wants {want} bytes of blocks, file has {}",
                bytes.len()
            )
        }
        let mut scales = Vec::with_capacity(n_blocks);
        let mut quants = Vec::with_capacity(n_blocks * QK8_0);
        for block in bytes.as_chunks::<{ Self::BLOCK_BYTES }>().0 {
            scales.push(f16::from_le_bytes([block[0], block[1]]));
            quants.extend(block[2..].iter().map(|&q| q as i8));
        }
        Self::upload(dev, &quants, &scales, &shape)
    }

    fn upload(dev: &Device, quants: &[i8], scales: &[f16], shape: &Shape) -> Result<Self> {
        let qs = dev.stream().clone_htod(quants)?;
        let scales = dev.stream().clone_htod(scales)?;
        Ok(Self { qs, scales, shape: shape.clone(), device: dev.clone() })
    }

    /// The weight as a dense `(n, k)` matrix.
    pub fn dequantize<T: WithDTypeF>(&self) -> Result<Tensor<T, Device>> {
        let (n, k) = self.shape.dims2()?;
        let out: Tensor<T, Device> =
            unsafe { Tensor::alloc_uninit(self.shape.clone(), &self.device)? };
        {
            let mut dst = out.storage_mut()?;
            let func = self.device.get_func(&kernel_name::<T>("dequant_q8"), PTXModule::Reduce)?;
            let numel = n * k;
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            let k = k as i32;
            let mut launch_args = self.device.stream.launch_builder(&func);
            launch_args.arg(&self.qs);
            launch_args.arg(&self.scales);
            launch_args.arg(&mut dst.data);
            launch_args.arg(&numel);
            launch_args.arg(&k);
            unsafe { launch_args.launch(cfg) }?;
        }
        Ok(out)
    }

    /// `lhs @ self^T`, with `lhs` of shape `(.., k)` and the result `(.., n)`.
    pub fn matmul_t<T: WithDTypeF>(&self, lhs: &Tensor<T, Device>) -> Result<Tensor<T, Device>> {
        let (n, k) = self.shape.dims2()?;
        let dims = lhs.dims();
        if dims.last() != Some(&k) {
            crate::bail!("cuda q8_0 matmul: input {dims:?} does not match weight (n={n}, k={k})")
        }
        let m = lhs.shape().elem_count() / k;
        // The kernel reads the input 16 bytes at a time, so its rows must start aligned.
        let aligned = (k * std::mem::size_of::<T>()).is_multiple_of(16);
        if m > KERNEL_MAX_ROWS || !aligned || n > i32::MAX as usize || k > i32::MAX as usize {
            return crate::ops::matmul_t(lhs, &self.dequantize::<T>()?);
        }
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(n);
        let out: Tensor<T, Device> =
            unsafe { Tensor::alloc_uninit(Shape::from(out_dims), &self.device)? };
        {
            let x = lhs.storage()?;
            let mut dst = out.storage_mut()?;
            let kname = format!("{}_m{}", kernel_name::<T>("matmul_q8"), m.next_power_of_two());
            let func = self.device.get_func(&kname, PTXModule::Reduce)?;
            // Four warps a block, two rows of the weight a warp.
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(8), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let (m, n, k) = (m as i32, n as i32, k as i32);
            let mut launch_args = self.device.stream.launch_builder(&func);
            launch_args.arg(&x.data);
            launch_args.arg(&self.qs);
            launch_args.arg(&self.scales);
            launch_args.arg(&mut dst.data);
            launch_args.arg(&m);
            launch_args.arg(&n);
            launch_args.arg(&k);
            launch_args.arg(&k);
            unsafe { launch_args.launch(cfg) }?;
        }
        Ok(out)
    }
}

/// A linear layer with q8_0 weights.
pub struct Q8Linear<T: WithDTypeF> {
    weight: Q8Tensor,
    bias: Option<Tensor<T, Device>>,
}

impl<T: WithDTypeF> Q8Linear<T> {
    pub fn new(weight: Q8Tensor, bias: Option<Tensor<T, Device>>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, xs: &Tensor<T, Device>) -> Result<Tensor<T, Device>> {
        let out = self.weight.matmul_t(xs)?;
        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}

impl<T: WithDTypeF> crate::ModuleT for Q8Linear<T> {
    type T = T;
    type B = Device;
    fn forward(&self, xs: &Tensor<T, Device>) -> Result<Tensor<T, Device>> {
        self.forward(xs)
    }
}

/// `BackendQ` with q8_0 weights for the linear layers and `T` for everything else.
#[derive(Clone, Copy)]
pub struct Q8<T: WithDTypeF>(std::marker::PhantomData<T>);

impl<T: WithDTypeF> crate::BackendQ for Q8<T> {
    type T = T;
    type B = Device;
    type LinearQ = Q8Linear<T>;

    /// Take a GGUF's q8_0 blocks as they are; quantize anything else.
    fn linear_load<V: std::borrow::Borrow<crate::nn::Path<Device>>>(
        vb: V,
        in_features: usize,
        out_features: usize,
    ) -> Result<Q8Linear<T>> {
        let vb = vb.borrow();
        match vb.qtensor("weight")? {
            Some(qt) if qt.dtype() == GgmlDType::Q8_0 => {
                let (n, k) = qt.shape().dims2()?;
                if (n, k) != (out_features, in_features) {
                    crate::bail!(
                        "cuda q8_0: weight is [{n}, {k}], expected [{out_features}, {in_features}]"
                    )
                }
                // Weight only, like `nn::Linear::load`.
                Ok(Q8Linear::new(Q8Tensor::from_q8_0(&qt, vb.device())?, None))
            }
            _ => Self::from_linear(crate::nn::Linear::load(vb, in_features, out_features)?),
        }
    }

    fn from_linear(l: crate::nn::Linear<T, Device>) -> Result<Q8Linear<T>> {
        let w = l.weight();
        let data: Vec<f32> = w.to_vec()?.into_iter().map(|v| v.to_f32()).collect();
        let weight = Q8Tensor::from_f32(w.device(), &data, w.shape())?;
        Ok(Q8Linear::new(weight, l.bias().cloned()))
    }
}
