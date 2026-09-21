//! q8_0 weights on the WebGPU backend.
//!
//! The rest of the backend computes in f32 and the quantized formats live on
//! the CPU, so a linear layer here is the one place where 8-bit weights reach
//! the GPU. Decode is bound by the weight stream, so the whole win is in the
//! bytes: q8_0 spends 8.5 bits per weight against f32's 32, and the
//! dequantization rides along inside the matmul instead of materializing an
//! f32 copy.
//!
//! Weights are quantized through [`BlockQ8_0::from_float`], the same routine
//! the CPU path uses, so a layer produces the same numbers on both backends up
//! to summation order. What differs is the layout: ggml's 34-byte block
//! (`f16` scale, then 32 `i8`) is split on upload into a `u32` quant stream and
//! an `f32` scale stream, because 34 is not a multiple of 4 and WGSL has no
//! narrower load than `u32`. See `webgpu-kernels/gemv_q8.wgsl`.

use super::{Buf, Device, Pc};
use crate::quantized::GgmlType;
use crate::quantized::k_quants::{BlockQ8_0, QK8_0};
use crate::{Backend, Result, Shape, Tensor};

/// A `(n, k)` weight matrix held on the GPU as q8_0.
pub struct Q8Tensor {
    /// `n * k/4` words, row `j` at `j * k/4`; each word is 4 `i8` quants.
    qs: Buf,
    /// `n * k/32` scales, row `j` at `j * k/32`.
    scales: Buf,
    shape: Shape,
    device: Device,
}

impl Drop for Q8Tensor {
    fn drop(&mut self) {
        self.device.defer_free(self.qs.dup());
        self.device.defer_free(self.scales.dup());
    }
}

impl Q8Tensor {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Quantize a row-major `(n, k)` f32 matrix and upload it. `k` must be a
    /// multiple of the 32-value q8_0 block.
    pub fn from_f32(dev: &Device, src: &[f32], shape: &Shape) -> Result<Self> {
        let (n, k) = shape.dims2()?;
        if !k.is_multiple_of(QK8_0) {
            crate::bail!("webgpu q8_0: k = {k} is not a multiple of the {QK8_0}-value block")
        }
        if src.len() != n * k {
            crate::bail!("webgpu q8_0: expected {} elements, got {}", n * k, src.len())
        }
        let blocks_per_row = k / QK8_0;
        let words_per_row = k / 4;

        let mut blocks = vec![BlockQ8_0::zeros(); n * blocks_per_row];
        BlockQ8_0::from_float(src, &mut blocks)?;

        // Split the interleaved blocks into the two aligned streams the kernel
        // reads. The `i8` quants are packed little-endian, four to a word, so
        // word `w` of a row carries elements `4w..4w+4` -- the order
        // `unpack4` undoes with `extractBits`.
        let mut qs = vec![0u32; n * words_per_row];
        let mut scales = vec![0f32; n * blocks_per_row];
        for (bi, block) in blocks.iter().enumerate() {
            scales[bi] = block.d.to_f32();
            let word0 = bi * (QK8_0 / 4);
            for w in 0..QK8_0 / 4 {
                let q = &block.qs;
                qs[word0 + w] = u32::from_le_bytes([
                    q[4 * w] as u8,
                    q[4 * w + 1] as u8,
                    q[4 * w + 2] as u8,
                    q[4 * w + 3] as u8,
                ]);
            }
        }

        let qs_buf = dev.alloc_buffer(std::mem::size_of_val(qs.as_slice()));
        let scales_buf = dev.alloc_buffer(std::mem::size_of_val(scales.as_slice()));
        // Fresh buffers: nothing recorded can reference them yet, so the
        // uploads need no flush and land before any command that reads them.
        dev.write_buffer_u32(&qs_buf, &qs);
        dev.write_buffer_data(&scales_buf, &scales);

        Ok(Self { qs: qs_buf, scales: scales_buf, shape: shape.clone(), device: dev.clone() })
    }

    /// `dst = lhs @ self^T`, with `lhs` of shape `(.., k)` and `dst` `(.., n)`.
    pub fn matmul_t(&self, lhs: &Tensor<f32, Device>) -> Result<Tensor<f32, Device>> {
        let (n, k) = self.shape.dims2()?;
        let dims = lhs.dims();
        if dims.is_empty() {
            crate::bail!("webgpu q8_0 matmul: input tensor is a scalar")
        }
        let last = dims[dims.len() - 1];
        if last != k {
            crate::bail!("webgpu q8_0 matmul: input {dims:?} does not match weight (n={n}, k={k})")
        }
        let m = lhs.shape().elem_count() / k;

        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(n);
        let out: Tensor<f32, Device> =
            unsafe { Tensor::alloc_uninit(Shape::from(out_dims), &self.device)? };

        let push = Pc::new().usize(m).usize(n).usize(k);
        {
            let lhs_s = lhs.storage()?;
            let out_s = out.storage()?;
            let buffers = [&out_s.buffer, &lhs_s.buffer, &self.qs, &self.scales];
            // Three kernels, gated on m, because the weight stream and the
            // reduction pull in opposite directions:
            //
            //   m == 1        one row, so a four-row tile would pay four
            //                 times the reduction for one useful result.
            //   2..=16       the four-row register tile: a few passes over
            //                 the weight, no workgroup staging.
            //   m > 16        the 32x32 tile, which stages the dequantized
            //                 weight in workgroup memory and so reads it once
            //                 regardless of m. Below the crossover its 32-row
            //                 tile is mostly empty; above it, the row-block
            //                 kernel's re-reads dominate.
            //
            // Measured crossover on an M5, row-block against tiled, summed
            // over Phonon's linear shapes: 4.9x at m = 4, 3.1x at 8, 1.7x at
            // 16, 1.2x at 24, then 0.81x at 32 and 0.60x at 64 as the
            // row-block kernel's weight re-reads pile up. They cross around
            // 28 on the total, so a gate at 16 leaves a little on the table
            // and keeps a margin.
            //
            // The total hides that the right kernel is really a question of
            // shape, not of m. What starves the tiled kernel is its grid --
            // ceil(n/32) * ceil(m/32) workgroups -- so 3072->768, which gives
            // it 24 of them, prefers the row-block kernel at every m measured
            // (118 vs 647 us at m = 16, 218 vs 578 at 32), while 512->2048
            // gives it 64 and already prefers it at m = 16. A gate on the
            // tiled grid size would serve both; this one is deliberately
            // simpler.
            const ROW_BLOCK_MAX: usize = 16;
            if m == 1 {
                let groups = (super::div_ceil(n, super::GEMV_Q8_TN), 1, 1);
                self.device.dispatch_nd("gemv_q8", &buffers, &push, groups)?;
            } else if m <= ROW_BLOCK_MAX {
                let groups = (
                    super::div_ceil(n, super::GEMM_Q8_TN),
                    super::div_ceil(m, super::GEMM_Q8_MR),
                    1,
                );
                self.device.dispatch_nd("gemm_q8", &buffers, &push, groups)?;
            } else {
                let groups = (
                    super::div_ceil(n, super::GEMM_Q8_TILE),
                    super::div_ceil(m, super::GEMM_Q8_TILE),
                    1,
                );
                self.device.dispatch_nd("gemm_q8_tiled", &buffers, &push, groups)?;
            }
        }
        Ok(out)
    }
}

/// A linear layer with q8_0 weights, computed on the GPU.
pub struct Q8Linear {
    weight: Q8Tensor,
    bias: Option<Tensor<f32, Device>>,
}

impl Q8Linear {
    pub fn new(weight: Q8Tensor, bias: Option<Tensor<f32, Device>>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, xs: &Tensor<f32, Device>) -> Result<Tensor<f32, Device>> {
        let out = self.weight.matmul_t(xs)?;
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

/// `BackendQ` selecting q8_0 weights on the WebGPU backend.
///
/// Weights arrive as f32 -- from safetensors directly, or dequantized out of a
/// GGUF by the var builder -- and are quantized once, here. Re-quantizing a
/// tensor that was already q8_0 in the file reproduces its scales and quants,
/// so the GGUF round trip costs load time rather than accuracy.
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
        // A k that is not block-aligned cannot be q8_0; such layers stay f32
        // by falling back to a dequantized weight rather than failing to load.
        if !k.is_multiple_of(QK8_0) {
            crate::bail!("webgpu q8_0: layer with in_features = {k} is not a multiple of {QK8_0}")
        }
        let data = {
            let storage = w.storage()?;
            <Device as Backend>::data(&storage, shape.elem_count())?.into_owned()
        };
        let weight = Q8Tensor::from_f32(w.device(), &data, &shape)?;
        Ok(Q8Linear::new(weight, l.bias().cloned()))
    }
}
