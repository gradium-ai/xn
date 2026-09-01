//! `q8_0` weights for the WebGPU backend.
//!
//! Status on Apple M5 / Metal: the kernel is correct and, measured in isolation
//! through the backend, 2.25x faster than the f16 dense path on this model's
//! shapes (see the `webgpu_q8_cost` example). End to end it does **not** pay
//! yet. Quantizing all 48 eligible layers makes the frame ~1.5x slower, and the
//! damage lands mostly on the Mimi decoder -- unrelated f32 work, in a later
//! submit, whose dispatch and workgroup counts are byte-identical between runs.
//! Quantizing a subset lands within run-to-run noise of plain f16. Ruled out so
//! far: the quantized buffers merely being resident (holding them while running
//! the dense path is exactly as fast as f16), the number of distinct buffers a
//! pass references (merging scales and quants into one allocation changed
//! nothing), thermal drift (no monotonic trend across iterations), dispatch
//! count (identical), and memory footprint (q8 uses less than f16). What remains
//! fits a power or clock effect: the quantized kernel reads its weights ~3x
//! faster than the dense path, and the cost appears in whatever runs next.
//! `XN_WEBGPU_Q8_LAYERS` and `XN_WEBGPU_Q8_DENSE` below exist to keep bisecting
//! this.
//!
//! Only the layers the model routes through `BackendQ::LinearQ` are quantized --
//! for phonon that is the flow-LM attention projections and both transformers'
//! feed-forward pairs, roughly 89M of the ~111M parameters a decode frame reads.
//! Everything else stays in the unquantized dtype `T`, so this composes with the
//! f16 path rather than replacing it.
//!
//! ggml's `q8_0` block is `{f16 d; i8 qs[32]}`, 34 bytes. WGSL storage buffers
//! are word-addressed and 34 is not a multiple of 4, so a block's scale and
//! quants straddle word boundaries differently in every block and the layout
//! cannot be read from a shader at all. The weights are therefore split at load
//! time into quants packed four per `u32`, which the kernel reads as
//! `vec4<u32>` (16 weights per load), followed by one `f32` scale per block --
//! both in a single allocation, because the cost of a compute pass grows sharply
//! with the number of distinct buffers it references and a decode frame issues
//! ~48 quantized dispatches into one pass. Quantization itself
//! goes through [`BlockQ8_0::from_float`], the same code the CPU path uses, so
//! the two produce identical numbers.

use super::{Device, GEMV_TPC_COLS, Pc, div_ceil};
use crate::quantized::GgmlType;
use crate::quantized::k_quants::{BlockQ8_0, QK8_0};
use crate::{Result, Tensor, WithDTypeF};

/// Bytes of quantized weight per 32-value block in the split layout: 32 quants
/// packed four per `u32`, plus one `f32` scale.
const BYTES_PER_BLOCK: usize = QK8_0 + 4;

/// A `[n, k]` weight matrix in the WGSL-addressable `q8_0` layout.
///
/// Buffers are created directly rather than taken from the device's recycling
/// pool: these live for the life of the model, so pooling them would only keep
/// large allocations pinned in a free list.
pub struct Q8Tensor {
    /// Quants then scales in one allocation: `n * k / 4` words of quants
    /// (row-major, four per `u32`), then `n * k / 32` `f32` scales. `n * k` is a
    /// multiple of 16 because `k` is a multiple of the 32-value block size, so
    /// the scales land 16-byte aligned and the one buffer can be read both as
    /// `array<vec4<u32>>` for the quants and as `array<f32>` for the scales.
    data: wgpu::Buffer,
    /// Index of the first scale with the buffer viewed as `array<f32>`.
    scale_word_offset: usize,
    n: usize,
    k: usize,
}

impl Q8Tensor {
    /// Quantize a dense `[n, k]` weight. `k` must be a multiple of 32, the
    /// `q8_0` block size.
    pub fn quantize<T: WithDTypeF>(w: &Tensor<T, Device>) -> Result<Self> {
        let (n, k) = w.shape().dims2()?;
        if !k.is_multiple_of(QK8_0) {
            crate::bail!("webgpu q8_0: k={k} is not a multiple of the {QK8_0}-value block size");
        }
        let dev = w.device().clone();
        let host: Vec<f32> = w.to_vec()?.into_iter().map(|v| v.to_f32()).collect();

        let blocks_per_row = k / QK8_0;
        let quant_words = n * k / 4;
        let mut quants = vec![0u32; quant_words];
        let mut scales = vec![0f32; n * blocks_per_row];
        let mut row = vec![BlockQ8_0::zeros(); blocks_per_row];
        for j in 0..n {
            BlockQ8_0::from_float(&host[j * k..(j + 1) * k], &mut row)?;
            for (b, blk) in row.iter().enumerate() {
                scales[j * blocks_per_row + b] = blk.d.to_f32();
                // Four signed bytes per word, little-endian, matching the
                // shader's `unpack4xI8`.
                for w4 in 0..QK8_0 / 4 {
                    let q = &blk.qs[w4 * 4..w4 * 4 + 4];
                    quants[j * (k / 4) + b * (QK8_0 / 4) + w4] =
                        u32::from_le_bytes([q[0] as u8, q[1] as u8, q[2] as u8, q[3] as u8]);
                }
            }
        }
        Ok(Self {
            data: dev.weight_buffer_2(bytemuck_u32(&quants), bytemuck_f32(&scales)),
            scale_word_offset: quant_words,
            n,
            k,
        })
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    /// Quantized bytes held on the device, against `n * k * 4` for f32 weights.
    pub fn size_in_bytes(&self) -> usize {
        self.n * self.k / QK8_0 * BYTES_PER_BLOCK
    }
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A linear layer with `q8_0` weights and an optional bias in dtype `T`.
///
/// The bias is folded into the matmul kernel rather than applied as a separate
/// `broadcast_add`, which saves both a dispatch and a full read-modify-write of
/// the output for every quantized layer.
pub struct Q8Linear<T: WithDTypeF> {
    weight: Q8Tensor,
    bias: Option<Tensor<T, Device>>,
    device: Device,
    /// Diagnostic, kept while the end-to-end regression is unexplained. When set
    /// the dense weight is retained and used for the forward, so the quantized
    /// buffers stay resident but the quantized kernel never runs -- which
    /// separates "the q8 buffers cost something" from "running the q8 pipeline
    /// costs something". Measured: the buffers cost nothing.
    dense: Option<crate::nn::Linear<T, Device>>,
}

impl<T: WithDTypeF> Q8Linear<T> {
    pub fn new(weight: Q8Tensor, bias: Option<Tensor<T, Device>>, device: Device) -> Self {
        Self { weight, bias, device, dense: None }
    }

    /// Whether this layer should keep its dense weight and skip the quantized
    /// kernel. Both switches are diagnostics for the regression described at the
    /// top of this module, not tuning knobs to rely on.
    fn dense_mode() -> bool {
        if std::env::var("XN_WEBGPU_Q8_DENSE").is_ok_and(|v| v == "1") {
            return true;
        }
        // XN_WEBGPU_Q8_LAYERS=<n> quantizes only the first n eligible layers in
        // load order, so the cost can be measured against how many quantized
        // dispatches a frame issues. The degradation of later f32 work grows
        // roughly with that count.
        match std::env::var("XN_WEBGPU_Q8_LAYERS").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(limit) => {
                static SEEN: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= limit
            }
            None => false,
        }
    }

    pub fn forward(&self, xs: &Tensor<T, Device>) -> Result<Tensor<T, Device>> {
        if let Some(dense) = &self.dense {
            return dense.forward(xs);
        }
        let (n, k) = self.weight.dims();
        let src_shape = xs.shape();
        if src_shape.rank() < 2 {
            crate::bail!("webgpu q8_0 linear: input {src_shape:?} has rank < 2")
        }
        let mut dst_dims = src_shape.dims().to_vec();
        let last = dst_dims.pop().expect("rank >= 2");
        if last != k {
            crate::bail!("webgpu q8_0 linear: input {src_shape:?} incompatible with [{n}, {k}]")
        }
        // Rows of the flattened batch; the kernel walks them MR at a time.
        let m: usize = dst_dims.iter().product();
        dst_dims.push(n);

        // The kernel reads `lhs` through a vec4 view. A `Tensor` (unlike a
        // `TensorView`) is contiguous over the whole of its storage at offset 0,
        // so that view is always valid and no staging copy is needed.
        let dst = unsafe { Tensor::<T, Device>::alloc_uninit(dst_dims, &self.device) }?;
        self.device.q8_matmul(&dst, xs, &self.weight, self.bias.as_ref(), m, n, k)?;
        Ok(dst)
    }
}

impl<T: WithDTypeF> crate::ModuleT for Q8Linear<T> {
    type T = T;
    type B = Device;
    fn forward(&self, xs: &Tensor<Self::T, Self::B>) -> Result<Tensor<Self::T, Self::B>> {
        self.forward(xs)
    }
}

macro_rules! backend_q8 {
    ($name:ident, $t:ty, $doc:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy)]
        pub struct $name;

        impl crate::BackendQ for $name {
            type T = $t;
            type B = Device;
            type LinearQ = Q8Linear<$t>;

            fn from_linear(l: crate::nn::Linear<Self::T, Self::B>) -> Result<Self::LinearQ> {
                let device = l.weight().device().clone();
                let weight = Q8Tensor::quantize(l.weight())?;
                let bias = l.bias().cloned();
                let mut q = Q8Linear::new(weight, bias, device);
                if Q8Linear::<Self::T>::dense_mode() {
                    q.dense = Some(l);
                }
                Ok(q)
            }
        }
    };
}

backend_q8!(
    Q80F16,
    half::f16,
    "`q8_0` weights with f16 activations. Preferred where the adapter supports \
     `shader-f16`: the layers this does not quantize still halve their traffic."
);
backend_q8!(Q80F32, f32, "`q8_0` weights with f32 activations.");

impl Device {
    /// A buffer holding model weights for the life of the model. Deliberately
    /// not from the recycling pool, which exists for per-token intermediates.
    fn weight_buffer_2(&self, head: &[u8], tail: &[u8]) -> wgpu::Buffer {
        let bytes = head.len() + tail.len();
        let size = (bytes.max(4) as u64).div_ceil(4) * 4;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xn-q8-weight"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, head);
        self.queue.write_buffer(&buf, head.len() as u64, tail);
        buf
    }

    /// Record the `q8_0` matmul. `dst`/`lhs` are contiguous and dtype `T`.
    fn q8_matmul<T: WithDTypeF>(
        &self,
        dst: &Tensor<T, Device>,
        lhs: &Tensor<T, Device>,
        w: &Q8Tensor,
        bias: Option<&Tensor<T, Device>>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let dt = self.dtype_suffix::<T>("q8_0 matmul")?;
        let dst_s = dst.storage_mut()?;
        let lhs_s = lhs.storage()?;
        let bias_s = bias.map(|b| b.storage()).transpose()?;
        // With no bias the binding still has to be filled; the scales buffer
        // stands in and the shader never reads it (`has_bias == 0`).
        let bias_buf: &wgpu::Buffer = match &bias_s {
            Some(b) => &b.buffer,
            None => &w.data,
        };
        // Decode is m == 1 and gets the kernel with no row blocking: carrying
        // the four-row machinery costs 1.8x there, because the extra live
        // registers cut occupancy whether or not the rows are used.
        let kernel = if m == 1 { "qgemv_q8" } else { "qgemm_q8" };
        let push = Pc::new()
            .usize(m)
            .usize(n)
            .usize(k)
            .u32(u32::from(bias.is_some()))
            .usize(w.scale_word_offset);
        self.dispatch(
            &format!("{kernel}_{dt}"),
            // The weight buffer is bound twice: as vec4<u32> for the quants and
            // as f32 for the scales that follow them.
            &[&dst_s.buffer, &lhs_s.buffer, &w.data, &w.data, bias_buf],
            &push,
            div_ceil(n, GEMV_TPC_COLS),
        )
    }
}
