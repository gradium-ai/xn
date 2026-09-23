use crate::error::Context;
use crate::{Backend, BinaryOp, Dim, Error, Result, Tensor, TensorOrView, WithDType, WithDTypeF};

/// Compute the broadcast output shape for two input shapes.
fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let out_rank = lhs.len().max(rhs.len());
    let mut out_shape = vec![0usize; out_rank];
    for (i, out_dim) in out_shape.iter_mut().enumerate() {
        let lhs_dim = if i < out_rank - lhs.len() { 1 } else { lhs[i - (out_rank - lhs.len())] };
        let rhs_dim = if i < out_rank - rhs.len() { 1 } else { rhs[i - (out_rank - rhs.len())] };

        *out_dim = if lhs_dim == rhs_dim {
            lhs_dim
        } else if lhs_dim == 1 {
            rhs_dim
        } else if rhs_dim == 1 {
            lhs_dim
        } else {
            crate::bail!("cannot broadcast between shapes {lhs:?} and {rhs:?}");
        };
    }

    Ok(out_shape)
}

fn check_same_shape<T: WithDType, B: Backend>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    op: &'static str,
) -> Result<()> {
    if a.shape != b.shape {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: a.shape.clone(),
            rhs: b.shape.clone(),
            op,
        }
        .bt());
    }
    Ok(())
}

macro_rules! binary_op {
    ($n:ident, $bn:ident, $v:ident) => {
        #[tracing::instrument(skip_all)]
        pub fn $n(&self, other: &Self) -> Result<Self> {
            self.binary(other, BinaryOp::$v)
        }

        #[tracing::instrument(skip_all)]
        pub fn $bn(&self, other: &Self) -> Result<Self> {
            self.broadcast_binary(other, BinaryOp::$v)
        }
    };
}

impl<B: Backend> Tensor<f32, B> {
    pub fn randn_like(&self, mean: f32, std: f32) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.randn_(mean, std)?;
        Ok(result)
    }

    pub fn rand_uniform_like(&self, lo: f32, up: f32) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rand_uniform_(lo, up)?;
        Ok(result)
    }

    pub fn randn(&self, shape: impl Into<crate::Shape>, mean: f32, std: f32) -> Result<Self> {
        let shape = shape.into();
        let result = unsafe { Tensor::alloc_uninit(shape, self.device()) }?;
        result.randn_(mean, std)?;
        Ok(result)
    }

    pub fn rand_uniform(&self, shape: impl Into<crate::Shape>, lo: f32, up: f32) -> Result<Self> {
        let shape = shape.into();
        let result = unsafe { Tensor::alloc_uninit(shape, self.device()) }?;
        result.rand_uniform_(lo, up)?;
        Ok(result)
    }
}

impl<T: WithDType, B: Backend> Tensor<T, B> {
    pub fn binary(&self, other: &Self, op: BinaryOp) -> Result<Self> {
        check_same_shape(self, other, op.as_str())?;
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.binary_(self, other, op)?;
        Ok(result)
    }

    pub fn broadcast_binary(&self, other: &Self, op: BinaryOp) -> Result<Self> {
        let out_shape = broadcast_shape(self.dims(), other.dims())?;
        let result = unsafe { Tensor::alloc_uninit(out_shape, self.device()) }?;
        result.broadcast_binary_(self, other, op)?;
        Ok(result)
    }

    binary_op!(add, broadcast_add, Add);
    binary_op!(sub, broadcast_sub, Sub);
    binary_op!(mul, broadcast_mul, Mul);
    binary_op!(div, broadcast_div, Div);
    binary_op!(minimum, broadcast_minimum, Minimum);
    binary_op!(maximum, broadcast_maximum, Maximum);

    /// Transpose two dimensions.
    /// Returns a `TensorView` (zero-copy). Call `.contiguous()?` on the result
    /// if you need a contiguous `Tensor`.
    #[tracing::instrument(skip_all)]
    pub fn transpose<D1: Dim, D2: Dim>(
        &self,
        dim1: D1,
        dim2: D2,
    ) -> Result<crate::TensorView<T, B>> {
        crate::TensorView::from(self).transpose(dim1, dim2)
    }

    pub fn copy(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.copy_(self)?;
        Ok(result)
    }

    pub fn full_like(&self, value: T) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.fill_(value)?;
        Ok(result)
    }

    pub fn scale(&self, m: T) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.scale_(self, m)?;
        Ok(result)
    }

    pub fn add_scalar(&self, a: T) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.add_scalar_(self, a)?;
        Ok(result)
    }

    pub fn scale_add(&self, scale: T, add: T) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.scale_add_(self, scale, add)?;
        Ok(result)
    }

    /// Cast tensor to a different dtype.
    pub fn to<U: WithDType>(&self) -> Result<Tensor<U, B>> {
        let result = if T::DTYPE == U::DTYPE {
            let slf = self as &dyn std::any::Any;
            slf.downcast_ref::<Tensor<U, B>>().context("failed to downcast tensor in to()")?.clone()
        } else {
            let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
            result.to_dtype_(self)?;
            result
        };
        Ok(result)
    }

    /// Flatten all dimensions into a single dimension.
    pub fn flatten_all(&self) -> Result<Self> {
        self.reshape(vec![self.elem_count()])
    }

    /// Flatten dimensions from start to end (inclusive) into a single dimension.
    pub fn flatten<D: Dim>(&self, start_dim: D, end_dim: D) -> Result<Self> {
        let start_dim = start_dim.to_index(self.shape(), "flatten start_dim")?;
        let end_dim = end_dim.to_index(self.shape(), "flatten end_dim")?;
        let dims = self.dims();
        if start_dim > end_dim {
            crate::bail!("flatten: start_dim {start_dim} > end_dim {end_dim}");
        }
        let flat_size: usize = dims[start_dim..=end_dim].iter().product();
        let mut new_dims = Vec::with_capacity(dims.len() - (end_dim - start_dim));
        new_dims.extend_from_slice(&dims[..start_dim]);
        new_dims.push(flat_size);
        new_dims.extend_from_slice(&dims[end_dim + 1..]);
        self.reshape(new_dims)
    }

    /// Create a tensor of zeros with the same shape.
    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape().clone(), self.device())
    }

    /// Transpose (swap last two dimensions).
    /// Returns a `TensorView` (zero-copy). Call `.contiguous()?` on the result
    /// if you need a contiguous `Tensor`.
    pub fn t(&self) -> Result<crate::TensorView<T, B>> {
        let rank = self.rank();
        if rank < 2 {
            crate::bail!("t requires at least 2 dimensions");
        }
        self.transpose(rank - 2, rank - 1)
    }

    /// Unsqueeze: add a dimension of size 1 at the given position.
    pub fn unsqueeze<D: Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index_plus_one(self.shape(), "unsqueeze")?;
        let mut new_dims = self.dims().to_vec();
        new_dims.insert(dim, 1);
        self.reshape(new_dims)
    }
}

impl<T: WithDTypeF, B: Backend> Tensor<T, B> {
    pub fn cos(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.cos_(self)?;
        Ok(result)
    }

    pub fn sin(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sin_(self)?;
        Ok(result)
    }

    pub fn silu(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.silu_(self)?;
        Ok(result)
    }

    #[tracing::instrument(skip_all)]
    pub fn softmax(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.softmax_(self)?;
        Ok(result)
    }

    /// Apply causality mask and return a new tensor.
    /// Shape: (batch * heads, seq_q, seq_kv) or (batch, heads, seq_q, seq_kv)
    /// Masks positions where key position > query position + offset (sets to -inf).
    /// offset: starting position of the first query token (for KV cache generation).
    #[tracing::instrument(skip_all)]
    pub fn apply_causality_mask(&self, offset: usize) -> Result<Self> {
        let result = self.copy()?;
        result.apply_causality_mask_(offset)?;
        Ok(result)
    }

    #[tracing::instrument(skip_all)]
    pub fn rms_norm(&self, alpha: &Self, eps: f32) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rms_norm_(self, alpha, eps)?;
        Ok(result)
    }

    #[tracing::instrument(skip_all)]
    pub fn layer_norm(&self, weight: &Self, bias: &Self, eps: f32) -> Result<Self> {
        self.layer_norm_rm(weight, bias, eps, true)
    }

    #[tracing::instrument(skip_all)]
    pub fn layer_norm_rm(
        &self,
        weight: &Self,
        bias: &Self,
        eps: f32,
        remove_mean: bool,
    ) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.layer_norm_(self, weight, bias, eps, remove_mean)?;
        Ok(result)
    }

    #[tracing::instrument(skip_all)]
    pub fn rope(&self, cos: &Self, sin: &Self, pos: usize) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rope_(self, cos, sin, pos)?;
        Ok(result)
    }

    /// Scaled-dot-product attention for a single query position (autoregressive decode).
    ///
    /// `self` is the query, shape `(b, 1, h, d)`. `k` and `v` are the accumulated keys and
    /// values, shape `(b, kv, h, d)` — the layout an attention cache is written in, so callers
    /// need no transposes. Returns `(b, 1, h * d)`, ready for the output projection.
    ///
    /// `mask`, when given, holds additive terms per key position (`0` to keep it, `-inf` to
    /// drop it), applied to every head. It is either `kv` values, shared by the whole batch,
    /// or `b * kv` values with `b` as its first dimension and `kv` as its last, one row per
    /// batch entry — which is how a batch of sequences padded to a common length hides each
    /// one's padding. The per-row form spells `kv` out even when it is 1, so `(b, 1)` rather
    /// than `(b,)`. A mask per head is not supported on either path.
    ///
    /// Backends advertising [`Backend::FUSED_SDPA_DECODE`] run this as one pass; otherwise it
    /// is composed from transpose/matmul/softmax, which is what the caller would have written
    /// by hand.
    #[tracing::instrument(name = "sdpa-decode", skip_all)]
    pub fn sdpa_decode(
        &self,
        k: &crate::TensorView<T, B>,
        v: &crate::TensorView<T, B>,
        mask: Option<&Self>,
        scale: f32,
    ) -> Result<Self> {
        let qd = self.shape.dims();
        if qd.len() != 4 || qd[1] != 1 {
            crate::bail!("sdpa_decode expects a (b, 1, h, d) query, got {:?}", self.shape)
        }
        let (b, h, d) = (qd[0], qd[2], qd[3]);
        for (name, t) in [("k", k), ("v", v)] {
            let td = t.dims();
            if td.len() != 4 || td[0] != b || td[2] != h || td[3] != d {
                crate::bail!(
                    "sdpa_decode: {name} shape {:?} incompatible with q {:?}",
                    t.shape(),
                    self.shape
                )
            }
        }
        let kv = k.dims()[1];
        if v.dims()[1] != kv {
            crate::bail!("sdpa_decode: k has {kv} positions but v has {}", v.dims()[1])
        }

        // The fused kernel addresses operands as (b, pos, head, dim) with `dim` innermost, so
        // it applies only when the trailing dims are laid out that way. A cache narrowed to its
        // filled prefix satisfies this even though it is shorter than its storage.
        let hd = h * d;
        let laid_out = |strides: &[usize], batch_stride: usize| {
            strides[3] == 1 && strides[2] == d && strides[1] == hd && strides[0] == batch_stride
        };
        // The kernel takes one batch stride for both k and v, so they must agree on it.
        let kv_batch_stride = k.strides()[0];
        let q_ok = laid_out(&self.shape.stride_contiguous(), hd);
        let k_ok = laid_out(k.strides(), kv_batch_stride);
        let v_ok = laid_out(v.strides(), kv_batch_stride);

        // A mask is `kv` terms shared by the batch, or one row of `kv` terms per batch entry.
        let mask_rows = match mask {
            None => 1,
            Some(m) => Self::sdpa_mask_rows(m, b, kv)?,
        };
        // A `Tensor` is always contiguous, so a mask needs no layout check of its own.
        if B::FUSED_SDPA_DECODE && d <= B::SDPA_MAX_HEAD_DIM && kv > 0 && q_ok && k_ok && v_ok {
            let out: Tensor<T, B> =
                unsafe { Tensor::alloc_uninit(crate::Shape::from((b, 1, hd)), self.device()) }?;
            {
                let qs = self.storage()?;
                let (ks, k_off) = k.storage_and_offset()?;
                let (vs, v_off) = v.storage_and_offset()?;
                let mut os = out.storage_mut()?;
                let ms = match mask {
                    Some(m) => Some(m.storage()?),
                    None => None,
                };
                let mask_batch_stride = if mask_rows == 1 { 0 } else { kv };
                B::sdpa_decode(
                    &mut os,
                    (&qs, 0),
                    (&ks, k_off),
                    (&vs, v_off),
                    ms.as_deref().map(|m| (m, 0, mask_batch_stride)),
                    kv_batch_stride,
                    b,
                    h,
                    d,
                    kv,
                    scale,
                )?;
            }
            return Ok(out);
        }

        self.sdpa_composed(k, v, mask, mask_rows, scale, b, hd)
    }

    /// How many rows of `kv` terms `mask` holds: 1 when shared by the batch, `b` when there is
    /// one per entry. Anything else is a shape error.
    fn sdpa_mask_rows(mask: &Self, b: usize, kv: usize) -> Result<usize> {
        let dims = mask.shape.dims();
        let last = dims.last().copied().unwrap_or(0);
        let n = mask.shape.elem_count();
        if last == kv && n == kv {
            Ok(1)
        } else if last == kv && n == b * kv && dims.first() == Some(&b) {
            Ok(b)
        } else {
            crate::bail!(
                "sdpa_decode: mask {:?} must be {kv} terms, or {b} rows of {kv} with the batch \
                 as its first dimension; a mask per head is not supported",
                mask.shape
            )
        }
    }

    /// The transpose/matmul/softmax sequence a caller would otherwise write by hand. Used when
    /// the backend has no fused kernel, or when the operands do not match the layout it needs.
    #[allow(clippy::too_many_arguments)]
    fn sdpa_composed(
        &self,
        k: &crate::TensorView<T, B>,
        v: &crate::TensorView<T, B>,
        mask: Option<&Self>,
        mask_rows: usize,
        scale: f32,
        b: usize,
        hd: usize,
    ) -> Result<Self> {
        let q = crate::TensorView::from(self).transpose(1, 2)?;
        let kt = k.transpose(1, 2)?;
        let vt = v.transpose(1, 2)?;
        let attn = q.matmul_t(&kt)?.scale(T::from_f32(scale))?;
        let attn = match mask {
            // Scores are (b, h, 1, kv); a mask row broadcasts over the heads, and a shared
            // mask over the batch as well.
            Some(m) => attn.broadcast_add(&m.reshape((mask_rows, 1, 1, kt.dims()[2]))?)?,
            None => attn,
        };
        let attn = attn.softmax()?;
        attn.matmul(&vt)?.transpose(1, 2)?.reshape((b, 1, hd))?.contiguous()
    }

    #[tracing::instrument(skip_all)]
    pub fn rope_i(&self, cos: &Self, sin: &Self, pos: usize) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rope_i_(self, cos, sin, pos)?;
        Ok(result)
    }

    /// 1D convolution.
    /// Input: (batch, in_channels, length)
    /// Kernel: (out_channels, in_channels/groups, kernel_size)
    /// Output: (batch, out_channels, out_length)
    #[tracing::instrument(skip_all)]
    pub fn conv1d(
        &self,
        kernel: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (batch, in_channels, length) = self.dims3()?;
        let (out_channels, kernel_in_channels, kernel_size) = kernel.dims3()?;

        if !in_channels.is_multiple_of(groups) {
            crate::bail!("in_channels ({in_channels}) must be divisible by groups ({groups})");
        }
        if !out_channels.is_multiple_of(groups) {
            crate::bail!("out_channels ({out_channels}) must be divisible by groups ({groups})",);
        }
        if kernel_in_channels != in_channels / groups {
            crate::bail!(
                "kernel in_channels/groups mismatch: expected {}, got {kernel_in_channels}",
                in_channels / groups,
            );
        }

        // Compute output length
        let out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

        let mut result =
            unsafe { Tensor::alloc_uninit((batch, out_channels, out_length), self.device()) }?;
        result.conv1d_(self, kernel, stride, padding, dilation, groups)?;

        // Add bias if provided
        if let Some(bias) = bias {
            let bias_dims = bias.dims();
            if bias_dims != [out_channels] {
                crate::bail!(
                    "bias shape mismatch: expected [{out_channels}], got {:?}",
                    bias.shape()
                );
            }
            // Reshape bias to (1, out_channels, 1) for broadcasting
            let bias = bias.reshape((1, out_channels, 1))?;
            result = result.broadcast_add(&bias)?;
        }

        Ok(result)
    }

    /// 1D transposed convolution.
    /// Input: (batch, in_channels, length)
    /// Kernel: (in_channels, out_channels/groups, kernel_size)
    /// Output: (batch, out_channels, out_length)
    #[tracing::instrument(skip_all)]
    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<Self> {
        let (batch, in_channels, length) = self.dims3()?;
        let (k_in_channels, out_channels_per_group, kernel_size) = kernel.dims3()?;

        let out_channels = out_channels_per_group * groups;

        if !in_channels.is_multiple_of(groups) {
            crate::bail!("in_channels ({in_channels}) must be divisible by groups ({groups})");
        }
        if k_in_channels != in_channels {
            crate::bail!(
                "kernel in_channels mismatch: expected {in_channels}, got {k_in_channels}",
            );
        }

        // Compute output length for transposed convolution
        // out_length = (length - 1) * stride - 2 * padding + kernel_size + output_padding
        let out_length = (length - 1) * stride + kernel_size + output_padding - 2 * padding;

        let mut result =
            unsafe { Tensor::alloc_uninit((batch, out_channels, out_length), self.device()) }?;
        result.conv_transpose1d_(self, kernel, stride, padding, output_padding, groups)?;

        // Add bias if provided
        if let Some(bias) = bias {
            let bias_dims = bias.dims();
            if bias_dims != [out_channels] {
                crate::bail!(
                    "bias shape mismatch: expected [{out_channels}], got {:?}",
                    bias.shape()
                );
            }
            // Reshape bias to (1, out_channels, 1) for broadcasting
            let bias = bias.reshape((1, out_channels, 1))?;
            result = result.broadcast_add(&bias)?;
        }

        Ok(result)
    }

    /// Element-wise square.
    pub fn sqr(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sqr_(self)?;
        Ok(result)
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sqrt_(self)?;
        Ok(result)
    }

    /// Element-wise reciprocal square root.
    pub fn rsqrt(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.rsqrt_(self)?;
        Ok(result)
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.abs_(self)?;
        Ok(result)
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.neg_(self)?;
        Ok(result)
    }

    /// Element-wise log.
    pub fn log(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.log_(self)?;
        Ok(result)
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.exp_(self)?;
        Ok(result)
    }

    /// Sum along dimensions, keeping the dimensions (with size 1).
    #[tracing::instrument(skip_all)]
    pub fn sum_keepdim(&self, dims: impl Into<Vec<usize>>) -> Result<Self> {
        let mut dims: Vec<usize> = dims.into();
        // Sort dims in descending order so we can reduce from the end
        dims.sort_by(|a, b| b.cmp(a));
        dims.dedup();

        let mut result: Option<Self> = None;
        for &dim in &dims {
            let src = result.as_ref().unwrap_or(self);
            if dim >= src.rank() {
                crate::bail!(
                    "sum_keepdim: dimension {} out of range for tensor of rank {}",
                    dim,
                    src.rank()
                );
            }
            // Reduce along dim, then reshape to keep the dimension with size 1
            let current_dims = src.dims().to_vec();

            // Output shape has dim reduced (removed)
            let mut reduced_dims: Vec<usize> = current_dims.clone();
            reduced_dims.remove(dim);
            if reduced_dims.is_empty() {
                reduced_dims.push(1);
            }

            let reduced = unsafe { Tensor::alloc_uninit(reduced_dims, src.device()) }?;
            reduced.reduce_sum_(src, dim)?;

            // Reshape to keep the dimension with size 1
            let mut keepdim_shape: Vec<usize> = current_dims;
            keepdim_shape[dim] = 1;
            result = Some(reduced.reshape(keepdim_shape)?);
        }

        match result {
            Some(result) => Ok(result),
            None => self.copy(),
        }
    }

    /// Maximum value along dimension.
    #[tracing::instrument(skip_all)]
    pub fn max<D: Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "max dim")?;
        let mut out_dims: Vec<usize> = self.dims().to_vec();
        out_dims.remove(dim);
        if out_dims.is_empty() {
            out_dims.push(1);
        }
        let result = unsafe { Tensor::alloc_uninit(out_dims, self.device()) }?;
        result.reduce_max_(self, dim)?;
        Ok(result)
    }

    /// Minimum value along dimension.
    #[tracing::instrument(skip_all)]
    pub fn min<D: Dim>(&self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "min dim")?;
        let mut out_dims: Vec<usize> = self.dims().to_vec();
        out_dims.remove(dim);
        if out_dims.is_empty() {
            out_dims.push(1);
        }
        let result = unsafe { Tensor::alloc_uninit(out_dims, self.device()) }?;
        result.reduce_min_(self, dim)?;
        Ok(result)
    }

    /// Argmin along dimension.
    /// Returns i64 indices.
    #[tracing::instrument(skip_all)]
    pub fn argmin<D: Dim>(&self, dim: D) -> Result<Tensor<i64, B>> {
        let dim = dim.to_index(self.shape(), "argmin dim")?;
        let mut out_dims: Vec<usize> = self.dims().to_vec();
        out_dims.remove(dim);
        if out_dims.is_empty() {
            out_dims.push(1);
        }
        let result: Tensor<i64, B> = unsafe { Tensor::alloc_uninit(out_dims, self.device()) }?;
        Self::reduce_argmin_(&result, self, dim)?;
        Ok(result)
    }

    /// Argmax along dimension.
    /// Returns i64 indices.
    #[tracing::instrument(skip_all)]
    pub fn argmax<D: Dim>(&self, dim: D) -> Result<Tensor<i64, B>> {
        let dim = dim.to_index(self.shape(), "argmax dim")?;
        let mut out_dims: Vec<usize> = self.dims().to_vec();
        out_dims.remove(dim);
        if out_dims.is_empty() {
            out_dims.push(1);
        }
        let result: Tensor<i64, B> = unsafe { Tensor::alloc_uninit(out_dims, self.device()) }?;
        Self::reduce_argmax_(&result, self, dim)?;
        Ok(result)
    }

    /// GELU activation with erf.
    #[tracing::instrument(skip_all)]
    pub fn gelu_erf(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.gelu_erf_(self)?;
        Ok(result)
    }

    /// ELU activation.
    #[tracing::instrument(skip_all)]
    pub fn elu(&self, alpha: f32) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.elu_(self, alpha)?;
        Ok(result)
    }

    /// ReLU activation.
    #[tracing::instrument(skip_all)]
    pub fn relu(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.relu_(self)?;
        Ok(result)
    }

    /// Tanh activation.
    #[tracing::instrument(skip_all)]
    pub fn tanh(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.tanh_(self)?;
        Ok(result)
    }

    /// Sigmoid activation.
    #[tracing::instrument(skip_all)]
    pub fn sigmoid(&self) -> Result<Self> {
        let result = unsafe { Tensor::alloc_uninit(self.shape.clone(), self.device()) }?;
        result.sigmoid_(self)?;
        Ok(result)
    }

    /// Expand tensor to a new shape (broadcasting).
    pub fn expand(&self, shape: impl Into<crate::Shape>) -> Result<crate::TensorView<T, B>> {
        crate::TensorView::from(self).expand(shape)
    }

    /// Pad with zeros along a dimension.
    pub fn pad_with_zeros<D: Dim>(&self, dim: D, left: usize, right: usize) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "pad_with_zeros")?;
        let dims = self.dims();
        let dim_size = dims[dim];

        // Compute new shape
        let mut new_dims = dims.to_vec();
        new_dims[dim] = dim_size + left + right;
        let new_shape = crate::Shape::from(new_dims);

        // Create output tensor filled with zeros
        let result = Self::zeros(new_shape, self.device())?;

        if dim_size == 0 || self.elem_count() == 0 {
            return Ok(result);
        }

        // Copy original data to the padded position
        let outer_size: usize = dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);
        let new_dim_size = dim_size + left + right;

        {
            let mut dst = result.storage_mut()?;
            let src = self.storage()?;
            B::copy2d(
                &mut *dst,
                &*src,
                outer_size,                // d1: number of outer blocks
                dim_size * inner_size,     // d2: elements per block
                new_dim_size * inner_size, // dst_s: stride in output
                dim_size * inner_size,     // src_s: stride in source
                left * inner_size,         // dst_o: offset to skip left padding
                0,                         // src_o: start from beginning of source
            )?;
        }

        Ok(result)
    }

    /// Pad by replicating boundary values.
    pub fn pad_with_same<D: Dim>(&self, dim: D, left: usize, right: usize) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "pad_with_same")?;
        let dims = self.dims();
        let dim_size = dims[dim];

        if dim_size == 0 {
            crate::bail!("cannot pad_with_same on dimension with size 0");
        }

        // Compute new shape
        let mut new_dims = dims.to_vec();
        new_dims[dim] = dim_size + left + right;

        let result = unsafe { Self::alloc_uninit(new_dims, self.device()) }?;

        let outer_size: usize = dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);
        let new_dim_size = dim_size + left + right;

        {
            let mut dst = result.storage_mut()?;
            let src = self.storage()?;

            // Copy original data to the center position
            B::copy2d(
                &mut *dst,
                &*src,
                outer_size,                // d1: number of outer blocks
                dim_size * inner_size,     // d2: elements per block
                new_dim_size * inner_size, // dst_s: stride in output
                dim_size * inner_size,     // src_s: stride in source
                left * inner_size,         // dst_o: offset to skip left padding
                0,                         // src_o: start from beginning of source
            )?;

            // Replicate first slice for left padding
            for l in 0..left {
                B::copy2d(
                    &mut *dst,
                    &*src,
                    outer_size,                // d1: number of outer blocks
                    inner_size,                // d2: one slice
                    new_dim_size * inner_size, // dst_s: stride in output
                    dim_size * inner_size,     // src_s: stride in source
                    l * inner_size,            // dst_o: position l in left padding
                    0,                         // src_o: first slice of source
                )?;
            }

            // Replicate last slice for right padding
            for r in 0..right {
                B::copy2d(
                    &mut *dst,
                    &*src,
                    outer_size,                         // d1: number of outer blocks
                    inner_size,                         // d2: one slice
                    new_dim_size * inner_size,          // dst_s: stride in output
                    dim_size * inner_size,              // src_s: stride in source
                    (left + dim_size + r) * inner_size, // dst_o: position after original data
                    (dim_size - 1) * inner_size,        // src_o: last slice of source
                )?;
            }
        }

        Ok(result)
    }

    pub fn matmul_t<R: TensorOrView<T, B>>(&self, rhs: &R) -> Result<Self> {
        matmul_t(self, rhs)
    }

    pub fn matmul<R: TensorOrView<T, B>>(&self, rhs: &R) -> Result<Self> {
        matmul(self, rhs)
    }
}

#[tracing::instrument(skip_all)]
pub fn matmul_with_t<T: WithDTypeF, B: Backend, L: TensorOrView<T, B>, R: TensorOrView<T, B>>(
    lhs: &L,
    rhs: &R,
    rhs_t: bool,
) -> Result<Tensor<T, B>> {
    if lhs.shape().rank() < 2 || rhs.shape().rank() < 2 {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
            msg: "matmul requires at least 2D tensors",
        }
        .bt());
    }

    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();

    // Get M, K from lhs (last two dims)
    let lhs_m = lhs_dims[lhs_dims.len() - 2];
    let lhs_k = lhs_dims[lhs_dims.len() - 1];

    // Get K, N from rhs (last two dims), accounting for transpose
    let (rhs_k, rhs_n) = if rhs_t {
        (rhs_dims[rhs_dims.len() - 1], rhs_dims[rhs_dims.len() - 2])
    } else {
        (rhs_dims[rhs_dims.len() - 2], rhs_dims[rhs_dims.len() - 1])
    };

    if lhs_k != rhs_k {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
            msg: "inner dimensions do not match in matmul",
        }
        .bt());
    }

    // Check batch dimensions are compatible
    // rhs can be 2D (no batch) which broadcasts to any lhs batch
    let lhs_batch = &lhs_dims[..lhs_dims.len() - 2];
    let rhs_batch = &rhs_dims[..rhs_dims.len() - 2];
    if !rhs_batch.is_empty() && lhs_batch != rhs_batch {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
            msg: "batch dimensions do not match in matmul",
        }
        .bt());
    }

    // Build output shape: lhs batch dims + [M, N]
    let mut target_shape = lhs_batch.to_vec();
    target_shape.push(lhs_m);
    target_shape.push(rhs_n);

    let dev = lhs.device();
    let result = unsafe { Tensor::<T, B>::alloc_uninit(target_shape, dev) }?;
    result.matmul_(lhs, rhs, rhs_t)?;
    Ok(result)
}

pub fn matmul<T: WithDTypeF, B: Backend, L: TensorOrView<T, B>, R: TensorOrView<T, B>>(
    lhs: &L,
    rhs: &R,
) -> Result<Tensor<T, B>> {
    matmul_with_t(lhs, rhs, false)
}

pub fn matmul_t<T: WithDTypeF, B: Backend, L: TensorOrView<T, B>, R: TensorOrView<T, B>>(
    lhs: &L,
    rhs: &R,
) -> Result<Tensor<T, B>> {
    matmul_with_t(lhs, rhs, true)
}
