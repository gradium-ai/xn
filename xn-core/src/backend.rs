use crate::Result;
use crate::{BinaryOp, UnaryOp};

pub trait Backend: Sized + Clone + 'static + Sync + Send + std::fmt::Debug {
    type Storage<T: crate::WithDType>: Sized + Sync + Send + 'static;

    fn name(&self) -> String;
    fn synchronize(&self) -> Result<()>;

    fn storage_len<T: crate::WithDType>(storage: &Self::Storage<T>) -> usize;

    fn storage_is_empty<T: crate::WithDType>(storage: &Self::Storage<T>) -> bool {
        Self::storage_len::<T>(storage) == 0
    }

    /// # Safety
    /// This function allocates an unitialized block of memory. It is the responsibility of the
    /// caller to set the memory before using or returning the block.
    unsafe fn alloc_uninit<T: crate::WithDType>(len: usize, dev: &Self)
    -> Result<Self::Storage<T>>;

    // TODO(laurent): Add a from_slice variant.
    fn from_vec<T: crate::WithDType>(v: Vec<T>, dev: &Self) -> Result<Self::Storage<T>>;

    fn cst<T: crate::WithDType>(v: T, len: usize, dev: &Self) -> Result<Self::Storage<T>> {
        let mut res = unsafe { Self::alloc_uninit(len, dev)? };
        Self::fill(&mut res, v, len)?;
        Ok(res)
    }

    fn fill<T: crate::WithDType>(dst: &mut Self::Storage<T>, elem: T, len: usize) -> Result<()>;

    fn rand_uniform(dst: &mut Self::Storage<f32>, len: usize, lo: f32, up: f32) -> Result<()>;

    fn randn(dst: &mut Self::Storage<f32>, len: usize, mean: f32, std: f32) -> Result<()>;

    fn copy<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn to_dtype<T: crate::WithDType, U: crate::WithDType>(
        dst: &mut Self::Storage<U>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()>;

    fn data<T: crate::WithDType>(
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<std::borrow::Cow<'_, [T]>>;

    fn inplace_unary<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()>;

    fn bin_assign<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()>;

    fn unary<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()>;

    fn binary<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()>;

    fn scale_add<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        scale: T,
        add: T,
        len: usize,
    ) -> Result<()>;

    fn transpose<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn copy2d<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        d1: usize,
        d2: usize,
        dst_s: usize,
        src_s: usize,
        dst_o: usize,
        src_o: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn rope<T: crate::WithDTypeF>(
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
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn rope_i<T: crate::WithDTypeF>(
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
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn gemm<T: crate::WithDType>(
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
    ) -> Result<()>;

    fn index_select<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &Self::Storage<i64>,
        num_ids: usize,
        dim: usize,
        dims: &[usize],
    ) -> Result<()>;

    fn apply_causality_mask<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()>;

    fn softmax<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()>;

    fn rms_norm<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()>;

    /// Layer normalization.
    /// Normalizes over the last dimension using mean and variance.
    /// When `remove_mean` is true: y = (x - mean) / sqrt(variance + eps) * weight + bias
    /// When `remove_mean` is false: y = x / sqrt(variance + eps) * weight + bias
    /// The mean is always used when computing the variance.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        weight: &Self::Storage<T>,
        bias: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
        remove_mean: bool,
    ) -> Result<()>;

    /// Reduce max along a dimension.
    /// dst has shape with the reduced dimension removed.
    /// dim_size is the size of the dimension being reduced.
    /// outer_size is the product of dimensions before the reduced dim.
    /// inner_size is the product of dimensions after the reduced dim.
    fn reduce_max<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()>;

    /// Reduce min along a dimension.
    fn reduce_min<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()>;

    /// Reduce argmin along a dimension.
    /// Returns i64 indices.
    fn reduce_argmin<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()>;

    /// Reduce argmax along a dimension.
    /// Returns i64 indices.
    fn reduce_argmax<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()>;

    /// Reduce sum along a dimension.
    fn reduce_sum<T: crate::WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()>;

    /// Scatter set operation.
    /// For each element position in src/ids (which share the same shape `src_dims`):
    ///   dst[..., ids[pos], ...] = src[pos]
    /// where the ids value replaces the coordinate at dimension `dim`.
    /// Copy from strided source to contiguous destination.
    /// `src_offset` is the starting offset in the source storage.
    /// `dims` is the shape, `src_strides` are the strides of the source layout.
    fn copy_strided<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        src_offset: usize,
        dims: &[usize],
        src_strides: &[usize],
    ) -> Result<()>;

    fn scatter_set<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &Self::Storage<i64>,
        dim: usize,
        dst_dims: &[usize],
        src_dims: &[usize],
    ) -> Result<()>;

    /// Broadcast binary operation
    /// lhs_strides and rhs_strides have 0 for broadcast dimensions.
    fn broadcast_binary<T: crate::WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        op: BinaryOp,
    ) -> Result<()>;

    /// 1D convolution.
    /// src: (batch, in_channels, length)
    /// kernel: (out_channels, in_channels/groups, kernel_size)
    /// dst: (batch, out_channels, out_length)
    #[allow(clippy::too_many_arguments)]
    fn conv1d<T: crate::WithDTypeF>(
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
    ) -> Result<()>;

    /// 1D transposed convolution.
    /// src: (batch, in_channels, length)
    /// kernel: (in_channels, out_channels/groups, kernel_size)
    /// dst: (batch, out_channels, out_length)
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d<T: crate::WithDTypeF>(
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
    ) -> Result<()>;

    /// Whether this backend implements [`Backend::sdpa_decode`]. When false, callers compose
    /// the operation from transpose/matmul/softmax instead.
    const FUSED_SDPA_DECODE: bool = false;

    /// Largest head dim [`Backend::sdpa_decode`] accepts. Callers fall back to the composed
    /// form above it, so a backend may cap this at whatever its kernel is sized for.
    const SDPA_MAX_HEAD_DIM: usize = 0;

    /// Fused scaled-dot-product attention for a single query position, i.e. one step of
    /// autoregressive decode.
    ///
    /// Operands keep the `(batch, pos, head, dim)` layout the attention cache is written in,
    /// with `dim` innermost and contiguous, so no transposes are needed:
    ///   `q[q_off + bi * h * d + hh * d + i]`
    ///   `k[k_off + bi * kv_batch_stride + j * h * d + hh * d + i]` (same for `v`)
    /// `dst` is contiguous `(b, h * d)`.
    ///
    /// `mask`, when given, is `(storage, offset, batch stride)`: `kv` contiguous additive
    /// terms per batch entry (`0` to keep a position, `-inf` to drop it), applied to every
    /// head, with entry `bi` reading `mask[offset + bi * batch_stride + j]`. A batch stride of
    /// `0` shares one row of terms across the batch; `kv` gives every entry its own row, which
    /// is how a batch of sequences padded to a common length hides each one's padding.
    ///
    /// Fusing matters here because the composed form is a batch of `h` tiny matmuls per
    /// projection plus a separate softmax pass, which is dominated by per-call overhead rather
    /// than arithmetic.
    #[allow(clippy::too_many_arguments)]
    fn sdpa_decode<T: crate::WithDTypeF>(
        _dst: &mut Self::Storage<T>,
        _q: (&Self::Storage<T>, usize),
        _k: (&Self::Storage<T>, usize),
        _v: (&Self::Storage<T>, usize),
        _mask: Option<(&Self::Storage<T>, usize, usize)>,
        _kv_batch_stride: usize,
        _b: usize,
        _h: usize,
        _d: usize,
        _kv: usize,
        _scale: f32,
    ) -> Result<()> {
        crate::bail!("sdpa_decode is not implemented for this backend")
    }
}
