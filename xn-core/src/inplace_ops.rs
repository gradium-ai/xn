use std::sync::Arc;

use crate::error::check_same_shape;
use crate::{Backend, Result, Tensor, TensorOrView, WithDType, WithDTypeF};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnaryOp {
    Cos,
    Sin,
    Exp,
    Log,
    Neg,
    Sqr,
    Sqrt,
    Rsqrt,
    Abs,
    GeluErf,
    Elu { alpha: f32 },
    Relu,
    Silu,
    Tanh,
    Sigmoid,
}

impl UnaryOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            UnaryOp::Cos => "cos",
            UnaryOp::Sin => "sin",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Neg => "neg",
            UnaryOp::Sqr => "sqr",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Rsqrt => "rsqrt",
            UnaryOp::Abs => "abs",
            UnaryOp::GeluErf => "gelu_erf",
            UnaryOp::Elu { .. } => "elu",
            UnaryOp::Relu => "relu",
            UnaryOp::Silu => "silu",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Sigmoid => "sigmoid",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
}

impl BinaryOp {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Maximum => "maximum",
            BinaryOp::Minimum => "minimum",
        }
    }
}

macro_rules! binary_op {
    ($ipn:ident, $n_:ident, $bn_:ident, $v:ident) => {
        pub fn $ipn(&self, other: &Self) -> Result<()> {
            self.inplace_binary(other, BinaryOp::$v)
        }

        pub fn $n_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
            self.binary_(lhs, rhs, BinaryOp::$v)
        }

        pub fn $bn_(&self, lhs: &Self, rhs: &Self) -> Result<()> {
            self.broadcast_binary_(lhs, rhs, BinaryOp::$v)
        }
    };
}

/// Uniform stride covering a row-major flat iteration of the batch dims (all but the
/// last two), when the layout admits one; size-1 dims never contribute to an offset
/// and are ignored. Contiguous tensors always admit one; strided views may not, e.g.
/// the (b, h) batch dims of a transpose(1, 2) view of a (b, t, h, d) tensor have
/// strides (t*h*d, d) which cannot be walked with a single stride when b > 1.
fn flat_batch_stride(dims: &[usize], strides: &[usize]) -> Option<usize> {
    let nb = dims.len().saturating_sub(2);
    let mut step = None;
    let mut suffix = 1usize;
    for j in (0..nb).rev() {
        if dims[j] == 1 {
            continue;
        }
        match step {
            None => step = Some(strides[j]),
            Some(s) => {
                if strides[j] != s * suffix {
                    return None;
                }
            }
        }
        suffix *= dims[j];
    }
    Some(step.unwrap_or(0))
}

/// Materialize a contiguous copy of an arbitrarily strided operand.
fn contiguous_copy<T: WithDType, B: Backend>(src: &dyn TensorOrView<T, B>) -> Result<Tensor<T, B>> {
    let dst: Tensor<T, B> = unsafe { Tensor::alloc_uninit(src.shape().clone(), src.device()) }?;
    {
        let (src_data, src_o) = src.storage_and_offset()?;
        let mut dst_data = dst.storage_mut()?;
        B::copy_strided(&mut dst_data, &*src_data, src_o, src.dims(), &src.strides())?;
    }
    Ok(dst)
}

impl<T: WithDType, B: Backend> Tensor<T, B> {
    fn check_not_same_storage(&self, other: &Self, op: &str) -> Result<()> {
        if Arc::ptr_eq(&self.data, &other.data) {
            crate::bail!("{op}: cannot use when dst and src share their storage");
        }
        Ok(())
    }

    pub(crate) fn inplace_binary(&self, other: &Self, op: BinaryOp) -> Result<()> {
        self.check_not_same_storage(other, op.as_str())?;
        check_same_shape(&self.shape, &other.shape, op.as_str())?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src = other.storage()?;
        B::bin_assign(&mut *dst, &*src, len, op)?;
        Ok(())
    }

    pub fn binary_(&self, lhs: &Self, rhs: &Self, op: BinaryOp) -> Result<()> {
        self.check_not_same_storage(lhs, op.as_str())?;
        self.check_not_same_storage(rhs, op.as_str())?;
        check_same_shape(&lhs.shape, &rhs.shape, op.as_str())?;
        check_same_shape(&self.shape, &lhs.shape, op.as_str())?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let lhs_data = lhs.storage()?;
        let rhs_data = rhs.storage()?;
        B::binary(&mut *dst, &*lhs_data, &*rhs_data, len, op)?;
        Ok(())
    }

    pub fn broadcast_binary_(&self, lhs: &Self, rhs: &Self, op: BinaryOp) -> Result<()> {
        self.check_not_same_storage(lhs, "broadcast_binary")?;
        self.check_not_same_storage(rhs, "broadcast_binary")?;
        let dst_shape = self.dims();
        let (dst_shape, lhs_strides, rhs_strides) =
            compute_broadcast_strides(dst_shape, lhs.dims(), rhs.dims())?;
        let mut dst = self.storage_mut()?;
        let lhs_data = lhs.storage()?;
        let rhs_data = rhs.storage()?;
        B::broadcast_binary(
            &mut *dst,
            &*lhs_data,
            &*rhs_data,
            &dst_shape,
            &lhs_strides,
            &rhs_strides,
            op,
        )?;
        Ok(())
    }

    binary_op!(inplace_add, add_, broadcast_add_, Add);
    binary_op!(inplace_sub, sub_, broadcast_sub_, Sub);
    binary_op!(inplace_mul, mul_, broadcast_mul_, Mul);
    binary_op!(inplace_div, div_, broadcast_div_, Div);
    binary_op!(inplace_maximum, maximum_, broadcast_maximum_, Maximum);
    binary_op!(inplace_minimum, minimum_, broadcast_minimum_, Minimum);

    pub fn to_dtype_<U: WithDType>(&self, src: &Tensor<U, B>) -> Result<()> {
        check_same_shape(&self.shape, &src.shape, "to_dtype")?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::to_dtype(&mut *dst, &*src_data, len)?;
        Ok(())
    }

    pub fn transpose_(&self, src: &Self, dim1: usize, dim2: usize) -> Result<()> {
        self.check_not_same_storage(src, "transpose_")?;
        let dims = src.dims();
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        if dim1 == dim2 {
            B::copy(&mut *dst, &*src_data, len)?;
        } else {
            B::transpose(&mut *dst, &*src_data, dim1, dim2, dims)?;
        }
        Ok(())
    }

    pub fn copy_(&self, src: &Self) -> Result<()> {
        self.check_not_same_storage(src, "copy_")?;
        check_same_shape(&self.shape, &src.shape, "copy_")?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::copy(&mut *dst, &*src_data, len)?;
        Ok(())
    }

    pub fn fill_(&self, value: T) -> Result<()> {
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        B::fill(&mut *dst, value, len)?;
        Ok(())
    }

    pub fn scale_(&self, src: &Self, m: T) -> Result<()> {
        self.scale_add_(src, m, T::zero())
    }

    pub fn add_scalar_(&self, src: &Self, a: T) -> Result<()> {
        self.scale_add_(src, T::one(), a)
    }

    pub fn scale_add_(&self, src: &Self, scale: T, add: T) -> Result<()> {
        self.check_not_same_storage(src, "scale_add_")?;
        check_same_shape(&self.shape, &src.shape, "scale_add_")?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::scale_add(&mut *dst, &*src_data, scale, add, len)?;
        Ok(())
    }
}

impl<B: Backend> Tensor<f32, B> {
    pub fn randn_(&self, mean: f32, std: f32) -> Result<()> {
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        B::randn(&mut *dst, len, mean, std)?;
        Ok(())
    }

    pub fn rand_uniform_(&self, lo: f32, up: f32) -> Result<()> {
        if up < lo {
            crate::bail!("rand_uniform: upper bound ({up}) must be >= lower bound ({lo})");
        }
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        B::rand_uniform(&mut *dst, len, lo, up)?;
        Ok(())
    }
}

impl<T: WithDTypeF, B: Backend> Tensor<T, B> {
    pub fn unary_(&self, src: &Self, op: UnaryOp) -> Result<()> {
        self.check_not_same_storage(src, op.as_str())?;
        check_same_shape(&self.shape, &src.shape, op.as_str())?;
        let len = self.elem_count();
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::unary(&mut *dst, &*src_data, len, op)?;
        Ok(())
    }

    pub fn cos_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Cos)
    }

    pub fn sin_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Sin)
    }

    pub fn exp_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Exp)
    }

    pub fn log_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Log)
    }

    pub fn neg_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Neg)
    }

    pub fn gelu_erf_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::GeluErf)
    }

    pub fn elu_(&self, src: &Self, alpha: f32) -> Result<()> {
        self.unary_(src, UnaryOp::Elu { alpha })
    }

    pub fn abs_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Abs)
    }

    pub fn sqr_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Sqr)
    }

    pub fn sqrt_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Sqrt)
    }

    pub fn rsqrt_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Rsqrt)
    }

    pub fn relu_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Relu)
    }

    pub fn tanh_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Tanh)
    }

    pub fn sigmoid_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Sigmoid)
    }

    pub fn silu_(&self, src: &Self) -> Result<()> {
        self.unary_(src, UnaryOp::Silu)
    }

    pub fn softmax_(&self, src: &Self) -> Result<()> {
        self.check_not_same_storage(src, "softmax_")?;
        check_same_shape(&self.shape, &src.shape, "softmax_")?;
        let dim_m1 = self.shape.dims().last().copied().unwrap_or(1);
        let d = self.elem_count() / dim_m1;
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::softmax(&mut *dst, &*src_data, dim_m1, d)?;
        Ok(())
    }

    /// Apply causality mask in-place.
    /// Shape: (batch * heads, seq_q, seq_kv) or (batch, heads, seq_q, seq_kv)
    /// Masks positions where key position > query position + offset (sets to -inf).
    /// offset: starting position of the first query token (for KV cache generation).
    pub fn apply_causality_mask_(&self, offset: usize) -> Result<()> {
        let dims = self.dims();
        let (bh, t1, t2) = match dims.len() {
            3 => (dims[0], dims[1], dims[2]),
            4 => (dims[0] * dims[1], dims[2], dims[3]),
            _ => crate::bail!(
                "apply_causality_mask expects 3D or 4D tensor, got shape {:?}",
                self.shape()
            ),
        };
        let mut dst = self.storage_mut()?;
        B::apply_causality_mask(&mut *dst, bh, t1, t2, offset)?;
        Ok(())
    }

    pub fn rms_norm_(&self, src: &Self, alpha: &Self, eps: f32) -> Result<()> {
        self.check_not_same_storage(src, "rms_norm_")?;
        self.check_not_same_storage(alpha, "rms_norm_")?;
        check_same_shape(&self.shape, &src.shape, "rms_norm_ src")?;
        if eps <= 0.0 {
            crate::bail!("rms_norm_ eps must be positive");
        }
        let dim_m1 = self.shape.dims().last().copied().unwrap_or(1);
        let d = self.elem_count() / dim_m1;
        let expected_shape_alpha = dim_m1.into();
        check_same_shape(&alpha.shape, &expected_shape_alpha, "rms_norm_ alpha")?;
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let alpha_data = alpha.storage()?;
        B::rms_norm(&mut *dst, &*src_data, &*alpha_data, dim_m1, d, eps)?;
        Ok(())
    }

    pub fn layer_norm_(
        &self,
        src: &Self,
        weight: &Self,
        bias: &Self,
        eps: f32,
        remove_mean: bool,
    ) -> Result<()> {
        self.check_not_same_storage(src, "layer_norm_")?;
        self.check_not_same_storage(weight, "layer_norm_")?;
        self.check_not_same_storage(bias, "layer_norm_")?;
        check_same_shape(&self.shape, &src.shape, "layer_norm_ src")?;
        if eps <= 0.0 {
            crate::bail!("layer_norm_ eps must be positive");
        }
        let dim_m1 = self.shape.dims().last().copied().unwrap_or(1);
        let d = self.elem_count() / dim_m1;
        let expected_shape = dim_m1.into();
        check_same_shape(&weight.shape, &expected_shape, "layer_norm_ weight")?;
        check_same_shape(&bias.shape, &expected_shape, "layer_norm_ bias")?;
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let weight_data = weight.storage()?;
        let bias_data = bias.storage()?;
        B::layer_norm(
            &mut *dst,
            &*src_data,
            &*weight_data,
            &*bias_data,
            dim_m1,
            d,
            eps,
            remove_mean,
        )?;
        Ok(())
    }

    pub fn matmul_<L: TensorOrView<T, B>, R: TensorOrView<T, B>>(
        &self,
        lhs: &L,
        rhs: &R,
        rhs_t: bool,
    ) -> Result<()> {
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();

        if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
            crate::bail!(
                "matmul requires at least 2D tensors, got lhs {:?}, rhs {:?}",
                lhs.shape(),
                rhs.shape()
            );
        }

        // Extract M, K from lhs (last two dimensions)
        let lhs_m = lhs_dims[lhs_dims.len() - 2];
        let lhs_k = lhs_dims[lhs_dims.len() - 1];

        // Extract K, N from rhs (last two dimensions), accounting for transpose
        let (rhs_k, rhs_n) = if rhs_t {
            (rhs_dims[rhs_dims.len() - 1], rhs_dims[rhs_dims.len() - 2])
        } else {
            (rhs_dims[rhs_dims.len() - 2], rhs_dims[rhs_dims.len() - 1])
        };

        if lhs_k != rhs_k {
            crate::bail!(
                "matmul inner dimension mismatch: lhs {:?}, rhs {:?}, rhs_t={rhs_t}",
                lhs.shape(),
                rhs.shape()
            );
        }

        // Compute batch dimensions
        let lhs_batch_dims = &lhs_dims[..lhs_dims.len() - 2];
        let rhs_batch_dims = &rhs_dims[..rhs_dims.len() - 2];
        let lhs_batch: usize = lhs_batch_dims.iter().product::<usize>().max(1);
        let rhs_batch: usize = rhs_batch_dims.iter().product::<usize>().max(1);

        // Check batch dimensions are compatible
        if rhs_batch != 1 && rhs_batch != lhs_batch {
            crate::bail!(
                "matmul batch dimension mismatch: lhs {:?}, rhs {:?}",
                lhs.shape(),
                rhs.shape()
            );
        }

        let (m, n, k) = (lhs_m, rhs_n, lhs_k);

        let dst_elems = lhs_batch * m * n;
        let dst_data = self.storage()?;
        let storage_len = B::storage_len(&*dst_data);
        drop(dst_data);

        if dst_elems > storage_len {
            crate::bail!(
                "matmul dst is too small, dst {} < {dst_elems}, lhs {:?} rhs {:?}",
                storage_len,
                lhs.shape(),
                rhs.shape()
            );
        }

        // The backend gemm walks the flattened batch space with a single uniform stride
        // per operand. When the batch dims of a strided view do not collapse to such a
        // stride (e.g. a transpose(1, 2) view of a (b, t, h, d) tensor with b > 1),
        // materialize a contiguous copy of the operand first — otherwise every batch
        // entry past the first would read from wrong offsets.
        let lhs_contiguous;
        let lhs: &dyn TensorOrView<T, B> = if flat_batch_stride(lhs_dims, &lhs.strides()).is_some()
        {
            lhs
        } else {
            lhs_contiguous = contiguous_copy(lhs)?;
            &lhs_contiguous
        };
        let rhs_contiguous;
        let rhs: &dyn TensorOrView<T, B> =
            if rhs_batch == 1 || flat_batch_stride(rhs_dims, &rhs.strides()).is_some() {
                rhs
            } else {
                rhs_contiguous = contiguous_copy(rhs)?;
                &rhs_contiguous
            };

        // Use actual strides from the TensorOrView to support non-contiguous inputs
        // (e.g. transposed views passed directly to matmul).
        let lhs_s = lhs.strides();
        let rhs_s = rhs.strides();

        let (dst_cs, dst_rs) = (1, n);
        let lhs_cs = lhs_s[lhs_s.len() - 1];
        let lhs_rs = lhs_s[lhs_s.len() - 2];
        let (rhs_cs, rhs_rs) = if rhs_t {
            // rhs is stored as (..., n, k) but gemm reads it as (k, n):
            // row stride (along k) = stride of last dim, col stride (along n) = stride of second-to-last
            (rhs_s[rhs_s.len() - 2], rhs_s[rhs_s.len() - 1])
        } else {
            (rhs_s[rhs_s.len() - 1], rhs_s[rhs_s.len() - 2])
        };

        let lhs_b_stride = match flat_batch_stride(lhs_dims, &lhs_s) {
            Some(s) if lhs_s.len() >= 3 => s,
            _ => m * k,
        };
        let rhs_b_stride = if rhs_batch == 1 {
            0
        } else {
            match flat_batch_stride(rhs_dims, &rhs_s) {
                Some(s) if rhs_s.len() >= 3 => s,
                _ => rhs_dims[rhs_dims.len() - 2] * rhs_dims[rhs_dims.len() - 1],
            }
        };

        let mut dst = self.storage_mut()?;
        let (lhs_data, lhs_o) = lhs.storage_and_offset()?;
        let (rhs_data, rhs_o) = rhs.storage_and_offset()?;

        // A broadcast rhs lets the lhs batch fold into M so that one gemm reads the rhs
        // once instead of once per batch entry: when the lhs rows run contiguously
        // across batches, and when there is a single row per batch entry (e.g. the last
        // token of each sequence, `x.narrow(1, t - 1..t)` on a `(b, t, h)` tensor), where
        // the batch stride is the row stride of the folded `(batch, k)` operand.
        let (lhs_batch, rhs_b_stride, m, n, k, lhs_rs) =
            if rhs_batch == 1 && lhs_b_stride == m * lhs_rs {
                (1, 1, lhs_batch * lhs_m, rhs_n, lhs_k, lhs_rs)
            } else if rhs_batch == 1 && m == 1 {
                (1, 1, lhs_batch, rhs_n, lhs_k, lhs_b_stride)
            } else {
                (lhs_batch, rhs_b_stride, m, n, k, lhs_rs)
            };
        B::gemm(
            &mut *dst,
            (&*lhs_data, lhs_o),
            (&*rhs_data, rhs_o),
            m,
            n,
            k,
            lhs_batch,
            lhs_b_stride,
            rhs_b_stride,
            (dst_cs, dst_rs),
            (lhs_cs, lhs_rs),
            (rhs_cs, rhs_rs),
        )?;

        Ok(())
    }

    pub fn rope_(&self, src: &Self, cos: &Self, sin: &Self, pos: usize) -> Result<()> {
        self.check_not_same_storage(src, "rope_")?;
        self.check_not_same_storage(cos, "rope_")?;
        self.check_not_same_storage(sin, "rope_")?;
        check_same_shape(&self.shape, &src.shape, "rope_ src")?;
        check_same_shape(&cos.shape, &sin.shape, "rope_ cos/sin")?;
        let (b, h, t, d) = self.dims4()?;
        let (max_pos, d_over_2) = rope_check_cs(cos.dims(), b)?;
        let unbatched_rope = cos.rank() == 3;
        if d_over_2 * 2 != d {
            crate::bail!(
                "rope_ requires even d dimension, got d={d}, {:?} {:?}",
                self.shape(),
                cos.shape()
            );
        }
        if pos + t > max_pos {
            crate::bail!(
                "rope_ position out of range, pos={pos} + t={t} > max_pos={max_pos}, {:?} {:?}",
                self.shape(),
                cos.shape()
            );
        }
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let cos_data = cos.storage()?;
        let sin_data = sin.storage()?;
        B::rope(&mut *dst, &*src_data, &*cos_data, &*sin_data, b, h, t, d, pos, unbatched_rope)?;
        Ok(())
    }

    pub fn rope_i_(&self, src: &Self, cos: &Self, sin: &Self, pos: usize) -> Result<()> {
        self.check_not_same_storage(src, "rope_i_")?;
        self.check_not_same_storage(cos, "rope_i_")?;
        self.check_not_same_storage(sin, "rope_i_")?;
        check_same_shape(&self.shape, &src.shape, "rope_i_ src")?;
        check_same_shape(&cos.shape, &sin.shape, "rope_i_ cos/sin")?;
        let (b, h, t, d) = self.dims4()?;
        let (max_pos, d_over_2) = rope_check_cs(cos.dims(), b)?;
        let unbatched_rope = cos.rank() == 3;
        if d_over_2 * 2 != d {
            crate::bail!(
                "rope_i_ requires even d dimension, got d={d}, {:?} {:?}",
                self.shape(),
                cos.shape()
            );
        }
        if pos + t > max_pos {
            crate::bail!(
                "rope_i_ position out of range, pos={pos} + t={t} > max_pos={max_pos}, {:?} {:?}",
                self.shape(),
                cos.shape()
            );
        }
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let cos_data = cos.storage()?;
        let sin_data = sin.storage()?;
        B::rope_i(&mut *dst, &*src_data, &*cos_data, &*sin_data, b, h, t, d, pos, unbatched_rope)?;
        Ok(())
    }

    pub fn reduce_max_(&self, src: &Self, dim: usize) -> Result<()> {
        self.check_not_same_storage(src, "reduce_max_")?;
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::reduce_max(&mut *dst, &*src_data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    pub fn reduce_min_(&self, src: &Self, dim: usize) -> Result<()> {
        self.check_not_same_storage(src, "reduce_min_")?;
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::reduce_min(&mut *dst, &*src_data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    pub fn reduce_argmin_<U: crate::WithDTypeF>(
        dst: &Tensor<i64, B>,
        src: &Tensor<U, B>,
        dim: usize,
    ) -> Result<()> {
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let mut dst_data = dst.storage_mut()?;
        let src_data = src.storage()?;
        B::reduce_argmin(&mut *dst_data, &*src_data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    pub fn reduce_argmax_<U: crate::WithDTypeF>(
        dst: &Tensor<i64, B>,
        src: &Tensor<U, B>,
        dim: usize,
    ) -> Result<()> {
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let mut dst_data = dst.storage_mut()?;
        let src_data = src.storage()?;
        B::reduce_argmax(&mut *dst_data, &*src_data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    pub fn reduce_sum_(&self, src: &Self, dim: usize) -> Result<()> {
        self.check_not_same_storage(src, "reduce_sum_")?;
        let src_dims = src.dims();
        let dim_size = src_dims[dim];
        let outer_size: usize = src_dims[..dim].iter().product::<usize>().max(1);
        let inner_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        B::reduce_sum(&mut *dst, &*src_data, dim_size, outer_size, inner_size)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_(
        &self,
        src: &Self,
        kernel: &Self,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<()> {
        self.check_not_same_storage(src, "conv1d_")?;
        self.check_not_same_storage(kernel, "conv1d_")?;
        let src_dims = src.dims();
        let kernel_dims = kernel.dims();
        if src_dims.len() != 3 {
            crate::bail!(
                "conv1d input must be 3D (batch, in_channels, length), got {:?}",
                src.shape()
            );
        }
        if kernel_dims.len() != 3 {
            crate::bail!(
                "conv1d kernel must be 3D (out_channels, in_channels/groups, kernel_size), got {:?}",
                kernel.shape()
            );
        }

        let batch = src_dims[0];
        let in_channels = src_dims[1];
        let length = src_dims[2];
        let out_channels = kernel_dims[0];
        let kernel_size = kernel_dims[2];

        // Compute output length
        let out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

        let dst_dims = self.dims();
        if dst_dims != [batch, out_channels, out_length] {
            crate::bail!(
                "conv1d output shape mismatch: expected {:?}, got {:?}",
                [batch, out_channels, out_length],
                dst_dims
            );
        }

        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let kernel_data = kernel.storage()?;
        // The kernel is a parameter, so the CPU gemm may keep it packed across calls.
        let parameter = crate::cpu_backend::parameter::<T, _>(&*kernel_data, kernel.elem_count());
        gemm::packed_cache::with_constant(parameter, || {
            B::conv1d(
                &mut *dst,
                &*src_data,
                &*kernel_data,
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
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d_(
        &self,
        src: &Self,
        kernel: &Self,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<()> {
        self.check_not_same_storage(src, "conv_transpose1d_")?;
        self.check_not_same_storage(kernel, "conv_transpose1d_")?;
        let src_dims = src.dims();
        let kernel_dims = kernel.dims();
        if src_dims.len() != 3 {
            crate::bail!(
                "conv_transpose1d input must be 3D (batch, in_channels, length), got {:?}",
                src.shape()
            );
        }
        if kernel_dims.len() != 3 {
            crate::bail!(
                "conv_transpose1d kernel must be 3D (in_channels, out_channels/groups, kernel_size), got {:?}",
                kernel.shape()
            );
        }

        let batch = src_dims[0];
        let in_channels = src_dims[1];
        let length = src_dims[2];
        let out_channels = kernel_dims[1] * groups;
        let kernel_size = kernel_dims[2];

        // Compute output length for transposed convolution
        // out_length = (length - 1) * stride - 2 * padding + kernel_size + output_padding
        let out_length = (length - 1) * stride + kernel_size + output_padding - 2 * padding;

        let dst_dims = self.dims();
        if dst_dims != [batch, out_channels, out_length] {
            crate::bail!(
                "conv_transpose1d output shape mismatch: expected {:?}, got {:?}",
                [batch, out_channels, out_length],
                dst_dims
            );
        }

        let mut dst = self.storage_mut()?;
        let src_data = src.storage()?;
        let kernel_data = kernel.storage()?;
        // The kernel is a parameter, so the CPU gemm may keep it packed across calls.
        let parameter = crate::cpu_backend::parameter::<T, _>(&*kernel_data, kernel.elem_count());
        gemm::packed_cache::with_constant(parameter, || {
            B::conv_transpose1d(
                &mut *dst,
                &*src_data,
                &*kernel_data,
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
        })
    }
}

/// Compute broadcast strides for lhs and rhs given the output shape.
/// Returns (lhs_strides, rhs_strides) where stride is 0 for broadcast dimensions.
fn compute_broadcast_strides(
    out_shape: &[usize],
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> crate::Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Broadcast {
        Lhs,
        Rhs,
        None,
    }
    let out_rank = out_shape.len();
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();

    let mut lro = Vec::with_capacity(out_rank);
    for (i, out_dim) in out_shape.iter().enumerate() {
        let lhs_dim =
            if i >= out_rank - lhs_rank { lhs_shape[i - (out_rank - lhs_rank)] } else { 1 };
        let rhs_dim =
            if i >= out_rank - rhs_rank { rhs_shape[i - (out_rank - rhs_rank)] } else { 1 };
        if lhs_dim != *out_dim && lhs_dim != 1 {
            crate::bail!("broadcast mismatch: lhs dim {i} is {lhs_dim} but output is {out_dim}",);
        }
        if rhs_dim != *out_dim && rhs_dim != 1 {
            crate::bail!("broadcast mismatch: rhs dim {i} is {rhs_dim} but output is {out_dim}",);
        }
        let broadcast = match (lhs_dim == 1, rhs_dim == 1) {
            (true, false) => Broadcast::Lhs,
            (false, true) => Broadcast::Rhs,
            (false, false) => Broadcast::None,
            (true, true) => continue,
        };
        lro.push((broadcast, *out_dim))
    }

    let mut compact_lro = Vec::with_capacity(lro.len());
    for (b, dim) in lro {
        if let Some((last_b, last_dim)) = compact_lro.last_mut()
            && *last_b == b
        {
            *last_dim *= dim;
            continue;
        }
        compact_lro.push((b, dim));
    }

    let out_rank = compact_lro.len();
    let mut lhs_strides = vec![0; out_rank];
    let mut rhs_strides = vec![0; out_rank];
    let mut lhs_stride = 1;
    let mut rhs_stride = 1;
    for i in (0..out_rank).rev() {
        let (b, dim) = compact_lro[i];
        match b {
            Broadcast::Lhs => {
                rhs_strides[i] = rhs_stride;
                lhs_strides[i] = 0;
                rhs_stride *= dim;
            }
            Broadcast::Rhs => {
                lhs_strides[i] = lhs_stride;
                rhs_strides[i] = 0;
                lhs_stride *= dim;
            }
            Broadcast::None => {
                lhs_strides[i] = lhs_stride;
                rhs_strides[i] = rhs_stride;
                lhs_stride *= dim;
                rhs_stride *= dim;
            }
        }
    }
    let out_shape = compact_lro.iter().map(|(_, dim)| *dim).collect();
    Ok((out_shape, lhs_strides, rhs_strides))
}

fn rope_check_cs(cs_dims: &[usize], b_sz: usize) -> Result<(usize, usize)> {
    match *cs_dims {
        [t, d] => Ok((t, d)),
        [b, t, d] => {
            if b != b_sz {
                crate::bail!("inconsistent batch size in rope {b_sz} {cs_dims:?}",)
            }
            Ok((t, d))
        }
        _ => crate::bail!("cos/sin has to be 2D or 3D in rope {b_sz} {cs_dims:?}"),
    }
}
