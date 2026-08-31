use crate::error::Context;
use crate::{BinaryOp, DType, Result, UnaryOp, WithDType, WithDTypeF};
use half::{bf16, f16};
use rayon::prelude::*;
use std::any::Any;

const USE_IM2COL_CONV1D: bool = true;
const USE_COL2IM_CONV1D_TR: bool = true;

/// Whether to use a parallel implementation for this many elements of work. The work must be
/// large enough to amortize the dispatch, and the pool must actually have more than one
/// participant: with a single-thread pool (e.g. RAYON_NUM_THREADS=1) every parallel call would
/// still pay a handoff for no benefit.
fn use_parallelism(work: usize) -> bool {
    work >= ELEMWISE_PAR_THRESHOLD && crate::threadpool::size() > 1
}

fn copy_strided_2d<T: WithDType>(
    dst: &mut [T],
    src: &[T],
    src_offset: usize,
    d0: usize,
    d1: usize,
    s0: usize,
) {
    let copy_row = |i0: usize, dst: &mut [T]| {
        let src_idx = src_offset + i0 * s0;
        dst.copy_from_slice(&src[src_idx..src_idx + d1]);
    };
    let dst = &mut dst[..d0 * d1];
    if use_parallelism(d0 * d1) {
        dst.par_chunks_mut(d1).with_min_len(4).enumerate().for_each(|(i0, dst)| copy_row(i0, dst));
    } else {
        dst.chunks_mut(d1).enumerate().for_each(|(i0, dst)| copy_row(i0, dst));
    }
}

fn copy_strided_3d<T: WithDType>(
    dst: &mut [T],
    src: &[T],
    src_offset: usize,
    dims: [usize; 3],
    strides: [usize; 2],
) {
    let [d0, d1, d2] = dims;
    let [s0, s1] = strides;
    let copy_block = |i0: usize, dst: &mut [T]| {
        let base = src_offset + i0 * s0;
        let mut dst_off = 0;
        for i1 in 0..d1 {
            let src_idx = base + i1 * s1;
            dst[dst_off..dst_off + d2].copy_from_slice(&src[src_idx..src_idx + d2]);
            dst_off += d2;
        }
    };
    let dst = &mut dst[..d0 * d1 * d2];
    if use_parallelism(d0 * d1 * d2) {
        dst.par_chunks_mut(d1 * d2).enumerate().for_each(|(i0, dst)| copy_block(i0, dst));
    } else {
        dst.chunks_mut(d1 * d2).enumerate().for_each(|(i0, dst)| copy_block(i0, dst));
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_<T: WithDType>(
    dst: &mut [T],
    (lhs, lhs_o): (&[T], usize),
    (rhs, rhs_o): (&[T], usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    lhs_b_stride: usize,
    rhs_b_stride: usize,
    (dst_cs, dst_rs): (usize, usize),
    (lhs_cs, lhs_rs): (usize, usize),
    (rhs_cs, rhs_rs): (usize, usize),
) -> Result<()> {
    let lhs = &lhs[lhs_o..];
    let rhs = &rhs[rhs_o..];

    // Column stripes, run on xn's own pool rather than letting `gemm` reach for rayon.
    //
    // Two pools would otherwise both be live inside a decode step -- this one for the f32
    // matmuls and xn's for everything else -- and their idle workers spin against each other
    // for cores. Splitting by output column is what `Parallelism::Rayon` does internally
    // anyway; the cost is that each stripe re-packs the lhs panel, which is negligible at the
    // batch sizes decode uses.
    let nth = crate::threadpool::size();
    // One stripe per participant. Measured against 0.25x, 0.5x, 2x and 4x that count: 1x and
    // 2x tie, everything else is worse. Splitting the rows instead when the output is too
    // narrow to stripe was also tried and nets nothing, so narrow outputs stay serial.
    let stripes = if nth > 1 && n >= nth * 4 && m * n * k >= 1 << 14 { nth } else { 1 };

    for b_idx in 0..lhs_b {
        let dst = &mut dst[b_idx * m * n..(b_idx + 1) * m * n];
        let lhs = &lhs[b_idx * lhs_b_stride..];
        let rhs = &rhs[b_idx * rhs_b_stride..];
        if stripes > 1 {
            let dst_p = dst.as_mut_ptr() as usize;
            let lhs_p = lhs.as_ptr() as usize;
            let rhs_p = rhs.as_ptr() as usize;
            crate::threadpool::par_units(stripes, |s| {
                let per = n.div_ceil(stripes);
                let n0 = (s * per).min(n);
                let n1 = (n0 + per).min(n);
                if n0 == n1 {
                    return;
                }
                // SAFETY: stripes own disjoint output columns, and every offset stays inside
                // the slices borrowed above, which outlive the dispatch.
                unsafe {
                    gemm::gemm(
                        m,
                        n1 - n0,
                        k,
                        (dst_p as *mut T).add(n0 * dst_cs),
                        dst_cs as isize,
                        dst_rs as isize,
                        false,
                        lhs_p as *const T,
                        lhs_cs as isize,
                        lhs_rs as isize,
                        (rhs_p as *const T).add(n0 * rhs_cs),
                        rhs_cs as isize,
                        rhs_rs as isize,
                        T::zero(),
                        T::one(),
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    )
                }
            });
            continue;
        }
        unsafe {
            gemm::gemm(
                /* m: usize = */ m,
                /* n: usize = */ n,
                /* k: usize = */ k,
                /* dst: *mut T = */ dst.as_mut_ptr(),
                /* dst_cs: isize = */ dst_cs as isize,
                /* dst_rs: isize = */ dst_rs as isize,
                /* read_dst: bool = */ false,
                /* lhs: *const T = */ lhs.as_ptr(),
                /* lhs_cs: isize = */ lhs_cs as isize,
                /* lhs_rs: isize = */ lhs_rs as isize,
                /* rhs: *const T = */ rhs.as_ptr(),
                /* rhs_cs: isize = */ rhs_cs as isize,
                /* rhs_rs: isize = */ rhs_rs as isize,
                /* alpha: T = */ T::zero(),
                /* beta: T = */ T::one(),
                /* conj_dst: bool = */ false,
                /* conj_lhs: bool = */ false,
                /* conj_rhs: bool = */ false,
                gemm::Parallelism::None,
            )
        }
    }
    Ok(())
}

impl crate::Backend for crate::CpuDevice {
    type Storage<T: WithDType> = Vec<T>;

    fn name(&self) -> String {
        "cpu".to_string()
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn storage_len<T: WithDType>(storage: &Self::Storage<T>) -> usize {
        storage.len()
    }

    unsafe fn alloc_uninit<T: WithDType>(len: usize, _: &Self) -> Result<Self::Storage<T>> {
        // All supported dtypes are plain-old-data for which any bit pattern is valid, and the
        // trait contract requires the caller to initialize the memory before reading it.
        let mut v = Vec::with_capacity(len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            v.set_len(len)
        };
        Ok(v)
    }

    fn from_vec<T: WithDType>(v: Vec<T>, _: &Self) -> Result<Self::Storage<T>> {
        Ok(v)
    }

    fn data<T: WithDType>(src: &Self::Storage<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        Ok(std::borrow::Cow::Borrowed(&src[..len]))
    }

    fn bin_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        match op {
            BinaryOp::Add => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| *d += s),
            BinaryOp::Sub => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| *d -= s),
            BinaryOp::Mul => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| *d *= s),
            BinaryOp::Div => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| *d /= s),
            BinaryOp::Maximum => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| {
                if s > *d {
                    *d = s
                }
            }),
            BinaryOp::Minimum => apply_bin_assign(&mut dst[..len], &src[..len], |d, s| {
                if s < *d {
                    *d = s
                }
            }),
        }
        Ok(())
    }

    fn inplace_unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        let chunk = unary_chunk(op, len);
        match op {
            UnaryOp::Cos => apply_inplace_unary(chunk, &mut dst[..len], |v| *v = v.cos()),
            UnaryOp::Sin => apply_inplace_unary(chunk, &mut dst[..len], |v| *v = v.sin()),
            UnaryOp::Exp => apply_inplace_unary(chunk, &mut dst[..len], |v| *v = v.exp()),
            UnaryOp::Log => apply_inplace_unary(chunk, &mut dst[..len], |v| *v = v.ln()),
            UnaryOp::Neg => apply_inplace_unary(chunk, &mut dst[..len], |v| *v = T::zero() - *v),
            UnaryOp::Sqr => apply_inplace_unary(chunk, &mut dst[..len], |v| *v = *v * *v),
            UnaryOp::Sqrt => apply_inplace_unary(chunk, &mut dst[..len], |v| *v = v.sqrt()),
            UnaryOp::Rsqrt => {
                apply_inplace_unary(chunk, &mut dst[..len], |v| *v = T::one() / v.sqrt())
            }
            UnaryOp::Abs => apply_inplace_unary(chunk, &mut dst[..len], |v| *v = v.abs()),
            UnaryOp::GeluErf => {
                let sqrt_2_inv = std::f32::consts::FRAC_1_SQRT_2;
                apply_inplace_unary(chunk, &mut dst[..len], |v| {
                    let x = v.to_f32();
                    let erf_val = libm::erff(x * sqrt_2_inv);
                    *v = T::from_f32(x * 0.5 * (1.0 + erf_val));
                })
            }
            UnaryOp::Elu { alpha } => apply_inplace_unary(chunk, &mut dst[..len], |v| {
                let x = v.to_f32();
                *v = T::from_f32(if x > 0.0 { x } else { alpha * (x.exp() - 1.0) });
            }),
            UnaryOp::Relu => apply_inplace_unary(chunk, &mut dst[..len], |v| {
                if *v < T::zero() {
                    *v = T::zero()
                }
            }),
            UnaryOp::Silu => apply_inplace_unary(chunk, &mut dst[..len], |v| {
                *v = *v / (T::one() + (T::zero() - *v).exp())
            }),
            UnaryOp::Tanh => apply_inplace_unary(chunk, &mut dst[..len], |v| *v = v.tanh()),
            UnaryOp::Sigmoid => apply_inplace_unary(chunk, &mut dst[..len], |v| {
                *v = T::one() / (T::one() + (T::zero() - *v).exp())
            }),
        }
        Ok(())
    }

    fn unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        let chunk = unary_chunk(op, len);
        match op {
            UnaryOp::Cos => apply_unary(chunk, &mut dst[..len], &src[..len], |s| s.cos()),
            UnaryOp::Sin => apply_unary(chunk, &mut dst[..len], &src[..len], |s| s.sin()),
            UnaryOp::Exp => apply_unary(chunk, &mut dst[..len], &src[..len], |s| s.exp()),
            UnaryOp::Log => apply_unary(chunk, &mut dst[..len], &src[..len], |s| s.ln()),
            UnaryOp::Neg => apply_unary(chunk, &mut dst[..len], &src[..len], |s| T::zero() - s),
            UnaryOp::Sqr => apply_unary(chunk, &mut dst[..len], &src[..len], |s| s * s),
            UnaryOp::Sqrt => apply_unary(chunk, &mut dst[..len], &src[..len], |s| s.sqrt()),
            UnaryOp::Rsqrt => {
                apply_unary(chunk, &mut dst[..len], &src[..len], |s| T::one() / s.sqrt())
            }
            UnaryOp::Abs => apply_unary(chunk, &mut dst[..len], &src[..len], |s| s.abs()),
            UnaryOp::GeluErf => {
                let sqrt_2_inv = std::f32::consts::FRAC_1_SQRT_2;
                apply_unary(chunk, &mut dst[..len], &src[..len], |s| {
                    let x = s.to_f32();
                    let erf_val = libm::erff(x * sqrt_2_inv);
                    T::from_f32(x * 0.5 * (1.0 + erf_val))
                })
            }
            UnaryOp::Elu { alpha } => apply_unary(chunk, &mut dst[..len], &src[..len], |s| {
                let x = s.to_f32();
                T::from_f32(if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            }),
            UnaryOp::Relu => {
                let zero = T::zero();
                apply_unary(
                    chunk,
                    &mut dst[..len],
                    &src[..len],
                    |s| if s < zero { zero } else { s },
                )
            }
            UnaryOp::Silu => apply_unary(chunk, &mut dst[..len], &src[..len], |s| {
                s / (T::one() + (T::zero() - s).exp())
            }),
            UnaryOp::Tanh => apply_unary(chunk, &mut dst[..len], &src[..len], |s| s.tanh()),
            UnaryOp::Sigmoid => apply_unary(chunk, &mut dst[..len], &src[..len], |s| {
                T::one() / (T::one() + (T::zero() - s).exp())
            }),
        }
        Ok(())
    }

    fn binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        match op {
            BinaryOp::Add => apply_binary(&mut dst[..len], &lhs[..len], &rhs[..len], |a, b| a + b),
            BinaryOp::Sub => apply_binary(&mut dst[..len], &lhs[..len], &rhs[..len], |a, b| a - b),
            BinaryOp::Mul => apply_binary(&mut dst[..len], &lhs[..len], &rhs[..len], |a, b| a * b),
            BinaryOp::Div => apply_binary(&mut dst[..len], &lhs[..len], &rhs[..len], |a, b| a / b),
            BinaryOp::Maximum => {
                apply_binary(
                    &mut dst[..len],
                    &lhs[..len],
                    &rhs[..len],
                    |a, b| {
                        if a > b { a } else { b }
                    },
                )
            }
            BinaryOp::Minimum => {
                apply_binary(
                    &mut dst[..len],
                    &lhs[..len],
                    &rhs[..len],
                    |a, b| {
                        if a < b { a } else { b }
                    },
                )
            }
        }
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
        // An fma per element, so the fixed chunk applies (see `unary_chunk`).
        let chunk = ELEMWISE_CHUNK;
        if add == zero && scale == one {
            Self::copy(dst, src, len)
        } else if add == zero {
            apply_unary(chunk, &mut dst[..len], &src[..len], |s| s * scale);
            Ok(())
        } else if scale == one {
            apply_unary(chunk, &mut dst[..len], &src[..len], |s| s + add);
            Ok(())
        } else {
            apply_unary(chunk, &mut dst[..len], &src[..len], |s| s * scale + add);
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
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
        for i1 in 0..d1 {
            let dst_idx = i1 * dst_s + dst_o;
            let src_idx = i1 * src_s + src_o;
            let dst = &mut dst[dst_idx..dst_idx + d2];
            let src = &src[src_idx..src_idx + d2];
            dst.copy_from_slice(src)
        }
        Ok(())
    }

    fn transpose<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()> {
        if dim1 == dim2 || dims.iter().filter(|v| **v != 1).count() <= 1 {
            dst.copy_from_slice(src);
        } else {
            let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
            let d_j = dims[dim1 + 1..dim2].iter().product::<usize>();
            let d_k = dims[(dim2 + 1)..].iter().product::<usize>();
            let d1 = dims[dim1];
            let d2 = dims[dim2];
            let parallel = use_parallelism(dst.len());
            if d_k == 1 {
                // The transposed dimension is the innermost one: use a tiled 2d transpose so
                // that both reads and writes stay within cached lines for a whole tile.
                const TILE: usize = 32;
                let transpose_block = |i: usize, dst: &mut [T]| {
                    let src = &src[i * d1 * d_j * d2..(i + 1) * d1 * d_j * d2];
                    for j in 0..d_j {
                        for a1_t in (0..d1).step_by(TILE) {
                            let a1_end = usize::min(a1_t + TILE, d1);
                            for a2_t in (0..d2).step_by(TILE) {
                                let a2_end = usize::min(a2_t + TILE, d2);
                                for a1 in a1_t..a1_end {
                                    let src_base = a1 * d_j * d2 + j * d2;
                                    let dst_base = j * d1 + a1;
                                    for a2 in a2_t..a2_end {
                                        dst[a2 * d_j * d1 + dst_base] = src[src_base + a2];
                                    }
                                }
                            }
                        }
                    }
                };
                if parallel {
                    dst.par_chunks_mut(d2 * d_j * d1)
                        .enumerate()
                        .for_each(|(i, dst)| transpose_block(i, dst));
                } else {
                    dst.chunks_mut(d2 * d_j * d1)
                        .enumerate()
                        .for_each(|(i, dst)| transpose_block(i, dst));
                }
            } else {
                let transpose_block = |i: usize, dst: &mut [T]| {
                    let src = &src[i * d1 * d_j * d2 * d_k..];
                    for a1 in 0..d1 {
                        for j in 0..d_j {
                            for a2 in 0..d2 {
                                let src_idx = a1 * d_j * d2 * d_k + j * d2 * d_k + a2 * d_k;
                                let dst_idx = a2 * d_j * d1 * d_k + j * d1 * d_k + a1 * d_k;
                                dst[dst_idx..dst_idx + d_k]
                                    .copy_from_slice(&src[src_idx..src_idx + d_k]);
                            }
                        }
                    }
                };
                if parallel {
                    dst.par_chunks_mut(d2 * d_j * d1 * d_k)
                        .enumerate()
                        .for_each(|(i, dst)| transpose_block(i, dst));
                } else {
                    dst.chunks_mut(d2 * d_j * d1 * d_k)
                        .enumerate()
                        .for_each(|(i, dst)| transpose_block(i, dst));
                }
            }
        }
        Ok(())
    }

    fn copy<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        l: usize,
    ) -> Result<()> {
        if !use_parallelism(l) {
            dst[..l].copy_from_slice(&src[..l]);
        } else {
            dst[..l]
                .par_chunks_mut(ELEMWISE_CHUNK)
                .zip(src[..l].par_chunks(ELEMWISE_CHUNK))
                .for_each(|(d, s)| d.copy_from_slice(s));
        }
        Ok(())
    }

    fn to_dtype<T: WithDType, U: WithDType>(
        dst: &mut Self::Storage<U>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()> {
        let src_any: &dyn Any = src;
        let dst_any: &mut dyn Any = dst;
        macro_rules! cast {
            ($src_ty:ty, $dst_ty:ty, |$v:ident| $conv:expr) => {{
                let src = src_any
                    .downcast_ref::<Vec<$src_ty>>()
                    .context("to_dtype src downcast failed")?;
                let dst = dst_any
                    .downcast_mut::<Vec<$dst_ty>>()
                    .context("to_dtype dst downcast failed")?;
                for (d, s) in dst[..len].iter_mut().zip(src[..len].iter()) {
                    let $v = *s;
                    *d = $conv;
                }
            }};
        }
        use DType::*;
        match (T::DTYPE, U::DTYPE) {
            // same type
            (F16, F16) => cast!(f16, f16, |v| v),
            (BF16, BF16) => cast!(bf16, bf16, |v| v),
            (F32, F32) => cast!(f32, f32, |v| v),
            (I64, I64) => cast!(i64, i64, |v| v),
            (U8, U8) => cast!(u8, u8, |v| v),
            // float <-> float
            (F32, F16) => cast!(f32, f16, |v| f16::from_f32(v)),
            (F32, BF16) => cast!(f32, bf16, |v| bf16::from_f32(v)),
            (F16, F32) => cast!(f16, f32, |v| v.to_f32()),
            (BF16, F32) => cast!(bf16, f32, |v| v.to_f32()),
            (F16, BF16) => cast!(f16, bf16, |v| bf16::from_f32(v.to_f32())),
            (BF16, F16) => cast!(bf16, f16, |v| f16::from_f32(v.to_f32())),
            // float -> int
            (F32, I64) => cast!(f32, i64, |v| v as i64),
            (F32, U8) => cast!(f32, u8, |v| v as u8),
            (F16, I64) => cast!(f16, i64, |v| v.to_f32() as i64),
            (F16, U8) => cast!(f16, u8, |v| v.to_f32() as u8),
            (BF16, I64) => cast!(bf16, i64, |v| v.to_f32() as i64),
            (BF16, U8) => cast!(bf16, u8, |v| v.to_f32() as u8),
            // int -> float
            (I64, F32) => cast!(i64, f32, |v| v as f32),
            (I64, F16) => cast!(i64, f16, |v| f16::from_f32(v as f32)),
            (I64, BF16) => cast!(i64, bf16, |v| bf16::from_f32(v as f32)),
            (U8, F32) => cast!(u8, f32, |v| v as f32),
            (U8, F16) => cast!(u8, f16, |v| f16::from_f32(v as f32)),
            (U8, BF16) => cast!(u8, bf16, |v| bf16::from_f32(v as f32)),
            // int <-> int
            (I64, U8) => cast!(i64, u8, |v| v as u8),
            (U8, I64) => cast!(u8, i64, |v| v as i64),
        }
        Ok(())
    }

    fn rand_uniform(dst: &mut Self::Storage<f32>, len: usize, lo: f32, up: f32) -> Result<()> {
        use rand::Rng;
        let range = up - lo;
        let fill = |chunk: &mut [f32]| {
            let mut rng = rand::rng();
            for v in chunk.iter_mut() {
                *v = rng.random::<f32>() * range + lo;
            }
        };
        if use_parallelism(len) {
            dst[..len].par_chunks_mut(ELEMWISE_CHUNK).for_each(fill);
        } else {
            fill(&mut dst[..len]);
        }
        Ok(())
    }

    fn randn(dst: &mut Self::Storage<f32>, len: usize, mean: f32, std: f32) -> Result<()> {
        use rand_distr::Distribution;

        let distr = match rand_distr::Normal::<f32>::new(mean, std) {
            Ok(d) => d,
            Err(e) => crate::bail!("failed to create normal distribution for randn: {e}"),
        };
        let fill = |chunk: &mut [f32]| {
            let mut rng = rand::rng();
            for v in chunk.iter_mut() {
                *v = distr.sample(&mut rng);
            }
        };
        if use_parallelism(len) {
            dst[..len].par_chunks_mut(ELEMWISE_CHUNK).for_each(fill);
        } else {
            fill(&mut dst[..len]);
        }
        Ok(())
    }

    fn fill<T: WithDType>(dst: &mut Self::Storage<T>, v: T, l: usize) -> Result<()> {
        if !use_parallelism(l) {
            dst[..l].fill(v);
        } else {
            dst[..l].par_chunks_mut(ELEMWISE_CHUNK).for_each(|d| d.fill(v));
        }
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
        if dst.len() != b * h * t * d {
            crate::bail!("rope unexpected size for dst {} {b} {h} {t} {d}", dst.len())
        }
        if src.len() != b * h * t * d {
            crate::bail!("rope unexpected size for src {} {b} {h} {t} {d}", src.len())
        }
        let cos = &cos[pos * d / 2..];
        let sin = &sin[pos * d / 2..];
        // Same gating as `rope_i`.
        let row = |bh_i: usize, src: &[T], dst: &mut [T]| {
            let base = if unbatched_rope { (bh_i / h) * t * d / 2 } else { 0 };
            for i_t in 0..t {
                for i_d in 0..d / 2 {
                    let i1 = i_t * d + i_d;
                    let i2 = i1 + d / 2;
                    let i_cs = base + i_t * (d / 2) + i_d;
                    dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                    dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                }
            }
        };
        if use_parallelism(b * h * t * d) {
            src.par_chunks(t * d)
                .zip(dst.par_chunks_mut(t * d))
                .enumerate()
                .for_each(|(bh_i, (s, d))| row(bh_i, s, d));
        } else {
            src.chunks(t * d)
                .zip(dst.chunks_mut(t * d))
                .enumerate()
                .for_each(|(bh_i, (s, d))| row(bh_i, s, d));
        }
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
        if dst.len() != b * h * t * d {
            crate::bail!("rope-i unexpected size for dst {} {b} {h} {t} {d}", dst.len())
        }
        if src.len() != b * h * t * d {
            crate::bail!("rope-i unexpected size for src {} {b} {h} {t} {d}", src.len())
        }
        let cos = &cos[pos * d / 2..];
        let sin = &sin[pos * d / 2..];
        // One (b, h) row per chunk. During single-token decode a row is `d` elements and there
        // are only b*h of them, so the rayon fork/join costs far more than the arithmetic;
        // gate it the same way the elementwise ops are gated.
        let row = |bh_i: usize, src: &[T], dst: &mut [T]| {
            // The batch offset is invariant across the row, so compute it once.
            let base = if unbatched_rope { (bh_i / h) * t * d / 2 } else { 0 };
            for i_over_2 in 0..t * d / 2 {
                let i = 2 * i_over_2;
                let rope_i = base + i_over_2;
                dst[i] = src[i] * cos[rope_i] - src[i + 1] * sin[rope_i];
                dst[i + 1] = src[i] * sin[rope_i] + src[i + 1] * cos[rope_i];
            }
        };
        if use_parallelism(b * h * t * d) {
            src.par_chunks(t * d)
                .zip(dst.par_chunks_mut(t * d))
                .enumerate()
                .for_each(|(bh_i, (s, d))| row(bh_i, s, d));
        } else {
            src.chunks(t * d)
                .zip(dst.chunks_mut(t * d))
                .enumerate()
                .for_each(|(bh_i, (s, d))| row(bh_i, s, d));
        }
        Ok(())
    }

    #[cfg(feature = "accelerate")]
    fn gemm<T: WithDType>(
        dst: &mut Self::Storage<T>,
        (lhs, lhs_o): (&Self::Storage<T>, usize),
        (rhs, rhs_o): (&Self::Storage<T>, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        lhs_b_stride: usize,
        rhs_b_stride: usize,
        (dst_cs, dst_rs): (usize, usize),
        (lhs_cs, lhs_rs): (usize, usize),
        (rhs_cs, rhs_rs): (usize, usize),
    ) -> Result<()> {
        let lhs = &lhs[lhs_o..];
        let rhs = &rhs[rhs_o..];
        let (lda, transa) = if (rhs_cs == 1 || n == 1) && (rhs_rs == n || k == 1) {
            (n as i32, b'N')
        } else if rhs_cs == k && rhs_rs == 1 {
            (k as i32, b'T')
        } else {
            return gemm_(
                dst,
                (lhs, lhs_o),
                (rhs, rhs_o),
                m,
                n,
                k,
                lhs_b,
                lhs_b_stride,
                rhs_b_stride,
                (dst_cs, dst_rs),
                (lhs_cs, lhs_rs),
                (rhs_cs, rhs_rs),
            );
        };
        // The b tensor has dims batching, m, k (lhs)
        let (ldb, transb) = if (lhs_cs == 1 || k == 1) && (lhs_rs == k || m == 1) {
            (k as i32, b'N')
        } else if lhs_cs == m && lhs_rs == 1 {
            (m as i32, b'T')
        } else {
            return gemm_(
                dst,
                (lhs, lhs_o),
                (rhs, rhs_o),
                m,
                n,
                k,
                lhs_b,
                lhs_b_stride,
                rhs_b_stride,
                (dst_cs, dst_rs),
                (lhs_cs, lhs_rs),
                (rhs_cs, rhs_rs),
            );
        };

        match T::DTYPE {
            crate::DType::F32 => {
                for b_idx in 0..lhs_b {
                    let dst_p = &mut dst[b_idx * m * n..(b_idx + 1) * m * n];
                    let lhs_p = &lhs[b_idx * lhs_b_stride..];
                    let rhs_p = &rhs[b_idx * rhs_b_stride..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f32;
                        let b = lhs_p.as_ptr() as *const f32;
                        let c = dst_p.as_mut_ptr() as *mut f32;
                        crate::accelerate::sgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            _ => {
                return gemm_(
                    dst,
                    (lhs, lhs_o),
                    (rhs, rhs_o),
                    m,
                    n,
                    k,
                    lhs_b,
                    lhs_b_stride,
                    rhs_b_stride,
                    (dst_cs, dst_rs),
                    (lhs_cs, lhs_rs),
                    (rhs_cs, rhs_rs),
                );
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "accelerate"))]
    fn gemm<T: WithDType>(
        dst: &mut Self::Storage<T>,
        (lhs, lhs_o): (&Self::Storage<T>, usize),
        (rhs, rhs_o): (&Self::Storage<T>, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        lhs_b_stride: usize,
        rhs_b_stride: usize,
        (dst_cs, dst_rs): (usize, usize),
        (lhs_cs, lhs_rs): (usize, usize),
        (rhs_cs, rhs_rs): (usize, usize),
    ) -> Result<()> {
        gemm_(
            dst,
            (lhs, lhs_o),
            (rhs, rhs_o),
            m,
            n,
            k,
            lhs_b,
            lhs_b_stride,
            rhs_b_stride,
            (dst_cs, dst_rs),
            (lhs_cs, lhs_rs),
            (rhs_cs, rhs_rs),
        )
    }

    fn copy_strided<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        src_offset: usize,
        dims: &[usize],
        src_strides: &[usize],
    ) -> Result<()> {
        let rank = dims.len();
        let total: usize = dims.iter().product();
        if rank == 1 && src_strides[0] == 1 {
            dst[..total].copy_from_slice(&src[src_offset..src_offset + total]);
            return Ok(());
        }
        if rank == 2 && src_strides[1] == 1 {
            copy_strided_2d(dst, src, src_offset, dims[0], dims[1], src_strides[0]);
            return Ok(());
        }
        if rank == 3 && src_strides[2] == 1 {
            copy_strided_3d(
                dst,
                src,
                src_offset,
                [dims[0], dims[1], dims[2]],
                [src_strides[0], src_strides[1]],
            );
            return Ok(());
        }
        if rank == 1 {
            let s0 = src_strides[0];
            for (i, dst_elem) in dst.iter_mut().enumerate().take(total) {
                *dst_elem = src[src_offset + i * s0];
            }
            return Ok(());
        }
        // General fallback: the innermost stride is not 1 (e.g. a transposed view). Process
        // the last two dims as a tiled 2d gather so reads and writes both get cache-line
        // reuse within a tile, and parallelize over the outer dims.
        let d_a = dims[rank - 2];
        let d_b = dims[rank - 1];
        let s_a = src_strides[rank - 2];
        let s_b = src_strides[rank - 1];
        let outer_dims = &dims[..rank - 2];
        let outer_strides = &src_strides[..rank - 2];
        const TILE: usize = 32;
        let tiled_2d_gather = |dst: &mut [T], off: usize| {
            // dst is a chunk of consecutive `a` rows starting at row `a_t`; `off` already
            // includes the `a_t * s_a` component.
            let a_cnt = dst.len() / d_b;
            for b_t in (0..d_b).step_by(TILE) {
                let b_end = usize::min(b_t + TILE, d_b);
                for a in 0..a_cnt {
                    let src_base = off + a * s_a;
                    let dst_base = a * d_b;
                    for b in b_t..b_end {
                        dst[dst_base + b] = src[src_base + b * s_b];
                    }
                }
            }
        };
        let n_outer = total / (d_a * d_b);
        let outer_offset = |o: usize| {
            let mut rem = o;
            let mut off = src_offset;
            for d in (0..outer_dims.len()).rev() {
                off += (rem % outer_dims[d]) * outer_strides[d];
                rem /= outer_dims[d];
            }
            off
        };
        if !use_parallelism(total) {
            // Serially, a plain row-major gather beats the tiled one: the strided read lines
            // for one output row generally stay in L2, and the tile bookkeeping costs more
            // than it saves. Tiling pays off once several threads share the caches.
            for o in 0..n_outer {
                let off = outer_offset(o);
                let dst = &mut dst[o * d_a * d_b..(o + 1) * d_a * d_b];
                for a in 0..d_a {
                    let src_base = off + a * s_a;
                    for (b, d) in dst[a * d_b..(a + 1) * d_b].iter_mut().enumerate() {
                        *d = src[src_base + b * s_b];
                    }
                }
            }
        } else if n_outer >= crate::get_num_threads() {
            dst[..total]
                .par_chunks_mut(d_a * d_b)
                .enumerate()
                .for_each(|(o, dst)| tiled_2d_gather(dst, outer_offset(o)));
        } else {
            // Few outer blocks (e.g. a plain 2d transpose): parallelize over row tiles
            // within each block instead.
            for o in 0..n_outer {
                let off = outer_offset(o);
                dst[o * d_a * d_b..(o + 1) * d_a * d_b]
                    .par_chunks_mut(TILE * d_b)
                    .enumerate()
                    .for_each(|(a_t, dst)| tiled_2d_gather(dst, off + a_t * TILE * s_a));
            }
        }
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
        let left_size: usize = src_dims[..dim].iter().product();
        let right_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let src_dim_size = src_dims[dim];
        let dst_dim_size = dst_dims[dim];

        for left in 0..left_size {
            for i in 0..src_dim_size {
                for right in 0..right_size {
                    let src_flat = left * src_dim_size * right_size + i * right_size + right;
                    let idx = ids[src_flat] as usize;
                    let dst_flat = left * dst_dim_size * right_size + idx * right_size + right;
                    dst[dst_flat] = src[src_flat];
                }
            }
        }
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
        let src_dim_size = dims[dim];

        for left in 0..left_size {
            for (i, &idx) in ids.iter().enumerate().take(num_ids) {
                let dst_offset = left * num_ids * right_size + i * right_size;
                if idx == -1 {
                    // An index of -1 selects zeros rather than a row of the source.
                    dst[dst_offset..dst_offset + right_size].fill(<T as num_traits::Zero>::zero());
                } else {
                    let src_offset = left * src_dim_size * right_size + idx as usize * right_size;
                    dst[dst_offset..dst_offset + right_size]
                        .copy_from_slice(&src[src_offset..src_offset + right_size]);
                }
            }
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
        for idx_b in 0..bh {
            for idx1 in 0..t1 {
                // Query at position offset + idx1 can attend to keys at positions 0..=offset+idx1
                // So mask positions where idx2 > offset + idx1
                for idx2 in (offset + idx1 + 1)..t2 {
                    let idx = idx_b * t1 * t2 + idx1 * t2 + idx2;
                    dst[idx] = T::neg_infinity()
                }
            }
        }
        Ok(())
    }

    fn softmax<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()> {
        let src = &src[..d * dim_m1];
        let dst = &mut dst[..d * dim_m1];
        // Rows are independent, so this is gated the same way as the elementwise ops: during
        // autoregressive decode a row is a single short attention vector, and the fork/join
        // cost dominates the handful of exps it saves.
        let softmax_row = |src: &[T], dst: &mut [T]| {
            let mut max = T::neg_infinity();
            for &v in src.iter() {
                max = T::max(v, max)
            }
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = (*s - max).exp();
            }
            let sum_exp = dst.iter().map(|v| <T as WithDTypeF>::to_f32(*v)).sum::<f32>();
            for d in dst.iter_mut() {
                *d = T::from_f32(d.to_f32() / sum_exp)
            }
        };
        if use_parallelism(d * dim_m1) {
            crate::threadpool::par_chunks_zip(dst, dim_m1, src, dim_m1, |_, d, s| {
                softmax_row(s, d)
            });
        } else {
            src.chunks(dim_m1).zip(dst.chunks_mut(dim_m1)).for_each(|(s, d)| softmax_row(s, d));
        }
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
        let src = &src[..d * dim_m1];
        let dst = &mut dst[..d * dim_m1];
        crate::threadpool::par_chunks_zip(dst, dim_m1, src, dim_m1, |_, dst, src| {
            let sum2 = src.iter().map(|&v| v.to_f32() * v.to_f32()).sum::<f32>();
            let m = (sum2 / dim_m1 as f32 + eps).sqrt();
            for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                *d = T::from_f32((*s).to_f32() / m * (*alpha).to_f32())
            }
        });
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
        let src = &src[..d * dim_m1];
        let dst = &mut dst[..d * dim_m1];
        let weight = &weight[..dim_m1];
        let bias = &bias[..dim_m1];
        crate::threadpool::par_chunks_zip(dst, dim_m1, src, dim_m1, |_, dst, src| {
            // Compute mean
            let sum: f32 = src.iter().map(|&v| v.to_f32()).sum();
            let mean = sum / dim_m1 as f32;

            // Compute variance (always uses mean)
            let var: f32 = src
                .iter()
                .map(|&v| {
                    let diff = v.to_f32() - mean;
                    diff * diff
                })
                .sum::<f32>()
                / dim_m1 as f32;

            let inv_std = 1.0 / (var + eps).sqrt();

            // Normalize and apply weight/bias
            for i in 0..dim_m1 {
                let centered = if remove_mean { src[i].to_f32() - mean } else { src[i].to_f32() };
                let normalized = centered * inv_std;
                dst[i] = T::from_f32(normalized * weight[i].to_f32() + bias[i].to_f32());
            }
        });
        Ok(())
    }

    fn reduce_max<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        reduce_combine(dst, src, dim_size, outer_size, inner_size, T::neg_infinity(), |a, b| {
            T::max(a, b)
        });
        Ok(())
    }

    fn reduce_min<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        reduce_combine(dst, src, dim_size, outer_size, inner_size, T::infinity(), |a, b| {
            T::min(a, b)
        });
        Ok(())
    }

    fn reduce_argmin<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        reduce_arg(dst, src, dim_size, outer_size, inner_size, T::infinity(), |v, best| {
            v.to_f32() < best.to_f32()
        });
        Ok(())
    }

    fn reduce_argmax<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        reduce_arg(dst, src, dim_size, outer_size, inner_size, T::neg_infinity(), |v, best| {
            v.to_f32() > best.to_f32()
        });
        Ok(())
    }

    fn reduce_sum<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        reduce_combine(dst, src, dim_size, outer_size, inner_size, T::zero(), |a, b| a + b);
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
        match op {
            BinaryOp::Add => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    a + b
                })
            }
            BinaryOp::Sub => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    a - b
                })
            }
            BinaryOp::Mul => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    a * b
                })
            }
            BinaryOp::Div => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    a / b
                })
            }
            BinaryOp::Maximum => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    if a > b { a } else { b }
                })
            }
            BinaryOp::Minimum => {
                broadcast_binary_op(dst, lhs, rhs, dst_shape, lhs_strides, rhs_strides, |a, b| {
                    if a < b { a } else { b }
                })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
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
        if USE_IM2COL_CONV1D && groups == 1 {
            // IM2COL approach: transform conv1d into matrix multiplication
            // 1. Im2Col: transform input [B, C, L] -> [B, L_out, C * K]
            // 2. Matmul: [B, L_out, C*K] x [C*K, out_channels] -> [B, L_out, out_channels]
            // 3. Transpose result to [B, out_channels, L_out]

            let k = in_channels * kernel_size;

            // Step 1: Im2Col transformation
            let col = im2col1d(
                src,
                batch,
                in_channels,
                length,
                out_length,
                kernel_size,
                stride,
                padding,
                dilation,
            );

            // Step 2: Matrix multiplication
            // col is [B, L_out, K] where K = in_channels * kernel_size
            // kernel is [out_channels, in_channels, kernel_size] = [out_channels, K]
            // We want [B, L_out, out_channels]
            let mut result = vec![T::zero(); batch * out_length * out_channels];

            Self::gemm(
                &mut result,
                (&col, 0),
                (kernel, 0),
                /* m */ out_length,
                /* n */ out_channels,
                /* k */ k,
                batch,
                out_length * k,
                0,
                (1, out_channels),
                (1, k),
                (k, 1),
            )?;

            // Step 3: Transpose from [B, L_out, out_channels] to [B, out_channels, L_out]
            Self::transpose(dst, &result, 1, 2, &[batch, out_length, out_channels])
        } else {
            // Fallback: original implementation for grouped convolutions
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

    const FUSED_SDPA_DECODE: bool = true;
    const SDPA_MAX_HEAD_DIM: usize = CPU_SDPA_MAX_HEAD_DIM;

    #[allow(clippy::too_many_arguments)]
    fn sdpa_decode<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        (q, q_off): (&Self::Storage<T>, usize),
        (k, k_off): (&Self::Storage<T>, usize),
        (v, v_off): (&Self::Storage<T>, usize),
        mask: Option<(&Self::Storage<T>, usize)>,
        kv_batch_stride: usize,
        b: usize,
        h: usize,
        d: usize,
        kv: usize,
        scale: f32,
    ) -> Result<()> {
        if d > CPU_SDPA_MAX_HEAD_DIM {
            crate::bail!("sdpa_decode: head dim {d} exceeds {CPU_SDPA_MAX_HEAD_DIM}")
        }
        if kv == 0 {
            crate::bail!("sdpa_decode: empty kv")
        }
        let hd = h * d;
        // One (batch, head) pair per unit of work; each owns a disjoint `d`-wide slice of dst.
        //
        // Streaming ("online") softmax: carry a running max, denominator and weighted sum so
        // that no kv-sized score buffer is needed and a single pass over k and v suffices,
        // while staying as numerically stable as a two-pass max-subtracted softmax.
        let head = |idx: usize, out: &mut [T]| {
            let bi = idx / h;
            let hh = idx % h;
            let qo = q_off + bi * hd + hh * d;
            let kb = k_off + bi * kv_batch_stride + hh * d;
            let vb = v_off + bi * kv_batch_stride + hh * d;
            let qrow = &q[qo..qo + d];
            let mut acc = [0f32; CPU_SDPA_MAX_HEAD_DIM];
            let acc = &mut acc[..d];
            let mut max = f32::NEG_INFINITY;
            let mut denom = 0f32;
            for j in 0..kv {
                let ko = kb + j * hd;
                let krow = &k[ko..ko + d];
                // Independent partial sums, the same shape as `reduce_combine`: a single
                // `s += q * k` accumulator is a serial dependency chain the compiler cannot
                // reorder (float addition is not associative), which pins the dot product at
                // one FMA per latency. Eight lanes -- two 4-wide f32 vectors -- let it both
                // vectorize and keep several FMAs in flight; eight measured distinctly
                // better than four here.
                const LANES: usize = 8;
                let mut lanes = [0f32; LANES];
                // Head dims are multiples of 8 in practice, so these tails are normally empty.
                let (q_chunks, q_rem) = qrow.as_chunks::<LANES>();
                let (k_chunks, k_rem) = krow.as_chunks::<LANES>();
                for (qc, kc) in q_chunks.iter().zip(k_chunks.iter()) {
                    for l in 0..LANES {
                        lanes[l] +=
                            <T as WithDTypeF>::to_f32(qc[l]) * <T as WithDTypeF>::to_f32(kc[l]);
                    }
                }
                let mut s: f32 = lanes.iter().sum();
                for (qq, kk) in q_rem.iter().zip(k_rem.iter()) {
                    s += <T as WithDTypeF>::to_f32(*qq) * <T as WithDTypeF>::to_f32(*kk);
                }
                s *= scale;
                if let Some((m, m_off)) = mask {
                    s += <T as WithDTypeF>::to_f32(m[m_off + j]);
                }
                if s == f32::NEG_INFINITY {
                    // Fully masked: contributes nothing to the sum or the denominator. Skipping
                    // it also keeps the running max finite, which `exp(max - s)` relies on.
                    continue;
                }
                if s > max {
                    // Rebase the running totals onto the new maximum. On the first step
                    // `max` is -inf, so this correctly zeroes them.
                    let c = (max - s).exp();
                    denom *= c;
                    for a in acc.iter_mut() {
                        *a *= c;
                    }
                    max = s;
                }
                let p = (s - max).exp();
                denom += p;
                let vo = vb + j * hd;
                for (a, vv) in acc.iter_mut().zip(v[vo..vo + d].iter()) {
                    *a += p * <T as WithDTypeF>::to_f32(*vv);
                }
            }
            let inv = 1.0 / denom;
            for (o, a) in out.iter_mut().zip(acc.iter()) {
                *o = T::from_f32(*a * inv);
            }
        };
        let dst = &mut dst[..b * hd];
        if use_parallelism(b * h * kv * d) {
            crate::threadpool::par_chunks_mut(dst, d, |i, o| head(i, o));
        } else {
            dst.chunks_mut(d).enumerate().for_each(|(i, o)| head(i, o));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
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

        if USE_COL2IM_CONV1D_TR && can_use_col2im {
            // COL2IM approach: matmul + col2im transformation
            // 1. Transpose input from [B, C_in, L_in] to [B, L_in, C_in]
            // 2. Matmul: [B, L_in, C_in] @ [C_in, C_out * K] -> [B, L_in, C_out * K]
            // 3. Col2Im: [B, L_in, C_out, K] -> [B, C_out, L_out]

            // Step 1: Transpose input to [B, L_in, C_in]
            let mut src_transposed = vec![T::zero(); batch * length * in_channels];
            Self::transpose(&mut src_transposed, src, 1, 2, &[batch, in_channels, length])?;

            // Step 2: Matrix multiplication
            // src_transposed: [B, L_in, C_in]
            // kernel: [C_in, C_out, K] stored row-major, treat as [C_in, C_out * K]
            // result: [B, L_in, C_out * K]
            let n = out_channels * kernel_size;
            let mut col = vec![T::zero(); batch * length * n];

            Self::gemm(
                &mut col,
                (&src_transposed, 0),
                (kernel, 0),
                /* m */ length,
                /* n */ n,
                /* k */ in_channels,
                batch,
                length * in_channels,
                0,
                (1, n),
                (1, in_channels),
                (1, n),
            )?;

            // Step 3: Col2Im transformation
            // col is [B, L_in, C_out * K] = [B, L_in, C_out, K]
            // output is [B, C_out, L_out]
            col2im1d(dst, &col, batch, length, out_channels, kernel_size, stride);

            Ok(())
        } else {
            // Fallback: original implementation for grouped convolutions or with padding
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

/// Reduce along a dimension with a binary combine function.
/// For the innermost dimension (`inner_size == 1`), each output is the reduction of a
/// contiguous row: rows are processed in parallel and the reduction uses several independent
/// accumulator lanes so it is not bound by the combine-op latency chain.
/// For non-innermost dimensions, the reduced slices are combined with contiguous streaming
/// passes rather than strided per-element loops.
fn reduce_combine<T: WithDType + Copy>(
    dst: &mut [T],
    src: &[T],
    dim_size: usize,
    outer_size: usize,
    inner_size: usize,
    init: T,
    combine: impl Fn(T, T) -> T + Sync,
) {
    if dim_size == 0 {
        dst[..outer_size * inner_size].fill(init);
        return;
    }
    let parallel = use_parallelism(outer_size * dim_size * inner_size);
    if inner_size == 1 {
        let reduce_row = |outer: usize, dst: &mut T| {
            let row = &src[outer * dim_size..(outer + 1) * dim_size];
            const LANES: usize = 16;
            let mut acc = [init; LANES];
            let (chunks, rem) = row.as_chunks::<LANES>();
            for chunk in chunks {
                for (a, &v) in acc.iter_mut().zip(chunk) {
                    *a = combine(*a, v);
                }
            }
            let mut res = init;
            for a in acc {
                res = combine(res, a);
            }
            for &v in rem {
                res = combine(res, v);
            }
            *dst = res;
        };
        let dst = &mut dst[..outer_size];
        if parallel {
            dst.par_iter_mut()
                .with_min_len(4)
                .enumerate()
                .for_each(|(outer, dst)| reduce_row(outer, dst));
        } else {
            dst.iter_mut().enumerate().for_each(|(outer, dst)| reduce_row(outer, dst));
        }
    } else {
        let reduce_block = |outer: usize, dst: &mut [T]| {
            let base = outer * dim_size * inner_size;
            dst.copy_from_slice(&src[base..base + inner_size]);
            for d in 1..dim_size {
                let s = &src[base + d * inner_size..base + (d + 1) * inner_size];
                for (dv, &sv) in dst.iter_mut().zip(s) {
                    *dv = combine(*dv, sv);
                }
            }
        };
        let dst = &mut dst[..outer_size * inner_size];
        if parallel {
            dst.par_chunks_mut(inner_size)
                .enumerate()
                .for_each(|(outer, dst)| reduce_block(outer, dst));
        } else {
            dst.chunks_mut(inner_size)
                .enumerate()
                .for_each(|(outer, dst)| reduce_block(outer, dst));
        }
    }
}

/// Arg-reduction (argmin/argmax) along a dimension. `better(v, best)` returns true when `v`
/// should replace the current best value.
fn reduce_arg<T: WithDType + Copy>(
    dst: &mut [i64],
    src: &[T],
    dim_size: usize,
    outer_size: usize,
    inner_size: usize,
    init: T,
    better: impl Fn(T, T) -> bool + Sync,
) {
    if dim_size == 0 {
        dst[..outer_size * inner_size].fill(0);
        return;
    }
    let parallel = use_parallelism(outer_size * dim_size * inner_size);
    if inner_size == 1 {
        let arg_row = |outer: usize, dst: &mut i64| {
            let row = &src[outer * dim_size..(outer + 1) * dim_size];
            let mut best = init;
            let mut best_idx = 0usize;
            for (d, &v) in row.iter().enumerate() {
                if better(v, best) {
                    best = v;
                    best_idx = d;
                }
            }
            *dst = best_idx as i64;
        };
        let dst = &mut dst[..outer_size];
        if parallel {
            dst.par_iter_mut()
                .with_min_len(4)
                .enumerate()
                .for_each(|(outer, dst)| arg_row(outer, dst));
        } else {
            dst.iter_mut().enumerate().for_each(|(outer, dst)| arg_row(outer, dst));
        }
    } else {
        let arg_block = |outer: usize, dst: &mut [i64]| {
            let base = outer * dim_size * inner_size;
            let mut best = src[base..base + inner_size].to_vec();
            dst.fill(0);
            for d in 1..dim_size {
                let s = &src[base + d * inner_size..base + (d + 1) * inner_size];
                for ((dv, bv), &sv) in dst.iter_mut().zip(best.iter_mut()).zip(s) {
                    if better(sv, *bv) {
                        *bv = sv;
                        *dv = d as i64;
                    }
                }
            }
        };
        let dst = &mut dst[..outer_size * inner_size];
        if parallel {
            dst.par_chunks_mut(inner_size)
                .enumerate()
                .for_each(|(outer, dst)| arg_block(outer, dst));
        } else {
            dst.chunks_mut(inner_size).enumerate().for_each(|(outer, dst)| arg_block(outer, dst));
        }
    }
}

/// Largest head dimension the fused attention kernel keeps its accumulator on the stack for.
/// Well above any head dim in practice; callers fall back to the composed path beyond it.
const CPU_SDPA_MAX_HEAD_DIM: usize = 256;

/// Below this many elements, elementwise ops run serially: the rayon fork/join overhead
/// (~10us) dwarfs the work itself.
const ELEMWISE_PAR_THRESHOLD: usize = 32 * 1024;
/// Chunk size for parallel elementwise ops; large enough that the per-chunk dispatch cost is
/// negligible and the inner loop auto-vectorizes.
const ELEMWISE_CHUNK: usize = 64 * 1024;
/// Floor on a pool-derived chunk (see `unary_chunk`): below this the per-chunk dispatch cost
/// stops being negligible and the inner loop has too few iterations to auto-vectorize well.
const ELEMWISE_MIN_CHUNK: usize = 4 * 1024;
/// Chunks per thread for a pool-derived chunk. Splitting exactly `current_num_threads()` ways
/// leaves rayon nothing to steal, and the cores here are not interchangeable: an E core takes
/// several times longer over its chunk than a P core, and with one chunk each the whole op
/// waits for it. Oversubscribing keeps the fast cores fed. This matters most with a small pool,
/// where the chunks are large -- on a 4-thread pool, 1M `silu_` goes 487us -> 421us and 128K
/// `silu_` 96us -> 80us just from this factor.
const ELEMWISE_CHUNKS_PER_THREAD: usize = 4;

/// Chunk size for a unary op over `len` elements.
///
/// `ELEMWISE_CHUNK` is a fixed 64K, so a length just past the parallelism threshold yields a
/// single chunk -- rayon is entered but everything runs on one worker -- and the whole pool
/// only gets work at `threads * ELEMWISE_CHUNK` (640K on a 10-thread pool). For a
/// transcendental that is a lot of idle threads: one element costs tens of cycles, so a serial
/// pass over 48K of them already costs ~50us against a ~20us fork/join, and splitting to match
/// the pool is worth 1.6-1.9x from 32K to 128K.
///
/// The cheap ops keep the fixed chunk on purpose. An `Exp` element is ~20x a `Sqr` element, so
/// for those a serial pass over the same 48K costs only a few us -- an order of magnitude below
/// the fork/join -- and a real split is a 4-10x *loss*. Sizing their chunks by the pool is the
/// one thing that must not happen here.
fn unary_chunk(op: UnaryOp, len: usize) -> usize {
    let transcendental = matches!(
        op,
        UnaryOp::Cos
            | UnaryOp::Sin
            | UnaryOp::Exp
            | UnaryOp::Log
            | UnaryOp::GeluErf
            | UnaryOp::Elu { .. }
            | UnaryOp::Silu
            | UnaryOp::Tanh
            | UnaryOp::Sigmoid
    );
    if !transcendental {
        return ELEMWISE_CHUNK;
    }
    let threads = rayon::current_num_threads().max(1);
    len.div_ceil(threads * ELEMWISE_CHUNKS_PER_THREAD).max(ELEMWISE_MIN_CHUNK)
}

/// Apply a binary operation in-place: dst[i] = op(dst[i], src[i])
#[inline(always)]
fn apply_bin_assign<T: Copy + Send + Sync, F>(dst: &mut [T], src: &[T], f: F)
where
    F: Fn(&mut T, T) + Sync,
{
    if !use_parallelism(dst.len()) {
        for (d, s) in dst.iter_mut().zip(src) {
            f(d, *s);
        }
    } else {
        crate::threadpool::par_chunks_zip(
            dst,
            ELEMWISE_CHUNK,
            src,
            ELEMWISE_CHUNK,
            |_, dst, src| {
                for (d, s) in dst.iter_mut().zip(src) {
                    f(d, *s);
                }
            },
        );
    }
}

/// Apply a unary operation in-place: dst[i] = op(dst[i])
#[inline(always)]
fn apply_inplace_unary<T: Copy + Send + Sync, F>(chunk: usize, dst: &mut [T], f: F)
where
    F: Fn(&mut T) + Sync,
{
    if !use_parallelism(dst.len()) {
        for d in dst.iter_mut() {
            f(d);
        }
    } else {
        crate::threadpool::par_chunks_mut(dst, chunk, |_, dst| {
            for d in dst.iter_mut() {
                f(d);
            }
        });
    }
}

/// Apply a unary operation: dst[i] = op(src[i])
#[inline(always)]
fn apply_unary<T: Copy + Send + Sync, F>(chunk: usize, dst: &mut [T], src: &[T], f: F)
where
    F: Fn(T) -> T + Sync,
{
    if !use_parallelism(dst.len()) {
        for (d, s) in dst.iter_mut().zip(src) {
            *d = f(*s);
        }
    } else {
        crate::threadpool::par_chunks_zip(dst, chunk, src, chunk, |_, dst, src| {
            for (d, s) in dst.iter_mut().zip(src) {
                *d = f(*s);
            }
        });
    }
}

/// Apply a binary operation: dst[i] = op(lhs[i], rhs[i])
#[inline(always)]
fn apply_binary<T: Copy + Send + Sync, F>(dst: &mut [T], lhs: &[T], rhs: &[T], f: F)
where
    F: Fn(T, T) -> T + Sync,
{
    if !use_parallelism(dst.len()) {
        for ((d, l), r) in dst.iter_mut().zip(lhs).zip(rhs) {
            *d = f(*l, *r);
        }
    } else {
        crate::threadpool::par_chunks_zip(
            dst,
            ELEMWISE_CHUNK,
            lhs,
            ELEMWISE_CHUNK,
            |c, dst, lhs| {
                // `rhs` is chunked the same way, so the piece is fixed by the chunk index.
                let start = c * ELEMWISE_CHUNK;
                let rhs = &rhs[start..(start + dst.len()).min(rhs.len())];
                for ((d, l), r) in dst.iter_mut().zip(lhs).zip(rhs) {
                    *d = f(*l, *r);
                }
            },
        );
    }
}

/// Im2Col transformation for 1D convolution.
/// Transforms input from [B, C, L] to [B, L_out, C * K] for matrix multiplication.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all)]
fn im2col1d<T: WithDTypeF>(
    src: &[T],
    batch: usize,
    in_channels: usize,
    length: usize,
    l_out: usize,
    l_k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Vec<T> {
    let k = in_channels * l_k;
    let mut dst = vec![T::zero(); batch * l_out * k];

    let fill_row = |bl: usize, dst_row: &mut [T]| {
        let b_idx = bl / l_out;
        let l_idx = bl % l_out;
        let src_b_offset = b_idx * in_channels * length;

        for c_idx in 0..in_channels {
            let src_c_offset = src_b_offset + c_idx * length;
            let dst_c_offset = c_idx * l_k;

            for (l_k_idx, dst) in dst_row[dst_c_offset..dst_c_offset + l_k].iter_mut().enumerate() {
                let src_l = l_idx * stride + l_k_idx * dilation;

                // Handle padding
                if src_l < padding || src_l >= length + padding {
                    // Zero padding - already initialized to zero
                    continue;
                }
                let src_l = src_l - padding;
                let src_idx = src_c_offset + src_l;
                *dst = src[src_idx];
            }
        }
    };
    if use_parallelism(batch * l_out * k) {
        crate::threadpool::par_chunks_mut(&mut dst, k, |bl, dst_row| fill_row(bl, dst_row));
    } else {
        dst.chunks_mut(k).enumerate().for_each(|(bl, dst_row)| fill_row(bl, dst_row));
    }

    dst
}

/// Direct conv1d implementation (fallback for grouped convolutions).
#[allow(clippy::too_many_arguments)]
fn conv1d_direct<T: WithDTypeF>(
    dst: &mut [T],
    src: &[T],
    kernel: &[T],
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
) -> crate::Result<()> {
    let in_c_per_group = in_channels / groups;

    // Initialize output to zero
    dst.iter_mut().for_each(|v| *v = T::zero());

    // Reorder input from [B, C, L] to [B, L, C] for better memory access in the inner loop
    let mut src_reordered = vec![T::zero(); batch * length * in_channels];
    for b in 0..batch {
        for l in 0..length {
            for c in 0..in_channels {
                let src_idx = b * in_channels * length + c * length + l;
                let dst_idx = b * length * in_channels + l * in_channels + c;
                src_reordered[dst_idx] = src[src_idx];
            }
        }
    }

    // Process each kernel offset
    for k_offset in 0..kernel_size {
        // Parallelize over output channels
        (0..out_channels).into_par_iter().for_each(|out_c| {
            let g = out_c / (out_channels / groups);
            let in_c_start = g * in_c_per_group;

            // Gather kernel weights for this output channel and kernel offset
            // kernel layout: [out_channels, in_c_per_group, kernel_size]
            let k_cont: Vec<T> = (0..in_c_per_group)
                .map(|ic| {
                    let k_idx = out_c * in_c_per_group * kernel_size + ic * kernel_size + k_offset;
                    kernel[k_idx]
                })
                .collect();

            for b in 0..batch {
                let dst_base = b * out_channels * out_length + out_c * out_length;

                for ol in 0..out_length {
                    let src_l = ol * stride + k_offset * dilation;

                    // Check padding bounds
                    if src_l < padding || src_l >= padding + length {
                        continue;
                    }
                    let src_l = src_l - padding;

                    // Compute dot product over input channels
                    let src_base = b * length * in_channels + src_l * in_channels + in_c_start;
                    let mut d = T::zero();
                    for ic in 0..in_c_per_group {
                        d += src_reordered[src_base + ic] * k_cont[ic];
                    }

                    // Accumulate into output
                    // Safety: each out_c is processed by a different thread, so no races
                    let dst_idx = dst_base + ol;
                    unsafe {
                        let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                        *ptr += d;
                    }
                }
            }
        });
    }
    Ok(())
}

/// Col2Im transformation for 1D transposed convolution.
/// Transforms col from [B, L_in, C_out, K] to output [B, C_out, L_out].
/// Following the reference implementation closely.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all)]
fn col2im1d<T: WithDTypeF>(
    dst: &mut [T],
    col: &[T],
    _b_size: usize,
    l_in: usize,
    c_out: usize,
    k_size: usize,
    stride: usize,
) {
    let l_out = (l_in - 1) * stride + k_size;

    // Strides for source [B, L_in, C_out, K] stored as [B, L_in, C_out * K]
    let (src_s0, src_s1, src_s2) = (c_out * k_size * l_in, c_out * k_size, k_size);

    // Each (b, c) pair owns a contiguous l_out-sized slice of dst, so the accumulation can
    // run in parallel over those slices with contiguous reads from col.
    let accumulate = |bc: usize, dst: &mut [T]| {
        let b_i = bc / c_out;
        let c_i = bc % c_out;
        dst.fill(T::zero());
        for l_in_i in 0..l_in {
            let src_base = b_i * src_s0 + l_in_i * src_s1 + c_i * src_s2;
            let dst_base = l_in_i * stride;
            for k_i in 0..k_size {
                dst[dst_base + k_i] += col[src_base + k_i];
            }
        }
    };
    if use_parallelism(dst.len().max(col.len())) {
        crate::threadpool::par_chunks_mut(dst, l_out, |bc, dst| accumulate(bc, dst));
    } else {
        dst.chunks_mut(l_out).enumerate().for_each(|(bc, dst)| accumulate(bc, dst));
    }
}

/// Direct conv_transpose1d implementation (fallback for grouped convolutions or with padding).
#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_direct<T: WithDTypeF>(
    dst: &mut [T],
    src: &[T],
    kernel: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    _output_padding: usize,
    groups: usize,
) -> crate::Result<()> {
    let in_c_per_group = in_channels / groups;
    let out_c_per_group = out_channels / groups;

    dst.fill(T::zero());

    // Reorder input from [B, C, L] to [B, L, C] for contiguous memory access
    let mut src_reordered = vec![T::zero(); batch * length * in_channels];
    for b in 0..batch {
        for l in 0..length {
            for c in 0..in_channels {
                let src_idx = b * in_channels * length + c * length + l;
                let dst_idx = b * length * in_channels + l * in_channels + c;
                src_reordered[dst_idx] = src[src_idx];
            }
        }
    }

    // Each (b, out_c) pair owns a contiguous `out_length` slice of dst, so the op splits over
    // those slices instead of over channels once per kernel offset. That means one fork/join
    // for the whole call rather than `kernel_size` of them, and safe mutable access instead of
    // raw pointers. Keeping k_offset outermost within a slice preserves the original
    // accumulation order into dst, so results are unchanged.
    let src_reordered = src_reordered.as_slice();
    let accumulate = |bc: usize, dst_row: &mut [T]| {
        let b = bc / out_channels;
        let out_c = bc % out_channels;
        let g = out_c / out_c_per_group;
        let oc_in_group = out_c % out_c_per_group;
        let in_c_start = g * in_c_per_group;
        let src_row = &src_reordered[b * length * in_channels..(b + 1) * length * in_channels];
        // Kernel layout: [in_channels, out_channels/groups, kernel_size]
        let k_base = in_c_start * out_c_per_group * kernel_size + oc_in_group * kernel_size;

        if in_c_per_group == 1 {
            // Depthwise: the dot product over input channels collapses to a single multiply,
            // and this channel's kernel taps are contiguous.
            let w = &kernel[k_base..k_base + kernel_size];
            for (k_offset, &wk) in w.iter().enumerate() {
                for il in 0..length {
                    let out_pos_raw = il * stride + k_offset;
                    if out_pos_raw < padding || out_pos_raw >= out_length + padding {
                        continue;
                    }
                    dst_row[out_pos_raw - padding] += src_row[il * in_channels + in_c_start] * wk;
                }
            }
        } else {
            // This channel's taps for a given k_offset sit `out_c_per_group * kernel_size`
            // apart in `kernel`, so gather them once per offset rather than reading them
            // strided inside the `il` loop: the dot product below then walks two contiguous
            // buffers and vectorizes. The gather costs one `in_c_per_group` copy per
            // (slice, offset), which is `length` times less work than the loop it feeds.
            let mut k_cont = vec![T::zero(); in_c_per_group];
            for k_offset in 0..kernel_size {
                for (ic, kc) in k_cont.iter_mut().enumerate() {
                    *kc = kernel[k_base + ic * out_c_per_group * kernel_size + k_offset];
                }
                for il in 0..length {
                    let out_pos_raw = il * stride + k_offset;
                    if out_pos_raw < padding || out_pos_raw >= out_length + padding {
                        continue;
                    }
                    let src_base = il * in_channels + in_c_start;
                    let mut acc = T::zero();
                    for (ic, &kc) in k_cont.iter().enumerate() {
                        acc += src_row[src_base + ic] * kc;
                    }
                    dst_row[out_pos_raw - padding] += acc;
                }
            }
        }
    };

    let dst = &mut dst[..batch * out_channels * out_length];
    if use_parallelism(batch * out_channels * kernel_size * length) {
        crate::threadpool::par_chunks_mut(dst, out_length, |bc, d| accumulate(bc, d));
    } else {
        dst.chunks_mut(out_length).enumerate().for_each(|(bc, d)| accumulate(bc, d));
    }
    Ok(())
}

/// Helper function for broadcast binary operations.
#[inline(always)]
fn broadcast_binary_op<T: WithDType>(
    dst: &mut [T],
    lhs: &[T],
    rhs: &[T],
    dst_shape: &[usize],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
    op: impl Fn(T, T) -> T + Sync,
) -> Result<()> {
    let lhs_no_zero = lhs_strides.iter().all(|&s| s > 0);
    let rhs_no_zero = rhs_strides.iter().all(|&s| s > 0);

    if lhs_no_zero && rhs_no_zero {
        apply_binary(dst, lhs, rhs, &op);
        return Ok(());
    }
    if lhs_no_zero && rhs_strides == [0, 1] {
        for idx0 in 0..dst_shape[0] {
            for (idx1, rhs) in rhs.iter().enumerate().take(dst_shape[1]) {
                let dst_idx = idx0 * dst_shape[1] + idx1;
                let lhs_idx = idx0 * lhs_strides[0] + idx1;
                dst[dst_idx] = op(lhs[lhs_idx], *rhs);
            }
        }
        return Ok(());
    }
    if lhs_no_zero && rhs_strides == [1, 0] {
        for (idx0, rhs) in rhs.iter().enumerate().take(dst_shape[0]) {
            for idx1 in 0..dst_shape[1] {
                let dst_idx = idx0 * dst_shape[1] + idx1;
                let lhs_idx = idx0 * lhs_strides[0] + idx1;
                dst[dst_idx] = op(lhs[lhs_idx], *rhs);
            }
        }
        return Ok(());
    }
    if rhs_no_zero && lhs_strides == [0, 1] {
        for idx0 in 0..dst_shape[0] {
            for (idx1, lhs) in lhs.iter().enumerate().take(dst_shape[1]) {
                let dst_idx = idx0 * dst_shape[1] + idx1;
                let rhs_idx = idx0 * rhs_strides[0] + idx1;
                dst[dst_idx] = op(*lhs, rhs[rhs_idx]);
            }
        }
        return Ok(());
    }
    if rhs_no_zero && lhs_strides == [1, 0] {
        for (idx0, lhs) in lhs.iter().enumerate().take(dst_shape[0]) {
            for idx1 in 0..dst_shape[1] {
                let dst_idx = idx0 * dst_shape[1] + idx1;
                let rhs_idx = idx0 * rhs_strides[0] + idx1;
                dst[dst_idx] = op(*lhs, rhs[rhs_idx]);
            }
        }
        return Ok(());
    }

    let total_elems: usize = dst_shape.iter().product();
    let rank = dst_shape.len();

    for (dst_idx, dst) in dst.iter_mut().enumerate().take(total_elems) {
        // Convert linear index to multi-dimensional indices
        let mut remaining = dst_idx;
        let mut lhs_idx = 0usize;
        let mut rhs_idx = 0usize;

        for d in 0..rank {
            let stride: usize = dst_shape[d + 1..].iter().product::<usize>().max(1);
            let coord = remaining / stride;
            remaining %= stride;

            lhs_idx += coord * lhs_strides[d];
            rhs_idx += coord * rhs_strides[d];
        }

        *dst = op(lhs[lhs_idx], rhs[rhs_idx]);
    }

    Ok(())
}
