use crate::error::Context;
use crate::{BinaryOp, DType, Result, UnaryOp, WithDType, WithDTypeF};
use half::{bf16, f16};
use std::any::Any;

const USE_IM2COL_CONV1D: bool = true;
const USE_COL2IM_CONV1D_TR: bool = true;

/// Whether to use a parallel implementation for this many elements of work. The work must be
/// large enough to amortize the dispatch, and the pool must actually have more than one
/// participant: with a single-thread pool (e.g. RAYON_NUM_THREADS=1) every parallel call would
/// still pay a handoff for no benefit.
fn use_parallelism(work: usize) -> bool {
    work >= ELEMWISE_PAR_THRESHOLD && crate::threadpool::threads_for(work) > 1
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
        crate::threadpool::par_chunks_mut(dst, d1, |i0, dst| copy_row(i0, dst));
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
        crate::threadpool::par_chunks_mut(dst, d1 * d2, |i0, dst| copy_block(i0, dst));
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

    let layout = arch::Gemm {
        m,
        n,
        k,
        batch: lhs_b,
        lhs_b_stride,
        rhs_b_stride,
        dst: (dst_cs, dst_rs),
        lhs: (lhs_cs, lhs_rs),
        rhs: (rhs_cs, rhs_rs),
    };
    if arch::gemm(dst, lhs, rhs, &layout) {
        return Ok(());
    }

    // Column stripes, run on xn's own pool rather than letting `gemm` reach for rayon.
    //
    // Two pools would otherwise both be live inside a decode step -- this one for the f32
    // matmuls and xn's for everything else -- and their idle workers spin against each other
    // for cores. Splitting by output column is what `Parallelism::Rayon` does internally
    // anyway; the cost is that each stripe re-packs the lhs panel, which is negligible at the
    // batch sizes decode uses.
    let nth = crate::threadpool::size();
    let worth = nth > 1 && m * n * k >= 1 << 14;
    // One stripe per participant. Measured against 0.25x, 0.5x, 2x and 4x that count: 1x and
    // 2x tie, everything else is worse. Splitting the rows instead when the output is too
    // narrow to stripe was also tried and nets nothing for outputs with few rows, so those
    // stay serial.
    //
    // A tall output, with more rows than columns and at least a microkernel tile of rows per
    // participant, splits its rows: a column stripe there would be a sliver of a tile wide and
    // still read the whole lhs. A convolution's `[out_channels, L_out]` over a short window is
    // the usual case.
    let by_rows = worth && m > n && m >= nth * 16;
    let stripes = if by_rows || (worth && n >= nth * 4) { nth } else { 1 };
    // A parameter the caller declared (see `parameter`) is declared per thread, so each stripe
    // declares it again wherever it runs.
    let parameter = gemm::packed_cache::constant();

    for b_idx in 0..lhs_b {
        let dst = &mut dst[b_idx * m * n..(b_idx + 1) * m * n];
        let lhs = &lhs[b_idx * lhs_b_stride..];
        let rhs = &rhs[b_idx * rhs_b_stride..];
        if stripes > 1 {
            let dst_p = dst.as_mut_ptr() as usize;
            let lhs_p = lhs.as_ptr() as usize;
            let rhs_p = rhs.as_ptr() as usize;
            crate::threadpool::par_units(stripes, |s| {
                let len = if by_rows { m } else { n };
                let per = len.div_ceil(stripes);
                let i0 = (s * per).min(len);
                let i1 = (i0 + per).min(len);
                if i0 == i1 {
                    return;
                }
                let (rows, cols, dst_o, lhs_o, rhs_o) = if by_rows {
                    (i1 - i0, n, i0 * dst_rs, i0 * lhs_rs, 0)
                } else {
                    (m, i1 - i0, i0 * dst_cs, 0, i0 * rhs_cs)
                };
                // SAFETY: stripes own disjoint output rows or columns, and every offset stays
                // inside the slices borrowed above, which outlive the dispatch.
                gemm::packed_cache::with_constant(parameter, || unsafe {
                    gemm::gemm(
                        rows,
                        cols,
                        k,
                        (dst_p as *mut T).add(dst_o),
                        dst_cs as isize,
                        dst_rs as isize,
                        false,
                        (lhs_p as *const T).add(lhs_o),
                        lhs_cs as isize,
                        lhs_rs as isize,
                        (rhs_p as *const T).add(rhs_o),
                        rhs_cs as isize,
                        rhs_rs as isize,
                        T::zero(),
                        T::one(),
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    )
                })
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

/// The first `len` elements of `storage`, as a span to declare to `gemm`'s packed operand cache
/// with `gemm::packed_cache::with_constant`. A gemm made inside the declaration whose packed
/// operand is this span keeps the packed copy and reuses it on later calls, rather than
/// packing it on every call; any other operand is packed per call as before. Empty for a
/// storage that is not on the CPU.
///
/// Only a model parameter may be declared: its contents must never change afterwards, since a
/// later call reads the copy. A key/value cache or streaming state written in place must not.
pub(crate) fn parameter<T: WithDType, S: Any>(
    storage: &S,
    len: usize,
) -> gemm::packed_cache::Constant {
    match (storage as &dyn Any).downcast_ref::<Vec<T>>() {
        Some(v) => gemm::packed_cache::Constant::new(v.as_ptr(), len.min(v.len())),
        None => gemm::packed_cache::Constant::default(),
    }
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
        if arch_unary_inplace(&mut dst[..len], op) {
            return Ok(());
        }
        match op {
            UnaryOp::Cos => apply_inplace_unary(&mut dst[..len], |v| *v = v.cos()),
            UnaryOp::Sin => apply_inplace_unary(&mut dst[..len], |v| *v = v.sin()),
            UnaryOp::Exp => apply_inplace_unary(&mut dst[..len], |v| *v = v.exp()),
            UnaryOp::Log => apply_inplace_unary(&mut dst[..len], |v| *v = v.ln()),
            UnaryOp::Neg => apply_inplace_unary(&mut dst[..len], |v| *v = T::zero() - *v),
            UnaryOp::Sqr => apply_inplace_unary(&mut dst[..len], |v| *v = *v * *v),
            UnaryOp::Sqrt => apply_inplace_unary(&mut dst[..len], |v| *v = v.sqrt()),
            UnaryOp::Rsqrt => apply_inplace_unary(&mut dst[..len], |v| *v = T::one() / v.sqrt()),
            UnaryOp::Abs => apply_inplace_unary(&mut dst[..len], |v| *v = v.abs()),
            UnaryOp::GeluErf => {
                let sqrt_2_inv = std::f32::consts::FRAC_1_SQRT_2;
                apply_inplace_unary(&mut dst[..len], |v| {
                    let x = v.to_f32();
                    let erf_val = libm::erff(x * sqrt_2_inv);
                    *v = T::from_f32(x * 0.5 * (1.0 + erf_val));
                })
            }
            UnaryOp::Elu { alpha } => apply_inplace_unary(&mut dst[..len], |v| {
                let x = v.to_f32();
                *v = T::from_f32(if x > 0.0 { x } else { alpha * (x.exp() - 1.0) });
            }),
            UnaryOp::Relu => apply_inplace_unary(&mut dst[..len], |v| {
                if *v < T::zero() {
                    *v = T::zero()
                }
            }),
            UnaryOp::Silu => apply_inplace_unary(&mut dst[..len], |v| {
                *v = *v / (T::one() + (T::zero() - *v).exp())
            }),
            UnaryOp::Tanh => apply_inplace_unary(&mut dst[..len], |v| *v = v.tanh()),
            UnaryOp::Sigmoid => apply_inplace_unary(&mut dst[..len], |v| {
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
        if arch_unary(&mut dst[..len], &src[..len], op) {
            return Ok(());
        }
        match op {
            UnaryOp::Cos => apply_unary(&mut dst[..len], &src[..len], |s| s.cos()),
            UnaryOp::Sin => apply_unary(&mut dst[..len], &src[..len], |s| s.sin()),
            UnaryOp::Exp => apply_unary(&mut dst[..len], &src[..len], |s| s.exp()),
            UnaryOp::Log => apply_unary(&mut dst[..len], &src[..len], |s| s.ln()),
            UnaryOp::Neg => apply_unary(&mut dst[..len], &src[..len], |s| T::zero() - s),
            UnaryOp::Sqr => apply_unary(&mut dst[..len], &src[..len], |s| s * s),
            UnaryOp::Sqrt => apply_unary(&mut dst[..len], &src[..len], |s| s.sqrt()),
            UnaryOp::Rsqrt => apply_unary(&mut dst[..len], &src[..len], |s| T::one() / s.sqrt()),
            UnaryOp::Abs => apply_unary(&mut dst[..len], &src[..len], |s| s.abs()),
            UnaryOp::GeluErf => {
                let sqrt_2_inv = std::f32::consts::FRAC_1_SQRT_2;
                apply_unary(&mut dst[..len], &src[..len], |s| {
                    let x = s.to_f32();
                    let erf_val = libm::erff(x * sqrt_2_inv);
                    T::from_f32(x * 0.5 * (1.0 + erf_val))
                })
            }
            UnaryOp::Elu { alpha } => apply_unary(&mut dst[..len], &src[..len], |s| {
                let x = s.to_f32();
                T::from_f32(if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            }),
            UnaryOp::Relu => {
                let zero = T::zero();
                apply_unary(&mut dst[..len], &src[..len], |s| if s < zero { zero } else { s })
            }
            UnaryOp::Silu => apply_unary(&mut dst[..len], &src[..len], |s| {
                s / (T::one() + (T::zero() - s).exp())
            }),
            UnaryOp::Tanh => apply_unary(&mut dst[..len], &src[..len], |s| s.tanh()),
            UnaryOp::Sigmoid => apply_unary(&mut dst[..len], &src[..len], |s| {
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
        if add == zero && scale == one {
            Self::copy(dst, src, len)
        } else if add == zero {
            apply_unary(&mut dst[..len], &src[..len], |s| s * scale);
            Ok(())
        } else if scale == one {
            apply_unary(&mut dst[..len], &src[..len], |s| s + add);
            Ok(())
        } else {
            apply_unary(&mut dst[..len], &src[..len], |s| s * scale + add);
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
                    crate::threadpool::par_chunks_mut(dst, d2 * d_j * d1, |i, dst| {
                        transpose_block(i, dst)
                    });
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
                    crate::threadpool::par_chunks_mut(dst, d2 * d_j * d1 * d_k, |i, dst| {
                        transpose_block(i, dst)
                    });
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
            crate::threadpool::par_chunks_zip(
                &mut dst[..l],
                ELEMWISE_CHUNK,
                &src[..l],
                ELEMWISE_CHUNK,
                |_, d, s| d.copy_from_slice(s),
            );
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
            crate::threadpool::par_chunks_mut(&mut dst[..len], ELEMWISE_CHUNK, |_, d| fill(d));
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
            crate::threadpool::par_chunks_mut(&mut dst[..len], ELEMWISE_CHUNK, |_, d| fill(d));
        } else {
            fill(&mut dst[..len]);
        }
        Ok(())
    }

    fn fill<T: WithDType>(dst: &mut Self::Storage<T>, v: T, l: usize) -> Result<()> {
        if !use_parallelism(l) {
            dst[..l].fill(v);
        } else {
            crate::threadpool::par_chunks_mut(&mut dst[..l], ELEMWISE_CHUNK, |_, d| d.fill(v));
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
        // Same gating as `rope_i`.
        if use_parallelism(b * h * t * d) {
            crate::threadpool::par_chunks_zip(dst, t * d, src, t * d, |bh_i, d, s| row(bh_i, s, d));
        } else {
            for (bh_i, (s, d)) in src.chunks(t * d).zip(dst.chunks_mut(t * d)).enumerate() {
                row(bh_i, s, d);
            }
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
        let row = |bh_i: usize, src: &[T], dst: &mut [T]| {
            let base = if unbatched_rope { (bh_i / h) * t * d / 2 } else { 0 };
            for i_over_2 in 0..t * d / 2 {
                let i = 2 * i_over_2;
                let rope_i = base + i_over_2;
                dst[i] = src[i] * cos[rope_i] - src[i + 1] * sin[rope_i];
                dst[i + 1] = src[i] * sin[rope_i] + src[i + 1] * cos[rope_i];
            }
        };
        // One (b, h) row per chunk. At single-token decode a row is `d` elements and there are
        // only b*h of them, so a fork/join costs more than the whole rotation; past the
        // elementwise threshold it goes to the same resident pool as every other operator.
        if use_parallelism(b * h * t * d) {
            crate::threadpool::par_chunks_zip(dst, t * d, src, t * d, |bh_i, d, s| row(bh_i, s, d));
        } else {
            for (bh_i, (s, d)) in src.chunks(t * d).zip(dst.chunks_mut(t * d)).enumerate() {
                row(bh_i, s, d);
            }
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
        // Accelerate comes before the target kernels of `gemm_`, even for one row: it runs on
        // the matrix unit, which beats the direct `x @ W^T` kernel at every shape but the
        // tiniest. Only the layouts it cannot take reach them.
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
                // The fallbacks hand `gemm_` the offsets; only this path applies them here.
                let (lhs, rhs) = (&lhs[lhs_o..], &rhs[rhs_o..]);
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
        // The width that will actually run it; `get_num_threads` is 1 off the pool.
        } else if n_outer >= crate::threadpool::size() {
            crate::threadpool::par_chunks_mut(&mut dst[..total], d_a * d_b, |o, dst| {
                tiled_2d_gather(dst, outer_offset(o))
            });
        } else {
            // Few outer blocks (e.g. a plain 2d transpose): parallelize over row tiles
            // within each block instead.
            for o in 0..n_outer {
                let off = outer_offset(o);
                crate::threadpool::par_chunks_mut(
                    &mut dst[o * d_a * d_b..(o + 1) * d_a * d_b],
                    TILE * d_b,
                    |a_t, dst| tiled_2d_gather(dst, off + a_t * TILE * s_a),
                );
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
            if arch::softmax_row(dst, src) {
                return;
            }
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
        if in_channels == 0 {
            // Nothing to sum, so every output is zero. The gemm paths below would be handed an
            // empty reduction, which BLAS rejects.
            dst.fill(T::zero());
            return Ok(());
        }
        let shape = arch::Conv1d {
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
        };
        if arch::conv1d(dst, src, kernel, &shape) {
            return Ok(());
        }
        if USE_IM2COL_CONV1D && groups == 1 {
            // im2col, laid out as [B, C * K, L_out]: row (c, k) is the input channel `c`
            // shifted by tap `k`, which for a stride-1 convolution is one contiguous copy.
            // The kernel [out, in, K] read as [out, C * K] then multiplies it straight into
            // the output layout, [B, out, L_out], with no transpose on either side.
            let k = in_channels * kernel_size;
            // A pointwise convolution's columns are the input itself.
            let pointwise = kernel_size == 1 && stride == 1 && padding == 0;
            let col_buf;
            let col: &Self::Storage<T> = if pointwise {
                src
            } else {
                col_buf = im2col1d(
                    src,
                    batch,
                    in_channels,
                    length,
                    out_length,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                )?;
                &col_buf
            };
            Self::gemm(
                dst,
                (kernel, 0),
                (col, 0),
                /* m */ out_channels,
                /* n */ out_length,
                /* k */ k,
                batch,
                0,
                k * out_length,
                (1, out_length),
                (1, k),
                (1, out_length),
            )
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
        mask: Option<(&Self::Storage<T>, usize, usize)>,
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
            // This entry's row of the mask; a batch stride of 0 makes every entry read row 0.
            let mask = mask.map(|(m, m_off, m_bs)| (m, m_off + bi * m_bs));
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
        if in_channels == 0 {
            // Nothing to sum, so every output is zero. The gemm paths below would be handed an
            // empty reduction, which BLAS rejects.
            dst.fill(T::zero());
            return Ok(());
        }
        if USE_COL2IM_CONV1D_TR && groups == 1 {
            // [B, L_in, C_in] @ [C_in, C_out * K] -> [B, L_in, C_out * K], then col2im
            // overlap-adds the taps into [B, C_out, L_out]. The input is [B, C_in, L_in]; the
            // gemm reads it transposed through its strides rather than through a copy.
            let n = out_channels * kernel_size;
            // SAFETY: the gemm writes every element; its reduction is not empty, since
            // `in_channels > 0` here.
            let mut col = unsafe { Self::alloc_uninit(batch * length * n, &crate::CPU)? };
            Self::gemm(
                &mut col,
                (src, 0),
                (kernel, 0),
                /* m */ length,
                /* n */ n,
                /* k */ in_channels,
                batch,
                length * in_channels,
                0,
                (1, n),
                (length, 1),
                (1, n),
            )?;
            col2im1d(dst, &col, length, out_channels, kernel_size, stride, padding, out_length);
            Ok(())
        } else {
            // Fallback for grouped convolutions.
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
            crate::threadpool::par_chunks_mut_by(outer_size * dim_size, dst, 4, |c, piece| {
                for (j, d) in piece.iter_mut().enumerate() {
                    reduce_row(c * 4 + j, d);
                }
            });
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
            crate::threadpool::par_chunks_mut_by(
                outer_size * dim_size * inner_size,
                dst,
                inner_size,
                |outer, dst| reduce_block(outer, dst),
            );
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
            crate::threadpool::par_chunks_mut_by(outer_size * dim_size, dst, 4, |c, piece| {
                for (j, d) in piece.iter_mut().enumerate() {
                    arg_row(c * 4 + j, d);
                }
            });
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
            crate::threadpool::par_chunks_mut_by(
                outer_size * dim_size * inner_size,
                dst,
                inner_size,
                |outer, dst| arg_block(outer, dst),
            );
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
/// Smallest chunk [`elemwise_chunk`] will cut, a quarter of [`ELEMWISE_PAR_THRESHOLD`]: a buffer
/// barely over that threshold still reaches four threads, and no chunk is ever small enough for
/// its dispatch to show.
const ELEMWISE_MIN_CHUNK: usize = ELEMWISE_PAR_THRESHOLD / 4;

/// Chunk size for a parallel pass over `len` elements: enough chunks to reach every thread the
/// pool would give this work, never below [`ELEMWISE_MIN_CHUNK`] and never above
/// [`ELEMWISE_CHUNK`], which keeps the chunks small enough to steal on a large buffer.
///
/// [`ELEMWISE_CHUNK`] on its own decides the chunk count from the buffer alone, so a buffer of
/// one or two chunks runs on one or two threads however many are idle -- and `par_chunks_mut`
/// runs a single chunk on the calling thread. Activations between a model's layers land in
/// exactly that range.
fn elemwise_chunk(len: usize) -> usize {
    let threads = crate::threadpool::threads_for(len).max(1);
    len.div_ceil(threads).clamp(ELEMWISE_MIN_CHUNK, ELEMWISE_CHUNK)
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

/// Whether [`arch::unary_inplace`] has a kernel for `op` on `T`: on an empty slice a kernel is
/// a no-op, so running it only answers the question.
fn has_arch_unary<T: 'static>(op: UnaryOp) -> bool {
    arch::unary_inplace::<T>(&mut [], op)
}

/// [`arch::unary_inplace`], split across the pool like [`apply_inplace_unary`] when the work is
/// large enough. Says whether there is a kernel for `op`.
fn arch_unary_inplace<T: Send + Sync + 'static>(dst: &mut [T], op: UnaryOp) -> bool {
    if !use_parallelism(dst.len()) {
        return arch::unary_inplace(dst, op);
    }
    has_arch_unary::<T>(op) && {
        let chunk = elemwise_chunk(dst.len());
        crate::threadpool::par_chunks_mut(dst, chunk, |_, d| {
            arch::unary_inplace(d, op);
        });
        true
    }
}

/// [`arch::unary`], split across the pool like [`apply_unary`] when the work is large enough.
/// Says whether there is a kernel for `op`.
fn arch_unary<T: Send + Sync + 'static>(dst: &mut [T], src: &[T], op: UnaryOp) -> bool {
    if !use_parallelism(dst.len()) {
        return arch::unary(dst, src, op);
    }
    has_arch_unary::<T>(op) && {
        let chunk = elemwise_chunk(dst.len());
        crate::threadpool::par_chunks_zip(dst, chunk, src, chunk, |_, d, s| {
            arch::unary(d, s, op);
        });
        true
    }
}

/// Apply a unary operation in-place: dst[i] = op(dst[i])
#[inline(always)]
fn apply_inplace_unary<T: Copy + Send + Sync, F>(dst: &mut [T], f: F)
where
    F: Fn(&mut T) + Sync,
{
    if !use_parallelism(dst.len()) {
        for d in dst.iter_mut() {
            f(d);
        }
    } else {
        crate::threadpool::par_chunks_mut(dst, ELEMWISE_CHUNK, |_, dst| {
            for d in dst.iter_mut() {
                f(d);
            }
        });
    }
}

/// Apply a unary operation: dst[i] = op(src[i])
#[inline(always)]
fn apply_unary<T: Copy + Send + Sync, F>(dst: &mut [T], src: &[T], f: F)
where
    F: Fn(T) -> T + Sync,
{
    if !use_parallelism(dst.len()) {
        for (d, s) in dst.iter_mut().zip(src) {
            *d = f(*s);
        }
    } else {
        crate::threadpool::par_chunks_zip(
            dst,
            ELEMWISE_CHUNK,
            src,
            ELEMWISE_CHUNK,
            |_, dst, src| {
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = f(*s);
                }
            },
        );
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

/// The `[B, C * K, L_out]` im2col buffer for [`crate::CpuDevice::conv1d`].
///
/// Row `(c, k)` holds input channel `c` sampled at `l * stride + k * dilation - padding` for
/// every output position `l`, zero where that falls outside the input. The positions inside
/// the input are one contiguous range of `l`, so a row is that range copied (a single
/// `copy_from_slice` at stride 1) with zeros on either side, and no bounds test per element.
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
) -> Result<Vec<T>> {
    let rows = batch * in_channels * l_k;
    // SAFETY: `fill_row` writes every element of its row.
    let mut dst =
        unsafe { <crate::CpuDevice as crate::Backend>::alloc_uninit(rows * l_out, &crate::CPU)? };

    let fill_row = |row: usize, dst_row: &mut [T]| {
        let k_idx = row % l_k;
        let c_idx = (row / l_k) % in_channels;
        let b_idx = row / (l_k * in_channels);
        let src_c = &src[(b_idx * in_channels + c_idx) * length..][..length];
        // Output `l` reads input position `l * stride + shift`, which is inside the input
        // for `l` in `l0..l1`.
        let shift = (k_idx * dilation) as isize - padding as isize;
        let first_inside = |pos: isize| match pos {
            ..=0 => 0,
            pos => (pos as usize).div_ceil(stride),
        };
        let l0 = first_inside(-shift).min(l_out);
        let l1 = first_inside(length as isize - shift).clamp(l0, l_out);
        let (before, rest) = dst_row.split_at_mut(l0);
        let (inside, after) = rest.split_at_mut(l1 - l0);
        before.fill(T::zero());
        after.fill(T::zero());
        if !inside.is_empty() {
            let src_c = &src_c[(l0 * stride).wrapping_add_signed(shift)..];
            if stride == 1 {
                inside.copy_from_slice(&src_c[..inside.len()]);
            } else {
                for (d, s) in inside.iter_mut().zip(src_c.iter().step_by(stride)) {
                    *d = *s;
                }
            }
        }
    };
    if use_parallelism(rows * l_out) {
        crate::threadpool::par_chunks_mut(&mut dst, l_out, |row, dst_row| fill_row(row, dst_row));
    } else {
        dst.chunks_mut(l_out).enumerate().for_each(|(row, dst_row)| fill_row(row, dst_row));
    }
    Ok(dst)
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
        crate::threadpool::par_units(out_channels, |out_c| {
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

/// Col2im for [`crate::CpuDevice::conv_transpose1d`]: overlap-adds `col`, laid out as
/// `[B, L_in, C_out, K]`, into the output `[B, C_out, L_out]`.
///
/// Input position `l` and tap `k` land on output `l * stride + k - padding`; taps that fall
/// outside `0..L_out` are cropped, and output positions no tap reaches stay zero.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all)]
fn col2im1d<T: WithDTypeF>(
    dst: &mut [T],
    col: &[T],
    l_in: usize,
    c_out: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
    l_out: usize,
) {
    // A block of output rows, (b, c) pairs, starting at row `r0`. The loop runs over input
    // positions outside and rows inside: col is then read front to back, and a row is added
    // to again only a block's worth of rows later. With the rows inside, consecutive adds
    // would each wait on the store of the one before, since their spans overlap.
    let accumulate = |r0: usize, dst: &mut [T]| {
        dst.fill(T::zero());
        let rows = dst.len() / l_out;
        let mut done = 0;
        while done < rows {
            // The rows left in this batch entry: consecutive channels, whose taps for one input
            // position sit next to each other in col.
            let (b_i, c_i) = ((r0 + done) / c_out, (r0 + done) % c_out);
            let n = (c_out - c_i).min(rows - done);
            let block = &mut dst[done * l_out..(done + n) * l_out];
            for l_in_i in 0..l_in {
                let start = (l_in_i * stride) as isize - padding as isize;
                let k0 = start.min(0).unsigned_abs().min(k_size);
                let k1 = (l_out as isize - start).clamp(k0 as isize, k_size as isize) as usize;
                if k0 == k1 {
                    continue;
                }
                let o = start.wrapping_add_unsigned(k0) as usize;
                let base = ((b_i * l_in + l_in_i) * c_out + c_i) * k_size;
                let taps = col[base..base + n * k_size].chunks_exact(k_size);
                for (row, taps) in block.chunks_exact_mut(l_out).zip(taps) {
                    add_assign(&mut row[o..o + k1 - k0], &taps[k0..k1]);
                }
            }
            done += n;
        }
    };
    let rows = dst.len() / l_out.max(1);
    // The work is the larger of the two buffers, for the pool's decision as for this one.
    let work = dst.len().max(col.len());
    if use_parallelism(work) {
        let per = rows.div_ceil(crate::threadpool::size());
        crate::threadpool::par_chunks_mut_by(work, dst, per * l_out, |i, dst| {
            accumulate(i * per, dst)
        });
    } else if rows > 0 {
        accumulate(0, dst);
    }
}

/// `dst += src`, elementwise, in fixed blocks of four so that even a short span is added
/// with vector instructions.
#[inline(always)]
fn add_assign<T: WithDType>(dst: &mut [T], src: &[T]) {
    let (dst4, dst_tail) = dst.as_chunks_mut::<4>();
    let (src4, src_tail) = src.as_chunks::<4>();
    for (d, s) in dst4.iter_mut().zip(src4) {
        for i in 0..4 {
            d[i] += s[i];
        }
    }
    for (d, s) in dst_tail.iter_mut().zip(src_tail) {
        *d += *s;
    }
}

/// Direct conv_transpose1d implementation (fallback for grouped convolutions).
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

    // Reorder input from [B, C, L] to [B, L, C] for contiguous memory access. A depthwise
    // convolution reads one channel at a time, which is contiguous already.
    let mut src_reordered = vec![];
    if in_c_per_group > 1 {
        src_reordered = vec![T::zero(); batch * length * in_channels];
        for b in 0..batch {
            for l in 0..length {
                for c in 0..in_channels {
                    let src_idx = b * in_channels * length + c * length + l;
                    let dst_idx = b * length * in_channels + l * in_channels + c;
                    src_reordered[dst_idx] = src[src_idx];
                }
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
        // Kernel layout: [in_channels, out_channels/groups, kernel_size]
        let k_base = in_c_start * out_c_per_group * kernel_size + oc_in_group * kernel_size;

        if in_c_per_group == 1 {
            // Depthwise: the dot product over input channels collapses to a single multiply,
            // so each input position adds its value times this channel's taps, which are
            // contiguous, to a contiguous span of the output. Walking the input backwards adds
            // into each output position in increasing tap order, as the general loop below does.
            let x = &src[(b * in_channels + in_c_start) * length..][..length];
            let w = &kernel[k_base..k_base + kernel_size];
            for (il, &xv) in x.iter().enumerate().rev() {
                let start = (il * stride) as isize - padding as isize;
                let k0 = start.min(0).unsigned_abs().min(kernel_size);
                let k1 = (out_length as isize - start).clamp(k0 as isize, kernel_size as isize);
                let (k1, o) = (k1 as usize, start.wrapping_add_unsigned(k0) as usize);
                if k0 < k1 {
                    for (d, &wk) in dst_row[o..o + k1 - k0].iter_mut().zip(&w[k0..k1]) {
                        *d += xv * wk;
                    }
                }
            }
        } else {
            let src_row = &src_reordered[b * length * in_channels..(b + 1) * length * in_channels];
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

mod arch {
    //! Target-specific kernels for the CPU backend, behind one interface. Each function returns
    //! whether it did the work; on a target without such a kernel it is a constant `false`, so the
    //! backend itself carries no `cfg`. Only `f32` has kernels; other dtypes always get `false`.

    use crate::UnaryOp;

    /// A 1d convolution's shape, as the backend's `conv1d` receives it.
    #[cfg_attr(
        not(any(
            all(target_arch = "wasm32", target_feature = "simd128"),
            all(target_arch = "aarch64", target_feature = "neon")
        )),
        allow(dead_code)
    )]
    pub(super) struct Conv1d {
        pub batch: usize,
        pub in_channels: usize,
        pub out_channels: usize,
        pub length: usize,
        pub out_length: usize,
        pub kernel_size: usize,
        pub stride: usize,
        pub padding: usize,
        pub dilation: usize,
        pub groups: usize,
    }

    /// A batched matmul's layout, as the backend's `gemm_` receives it: `(col, row)` strides.
    #[cfg_attr(
        not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        allow(dead_code)
    )]
    pub(super) struct Gemm {
        pub m: usize,
        pub n: usize,
        pub k: usize,
        pub batch: usize,
        pub lhs_b_stride: usize,
        pub rhs_b_stride: usize,
        pub dst: (usize, usize),
        pub lhs: (usize, usize),
        pub rhs: (usize, usize),
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    mod imp {
        use super::{Conv1d, Gemm};
        pub use crate::simd_math::{softmax_row, unary, unary_inplace};
        use crate::simd128_conv as k;

        /// Stride 1, no padding, one group: read the input in place, no im2col or transpose.
        pub fn conv1d(dst: &mut [f32], src: &[f32], w: &[f32], s: &Conv1d) -> bool {
            let span = (s.kernel_size.max(1) - 1) * s.dilation;
            let fits = s.out_length == 0 || s.out_length - 1 + span < s.length;
            if s.groups != 1 || s.stride != 1 || s.padding != 0 || s.kernel_size == 0 || !fits {
                return false;
            }
            let (l_in, l_out) = (s.length, s.out_length);
            k::conv1d(
                dst,
                src,
                w,
                s.batch,
                s.in_channels,
                s.out_channels,
                l_in,
                l_out,
                s.kernel_size,
                s.dilation,
            );
            true
        }

        /// `x @ W^T` on a row-major weight, both operands contiguous along `k`, into a dense
        /// output: each batch entry is `m * n` floats, as `gemm_` lays them out.
        pub fn gemm(dst: &mut [f32], lhs: &[f32], rhs: &[f32], g: &Gemm) -> bool {
            let (((dst_cs, dst_rs), (lhs_cs, lhs_rs)), (rhs_cs, rhs_rs)) = ((g.dst, g.lhs), g.rhs);
            let dense = dst_rs == g.n || g.m <= 1;
            if lhs_cs != 1 || dst_cs != 1 || rhs_rs != 1 || !dense {
                return false;
            }
            let (m, n, kk) = (g.m, g.n, g.k);
            for b in 0..g.batch {
                let dst = &mut dst[b * m * n..(b + 1) * m * n];
                let (lhs, rhs) = (&lhs[b * g.lhs_b_stride..], &rhs[b * g.rhs_b_stride..]);
                k::gemm_dot(dst, dst_rs, lhs, lhs_rs, rhs, rhs_cs, m, n, kk);
            }
            true
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    mod imp {
        use super::{Conv1d, Gemm};
        use crate::neon_conv as k;
        pub use crate::simd_math::{softmax_row, unary, unary_inplace};

        /// Stride 1, no padding, one group, and a shape `neon_conv::takes`: read the input in
        /// place, no im2col or transpose.
        pub fn conv1d(dst: &mut [f32], src: &[f32], w: &[f32], s: &Conv1d) -> bool {
            let span = (s.kernel_size.max(1) - 1) * s.dilation;
            let fits = s.out_length == 0 || s.out_length - 1 + span < s.length;
            if s.groups != 1 || s.stride != 1 || s.padding != 0 || s.kernel_size == 0 || !fits {
                return false;
            }
            if !k::takes(s.in_channels, s.out_channels, s.length, s.out_length) {
                return false;
            }
            k::conv1d(
                dst,
                src,
                w,
                s.batch,
                s.in_channels,
                s.out_channels,
                s.length,
                s.out_length,
                s.kernel_size,
                s.dilation,
            );
            true
        }

        /// `x @ W^T` on a row-major weight, both operands contiguous along `k`, for up to
        /// `MAX_ROWS` rows, into a dense output: each batch entry is `m * n` floats, as `gemm_`
        /// lays them out. Skipping the packing pays for a handful of rows; past a few dozen,
        /// xn-gemm's packed kernel, which loads fewer operands per multiply-add, catches up and
        /// then wins.
        pub fn gemm(dst: &mut [f32], lhs: &[f32], rhs: &[f32], g: &Gemm) -> bool {
            #[cfg(xn_kai)]
            if crate::kai_f32::gemm(
                dst,
                lhs,
                rhs,
                (g.m, g.n, g.k),
                g.batch,
                (g.lhs_b_stride, g.rhs_b_stride),
                g.dst,
                g.lhs,
                g.rhs,
            ) {
                return true;
            }
            const MAX_ROWS: usize = 64;
            let (((dst_cs, dst_rs), (lhs_cs, lhs_rs)), (rhs_cs, rhs_rs)) = ((g.dst, g.lhs), g.rhs);
            let dense = dst_rs == g.n || g.m <= 1;
            if lhs_cs != 1 || dst_cs != 1 || rhs_rs != 1 || !dense || g.m > MAX_ROWS {
                return false;
            }
            let batch =
                crate::neon_gemm::Batch { n: g.batch, lhs: g.lhs_b_stride, rhs: g.rhs_b_stride };
            let (m, n, kk) = (g.m, g.n, g.k);
            crate::neon_gemm::gemm_dot_batched(
                dst, dst_rs, lhs, lhs_rs, rhs, rhs_cs, m, n, kk, batch,
            );
            true
        }
    }

    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    mod imp {
        use super::{Conv1d, Gemm};
        use crate::UnaryOp;
        #[inline(always)]
        pub fn unary_inplace(_: &mut [f32], _: UnaryOp) -> bool {
            false
        }
        #[inline(always)]
        pub fn unary(_: &mut [f32], _: &[f32], _: UnaryOp) -> bool {
            false
        }
        #[inline(always)]
        pub fn softmax_row(_: &mut [f32], _: &[f32]) -> bool {
            false
        }
        #[inline(always)]
        pub fn conv1d(_: &mut [f32], _: &[f32], _: &[f32], _: &Conv1d) -> bool {
            false
        }
        #[inline(always)]
        pub fn gemm(_: &mut [f32], _: &[f32], _: &[f32], _: &Gemm) -> bool {
            false
        }
    }

    pub(super) fn unary_inplace<T: 'static>(dst: &mut [T], op: UnaryOp) -> bool {
        as_f32_mut(dst).is_some_and(|dst| imp::unary_inplace(dst, op))
    }

    pub(super) fn unary<T: 'static>(dst: &mut [T], src: &[T], op: UnaryOp) -> bool {
        match (as_f32_mut(dst), as_f32(src)) {
            (Some(dst), Some(src)) => imp::unary(dst, src, op),
            _ => false,
        }
    }

    /// One row of a softmax over the last dimension.
    pub(super) fn softmax_row<T: 'static>(dst: &mut [T], src: &[T]) -> bool {
        match (as_f32_mut(dst), as_f32(src)) {
            (Some(dst), Some(src)) => imp::softmax_row(dst, src),
            _ => false,
        }
    }

    pub(super) fn conv1d<T: 'static>(dst: &mut [T], src: &[T], w: &[T], s: &Conv1d) -> bool {
        match (as_f32_mut(dst), as_f32(src), as_f32(w)) {
            (Some(dst), Some(src), Some(w)) => imp::conv1d(dst, src, w, s),
            _ => false,
        }
    }

    pub(super) fn gemm<T: 'static>(dst: &mut [T], lhs: &[T], rhs: &[T], g: &Gemm) -> bool {
        match (as_f32_mut(dst), as_f32(lhs), as_f32(rhs)) {
            (Some(dst), Some(lhs), Some(rhs)) => imp::gemm(dst, lhs, rhs, g),
            _ => false,
        }
    }

    /// `s` as `f32` when `T` is `f32`, else `None`.
    fn as_f32<T: 'static>(s: &[T]) -> Option<&[f32]> {
        (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>())
            // SAFETY: `T` is `f32`; length and alignment carry over unchanged.
            .then(|| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, s.len()) })
    }

    /// `s` as `f32` when `T` is `f32`, else `None`.
    fn as_f32_mut<T: 'static>(s: &mut [T]) -> Option<&mut [f32]> {
        (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>())
            // SAFETY: `T` is `f32`; length and alignment carry over unchanged.
            .then(|| unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut f32, s.len()) })
    }
}

#[cfg(test)]
mod tests {
    use crate::{CPU, CpuDevice, Tensor};

    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    fn tensor(v: &[f32], dims: &[usize]) -> Tensor<f32, CpuDevice> {
        Tensor::from_vec(v.to_vec(), dims.to_vec(), &CPU).unwrap()
    }

    /// `want[b, i, j] = sum_t x(b, i, t) * w(b, j, t)`.
    fn reference(
        (batch, m, n, k): (usize, usize, usize, usize),
        x: impl Fn(usize, usize, usize) -> f32,
        w: impl Fn(usize, usize, usize) -> f32,
    ) -> Vec<f32> {
        let mut want = vec![0f32; batch * m * n];
        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    want[(b * m + i) * n + j] = (0..k).map(|t| x(b, i, t) * w(b, j, t)).sum();
                }
            }
        }
        want
    }

    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{what}[{i}]: got {g}, want {w}");
        }
    }

    /// A batched input against one 2-D weight: the weight's batch stride is 0, which the
    /// kernels only see from `gemm_`.
    #[test]
    fn batched_matmul_t_with_a_broadcast_weight() {
        for (batch, m, n, k) in [(3, 5, 20, 9), (2, 16, 72, 64), (1, 1, 300, 256)] {
            let (xv, wv) = (data(batch * m * k, 7), data(n * k, 8));
            let got = tensor(&xv, &[batch, m, k]).matmul_t(&tensor(&wv, &[n, k])).unwrap();
            let want = reference(
                (batch, m, n, k),
                |b, i, t| xv[(b * m + i) * k + t],
                |_, j, t| wv[j * k + t],
            );
            assert_close(&got.to_vec().unwrap(), &want, &format!("{batch}x{m}x{n}x{k}"));
        }
    }

    /// One weight per batch entry, as attention's `q @ k^T` has.
    #[test]
    fn batched_matmul_t_with_a_weight_per_entry() {
        let (batch, m, n, k) = (8, 16, 37, 64);
        let (xv, wv) = (data(batch * m * k, 9), data(batch * n * k, 10));
        let got = tensor(&xv, &[batch, m, k]).matmul_t(&tensor(&wv, &[batch, n, k])).unwrap();
        let want = reference(
            (batch, m, n, k),
            |b, i, t| xv[(b * m + i) * k + t],
            |b, j, t| wv[(b * n + j) * k + t],
        );
        assert_close(&got.to_vec().unwrap(), &want, "per-entry weight");
    }

    /// Layouts the direct kernel does not take must still come out right: a transposed lhs,
    /// and a weight that is contiguous along `n` rather than `k`.
    #[test]
    fn matmul_layouts_that_fall_back() {
        let (m, n, k) = (6, 10, 12);
        let (xv, wv) = (data(m * k, 11), data(n * k, 12));
        // `x` stored as `[k, m]` and transposed: lhs columns are strided.
        let xt = tensor(&xv, &[k, m]);
        let got = xt.t().unwrap().matmul_t(&tensor(&wv, &[n, k])).unwrap();
        let want = reference((1, m, n, k), |_, i, t| xv[t * m + i], |_, j, t| wv[j * k + t]);
        assert_close(&got.to_vec().unwrap(), &want, "transposed lhs");
        // `w` stored as `[k, n]`: `x @ w`.
        let got = tensor(&xv, &[m, k]).matmul(&tensor(&wv, &[k, n])).unwrap();
        let want = reference((1, m, n, k), |_, i, t| xv[i * k + t], |_, j, t| wv[t * n + j]);
        assert_close(&got.to_vec().unwrap(), &want, "x @ w");
    }

    /// Which matmuls the target kernel takes: `f32` only, with `lhs` and the output contiguous
    /// along their rows and `rhs` along `k`, and on aarch64 only up to 64 rows. KleidiAI's
    /// kernels, when in use, also take more rows and the other layouts of `lhs` and `rhs`.
    #[test]
    fn arch_gemm_guards() {
        use super::arch::{Gemm, gemm};
        let (n, k) = (8, 16);
        let layout = |m, dst, lhs, rhs| Gemm {
            m,
            n,
            k,
            batch: 1,
            lhs_b_stride: m * k,
            rhs_b_stride: 0,
            dst,
            lhs,
            rhs,
        };
        let (lhs, rhs) = (data(65 * k, 13), data(n * k, 14));
        let mut dst = vec![0f32; 65 * n];
        let wasm = cfg!(all(target_arch = "wasm32", target_feature = "simd128"));
        let neon = cfg!(all(target_arch = "aarch64", target_feature = "neon"));
        #[cfg(xn_kai)]
        let kai = crate::quantized::kai::active();
        #[cfg(not(xn_kai))]
        let kai = false;
        let x_wt = |m| layout(m, (1, n), (1, k), (k, 1));
        assert_eq!(gemm(&mut dst, &lhs, &rhs, &x_wt(4)), wasm || neon);
        assert_eq!(gemm(&mut dst, &lhs, &rhs, &x_wt(65)), wasm || kai, "65 rows");
        let (lhs64, rhs64): (Vec<f64>, Vec<f64>) =
            (lhs.iter().map(|&v| v as f64).collect(), rhs.iter().map(|&v| v as f64).collect());
        assert!(!gemm(&mut vec![0f64; 4 * n], &lhs64, &rhs64, &x_wt(4)), "f64");
        assert_eq!(gemm(&mut dst, &lhs, &rhs, &layout(4, (1, n), (4, 1), (k, 1))), kai, "kxm lhs");
        assert_eq!(gemm(&mut dst, &lhs, &rhs, &layout(4, (1, n), (1, k), (1, n))), kai, "x @ w");
        assert!(!gemm(&mut dst, &lhs, &rhs, &layout(4, (4, 1), (1, k), (k, 1))), "strided dst");
        // Rows of the output further apart than `n` are not the dense layout the hook writes.
        let padded = layout(4, (1, n + 3), (1, k), (k, 1));
        assert_eq!(gemm(&mut vec![0f32; 4 * (n + 3)], &lhs, &rhs, &padded), kai, "padded dst rows");
        // One row has no row stride to speak of.
        assert_eq!(
            gemm(&mut dst, &lhs, &rhs, &layout(1, (1, n + 3), (1, k), (k, 1))),
            wasm || neon
        );
    }

    /// A buffer past the parallel threshold is cut into enough chunks to reach every thread the
    /// pool would give it, which a fixed chunk size does not do in the range activations fall in.
    #[test]
    fn elemwise_chunks_reach_every_thread() {
        use super::{ELEMWISE_CHUNK, ELEMWISE_MIN_CHUNK, ELEMWISE_PAR_THRESHOLD, elemwise_chunk};

        for len in [ELEMWISE_PAR_THRESHOLD, 61_440, 122_880, 8 * 1024 * 1024] {
            let chunk = elemwise_chunk(len);
            assert!(chunk >= ELEMWISE_MIN_CHUNK, "len {len}: chunk {chunk} under the floor");
            assert!(chunk <= ELEMWISE_CHUNK, "len {len}: chunk {chunk} over the cap");
            let threads = crate::threadpool::threads_for(len).max(1);
            let want = threads.min(len / ELEMWISE_MIN_CHUNK);
            assert!(
                len.div_ceil(chunk) >= want,
                "len {len}: {} chunks for {want} threads",
                len.div_ceil(chunk)
            );
        }
    }
}
