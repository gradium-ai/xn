// Included from `mod.rs`. Implements the `Backend` trait for the Vulkan
// `Device`, plus host fallbacks for non-f32 data-movement ops.

/// Convert a float-typed scalar value to `f32`. Only called on the GPU path,
/// where `float_suffix` has already restricted `T` to {f32, f16, bf16}.
fn scalar_to_f32<T: WithDType>(v: T) -> f32 {
    match T::DTYPE {
        DType::F32 => unsafe { *(&v as *const T as *const f32) },
        DType::F16 => unsafe { (*(&v as *const T as *const half::f16)).to_f32() },
        DType::BF16 => unsafe { (*(&v as *const T as *const half::bf16)).to_f32() },
        d => unreachable!("scalar_to_f32 on non-float dtype {d:?}"),
    }
}

fn bin_apply<T: WithDType>(op: BinaryOp, a: T, b: T) -> T {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Maximum => {
            if a > b {
                a
            } else {
                b
            }
        }
        BinaryOp::Minimum => {
            if a < b {
                a
            } else {
                b
            }
        }
    }
}

impl crate::Backend for Device {
    type Storage<T: WithDType> = Storage<T>;

    fn name(&self) -> String {
        format!("Vulkan ({})", self.device_name)
    }

    fn synchronize(&self) -> Result<()> {
        // Submit any pending batch and wait for it.
        self.flush("synchronize")?;
        unsafe { self.device.device_wait_idle() }.map_err(vkerr("device_wait_idle"))?;
        Ok(())
    }

    fn storage_len<T: WithDType>(storage: &Self::Storage<T>) -> usize {
        storage.len
    }

    unsafe fn alloc_uninit<T: WithDType>(len: usize, dev: &Self) -> Result<Self::Storage<T>> {
        let (buffer, memory, ptr, class) = dev.alloc_buffer(len * T::BYTE_SIZE)?;
        Ok(Storage { buffer, memory, ptr, len, class, device: dev.clone(), _t: PhantomData })
    }

    fn from_vec<T: WithDType>(v: Vec<T>, dev: &Self) -> Result<Self::Storage<T>> {
        let len = v.len();
        let storage = unsafe { Self::alloc_uninit::<T>(len, dev)? };
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, storage.ptr, len * T::BYTE_SIZE);
        }
        Ok(storage)
    }

    fn fill<T: WithDType>(dst: &mut Self::Storage<T>, elem: T, len: usize) -> Result<()> {
        // GPU fill for float dtypes so zeros/full stay inside the batch.
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(len).f32(scalar_to_f32(elem));
            return dst.device.dispatch(
                &format!("fill_{dt}"),
                &[dst.buffer],
                &push,
                div_ceil(len, WORKGROUP_SIZE),
            );
        }
        dst.device.flush("fill-host")?;
        dst.as_mut_slice()[..len].fill(elem);
        Ok(())
    }

    fn rand_uniform(dst: &mut Self::Storage<f32>, len: usize, lo: f32, up: f32) -> Result<()> {
        // Values are generated on the host, but into a *fresh* scratch buffer
        // (no pending GPU references) and copied into dst inside the batch, so
        // no flush is needed.
        let range = up - lo;
        let data: Vec<f32> = (0..len).map(|_| rand::random::<f32>() * range + lo).collect();
        let scratch = dst.device.scratch_from_slice(&data)?;
        let res = dst.device.record_copy(dst.buffer, scratch.buffer, len * 4);
        dst.device.defer_free(scratch);
        res
    }

    fn randn(dst: &mut Self::Storage<f32>, len: usize, mean: f32, std: f32) -> Result<()> {
        use rand_distr::Distribution;
        let distr = match rand_distr::Normal::<f32>::new(mean, std) {
            Ok(d) => d,
            Err(e) => crate::bail!("failed to create normal distribution for randn: {e}"),
        };
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..len).map(|_| distr.sample(&mut rng)).collect();
        let scratch = dst.device.scratch_from_slice(&data)?;
        let res = dst.device.record_copy(dst.buffer, scratch.buffer, len * 4);
        dst.device.defer_free(scratch);
        res
    }

    fn copy<T: WithDType>(dst: &mut Self::Storage<T>, src: &Self::Storage<T>, len: usize) -> Result<()> {
        // Recorded as a GPU buffer copy so it stays in the batch.
        dst.device.record_copy(dst.buffer, src.buffer, len * T::BYTE_SIZE)
    }

    fn to_dtype<T: WithDType, U: WithDType>(
        dst: &mut Self::Storage<U>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()> {
        use half::{bf16, f16};
        let dev = &src.device;
        // Same dtype: a plain in-batch buffer copy.
        if T::DTYPE == U::DTYPE {
            return dev.record_copy(dst.buffer, src.buffer, len * T::BYTE_SIZE);
        }
        // GPU cast kernels for the hot pairs (models cast around rope and
        // sampling every step; a host cast would force a flush per call).
        let f16_ok = dev.supports_f16;
        let bf16_ok = dev.supports_bf16;
        let kname = match (T::DTYPE, U::DTYPE) {
            (DType::F32, DType::F16) if f16_ok => Some("cast_f32_f16"),
            (DType::F16, DType::F32) if f16_ok => Some("cast_f16_f32"),
            (DType::F32, DType::BF16) if bf16_ok => Some("cast_f32_bf16"),
            (DType::BF16, DType::F32) if bf16_ok => Some("cast_bf16_f32"),
            (DType::F16, DType::BF16) if f16_ok && bf16_ok => Some("cast_f16_bf16"),
            (DType::BF16, DType::F16) if f16_ok && bf16_ok => Some("cast_bf16_f16"),
            (DType::I64, DType::F32) => Some("cast_i64_f32"),
            _ => None,
        };
        if let Some(kname) = kname {
            let push = Pc::new().usize(len);
            return dev.dispatch(
                kname,
                &[src.buffer, dst.buffer],
                &push,
                div_ceil(len, WORKGROUP_SIZE),
            );
        }
        // Host fallback for the remaining pairs.
        dev.flush("to_dtype-host")?;
        macro_rules! cast {
            ($s:ty, $d:ty, |$v:ident| $e:expr) => {{
                let s = unsafe { std::slice::from_raw_parts(src.ptr as *const $s, len) };
                let d = unsafe { std::slice::from_raw_parts_mut(dst.ptr as *mut $d, len) };
                for (o, i) in d.iter_mut().zip(s.iter()) {
                    let $v = *i;
                    *o = $e;
                }
            }};
        }
        use DType::*;
        match (T::DTYPE, U::DTYPE) {
            (F16, F16) => cast!(f16, f16, |v| v),
            (BF16, BF16) => cast!(bf16, bf16, |v| v),
            (F32, F32) => cast!(f32, f32, |v| v),
            (I64, I64) => cast!(i64, i64, |v| v),
            (U8, U8) => cast!(u8, u8, |v| v),
            (F32, F16) => cast!(f32, f16, |v| f16::from_f32(v)),
            (F32, BF16) => cast!(f32, bf16, |v| bf16::from_f32(v)),
            (F16, F32) => cast!(f16, f32, |v| v.to_f32()),
            (BF16, F32) => cast!(bf16, f32, |v| v.to_f32()),
            (F16, BF16) => cast!(f16, bf16, |v| bf16::from_f32(v.to_f32())),
            (BF16, F16) => cast!(bf16, f16, |v| f16::from_f32(v.to_f32())),
            (F32, I64) => cast!(f32, i64, |v| v as i64),
            (F32, U8) => cast!(f32, u8, |v| v as u8),
            (F16, I64) => cast!(f16, i64, |v| v.to_f32() as i64),
            (F16, U8) => cast!(f16, u8, |v| v.to_f32() as u8),
            (BF16, I64) => cast!(bf16, i64, |v| v.to_f32() as i64),
            (BF16, U8) => cast!(bf16, u8, |v| v.to_f32() as u8),
            (I64, F32) => cast!(i64, f32, |v| v as f32),
            (I64, F16) => cast!(i64, f16, |v| f16::from_f32(v as f32)),
            (I64, BF16) => cast!(i64, bf16, |v| bf16::from_f32(v as f32)),
            (U8, F32) => cast!(u8, f32, |v| v as f32),
            (U8, F16) => cast!(u8, f16, |v| f16::from_f32(v as f32)),
            (U8, BF16) => cast!(u8, bf16, |v| bf16::from_f32(v as f32)),
            (I64, U8) => cast!(i64, u8, |v| v as u8),
            (U8, I64) => cast!(u8, i64, |v| v as i64),
        }
        Ok(())
    }

    fn data<T: WithDType>(src: &Self::Storage<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        src.device.flush("data-readback")?;
        Ok(std::borrow::Cow::Owned(src.as_slice()[..len].to_vec()))
    }

    fn inplace_unary<T: WithDTypeF>(dst: &mut Self::Storage<T>, len: usize, op: UnaryOp) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "inplace_unary")?;
        let (code, alpha) = unary_op_code(op);
        let push = Pc::new().usize(len).u32(code).f32(alpha);
        dst.device.dispatch(
            &format!("unary_{dt}"),
            &[dst.buffer, dst.buffer],
            &push,
            div_ceil(len, WORKGROUP_SIZE),
        )
    }

    fn unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "unary")?;
        let (code, alpha) = unary_op_code(op);
        let push = Pc::new().usize(len).u32(code).f32(alpha);
        dst.device.dispatch(
            &format!("unary_{dt}"),
            &[src.buffer, dst.buffer],
            &push,
            div_ceil(len, WORKGROUP_SIZE),
        )
    }

    fn bin_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        s: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(len).u32(binary_op_code(op));
            dst.device.dispatch(
                &format!("binary_{dt}"),
                &[dst.buffer, s.buffer, dst.buffer],
                &push,
                div_ceil(len, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush("bin_assign-host")?;
            let src = s.as_slice()[..len].to_vec();
            for (d, sv) in dst.as_mut_slice()[..len].iter_mut().zip(src) {
                *d = bin_apply(op, *d, sv);
            }
            Ok(())
        }
    }

    fn binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(len).u32(binary_op_code(op));
            dst.device.dispatch(
                &format!("binary_{dt}"),
                &[lhs.buffer, rhs.buffer, dst.buffer],
                &push,
                div_ceil(len, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush("binary-host")?;
            let l = lhs.as_slice();
            let r = rhs.as_slice();
            let d = dst.as_mut_slice();
            for i in 0..len {
                d[i] = bin_apply(op, l[i], r[i]);
            }
            Ok(())
        }
    }

    fn scale_add<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        scale: T,
        add: T,
        len: usize,
    ) -> Result<()> {
        if add == T::zero() && scale == T::one() {
            return Self::copy(dst, src, len);
        }
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(len).f32(scalar_to_f32(scale)).f32(scalar_to_f32(add));
            dst.device.dispatch(
                &format!("scale_add_{dt}"),
                &[src.buffer, dst.buffer],
                &push,
                div_ceil(len, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush("scale_add-host")?;
            let s = src.as_slice()[..len].to_vec();
            for (d, sv) in dst.as_mut_slice()[..len].iter_mut().zip(s) {
                *d = sv * scale + add;
            }
            Ok(())
        }
    }

    fn transpose<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()> {
        let numel: usize = dims.iter().product();
        if dim1 == dim2 || dims.iter().filter(|v| **v != 1).count() <= 1 {
            return Self::copy(dst, src, numel);
        }
        let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
        let d_i: usize = dims[..dim1].iter().product();
        let d_j: usize = dims[dim1 + 1..dim2].iter().product();
        let d_k: usize = dims[(dim2 + 1)..].iter().product();
        let d1 = dims[dim1];
        let d2 = dims[dim2];
        if let Some(dt) = movement_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(numel).usize(d1).usize(d2).usize(d_i).usize(d_j).usize(d_k);
            dst.device.dispatch(
                &format!("transpose_{dt}"),
                &[src.buffer, dst.buffer],
                &push,
                div_ceil(numel, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush("transpose-host")?;
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            for dst_idx in 0..numel {
                let mut rem = dst_idx;
                let i = rem / (d2 * d_j * d1 * d_k);
                rem -= i * (d2 * d_j * d1 * d_k);
                let a2 = rem / (d_j * d1 * d_k);
                rem -= a2 * (d_j * d1 * d_k);
                let j = rem / (d1 * d_k);
                rem -= j * (d1 * d_k);
                let a1 = rem / d_k;
                rem -= a1 * d_k;
                let k = rem;
                let src_idx = i * d1 * d_j * d2 * d_k
                    + a1 * d_j * d2 * d_k
                    + j * d2 * d_k
                    + a2 * d_k
                    + k;
                d[dst_idx] = s[src_idx];
            }
            Ok(())
        }
    }

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
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        if let Some(dt) = movement_suffix::<T>(&dst.device) {
            let push =
                Pc::new().usize(d1).usize(d2).usize(src_s).usize(dst_s).usize(src_o).usize(dst_o);
            dst.device.dispatch(
                &format!("copy2d_{dt}"),
                &[src.buffer, dst.buffer],
                &push,
                div_ceil(d1 * d2, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush("copy2d-host")?;
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            for i1 in 0..d1 {
                for i2 in 0..d2 {
                    d[dst_o + i1 * dst_s + i2] = s[src_o + i1 * src_s + i2];
                }
            }
            Ok(())
        }
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
        let dt = dtype_suffix::<T>(&dst.device, "rope")?;
        let bh = b * h;
        let td = t * d;
        let cs_stride_b = if unbatched_rope { t * d / 2 } else { 0 };
        let off = pos * d / 2;
        let push = Pc::new()
            .usize(bh)
            .usize(td)
            .usize(d)
            .usize(h)
            .usize(cs_stride_b)
            .usize(off)
            .usize(off);
        dst.device.dispatch(
            &format!("rope_{dt}"),
            &[cos.buffer, sin.buffer, src.buffer, dst.buffer],
            &push,
            div_ceil(bh * td / 2, WORKGROUP_SIZE),
        )
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
        let dt = dtype_suffix::<T>(&dst.device, "rope_i")?;
        let bh = b * h;
        let td = t * d;
        let cs_stride_b = if unbatched_rope { t * d / 2 } else { 0 };
        let off = pos * d / 2;
        let push = Pc::new().usize(bh).usize(td).usize(h).usize(cs_stride_b).usize(off).usize(off);
        dst.device.dispatch(
            &format!("rope_i_{dt}"),
            &[cos.buffer, sin.buffer, src.buffer, dst.buffer],
            &push,
            div_ceil(bh * td / 2, WORKGROUP_SIZE),
        )
    }

    fn gemm<T: WithDType>(
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
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "gemm")?;
        if T::DTYPE == DType::F32
            && row_block_gemm(
                dst,
                lhs,
                rhs,
                m,
                n,
                k,
                lhs_b,
                lhs_b_stride,
                rhs_b_stride,
                dst_strides,
                lhs_strides,
                rhs_strides,
            )?
        {
            return Ok(());
        }
        let (dst_cs, dst_rs) = dst_strides;
        let (lhs_cs, lhs_rs) = lhs_strides;
        let (rhs_cs, rhs_rs) = rhs_strides;
        let push = Pc::new()
            .usize(m)
            .usize(n)
            .usize(k)
            .usize(lhs_b)
            .usize(lhs_b_stride)
            .usize(rhs_b_stride)
            .usize(lhs_cs)
            .usize(lhs_rs)
            .usize(rhs_cs)
            .usize(rhs_rs)
            .usize(dst_rs)
            .usize(dst_cs)
            .usize(lhs.1)
            .usize(rhs.1);
        // Profile rows split by shape and layout: `l`/`r` are the (col, row)
        // element strides of lhs and rhs, so `r1,k` is a row-major weight
        // read as `matmul_t` and `rn,1` a row-major rhs read as `matmul`.
        let label = |kernel: &str| {
            dst.device.profile_enabled.then(|| {
                format!("{kernel} m{m} n{n} k{k} b{lhs_b} l{lhs_cs},{lhs_rs} r{rhs_cs},{rhs_rs}")
            })
        };
        if m == 1 {
            // Decode path: one workgroup per output column, grid (n, batch, 1).
            // rhs is bound twice: once scalar, once as a vec4 view for the
            // shader's aligned-fast-path loads (see gemv.comp).
            let buffers = [dst.buffer, lhs.0.buffer, rhs.0.buffer, rhs.0.buffer];
            let kernel = format!("gemv_{dt}");
            let groups = (n as u32, lhs_b as u32, 1);
            dst.device.dispatch_labeled(&kernel, label(&kernel), &buffers, &push, groups)
        } else {
            let buffers = [dst.buffer, lhs.0.buffer, rhs.0.buffer];
            // Tiled kernel: grid (ceil(n/16), ceil(m/16), batch), local (16, 16, 1).
            const TILE: u32 = 16;
            let groups = (div_ceil(n, TILE), div_ceil(m, TILE), lhs_b as u32);
            let kernel = format!("gemm_tiled_{dt}");
            dst.device.dispatch_labeled(&kernel, label(&kernel), &buffers, &push, groups)
        }
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
        let total = left_size * num_ids * right_size;
        if let Some(dt) = movement_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(left_size).usize(num_ids).usize(right_size).usize(src_dim_size);
            dst.device.dispatch(
                &format!("index_select_{dt}"),
                &[src.buffer, dst.buffer, ids.buffer],
                &push,
                div_ceil(total, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush("index_select-host")?;
            let ids_h = ids.as_slice();
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            for left in 0..left_size {
                for id_i in 0..num_ids {
                    let idx = ids_h[id_i];
                    for r in 0..right_size {
                        let dst_off = (left * num_ids + id_i) * right_size + r;
                        if idx == -1 {
                            d[dst_off] = T::zero();
                        } else {
                            let src_off = (left * src_dim_size + idx as usize) * right_size + r;
                            d[dst_off] = s[src_off];
                        }
                    }
                }
            }
            Ok(())
        }
    }

    fn apply_causality_mask<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "apply_causality_mask")?;
        let total = bh * t1 * t2;
        let push = Pc::new().usize(bh).usize(t1).usize(t2).usize(offset);
        dst.device.dispatch(
            &format!("causality_mask_{dt}"),
            &[dst.buffer],
            &push,
            div_ceil(total, WORKGROUP_SIZE),
        )
    }

    fn softmax<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "softmax")?;
        let push = Pc::new().usize(dim_m1);
        dst.device.dispatch(&format!("softmax_{dt}"), &[src.buffer, dst.buffer], &push, d as u32)
    }

    fn rms_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "rms_norm")?;
        let push = Pc::new().usize(dim_m1).f32(eps);
        dst.device.dispatch(
            &format!("rmsnorm_{dt}"),
            &[src.buffer, dst.buffer, alpha.buffer],
            &push,
            d as u32,
        )
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
        let dt = dtype_suffix::<T>(&dst.device, "layer_norm")?;
        let push = Pc::new().usize(dim_m1).f32(eps).u32(if remove_mean { 1 } else { 0 });
        dst.device.dispatch(
            &format!("layernorm_{dt}"),
            &[src.buffer, dst.buffer, weight.buffer, bias.buffer],
            &push,
            d as u32,
        )
    }

    fn reduce_max<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        dst.device.clone().reduce(dst, src, dim_size, outer_size, inner_size, 1)
    }

    fn reduce_min<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        dst.device.clone().reduce(dst, src, dim_size, outer_size, inner_size, 2)
    }

    fn reduce_sum<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        dst.device.clone().reduce(dst, src, dim_size, outer_size, inner_size, 0)
    }

    fn reduce_argmin<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        dst.device.clone().reduce_arg(dst, src, dim_size, outer_size, inner_size, 0)
    }

    fn reduce_argmax<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        dst.device.clone().reduce_arg(dst, src, dim_size, outer_size, inner_size, 1)
    }

    fn copy_strided<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        src_offset: usize,
        dims: &[usize],
        src_strides: &[usize],
    ) -> Result<()> {
        let numel: usize = dims.iter().product();
        if numel == 0 {
            return Ok(());
        }
        if let Some(dt) = movement_suffix::<T>(&dst.device) {
            let info: Vec<u32> =
                dims.iter().chain(src_strides.iter()).map(|&v| v as u32).collect();
            let scratch = dst.device.scratch_from_slice(&info)?;
            let push = Pc::new().usize(numel).usize(dims.len()).usize(src_offset);
            let res = dst.device.dispatch(
                &format!("copy_strided_{dt}"),
                &[src.buffer, dst.buffer, scratch.buffer],
                &push,
                div_ceil(numel, WORKGROUP_SIZE),
            );
            // Only defer after the dispatch is recorded (see defer_free).
            dst.device.defer_free(scratch);
            res
        } else {
            dst.device.flush("copy_strided-host")?;
            let n = dims.len();
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            for idx in 0..numel {
                let mut si = 0usize;
                let mut rem = idx;
                for di in (0..n).rev() {
                    si += (rem % dims[di]) * src_strides[di];
                    rem /= dims[di];
                }
                d[idx] = s[src_offset + si];
            }
            Ok(())
        }
    }

    fn scatter_set<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &Self::Storage<i64>,
        dim: usize,
        dst_dims: &[usize],
        src_dims: &[usize],
    ) -> Result<()> {
        let right_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let src_dim_size = src_dims[dim];
        let dst_dim_size = dst_dims[dim];
        let numel: usize = src_dims.iter().product();
        if numel == 0 {
            return Ok(());
        }
        if let Some(dt) = movement_suffix::<T>(&dst.device) {
            let push = Pc::new().usize(numel).usize(right_size).usize(src_dim_size).usize(dst_dim_size);
            dst.device.dispatch(
                &format!("scatter_set_{dt}"),
                &[dst.buffer, src.buffer, ids.buffer],
                &push,
                div_ceil(numel, WORKGROUP_SIZE),
            )
        } else {
            dst.device.flush("scatter_set-host")?;
            let ids_h = ids.as_slice();
            let s = src.as_slice();
            let d = dst.as_mut_slice();
            for i in 0..numel {
                let right = i % right_size;
                let left = i / (right_size * src_dim_size);
                let idx = ids_h[i] as usize;
                let dst_off = left * dst_dim_size * right_size + idx * right_size + right;
                d[dst_off] = s[i];
            }
            Ok(())
        }
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
        let numel: usize = dst_shape.iter().product();
        if numel == 0 {
            return Ok(());
        }
        if let Some(dt) = float_suffix::<T>(&dst.device) {
            let info: Vec<u32> = dst_shape
                .iter()
                .chain(lhs_strides.iter())
                .chain(rhs_strides.iter())
                .map(|&v| v as u32)
                .collect();
            let scratch = dst.device.scratch_from_slice(&info)?;
            let push = Pc::new().usize(numel).usize(dst_shape.len()).u32(binary_op_code(op));
            let res = dst.device.dispatch(
                &format!("broadcast_{dt}"),
                &[lhs.buffer, rhs.buffer, dst.buffer, scratch.buffer],
                &push,
                div_ceil(numel, WORKGROUP_SIZE),
            );
            // Only defer after the dispatch is recorded (see defer_free).
            dst.device.defer_free(scratch);
            res
        } else {
            dst.device.flush("broadcast-host")?;
            let n = dst_shape.len();
            let l = lhs.as_slice();
            let r = rhs.as_slice();
            let d = dst.as_mut_slice();
            for idx in 0..numel {
                let mut li = 0usize;
                let mut ri = 0usize;
                let mut rem = idx;
                for di in (0..n).rev() {
                    let coord = rem % dst_shape[di];
                    rem /= dst_shape[di];
                    li += coord * lhs_strides[di];
                    ri += coord * rhs_strides[di];
                }
                d[idx] = bin_apply(op, l[li], r[ri]);
            }
            Ok(())
        }
    }

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
        check_f32::<T>("conv1d")?;
        if groups == 1 {
            return conv1d_im2col(
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
            );
        }
        let total = batch * out_channels * out_length;
        let push = Pc::new()
            .usize(batch)
            .usize(in_channels)
            .usize(out_channels)
            .usize(length)
            .usize(out_length)
            .usize(kernel_size)
            .usize(stride)
            .usize(padding)
            .usize(dilation)
            .usize(groups);
        dst.device.dispatch(
            "conv1d_f32",
            &[dst.buffer, src.buffer, kernel.buffer],
            &push,
            div_ceil(total, WORKGROUP_SIZE),
        )
    }

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
        check_f32::<T>("conv_transpose1d")?;
        // col2im assumes groups == 1, no padding/output_padding and (per the
        // Backend trait, which has no dilation param here) dilation == 1.
        if groups == 1 && padding == 0 && output_padding == 0 {
            return conv_transpose1d_col2im(
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
            );
        }
        let total = batch * out_channels * out_length;
        let push = Pc::new()
            .usize(batch)
            .usize(in_channels)
            .usize(out_channels)
            .usize(length)
            .usize(out_length)
            .usize(kernel_size)
            .usize(stride)
            .usize(padding)
            .usize(groups);
        dst.device.dispatch(
            "conv_transpose1d_f32",
            &[dst.buffer, src.buffer, kernel.buffer],
            &push,
            div_ceil(total, WORKGROUP_SIZE),
        )
    }
}

/// `groups == 1` conv1d via im2col + GEMM + transpose (mirrors the CUDA
/// backend): unfold `src` into `col` [batch, out_length, in_channels*kernel_size],
/// multiply by the [out_channels, in_channels*kernel_size] weight matrix
/// (matmul_t), then transpose the [batch, out_length, out_channels] GEMM
/// result into dst's [batch, out_channels, out_length] layout. This trades
/// conv1d's naive per-output-element gather loop for the same tiled/vectorized
/// GEMM path used everywhere else, at the cost of the `col` scratch buffer.
#[allow(clippy::too_many_arguments)]
fn conv1d_im2col<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<()> {
    let dev = dst.device.clone();
    let k = in_channels * kernel_size;

    let col = unsafe { <Device as crate::Backend>::alloc_uninit::<T>(batch * out_length * k, &dev)? };
    let push = Pc::new()
        .usize(batch)
        .usize(in_channels)
        .usize(length)
        .usize(out_length)
        .usize(kernel_size)
        .usize(stride)
        .usize(padding)
        .usize(dilation);
    dev.dispatch(
        "im2col1d_f32",
        &[col.buffer, src.buffer],
        &push,
        div_ceil(batch * out_length * k, WORKGROUP_SIZE),
    )?;

    // result[b, l, oc] = sum_k col[b, l, k] * kernel[oc, k]
    let mut result = unsafe { <Device as crate::Backend>::alloc_uninit::<T>(batch * out_length * out_channels, &dev)? };
    <Device as crate::Backend>::gemm(
        &mut result,
        (&col, 0),
        (kernel, 0),
        out_length,
        out_channels,
        k,
        batch,
        out_length * k,
        0,
        (1, out_channels),
        (1, k),
        (k, 1),
    )?;

    // [batch, out_length, out_channels] -> dst's [batch, out_channels, out_length].
    <Device as crate::Backend>::transpose(dst, &result, 1, 2, &[batch, out_length, out_channels])
}

/// `groups == 1`, no padding/output_padding conv_transpose1d via transpose +
/// GEMM + col2im (mirrors the CUDA backend): transpose `src` to
/// [batch, length, in_channels], multiply by the [in_channels, out_channels*kernel_size]
/// weight matrix to get `col` [batch, length, out_channels*kernel_size], then
/// fold `col` into dst via col2im's gather (each output position sums the
/// compatible (input position, kernel offset) pairs directly, no atomics).
#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_col2im<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    kernel: &Storage<T>,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    out_length: usize,
    kernel_size: usize,
    stride: usize,
) -> Result<()> {
    let dev = dst.device.clone();
    let n = out_channels * kernel_size;

    let mut src_t = unsafe { <Device as crate::Backend>::alloc_uninit::<T>(batch * length * in_channels, &dev)? };
    <Device as crate::Backend>::transpose(&mut src_t, src, 1, 2, &[batch, in_channels, length])?;

    // col[b, l, j] = sum_c src_t[b, l, c] * kernel[c, j]
    let mut col = unsafe { <Device as crate::Backend>::alloc_uninit::<T>(batch * length * n, &dev)? };
    <Device as crate::Backend>::gemm(
        &mut col,
        (&src_t, 0),
        (kernel, 0),
        length,
        n,
        in_channels,
        batch,
        length * in_channels,
        0,
        (1, n),
        (1, in_channels),
        (1, n),
    )?;

    let push = Pc::new()
        .usize(batch)
        .usize(length)
        .usize(out_channels)
        .usize(out_length)
        .usize(kernel_size)
        .usize(stride);
    let total = batch * out_channels * out_length;
    dev.dispatch("col2im1d_f32", &[dst.buffer, col.buffer], &push, div_ceil(total, WORKGROUP_SIZE))
}

impl Device {
    fn reduce<T: WithDTypeF>(
        &self,
        dst: &mut Storage<T>,
        src: &Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
        op: u32,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&dst.device, "reduce")?;
        let num_outputs = outer_size * inner_size;
        if num_outputs == 0 {
            return Ok(());
        }
        let push = Pc::new().usize(num_outputs).usize(dim_size).usize(inner_size).u32(op);
        dst.device.dispatch(&format!("reduce_{dt}"), &[src.buffer, dst.buffer], &push, num_outputs as u32)
    }

    fn reduce_arg<T: WithDTypeF>(
        &self,
        dst: &mut Storage<i64>,
        src: &Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
        op: u32,
    ) -> Result<()> {
        let dt = dtype_suffix::<T>(&src.device, "reduce_arg")?;
        let num_outputs = outer_size * inner_size;
        if num_outputs == 0 {
            return Ok(());
        }
        let push = Pc::new().usize(num_outputs).usize(dim_size).usize(inner_size).u32(op);
        dst.device.dispatch(
            &format!("reduce_arg_{dt}"),
            &[src.buffer, dst.buffer],
            &push,
            num_outputs as u32,
        )
    }
}

/// Tiles below which a tiled kernel leaves too much of the GPU idle and a
/// row-block kernel, even re-reading one operand, is faster.
const TILED_MIN_TILES: usize = 48;
/// Largest value of `min(m, n)` the NT row-block kernel takes when the tiled
/// kernel is short of tiles. Past it both operands are long and the tiled
/// kernel's reuse wins even on a small grid.
const ROW_BLOCK_NT_MAX_SHORT_SIDE: usize = 128;
/// Largest `m` the NN split-k kernel takes; it re-reads rhs once per 32 rows
/// of lhs.
const ROW_BLOCK_NN_MAX_M: usize = 512;
/// Smallest `n` for the NN kernel: below it the 16-row tile measured faster.
const ROW_BLOCK_NN_MIN_N: usize = 128;
/// 16x16 tiles below which the chunked 16-row tile kernel is short of
/// workgroups and a row-block kernel does better.
const TILED16_MIN_TILES: usize = 24;
/// Largest k treated as "short": one staged chunk covers it.
const SHORT_K: usize = 128;
/// Outputs below which a short-k problem takes the 16-row tile rather than
/// the 32-row one.
const SHORT_K_TILED16_MAX_OUTPUTS: usize = 32 * 1024;

/// The 14-word push constant block every f32 GEMM kernel shares.
#[allow(clippy::too_many_arguments)]
fn gemm_push(
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    lhs_b_stride: usize,
    rhs_b_stride: usize,
    (dst_cs, dst_rs): (usize, usize),
    (lhs_cs, lhs_rs): (usize, usize),
    (rhs_cs, rhs_rs): (usize, usize),
    lhs_o: usize,
    rhs_o: usize,
) -> Pc {
    Pc::new()
        .usize(m)
        .usize(n)
        .usize(k)
        .usize(lhs_b)
        .usize(lhs_b_stride)
        .usize(rhs_b_stride)
        .usize(lhs_cs)
        .usize(lhs_rs)
        .usize(rhs_cs)
        .usize(rhs_rs)
        .usize(dst_rs)
        .usize(dst_cs)
        .usize(lhs_o)
        .usize(rhs_o)
}

/// `lhs @ w^T` for a contiguous `(.., k)` lhs and a contiguous `(n, k)` w,
/// straight through the 64x64 tiled kernel whatever the shape. The q8 path
/// uses it after dequantizing a long prefill's weight, where the shape
/// heuristics of [`row_block_gemm`] would pick a row-block kernel and
/// re-read the freshly written f32 weight several times.
pub(crate) fn tiled_matmul_t(
    lhs: &crate::Tensor<f32, Device>,
    w: &crate::Tensor<f32, Device>,
) -> Result<crate::Tensor<f32, Device>> {
    let (n, k) = w.shape().dims2()?;
    let dims = lhs.dims();
    let m = lhs.shape().elem_count() / k;
    let mut out_dims = dims[..dims.len() - 1].to_vec();
    out_dims.push(n);
    let dev = lhs.device();
    let out: crate::Tensor<f32, Device> =
        unsafe { crate::Tensor::alloc_uninit(crate::Shape::from(out_dims), dev)? };
    if m == 0 {
        return Ok(out);
    }
    let push = gemm_push(m, n, k, 1, m * k, 0, (1, n), (1, k), (k, 1), 0, 0);
    {
        let out_s = out.storage()?;
        let lhs_s = lhs.storage()?;
        let w_s = w.storage()?;
        let buffers = [out_s.buffer, lhs_s.buffer, w_s.buffer];
        let label =
            |kernel: &str| dev.profile_enabled.then(|| format!("{kernel} m{m} n{n} k{k} b1 l1,{k} r{k},1"));
        dispatch_tiled_gemm(dev, out_s.buffer, buffers, &push, label, m, n, k, 1, true)?;
    }
    Ok(out)
}

/// The tiled kernel, its row-tile height and its k split for a problem: 64
/// rows when that still fills the GPU, else 32, which doubles the grid for
/// the price of re-reading rhs once more; and when even that leaves the grid
/// short, k is split across workgroups (up to 8 ways, keeping at least 128 k
/// per split) with a reduce pass over the partial results.
fn tiled_variant(m: usize, n: usize, batch: usize, k: usize) -> (&'static str, u32, usize) {
    let tiles64 = m.div_ceil(64) * n.div_ceil(64) * batch;
    if tiles64 >= TILED_MIN_TILES {
        return ("gemm_tiled64", 64, 1);
    }
    // Half-height tiles are half the threads, so aim for twice the grid,
    // and split k towards it when the tile count alone falls short.
    let tiles32 = m.div_ceil(32) * n.div_ceil(64) * batch;
    let splits = (2 * TILED_MIN_TILES).div_ceil(tiles32).clamp(1, 8).min(k / 128).max(1);
    ("gemm_tiled32", 32, splits)
}

/// Records a tiled GEMM: the kernel picked by [`tiled_variant`] and, when it
/// splits k, the workspace and the reduce pass. `dst` must be the contiguous
/// `(batch, m, n)` result when splitting (`dst_strides == (1, n)`); callers
/// pass `splittable` accordingly.
#[allow(clippy::too_many_arguments)]
fn dispatch_tiled_gemm(
    dev: &Device,
    dst: vk::Buffer,
    buffers: [vk::Buffer; 3],
    push: &Pc,
    label: impl Fn(&str) -> Option<String>,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    splittable: bool,
) -> Result<()> {
    let (kernel, tm, splits) = tiled_variant(m, n, batch, k);
    let splits = if splittable { splits } else { 1 };
    let push_k = push.clone().usize(splits);
    let groups = (div_ceil(n, 64), div_ceil(m, tm), (batch * splits) as u32);
    if splits == 1 {
        return dev.dispatch_labeled(kernel, label(kernel), &buffers, &push_k, groups);
    }
    let total = batch * m * n;
    let ws: crate::Tensor<f32, Device> =
        unsafe { crate::Tensor::alloc_uninit((splits * total,), dev)? };
    let ws_s = ws.storage()?;
    let lbl = label(kernel).map(|l| format!("{l} ksplit{splits}"));
    dev.dispatch_labeled(kernel, lbl, &[ws_s.buffer, buffers[1], buffers[2]], &push_k, groups)?;
    let rpush = Pc::new().usize(total).usize(splits);
    dev.dispatch_nd("ksplit_reduce", &[dst, ws_s.buffer], &rpush, (div_ceil(total, WORKGROUP_SIZE), 1, 1))
}

/// The f32 GEMM fast paths, picked by shape and layout:
///
/// * a pure GEMV (m or n is 1) with k contiguous on both sides goes to the
///   subgroup row-block kernel, which streams the long operand coalesced;
/// * a short k with enough outputs goes to the thread-per-output kernel;
/// * a problem long in both m and n with enough 64x64 tiles goes to the tiled
///   kernel, which reads each operand once;
/// * otherwise a row-block kernel streams the longer operand against a block
///   of the shorter one (gemm_nt_sg.comp when k is contiguous on both sides,
///   gemm_nn_rows.comp for a row-major `matmul` rhs);
/// * a problem long in both dimensions that fits none of those still takes
///   the tiled kernel on a small grid.
///
/// Returns `false` when nothing applies and the caller falls back to the
/// generic 16x16 tiled kernel.
#[allow(clippy::too_many_arguments)]
fn row_block_gemm<T: WithDType>(
    dst: &Storage<T>,
    lhs: (&Storage<T>, usize),
    rhs: (&Storage<T>, usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    lhs_b_stride: usize,
    rhs_b_stride: usize,
    (dst_cs, dst_rs): (usize, usize),
    (lhs_cs, lhs_rs): (usize, usize),
    (rhs_cs, rhs_rs): (usize, usize),
) -> Result<bool> {
    let dev = &dst.device;
    let Some(cols_per_wg) = WORKGROUP_SIZE.checked_div(dev.subgroup_size()) else {
        return Ok(false);
    };
    if m == 0 || n == 0 || k == 0 || lhs_b == 0 {
        return Ok(false);
    }
    let label = |kernel: &str| {
        dev.profile_enabled.then(|| {
            format!("{kernel} m{m} n{n} k{k} b{lhs_b} l{lhs_cs},{lhs_rs} r{rhs_cs},{rhs_rs}")
        })
    };
    let push_all = || {
        gemm_push(
            m,
            n,
            k,
            lhs_b,
            lhs_b_stride,
            rhs_b_stride,
            (dst_cs, dst_rs),
            (lhs_cs, lhs_rs),
            (rhs_cs, rhs_rs),
            lhs.1,
            rhs.1,
        )
    };
    let buffers = [dst.buffer, lhs.0.buffer, rhs.0.buffer];
    // Batch strides only matter past the first batch; `matmul_` passes a
    // placeholder for a single one.
    let batch_aligned = lhs_b == 1 || (lhs_b_stride.is_multiple_of(4) && rhs_b_stride.is_multiple_of(4));
    // NT: k contiguous on both sides, every row start 4-aligned for vec4 loads.
    let nt = lhs_cs == 1
        && rhs_rs == 1
        && k.is_multiple_of(4)
        && batch_aligned
        && [lhs.1, lhs_rs, rhs.1, rhs_cs].iter().all(|v| v.is_multiple_of(4));
    // NN: n contiguous in rhs, k contiguous in lhs, lhs rows 4-aligned.
    let nn = lhs_cs == 1
        && rhs_cs == 1
        && k.is_multiple_of(4)
        && batch_aligned
        && [lhs.1, lhs_rs].iter().all(|v| v.is_multiple_of(4));

    // The NT row-block kernel streams whichever operand has more rows and
    // blocks the other 16 at a time; the kernel is symmetric in its two
    // operands up to the dst strides, so a swap is a matter of which buffer
    // sits in which binding.
    let dispatch_nt = |dev: &Device| -> Result<()> {
        let swap = m > n;
        let (bm, sn) = if swap { (n, m) } else { (m, n) };
        let (kernel, mr) = row_block_kernel("gemm_nt_sg", bm, 16);
        let (push, buffers) = if swap {
            let push = gemm_push(
                n,
                m,
                k,
                lhs_b,
                rhs_b_stride,
                lhs_b_stride,
                (dst_rs, dst_cs),
                (rhs_rs, rhs_cs),
                (lhs_rs, lhs_cs),
                rhs.1,
                lhs.1,
            );
            (push, [dst.buffer, rhs.0.buffer, lhs.0.buffer])
        } else {
            (push_all(), buffers)
        };
        let groups = (div_ceil(sn, cols_per_wg), div_ceil(bm, mr), lhs_b as u32);
        dev.dispatch_labeled(&kernel, label(&kernel), &buffers, &push, groups)
    };
    let splittable = dst_cs == 1 && dst_rs == n;
    let dispatch_tiled = |dev: &Device| -> Result<()> {
        dispatch_tiled_gemm(dev, dst.buffer, buffers, &push_all(), label, m, n, k, lhs_b, splittable)
    };

    let tiles16 = m.div_ceil(16) * n.div_ceil(16) * lhs_b;
    let dispatch_tiled16 = |dev: &Device| -> Result<()> {
        let groups = (div_ceil(n, 16), div_ceil(m, 16), lhs_b as u32);
        dev.dispatch_labeled("gemm_tiled16", label("gemm_tiled16"), &buffers, &push_all(), groups)
    };
    let dispatch_nn = |dev: &Device| -> Result<()> {
        let (kernel, mr) = row_block_kernel("gemm_nn_rows", m, 16);
        let kernel = if nn { kernel } else { format!("{kernel}s") };
        let groups = (div_ceil(n, 32), div_ceil(m, mr), lhs_b as u32);
        dev.dispatch_labeled(&kernel, label(&kernel), &buffers, &push_all(), groups)
    };
    let nn_any = lhs_cs == 1 && rhs_cs == 1;
    // The order below follows measurements over Phonon's shapes
    // (`examples/gemm_shapes_bench.rs`), which is also what the thresholds
    // encode; `XN_VULKAN_GEMM` overrides it for re-measuring.

    // A pure GEMV streams the long operand coalesced.
    if nt && m.min(n) == 1 {
        dispatch_nt(dev)?;
        return Ok(true);
    }
    // A short k: every kernel is bound by moving the operands, and the 16-row
    // tile wins on small outputs, the 32-row tile on larger ones.
    if k <= SHORT_K {
        if m * n * lhs_b <= SHORT_K_TILED16_MAX_OUTPUTS || m < 32 || n < 64 {
            dispatch_tiled16(dev)?;
        } else {
            dispatch_tiled(dev)?;
        }
        return Ok(true);
    }
    // A row-major `matmul` rhs: the split-k column kernel up to 512 rows,
    // when there are enough columns for it.
    if nn_any && n >= ROW_BLOCK_NN_MIN_N && m <= ROW_BLOCK_NN_MAX_M {
        dispatch_nn(dev)?;
        return Ok(true);
    }
    // Both operands k-contiguous with one side short: stream the long one.
    if nt && m.min(n) <= ROW_BLOCK_NT_MAX_SHORT_SIDE {
        dispatch_nt(dev)?;
        return Ok(true);
    }
    // Long in both dimensions: read each operand once, splitting k when the
    // grid is short.
    if m >= 32 && n >= 32 {
        dispatch_tiled(dev)?;
        return Ok(true);
    }
    if tiles16 >= TILED16_MIN_TILES {
        dispatch_tiled16(dev)?;
        return Ok(true);
    }
    Ok(false)
}
