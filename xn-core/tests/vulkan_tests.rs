#![cfg(feature = "vulkan")]
//! Vulkan backend tests. Most tests run an op on both the CPU backend and the
//! Vulkan backend from identical inputs and assert the outputs match, so the
//! CPU backend acts as the reference oracle. A few tests also check explicit
//! expected values.

use std::cell::RefCell;
use xn::{CPU, Result, Tensor, vulkan_backend::Device as Vk};

// One device per test thread. Each test runs on its own thread, so this gives
// every test an isolated device (the backend batches commands on a single
// per-device stream and is meant to be driven from one thread at a time).
thread_local! {
    static DEVICE: RefCell<Option<Vk>> = const { RefCell::new(None) };
}

fn dev() -> Vk {
    DEVICE.with(|d| {
        d.borrow_mut()
            .get_or_insert_with(|| Vk::new(0).expect("failed to init vulkan device"))
            .clone()
    })
}

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x == y || (x.is_nan() && y.is_nan()) {
            continue;
        }
        let d = (x - y).abs();
        let scale = x.abs().max(y.abs());
        assert!(d <= tol + tol * scale, "mismatch at {i}: cpu={x} vk={y} (|d|={d})");
    }
}

fn iota(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect()
}

// -----------------------------------------------------------------------------
// Storage roundtrips
// -----------------------------------------------------------------------------

#[test]
fn roundtrip_f32() -> Result<()> {
    let data = vec![1.0f32, -2.5, 3.0, 4.25, 5.0];
    let t: Tensor<f32, Vk> = Tensor::from_vec(data.clone(), vec![5], &dev())?;
    assert_eq!(t.to_vec()?, data);
    Ok(())
}

#[test]
fn roundtrip_f16_bf16() -> Result<()> {
    let d16: Vec<half::f16> = [1.0f32, 2.0, 3.0].into_iter().map(half::f16::from_f32).collect();
    let t: Tensor<half::f16, Vk> = Tensor::from_vec(d16.clone(), vec![3], &dev())?;
    assert_eq!(t.to_vec()?, d16);
    let db: Vec<half::bf16> = [1.0f32, 2.0, 3.0].into_iter().map(half::bf16::from_f32).collect();
    let t: Tensor<half::bf16, Vk> = Tensor::from_vec(db.clone(), vec![3], &dev())?;
    assert_eq!(t.to_vec()?, db);
    Ok(())
}

#[test]
fn zeros_and_full() -> Result<()> {
    let z: Tensor<f32, Vk> = Tensor::zeros(vec![3, 4], &dev())?;
    assert!(z.to_vec()?.iter().all(|&x| x == 0.0));
    let f: Tensor<f32, Vk> = Tensor::full(42.0, vec![2, 3], &dev())?;
    assert!(f.to_vec()?.iter().all(|&x| x == 42.0));
    Ok(())
}

#[test]
fn dtype_roundtrip_conversions() -> Result<()> {
    // f32 -> f16 -> f32 on Vulkan should match doing it via CPU.
    let data = iota(20);
    let vk: Tensor<f32, Vk> = Tensor::from_vec(data.clone(), vec![20], &dev())?;
    let cpu: Tensor<f32, _> = Tensor::from_vec(data, vec![20], &CPU)?;
    let vk = vk.to::<half::f16>()?.to::<f32>()?;
    let cpu = cpu.to::<half::f16>()?.to::<f32>()?;
    assert_close(&cpu.to_vec()?, &vk.to_vec()?, 1e-6);
    Ok(())
}

#[test]
fn cast_pairs_cmp() -> Result<()> {
    // Exercises the GPU cast kernels (f32/f16/bf16 pairs + i64->f32) against
    // the CPU backend doing the same conversions.
    let d = dev();
    let data = iota(64);
    let vk: Tensor<f32, Vk> = Tensor::from_vec(data.clone(), vec![64], &d)?;
    let cpu: Tensor<f32, _> = Tensor::from_vec(data, vec![64], &CPU)?;
    if d.supports_f16() {
        let (a, b) = (vk.to::<half::f16>()?, cpu.to::<half::f16>()?);
        assert_eq!(a.to_vec()?, b.to_vec()?);
        assert_eq!(a.to::<f32>()?.to_vec()?, b.to::<f32>()?.to_vec()?);
        if d.supports_bf16() {
            assert_eq!(a.to::<half::bf16>()?.to_vec()?, b.to::<half::bf16>()?.to_vec()?);
        }
    }
    if d.supports_bf16() {
        let (a, b) = (vk.to::<half::bf16>()?, cpu.to::<half::bf16>()?);
        assert_eq!(a.to_vec()?, b.to_vec()?);
        assert_eq!(a.to::<f32>()?.to_vec()?, b.to::<f32>()?.to_vec()?);
        if d.supports_f16() {
            assert_eq!(a.to::<half::f16>()?.to_vec()?, b.to::<half::f16>()?.to_vec()?);
        }
    }
    // i64 -> f32 (rope position path).
    let ids = vec![0i64, 1, -1, 42, -12345, 1 << 20];
    let iv: Tensor<i64, Vk> = Tensor::from_vec(ids.clone(), vec![6], &d)?;
    let ic: Tensor<i64, _> = Tensor::from_vec(ids, vec![6], &CPU)?;
    assert_eq!(iv.to::<f32>()?.to_vec()?, ic.to::<f32>()?.to_vec()?);
    Ok(())
}

#[test]
fn fill_16bit_and_rand() -> Result<()> {
    let d = dev();
    // GPU fill for 16-bit dtypes (zeros/full go through the fill kernel).
    if d.supports_f16() {
        let f: Tensor<half::f16, Vk> = Tensor::full(half::f16::from_f32(1.5), vec![33], &d)?;
        assert!(f.to_vec()?.iter().all(|&x| x == half::f16::from_f32(1.5)));
    }
    if d.supports_bf16() {
        let f: Tensor<half::bf16, Vk> = Tensor::full(half::bf16::from_f32(-2.25), vec![33], &d)?;
        assert!(f.to_vec()?.iter().all(|&x| x == half::bf16::from_f32(-2.25)));
    }
    // rand_uniform: bounds + rough mean (flushless scratch-copy path).
    let base: Tensor<f32, Vk> = Tensor::zeros(vec![4096], &d)?;
    let r = base.rand_uniform_like(2.0, 3.0)?;
    let v = r.to_vec()?;
    assert!(v.iter().all(|&x| (2.0..3.0).contains(&x)), "values out of range");
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    assert!((mean - 2.5).abs() < 0.05, "mean {mean} too far from 2.5");
    Ok(())
}

// -----------------------------------------------------------------------------
// Elementwise binary / unary, compared against CPU
// -----------------------------------------------------------------------------

macro_rules! cmp_binary {
    ($name:ident, $method:ident) => {
        #[test]
        fn $name() -> Result<()> {
            let a = iota(64);
            let b: Vec<f32> = iota(64).iter().map(|x| x + 0.5).collect();
            let av: Tensor<f32, Vk> = Tensor::from_vec(a.clone(), vec![8, 8], &dev())?;
            let bv: Tensor<f32, Vk> = Tensor::from_vec(b.clone(), vec![8, 8], &dev())?;
            let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![8, 8], &CPU)?;
            let bc: Tensor<f32, _> = Tensor::from_vec(b, vec![8, 8], &CPU)?;
            assert_close(&ac.$method(&bc)?.to_vec()?, &av.$method(&bv)?.to_vec()?, 1e-6);
            Ok(())
        }
    };
}
cmp_binary!(binary_add, add);
cmp_binary!(binary_sub, sub);
cmp_binary!(binary_mul, mul);
cmp_binary!(binary_div, div);
cmp_binary!(binary_maximum, maximum);
cmp_binary!(binary_minimum, minimum);

macro_rules! cmp_unary {
    ($name:ident, $method:ident, $tol:expr) => {
        #[test]
        fn $name() -> Result<()> {
            let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 + 0.1).collect();
            let av: Tensor<f32, Vk> = Tensor::from_vec(a.clone(), vec![64], &dev())?;
            let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![64], &CPU)?;
            assert_close(&ac.$method()?.to_vec()?, &av.$method()?.to_vec()?, $tol);
            Ok(())
        }
    };
}
cmp_unary!(unary_relu, relu, 1e-6);
cmp_unary!(unary_silu, silu, 1e-6);
cmp_unary!(unary_sqr, sqr, 1e-6);
cmp_unary!(unary_sqrt, sqrt, 1e-6);
cmp_unary!(unary_exp, exp, 1e-5);
cmp_unary!(unary_abs, abs, 1e-6);
cmp_unary!(unary_neg, neg, 1e-6);

#[test]
fn scale_affine() -> Result<()> {
    let a = iota(32);
    let av: Tensor<f32, Vk> = Tensor::from_vec(a.clone(), vec![32], &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![32], &CPU)?;
    assert_close(&ac.scale(3.5)?.to_vec()?, &av.scale(3.5)?.to_vec()?, 1e-6);
    Ok(())
}

// -----------------------------------------------------------------------------
// Matmul
// -----------------------------------------------------------------------------

#[test]
fn matmul_2d_explicit() -> Result<()> {
    let a: Tensor<f32, Vk> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &dev())?;
    let b: Tensor<f32, Vk> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], &dev())?;
    assert_close(&a.matmul(&b)?.to_vec()?, &[22.0, 28.0, 49.0, 64.0], 1e-5);
    Ok(())
}

fn cmp_matmul(m: usize, k: usize, n: usize, batch: usize) -> Result<()> {
    let a = iota(batch * m * k);
    let b = iota(batch * k * n);
    let (as_, bs): (Vec<usize>, Vec<usize>) =
        if batch == 1 { (vec![m, k], vec![k, n]) } else { (vec![batch, m, k], vec![batch, k, n]) };
    let av: Tensor<f32, Vk> = Tensor::from_vec(a.clone(), as_.clone(), &dev())?;
    let bv: Tensor<f32, Vk> = Tensor::from_vec(b.clone(), bs.clone(), &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a, as_, &CPU)?;
    let bc: Tensor<f32, _> = Tensor::from_vec(b, bs, &CPU)?;
    assert_close(&ac.matmul(&bc)?.to_vec()?, &av.matmul(&bv)?.to_vec()?, 1e-4);
    Ok(())
}

#[test]
fn matmul_shapes() -> Result<()> {
    cmp_matmul(4, 5, 6, 1)?;
    cmp_matmul(1, 32, 17, 1)?;
    cmp_matmul(3, 4, 5, 2)?;
    cmp_matmul(8, 8, 8, 3)?;
    Ok(())
}

#[test]
fn matmul_t_and_transposed_view() -> Result<()> {
    // matmul_t exercises a non-contiguous rhs stride pattern.
    let a = iota(6 * 4);
    let b = iota(5 * 4);
    let av: Tensor<f32, Vk> = Tensor::from_vec(a.clone(), vec![6, 4], &dev())?;
    let bv: Tensor<f32, Vk> = Tensor::from_vec(b.clone(), vec![5, 4], &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![6, 4], &CPU)?;
    let bc: Tensor<f32, _> = Tensor::from_vec(b, vec![5, 4], &CPU)?;
    assert_close(&ac.matmul_t(&bc)?.to_vec()?, &av.matmul_t(&bv)?.to_vec()?, 1e-4);
    Ok(())
}

// -----------------------------------------------------------------------------
// Layout: transpose, cat/narrow (copy2d / copy_strided)
// -----------------------------------------------------------------------------

#[test]
fn transpose_various() -> Result<()> {
    for dims in [vec![3usize, 4], vec![2, 3, 4], vec![2, 3, 4, 5]] {
        let n: usize = dims.iter().product();
        let data = iota(n);
        let vk: Tensor<f32, Vk> = Tensor::from_vec(data.clone(), dims.clone(), &dev())?;
        let cpu: Tensor<f32, _> = Tensor::from_vec(data, dims.clone(), &CPU)?;
        let (d1, d2) = (0, dims.len() - 1);
        assert_close(
            &cpu.transpose(d1, d2)?.contiguous()?.to_vec()?,
            &vk.transpose(d1, d2)?.contiguous()?.to_vec()?,
            1e-6,
        );
    }
    Ok(())
}

#[test]
fn cat_and_narrow() -> Result<()> {
    let a = iota(2 * 3 * 4);
    let b: Vec<f32> = iota(2 * 2 * 4).iter().map(|x| x + 100.0).collect();
    let av: Tensor<f32, Vk> = Tensor::from_vec(a.clone(), vec![2, 3, 4], &dev())?;
    let bv: Tensor<f32, Vk> = Tensor::from_vec(b.clone(), vec![2, 2, 4], &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![2, 3, 4], &CPU)?;
    let bc: Tensor<f32, _> = Tensor::from_vec(b, vec![2, 2, 4], &CPU)?;
    let cv = Tensor::cat(&[&av, &bv], 1)?;
    let cc = Tensor::cat(&[&ac, &bc], 1)?;
    assert_close(&cc.to_vec()?, &cv.to_vec()?, 1e-6);
    // narrow (strided copy back to contiguous)
    let nv = cv.narrow(1, 1..4)?.contiguous()?;
    let nc = cc.narrow(1, 1..4)?.contiguous()?;
    assert_close(&nc.to_vec()?, &nv.to_vec()?, 1e-6);
    Ok(())
}

// -----------------------------------------------------------------------------
// Softmax / norms
// -----------------------------------------------------------------------------

#[test]
fn softmax_cmp() -> Result<()> {
    let data = iota(6 * 10);
    let vk: Tensor<f32, Vk> = Tensor::from_vec(data.clone(), vec![6, 10], &dev())?;
    let cpu: Tensor<f32, _> = Tensor::from_vec(data, vec![6, 10], &CPU)?;
    assert_close(&cpu.softmax()?.to_vec()?, &vk.softmax()?.to_vec()?, 1e-6);
    Ok(())
}

#[test]
fn rms_norm_cmp() -> Result<()> {
    let data = iota(4 * 16);
    let w: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 * 0.03).collect();
    let vk: Tensor<f32, Vk> = Tensor::from_vec(data.clone(), vec![4, 16], &dev())?;
    let wv: Tensor<f32, Vk> = Tensor::from_vec(w.clone(), vec![16], &dev())?;
    let cpu: Tensor<f32, _> = Tensor::from_vec(data, vec![4, 16], &CPU)?;
    let wc: Tensor<f32, _> = Tensor::from_vec(w, vec![16], &CPU)?;
    assert_close(&cpu.rms_norm(&wc, 1e-5)?.to_vec()?, &vk.rms_norm(&wv, 1e-5)?.to_vec()?, 1e-5);
    Ok(())
}

#[test]
fn layer_norm_cmp() -> Result<()> {
    let data = iota(4 * 16);
    let w: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 * 0.03).collect();
    let bias: Vec<f32> = (0..16).map(|i| -0.2 + i as f32 * 0.01).collect();
    let vk: Tensor<f32, Vk> = Tensor::from_vec(data.clone(), vec![4, 16], &dev())?;
    let wv: Tensor<f32, Vk> = Tensor::from_vec(w.clone(), vec![16], &dev())?;
    let bv: Tensor<f32, Vk> = Tensor::from_vec(bias.clone(), vec![16], &dev())?;
    let cpu: Tensor<f32, _> = Tensor::from_vec(data, vec![4, 16], &CPU)?;
    let wc: Tensor<f32, _> = Tensor::from_vec(w, vec![16], &CPU)?;
    let bc: Tensor<f32, _> = Tensor::from_vec(bias, vec![16], &CPU)?;
    assert_close(
        &cpu.layer_norm(&wc, &bc, 1e-5)?.to_vec()?,
        &vk.layer_norm(&wv, &bv, 1e-5)?.to_vec()?,
        1e-5,
    );
    Ok(())
}

// -----------------------------------------------------------------------------
// Reductions
// -----------------------------------------------------------------------------

#[test]
fn reductions_cmp() -> Result<()> {
    let data: Vec<f32> = (0..2 * 3 * 4).map(|i| ((i * 7 + 3) % 11) as f32 - 5.0).collect();
    for dims in [vec![2usize, 3, 4], vec![24]] {
        let n: usize = dims.iter().product();
        let d = data[..n].to_vec();
        let vk: Tensor<f32, Vk> = Tensor::from_vec(d.clone(), dims.clone(), &dev())?;
        let cpu: Tensor<f32, _> = Tensor::from_vec(d, dims.clone(), &CPU)?;
        for dim in 0..dims.len() {
            assert_close(&cpu.max(dim)?.to_vec()?, &vk.max(dim)?.to_vec()?, 1e-6);
            assert_close(&cpu.min(dim)?.to_vec()?, &vk.min(dim)?.to_vec()?, 1e-6);
            assert_close(
                &cpu.sum_keepdim(vec![dim])?.to_vec()?,
                &vk.sum_keepdim(vec![dim])?.to_vec()?,
                1e-5,
            );
            assert_eq!(cpu.argmax(dim)?.to_vec()?, vk.argmax(dim)?.to_vec()?);
            assert_eq!(cpu.argmin(dim)?.to_vec()?, vk.argmin(dim)?.to_vec()?);
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Broadcast
// -----------------------------------------------------------------------------

#[test]
fn broadcast_ops() -> Result<()> {
    // 2D + row/col, and 3D broadcasting, all vs CPU.
    let a = iota(2 * 3);
    let row = vec![10.0f32, 20.0, 30.0];
    let av: Tensor<f32, Vk> = Tensor::from_vec(a.clone(), vec![2, 3], &dev())?;
    let rv: Tensor<f32, Vk> = Tensor::from_vec(row.clone(), vec![1, 3], &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![2, 3], &CPU)?;
    let rc: Tensor<f32, _> = Tensor::from_vec(row, vec![1, 3], &CPU)?;
    assert_close(&ac.broadcast_add(&rc)?.to_vec()?, &av.broadcast_add(&rv)?.to_vec()?, 1e-6);
    assert_close(&ac.broadcast_mul(&rc)?.to_vec()?, &av.broadcast_mul(&rv)?.to_vec()?, 1e-6);
    assert_close(&ac.broadcast_sub(&rc)?.to_vec()?, &av.broadcast_sub(&rv)?.to_vec()?, 1e-6);

    let col = vec![1.0f32, 2.0];
    let cv: Tensor<f32, Vk> = Tensor::from_vec(col.clone(), vec![2, 1], &dev())?;
    let cc: Tensor<f32, _> = Tensor::from_vec(col, vec![2, 1], &CPU)?;
    assert_close(&ac.broadcast_add(&cc)?.to_vec()?, &av.broadcast_add(&cv)?.to_vec()?, 1e-6);
    Ok(())
}

#[test]
fn forced_midbatch_flush_with_scratch() -> Result<()> {
    // Regression test for the scratch-buffer recycle race: >4096 dispatches
    // without a host readback force a mid-batch flush inside dispatch_nd. The
    // op straddling that flush uses a scratch `info` buffer (broadcast dims and
    // strides); if the scratch is deferred to the pool before its dispatch is
    // recorded, the flush recycles it and the *next* op's scratch overwrites it
    // while the recorded dispatch still references it. Shapes alternate so
    // consecutive scratch contents differ and the corruption is observable.
    // The tensors are 1024 elements (4 KiB size class) so the 24-byte scratch
    // buffers are alone in their 256 B size class: the first same-class alloc
    // after the forced flush is the next op's scratch_u32, whose host memcpy
    // lands in the still-referenced buffer.
    let d = dev();
    let n = 1024usize;
    let base: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let row32: Vec<f32> = (0..32).map(|i| (i % 5) as f32).collect();
    let row64: Vec<f32> = (0..64).map(|i| (i % 3) as f32).collect();

    let mut xv: Tensor<f32, Vk> = Tensor::from_vec(base.clone(), vec![32, 32], &d)?;
    let r32v: Tensor<f32, Vk> = Tensor::from_vec(row32.clone(), vec![1, 32], &d)?;
    let r64v: Tensor<f32, Vk> = Tensor::from_vec(row64.clone(), vec![1, 64], &d)?;
    let mut xc: Tensor<f32, _> = Tensor::from_vec(base, vec![32, 32], &CPU)?;
    let r32c: Tensor<f32, _> = Tensor::from_vec(row32, vec![1, 32], &CPU)?;
    let r64c: Tensor<f32, _> = Tensor::from_vec(row64, vec![1, 64], &CPU)?;

    for i in 0..4200 {
        if i % 2 == 0 {
            xv = xv.broadcast_add(&r32v)?;
            xc = xc.broadcast_add(&r32c)?;
        } else {
            xv = xv.reshape((16, 64))?.broadcast_add(&r64v)?.reshape((32, 32))?;
            xc = xc.reshape((16, 64))?.broadcast_add(&r64c)?.reshape((32, 32))?;
        }
    }
    assert_close(&xc.to_vec()?, &xv.to_vec()?, 1e-4);
    Ok(())
}

// -----------------------------------------------------------------------------
// index_select / scatter (kv-cache-like)
// -----------------------------------------------------------------------------

#[test]
fn index_select_cmp() -> Result<()> {
    let data = iota(5 * 3);
    let ids = vec![0i64, 2, 4, 1, -1];
    let dv: Tensor<f32, Vk> = Tensor::from_vec(data.clone(), vec![5, 3], &dev())?;
    let iv: Tensor<i64, Vk> = Tensor::from_vec(ids.clone(), vec![5], &dev())?;
    let dc: Tensor<f32, _> = Tensor::from_vec(data, vec![5, 3], &CPU)?;
    let ic: Tensor<i64, _> = Tensor::from_vec(ids, vec![5], &CPU)?;
    assert_close(&dc.index_select(&ic, 0)?.to_vec()?, &dv.index_select(&iv, 0)?.to_vec()?, 1e-6);
    Ok(())
}

// -----------------------------------------------------------------------------
// RoPE
// -----------------------------------------------------------------------------

#[test]
fn rope_cmp() -> Result<()> {
    // x: [b=1, h=2, t=3, d=4]; cos/sin: [max_pos, d/2]
    let b = 1;
    let h = 2;
    let t = 3;
    let d = 4;
    let x = iota(b * h * t * d);
    let max_pos = 10;
    let cos: Vec<f32> = (0..max_pos * d / 2).map(|i| (i as f32 * 0.3).cos()).collect();
    let sin: Vec<f32> = (0..max_pos * d / 2).map(|i| (i as f32 * 0.3).sin()).collect();
    let xv: Tensor<f32, Vk> = Tensor::from_vec(x.clone(), vec![b, h, t, d], &dev())?;
    let cv: Tensor<f32, Vk> = Tensor::from_vec(cos.clone(), vec![max_pos, d / 2], &dev())?;
    let sv: Tensor<f32, Vk> = Tensor::from_vec(sin.clone(), vec![max_pos, d / 2], &dev())?;
    let xc: Tensor<f32, _> = Tensor::from_vec(x, vec![b, h, t, d], &CPU)?;
    let cc: Tensor<f32, _> = Tensor::from_vec(cos, vec![max_pos, d / 2], &CPU)?;
    let sc: Tensor<f32, _> = Tensor::from_vec(sin, vec![max_pos, d / 2], &CPU)?;
    for pos in [0usize, 2, 5] {
        assert_close(&xc.rope(&cc, &sc, pos)?.to_vec()?, &xv.rope(&cv, &sv, pos)?.to_vec()?, 1e-5);
        assert_close(
            &xc.rope_i(&cc, &sc, pos)?.to_vec()?,
            &xv.rope_i(&cv, &sv, pos)?.to_vec()?,
            1e-5,
        );
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Convolutions
// -----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn cmp_conv1d(
    batch: usize,
    in_c: usize,
    out_c: usize,
    len: usize,
    ks: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<()> {
    cmp_conv1d_dilated(batch, in_c, out_c, len, ks, stride, padding, 1, groups)
}

#[allow(clippy::too_many_arguments)]
fn cmp_conv1d_dilated(
    batch: usize,
    in_c: usize,
    out_c: usize,
    len: usize,
    ks: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<()> {
    let src = iota(batch * in_c * len);
    let kern = iota(out_c * (in_c / groups) * ks);
    let sv: Tensor<f32, Vk> = Tensor::from_vec(src.clone(), vec![batch, in_c, len], &dev())?;
    let kv: Tensor<f32, Vk> =
        Tensor::from_vec(kern.clone(), vec![out_c, in_c / groups, ks], &dev())?;
    let sc: Tensor<f32, _> = Tensor::from_vec(src, vec![batch, in_c, len], &CPU)?;
    let kc: Tensor<f32, _> = Tensor::from_vec(kern, vec![out_c, in_c / groups, ks], &CPU)?;
    let rc = sc.conv1d(&kc, None, stride, padding, dilation, groups)?;
    let rv = sv.conv1d(&kv, None, stride, padding, dilation, groups)?;
    assert_close(&rc.to_vec()?, &rv.to_vec()?, 1e-4);
    Ok(())
}

#[test]
fn conv1d_cmp() -> Result<()> {
    cmp_conv1d(1, 1, 1, 5, 3, 1, 0, 1)?;
    cmp_conv1d(1, 1, 1, 5, 3, 1, 1, 1)?;
    cmp_conv1d(1, 1, 1, 6, 3, 2, 0, 1)?;
    cmp_conv1d(2, 3, 4, 7, 3, 1, 1, 1)?;
    cmp_conv1d(1, 4, 4, 7, 3, 1, 1, 2)?;
    // Dilated, groups == 1 (the im2col fast path): matches Mimi's SEANet
    // residual blocks, which use dilation > 1 with groups == 1.
    cmp_conv1d_dilated(2, 3, 4, 20, 3, 1, 2, 3, 1)?;
    cmp_conv1d_dilated(1, 2, 2, 25, 3, 1, 9, 9, 1)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// f16 compute path (validated against the f32 CPU reference, loose tolerance
// since inputs/outputs are f16-rounded; shaders accumulate in f32).
// -----------------------------------------------------------------------------

fn to_f16_vec(v: &[f32]) -> Vec<half::f16> {
    v.iter().map(|&x| half::f16::from_f32(x)).collect()
}

fn f16_to_f32(v: &[half::f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

#[test]
fn f16_matmul_gemm_and_gemv() -> Result<()> {
    if !dev().supports_f16() {
        eprintln!("skipping: device lacks f16 support");
        return Ok(());
    }
    // GEMV (m=1) and a general GEMM (m>1).
    for (m, k, n) in [(1usize, 512usize, 384usize), (16, 128, 64)] {
        let a = iota(m * k);
        let b = iota(k * n);
        let av: Tensor<half::f16, Vk> = Tensor::from_vec(to_f16_vec(&a), vec![m, k], &dev())?;
        let bv: Tensor<half::f16, Vk> = Tensor::from_vec(to_f16_vec(&b), vec![k, n], &dev())?;
        let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![m, k], &CPU)?;
        let bc: Tensor<f32, _> = Tensor::from_vec(b, vec![k, n], &CPU)?;
        let got = f16_to_f32(&av.matmul(&bv)?.to_vec()?);
        assert_close(&ac.matmul(&bc)?.to_vec()?, &got, 3e-2);
    }
    Ok(())
}

#[test]
fn f16_elementwise_and_norm() -> Result<()> {
    if !dev().supports_f16() {
        return Ok(());
    }
    let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 + 0.1).collect();
    let av: Tensor<half::f16, Vk> = Tensor::from_vec(to_f16_vec(&a), vec![64], &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a.clone(), vec![64], &CPU)?;
    assert_close(&ac.silu()?.to_vec()?, &f16_to_f32(&av.silu()?.to_vec()?), 3e-2);

    let b: Vec<f32> = a.iter().map(|x| x + 0.5).collect();
    let bv: Tensor<half::f16, Vk> = Tensor::from_vec(to_f16_vec(&b), vec![64], &dev())?;
    let bc: Tensor<f32, _> = Tensor::from_vec(b, vec![64], &CPU)?;
    assert_close(&ac.add(&bc)?.to_vec()?, &f16_to_f32(&av.add(&bv)?.to_vec()?), 3e-2);

    // rms_norm and softmax over rows.
    let data = iota(4 * 16);
    let w: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 * 0.03).collect();
    let dv: Tensor<half::f16, Vk> = Tensor::from_vec(to_f16_vec(&data), vec![4, 16], &dev())?;
    let wv: Tensor<half::f16, Vk> = Tensor::from_vec(to_f16_vec(&w), vec![16], &dev())?;
    let dc: Tensor<f32, _> = Tensor::from_vec(data.clone(), vec![4, 16], &CPU)?;
    let wc: Tensor<f32, _> = Tensor::from_vec(w, vec![16], &CPU)?;
    assert_close(
        &dc.rms_norm(&wc, 1e-5)?.to_vec()?,
        &f16_to_f32(&dv.rms_norm(&wv, 1e-5)?.to_vec()?),
        3e-2,
    );
    assert_close(&dc.softmax()?.to_vec()?, &f16_to_f32(&dv.softmax()?.to_vec()?), 3e-2);
    Ok(())
}

#[test]
fn f16_index_select_and_rope() -> Result<()> {
    if !dev().supports_f16() {
        return Ok(());
    }
    // index_select (embedding-like) with f16 data.
    let data = iota(5 * 3);
    let ids = vec![0i64, 2, 4, 1];
    let dv: Tensor<half::f16, Vk> = Tensor::from_vec(to_f16_vec(&data), vec![5, 3], &dev())?;
    let iv: Tensor<i64, Vk> = Tensor::from_vec(ids.clone(), vec![4], &dev())?;
    let dc: Tensor<f32, _> = Tensor::from_vec(data, vec![5, 3], &CPU)?;
    let ic: Tensor<i64, _> = Tensor::from_vec(ids, vec![4], &CPU)?;
    assert_close(
        &dc.index_select(&ic, 0)?.to_vec()?,
        &f16_to_f32(&dv.index_select(&iv, 0)?.to_vec()?),
        1e-3,
    );

    // rope
    let (b, h, t, d, mp) = (1, 2, 3, 4, 10);
    let x = iota(b * h * t * d);
    let cos: Vec<f32> = (0..mp * d / 2).map(|i| (i as f32 * 0.3).cos()).collect();
    let sin: Vec<f32> = (0..mp * d / 2).map(|i| (i as f32 * 0.3).sin()).collect();
    let xv: Tensor<half::f16, Vk> = Tensor::from_vec(to_f16_vec(&x), vec![b, h, t, d], &dev())?;
    let cv: Tensor<half::f16, Vk> = Tensor::from_vec(to_f16_vec(&cos), vec![mp, d / 2], &dev())?;
    let sv: Tensor<half::f16, Vk> = Tensor::from_vec(to_f16_vec(&sin), vec![mp, d / 2], &dev())?;
    let xc: Tensor<f32, _> = Tensor::from_vec(x, vec![b, h, t, d], &CPU)?;
    let cc: Tensor<f32, _> = Tensor::from_vec(cos, vec![mp, d / 2], &CPU)?;
    let sc: Tensor<f32, _> = Tensor::from_vec(sin, vec![mp, d / 2], &CPU)?;
    assert_close(
        &xc.rope(&cc, &sc, 2)?.to_vec()?,
        &f16_to_f32(&xv.rope(&cv, &sv, 2)?.to_vec()?),
        3e-2,
    );
    Ok(())
}

// -----------------------------------------------------------------------------
// bf16 compute path (uint16_t storage, f32 compute; validated against the f32
// CPU reference with a loose tolerance — bf16 has 8 mantissa bits).
// -----------------------------------------------------------------------------

fn to_bf16_vec(v: &[f32]) -> Vec<half::bf16> {
    v.iter().map(|&x| half::bf16::from_f32(x)).collect()
}

fn bf16_to_f32(v: &[half::bf16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

#[test]
fn bf16_gpu_roundtrip_exact() -> Result<()> {
    if !dev().supports_bf16() {
        eprintln!("skipping: device lacks bf16 support");
        return Ok(());
    }
    // a + 0 forces a GPU LOAD/STORE round trip through the bf16 shader path;
    // bf16 -> f32 is exact and f32 -> bf16 RNE round-trips exactly, so the
    // output must be bit-identical to the input (incl. negatives and zeros).
    let data: Vec<half::bf16> =
        to_bf16_vec(&[0.0, -0.0, 1.0, -1.5, std::f32::consts::PI, -65504.0, 1e-8, 3.0e38]);
    let n = data.len();
    let a: Tensor<half::bf16, Vk> = Tensor::from_vec(data.clone(), vec![n], &dev())?;
    let z: Tensor<half::bf16, Vk> = Tensor::zeros(vec![n], &dev())?;
    assert_eq!(a.add(&z)?.to_vec()?, data);
    Ok(())
}

#[test]
fn bf16_matmul_gemm_and_gemv() -> Result<()> {
    if !dev().supports_bf16() {
        return Ok(());
    }
    for (m, k, n) in [(1usize, 512usize, 384usize), (16, 128, 64)] {
        let a = iota(m * k);
        let b = iota(k * n);
        let av: Tensor<half::bf16, Vk> = Tensor::from_vec(to_bf16_vec(&a), vec![m, k], &dev())?;
        let bv: Tensor<half::bf16, Vk> = Tensor::from_vec(to_bf16_vec(&b), vec![k, n], &dev())?;
        let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![m, k], &CPU)?;
        let bc: Tensor<f32, _> = Tensor::from_vec(b, vec![k, n], &CPU)?;
        let got = bf16_to_f32(&av.matmul(&bv)?.to_vec()?);
        assert_close(&ac.matmul(&bc)?.to_vec()?, &got, 5e-2);
    }
    Ok(())
}

#[test]
fn bf16_elementwise_and_norm() -> Result<()> {
    if !dev().supports_bf16() {
        return Ok(());
    }
    let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 + 0.1).collect();
    let av: Tensor<half::bf16, Vk> = Tensor::from_vec(to_bf16_vec(&a), vec![64], &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a.clone(), vec![64], &CPU)?;
    assert_close(&ac.silu()?.to_vec()?, &bf16_to_f32(&av.silu()?.to_vec()?), 5e-2);

    let b: Vec<f32> = a.iter().map(|x| x + 0.5).collect();
    let bv: Tensor<half::bf16, Vk> = Tensor::from_vec(to_bf16_vec(&b), vec![64], &dev())?;
    let bc: Tensor<f32, _> = Tensor::from_vec(b, vec![64], &CPU)?;
    assert_close(&ac.add(&bc)?.to_vec()?, &bf16_to_f32(&av.add(&bv)?.to_vec()?), 5e-2);

    let data = iota(4 * 16);
    let w: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 * 0.03).collect();
    let dv: Tensor<half::bf16, Vk> = Tensor::from_vec(to_bf16_vec(&data), vec![4, 16], &dev())?;
    let wv: Tensor<half::bf16, Vk> = Tensor::from_vec(to_bf16_vec(&w), vec![16], &dev())?;
    let dc: Tensor<f32, _> = Tensor::from_vec(data.clone(), vec![4, 16], &CPU)?;
    let wc: Tensor<f32, _> = Tensor::from_vec(w, vec![16], &CPU)?;
    assert_close(
        &dc.rms_norm(&wc, 1e-5)?.to_vec()?,
        &bf16_to_f32(&dv.rms_norm(&wv, 1e-5)?.to_vec()?),
        5e-2,
    );
    assert_close(&dc.softmax()?.to_vec()?, &bf16_to_f32(&dv.softmax()?.to_vec()?), 5e-2);
    Ok(())
}

#[test]
fn bf16_index_select_and_rope() -> Result<()> {
    if !dev().supports_bf16() {
        return Ok(());
    }
    let data = iota(5 * 3);
    let ids = vec![0i64, 2, 4, 1, -1];
    let dv: Tensor<half::bf16, Vk> = Tensor::from_vec(to_bf16_vec(&data), vec![5, 3], &dev())?;
    let iv: Tensor<i64, Vk> = Tensor::from_vec(ids.clone(), vec![5], &dev())?;
    let dc: Tensor<f32, _> = Tensor::from_vec(data, vec![5, 3], &CPU)?;
    let ic: Tensor<i64, _> = Tensor::from_vec(ids, vec![5], &CPU)?;
    assert_close(
        &dc.index_select(&ic, 0)?.to_vec()?,
        &bf16_to_f32(&dv.index_select(&iv, 0)?.to_vec()?),
        5e-3,
    );

    let (b, h, t, d, mp) = (1, 2, 3, 4, 10);
    let x = iota(b * h * t * d);
    let cos: Vec<f32> = (0..mp * d / 2).map(|i| (i as f32 * 0.3).cos()).collect();
    let sin: Vec<f32> = (0..mp * d / 2).map(|i| (i as f32 * 0.3).sin()).collect();
    let xv: Tensor<half::bf16, Vk> = Tensor::from_vec(to_bf16_vec(&x), vec![b, h, t, d], &dev())?;
    let cv: Tensor<half::bf16, Vk> = Tensor::from_vec(to_bf16_vec(&cos), vec![mp, d / 2], &dev())?;
    let sv: Tensor<half::bf16, Vk> = Tensor::from_vec(to_bf16_vec(&sin), vec![mp, d / 2], &dev())?;
    let xc: Tensor<f32, _> = Tensor::from_vec(x, vec![b, h, t, d], &CPU)?;
    let cc: Tensor<f32, _> = Tensor::from_vec(cos, vec![mp, d / 2], &CPU)?;
    let sc: Tensor<f32, _> = Tensor::from_vec(sin, vec![mp, d / 2], &CPU)?;
    assert_close(
        &xc.rope(&cc, &sc, 2)?.to_vec()?,
        &bf16_to_f32(&xv.rope(&cv, &sv, 2)?.to_vec()?),
        5e-2,
    );
    Ok(())
}

#[test]
fn conv_transpose1d_cmp() -> Result<()> {
    for (b, ic, oc, len, ks, stride) in [(1, 1, 1, 3, 3, 1), (1, 2, 3, 4, 3, 2), (2, 2, 2, 5, 2, 2)]
    {
        let src = iota(b * ic * len);
        let kern = iota(ic * oc * ks);
        let sv: Tensor<f32, Vk> = Tensor::from_vec(src.clone(), vec![b, ic, len], &dev())?;
        let kv: Tensor<f32, Vk> = Tensor::from_vec(kern.clone(), vec![ic, oc, ks], &dev())?;
        let sc: Tensor<f32, _> = Tensor::from_vec(src, vec![b, ic, len], &CPU)?;
        let kc: Tensor<f32, _> = Tensor::from_vec(kern, vec![ic, oc, ks], &CPU)?;
        let rc = sc.conv_transpose1d(&kc, None, stride, 0, 0, 1)?;
        let rv = sv.conv_transpose1d(&kv, None, stride, 0, 0, 1)?;
        assert_close(&rc.to_vec()?, &rv.to_vec()?, 1e-4);
    }
    Ok(())
}

// q8_0 weights on the GPU.
//
// The oracle here cannot be the f32 matmul directly: quantizing to 8 bits is
// lossy, so these compare against the *dequantized* weight rather than the
// original. That isolates the kernel from the quantizer -- a wrong unpack,
// scale index or reduction shows up immediately, while the ~0.4% the
// quantizer itself costs does not.

/// Round-trip a weight through q8_0 on the host, mirroring what the GPU
/// upload does, so tests can multiply by exactly the weight the kernel sees.
fn q8_roundtrip(w: &[f32]) -> Vec<f32> {
    use xn::quantized::GgmlType;
    use xn::quantized::k_quants::BlockQ8_0;
    let mut blocks = vec![BlockQ8_0::zeros(); w.len() / 32];
    BlockQ8_0::from_float(w, &mut blocks).unwrap();
    let mut out = vec![0f32; w.len()];
    BlockQ8_0::to_float(&blocks, &mut out).unwrap();
    out
}

fn cmp_q8_matmul(m: usize, k: usize, n: usize) -> Result<()> {
    use xn::Shape;
    use xn::vulkan_backend::quantization::Q8Tensor;
    let d = dev();

    // Spread of magnitudes across blocks so per-block scales actually differ.
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i % 71) as f32 - 35.0) * 0.013 * (1.0 + (i / k) as f32 * 0.1))
        .collect();
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 53) as f32 - 26.0) * 0.021).collect();

    let wq = Q8Tensor::from_f32(&d, &w, &Shape::from((n, k)))?;
    let xt: Tensor<f32, Vk> = Tensor::from_vec(x.clone(), (m, k), &d)?;
    let got = wq.matmul_t(&xt)?.to_vec()?;

    // Reference: the dequantized weight, multiplied on the CPU backend.
    let wd = q8_roundtrip(&w);
    let wt: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(wd, (n, k), &CPU)?;
    let xc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(x, (m, k), &CPU)?;
    let want = xc.matmul_t(&wt)?.to_vec()?;

    assert_eq!(got.len(), want.len(), "m={m} k={k} n={n}");
    // f32 accumulation in a different order than the CPU reference.
    assert_close(&want, &got, 1e-4);
    Ok(())
}

#[test]
fn q8_matmul_decode() -> Result<()> {
    // m == 1 takes the gemv_q8 path.
    for (k, n) in [(32, 4), (64, 1), (256, 7), (1024, 256), (1536, 512)] {
        cmp_q8_matmul(1, k, n)?;
    }
    Ok(())
}

#[test]
fn q8_matmul_rows() -> Result<()> {
    // 2..=16 takes gemm_q8, including partial row and column tiles.
    for (m, k, n) in [(2, 64, 8), (3, 256, 5), (4, 512, 64), (6, 1024, 130), (16, 128, 33)] {
        cmp_q8_matmul(m, k, n)?;
    }
    Ok(())
}

#[test]
fn q8_matmul_row_blocks() -> Result<()> {
    // m > 16 walks m in blocks of 32 (gemm_q8_sg_r32 with a 2D grid),
    // including partial last blocks and the 125-row voice prompt.
    for (m, k, n) in [(17, 64, 4), (30, 288, 96), (33, 128, 33), (64, 512, 128), (125, 1024, 96)] {
        cmp_q8_matmul(m, k, n)?;
    }
    Ok(())
}

#[test]
fn q8_matmul_batched_shape() -> Result<()> {
    use xn::Shape;
    use xn::vulkan_backend::quantization::Q8Tensor;
    // Leading dims are flattened into m and restored on the output.
    let (b, t, k, n) = (2usize, 3usize, 128usize, 16usize);
    let d = dev();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 29) as f32 - 14.0) * 0.02).collect();
    let x: Vec<f32> = (0..b * t * k).map(|i| ((i % 37) as f32 - 18.0) * 0.011).collect();
    let wq = Q8Tensor::from_f32(&d, &w, &Shape::from((n, k)))?;
    let xt: Tensor<f32, Vk> = Tensor::from_vec(x.clone(), (b, t, k), &d)?;
    let out = wq.matmul_t(&xt)?;
    assert_eq!(out.dims(), &[b, t, n]);

    let wd = q8_roundtrip(&w);
    let wt: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(wd, (n, k), &CPU)?;
    let xc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(x, (b, t, k), &CPU)?;
    assert_close(&xc.matmul_t(&wt)?.to_vec()?, &out.to_vec()?, 1e-4);
    Ok(())
}

#[test]
fn q8_gguf_blocks_match_in_process_quantization() -> Result<()> {
    // Uploading a QTensor's q8_0 blocks must give the same weight as
    // quantizing the f32 data in process.
    use xn::Shape;
    use xn::quantized::{GgmlDType, QTensor};
    use xn::vulkan_backend::quantization::Q8Tensor;
    let (k, n, m) = (256usize, 24usize, 3usize);
    let d = dev();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 67) as f32 - 33.0) * 0.019).collect();
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 41) as f32 - 20.0) * 0.027).collect();
    let qt = QTensor::quantize_f32(&w, &Shape::from((n, k)), GgmlDType::Q8_0)?;
    let from_blocks = Q8Tensor::from_qtensor(&d, &qt)?;
    let from_f32 = Q8Tensor::from_f32(&d, &w, &Shape::from((n, k)))?;
    let xt: Tensor<f32, Vk> = Tensor::from_vec(x, (m, k), &d)?;
    assert_eq!(from_blocks.matmul_t(&xt)?.to_vec()?, from_f32.matmul_t(&xt)?.to_vec()?);
    Ok(())
}

#[test]
fn q8_linear_matches_dequantized() -> Result<()> {
    // The BackendQ entry point, bias included: quantizing a Linear and running
    // it must match running the dequantized weight through the f32 path.
    use xn::BackendQ;
    use xn::nn::Linear;
    use xn::vulkan_backend::quantization::Q8F32;
    let d = dev();
    let (k, n, m) = (256usize, 64usize, 2usize);
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 61) as f32 - 30.0) * 0.017).collect();
    let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    let wt: Tensor<f32, Vk> = Tensor::from_vec(w.clone(), (n, k), &d)?;
    let bt: Tensor<f32, Vk> = Tensor::from_vec(bias.clone(), (n,), &d)?;
    let lin = Linear::new(wt).with_bias(bt);
    let q = Q8F32::from_linear(lin)?;

    let x: Vec<f32> = (0..m * k).map(|i| ((i % 43) as f32 - 21.0) * 0.03).collect();
    let xt: Tensor<f32, Vk> = Tensor::from_vec(x.clone(), (m, k), &d)?;
    let got = q.forward(&xt)?.to_vec()?;

    let wd = q8_roundtrip(&w);
    let wc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(wd, (n, k), &CPU)?;
    let bc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(bias, (n,), &CPU)?;
    let xc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(x, (m, k), &CPU)?;
    let want = xc.matmul_t(&wc)?.broadcast_add(&bc)?.to_vec()?;
    assert_close(&want, &got, 1e-4);
    Ok(())
}
