#![cfg(feature = "webgpu")]
//! WebGPU backend tests. Most tests run an op on both the CPU backend and the
//! WebGPU backend from identical inputs and assert the outputs match, so the
//! CPU backend acts as the reference oracle. A few tests also check explicit
//! expected values.
//!
//! The WebGPU backend computes in f32; f16/bf16/i64/u8 storage is transferred
//! and cast on the host. These tests therefore focus on the f32 GPU path plus
//! the host-fallback data movement (storage roundtrips, casts, index_select on
//! non-f32 data).

use std::sync::OnceLock;
use xn::{CPU, Result, Tensor, webgpu_backend::Device as Wg};

// A single WebGPU device shared across the whole test binary. Unlike the native
// backends, opening many independent `wgpu` devices (one per test thread) and
// driving them concurrently is unreliable, so all tests share one device. The
// backend serializes GPU submission internally (per-device command stream
// behind a mutex), and each tensor owns a distinct buffer, so tests running on
// different threads interleave their batches without corrupting each other.
fn dev() -> Wg {
    static DEVICE: OnceLock<Wg> = OnceLock::new();
    DEVICE.get_or_init(|| Wg::new(0).expect("failed to init webgpu device")).clone()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x == y || (x.is_nan() && y.is_nan()) {
            continue;
        }
        let d = (x - y).abs();
        let scale = x.abs().max(y.abs());
        assert!(d <= tol + tol * scale, "mismatch at {i}: cpu={x} wgpu={y} (|d|={d})");
    }
}

fn iota(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect()
}

// -----------------------------------------------------------------------------
// Storage roundtrips (host upload / readback, any dtype)
// -----------------------------------------------------------------------------

#[test]
fn roundtrip_f32() -> Result<()> {
    let data = vec![1.0f32, -2.5, 3.0, 4.25, 5.0];
    let t: Tensor<f32, Wg> = Tensor::from_vec(data.clone(), vec![5], &dev())?;
    assert_eq!(t.to_vec()?, data);
    Ok(())
}

#[test]
fn roundtrip_non_f32() -> Result<()> {
    let d = dev();
    let d16: Vec<half::f16> = [1.0f32, 2.0, 3.0].into_iter().map(half::f16::from_f32).collect();
    let t: Tensor<half::f16, Wg> = Tensor::from_vec(d16.clone(), vec![3], &d)?;
    assert_eq!(t.to_vec()?, d16);
    let db: Vec<half::bf16> = [1.0f32, 2.0, 3.0].into_iter().map(half::bf16::from_f32).collect();
    let t: Tensor<half::bf16, Wg> = Tensor::from_vec(db.clone(), vec![3], &d)?;
    assert_eq!(t.to_vec()?, db);
    // i64 and u8 (odd length exercises the 4-byte copy/write padding).
    let di = vec![0i64, -1, 42, -12345, 1 << 40];
    let t: Tensor<i64, Wg> = Tensor::from_vec(di.clone(), vec![5], &d)?;
    assert_eq!(t.to_vec()?, di);
    let du = vec![0u8, 1, 127, 255, 3];
    let t: Tensor<u8, Wg> = Tensor::from_vec(du.clone(), vec![5], &d)?;
    assert_eq!(t.to_vec()?, du);
    Ok(())
}

#[test]
fn zeros_and_full() -> Result<()> {
    let z: Tensor<f32, Wg> = Tensor::zeros(vec![3, 4], &dev())?;
    assert!(z.to_vec()?.iter().all(|&x| x == 0.0));
    let f: Tensor<f32, Wg> = Tensor::full(42.0, vec![2, 3], &dev())?;
    assert!(f.to_vec()?.iter().all(|&x| x == 42.0));
    // zeros of a non-f32 dtype takes the host fill path.
    let zi: Tensor<i64, Wg> = Tensor::zeros(vec![7], &dev())?;
    assert_eq!(zi.to_vec()?, vec![0i64; 7]);
    Ok(())
}

#[test]
fn cast_pairs_cmp() -> Result<()> {
    // Host cast kernels (f32/f16/bf16 pairs + i64/u8) against the CPU backend.
    let d = dev();
    let data = iota(64);
    let wg: Tensor<f32, Wg> = Tensor::from_vec(data.clone(), vec![64], &d)?;
    let cpu: Tensor<f32, _> = Tensor::from_vec(data, vec![64], &CPU)?;
    assert_eq!(wg.to::<half::f16>()?.to_vec()?, cpu.to::<half::f16>()?.to_vec()?);
    assert_eq!(wg.to::<half::bf16>()?.to_vec()?, cpu.to::<half::bf16>()?.to_vec()?);
    assert_eq!(
        wg.to::<half::f16>()?.to::<f32>()?.to_vec()?,
        cpu.to::<half::f16>()?.to::<f32>()?.to_vec()?
    );
    // i64 -> f32 (rope position path) and f32 -> i64.
    let ids = vec![0i64, 1, -1, 42, -12345, 1 << 20];
    let iv: Tensor<i64, Wg> = Tensor::from_vec(ids.clone(), vec![6], &d)?;
    let ic: Tensor<i64, _> = Tensor::from_vec(ids, vec![6], &CPU)?;
    assert_eq!(iv.to::<f32>()?.to_vec()?, ic.to::<f32>()?.to_vec()?);
    Ok(())
}

#[test]
fn rand_uniform_bounds() -> Result<()> {
    let base: Tensor<f32, Wg> = Tensor::zeros(vec![4096], &dev())?;
    let r = base.rand_uniform_like(2.0, 3.0)?;
    let v = r.to_vec()?;
    assert!(v.iter().all(|&x| (2.0..3.0).contains(&x)), "values out of range");
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    assert!((mean - 2.5).abs() < 0.05, "mean {mean} too far from 2.5");
    Ok(())
}

// -----------------------------------------------------------------------------
// Elementwise binary / unary / scale, compared against CPU
// -----------------------------------------------------------------------------

macro_rules! cmp_binary {
    ($name:ident, $method:ident) => {
        #[test]
        fn $name() -> Result<()> {
            let a = iota(64);
            let b: Vec<f32> = iota(64).iter().map(|x| x + 0.5).collect();
            let av: Tensor<f32, Wg> = Tensor::from_vec(a.clone(), vec![8, 8], &dev())?;
            let bv: Tensor<f32, Wg> = Tensor::from_vec(b.clone(), vec![8, 8], &dev())?;
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
            let av: Tensor<f32, Wg> = Tensor::from_vec(a.clone(), vec![64], &dev())?;
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
cmp_unary!(unary_gelu, gelu_erf, 1e-5);
cmp_unary!(unary_tanh, tanh, 1e-6);
cmp_unary!(unary_sigmoid, sigmoid, 1e-6);

#[test]
fn scale_affine() -> Result<()> {
    let a = iota(32);
    let av: Tensor<f32, Wg> = Tensor::from_vec(a.clone(), vec![32], &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![32], &CPU)?;
    assert_close(&ac.scale(3.5)?.to_vec()?, &av.scale(3.5)?.to_vec()?, 1e-6);
    Ok(())
}

// -----------------------------------------------------------------------------
// Matmul (gemv m==1, tiled gemm, batched, transposed rhs)
// -----------------------------------------------------------------------------

#[test]
fn matmul_2d_explicit() -> Result<()> {
    let a: Tensor<f32, Wg> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &dev())?;
    let b: Tensor<f32, Wg> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], &dev())?;
    assert_close(&a.matmul(&b)?.to_vec()?, &[22.0, 28.0, 49.0, 64.0], 1e-5);
    Ok(())
}

fn cmp_matmul(m: usize, k: usize, n: usize, batch: usize) -> Result<()> {
    let a = iota(batch * m * k);
    let b = iota(batch * k * n);
    let (as_, bs): (Vec<usize>, Vec<usize>) =
        if batch == 1 { (vec![m, k], vec![k, n]) } else { (vec![batch, m, k], vec![batch, k, n]) };
    let av: Tensor<f32, Wg> = Tensor::from_vec(a.clone(), as_.clone(), &dev())?;
    let bv: Tensor<f32, Wg> = Tensor::from_vec(b.clone(), bs.clone(), &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a, as_, &CPU)?;
    let bc: Tensor<f32, _> = Tensor::from_vec(b, bs, &CPU)?;
    assert_close(&ac.matmul(&bc)?.to_vec()?, &av.matmul(&bv)?.to_vec()?, 1e-4);
    Ok(())
}

#[test]
fn matmul_shapes() -> Result<()> {
    cmp_matmul(4, 5, 6, 1)?;
    cmp_matmul(1, 32, 17, 1)?; // gemv
    cmp_matmul(1, 4096, 4096, 1)?; // large gemv
    cmp_matmul(3, 4, 5, 2)?;
    cmp_matmul(8, 8, 8, 3)?;
    cmp_matmul(33, 17, 19, 1)?; // non-tile-aligned gemm
    // Interior and tail tiles of the register-tiled gemm and the multi-column
    // gemv. `cmp_matmul` leaves rhs as [k, n], so rhs_rs = n and both kernels
    // take their strided branch; `matmul_t_shapes` covers rhs_rs == 1, which is
    // the layout the model actually uses.
    cmp_matmul(32, 64, 32, 1)?; // exactly one 32x32 tile, k a multiple of KSTEP
    cmp_matmul(64, 8, 64, 2)?; // several whole tiles, batched
    cmp_matmul(31, 9, 33, 1)?; // one short of a tile in every dimension
    cmp_matmul(1, 576, 1152, 1)?; // gemv, n a multiple of GEMV_TN
    cmp_matmul(1, 576, 1151, 1)?; // ...and with a gemv column tail
    cmp_matmul(128, 576, 1152, 1)?; // gemm, prefill-shaped
    Ok(())
}

/// As `cmp_matmul`, but with rhs stored [n, k] and multiplied with `matmul_t`,
/// giving rhs_rs = 1 and rhs_cs = k.
fn cmp_matmul_t(m: usize, k: usize, n: usize, batch: usize) -> Result<()> {
    let a = iota(batch * m * k);
    let b = iota(batch * n * k);
    let (as_, bs): (Vec<usize>, Vec<usize>) =
        if batch == 1 { (vec![m, k], vec![n, k]) } else { (vec![batch, m, k], vec![batch, n, k]) };
    let av: Tensor<f32, Wg> = Tensor::from_vec(a.clone(), as_.clone(), &dev())?;
    let bv: Tensor<f32, Wg> = Tensor::from_vec(b.clone(), bs.clone(), &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a, as_, &CPU)?;
    let bc: Tensor<f32, _> = Tensor::from_vec(b, bs, &CPU)?;
    assert_close(&ac.matmul_t(&bc)?.to_vec()?, &av.matmul_t(&bv)?.to_vec()?, 1e-4);
    Ok(())
}

/// The rhs_rs == 1 half of both kernels: the GEMV `vec4` path and the GEMM
/// contiguous staging branch. This is the layout of every linear layer in the
/// model, and the one `matmul_shapes` cannot reach.
#[test]
fn matmul_t_shapes() -> Result<()> {
    cmp_matmul_t(1, 576, 1152, 1)?; // decode gemv, vectorized, n a multiple of GEMV_TN
    cmp_matmul_t(1, 576, 1151, 1)?; // ...and with a gemv column tail
    cmp_matmul_t(1, 577, 64, 1)?; // rhs_cs not 4-aligned: gemv scalar fallback
    cmp_matmul_t(128, 576, 1152, 1)?; // prefill gemm, contiguous rhs staging
    cmp_matmul_t(32, 64, 32, 1)?; // exactly one 32x32 tile
    cmp_matmul_t(31, 9, 33, 1)?; // one short of a tile in every dimension
    cmp_matmul_t(3, 4, 5, 2)?; // batched
    Ok(())
}

/// The scalar mop-up at the end of the vectorized GEMV loop.
///
/// A contiguous [n, k] rhs has rhs_cs == k, so that remainder is unreachable
/// through `cmp_matmul_t`: the same k that is not a multiple of 4 also fails
/// the `rhs_cs & 3` alignment check and drops the kernel to the fully scalar
/// path instead. Narrowing k away from the row stride is what separates the
/// two, leaving rhs_cs 4-aligned while k is not.
#[test]
fn matmul_t_gemv_vec4_remainder() -> Result<()> {
    let (n, k_full, k) = (256usize, 576usize, 573usize);
    let a = iota(k_full);
    let b = iota(n * k_full);
    let av: Tensor<f32, Wg> = Tensor::from_vec(a.clone(), vec![1, k_full], &dev())?;
    let bv: Tensor<f32, Wg> = Tensor::from_vec(b.clone(), vec![n, k_full], &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![1, k_full], &CPU)?;
    let bc: Tensor<f32, _> = Tensor::from_vec(b, vec![n, k_full], &CPU)?;
    let (avn, bvn) = (av.narrow(1, 0..k)?, bv.narrow(1, 0..k)?);
    let (acn, bcn) = (ac.narrow(1, 0..k)?, bc.narrow(1, 0..k)?);
    assert_close(&acn.matmul_t(&bcn)?.to_vec()?, &avn.matmul_t(&bvn)?.to_vec()?, 1e-4);
    Ok(())
}

#[test]
fn matmul_t_and_transposed_view() -> Result<()> {
    // matmul_t exercises a non-contiguous rhs stride pattern.
    let a = iota(6 * 4);
    let b = iota(5 * 4);
    let av: Tensor<f32, Wg> = Tensor::from_vec(a.clone(), vec![6, 4], &dev())?;
    let bv: Tensor<f32, Wg> = Tensor::from_vec(b.clone(), vec![5, 4], &dev())?;
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
        let wg: Tensor<f32, Wg> = Tensor::from_vec(data.clone(), dims.clone(), &dev())?;
        let cpu: Tensor<f32, _> = Tensor::from_vec(data, dims.clone(), &CPU)?;
        let (d1, d2) = (0, dims.len() - 1);
        assert_close(
            &cpu.transpose(d1, d2)?.contiguous()?.to_vec()?,
            &wg.transpose(d1, d2)?.contiguous()?.to_vec()?,
            1e-6,
        );
    }
    Ok(())
}

#[test]
fn cat_and_narrow() -> Result<()> {
    let a = iota(2 * 3 * 4);
    let b: Vec<f32> = iota(2 * 2 * 4).iter().map(|x| x + 100.0).collect();
    let av: Tensor<f32, Wg> = Tensor::from_vec(a.clone(), vec![2, 3, 4], &dev())?;
    let bv: Tensor<f32, Wg> = Tensor::from_vec(b.clone(), vec![2, 2, 4], &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![2, 3, 4], &CPU)?;
    let bc: Tensor<f32, _> = Tensor::from_vec(b, vec![2, 2, 4], &CPU)?;
    let cv = Tensor::cat(&[&av, &bv], 1)?;
    let cc = Tensor::cat(&[&ac, &bc], 1)?;
    assert_close(&cc.to_vec()?, &cv.to_vec()?, 1e-6);
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
    let wg: Tensor<f32, Wg> = Tensor::from_vec(data.clone(), vec![6, 10], &dev())?;
    let cpu: Tensor<f32, _> = Tensor::from_vec(data, vec![6, 10], &CPU)?;
    assert_close(&cpu.softmax()?.to_vec()?, &wg.softmax()?.to_vec()?, 1e-6);
    Ok(())
}

#[test]
fn rms_norm_cmp() -> Result<()> {
    let data = iota(4 * 16);
    let w: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 * 0.03).collect();
    let wg: Tensor<f32, Wg> = Tensor::from_vec(data.clone(), vec![4, 16], &dev())?;
    let wv: Tensor<f32, Wg> = Tensor::from_vec(w.clone(), vec![16], &dev())?;
    let cpu: Tensor<f32, _> = Tensor::from_vec(data, vec![4, 16], &CPU)?;
    let wc: Tensor<f32, _> = Tensor::from_vec(w, vec![16], &CPU)?;
    assert_close(&cpu.rms_norm(&wc, 1e-5)?.to_vec()?, &wg.rms_norm(&wv, 1e-5)?.to_vec()?, 1e-5);
    Ok(())
}

#[test]
fn layer_norm_cmp() -> Result<()> {
    let data = iota(4 * 16);
    let w: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 * 0.03).collect();
    let bias: Vec<f32> = (0..16).map(|i| -0.2 + i as f32 * 0.01).collect();
    let wg: Tensor<f32, Wg> = Tensor::from_vec(data.clone(), vec![4, 16], &dev())?;
    let wv: Tensor<f32, Wg> = Tensor::from_vec(w.clone(), vec![16], &dev())?;
    let bv: Tensor<f32, Wg> = Tensor::from_vec(bias.clone(), vec![16], &dev())?;
    let cpu: Tensor<f32, _> = Tensor::from_vec(data, vec![4, 16], &CPU)?;
    let wc: Tensor<f32, _> = Tensor::from_vec(w, vec![16], &CPU)?;
    let bc: Tensor<f32, _> = Tensor::from_vec(bias, vec![16], &CPU)?;
    assert_close(
        &cpu.layer_norm(&wc, &bc, 1e-5)?.to_vec()?,
        &wg.layer_norm(&wv, &bv, 1e-5)?.to_vec()?,
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
        let wg: Tensor<f32, Wg> = Tensor::from_vec(d.clone(), dims.clone(), &dev())?;
        let cpu: Tensor<f32, _> = Tensor::from_vec(d, dims.clone(), &CPU)?;
        for dim in 0..dims.len() {
            assert_close(&cpu.max(dim)?.to_vec()?, &wg.max(dim)?.to_vec()?, 1e-6);
            assert_close(&cpu.min(dim)?.to_vec()?, &wg.min(dim)?.to_vec()?, 1e-6);
            assert_close(
                &cpu.sum_keepdim(vec![dim])?.to_vec()?,
                &wg.sum_keepdim(vec![dim])?.to_vec()?,
                1e-5,
            );
            assert_eq!(cpu.argmax(dim)?.to_vec()?, wg.argmax(dim)?.to_vec()?);
            assert_eq!(cpu.argmin(dim)?.to_vec()?, wg.argmin(dim)?.to_vec()?);
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Broadcast
// -----------------------------------------------------------------------------

#[test]
fn broadcast_ops() -> Result<()> {
    let a = iota(2 * 3);
    let row = vec![10.0f32, 20.0, 30.0];
    let av: Tensor<f32, Wg> = Tensor::from_vec(a.clone(), vec![2, 3], &dev())?;
    let rv: Tensor<f32, Wg> = Tensor::from_vec(row.clone(), vec![1, 3], &dev())?;
    let ac: Tensor<f32, _> = Tensor::from_vec(a, vec![2, 3], &CPU)?;
    let rc: Tensor<f32, _> = Tensor::from_vec(row, vec![1, 3], &CPU)?;
    assert_close(&ac.broadcast_add(&rc)?.to_vec()?, &av.broadcast_add(&rv)?.to_vec()?, 1e-6);
    assert_close(&ac.broadcast_mul(&rc)?.to_vec()?, &av.broadcast_mul(&rv)?.to_vec()?, 1e-6);
    assert_close(&ac.broadcast_sub(&rc)?.to_vec()?, &av.broadcast_sub(&rv)?.to_vec()?, 1e-6);

    let col = vec![1.0f32, 2.0];
    let cv: Tensor<f32, Wg> = Tensor::from_vec(col.clone(), vec![2, 1], &dev())?;
    let cc: Tensor<f32, _> = Tensor::from_vec(col, vec![2, 1], &CPU)?;
    assert_close(&ac.broadcast_add(&cc)?.to_vec()?, &av.broadcast_add(&cv)?.to_vec()?, 1e-6);
    Ok(())
}

#[test]
fn many_broadcasts_in_one_batch() -> Result<()> {
    // Long unflushed chain: hundreds of dispatches (each with its own `info`
    // scratch buffer) accumulate into one command batch before a single
    // readback flushes them, exercising the deferred scratch/buffer recycling.
    let d = dev();
    let n = 1024usize;
    let base: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let row: Vec<f32> = (0..32).map(|i| (i % 5) as f32).collect();
    let mut xv: Tensor<f32, Wg> = Tensor::from_vec(base.clone(), vec![32, 32], &d)?;
    let rv: Tensor<f32, Wg> = Tensor::from_vec(row.clone(), vec![1, 32], &d)?;
    let mut xc: Tensor<f32, _> = Tensor::from_vec(base, vec![32, 32], &CPU)?;
    let rc: Tensor<f32, _> = Tensor::from_vec(row, vec![1, 32], &CPU)?;
    for _ in 0..500 {
        xv = xv.broadcast_add(&rv)?;
        xc = xc.broadcast_add(&rc)?;
    }
    assert_close(&xc.to_vec()?, &xv.to_vec()?, 1e-3);
    Ok(())
}

// -----------------------------------------------------------------------------
// index_select / scatter (kv-cache-like)
// -----------------------------------------------------------------------------

#[test]
fn index_select_cmp() -> Result<()> {
    // f32 data on the GPU path; -1 indices select zeros.
    let data = iota(5 * 3);
    let ids = vec![0i64, 2, 4, 1, -1];
    let dv: Tensor<f32, Wg> = Tensor::from_vec(data.clone(), vec![5, 3], &dev())?;
    let iv: Tensor<i64, Wg> = Tensor::from_vec(ids.clone(), vec![5], &dev())?;
    let dc: Tensor<f32, _> = Tensor::from_vec(data, vec![5, 3], &CPU)?;
    let ic: Tensor<i64, _> = Tensor::from_vec(ids, vec![5], &CPU)?;
    assert_close(&dc.index_select(&ic, 0)?.to_vec()?, &dv.index_select(&iv, 0)?.to_vec()?, 1e-6);
    Ok(())
}

#[test]
fn index_select_f16_host_fallback() -> Result<()> {
    // f16 data goes through the host fallback (WebGPU compute is f32-only).
    let data = iota(5 * 3);
    let ids = vec![0i64, 2, 4, 1];
    let f16 = |v: &[f32]| v.iter().map(|&x| half::f16::from_f32(x)).collect::<Vec<_>>();
    let dv: Tensor<half::f16, Wg> = Tensor::from_vec(f16(&data), vec![5, 3], &dev())?;
    let iv: Tensor<i64, Wg> = Tensor::from_vec(ids.clone(), vec![4], &dev())?;
    let dc: Tensor<half::f16, _> = Tensor::from_vec(f16(&data), vec![5, 3], &CPU)?;
    let ic: Tensor<i64, _> = Tensor::from_vec(ids, vec![4], &CPU)?;
    assert_eq!(dc.index_select(&ic, 0)?.to_vec()?, dv.index_select(&iv, 0)?.to_vec()?);
    Ok(())
}

// -----------------------------------------------------------------------------
// RoPE
// -----------------------------------------------------------------------------

#[test]
fn rope_cmp() -> Result<()> {
    let (b, h, t, d, max_pos) = (1, 2, 3, 4, 10);
    let x = iota(b * h * t * d);
    let cos: Vec<f32> = (0..max_pos * d / 2).map(|i| (i as f32 * 0.3).cos()).collect();
    let sin: Vec<f32> = (0..max_pos * d / 2).map(|i| (i as f32 * 0.3).sin()).collect();
    let xv: Tensor<f32, Wg> = Tensor::from_vec(x.clone(), vec![b, h, t, d], &dev())?;
    let cv: Tensor<f32, Wg> = Tensor::from_vec(cos.clone(), vec![max_pos, d / 2], &dev())?;
    let sv: Tensor<f32, Wg> = Tensor::from_vec(sin.clone(), vec![max_pos, d / 2], &dev())?;
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
    dilation: usize,
    groups: usize,
) -> Result<()> {
    let src = iota(batch * in_c * len);
    let kern = iota(out_c * (in_c / groups) * ks);
    let sv: Tensor<f32, Wg> = Tensor::from_vec(src.clone(), vec![batch, in_c, len], &dev())?;
    let kv: Tensor<f32, Wg> =
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
    cmp_conv1d(1, 1, 1, 5, 3, 1, 0, 1, 1)?;
    cmp_conv1d(1, 1, 1, 5, 3, 1, 1, 1, 1)?;
    cmp_conv1d(1, 1, 1, 6, 3, 2, 0, 1, 1)?;
    cmp_conv1d(2, 3, 4, 7, 3, 1, 1, 1, 1)?;
    cmp_conv1d(1, 4, 4, 7, 3, 1, 1, 1, 2)?; // grouped -> direct kernel
    // Dilated, groups == 1 (im2col fast path).
    cmp_conv1d(2, 3, 4, 20, 3, 1, 2, 3, 1)?;
    cmp_conv1d(1, 2, 2, 25, 3, 1, 9, 9, 1)?;
    Ok(())
}

#[test]
fn conv_transpose1d_cmp() -> Result<()> {
    for (b, ic, oc, len, ks, stride) in [(1, 1, 1, 3, 3, 1), (1, 2, 3, 4, 3, 2), (2, 2, 2, 5, 2, 2)]
    {
        let src = iota(b * ic * len);
        let kern = iota(ic * oc * ks);
        let sv: Tensor<f32, Wg> = Tensor::from_vec(src.clone(), vec![b, ic, len], &dev())?;
        let kv: Tensor<f32, Wg> = Tensor::from_vec(kern.clone(), vec![ic, oc, ks], &dev())?;
        let sc: Tensor<f32, _> = Tensor::from_vec(src, vec![b, ic, len], &CPU)?;
        let kc: Tensor<f32, _> = Tensor::from_vec(kern, vec![ic, oc, ks], &CPU)?;
        let rc = sc.conv_transpose1d(&kc, None, stride, 0, 0, 1)?;
        let rv = sv.conv_transpose1d(&kv, None, stride, 0, 0, 1)?;
        assert_close(&rc.to_vec()?, &rv.to_vec()?, 1e-4);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
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
    use xn::webgpu_backend::quantization::Q8Tensor;
    let d = dev();

    // Spread of magnitudes across blocks so per-block scales actually differ.
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i % 71) as f32 - 35.0) * 0.013 * (1.0 + (i / k) as f32 * 0.1))
        .collect();
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 53) as f32 - 26.0) * 0.021).collect();

    let wq = Q8Tensor::from_f32(&d, &w, &Shape::from((n, k)))?;
    let xt: Tensor<f32, Wg> = Tensor::from_vec(x.clone(), (m, k), &d)?;
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
    // 1 < m <= 16 takes gemm_q8, including partial row and column tiles.
    for (m, k, n) in
        [(2, 64, 8), (3, 256, 5), (4, 512, 64), (6, 1024, 130), (8, 128, 33), (9, 64, 4)]
    {
        cmp_q8_matmul(m, k, n)?;
    }
    Ok(())
}

#[test]
fn q8_matmul_tiled() -> Result<()> {
    // m > 16 takes gemm_q8_tiled. `k` is always a multiple of the 32-value
    // q8_0 block, so only m and n can be partial; (17, 64, 3) is partial in
    // both at once, just past the gate.
    for (m, k, n) in [(17, 64, 3), (30, 288, 96), (33, 128, 33), (64, 512, 128), (120, 96, 65)] {
        cmp_q8_matmul(m, k, n)?;
    }
    Ok(())
}

#[test]
fn q8_matmul_batched_shape() -> Result<()> {
    use xn::Shape;
    use xn::webgpu_backend::quantization::Q8Tensor;
    // Leading dims are flattened into m and restored on the output.
    let (b, t, k, n) = (2usize, 3usize, 128usize, 16usize);
    let d = dev();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 29) as f32 - 14.0) * 0.02).collect();
    let x: Vec<f32> = (0..b * t * k).map(|i| ((i % 37) as f32 - 18.0) * 0.011).collect();
    let wq = Q8Tensor::from_f32(&d, &w, &Shape::from((n, k)))?;
    let xt: Tensor<f32, Wg> = Tensor::from_vec(x.clone(), (b, t, k), &d)?;
    let out = wq.matmul_t(&xt)?;
    assert_eq!(out.dims(), &[b, t, n]);

    let wd = q8_roundtrip(&w);
    let wt: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(wd, (n, k), &CPU)?;
    let xc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(x, (b, t, k), &CPU)?;
    assert_close(&xc.matmul_t(&wt)?.to_vec()?, &out.to_vec()?, 1e-4);
    Ok(())
}

#[test]
fn q8_linear_matches_dequantized() -> Result<()> {
    // The BackendQ entry point, bias included: quantizing a Linear and running
    // it must match running the dequantized weight through the f32 path.
    use xn::BackendQ;
    use xn::nn::Linear;
    use xn::webgpu_backend::quantization::Q8F32;
    let d = dev();
    let (k, n, m) = (256usize, 64usize, 2usize);
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 61) as f32 - 30.0) * 0.017).collect();
    let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    let wt: Tensor<f32, Wg> = Tensor::from_vec(w.clone(), (n, k), &d)?;
    let bt: Tensor<f32, Wg> = Tensor::from_vec(bias.clone(), (n,), &d)?;
    let lin = Linear::new(wt).with_bias(bt);
    let q = Q8F32::from_linear(lin)?;

    let x: Vec<f32> = (0..m * k).map(|i| ((i % 43) as f32 - 21.0) * 0.03).collect();
    let xt: Tensor<f32, Wg> = Tensor::from_vec(x.clone(), (m, k), &d)?;
    let got = q.forward(&xt)?.to_vec()?;

    let wd = q8_roundtrip(&w);
    let wc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(wd, (n, k), &CPU)?;
    let bc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(bias, (n,), &CPU)?;
    let xc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(x, (m, k), &CPU)?;
    let want = xc.matmul_t(&wc)?.broadcast_add(&bc)?.to_vec()?;
    assert_close(&want, &got, 1e-4);
    Ok(())
}

#[test]
fn q8_quantization_error_is_small() -> Result<()> {
    // End to end against the *unquantized* weight: confirms the whole path
    // (quantize, split, unpack, scale) lands within q8_0's error budget rather
    // than merely being self-consistent.
    use xn::Shape;
    use xn::webgpu_backend::quantization::Q8Tensor;
    let d = dev();
    let (k, n) = (1024usize, 128usize);
    let w: Vec<f32> = (0..n * k).map(|i| (i * 7919 % 1000) as f32 / 500.0 - 1.0).collect();
    let x: Vec<f32> = (0..k).map(|i| (i * 104729 % 1000) as f32 / 500.0 - 1.0).collect();

    let wq = Q8Tensor::from_f32(&d, &w, &Shape::from((n, k)))?;
    let xt: Tensor<f32, Wg> = Tensor::from_vec(x.clone(), (1, k), &d)?;
    let got = wq.matmul_t(&xt)?.to_vec()?;

    let wc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(w, (n, k), &CPU)?;
    let xc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(x, (1, k), &CPU)?;
    let want = xc.matmul_t(&wc)?.to_vec()?;

    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in want.iter().zip(got.iter()) {
        num += ((a - b) as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    let rel = (num / den).sqrt();
    assert!(rel < 0.01, "q8_0 relative error {rel} is above the 1% budget");
    Ok(())
}

// `from_q8_0` parses blocks as they sit in a GGUF: a 2-byte f16 scale then 32
// `i8` quants, per block. `from_f32` reaches the same layout by quantizing.
// Feeding both the same weights and comparing the matmul checks the parsing,
// which nothing else does: every other q8 test goes through `from_linear`.
#[test]
fn q8_from_gguf_blocks_matches_quantizing() -> Result<()> {
    use xn::quantized::{GgmlDType, QTensor};
    use xn::webgpu_backend::quantization::Q8Tensor;
    let d = dev();
    let (k, n, m) = (128usize, 32usize, 3usize);
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 71) as f32 - 35.0) * 0.013).collect();
    let shape = xn::Shape::from((n, k));

    // The file route: quantize to q8_0 blocks, then read them back as a GGUF would.
    let qt = QTensor::quantize_f32(&w, &shape, GgmlDType::Q8_0)?;
    let from_blocks = Q8Tensor::from_q8_0(&qt, &d)?;
    // The dense route, for comparison.
    let from_floats = Q8Tensor::from_f32(&d, &w, &shape)?;

    let x: Vec<f32> = (0..m * k).map(|i| ((i % 37) as f32 - 18.0) * 0.021).collect();
    let xt: Tensor<f32, Wg> = Tensor::from_vec(x, (m, k), &d)?;
    let blocks = from_blocks.matmul_t(&xt)?.to_vec()?;
    let floats = from_floats.matmul_t(&xt)?.to_vec()?;
    // Same blocks either way, so this is exact rather than merely close.
    assert_eq!(blocks, floats, "from_q8_0 and from_f32 disagree");
    Ok(())
}

// A weight whose shape does not match the layer is a checkpoint mismatch, and
// silently loading it would give garbage rather than an error.
#[test]
fn q8_from_gguf_rejects_a_wrong_shape() -> Result<()> {
    use xn::quantized::{GgmlDType, QTensor};
    use xn::webgpu_backend::quantization::Q8Tensor;
    let d = dev();
    // k = 100 is not a multiple of the 32-value block.
    let w = vec![0.5f32; 4 * 100];
    let qt = QTensor::quantize_f32(&w, &xn::Shape::from((4usize, 100usize)), GgmlDType::Q8_0);
    // Either the quantizer refuses it, or `from_q8_0` does. Both are fine; what
    // matters is that no misaligned weight reaches the kernels.
    if let Ok(qt) = qt {
        assert!(Q8Tensor::from_q8_0(&qt, &d).is_err(), "misaligned k should not load");
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// In-place ops, which bind one buffer at two bindings
// -----------------------------------------------------------------------------

// `bin_assign` dispatches `binary` with `[dst, src, dst]`, so the destination
// buffer is bound twice in one dispatch. WebGPU's usage-scope rules constrain
// what those two bindings may declare, and a validation failure there is a lost
// device rather than a wrong number -- so this exercises the dispatch rather
// than trusting the layout to be right.
#[test]
fn inplace_add_binds_dst_twice() -> Result<()> {
    let d = dev();
    let n = 512;
    let a = iota(n);
    let b: Vec<f32> = a.iter().map(|v| v * 2.0 - 0.5).collect();

    let cpu_dst = Tensor::from_vec(a.clone(), n, &CPU)?;
    cpu_dst.inplace_add(&Tensor::from_vec(b.clone(), n, &CPU)?)?;

    let gpu_dst = Tensor::from_vec(a, n, &d)?;
    gpu_dst.inplace_add(&Tensor::from_vec(b, n, &d)?)?;

    assert_close(&cpu_dst.to_vec()?, &gpu_dst.to_vec()?, 1e-6);
    Ok(())
}

#[test]
fn inplace_mul_binds_dst_twice() -> Result<()> {
    let d = dev();
    let n = 300;
    let a = iota(n);
    let b: Vec<f32> = a.iter().map(|v| 0.5 - v).collect();

    let cpu_dst = Tensor::from_vec(a.clone(), n, &CPU)?;
    cpu_dst.inplace_mul(&Tensor::from_vec(b.clone(), n, &CPU)?)?;

    let gpu_dst = Tensor::from_vec(a, n, &d)?;
    gpu_dst.inplace_mul(&Tensor::from_vec(b, n, &d)?)?;

    assert_close(&cpu_dst.to_vec()?, &gpu_dst.to_vec()?, 1e-6);
    Ok(())
}
