#![cfg(feature = "webgpu")]
//! `q8_0` correctness tests for the WebGPU backend.
//!
//! The reference is not the dense matmul -- quantization loses real precision
//! and comparing against dense would only measure that loss. Instead the weight
//! is quantized and dequantized on the host with the same [`BlockQ8_0`] code the
//! GPU layout is built from, and the reference matmul runs on those dequantized
//! values in f32 on the CPU. The GPU should then agree to f32 accumulation order
//! plus one rounding of the output, which keeps the tolerance tight enough to
//! catch a mis-packed word or a wrong scale index.
//!
//! Shapes cover the model's real `(m, n, k)` plus the MR = 4 row-blocking
//! boundary in the kernel, and both the bias and no-bias paths.

use half::f16;
use xn::quantized::GgmlType;
use xn::quantized::k_quants::{BlockQ8_0, QK8_0};
use xn::webgpu_backend::Device as Wg;
use xn::webgpu_backend::quantization::{Q80F16, Q80F32};
use xn::{BackendQ, CPU, ModuleT, Result, Tensor, WithDTypeF, nn};

fn wg() -> Wg {
    use std::sync::OnceLock;
    static DEVICE: OnceLock<Wg> = OnceLock::new();
    DEVICE.get_or_init(|| Wg::new(0).expect("init webgpu device")).clone()
}

fn rnd(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x1234_5678);
    if s == 0 {
        s = 0xDEAD_BEEF;
    }
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = (s.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32;
            lo + u * (hi - lo)
        })
        .collect()
}

/// Round-trip a `[n, k]` weight through `q8_0`, giving the values the GPU kernel
/// effectively multiplies by.
fn dequantized(w: &[f32], n: usize, k: usize) -> Vec<f32> {
    let bpr = k / QK8_0;
    let mut out = vec![0f32; n * k];
    let mut blocks = vec![BlockQ8_0::zeros(); bpr];
    for j in 0..n {
        BlockQ8_0::from_float(&w[j * k..(j + 1) * k], &mut blocks).unwrap();
        BlockQ8_0::to_float(&blocks, &mut out[j * k..(j + 1) * k]).unwrap();
    }
    out
}

fn cmp(label: &str, reference: &[f32], got: &[f32], rtol: f32, atol: f32) {
    assert_eq!(reference.len(), got.len(), "{label}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_i = usize::MAX;
    for (i, (&r, &g)) in reference.iter().zip(got).enumerate() {
        if r == g {
            continue;
        }
        let over = (r - g).abs() - (atol + rtol * r.abs().max(g.abs()));
        if over > worst {
            worst = over;
            worst_i = i;
        }
    }
    if worst_i != usize::MAX {
        panic!(
            "{label}: mismatch at {worst_i}: ref={} got={} (over tol by {worst})",
            reference[worst_i], got[worst_i]
        );
    }
}

/// Run the quantized layer on WebGPU in dtype `T`.
fn run_q8<T: WithDTypeF, Q: BackendQ<T = T, B = Wg>>(
    xs: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32>
where
    Q::LinearQ: ModuleT<T = T, B = Wg>,
{
    let dev = wg();
    let to = |d: &[f32], sh: Vec<usize>| -> Tensor<T, Wg> {
        Tensor::from_vec(d.iter().map(|&v| T::from_f32(v)).collect(), sh, &dev).unwrap()
    };
    let mut lin = nn::Linear::new(to(w, vec![n, k]));
    if let Some(b) = bias {
        lin = lin.with_bias(to(b, vec![n]));
    }
    let q = Q::from_linear(lin).unwrap();
    let x = to(xs, vec![m, k]);
    q.forward(&x).unwrap().to_vec().unwrap().into_iter().map(|v| v.to_f32()).collect()
}

/// CPU f32 reference over the dequantized weight.
fn reference(
    xs: &[f32],
    wdq: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let x: Tensor<f32, _> = Tensor::from_vec(xs.to_vec(), vec![m, k], &CPU).unwrap();
    let w: Tensor<f32, _> = Tensor::from_vec(wdq.to_vec(), vec![n, k], &CPU).unwrap();
    let mut lin = nn::Linear::new(w);
    if let Some(b) = bias {
        lin = lin.with_bias(Tensor::from_vec(b.to_vec(), vec![n], &CPU).unwrap());
    }
    lin.forward(&x).unwrap().to_vec().unwrap()
}

// (m, n, k); k must be a multiple of 32. m values straddle the kernel's MR = 4.
const CASES: &[(usize, usize, usize)] = &[
    (1, 2304, 768),  // flow_lm in_proj, decode
    (1, 768, 768),   // flow_lm out_proj
    (1, 3072, 768),  // flow_lm linear1
    (1, 768, 3072),  // flow_lm linear2
    (16, 2048, 512), // mimi transformer linear1, t = 16
    (16, 512, 2048), // mimi transformer linear2
    (2, 64, 32),     // minimum block size
    (3, 65, 96),     // MR boundary, n not a multiple of 64
    (4, 33, 64),
    (5, 128, 128),
    (17, 96, 160),
    (64, 32, 32),
];

#[test]
fn q8_matmul_f32() -> Result<()> {
    for (i, &(m, n, k)) in CASES.iter().enumerate() {
        let xs = rnd(i as u64 + 1, m * k, -1.0, 1.0);
        let w = rnd(i as u64 + 500, n * k, -1.0, 1.0);
        let wdq = dequantized(&w, n, k);
        for bias in [None, Some(rnd(i as u64 + 900, n, -0.5, 0.5))] {
            let b = bias.as_deref();
            let want = reference(&xs, &wdq, b, m, n, k);
            let got = run_q8::<f32, Q80F32>(&xs, &w, b, m, n, k);
            // Same arithmetic as the reference, differing only in summation
            // order over k, so this is tight on purpose.
            cmp(&format!("q8 f32 {m}x{n}x{k} bias={}", b.is_some()), &want, &got, 2e-5, 2e-5);
        }
    }
    Ok(())
}

/// The quantized layer must reject a `k` that is not a whole number of blocks
/// rather than silently reading past the packed rows.
#[test]
fn q8_rejects_unaligned_k() -> Result<()> {
    let dev = wg();
    let w: Tensor<f32, Wg> = Tensor::from_vec(rnd(7, 4 * 20, -1.0, 1.0), vec![4, 20], &dev)?;
    let err = Q80F32::from_linear(nn::Linear::new(w));
    assert!(err.is_err(), "k = 20 is not a multiple of {QK8_0} and must be rejected");
    Ok(())
}

/// The split layout should be ~1.125 bytes per weight: one byte of quant plus a
/// 4-byte scale per 32 values.
#[test]
fn q8_size_on_device() -> Result<()> {
    use xn::webgpu_backend::quantization::Q8Tensor;
    let dev = wg();
    let (n, k) = (128usize, 256usize);
    let w: Tensor<f32, Wg> = Tensor::from_vec(rnd(11, n * k, -1.0, 1.0), vec![n, k], &dev)?;
    let qt = Q8Tensor::quantize(&w)?;
    assert_eq!(qt.dims(), (n, k));
    // 32 quant bytes plus a 4-byte scale per block: 1.125 bytes per weight.
    assert_eq!(qt.size_in_bytes(), n * k / 32 * 36);
    assert_eq!(qt.size_in_bytes() * 8, n * k * 9);
    // 3.56x smaller than the same weight in f32.
    assert!(qt.size_in_bytes() < n * k * 4);
    Ok(())
}
