//! Checks the XNNPACK matmul path against the same product computed on the gemm crate, on
//! the shapes and both weight layouts that the pocket-tts workload actually issues.
#![cfg(feature = "xnnpack")]

use xn::{CpuDevice, Tensor};

/// Deterministic, non-degenerate data: a constant or a low-rank pattern would agree even if
/// the two paths disagreed about the layout.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / 8388608.0) - 1.0
        })
        .collect()
}

/// `matmul_t`: weight stored `[n, k]`, which is what `nn::Linear` and conv1d's im2col pass.
fn check_matmul_t(m: usize, n: usize, k: usize) {
    let lhs: Tensor<f32, CpuDevice> =
        Tensor::from_vec(fill(m * k, 0x1234 + m as u64), (m, k), &CpuDevice).unwrap();
    let w: Tensor<f32, CpuDevice> =
        Tensor::from_vec(fill(n * k, 0x9876 + n as u64), (n, k), &CpuDevice).unwrap();

    let got = lhs.matmul_t(&w).unwrap().to_vec().unwrap();

    // Independent reference, accumulated in f64 so neither path is being compared to itself.
    let a = lhs.to_vec().unwrap();
    let b = w.to_vec().unwrap();
    let mut worst = 0f64;
    let mut scale = 0f64;
    for i in 0..m {
        for j in 0..n {
            let want: f64 =
                (0..k).map(|p| a[i * k + p] as f64 * b[j * k + p] as f64).sum();
            scale = scale.max(want.abs());
            worst = worst.max((got[i * n + j] as f64 - want).abs());
        }
    }
    let rel = worst / scale.max(1e-30);
    assert!(
        rel < 1e-5,
        "matmul_t m={m} n={n} k={k}: relative error {rel:.3e} against an f64 reference"
    );
}

/// `conv_transpose1d`'s gemm: weight stored `[k, n]`, the layout XNNPACK has to transpose.
fn check_kn_layout(m: usize, n: usize, k: usize) {
    let lhs: Tensor<f32, CpuDevice> =
        Tensor::from_vec(fill(m * k, 0x2468 + m as u64), (m, k), &CpuDevice).unwrap();
    let w: Tensor<f32, CpuDevice> =
        Tensor::from_vec(fill(k * n, 0x1357 + n as u64), (k, n), &CpuDevice).unwrap();

    let got = lhs.matmul(&w).unwrap().to_vec().unwrap();

    let a = lhs.to_vec().unwrap();
    let b = w.to_vec().unwrap();
    let mut worst = 0f64;
    let mut scale = 0f64;
    for i in 0..m {
        for j in 0..n {
            let want: f64 = (0..k).map(|p| a[i * k + p] as f64 * b[p * n + j] as f64).sum();
            scale = scale.max(want.abs());
            worst = worst.max((got[i * n + j] as f64 - want).abs());
        }
    }
    let rel = worst / scale.max(1e-30);
    assert!(
        rel < 1e-5,
        "matmul (k,n) m={m} n={n} k={k}: relative error {rel:.3e} against an f64 reference"
    );
}

/// Every distinct gemm shape a pocket-tts decode step issues, as reported by
/// `XN_XNNPACK_STATS=1`.
const SHAPES: &[(usize, usize, usize)] = &[
    // Mimi decoder transformer, 16 timesteps per frame.
    (16, 1536, 512),
    (16, 512, 512),
    (16, 2048, 512),
    (16, 512, 2048),
    // Mimi seanet.
    (16, 512, 3584),
    (16, 3072, 512),
    (96, 128, 768),
    (96, 256, 128),
    (96, 1280, 256),
    (480, 64, 384),
    (480, 128, 64),
    (480, 512, 128),
    (1920, 32, 192),
    (1920, 64, 32),
    (1920, 1, 192),
    // Flow LM head, one row at a time.
    (1, 1536, 512),
    (1, 512, 512),
    (1, 1024, 512),
    (1, 512, 256),
    (1, 32, 512),
    (1, 1, 768),
    (32, 768, 16),
];

#[test]
fn xnnpack_matmul_t_matches_reference() {
    for &(m, n, k) in SHAPES {
        check_matmul_t(m, n, k);
    }
}

#[test]
fn xnnpack_kn_layout_matches_reference() {
    for &(m, n, k) in SHAPES {
        check_kn_layout(m, n, k);
    }
}

/// The cache is keyed partly on the weight's address, so a second weight that lands on a
/// recycled allocation must not be served the first one's packed copy.
#[test]
fn xnnpack_does_not_serve_a_stale_packed_weight() {
    let (m, n, k) = (16usize, 512usize, 512usize);
    let lhs: Tensor<f32, CpuDevice> =
        Tensor::from_vec(fill(m * k, 7), (m, k), &CpuDevice).unwrap();

    let mut results = Vec::new();
    for seed in [11u64, 22, 33] {
        // Dropped before the next iteration, so the allocator is free to hand back the same
        // address with different contents.
        let w: Tensor<f32, CpuDevice> =
            Tensor::from_vec(fill(n * k, seed), (n, k), &CpuDevice).unwrap();
        results.push((seed, lhs.matmul_t(&w).unwrap().to_vec().unwrap(), w.to_vec().unwrap()));
    }
    for (seed, got, b) in results {
        let a = lhs.to_vec().unwrap();
        let mut worst = 0f64;
        let mut scale = 0f64;
        for i in 0..m {
            for j in 0..n {
                let want: f64 = (0..k).map(|p| a[i * k + p] as f64 * b[j * k + p] as f64).sum();
                scale = scale.max(want.abs());
                worst = worst.max((got[i * n + j] as f64 - want).abs());
            }
        }
        let rel = worst / scale.max(1e-30);
        assert!(rel < 1e-5, "seed {seed}: relative error {rel:.3e}, likely a stale packed weight");
    }
}
