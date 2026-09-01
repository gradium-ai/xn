//! Separates the WebGPU backend's fixed per-dispatch cost from real kernel work.
//!
//! Every loop below records dispatches without ever reading back, so they batch
//! into one compute pass and one submit; the single `synchronize` at the end
//! pays for all of them. Dividing by the iteration count therefore gives the
//! marginal cost of one dispatch, which is the number that decides whether an
//! autoregressive decode (hundreds of tiny ops per frame) can ever be fast on
//! this backend.
//!
//! Run with: cargo run --release --features webgpu --example webgpu_dispatch_cost

use std::time::Instant;
use xn::{Backend, Result, Tensor, webgpu_backend::Device};

fn bench(label: &str, dev: &Device, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<()> {
    for _ in 0..8 {
        f()?;
    }
    dev.synchronize()?;
    let start = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    dev.synchronize()?;
    let us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("{label:<46} {us:>8.2} us/op");
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    println!("device: {}", dev.name());
    println!("adapter shader-f16: {}\n", dev.adapter_supports_f16());

    // Floor: the smallest dispatch the backend can issue -- one workgroup.
    let tiny: Tensor<f32, Device> = Tensor::from_vec(vec![1f32; 256], (256,), &dev)?;
    let tiny_dst: Tensor<f32, Device> = Tensor::from_vec(vec![0f32; 256], (256,), &dev)?;
    bench("unary, 256 elems (1 workgroup)", &dev, 4000, || {
        tiny_dst.silu_(&tiny)?;
        Ok(())
    })?;

    // Same op with enough data to actually occupy the GPU. If this costs about
    // the same as the 256-element case, the backend is dispatch-bound.
    // The same op letting the result allocate, which is what model code does and
    // what exercises the buffer pool.
    bench("unary, 256 elems, allocating", &dev, 400, || {
        tiny.silu()?;
        Ok(())
    })?;

    let med: Tensor<f32, Device> = Tensor::from_vec(vec![1f32; 1 << 16], (1 << 16,), &dev)?;
    let med_dst: Tensor<f32, Device> = Tensor::from_vec(vec![0f32; 1 << 16], (1 << 16,), &dev)?;
    bench("unary, 65536 elems (256 workgroups)", &dev, 500, || {
        med_dst.silu_(&med)?;
        Ok(())
    })?;

    // The decode-shaped GEMVs the backbone and MLP actually run (m == 1).
    // `Linear::forward` calls `matmul_t`, whose weight rows are contiguous and
    // hit the kernel's vectorised path; plain `matmul` leaves the k stride at
    // `n`, so successive threads read 4n bytes apart and coalescing is lost.
    // Both are measured because the model contains both.
    let iters = 1500;
    for (k, n) in [(768usize, 768usize), (768, 3072), (3072, 768)] {
        let a: Tensor<f32, Device> = Tensor::from_vec(vec![0.01f32; k], (1, k), &dev)?;
        let flops = 2.0 * (k * n) as f64;
        let bytes = (k * n * 4) as f64;

        // matmul_t: weight is [n, k], each output row contiguous.
        let wt: Tensor<f32, Device> = Tensor::from_vec(vec![0.01f32; k * n], (n, k), &dev)?;
        let out: Tensor<f32, Device> = Tensor::from_vec(vec![0f32; n], (1, n), &dev)?;
        for _ in 0..8 {
            out.matmul_(&a, &wt, true)?;
        }
        dev.synchronize()?;
        let t = Instant::now();
        for _ in 0..iters {
            out.matmul_(&a, &wt, true)?;
        }
        dev.synchronize()?;
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "{:<46} {us:>8.2} us/op {:>7.1} GFLOP/s {:>6.0} GB/s",
            format!("gemv_t 1x{k} @ {n}x{k} (Linear path)"),
            flops / (us * 1e3),
            bytes / (us * 1e3)
        );

        // Same shape in f16, to line up with the browser harness's report.
        if dev.adapter_supports_f16() {
            use half::f16;
            let a16: Tensor<f16, Device> =
                Tensor::from_vec(vec![f16::from_f32(0.01); k], (1, k), &dev)?;
            let w16: Tensor<f16, Device> =
                Tensor::from_vec(vec![f16::from_f32(0.01); k * n], (n, k), &dev)?;
            let out16: Tensor<f16, Device> =
                Tensor::from_vec(vec![f16::from_f32(0.0); n], (1, n), &dev)?;
            for _ in 0..8 {
                out16.matmul_(&a16, &w16, true)?;
            }
            dev.synchronize()?;
            let t = Instant::now();
            for _ in 0..iters {
                out16.matmul_(&a16, &w16, true)?;
            }
            dev.synchronize()?;
            let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "{:<46} {us:>8.2} us/op {:>7.1} GFLOP/s {:>6.0} GB/s",
                format!("gemv_t f16 1x{k} @ {n}x{k}"),
                flops / (us * 1e3),
                (k * n * 2) as f64 / (us * 1e3)
            );
        }

        // Plain matmul: weight is [k, n].
        let b: Tensor<f32, Device> = Tensor::from_vec(vec![0.01f32; k * n], (k, n), &dev)?;
        for _ in 0..8 {
            a.matmul(&b)?;
        }
        dev.synchronize()?;
        let t = Instant::now();
        for _ in 0..iters {
            a.matmul(&b)?;
        }
        dev.synchronize()?;
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "{:<46} {us:>8.2} us/op {:>7.1} GFLOP/s {:>6.0} GB/s",
            format!("gemv   1x{k} @ {k}x{n} (strided)"),
            flops / (us * 1e3),
            bytes / (us * 1e3)
        );
    }
    Ok(())
}
