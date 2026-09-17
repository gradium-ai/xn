//! Per-shape matmul microbenchmark over every f32 GEMM / GEMV shape Phonon's
//! decode step issues (flow net, attention and the Mimi decoder convolutions
//! after im2col), plus the q8_0 shapes of the flow LM. The list was taken from
//! `XN_VULKAN_PROFILE=1` on the ptts `bench` example, one iteration, and the
//! counts are per generated frame.
//!
//! Build with exactly one of `--features cuda` or `--features vulkan`.
//! Usage: `gemm_shapes_bench [iters] [NT|NN|Q8]`.
//!
//! Each shape gets `CALLS` distinct weights (the model has one per layer, so
//! consecutive calls never hit the same lines) and is timed as `iters` passes
//! of `CALLS` calls, each pass ending with a synchronize so Vulkan's buffer
//! pool recycles the outputs. The synchronize itself is timed separately with
//! an empty pass and subtracted, so a row is the per-call cost including
//! submission but not the end-of-frame wait. `NT` is `lhs.matmul_t(w)` with
//! `w` row-major `(n, k)`, the layout of a weight matrix; `NN` is
//! `lhs.matmul(w)` with `w` row-major `(k, n)`.

#[cfg(any(feature = "cuda", feature = "vulkan"))]
fn main() -> xn::Result<()> {
    use std::time::Instant;
    use xn::{Shape, Tensor};

    #[cfg(feature = "cuda")]
    use xn::cuda_backend::{Device, q8::Q8Tensor};
    #[cfg(all(feature = "vulkan", not(feature = "cuda")))]
    use xn::vulkan_backend::{Device, quantization::Q8Tensor};

    #[derive(Clone, Copy, PartialEq)]
    enum Layout {
        Nt,
        Nn,
        Q8,
    }
    // (layout, m, n, k, batch, calls per frame)
    let shapes: &[(Layout, usize, usize, usize, usize, usize)] = &[
        // flow net and conditioner GEMVs
        (Layout::Nt, 1, 512, 512, 1, 14),
        (Layout::Nt, 1, 1536, 512, 1, 6),
        (Layout::Nt, 1, 512, 32, 1, 2),
        (Layout::Nt, 1, 512, 256, 1, 2),
        (Layout::Nt, 1, 512, 1024, 1, 1),
        (Layout::Nt, 1, 1024, 512, 1, 1),
        (Layout::Nt, 1, 1024, 32, 1, 1),
        (Layout::Nt, 1, 32, 512, 1, 1),
        (Layout::Nt, 1, 1, 1024, 1, 1),
        // batched attention over a ~176-token cache, 16 heads
        (Layout::Nt, 1, 176, 64, 16, 1),
        // Mimi decoder convolutions (im2col + GEMM)
        (Layout::Nt, 16, 512, 2048, 1, 2),
        (Layout::Nt, 16, 512, 3584, 1, 1),
        (Layout::Nt, 16, 2048, 512, 1, 2),
        (Layout::Nt, 16, 1536, 512, 1, 2),
        (Layout::Nt, 16, 512, 512, 1, 2),
        (Layout::Nt, 96, 128, 768, 1, 1),
        (Layout::Nt, 480, 64, 384, 1, 1),
        (Layout::Nt, 1920, 32, 192, 1, 1),
        (Layout::Nt, 1920, 1, 192, 1, 1),
        (Layout::Nt, 96, 256, 128, 1, 1),
        (Layout::Nt, 480, 128, 64, 1, 1),
        (Layout::Nt, 1920, 64, 32, 1, 1),
        (Layout::Nt, 16, 266, 64, 8, 1),
        (Layout::Nn, 16, 3072, 512, 1, 1),
        (Layout::Nn, 96, 1280, 256, 1, 1),
        (Layout::Nn, 480, 512, 128, 1, 1),
        (Layout::Nn, 16, 64, 266, 8, 1),
        // flow LM q8_0 linears: decode (m=1), text prompt (m=16) and voice
        // conditioning (m=125; once per utterance rather than per frame)
        (Layout::Q8, 1, 3072, 1024, 1, 6),
        (Layout::Q8, 1, 1024, 1024, 1, 6),
        (Layout::Q8, 1, 4096, 1024, 1, 6),
        (Layout::Q8, 1, 1024, 4096, 1, 6),
        (Layout::Q8, 16, 3072, 1024, 1, 6),
        (Layout::Q8, 16, 1024, 1024, 1, 6),
        (Layout::Q8, 16, 4096, 1024, 1, 6),
        (Layout::Q8, 16, 1024, 4096, 1, 6),
        (Layout::Q8, 125, 3072, 1024, 1, 6),
        (Layout::Q8, 125, 1024, 1024, 1, 6),
        (Layout::Q8, 125, 4096, 1024, 1, 6),
        (Layout::Q8, 125, 1024, 4096, 1, 6),
    ];

    const CALLS: usize = 32;
    let args: Vec<String> = std::env::args().collect();
    let iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let only = args.get(2).cloned();
    let dev = Device::new(0)?;

    // Cost of an empty pass: submit nothing, wait for the device.
    let sync_us = {
        let t: Tensor<f32, Device> = Tensor::zeros((16,), &dev)?;
        for _ in 0..20 {
            let _ = t.to_vec()?;
            xn::Backend::synchronize(&dev)?;
        }
        let start = Instant::now();
        for _ in 0..200 {
            xn::Backend::synchronize(&dev)?;
        }
        start.elapsed().as_secs_f64() * 1e6 / 200.0
    };
    println!("synchronize: {sync_us:.1} us (subtracted from every row)");

    enum W {
        F32(Tensor<f32, Device>),
        Q8(Q8Tensor),
    }

    println!(
        "{:<3} {:>5} {:>5} {:>5} {:>3} {:>5} {:>9} {:>8} {:>8} {:>8} {:>9}",
        "lay", "m", "n", "k", "b", "calls", "us/call", "issue", "rest", "GB/s", "us/frame"
    );
    let mut frame_us = 0.0;
    for &(layout, m, n, k, b, count) in shapes {
        let lay = match layout {
            Layout::Nt => "NT",
            Layout::Nn => "NN",
            Layout::Q8 => "Q8",
        };
        if only.as_deref().is_some_and(|o| o != lay) {
            continue;
        }
        let lhs_dims: Vec<usize> = if b == 1 { vec![m, k] } else { vec![b, m, k] };
        let x: Vec<f32> = (0..b * m * k).map(|i| ((i % 53) as f32 - 26.0) * 0.021).collect();
        let lhs: Tensor<f32, Device> = Tensor::from_vec(x, Shape::from(lhs_dims), &dev)?;
        let mut weights = Vec::with_capacity(CALLS);
        for c in 0..CALLS {
            let w: Vec<f32> = (0..b * n * k)
                .map(|i| ((i % 71) as f32 - 35.0) * 0.013 * (1.0 + c as f32 * 0.01))
                .collect();
            weights.push(match layout {
                Layout::Nt => {
                    let dims: Vec<usize> = if b == 1 { vec![n, k] } else { vec![b, n, k] };
                    W::F32(Tensor::from_vec(w, Shape::from(dims), &dev)?)
                }
                Layout::Nn => {
                    let dims: Vec<usize> = if b == 1 { vec![k, n] } else { vec![b, k, n] };
                    W::F32(Tensor::from_vec(w, Shape::from(dims), &dev)?)
                }
                Layout::Q8 => W::Q8(Q8Tensor::from_f32(&dev, &w, &Shape::from((n, k)))?),
            });
        }
        // Returns the time spent issuing the calls, before the synchronize:
        // on Vulkan that is command recording the GPU has not started on
        // yet; on CUDA the launches overlap with execution.
        let pass = || -> xn::Result<f64> {
            let t = Instant::now();
            for w in &weights {
                let _ = match (layout, w) {
                    (Layout::Nt, W::F32(w)) => lhs.matmul_t(w)?,
                    (Layout::Nn, W::F32(w)) => lhs.matmul(w)?,
                    (_, W::Q8(w)) => w.matmul_t(&lhs)?,
                    _ => unreachable!(),
                };
            }
            let issue = t.elapsed().as_secs_f64() * 1e6;
            xn::Backend::synchronize(&dev)?;
            Ok(issue)
        };
        for _ in 0..5 {
            pass()?;
        }
        let start = Instant::now();
        let mut issue_us = 0.0;
        for _ in 0..iters {
            issue_us += pass()?;
        }
        let pass_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64 - sync_us;
        let us = pass_us / CALLS as f64;
        let issue = issue_us / iters as f64 / CALLS as f64;
        // What is left once issuing is taken out: on Vulkan the GPU-side
        // time per call, since execution starts at the submit.
        let rest = (pass_us - issue_us / iters as f64).max(0.0) / CALLS as f64;
        let w_bytes =
            if layout == Layout::Q8 { (n * k) as f64 * 1.125 } else { (b * n * k * 4) as f64 };
        let bytes = w_bytes + (b * m * k * 4) as f64 + (b * m * n * 4) as f64;
        // The m=125 q8 shapes run once per utterance, not per frame.
        let per_frame = if layout == Layout::Q8 && m > 16 { 0.0 } else { us * count as f64 };
        frame_us += per_frame;
        println!(
            "{lay:<3} {m:>5} {n:>5} {k:>5} {b:>3} {count:>5} {us:>9.2} {issue:>8.2} {rest:>8.2} {:>8.1} {per_frame:>9.1}",
            bytes / us / 1e3
        );
    }
    println!("per-frame matmul total: {frame_us:.1} us");
    Ok(())
}

#[cfg(not(any(feature = "cuda", feature = "vulkan")))]
fn main() {
    eprintln!("build with --features cuda or --features vulkan");
}
