//! Microbenchmark for the q8_0 dequantize-fused matmul on a GPU backend, over
//! the linear shapes Phonon's flow LM runs at decode time.
//!
//! Build with exactly one of `--features cuda` or `--features vulkan`.
//! Usage: `q8_gemv_bench [m] [layers] [iters]`.
//!
//! A pass runs the four linears of every layer in order, the way a decode step
//! does, so the weight working set is `layers` times 12 MiB: with the default
//! six layers that is 74 MB, past the L2 of any current GPU, and the kernels
//! stream their weights from DRAM as they do in the model. (A single layer
//! fits in a 48 MB L2 and reports bandwidth well above what the card has.)
//! Each pass ends with a synchronize, as a decode step ends with reading the
//! frame back, so the number includes submission cost the way the model pays
//! it and, on Vulkan, lets the buffer pool recycle the outputs (without a
//! flush every call would allocate fresh device memory).

#[cfg(any(feature = "cuda", feature = "vulkan"))]
fn main() -> xn::Result<()> {
    use std::time::Instant;
    use xn::{Shape, Tensor};

    #[cfg(feature = "cuda")]
    use xn::cuda_backend::{Device, q8::Q8Tensor};
    #[cfg(all(feature = "vulkan", not(feature = "cuda")))]
    use xn::vulkan_backend::{Device, quantization::Q8Tensor};

    let args: Vec<String> = std::env::args().collect();
    let arg =
        |i: usize, default: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default);
    let m = arg(1, 1);
    let layers = arg(2, 6);
    let iters = arg(3, 200);
    let dev = Device::new(0)?;

    // (name, k, n): in_proj, out_proj, linear1, linear2 of the 1024-wide,
    // 4096-ffn transformer layer.
    let shapes = [
        ("in_proj", 1024, 3072),
        ("out_proj", 1024, 1024),
        ("linear1", 1024, 4096),
        ("linear2", 4096, 1024),
    ];
    let mut weights = Vec::new();
    let mut inputs = Vec::new();
    let mut weight_bytes = 0f64;
    for layer in 0..layers {
        for (_, k, n) in shapes {
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i % 71) as f32 - 35.0) * 0.013 * (1.0 + layer as f32 * 0.01))
                .collect();
            weights.push(Q8Tensor::from_f32(&dev, &w, &Shape::from((n, k)))?);
            // int8 quants plus one f32 scale per 32 weights.
            weight_bytes += (n * k) as f64 * (1.0 + 4.0 / 32.0);
        }
    }
    for (_, k, _) in shapes {
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 53) as f32 - 26.0) * 0.021).collect();
        inputs.push(Tensor::<f32, Device>::from_vec(x, (m, k), &dev)?);
    }
    let pass = |weights: &[Q8Tensor]| -> xn::Result<()> {
        for (i, w) in weights.iter().enumerate() {
            let _ = w.matmul_t(&inputs[i % shapes.len()])?;
        }
        xn::Backend::synchronize(&dev)
    };

    // Warm up: pipeline creation, module load, allocator.
    for _ in 0..10 {
        pass(&weights)?;
    }
    let start = Instant::now();
    for _ in 0..iters {
        pass(&weights)?;
    }
    let us_pass = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let calls = weights.len();
    println!(
        "m {m}  layers {layers}  weights {:.1} MB  pass {us_pass:.1} us  per call {:.2} us  {:.1} GB/s",
        weight_bytes / 1e6,
        us_pass / calls as f64,
        weight_bytes / us_pass / 1e3
    );
    Ok(())
}

#[cfg(not(any(feature = "cuda", feature = "vulkan")))]
fn main() {
    eprintln!("build with --features cuda or --features vulkan");
}
