//! f32 against q8_0 weights on the WebGPU backend, at Phonon's linear shapes.
//!
//!   cargo run --release --no-default-features --features webgpu \
//!     --example bench_webgpu_q8
//!
//! Reports microseconds and the weight bandwidth each variant sustains. Decode
//! (`m = 1`) streams the whole weight per call and nothing else of consequence,
//! so GB/s is the number that says whether a kernel is finished: the ratio
//! between the two dtypes should approach 4x, the ratio of their bytes.

#[cfg(not(feature = "webgpu"))]
fn main() {
    eprintln!("build with --features webgpu");
}

#[cfg(feature = "webgpu")]
fn main() -> xn::Result<()> {
    use xn::webgpu_backend::Device;
    use xn::webgpu_backend::quantization::Q8Tensor;
    use xn::{Backend, Shape, Tensor};

    let dev = Device::new(0)?;
    println!("device: {}\n", dev.name());

    // (in_features, out_features, label). The backbone and flow-net linears of
    // the Pocket TTS checkpoint, plus the 1024-wide variant.
    let shapes = [
        (768, 2304, "backbone qkv   768->2304"),
        (768, 768, "backbone out    768->768"),
        (768, 3072, "backbone ffn1   768->3072"),
        (3072, 768, "backbone ffn2  3072->768"),
        (512, 1536, "flow qkv        512->1536"),
        (512, 2048, "flow ffn1       512->2048"),
        (2048, 512, "flow ffn2      2048->512"),
        (1024, 4096, "wide ffn1      1024->4096"),
        (4096, 1024, "wide ffn2      4096->1024"),
    ];

    for &m in &[1usize, 4, 8, 16, 30, 64, 120] {
        println!("m = {m}");
        println!(
            "{:<26} {:>10} {:>10} {:>8} {:>10} {:>10}",
            "shape", "f32 us", "q8 us", "speedup", "f32 GB/s", "q8 GB/s"
        );
        let mut tot_f32 = 0f64;
        let mut tot_q8 = 0f64;
        for &(k, n, label) in &shapes {
            let w: Vec<f32> = (0..n * k).map(|i| ((i % 251) as f32 - 125.0) * 0.004).collect();
            let x: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();

            let wt: Tensor<f32, Device> = Tensor::from_vec(w.clone(), (n, k), &dev)?;
            let xt: Tensor<f32, Device> = Tensor::from_vec(x, (m, k), &dev)?;
            let wq = Q8Tensor::from_f32(&dev, &w, &Shape::from((n, k)))?;

            let iters = 100;
            // Warm up pipelines, bind groups and the buffer pool first: the
            // first call of each compiles a shader.
            for _ in 0..5 {
                let _ = xt.matmul_t(&wt)?;
                let _ = wq.matmul_t(&xt)?;
            }
            dev.synchronize()?;

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = xt.matmul_t(&wt)?;
            }
            dev.synchronize()?;
            let us_f32 = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = wq.matmul_t(&xt)?;
            }
            dev.synchronize()?;
            let us_q8 = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // Weight bytes only: q8_0 is 32 quants plus an f16 scale per block.
            let bytes_f32 = (n * k * 4) as f64;
            let bytes_q8 = (n * k) as f64 + (n * k / 32 * 2) as f64;
            tot_f32 += us_f32;
            tot_q8 += us_q8;
            println!(
                "{label:<26} {us_f32:>10.1} {us_q8:>10.1} {:>7.2}x {:>10.1} {:>10.1}",
                us_f32 / us_q8,
                bytes_f32 / us_f32 / 1e3,
                bytes_q8 / us_q8 / 1e3,
            );
        }
        println!("{:<26} {tot_f32:>10.1} {tot_q8:>10.1} {:>7.2}x\n", "total", tot_f32 / tot_q8);
    }
    Ok(())
}
