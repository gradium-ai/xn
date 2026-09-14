//! Times the GEMM shapes Phonon's Mimi decoder actually issues, so the tile can
//! be chosen against them.
//!
//! Shapes and per-frame counts come from `XN_WEBGPU_GEMM_SHAPES=1` on a q8 run
//! of `ptts/examples/bench`. The question this answers is narrower than it
//! looks: the vocoder costs ~17 ms of a 23 ms frame, and if these GEMMs do not
//! add up to most of that, the tile is the wrong thing to work on.
//!
//! Run with:
//!   cargo run --release --no-default-features --features webgpu \
//!     --example webgpu_vocoder_gemm

#[cfg(not(feature = "webgpu"))]
fn main() {
    eprintln!("build with --features webgpu");
}

#[cfg(feature = "webgpu")]
fn main() -> xn::Result<()> {
    use xn::webgpu_backend::Device;
    use xn::{Backend, Tensor};

    let dev = Device::new(0)?;
    println!("device: {}\n", dev.name());

    // (m, n, k, per-frame count). Counts are the histogram divided by the 54
    // frames an utterance produced, rounded to the nearest tenth.
    let shapes: &[(usize, usize, usize, f64)] = &[
        (16, 2048, 512, 2.0),
        (16, 1536, 512, 2.0),
        (16, 512, 2048, 2.0),
        (16, 512, 512, 2.0),
        (16, 64, 266, 1.4),
        (16, 266, 64, 1.4),
        (480, 512, 128, 1.0),
        (96, 256, 128, 1.0),
        (96, 128, 768, 1.0),
        (16, 512, 3584, 1.0),
        (16, 3072, 512, 1.0),
        (96, 1280, 256, 1.0),
    ];

    println!(
        "{:<22} {:>9} {:>11} {:>9} {:>11}",
        "shape", "us/op", "GFLOP/s", "per frame", "ms/frame"
    );
    let mut total_ms = 0f64;
    for &(m, n, k, per_frame) in shapes {
        let a: Tensor<f32, Device> =
            Tensor::from_vec((0..m * k).map(|i| (i % 97) as f32 * 0.01).collect(), (m, k), &dev)?;
        // `matmul_t` is what a Linear and an im2col conv both produce.
        let b: Tensor<f32, Device> =
            Tensor::from_vec((0..n * k).map(|i| (i % 89) as f32 * 0.01).collect(), (n, k), &dev)?;

        for _ in 0..5 {
            let _ = a.matmul_t(&b)?;
        }
        dev.synchronize()?;
        let iters = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = a.matmul_t(&b)?;
        }
        dev.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let gflops = 2.0 * (m * n * k) as f64 / (us * 1e3);
        let ms_frame = us * per_frame / 1e3;
        total_ms += ms_frame;
        println!(
            "m={m:<4} n={n:<5} k={k:<6} {us:>9.1} {gflops:>11.1} {per_frame:>9.1} {ms_frame:>11.3}"
        );
    }
    println!("\ntotal GEMM cost per frame: {total_ms:.2} ms");
    println!("(vocoder measures ~17.3 ms/frame in the q8 bench)");
    Ok(())
}
