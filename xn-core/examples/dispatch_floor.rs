//! Per-dispatch floor of a GPU backend: a chain of tiny dependent elementwise
//! ops, each one kernel on one small tensor, timed end to end.
//!
//! Build with exactly one of `--features cuda` or `--features vulkan`.
//! Usage: `dispatch_floor [elements] [ops per pass] [passes]`.
//!
//! Each pass runs `ops` dependent `scale` ops (out of place, so every op also
//! allocates from the pool) and ends with a synchronize, the way a decoded
//! frame ends with a readback. The number is what a frame of ~180 small
//! kernels pays per kernel on top of any arithmetic.

#[cfg(any(feature = "cuda", feature = "vulkan"))]
fn main() -> xn::Result<()> {
    use std::time::Instant;
    use xn::Tensor;

    #[cfg(feature = "cuda")]
    use xn::cuda_backend::Device;
    #[cfg(all(feature = "vulkan", not(feature = "cuda")))]
    use xn::vulkan_backend::Device;

    let args: Vec<String> = std::env::args().collect();
    let arg =
        |i: usize, default: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default);
    let n = arg(1, 1024);
    let ops = arg(2, 180);
    let passes = arg(3, 200);
    let dev = Device::new(0)?;

    let seed: Tensor<f32, Device> = Tensor::full(1.0, (n,), &dev)?;
    let pass = || -> xn::Result<()> {
        let mut t = seed.clone();
        for _ in 0..ops {
            t = t.scale(1.0001)?;
        }
        let _ = t.to_vec()?;
        Ok(())
    };
    for _ in 0..10 {
        pass()?;
    }
    let start = Instant::now();
    for _ in 0..passes {
        pass()?;
    }
    let us = start.elapsed().as_secs_f64() * 1e6 / passes as f64;
    println!(
        "{n} elements, {ops} dependent ops per pass: {us:.1} us per pass, {:.2} us per op",
        us / ops as f64
    );
    Ok(())
}

#[cfg(not(any(feature = "cuda", feature = "vulkan")))]
fn main() {
    eprintln!("build with --features cuda or --features vulkan");
}
