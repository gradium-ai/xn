//! What one host round trip costs on this backend, with no GPU work in it.
//!
//! `to_vec` on a tensor is flush + submit + `poll(Wait)` + staging copy + map.
//! An autoregressive decode does that once or twice per frame, so if the fixed
//! part is milliseconds rather than microseconds it sets the frame time no
//! matter how good the kernels are. Phonon's q8 frame spends ~21 ms against
//! Metal's 6 ms for the same graph, and the GEMMs in it account for 5 ms --
//! this is the probe that says whether the difference is latency.
//!
//! Run with:
//!   cargo run --release --no-default-features --features webgpu \
//!     --example webgpu_sync_cost

#[cfg(not(feature = "webgpu"))]
fn main() {
    eprintln!("build with --features webgpu");
}

#[cfg(feature = "webgpu")]
fn main() -> xn::Result<()> {
    use std::time::Instant;
    use xn::webgpu_backend::Device;
    use xn::{Backend, Tensor};

    let dev = Device::new(0)?;
    println!("device: {}\n", dev.name());

    let iters = 200;

    // 1. synchronize() with nothing recorded: the floor of a flush.
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        dev.synchronize()?;
    }
    let empty_sync = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("synchronize(), nothing recorded          {empty_sync:>9.1} us");

    // 2. A 1-element readback, nothing else recorded. This is the eos check.
    let one: Tensor<f32, Device> = Tensor::from_vec(vec![1.0f32], (1, 1), &dev)?;
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = one.to_vec()?;
    }
    let one_rb = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("to_vec() of 1 element                    {one_rb:>9.1} us");

    // 3. A 1920-element readback: the audio chunk.
    let pcm: Tensor<f32, Device> = Tensor::from_vec(vec![0.5f32; 1920], (1, 1920), &dev)?;
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = pcm.to_vec()?;
    }
    let pcm_rb = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("to_vec() of 1920 elements                {pcm_rb:>9.1} us");

    // 4. Both, as a frame does them.
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = pcm.to_vec()?;
        let _ = one.to_vec()?;
    }
    let both = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("to_vec() 1920 then 1 (a frame's pair)    {both:>9.1} us");

    // 5. For scale: one dispatch, no readback, amortized over the batch.
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![1.0f32; 65536], (65536,), &dev)?;
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = x.silu()?;
    }
    dev.synchronize()?;
    let disp = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("one silu dispatch, batched, no readback  {disp:>9.1} us");

    println!("\nA frame does the pair in row 4. Against a 21 ms frame, that is");
    println!("the share of it that no kernel change can touch.");
    Ok(())
}
