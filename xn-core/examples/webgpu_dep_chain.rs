//! Independent dispatches against dependent ones.
//!
//! `webgpu_dispatch_cost` measures dispatches that do not read each other's
//! output, so the GPU can overlap them and the marginal cost comes out at
//! ~4.5 us. A model is the opposite: nearly every op consumes the previous
//! one's result, so consecutive dispatches cannot overlap and each pays the
//! full pipeline latency instead of a slot in it.
//!
//! Phonon's q8 frame issues ~559 dispatches and takes ~17 ms of GPU time,
//! while the arithmetic in it accounts for well under 1 ms. If a dependent
//! dispatch costs tens of microseconds, that product is the frame -- and the
//! only lever is issuing fewer of them.
//!
//! Run with:
//!   cargo run --release --no-default-features --features webgpu \
//!     --example webgpu_dep_chain

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

    let n = 4096; // 16 workgroups: small enough that work is not the story
    let chain = 512;

    // Independent: every dispatch reads the same input, so nothing orders them.
    let src: Tensor<f32, Device> =
        Tensor::from_vec((0..n).map(|i| (i % 13) as f32 * 0.1).collect(), (n,), &dev)?;
    for _ in 0..8 {
        let _ = src.silu()?;
    }
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..chain {
        let _ = src.silu()?;
    }
    dev.synchronize()?;
    let indep = t0.elapsed().as_secs_f64() * 1e6 / chain as f64;

    // Dependent: each dispatch consumes the previous result, as a model does.
    let mut x = src.clone();
    for _ in 0..8 {
        x = x.silu()?;
    }
    dev.synchronize()?;
    let t0 = Instant::now();
    let mut x = src.clone();
    for _ in 0..chain {
        x = x.silu()?;
    }
    dev.synchronize()?;
    let dep = t0.elapsed().as_secs_f64() * 1e6 / chain as f64;
    // Keep the chain from being optimized away.
    let _ = x.to_vec()?;

    println!("independent silu dispatches   {indep:>8.2} us/op");
    println!("dependent silu chain          {dep:>8.2} us/op   ({:.1}x)", dep / indep);
    println!();
    println!("At {dep:.1} us, Phonon's 559 dispatches per frame come to");
    println!("{:.1} ms of latency alone, against a ~17 ms measured frame.", dep * 559.0 / 1e3);
    Ok(())
}
