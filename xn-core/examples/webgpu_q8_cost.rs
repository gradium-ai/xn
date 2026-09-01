//! Times a `q8_0` linear layer against an f16 one, through the real backend, at
//! phonon's decode shapes.
//!
//! The kernel lab says q8_0 should beat f16 by ~2.4x on these shapes, but
//! end-to-end the q8 model run is slower than the f16 one *and* slows down the
//! Mimi decoder, which shares no q8 code. This isolates which of the two it is:
//! a kernel that is slower than the lab measured, or something systemic about
//! having the quantized layers resident.
//!
//! Run with: cargo run --release --features webgpu --example webgpu_q8_cost

use half::f16;
use std::time::Instant;
use xn::webgpu_backend::Device;
use xn::webgpu_backend::quantization::Q80F16;
use xn::{BackendQ, Result, Tensor, nn};

// (label, n, k, dispatches per frame)
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("flow_lm in_proj ", 2304, 768, 12),
    ("flow_lm out_proj", 768, 768, 12),
    ("flow_lm linear1 ", 3072, 768, 12),
    ("flow_lm linear2 ", 768, 3072, 12),
];

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    println!("device: {}", <Device as xn::Backend>::name(&dev));
    println!("f16 compute: {}\n", dev.supports_f16());

    let iters = 200usize;
    let mut tot_f16 = 0f64;
    let mut tot_q8 = 0f64;

    for &(label, n, k, per_frame) in SHAPES {
        let wdata: Vec<f16> =
            (0..n * k).map(|i| f16::from_f32(((i % 251) as f32 - 125.0) / 256.0)).collect();
        let xdata: Vec<f16> =
            (0..k).map(|i| f16::from_f32(((i % 17) as f32 - 8.0) / 8.0)).collect();
        let x: Tensor<f16, Device> = Tensor::from_vec(xdata, vec![1, k], &dev)?;

        let w: Tensor<f16, Device> = Tensor::from_vec(wdata.clone(), vec![n, k], &dev)?;
        let dense = nn::Linear::new(w);

        let w2: Tensor<f16, Device> = Tensor::from_vec(wdata, vec![n, k], &dev)?;
        let q8 = Q80F16::from_linear(nn::Linear::new(w2))?;

        // Time a run of forwards inside one batch, so the number is the marginal
        // cost of the layer rather than of submission.
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            for _ in 0..8 {
                f()?;
            }
            <Device as xn::Backend>::synchronize(&dev)?;
            let t = Instant::now();
            for _ in 0..iters {
                f()?;
            }
            <Device as xn::Backend>::synchronize(&dev)?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / iters as f64)
        };

        let us_f16 = time(&|| dense.forward(&x).map(|_| ()))?;
        let us_q8 = time(&|| q8.forward(&x).map(|_| ()))?;
        tot_f16 += us_f16 * per_frame as f64;
        tot_q8 += us_q8 * per_frame as f64;

        let f16_bytes = (n * k * 2) as f64;
        let q8_bytes = (n * k) as f64 * 1.125;
        println!(
            "{label} n={n:<5} k={k:<5}  f16 {us_f16:>7.2} us ({:>5.0} GB/s)   \
             q8 {us_q8:>7.2} us ({:>5.0} GB/s)   {:>4.2}x",
            f16_bytes / (us_f16 * 1e3),
            q8_bytes / (us_q8 * 1e3),
            us_f16 / us_q8
        );
    }

    println!(
        "\nweighted per-frame: f16 {:.3} ms, q8 {:.3} ms  ({:.2}x)",
        tot_f16 / 1e3,
        tot_q8 / 1e3,
        tot_f16 / tot_q8
    );

    interference(&dev)?;
    Ok(())
}

/// Does having the quantized pipeline in a batch slow down unrelated work?
///
/// End to end, enabling q8 slows the Mimi decoder by ~1.8x even though Mimi runs
/// f32 kernels and shares no q8 code, while the quantized layers themselves get
/// 2.35x faster in isolation. If a register-heavy pipeline in the same compute
/// pass cost its neighbours occupancy, that would explain it: A alone, B alone,
/// then A and B interleaved in one batch should be super-additive.
fn interference(dev: &Device) -> Result<()> {
    println!("\n=== interference: unrelated work alongside q8, one batch ===");
    let n = 3072usize;
    let k = 768usize;
    let wdata: Vec<f16> =
        (0..n * k).map(|i| f16::from_f32(((i % 251) as f32 - 125.0) / 256.0)).collect();
    let xdata: Vec<f16> = (0..k).map(|i| f16::from_f32(((i % 17) as f32 - 8.0) / 8.0)).collect();
    let x: Tensor<f16, Device> = Tensor::from_vec(xdata, vec![1, k], dev)?;
    let q8 = Q80F16::from_linear(nn::Linear::new(Tensor::from_vec(wdata, vec![n, k], dev)?))?;

    // Stand-in for Mimi's work: an f32 elementwise pass over a comparable
    // number of elements, sharing nothing with the quantized layer.
    let big: Tensor<f32, Device> = Tensor::zeros(vec![1 << 18], dev)?;

    let reps = 40usize;
    let run = |q: bool, u: bool| -> Result<f64> {
        for _ in 0..4 {
            if q {
                q8.forward(&x)?;
            }
            if u {
                big.silu()?;
            }
        }
        <Device as xn::Backend>::synchronize(dev)?;
        let t = Instant::now();
        for _ in 0..reps {
            if q {
                q8.forward(&x)?;
            }
            if u {
                big.silu()?;
            }
        }
        <Device as xn::Backend>::synchronize(dev)?;
        Ok(t.elapsed().as_secs_f64() * 1e6 / reps as f64)
    };

    let only_q8 = run(true, false)?;
    let only_unary = run(false, true)?;
    let both = run(true, true)?;
    println!("  q8 alone                {only_q8:>8.2} us/rep");
    println!("  f32 silu alone          {only_unary:>8.2} us/rep");
    println!("  interleaved in one pass {both:>8.2} us/rep");
    println!(
        "  sum of parts            {:>8.2} us/rep  -> interleaved is {:.2}x the sum",
        only_q8 + only_unary,
        both / (only_q8 + only_unary)
    );
    Ok(())
}
