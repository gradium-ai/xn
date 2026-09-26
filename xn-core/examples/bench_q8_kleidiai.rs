//! The `q8_0` storages side by side: plain (row-tiled sgemm), the column-interleaved
//! `q8_0_4x8` layout, and every KleidiAI family this CPU can run, on the linear shapes of the
//! Phonon / Pocket TTS flow-LM transformer.
//!
//! Arms are timed in interleaved rounds (A, B, C, A, B, C, ...) so machine drift lands on all
//! of them, and each arm reports its best round next to the spread across rounds: a ratio that
//! sits inside the spread is noise, not a finding. Every arm starts from the same `q8_0`
//! blocks and includes its own activation quantization, so a row is the whole of what
//! `QLinear::forward` does apart from the bias add.
//!
//! Accuracy is the relative L2 distance of each arm's output from an f32 matmul against the
//! dequantized `q8_0` weights, which is what a lossless kernel would compute up to activation
//! rounding. The KleidiAI arms requantize the weights per row, so their number also carries
//! that loss.
//!
//! ```text
//! RAYON_NUM_THREADS=4 cargo run --release --features kleidiai --example bench_q8_kleidiai
//! XN_KLEIDIAI_THREADS=1 ...   # cap the KleidiAI arms' threads, e.g. to probe SME sharing
//! ```
use std::time::Instant;
use xn::quantized::QuantizedType;
use xn::quantized::k_quants::{BlockQ8_0, GgmlType, QK8_0};
use xn::quantized::kleidiai::{Family, Q8_0Kai};
use xn::quantized::repack::Q8_0x4;

struct Arm {
    name: String,
    storage: Box<dyn QuantizedType>,
}

fn arms(plain: &[BlockQ8_0], n: usize, k: usize, families: &[Family]) -> xn::Result<Vec<Arm>> {
    let mut arms = vec![
        Arm { name: "plain".into(), storage: Box::new(plain.to_vec()) },
        Arm { name: "interleaved".into(), storage: Box::new(Q8_0x4::from_q8_0(plain, n, k)?) },
    ];
    for &family in families {
        arms.push(Arm {
            name: format!("kai-{}", family.name()),
            storage: Box::new(Q8_0Kai::from_q8_0_for(family, plain, n, k)?),
        });
    }
    Ok(arms)
}

fn ref_matmul(w: &[f32], lhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut dst = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            dst[i * n + j] = (0..k).map(|l| lhs[i * k + l] * w[j * k + l]).sum();
        }
    }
    dst
}

fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
    let num: f32 = got.iter().zip(want).map(|(g, w)| (g - w).powi(2)).sum();
    let den: f32 = want.iter().map(|w| w.powi(2)).sum();
    (num / den.max(1e-20)).sqrt()
}

/// Microseconds per call over `iters` calls, after a short warm-up.
fn time_us<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn parse_list(s: &str) -> Vec<usize> {
    s.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().expect("an integer")).collect()
}

fn main() -> xn::Result<()> {
    let mut ms = vec![1usize, 16, 64, 256];
    let mut rounds = 5usize;
    let mut families: Vec<Family> = [Family::Sme2, Family::I8mm, Family::Dotprod]
        .into_iter()
        .filter(|f| f.available())
        .collect();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--m" => ms = parse_list(&args.next().expect("--m needs a list")),
            "--rounds" => rounds = args.next().expect("--rounds needs a number").parse().unwrap(),
            "--threads" => {
                xn::set_num_threads(args.next().expect("--threads needs a number").parse().unwrap())
            }
            "--families" => {
                let names = args.next().expect("--families needs a list");
                families = [Family::Sme2, Family::I8mm, Family::Dotprod]
                    .into_iter()
                    .filter(|f| names.split(',').any(|n| n == f.name()))
                    .collect();
            }
            other => panic!("unknown argument {other}"),
        }
    }
    println!(
        "threads={} kleidiai families: {}",
        xn::get_num_threads(),
        families.iter().map(|f| f.name()).collect::<Vec<_>>().join(",")
    );

    // Phonon: d_model 512, dim_feedforward 2048, fused qkv.
    let shapes: [(&str, usize, usize); 4] = [
        ("in_proj", 1536, 512),
        ("out_proj", 512, 512),
        ("linear1", 2048, 512),
        ("linear2", 512, 2048),
    ];

    // Best time per (m, arm) summed over the four shapes: one transformer layer's linears.
    let mut layer: Vec<Vec<f64>> = Vec::new();
    let mut arm_names: Vec<String> = Vec::new();

    for &m in &ms {
        let mut per_arm_total: Vec<f64> = vec![0.0; 2 + families.len()];
        for &(name, n, k) in &shapes {
            // Gaussian-ish weights with one heavier block per row, so the per-row
            // requantization has something to lose, as real rows do.
            let raw: Vec<f32> = (0..n * k)
                .map(|i| {
                    let u = ((i as u64).wrapping_mul(2654435761) % 4093) as f32 / 4093.0 - 0.5;
                    let boost = if (i % k) / QK8_0 == (i / k) % (k / QK8_0) { 3.0 } else { 1.0 };
                    u * boost / 8.0
                })
                .collect();
            let mut plain = vec![BlockQ8_0::zeros(); n * k / QK8_0];
            BlockQ8_0::from_float(&raw, &mut plain)?;
            let mut dq = vec![0f32; n * k];
            BlockQ8_0::to_float(&plain, &mut dq)?;
            let arms = arms(&plain, n, k, &families)?;
            if arm_names.is_empty() {
                arm_names = arms.iter().map(|a| a.name.clone()).collect();
            }

            let lhs: Vec<f32> =
                (0..m * k).map(|i| (((i * 40503) % 1021) as f32 - 510.0) / 256.0).collect();
            let want = ref_matmul(&dq, &lhs, m, k, n);
            let mut dst = vec![0f32; m * n];

            let iters = ((3e8 / (m * n * k) as f64) as usize).clamp(3, 400);
            let mut best = vec![f64::MAX; arms.len()];
            let mut worst = vec![0f64; arms.len()];
            let mut errs = vec![0f32; arms.len()];
            for _ in 0..rounds {
                for (i, arm) in arms.iter().enumerate() {
                    let us =
                        time_us(iters, || arm.storage.matmul_t((m, k, n), &lhs, &mut dst).unwrap());
                    best[i] = best[i].min(us);
                    worst[i] = worst[i].max(us);
                    errs[i] = rel_l2(&dst, &want);
                }
            }
            println!("\n{name} [{n}, {k}] m={m} ({iters} iters x {rounds} rounds)");
            println!(
                "  {:<14} {:>10} {:>8} {:>9} {:>8}",
                "arm", "best us", "spread", "GMAC/s", "rel-l2"
            );
            for (i, arm) in arms.iter().enumerate() {
                let gmacs = (m * n * k) as f64 / best[i] / 1e3;
                let spread = (worst[i] - best[i]) / best[i] * 100.0;
                let vs = best[0] / best[i];
                println!(
                    "  {:<14} {:>10.1} {:>7.0}% {:>9.1} {:>8.1e}  x{vs:.2} vs plain",
                    arm.name, best[i], spread, gmacs, errs[i]
                );
                per_arm_total[i] += best[i];
            }
        }
        layer.push(per_arm_total);
    }

    println!("\nOne layer's four linears, best-of-round sums (us):");
    print!("  {:<6}", "m");
    for name in &arm_names {
        print!(" {name:>12}");
    }
    println!();
    for (m, totals) in ms.iter().zip(&layer) {
        print!("  {m:<6}");
        for t in totals {
            print!(" {t:>12.1}");
        }
        println!();
    }
    Ok(())
}
