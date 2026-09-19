//! Benchmark the native CUDA backend against the XLA backend on the same GPU.
//!
//! Run with:
//!   cargo run --release --features cuda,xla --example backend_bench -- \
//!     --bench matmul --backend cuda --dtype bf16
//!
//! Workloads:
//! - matmul:        C = A x B^T with A=[M, K], B=[N, K] (decode-shaped gemm)
//! - llama-prefill: SmolLM-135M forward over a fixed-length prompt, no cache
//!                  (static shapes: the xla executable cache gets hits)
//! - llama-decode:  SmolLM-135M autoregressive decode with a growing kv-cache
//!                  (shapes change every step: worst case for eager xla)
//! - mimi-encode:   Mimi audio tokenizer, streaming encode of 1920-sample
//!                  chunks (static shapes per step)

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::time::Instant;
use xn::models::llama::{Config, KvCache, Llama};
use xn::nn::VB;
use xn::{Backend, Tensor};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Bench {
    Matmul,
    LlamaPrefill,
    LlamaDecode,
    MimiEncode,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendArg {
    Cuda,
    Xla,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DtypeArg {
    F32,
    F16,
    Bf16,
}

#[derive(Parser, Debug, Clone, Copy)]
#[command(about = "Benchmark native cuda vs xla backends")]
struct Args {
    #[arg(long, value_enum)]
    bench: Bench,

    #[arg(long, value_enum)]
    backend: BackendArg,

    #[arg(long, value_enum, default_value_t = DtypeArg::F32)]
    dtype: DtypeArg,

    /// Tokens to generate in llama-decode.
    #[arg(long, default_value_t = 64)]
    tokens: usize,

    /// Prompt length for llama-prefill.
    #[arg(long, default_value_t = 128)]
    prefill: usize,

    /// Iterations for matmul / llama-prefill / mimi chunks.
    #[arg(long, default_value_t = 200)]
    iters: usize,

    /// Preallocated kv-cache capacity for llama-decode (0 = growing cache).
    #[arg(long, default_value_t = 0)]
    kv_cap: usize,

    /// Warmup iterations.
    #[arg(long, default_value_t = 20)]
    warmup: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.backend {
        BackendArg::Cuda => {
            let dev = xn::cuda_backend::Device::new(0)?;
            unsafe {
                dev.disable_event_tracking();
            }
            println!("backend: cuda (native)");
            dispatch(args, dev)
        }
        BackendArg::Xla => {
            let dev = xn::xla_backend::Device::new(0)?;
            println!("backend: xla ({})", dev.platform_name());
            dispatch(args, dev)
        }
    }
}

fn dispatch<B: Backend>(args: Args, dev: B) -> Result<()> {
    match args.dtype {
        DtypeArg::F32 => run::<f32, B>(args, dev),
        DtypeArg::F16 => run::<half::f16, B>(args, dev),
        DtypeArg::Bf16 => run::<half::bf16, B>(args, dev),
    }
}

fn run<T: xn::WithDTypeF, B: Backend>(args: Args, dev: B) -> Result<()> {
    println!("dtype: {:?}", T::DTYPE);
    match args.bench {
        Bench::Matmul => bench_matmul::<T, B>(args, dev),
        Bench::LlamaPrefill => bench_llama_prefill::<T, B>(args, dev),
        Bench::LlamaDecode => bench_llama_decode::<T, B>(args, dev),
        Bench::MimiEncode => bench_mimi_encode(args, dev),
    }
}

fn random_vec<T: xn::WithDTypeF>(n: usize, seed: u64) -> Vec<T> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            T::from_f32((s & 0xFFFF) as f32 / 65535.0 * 2.0 - 1.0)
        })
        .collect()
}

fn stats(times: &[f64]) -> (f64, f64, f64) {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    (median, min, max)
}

// =========================================================================
// matmul
// =========================================================================

fn bench_matmul<T: xn::WithDTypeF, B: Backend>(args: Args, dev: B) -> Result<()> {
    const M: usize = 32;
    const K: usize = 2048;
    const N: usize = 11264;
    println!("matmul: A=[{M}, {K}] x B^T=[{N}, {K}] -> C=[{M}, {N}]");

    let a: Tensor<T, B> = Tensor::from_vec(random_vec::<T>(M * K, 42), (M, K), &dev)?;
    let b: Tensor<T, B> = Tensor::from_vec(random_vec::<T>(N * K, 123), (N, K), &dev)?;

    // Synchronize inside the loop: with a lazy backend, dropping the result
    // without a sync would skip the computation entirely.
    for _ in 0..args.warmup {
        let c = a.matmul_t(&b)?;
        dev.synchronize()?;
        drop(c);
    }

    let t0 = Instant::now();
    for _ in 0..args.iters {
        let c = a.matmul_t(&b)?;
        dev.synchronize()?;
        drop(c);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let us = elapsed * 1e6 / args.iters as f64;
    let tflops = 2.0 * (M * N * K) as f64 * args.iters as f64 / elapsed / 1e12;
    println!("{} iters | {us:.1} us/iter | {tflops:.2} TFLOP/s", args.iters);
    Ok(())
}

// =========================================================================
// llama
// =========================================================================

fn smol_lm_weights() -> Result<std::path::PathBuf> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    let api = Api::new()?;
    let repo = api.repo(Repo::new("HuggingFaceTB/SmolLM-135M".to_string(), RepoType::Model));
    repo.get("model.safetensors").context("model.safetensors not found")
}

fn load_smol_lm<T: xn::WithDTypeF, B: Backend>(dev: &B) -> Result<Llama<T, B>> {
    let path = smol_lm_weights()?;
    let vb = VB::load(&[path], dev.clone())?;
    let model = Llama::load(&vb.root(), &Config::smol_lm_135m())?;
    Ok(model)
}

fn bench_llama_prefill<T: xn::WithDTypeF, B: Backend>(args: Args, dev: B) -> Result<()> {
    println!("llama-prefill: SmolLM-135M, prompt len {}, no kv-cache", args.prefill);
    let model = load_smol_lm::<T, B>(&dev)?;
    let tokens: Vec<u32> = (0..args.prefill).map(|i| 1000 + (i as u32 % 1000)).collect();

    for _ in 0..args.warmup.min(5) {
        let (_logits, _kv) = model.forward(&tokens, 0, None)?;
        dev.synchronize()?;
    }

    let mut times = Vec::with_capacity(args.iters);
    let t_all = Instant::now();
    for _ in 0..args.iters {
        let t0 = Instant::now();
        let (_logits, _kv) = model.forward(&tokens, 0, None)?;
        dev.synchronize()?;
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let elapsed = t_all.elapsed().as_secs_f64();
    let (median, min, max) = stats(&times);
    let tok_s = args.prefill as f64 * args.iters as f64 / elapsed;
    println!(
        "{} iters | median {median:.2} ms | min {min:.2} ms | max {max:.2} ms | {tok_s:.0} tok/s",
        args.iters
    );
    Ok(())
}

fn bench_llama_decode<T: xn::WithDTypeF, B: Backend>(args: Args, dev: B) -> Result<()> {
    println!(
        "llama-decode: SmolLM-135M, 16-token prompt, {} decode steps, kv-cache: {}",
        args.tokens,
        if args.kv_cap == 0 { "growing".to_string() } else { format!("fixed[{}]", args.kv_cap) },
    );
    let model = load_smol_lm::<T, B>(&dev)?;
    let prompt: Vec<u32> = (0..16).map(|i| 1000 + i as u32).collect();

    let mut kv_cache: Option<KvCache<T, B>> = if args.kv_cap == 0 {
        None
    } else {
        anyhow::ensure!(16 + args.tokens + 1 <= args.kv_cap, "kv-cap too small");
        Some(KvCache::fixed(&Config::smol_lm_135m(), 1, args.kv_cap, &dev)?)
    };
    let mut prefill_done = false;
    let mut generated: Vec<u32> = Vec::new();
    let mut pos = 0;
    let mut last_token = *prompt.last().unwrap();
    let mut prefill_ms = 0f64;
    let mut times = Vec::with_capacity(args.tokens);

    for step in 0..=args.tokens {
        let input: Vec<u32> = if !prefill_done { prompt.clone() } else { vec![last_token] };
        prefill_done = true;
        let t0 = Instant::now();
        let (logits, new_kv) = model.forward(&input, pos, kv_cache.as_ref())?;
        // Greedy sampling on the host; the transfer also acts as a sync point.
        let logits = logits.to::<f32>()?.to_vec()?;
        let elapsed = t0.elapsed().as_secs_f64() * 1e3;
        kv_cache = Some(new_kv);
        pos += input.len();
        let vocab = logits.len() / input.len();
        let last = &logits[logits.len() - vocab..];
        last_token = last
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        generated.push(last_token);
        if step == 0 {
            prefill_ms = elapsed;
        } else {
            times.push(elapsed);
        }
    }

    println!("first tokens: {:?}", &generated[..generated.len().min(8)]);
    let (median, min, max) = stats(&times);
    let total: f64 = times.iter().sum();
    let tok_s = times.len() as f64 / (total / 1e3);
    println!(
        "prefill {prefill_ms:.1} ms | {} steps | median {median:.2} ms | min {min:.2} ms | max {max:.2} ms | {tok_s:.1} tok/s",
        times.len()
    );
    Ok(())
}

// =========================================================================
// mimi
// =========================================================================

fn bench_mimi_encode<B: Backend>(args: Args, dev: B) -> Result<()> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    use xn::models::mimi::{Config as MimiConfig, Mimi, StreamMask, StreamTensor};

    println!("mimi-encode: streaming encode, 1920-sample chunks (f32 model)");
    let api = Api::new()?;
    let repo = api.repo(Repo::new("kyutai/moshiko-candle-q8".to_string(), RepoType::Model));
    let model_path = repo
        .get("tokenizer-e351c8d8-checkpoint125.safetensors")
        .context("mimi weights not found")?;
    let vb = VB::load(&[model_path], dev.clone())?;
    let config = MimiConfig::v0_1_no_weight_norm(Some(8));
    let mut model: Mimi<f32, B> = Mimi::load(&vb.root(), config, &dev)?;
    model.reset_state();
    let mask = StreamMask::empty();

    const CHUNK: usize = 1920;
    let pcm = random_vec::<f32>(CHUNK, 42);

    // The streaming transformer inside mimi keeps a kv-cache that grows until
    // it reaches its fixed context (250 frames), so shapes only stabilize
    // after ~250 chunks: use `--warmup 260` or more to measure steady state.
    let mut times = Vec::with_capacity(args.iters);
    for it in 0..args.warmup + args.iters {
        let audio: Tensor<f32, B> = Tensor::from_vec(pcm.clone(), (1, 1, CHUNK), &dev)?;
        let t0 = Instant::now();
        let codes = model.encode_step(&StreamTensor::from_tensor(audio), &mask)?;
        if let Some(codes) = codes.as_option() {
            let _v = codes.to_vec()?;
        }
        dev.synchronize()?;
        if it >= args.warmup {
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }

    let (median, min, max) = stats(&times);
    // Each 1920-sample chunk is 80ms of 24kHz audio.
    println!(
        "{} chunks | median {median:.2} ms | min {min:.2} ms | max {max:.2} ms | rtf {:.1}x",
        times.len(),
        80.0 / median,
    );
    Ok(())
}
