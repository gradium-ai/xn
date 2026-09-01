//! Runs the WebGPU backend in a browser and checks its results.
//!
//! The shader-level check next door compiles every kernel with the browser's WGSL
//! implementation; this exercises the backend itself -- device creation, the
//! uniform parameter ring, batching, the buffer pool, and the async readback --
//! against expected values computed on the spot.
//!
//! Ops only record into the pending batch, so they run through the ordinary
//! synchronous `Backend` trait here exactly as they do natively. Only the
//! readbacks are awaited, via `Device::tensor_to_vec`.

use half::f16;
use xn::webgpu_backend::Device;
use xn::{Tensor, WithDTypeF};

/// One check's outcome, serialised by hand to avoid a serde dependency.
struct Check {
    name: String,
    ok: bool,
    detail: String,
}

fn json_escape(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect(),
            '\n' => "\\n".chars().collect(),
            c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32).chars().collect(),
            c => vec![c],
        })
        .collect()
}

/// Deterministic xorshift64* in `[lo, hi)`, rounded through f16 so the f16 and
/// f32 runs see identical inputs.
fn rnd(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x1234_5678);
    if s == 0 {
        s = 0xDEAD_BEEF;
    }
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = (s.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32;
            f16::from_f32(lo + u * (hi - lo)).to_f32()
        })
        .collect()
}

fn close(a: &[f32], b: &[f32], rtol: f32, atol: f32) -> Option<String> {
    if a.len() != b.len() {
        return Some(format!("length {} vs {}", a.len(), b.len()));
    }
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x == y || (x.is_nan() && y.is_nan()) {
            continue;
        }
        if (x - y).abs() > atol + rtol * x.abs().max(y.abs()) {
            return Some(format!("at {i}: expected {x}, got {y}"));
        }
    }
    None
}

fn tt<T: WithDTypeF>(dev: &Device, d: &[f32], shape: &[usize]) -> Tensor<T, Device> {
    Tensor::from_vec(d.iter().map(|&v| T::from_f32(v)).collect(), shape.to_vec(), dev).unwrap()
}

/// Runs every check for one dtype and appends to `out`.
async fn run_dtype<T: WithDTypeF>(dev: &Device, tag: &str, out: &mut Vec<Check>) {
    let rtol = if tag == "f16" { 6e-3 } else { 2e-5 };
    let atol = if tag == "f16" { 3e-3 } else { 1e-6 };

    macro_rules! check {
        ($name:expr, $want:expr, $got:expr) => {{
            let want: Vec<f32> = $want;
            let got: Result<Vec<T>, _> = $got;
            let c = match got {
                Err(e) => Check {
                    name: format!("{} {}", tag, $name),
                    ok: false,
                    detail: format!("error: {e}"),
                },
                Ok(v) => {
                    let v: Vec<f32> = v.into_iter().map(|x| x.to_f32()).collect();
                    match close(&want, &v, rtol, atol) {
                        None => Check {
                            name: format!("{} {}", tag, $name),
                            ok: true,
                            detail: String::new(),
                        },
                        Some(d) => Check {
                            name: format!("{} {}", tag, $name),
                            ok: false,
                            detail: d,
                        },
                    }
                }
            };
            out.push(c);
        }};
    }

    // Roundtrip: upload then read back. An odd length is not a multiple of 4
    // bytes in f16, which every WebGPU write and copy must be.
    let d = rnd(1, 1001, -4.0, 4.0);
    let t = tt::<T>(dev, &d, &[1001]);
    check!("roundtrip odd len", d.clone(), dev.tensor_to_vec(&t).await);

    // Elementwise, exercising the shared ops library and the uniform ring.
    let a = rnd(2, 257, -3.0, 3.0);
    let b = rnd(3, 257, 0.5, 3.0);
    let ta = tt::<T>(dev, &a, &[257]);
    let tb = tt::<T>(dev, &b, &[257]);
    check!(
        "add",
        a.iter().zip(&b).map(|(x, y)| x + y).collect(),
        dev.tensor_to_vec(&ta.add(&tb).unwrap()).await
    );
    check!(
        "mul",
        a.iter().zip(&b).map(|(x, y)| x * y).collect(),
        dev.tensor_to_vec(&ta.mul(&tb).unwrap()).await
    );
    check!(
        "silu",
        a.iter().map(|x| x / (1.0 + (-x).exp())).collect(),
        dev.tensor_to_vec(&ta.silu().unwrap()).await
    );
    check!(
        "relu",
        a.iter().map(|x| x.max(0.0)).collect(),
        dev.tensor_to_vec(&ta.relu().unwrap()).await
    );
    // bin_assign, one of the two ops that used to alias a writable buffer. The
    // other, `Backend::inplace_unary`, is unreachable from here: `UnaryOp` is
    // `pub(crate)` in xn even though the public trait takes it, and nothing in
    // xn's own Tensor API routes to that method for any backend.
    let acc = tt::<T>(dev, &a, &[257]);
    acc.inplace_add(&tb).unwrap();
    check!(
        "inplace_add",
        a.iter().zip(&b).map(|(x, y)| x + y).collect(),
        dev.tensor_to_vec(&acc).await
    );

    // Broadcast, the shape a bias add takes.
    let bias = rnd(4, 16, -1.0, 1.0);
    let rows = rnd(5, 4 * 16, -2.0, 2.0);
    let trows = tt::<T>(dev, &rows, &[4, 16]);
    let tbias = tt::<T>(dev, &bias, &[16]);
    check!(
        "broadcast_add",
        (0..4 * 16).map(|i| rows[i] + bias[i % 16]).collect(),
        dev.tensor_to_vec(&trows.broadcast_add(&tbias).unwrap()).await
    );

    // Row reductions with a workgroup barrier tree.
    let sm = rnd(6, 2 * 300, -3.0, 3.0);
    let tsm = tt::<T>(dev, &sm, &[2, 300]);
    let mut want_sm = Vec::with_capacity(600);
    for r in 0..2 {
        let row = &sm[r * 300..(r + 1) * 300];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ex: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
        let s: f32 = ex.iter().sum();
        want_sm.extend(ex.iter().map(|v| v / s));
    }
    check!("softmax", want_sm, dev.tensor_to_vec(&tsm.softmax().unwrap()).await);

    // Matmul: all three kernels. k < 1024 is gemv_tpc, k >= 1024 the cooperative
    // gemv, m > 1 the tiled one.
    for &(m, k, n) in &[(1usize, 768usize, 512usize), (1, 2048, 64), (16, 128, 96)] {
        let x = rnd(7, m * k, -1.0, 1.0);
        let w = rnd(8, n * k, -1.0, 1.0);
        let tx = tt::<T>(dev, &x, &[m, k]);
        let tw = tt::<T>(dev, &w, &[n, k]);
        let mut want = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                want.push((0..k).map(|l| x[i * k + l] * w[j * k + l]).sum::<f32>());
            }
        }
        // A dot product sums k terms in a different order than the reference,
        // and with mixed signs the result can be orders of magnitude smaller than
        // the terms, so the tolerance has to be absolute and scaled by k rather
        // than relative to a near-cancelling sum. f16 additionally rounds at the
        // store.
        let scale = (k as f32).sqrt();
        let (rtol, atol) =
            if tag == "f16" { (3e-2, 3e-2) } else { (rtol, 1e-6 * scale.max(1.0)) };
        let got = dev.tensor_to_vec(&tx.matmul_t(&tw).unwrap()).await;
        let c = match got {
            Err(e) => Check {
                name: format!("{tag} matmul_t {m}x{k}x{n}"),
                ok: false,
                detail: format!("error: {e}"),
            },
            Ok(v) => {
                let v: Vec<f32> = v.into_iter().map(|x| x.to_f32()).collect();
                match close(&want, &v, rtol, atol) {
                    None => Check {
                        name: format!("{tag} matmul_t {m}x{k}x{n}"),
                        ok: true,
                        detail: String::new(),
                    },
                    Some(d) => Check {
                        name: format!("{tag} matmul_t {m}x{k}x{n}"),
                        ok: false,
                        detail: d,
                    },
                }
            }
        };
        out.push(c);
    }

    // Layout movement, which is exact.
    let mv = rnd(9, 2 * 3 * 4, -5.0, 5.0);
    let tmv = tt::<T>(dev, &mv, &[2, 3, 4]);
    let mut want_t = Vec::with_capacity(24);
    for i in 0..2 {
        for k in 0..4 {
            for j in 0..3 {
                want_t.push(mv[i * 12 + j * 4 + k]);
            }
        }
    }
    check!(
        "transpose 1,2",
        want_t,
        dev.tensor_to_vec(&tmv.transpose(1, 2).unwrap().contiguous().unwrap()).await
    );

    // A long batch, to push the uniform parameter ring past one slot and force
    // the flush path that reuses it.
    let mut chain = tt::<T>(dev, &vec![1.0f32; 64], &[64]);
    for _ in 0..200 {
        chain = chain.add(&tt::<T>(dev, &vec![0.5f32; 64], &[64])).unwrap();
    }
    check!("200-op chain", vec![101.0f32; 64], dev.tensor_to_vec(&chain).await);
}

#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_all() -> String {
    console_error_panic_hook::set_once();
    let mut out: Vec<Check> = Vec::new();
    let mut header = String::new();

    match Device::new_async(0).await {
        Err(e) => {
            header = format!("\"deviceError\": \"{}\",", json_escape(&format!("{e}")));
        }
        Ok(dev) => {
            let name = <Device as xn::Backend>::name(&dev);
            header = format!(
                "\"device\": \"{}\", \"f16\": {},",
                json_escape(&name),
                dev.supports_f16()
            );
            run_dtype::<f32>(&dev, "f32", &mut out).await;
            if dev.supports_f16() {
                run_dtype::<f16>(&dev, "f16", &mut out).await;
            }
            // The blocking readback must fail loudly rather than deadlock.
            let t = tt::<f32>(&dev, &[1.0, 2.0], &[2]);
            let blocked = t.to_vec();
            out.push(Check {
                name: "blocking to_vec refused".into(),
                ok: blocked.is_err(),
                detail: match blocked {
                    Err(_) => String::new(),
                    Ok(_) => "expected an error, got values".into(),
                },
            });
        }
    }

    let failed = out.iter().filter(|c| !c.ok).count();
    let results: Vec<String> = out
        .iter()
        .filter(|c| !c.ok)
        .map(|c| format!("{{\"name\": \"{}\", \"detail\": \"{}\"}}", json_escape(&c.name), json_escape(&c.detail)))
        .collect();
    format!(
        "{{{header} \"total\": {}, \"failed\": {}, \"failures\": [{}]}}",
        out.len(),
        failed,
        results.join(", ")
    )
}
