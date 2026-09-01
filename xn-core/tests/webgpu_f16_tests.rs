#![cfg(feature = "webgpu")]
//! f16 equivalence tests for the WebGPU backend.
//!
//! Each op runs twice from the same inputs: f32 on the CPU (the oracle) and f16
//! on WebGPU. Inputs are rounded through f16 *before* either side sees them, so
//! both backends compute on numerically identical values and the only expected
//! divergence is f16 rounding of the outputs and of intermediate storage --
//! arithmetic inside the kernels is f32 either way. That keeps tolerances tight
//! enough to catch real bugs instead of hiding them behind f16 slack.
//!
//! These tests also serve as the compile check for the f16 kernel variants: WGSL
//! is compiled lazily on first dispatch, so an op with no test here could ship a
//! shader that fails to build.

use half::f16;
use xn::{Backend, CPU, Result, Tensor, WithDTypeF};

fn wg() -> xn::webgpu_backend::Device {
    use std::sync::OnceLock;
    static DEVICE: OnceLock<xn::webgpu_backend::Device> = OnceLock::new();
    DEVICE.get_or_init(|| xn::webgpu_backend::Device::new(0).expect("init webgpu device")).clone()
}

/// f16 compute is adapter-dependent. Where it is missing the backend falls back
/// to host loops, which these tests are not about, so they no-op instead of
/// reporting a pass that never ran.
fn f16_ready() -> bool {
    let ok = wg().supports_f16();
    if !ok {
        eprintln!("skipping: adapter does not advertise WGSL shader-f16");
    }
    ok
}

/// xorshift64* in [lo, hi), rounded through f16 so both backends see the same
/// values exactly.
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

fn cmp(label: &str, reference: &[f32], got: &[f32], rtol: f32, atol: f32) {
    assert_eq!(reference.len(), got.len(), "{label}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_i = usize::MAX;
    for (i, (&r, &g)) in reference.iter().zip(got).enumerate() {
        if r == g || (r.is_nan() && g.is_nan()) {
            continue;
        }
        let over = (r - g).abs() - (atol + rtol * r.abs().max(g.abs()));
        if over > worst {
            worst = over;
            worst_i = i;
        }
    }
    if worst_i != usize::MAX {
        let (r, g) = (reference[worst_i], got[worst_i]);
        panic!(
            "{label}: mismatch at {worst_i}: cpu_f32={r} wgpu_f16={g} |d|={} (over tol by {worst})",
            (r - g).abs()
        );
    }
}

/// Run a dtype-generic runner on CPU/f32 and WebGPU/f16 and compare.
macro_rules! verify16 {
    // `$run` is an `ident`, not a `path`: a parsed path fragment cannot take a
    // turbofish afterwards, and these runners are generic over the dtype.
    ($label:expr, $rtol:expr, $atol:expr, $run:ident $(, $arg:expr)* $(,)?) => {{
        let reference = $run::<f32, _>(&CPU $(, $arg)*);
        let got = $run::<f16, _>(&wg() $(, $arg)*);
        cmp($label, &reference, &got, $rtol, $atol);
    }};
}

fn tt<T: WithDTypeF, B: Backend>(dev: &B, data: &[f32], shape: &[usize]) -> Tensor<T, B> {
    let v: Vec<T> = data.iter().map(|&x| T::from_f32(x)).collect();
    Tensor::from_vec(v, shape.to_vec(), dev).unwrap()
}

fn out<T: WithDTypeF, B: Backend>(t: Tensor<T, B>) -> Vec<f32> {
    t.to_vec().unwrap().into_iter().map(|v| <T as WithDTypeF>::to_f32(v)).collect()
}

// -----------------------------------------------------------------------------
// Elementwise
// -----------------------------------------------------------------------------

macro_rules! def_unary {
    ($fname:ident, $m:ident) => {
        fn $fname<T: WithDTypeF, B: Backend>(dev: &B, d: &[f32], sh: &[usize]) -> Vec<f32> {
            out(tt::<T, B>(dev, d, sh).$m().unwrap())
        }
    };
}
def_unary!(run_relu, relu);
def_unary!(run_silu, silu);
def_unary!(run_sqr, sqr);
def_unary!(run_sqrt, sqrt);
def_unary!(run_exp, exp);
def_unary!(run_log, log);
def_unary!(run_abs, abs);
def_unary!(run_neg, neg);
def_unary!(run_gelu, gelu_erf);
def_unary!(run_tanh, tanh);
def_unary!(run_sigmoid, sigmoid);
def_unary!(run_cos, cos);
def_unary!(run_sin, sin);
def_unary!(run_rsqrt, rsqrt);

fn run_elu<T: WithDTypeF, B: Backend>(dev: &B, d: &[f32], sh: &[usize]) -> Vec<f32> {
    out(tt::<T, B>(dev, d, sh).elu(1.0).unwrap())
}

const SHAPES_1D: &[&[usize]] = &[&[1], &[7], &[255], &[256], &[257], &[1000], &[3, 5, 7]];

#[test]
fn unary_ops() -> Result<()> {
    if !f16_ready() {
        return Ok(());
    }
    for (i, &sh) in SHAPES_1D.iter().enumerate() {
        let n: usize = sh.iter().product();
        let pos = rnd(i as u64 + 1, n, 0.25, 4.0);
        let any = rnd(i as u64 + 90, n, -4.0, 4.0);
        // f16 has ~3 decimal digits, and the final store rounds; 3e-3 relative
        // is a few ulp at these magnitudes.
        let (r, a) = (3e-3f32, 1e-3f32);
        verify16!("relu", r, a, run_relu, &any, sh);
        verify16!("silu", r, a, run_silu, &any, sh);
        verify16!("sqr", r, a, run_sqr, &any, sh);
        verify16!("neg", r, a, run_neg, &any, sh);
        verify16!("abs", r, a, run_abs, &any, sh);
        verify16!("gelu", r, a, run_gelu, &any, sh);
        verify16!("tanh", r, a, run_tanh, &any, sh);
        verify16!("sigmoid", r, a, run_sigmoid, &any, sh);
        verify16!("cos", r, a, run_cos, &any, sh);
        verify16!("sin", r, a, run_sin, &any, sh);
        verify16!("elu", r, a, run_elu, &any, sh);
        verify16!("sqrt", r, a, run_sqrt, &pos, sh);
        verify16!("log", r, a, run_log, &pos, sh);
        verify16!("exp", r, a, run_exp, &pos, sh);
        verify16!("rsqrt", r, a, run_rsqrt, &pos, sh);
    }
    Ok(())
}

fn run_bin<T: WithDTypeF, B: Backend>(
    dev: &B,
    a: &[f32],
    b: &[f32],
    sh: &[usize],
    op: &str,
) -> Vec<f32> {
    let (x, y) = (tt::<T, B>(dev, a, sh), tt::<T, B>(dev, b, sh));
    let r = match op {
        "add" => x.add(&y),
        "sub" => x.sub(&y),
        "mul" => x.mul(&y),
        "div" => x.div(&y),
        "max" => x.maximum(&y),
        "min" => x.minimum(&y),
        _ => unreachable!(),
    };
    out(r.unwrap())
}

fn run_scale<T: WithDTypeF, B: Backend>(dev: &B, a: &[f32], sh: &[usize], s: f32) -> Vec<f32> {
    out(tt::<T, B>(dev, a, sh).scale(T::from_f32(s)).unwrap())
}

#[test]
fn binary_ops() -> Result<()> {
    if !f16_ready() {
        return Ok(());
    }
    for (i, &sh) in SHAPES_1D.iter().enumerate() {
        let n: usize = sh.iter().product();
        let a = rnd(i as u64 + 1, n, -4.0, 4.0);
        let b = rnd(i as u64 + 500, n, 0.5, 4.0);
        for op in ["add", "sub", "mul", "div", "max", "min"] {
            verify16!("binary", 3e-3, 1e-3, run_bin, &a, &b, sh, op);
        }
        verify16!("scale", 3e-3, 1e-3, run_scale, &a, sh, 3.25);
    }
    Ok(())
}

fn run_bcast<T: WithDTypeF, B: Backend>(
    dev: &B,
    a: &[f32],
    sa: &[usize],
    b: &[f32],
    sb: &[usize],
    op: &str,
) -> Vec<f32> {
    let (x, y) = (tt::<T, B>(dev, a, sa), tt::<T, B>(dev, b, sb));
    let r = match op {
        "add" => x.broadcast_add(&y),
        "mul" => x.broadcast_mul(&y),
        "sub" => x.broadcast_sub(&y),
        _ => unreachable!(),
    };
    out(r.unwrap())
}

#[test]
fn broadcast_ops() -> Result<()> {
    if !f16_ready() {
        return Ok(());
    }
    // Bias-add shaped cases, which is how the model uses broadcast.
    let cases: &[(&[usize], &[usize])] =
        &[(&[4, 5], &[5]), (&[1, 1, 768], &[768]), (&[2, 3, 4], &[3, 4]), (&[7, 1, 9], &[9])];
    for (i, &(sa, sb)) in cases.iter().enumerate() {
        let a = rnd(i as u64 + 3, sa.iter().product(), -3.0, 3.0);
        let b = rnd(i as u64 + 77, sb.iter().product(), 0.5, 3.0);
        for op in ["add", "mul", "sub"] {
            verify16!("broadcast", 3e-3, 1e-3, run_bcast, &a, sa, &b, sb, op);
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Norms and softmax (workgroup reductions, f32 accumulation)
// -----------------------------------------------------------------------------

fn run_softmax<T: WithDTypeF, B: Backend>(dev: &B, d: &[f32], sh: &[usize]) -> Vec<f32> {
    out(tt::<T, B>(dev, d, sh).softmax().unwrap())
}
fn run_rmsnorm<T: WithDTypeF, B: Backend>(
    dev: &B,
    d: &[f32],
    w: &[f32],
    sh: &[usize],
    eps: f32,
) -> Vec<f32> {
    let ncols = *sh.last().unwrap();
    let wt = tt::<T, B>(dev, w, &[ncols]);
    out(tt::<T, B>(dev, d, sh).rms_norm(&wt, eps).unwrap())
}
fn run_layernorm<T: WithDTypeF, B: Backend>(
    dev: &B,
    d: &[f32],
    w: &[f32],
    b: &[f32],
    sh: &[usize],
    eps: f32,
) -> Vec<f32> {
    let ncols = *sh.last().unwrap();
    let (wt, bt) = (tt::<T, B>(dev, w, &[ncols]), tt::<T, B>(dev, b, &[ncols]));
    out(tt::<T, B>(dev, d, sh).layer_norm(&wt, &bt, eps).unwrap())
}

#[test]
fn softmax_and_norms() -> Result<()> {
    if !f16_ready() {
        return Ok(());
    }
    // Row widths straddling the 256-lane workgroup boundary.
    let shapes: &[&[usize]] = &[&[1, 7], &[3, 255], &[3, 256], &[3, 257], &[2, 768], &[1, 2048]];
    for (i, &sh) in shapes.iter().enumerate() {
        let n: usize = sh.iter().product();
        let ncols = *sh.last().unwrap();
        let d = rnd(i as u64 + 11, n, -3.0, 3.0);
        let w = rnd(i as u64 + 22, ncols, 0.5, 1.5);
        let b = rnd(i as u64 + 33, ncols, -0.5, 0.5);
        // softmax outputs live in [0, 1], so absolute tolerance carries it.
        verify16!("softmax", 3e-3, 1e-3, run_softmax, &d, sh);
        verify16!("rmsnorm", 6e-3, 2e-3, run_rmsnorm, &d, &w, sh, 1e-5);
        verify16!("layernorm", 6e-3, 2e-3, run_layernorm, &d, &w, &b, sh, 1e-5);
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Reductions
// -----------------------------------------------------------------------------

fn run_reduce<T: WithDTypeF, B: Backend>(
    dev: &B,
    d: &[f32],
    sh: &[usize],
    dim: usize,
    op: &str,
) -> Vec<f32> {
    let x = tt::<T, B>(dev, d, sh);
    let r = match op {
        "max" => x.max(dim),
        "min" => x.min(dim),
        "sum" => x.sum_keepdim(vec![dim]),
        _ => unreachable!(),
    };
    out(r.unwrap())
}

/// Arg-reductions return indices, so this compares them exactly. Inputs are
/// f16-representable and drawn to avoid exact ties, so both backends must agree.
fn run_argreduce<T: WithDTypeF, B: Backend>(
    dev: &B,
    d: &[f32],
    sh: &[usize],
    dim: usize,
    max: bool,
) -> Vec<f32> {
    let x = tt::<T, B>(dev, d, sh);
    let r = if max { x.argmax(dim) } else { x.argmin(dim) };
    r.unwrap().to_vec().unwrap().into_iter().map(|v| v as f32).collect()
}

#[test]
fn reductions() -> Result<()> {
    if !f16_ready() {
        return Ok(());
    }
    let shapes: &[&[usize]] = &[&[3, 255], &[3, 256], &[3, 257], &[2, 3, 400], &[2, 768]];
    for (i, &sh) in shapes.iter().enumerate() {
        let n: usize = sh.iter().product();
        // Sums of ~800 f16 values in an f32 accumulator: the store back to f16
        // is the dominant error, but magnitudes grow, so lean on relative.
        let d = rnd(i as u64 + 44, n, -2.0, 2.0);
        for dim in 0..sh.len() {
            for op in ["max", "min", "sum"] {
                verify16!("reduce", 8e-3, 2e-3, run_reduce, &d, sh, dim, op);
            }
            verify16!("argmax", 0.0, 0.0, run_argreduce, &d, sh, dim, true);
            verify16!("argmin", 0.0, 0.0, run_argreduce, &d, sh, dim, false);
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Layout / movement
// -----------------------------------------------------------------------------

fn run_transpose<T: WithDTypeF, B: Backend>(
    dev: &B,
    d: &[f32],
    sh: &[usize],
    d1: usize,
    d2: usize,
) -> Vec<f32> {
    out(tt::<T, B>(dev, d, sh).transpose(d1, d2).unwrap().contiguous().unwrap())
}

fn run_cat_narrow<T: WithDTypeF, B: Backend>(
    dev: &B,
    a: &[f32],
    b: &[f32],
    sa: &[usize],
    sb: &[usize],
    dim: usize,
    lo: usize,
    hi: usize,
) -> Vec<f32> {
    let x = tt::<T, B>(dev, a, sa);
    let y = tt::<T, B>(dev, b, sb);
    let c = Tensor::cat(&[&x, &y], dim).unwrap();
    out(c.narrow(dim, lo..hi).unwrap().contiguous().unwrap())
}

fn run_index_select<T: WithDTypeF, B: Backend>(
    dev: &B,
    d: &[f32],
    sh: &[usize],
    ids: &[i64],
    dim: usize,
) -> Vec<f32> {
    let x = tt::<T, B>(dev, d, sh);
    let iv = Tensor::<i64, B>::from_vec(ids.to_vec(), vec![ids.len()], dev).unwrap();
    out(x.index_select(&iv, dim).unwrap())
}

fn run_scatter<T: WithDTypeF, B: Backend>(
    dev: &B,
    dst: &[f32],
    sd: &[usize],
    src: &[f32],
    ss: &[usize],
    ids: &[i64],
    dim: usize,
) -> Vec<f32> {
    let d = tt::<T, B>(dev, dst, sd);
    let s = tt::<T, B>(dev, src, ss);
    let iv = Tensor::<i64, B>::from_vec(ids.to_vec(), ss.to_vec(), dev).unwrap();
    out(d.scatter(&iv, &s, dim).unwrap())
}

#[test]
fn movement_ops() -> Result<()> {
    if !f16_ready() {
        return Ok(());
    }
    // Data movement is exact: S -> S with no conversion.
    let d = rnd(7, 2 * 3 * 4 * 5, -5.0, 5.0);
    let sh: &[usize] = &[2, 3, 4, 5];
    for d1 in 0..4 {
        for d2 in 0..4 {
            verify16!("transpose", 0.0, 0.0, run_transpose, &d, sh, d1, d2);
        }
    }
    // The decode-shaped transpose: swapping an extent-1 dim must stay exact.
    let q = rnd(8, 12 * 64, -3.0, 3.0);
    verify16!("transpose bthd", 0.0, 0.0, run_transpose, &q, &[1, 1, 12, 64], 1, 2);

    let a = rnd(9, 24, -3.0, 3.0);
    let b = rnd(10, 24, -3.0, 3.0);
    verify16!("cat/narrow", 0.0, 0.0, run_cat_narrow, &a, &b, &[2, 3, 4], &[2, 3, 4], 2, 1, 6);
    verify16!("cat/narrow d1", 0.0, 0.0, run_cat_narrow, &a, &b, &[2, 3, 4], &[2, 3, 4], 1, 0, 5);

    let e = rnd(11, 5 * 7, -3.0, 3.0);
    verify16!("index_select", 0.0, 0.0, run_index_select, &e, &[5, 7], &[3i64, 0, 4, 1], 0);

    let dst = rnd(12, 12, -2.0, 2.0);
    let src = rnd(13, 6, -2.0, 2.0);
    verify16!(
        "scatter",
        0.0,
        0.0,
        run_scatter,
        &dst,
        &[3, 4],
        &src,
        &[3, 2],
        &[0i64, 2, 1, 3, 3, 0],
        1
    );
    Ok(())
}

// -----------------------------------------------------------------------------
// RoPE
// -----------------------------------------------------------------------------

fn run_rope<T: WithDTypeF, B: Backend>(
    dev: &B,
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    dims: (usize, usize, usize, usize),
    max_pos: usize,
    pos: usize,
    interleaved: bool,
) -> Vec<f32> {
    let (b, h, tl, d) = dims;
    let xt = tt::<T, B>(dev, x, &[b, h, tl, d]);
    let ct = tt::<T, B>(dev, cos, &[max_pos, d / 2]);
    let st = tt::<T, B>(dev, sin, &[max_pos, d / 2]);
    let r = if interleaved { xt.rope_i(&ct, &st, pos) } else { xt.rope(&ct, &st, pos) };
    out(r.unwrap())
}

#[test]
fn rope() -> Result<()> {
    if !f16_ready() {
        return Ok(());
    }
    // (b, h, t, d) including the decode shape the model runs.
    let cases: &[(usize, usize, usize, usize)] = &[(1, 1, 1, 64), (1, 12, 1, 64), (2, 3, 5, 32)];
    for (i, &(b, h, tl, d)) in cases.iter().enumerate() {
        let max_pos = tl + 8;
        let x = rnd(i as u64 + 55, b * h * tl * d, -3.0, 3.0);
        let cos = rnd(i as u64 + 66, max_pos * d / 2, -1.0, 1.0);
        let sin = rnd(i as u64 + 67, max_pos * d / 2, -1.0, 1.0);
        for pos in [0usize, 3] {
            verify16!(
                "rope",
                4e-3,
                2e-3,
                run_rope,
                &x,
                &cos,
                &sin,
                (b, h, tl, d),
                max_pos,
                pos,
                false
            );
            verify16!(
                "rope_i",
                4e-3,
                2e-3,
                run_rope,
                &x,
                &cos,
                &sin,
                (b, h, tl, d),
                max_pos,
                pos,
                true
            );
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Matmul: all three kernels (gemv_tpc for k < 1024, gemv above, gemm_tiled m > 1)
// -----------------------------------------------------------------------------

fn run_matmul<T: WithDTypeF, B: Backend>(
    dev: &B,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    transpose_rhs: bool,
) -> Vec<f32> {
    let (sa, sb) = if batch == 1 {
        (vec![m, k], if transpose_rhs { vec![n, k] } else { vec![k, n] })
    } else {
        (vec![batch, m, k], if transpose_rhs { vec![batch, n, k] } else { vec![batch, k, n] })
    };
    let (x, y) = (tt::<T, B>(dev, a, &sa), tt::<T, B>(dev, b, &sb));
    let r = if transpose_rhs { x.matmul_t(&y) } else { x.matmul(&y) };
    out(r.unwrap())
}

#[test]
fn matmul_shapes() -> Result<()> {
    if !f16_ready() {
        return Ok(());
    }
    // (m, k, n, batch). m == 1 with k < 1024 hits gemv_tpc; k >= 1024 hits the
    // cooperative gemv; m > 1 hits gemm_tiled. Non-tile-aligned dims included.
    let cases: &[(usize, usize, usize, usize)] = &[
        (1, 32, 512, 1),
        (1, 256, 512, 1),
        (1, 768, 768, 1),
        (1, 768, 2304, 1),
        (1, 1024, 512, 1),
        (1, 2048, 512, 1),
        (1, 3072, 768, 1),
        (1, 100, 37, 1),
        (16, 512, 512, 1),
        (16, 3584, 512, 1),
        (17, 33, 65, 1),
        (96, 768, 128, 1),
        (4, 64, 64, 3),
        (1, 64, 64, 4),
    ];
    for (i, &(m, k, n, batch)) in cases.iter().enumerate() {
        // Small magnitudes keep an f32 accumulator over k terms well inside f16
        // range at the store, so the tolerance reflects rounding, not overflow.
        let a = rnd(i as u64 + 101, batch * m * k, -1.0, 1.0);
        let b = rnd(i as u64 + 202, batch * k * n, -1.0, 1.0);
        for t_rhs in [true, false] {
            // A k-term dot product of f16 inputs rounds once at the store; the
            // f32 accumulation means error grows far slower than sqrt(k).
            verify16!("matmul", 2e-2, 2e-2, run_matmul, &a, &b, m, k, n, batch, t_rhs);
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Convolutions: im2col route, direct route, and transposed
// -----------------------------------------------------------------------------

fn run_conv1d<T: WithDTypeF, B: Backend>(
    dev: &B,
    src: &[f32],
    kern: &[f32],
    b: usize,
    ic: usize,
    oc: usize,
    len: usize,
    ks: usize,
    stride: usize,
    pad: usize,
    dil: usize,
    groups: usize,
) -> Vec<f32> {
    let s = tt::<T, B>(dev, src, &[b, ic, len]);
    let k = tt::<T, B>(dev, kern, &[oc, ic / groups, ks]);
    out(s.conv1d(&k, None, stride, pad, dil, groups).unwrap())
}

fn run_conv_transpose1d<T: WithDTypeF, B: Backend>(
    dev: &B,
    src: &[f32],
    kern: &[f32],
    b: usize,
    ic: usize,
    oc: usize,
    len: usize,
    ks: usize,
    stride: usize,
    groups: usize,
) -> Vec<f32> {
    let s = tt::<T, B>(dev, src, &[b, ic, len]);
    let k = tt::<T, B>(dev, kern, &[ic, oc / groups, ks]);
    out(s.conv_transpose1d(&k, None, stride, 0, 0, groups).unwrap())
}

#[test]
fn convolutions() -> Result<()> {
    if !f16_ready() {
        return Ok(());
    }
    // (b, ic, oc, len, ks, stride, pad, dil, groups). The wide-output/few-channel
    // cases take the direct kernel, the rest im2col + GEMM.
    let cases: &[(usize, usize, usize, usize, usize, usize, usize, usize, usize)] = &[
        (1, 1, 1, 8, 3, 1, 0, 1, 1),
        (1, 4, 6, 16, 3, 1, 1, 1, 1),
        (1, 64, 32, 128, 3, 1, 0, 1, 1),
        (1, 64, 1, 128, 3, 1, 0, 1, 1),
        (1, 32, 64, 16, 7, 1, 0, 2, 1),
        (2, 4, 4, 12, 3, 2, 1, 1, 2),
        (1, 8, 8, 16, 4, 2, 0, 1, 8),
    ];
    for (i, &(b, ic, oc, len, ks, stride, pad, dil, groups)) in cases.iter().enumerate() {
        let src = rnd(i as u64 + 301, b * ic * len, -1.0, 1.0);
        let kern = rnd(i as u64 + 401, oc * (ic / groups) * ks, -1.0, 1.0);
        verify16!(
            "conv1d", 2e-2, 1e-2, run_conv1d, &src, &kern, b, ic, oc, len, ks, stride, pad, dil,
            groups
        );
    }

    // (b, ic, oc, len, ks, stride, groups); groups == ic is the depthwise
    // upsample Mimi uses, which takes the direct transposed kernel.
    let tcases: &[(usize, usize, usize, usize, usize, usize, usize)] = &[
        (1, 1, 1, 4, 3, 1, 1),
        (1, 4, 6, 8, 4, 2, 1),
        (2, 4, 4, 6, 2, 2, 1),
        (1, 8, 8, 5, 4, 2, 8),
    ];
    for (i, &(b, ic, oc, len, ks, stride, groups)) in tcases.iter().enumerate() {
        let src = rnd(i as u64 + 501, b * ic * len, -1.0, 1.0);
        let kern = rnd(i as u64 + 601, ic * (oc / groups) * ks, -1.0, 1.0);
        verify16!(
            "conv_transpose1d",
            2e-2,
            1e-2,
            run_conv_transpose1d,
            &src,
            &kern,
            b,
            ic,
            oc,
            len,
            ks,
            stride,
            groups
        );
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Fill / roundtrip
// -----------------------------------------------------------------------------

fn run_zeros<T: WithDTypeF, B: Backend>(dev: &B, n: usize) -> Vec<f32> {
    out(Tensor::<T, B>::zeros(vec![n], dev).unwrap())
}
fn run_full<T: WithDTypeF, B: Backend>(dev: &B, v: f32, n: usize) -> Vec<f32> {
    out(Tensor::<T, B>::full(T::from_f32(v), vec![n], dev).unwrap())
}

#[test]
fn fill_and_roundtrip() -> Result<()> {
    if !f16_ready() {
        return Ok(());
    }
    // Odd counts: an f16 tensor with an odd length is not a multiple of 4 bytes,
    // which every buffer write and copy has to be.
    for n in [1usize, 3, 255, 257, 1001] {
        verify16!("zeros", 0.0, 0.0, run_zeros, n);
        verify16!("full", 0.0, 0.0, run_full, -3.5, n);
    }
    let d = rnd(999, 1001, -6.0, 6.0);
    let t: Tensor<f16, _> = tt(&wg(), &d, &[1001]);
    let got = out(t);
    cmp("roundtrip odd len", &d, &got, 0.0, 0.0);
    Ok(())
}
