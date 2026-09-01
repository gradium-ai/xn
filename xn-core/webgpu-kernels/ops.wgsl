// Shared elementwise op bodies.
//
// WGSL has no preprocessor or include mechanism, so snippets are composed
// Rust-side: kernels that set `needs_ops` in `kernel_src` get this prepended
// after the dtype preamble. It exists so the in-place and out-of-place variants
// of unary/binary/broadcast share one definition of each op rather than
// duplicating a 15-arm switch four times.
//
// `op` codes match the `UnaryOp` / `BinaryOp` order used by the other backends.

/// Abramowitz & Stegun 7.1.26 approximation of erf, max abs error ~1.5e-7.
fn erf_approx(x0: f32) -> f32 {
    let s = sign(x0);
    let x = abs(x0);
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                   - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return s * y;
}

fn apply_unary(x: f32, op: u32, alpha: f32) -> f32 {
    switch op {
        case 0u: { return cos(x); }
        case 1u: { return sin(x); }
        case 2u: { return exp(x); }
        case 3u: { return log(x); }
        case 4u: { return -x; }
        case 5u: { return x * x; }
        case 6u: { return sqrt(x); }
        case 7u: { return inverseSqrt(x); }
        case 8u: { return abs(x); }
        case 9u: { return x * 0.5 * (1.0 + erf_approx(x * 0.7071067811865476)); }
        case 10u: { if x > 0.0 { return x; } return alpha * (exp(x) - 1.0); }
        case 11u: { return max(x, 0.0); }
        case 12u: { return x / (1.0 + exp(-x)); }
        case 13u: { return tanh(x); }
        case 14u: { return 1.0 / (1.0 + exp(-x)); }
        default: { return x; }
    }
}

fn apply_binary(a: f32, b: f32, op: u32) -> f32 {
    switch op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return max(a, b); }
        case 5u: { return min(a, b); }
        default: { return a; }
    }
}
