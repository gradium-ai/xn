// Elementwise unary ops. For in-place use, bind the same buffer to src and dst.
// `op` matches the `UnaryOp` order used by the CPU/CUDA backends.
struct Params { n: u32, op: u32, alpha: f32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> src: array<S>;
@group(0) @binding(1) var<storage, read_write> dst: array<S>;

// Abramowitz & Stegun 7.1.26 approximation of erf, max abs error ~1.5e-7.
fn erf_approx(x0: f32) -> f32 {
    let s = sign(x0);
    let x = abs(x0);
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                   - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return s * y;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    let x = f32(src[i]);
    var r: f32;
    switch pc.op {
        case 0u: { r = cos(x); }
        case 1u: { r = sin(x); }
        case 2u: { r = exp(x); }
        case 3u: { r = log(x); }
        case 4u: { r = -x; }
        case 5u: { r = x * x; }
        case 6u: { r = sqrt(x); }
        case 7u: { r = inverseSqrt(x); }
        case 8u: { r = abs(x); }
        case 9u: { r = x * 0.5 * (1.0 + erf_approx(x * 0.7071067811865476)); }
        case 10u: { if x > 0.0 { r = x; } else { r = pc.alpha * (exp(x) - 1.0); } }
        case 11u: { r = max(x, 0.0); }
        case 12u: { r = x / (1.0 + exp(-x)); }
        case 13u: { r = tanh(x); }
        case 14u: { r = 1.0 / (1.0 + exp(-x)); }
        default: { r = x; }
    }
    dst[i] = S(r);
}
