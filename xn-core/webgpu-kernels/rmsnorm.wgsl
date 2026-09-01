// Row-wise RMSNorm, f32 accumulation. One workgroup per row.
// dst = x * rsqrt(mean(x^2) + eps) * alpha
struct Params { ncols: u32, eps: f32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> src: array<S>;
@group(0) @binding(1) var<storage, read_write> dst: array<S>;
@group(0) @binding(2) var<storage, read_write> alpha: array<S>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    let base = row * pc.ncols;

    var acc = 0.0;
    for (var c = tid; c < pc.ncols; c = c + 256u) {
        let x = f32(src[base + c]);
        acc = acc + x * x;
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    let mean = sh[0] / f32(pc.ncols);
    let scale = inverseSqrt(mean + pc.eps);
    workgroupBarrier();

    for (var c = tid; c < pc.ncols; c = c + 256u) {
        dst[base + c] = S(scale * f32(src[base + c]) * f32(alpha[c]));
    }
}
