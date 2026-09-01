// Row-wise softmax. One workgroup per row; `ncols` elements per row.
// Reductions accumulate in f32.
struct Params { ncols: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> src: array<S>;
@group(0) @binding(1) var<storage, read_write> dst: array<S>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    let base = row * pc.ncols;

    var m = -3.402823466e+38;
    for (var c = tid; c < pc.ncols; c = c + 256u) { m = max(m, f32(src[base + c])); }
    sh[tid] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = max(sh[tid], sh[tid + s]); }
        workgroupBarrier();
    }
    let maxv = sh[0];
    workgroupBarrier();

    var sum = 0.0;
    for (var c = tid; c < pc.ncols; c = c + 256u) {
        let e = exp(f32(src[base + c]) - maxv);
        dst[base + c] = S(e);
        sum = sum + e;
    }
    sh[tid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sh[0];
    workgroupBarrier();

    for (var c = tid; c < pc.ncols; c = c + 256u) { dst[base + c] = S(f32(dst[base + c]) * inv); }
}
