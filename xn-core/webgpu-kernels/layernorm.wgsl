// Row-wise LayerNorm, f32 accumulation. One workgroup per row.
// remove_mean == 1: y = (x - mean) / sqrt(var + eps) * weight + bias
// remove_mean == 0: y =  x        / sqrt(var + eps) * weight + bias
struct Params { ncols: u32, eps: f32, remove_mean: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

var<workgroup> sh_sum: array<f32, 256>;
var<workgroup> sh_sq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    let base = row * pc.ncols;

    var s1 = 0.0;
    var s2 = 0.0;
    for (var c = tid; c < pc.ncols; c = c + 256u) {
        let x = src[base + c];
        s1 = s1 + x;
        s2 = s2 + x * x;
    }
    sh_sum[tid] = s1;
    sh_sq[tid] = s2;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s {
            sh_sum[tid] = sh_sum[tid] + sh_sum[tid + s];
            sh_sq[tid] = sh_sq[tid] + sh_sq[tid + s];
        }
        workgroupBarrier();
    }
    let mean = sh_sum[0] / f32(pc.ncols);
    let variance = sh_sq[0] / f32(pc.ncols) - mean * mean;
    let inv_std = inverseSqrt(variance + pc.eps);
    var mean_off = 0.0;
    if pc.remove_mean != 0u { mean_off = mean; }
    workgroupBarrier();

    for (var c = tid; c < pc.ncols; c = c + 256u) {
        let l = (src[base + c] - mean_off) * inv_std;
        dst[base + c] = l * weight[c] + bias[c];
    }
}
