// Reduction over one dimension. One workgroup per output; f32 accumulation.
// Iteration shape (outer, inner, dim); physical layout (outer, dim, inner).
//   op: 0 = sum, 1 = max, 2 = min
struct Params { num_outputs: u32, dim_size: u32, inner_size: u32, op: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let o = wid.x;
    let tid = lid.x;
    let a_inner = o % pc.inner_size;
    let a_outer = o / pc.inner_size;
    let outer_base = a_outer * pc.dim_size * pc.inner_size + a_inner;

    var acc: f32;
    if pc.op == 0u { acc = 0.0; } else if pc.op == 1u { acc = -3.402823466e+38; } else { acc = 3.402823466e+38; }
    for (var k = tid; k < pc.dim_size; k = k + 256u) {
        let v = src[outer_base + k * pc.inner_size];
        if pc.op == 0u { acc = acc + v; } else if pc.op == 1u { acc = max(acc, v); } else { acc = min(acc, v); }
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s {
            if pc.op == 0u { sh[tid] = sh[tid] + sh[tid + s]; }
            else if pc.op == 1u { sh[tid] = max(sh[tid], sh[tid + s]); }
            else { sh[tid] = min(sh[tid], sh[tid + s]); }
        }
        workgroupBarrier();
    }
    if tid == 0u { dst[o] = sh[0]; }
}
