// Arg-reduction over one dimension (-> i64 index). One workgroup per output.
// The output buffer is an i64 array, viewed here as pairs of u32 (little-endian):
// low word = index, high word = 0.
//   op: 0 = argmin, 1 = argmax
struct Params { num_outputs: u32, dim_size: u32, inner_size: u32, op: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

var<workgroup> shv: array<f32, 256>;
var<workgroup> shi: array<u32, 256>;

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

    var best: f32;
    if pc.op == 1u { best = -3.402823466e+38; } else { best = 3.402823466e+38; }
    var bidx = 0u;
    var have = false;
    for (var k = tid; k < pc.dim_size; k = k + 256u) {
        let v = src[outer_base + k * pc.inner_size];
        var better = v < best;
        if pc.op == 1u { better = v > best; }
        if !have || better {
            best = v;
            bidx = k;
            have = true;
        }
    }
    shv[tid] = best;
    shi[tid] = bidx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s {
            var better = shv[tid + s] < shv[tid];
            if pc.op == 1u { better = shv[tid + s] > shv[tid]; }
            let tie_lower = shv[tid + s] == shv[tid] && shi[tid + s] < shi[tid];
            if better || tie_lower {
                shv[tid] = shv[tid + s];
                shi[tid] = shi[tid + s];
            }
        }
        workgroupBarrier();
    }
    if tid == 0u {
        dst[2u * o] = shi[0];
        dst[2u * o + 1u] = 0u;
    }
}
