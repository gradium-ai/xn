// index_select over `dim`. ids is an i64 array viewed as u32 pairs
// (little-endian): only the low word is read. An index of -1 selects zeros.
//   dst[(left*num_ids + id)*right + r] = src[(left*src_dim_size + ids[id])*right + r]
struct Params { left_size: u32, num_ids: u32, right_size: u32, src_dim_size: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let total = pc.left_size * pc.num_ids * pc.right_size;
    if g >= total { return; }

    let r = g % pc.right_size;
    let tmp = g / pc.right_size;
    let id_i = tmp % pc.num_ids;
    let left = tmp / pc.num_ids;

    let idx = bitcast<i32>(ids[2u * id_i]);
    let dst_off = (left * pc.num_ids + id_i) * pc.right_size + r;
    if idx == -1 {
        dst[dst_off] = 0.0;
        return;
    }
    let src_off = (left * pc.src_dim_size + u32(idx)) * pc.right_size + r;
    dst[dst_off] = src[src_off];
}
