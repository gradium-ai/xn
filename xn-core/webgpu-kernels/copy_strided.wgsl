// Copy from a strided source to a contiguous destination.
// `info` packs [dims (num_dims), src_strides (num_dims)].
struct Params { numel: u32, num_dims: u32, src_offset: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> info: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= pc.numel { return; }
    let nd = pc.num_dims;
    var si = 0u;
    var rem = idx;
    for (var d = 0u; d < nd; d = d + 1u) {
        let di = nd - 1u - d;
        let dimv = info[di];
        si = si + (rem % dimv) * info[nd + di];
        rem = rem / dimv;
    }
    dst[idx] = src[pc.src_offset + si];
}
