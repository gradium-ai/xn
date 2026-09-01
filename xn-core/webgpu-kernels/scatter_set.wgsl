// scatter_set: for each src element at flat position i,
//   dst[left*dst_dim_size*right + ids[i]*right + right_idx] = src[i]
// where left = i/(right*src_dim_size), right_idx = i % right.
// ids is an i64 array viewed as u32 pairs; only the low word is read.
struct Params { numel: u32, right_size: u32, src_dim_size: u32, dst_dim_size: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed to
// this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;
@group(0) @binding(1) var<storage, read> src: array<S>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.numel { return; }
    let right = i % pc.right_size;
    let left = i / (pc.right_size * pc.src_dim_size);
    let idx = ids[2u * i];
    let dst_off = left * pc.dst_dim_size * pc.right_size + idx * pc.right_size + right;
    dst[dst_off] = src[i];
}
