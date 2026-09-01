// Transpose two dims of a tensor whose shape factors as
// (d_i, d1, d_j, d2, d_k), swapping d1 and d2. Mirrors layout.cu.
struct Params { numel: u32, d1: u32, d2: u32, d_i: u32, d_j: u32, d_k: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed to
// this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read> src: array<S>;
@group(0) @binding(1) var<storage, read_write> dst: array<S>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_idx = gid.x;
    if dst_idx >= pc.numel { return; }

    var rem = dst_idx;
    let i = rem / (pc.d2 * pc.d_j * pc.d1 * pc.d_k);
    rem = rem - i * (pc.d2 * pc.d_j * pc.d1 * pc.d_k);
    let a2 = rem / (pc.d_j * pc.d1 * pc.d_k);
    rem = rem - a2 * (pc.d_j * pc.d1 * pc.d_k);
    let j = rem / (pc.d1 * pc.d_k);
    rem = rem - j * (pc.d1 * pc.d_k);
    let a1 = rem / pc.d_k;
    rem = rem - a1 * pc.d_k;
    let k = rem;

    let src_idx = i * pc.d1 * pc.d_j * pc.d2 * pc.d_k
                + a1 * pc.d_j * pc.d2 * pc.d_k
                + j * pc.d2 * pc.d_k
                + a2 * pc.d_k
                + k;
    dst[dst_idx] = src[src_idx];
}
