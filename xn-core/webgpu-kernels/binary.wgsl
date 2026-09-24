// Elementwise binary op: dst = lhs op rhs, same shape/contiguous, out of place.
// `bin_assign` goes to binary_inplace.wgsl. Op bodies live in ops.wgsl.
struct Params { n: u32, op: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed to
// this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read> lhs: array<S>;
@group(0) @binding(1) var<storage, read> rhs: array<S>;
@group(0) @binding(2) var<storage, read_write> dst: array<S>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = S(apply_binary(f32(lhs[i]), f32(rhs[i]), pc.op));
}
