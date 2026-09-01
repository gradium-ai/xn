// Elementwise unary ops, out of place. In-place goes to unary_inplace.wgsl:
// WebGPU forbids aliasing a buffer across bindings when one of them is writable.
// Op bodies live in ops.wgsl.
struct Params { n: u32, op: u32, alpha: f32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed to
// this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read> src: array<S>;
@group(0) @binding(1) var<storage, read_write> dst: array<S>;


@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = S(apply_unary(f32(src[i]), pc.op, pc.alpha));
}
