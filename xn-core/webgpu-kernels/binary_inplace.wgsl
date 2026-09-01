// In-place binary: `dst = op(dst, src)`, two bindings.
//
// See unary_inplace.wgsl: the out-of-place kernel would have to alias `dst` into
// two bindings, one of them writable, which WebGPU forbids.
struct Params { n: u32, op: u32 };
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;
@group(0) @binding(1) var<storage, read> src: array<S>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = S(apply_binary(f32(dst[i]), f32(src[i]), pc.op));
}
