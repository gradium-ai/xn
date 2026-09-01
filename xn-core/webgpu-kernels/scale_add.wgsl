// dst = src * scale + add   (elementwise)
struct Params { n: u32, scale: f32, add: f32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> src: array<S>;
@group(0) @binding(1) var<storage, read_write> dst: array<S>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = S(f32(src[i]) * pc.scale + pc.add);
}
