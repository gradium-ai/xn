// Fill a buffer with a constant. Keeps zeros/full inside the recorded batch.
struct Params { n: u32, v: f32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = S(pc.v);
}
