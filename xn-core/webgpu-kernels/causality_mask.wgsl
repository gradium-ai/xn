// Apply causal mask in place: set dst to -inf where idx2 > offset + idx1.
// Linear index decomposes as (b, idx1, idx2) over (bh, t1, t2).
struct Params { bh: u32, t1: u32, t2: u32, offset: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = pc.bh * pc.t1 * pc.t2;
    if idx >= total { return; }
    let idx2 = idx % pc.t2;
    let tmp = idx / pc.t2;
    let idx1 = tmp % pc.t1;
    if idx2 > pc.offset + idx1 {
        dst[idx] = S_NEG_BIG; // -inf for f32, most-negative finite for f16
    }
}
