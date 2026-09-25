// Fill a buffer with a constant. Keeps zeros/full inside the recorded batch.
struct Params { n: u32, v: f32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = pc.v;
}
