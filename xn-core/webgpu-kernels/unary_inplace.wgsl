// In-place unary: `dst = f(dst)`, one binding.
//
// The out-of-place `unary.wgsl` cannot serve this case by binding the same buffer
// to both of its slots: WebGPU forbids aliasing a buffer across bindings when any
// of them is writable, so an in-place op needs a kernel that reads and writes
// through a single read_write binding.
struct Params { n: u32, op: u32, alpha: f32 };
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    let x = f32(dst[i]);
    dst[i] = S(apply_unary(x, pc.op, pc.alpha));
}
