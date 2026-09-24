// Elementwise binary op: dst = lhs op rhs, same shape/contiguous.
// For `bin_assign` (dst = dst op s), bind lhs=dst and rhs=s.
struct Params { n: u32, op: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    let a = lhs[i];
    let b = rhs[i];
    var r: f32;
    switch pc.op {
        case 0u: { r = a + b; }
        case 1u: { r = a - b; }
        case 2u: { r = a * b; }
        case 3u: { r = a / b; }
        case 4u: { r = max(a, b); }
        case 5u: { r = min(a, b); }
        default: { r = a; }
    }
    dst[i] = r;
}
