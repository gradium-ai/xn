// General strided broadcast binary op. `info` packs, per dimension:
//   [dims (num_dims), lhs_strides (num_dims), rhs_strides (num_dims)]
// A broadcast dimension has stride 0. Destination is contiguous row-major.
struct Params { numel: u32, num_dims: u32, op: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> lhs: array<f32>;
@group(0) @binding(1) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= pc.numel { return; }
    let nd = pc.num_dims;
    var li = 0u;
    var ri = 0u;
    var rem = idx;
    for (var d = 0u; d < nd; d = d + 1u) {
        let di = nd - 1u - d;
        let dimv = info[di];
        let coord = rem % dimv;
        rem = rem / dimv;
        li = li + coord * info[nd + di];
        ri = ri + coord * info[2u * nd + di];
    }
    let a = lhs[li];
    let b = rhs[ri];
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
    dst[idx] = r;
}
