// 2D strided copy: dst[dst_o + i*dst_s + j] = src[src_o + i*src_s + j]
// for i in [0,d1), j in [0,d2).
struct Params { d1: u32, d2: u32, src_s: u32, dst_s: u32, src_o: u32, dst_o: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= pc.d1 * pc.d2 { return; }
    let i1 = idx / pc.d2;
    let i2 = idx - pc.d2 * i1;
    dst[pc.dst_o + i1 * pc.dst_s + i2] = src[pc.src_o + i1 * pc.src_s + i2];
}
