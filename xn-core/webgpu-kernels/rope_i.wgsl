// Rotary position embedding, interleaved.
struct Params { bh: u32, td: u32, h: u32, cs_stride_b: u32, cos_off: u32, sin_off: u32 };
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read> cosb: array<f32>;
@group(0) @binding(1) var<storage, read> sinb: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if 2u * idx >= pc.bh * pc.td { return; }

    let half_td = pc.td / 2u;
    let i_bh = idx / half_td;
    var cos_idx = idx % half_td;
    if pc.cs_stride_b > 0u { cos_idx = cos_idx + (i_bh / pc.h) * pc.cs_stride_b; }

    let c = cosb[pc.cos_off + cos_idx];
    let s = sinb[pc.sin_off + cos_idx];
    let a = src[2u * idx];
    let b = src[2u * idx + 1u];
    dst[2u * idx] = a * c - b * s;
    dst[2u * idx + 1u] = a * s + b * c;
}
