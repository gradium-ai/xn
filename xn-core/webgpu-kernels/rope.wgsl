// Rotary position embedding, non-interleaved (GPT-NeoX style).
struct Params { bh: u32, td: u32, d: u32, h: u32, cs_stride_b: u32, cos_off: u32, sin_off: u32 };
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
    let half_d = pc.d / 2u;
    let i_bh = idx / half_td;
    let i_td = idx - half_td * i_bh;
    let i_t = i_td / half_d;
    let i_d = i_td - half_d * i_t;
    let i1 = i_bh * pc.td + i_t * pc.d + i_d;
    let i2 = i1 + half_d;
    var i_cs = i_t * half_d + i_d;
    if pc.cs_stride_b > 0u { i_cs = i_cs + (i_bh / pc.h) * pc.cs_stride_b; }

    let c = cosb[pc.cos_off + i_cs];
    let s = sinb[pc.sin_off + i_cs];
    let a = src[i1];
    let b = src[i2];
    dst[i1] = a * c - b * s;
    dst[i2] = a * s + b * c;
}
