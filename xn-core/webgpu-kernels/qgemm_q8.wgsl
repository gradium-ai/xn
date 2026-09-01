// q8_0 matmul for m > 1, which for this model means the flow LM consuming its
// text prefix -- once per utterance, not per frame. Decode (m == 1) goes to
// qgemv_q8.wgsl, which is ~1.8x faster for lack of the row blocking here.
//
// Rows are handled MR at a time so a block's unpacked weights serve MR outputs;
// re-reading the weight row once per output row would multiply the dominant
// traffic by m. The eight lanes stay in named locals rather than an array,
// because dynamic indexing pushes them out of registers (measured 3.1x).
const MR: u32 = 4u;

struct Params { m: u32, n: u32, k: u32, has_bias: u32, scale_off: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;
@group(0) @binding(1) var<storage, read_write> lhs4: array<S4>;
@group(0) @binding(2) var<storage, read_write> q: array<vec4<u32>>;
// Same buffer as `q`, viewed as f32: the scales follow the quants, starting at
// word `pc.scale_off`. One buffer instead of two keeps the number of distinct
// buffers a compute pass references down, which measurably dominates.
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> bias: array<S>;

/// One 32-value block against `lhs4[base ..]`. The weights arrive as parameters
/// rather than an array so they stay in registers; this inlines.
fn dot_block(
    w0: vec4<f32>, w1: vec4<f32>, w2: vec4<f32>, w3: vec4<f32>,
    w4: vec4<f32>, w5: vec4<f32>, w6: vec4<f32>, w7: vec4<f32>,
    base: u32,
) -> f32 {
    var a = w0 * vec4<f32>(lhs4[base]);
    a = a + w1 * vec4<f32>(lhs4[base + 1u]);
    a = a + w2 * vec4<f32>(lhs4[base + 2u]);
    a = a + w3 * vec4<f32>(lhs4[base + 3u]);
    a = a + w4 * vec4<f32>(lhs4[base + 4u]);
    a = a + w5 * vec4<f32>(lhs4[base + 5u]);
    a = a + w6 * vec4<f32>(lhs4[base + 6u]);
    a = a + w7 * vec4<f32>(lhs4[base + 7u]);
    return a.x + a.y + a.z + a.w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }

    let nb = pc.k >> 5u;
    let qrow = j * (pc.k >> 4u);
    let srow = pc.scale_off + j * nb;
    let l4 = pc.k >> 2u;

    var b_add = 0.0;
    if pc.has_bias != 0u { b_add = f32(bias[j]); }

    var m0 = 0u;
    while m0 < pc.m {
        let rows = pc.m - m0;
        var t0 = 0.0;
        var t1 = 0.0;
        var t2 = 0.0;
        var t3 = 0.0;
        let r0 = m0 * l4;
        for (var b = 0u; b < nb; b = b + 1u) {
            let lo = q[qrow + b * 2u];
            let hi = q[qrow + b * 2u + 1u];
            let w0 = vec4<f32>(unpack4xI8(lo.x));
            let w1 = vec4<f32>(unpack4xI8(lo.y));
            let w2 = vec4<f32>(unpack4xI8(lo.z));
            let w3 = vec4<f32>(unpack4xI8(lo.w));
            let w4 = vec4<f32>(unpack4xI8(hi.x));
            let w5 = vec4<f32>(unpack4xI8(hi.y));
            let w6 = vec4<f32>(unpack4xI8(hi.z));
            let w7 = vec4<f32>(unpack4xI8(hi.w));
            let sc = scales[srow + b];
            let l0 = b * 8u;
            t0 = t0 + sc * dot_block(w0, w1, w2, w3, w4, w5, w6, w7, r0 + l0);
            if rows > 1u {
                t1 = t1 + sc * dot_block(w0, w1, w2, w3, w4, w5, w6, w7, r0 + l4 + l0);
            }
            if rows > 2u {
                t2 = t2 + sc * dot_block(w0, w1, w2, w3, w4, w5, w6, w7, r0 + 2u * l4 + l0);
            }
            if rows > 3u {
                t3 = t3 + sc * dot_block(w0, w1, w2, w3, w4, w5, w6, w7, r0 + 3u * l4 + l0);
            }
        }
        dst[m0 * pc.n + j] = S(t0 + b_add);
        if rows > 1u { dst[(m0 + 1u) * pc.n + j] = S(t1 + b_add); }
        if rows > 2u { dst[(m0 + 2u) * pc.n + j] = S(t2 + b_add); }
        if rows > 3u { dst[(m0 + 3u) * pc.n + j] = S(t3 + b_add); }
        m0 = m0 + MR;
    }
}
