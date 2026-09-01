// q8_0 matvec: `dst[j] = sum_l lhs[l] * dequant(w[j, l]) + bias[j]`, m == 1.
//
// This is the decode hot path -- roughly 48 dispatches per frame -- and is kept
// deliberately minimal. Two measured constraints shape it (see the
// `webgpu_dtype_lab` and `webgpu_q8_cost` examples, Apple M5):
//
//   * Quants load as `vec4<u32>`, 16 weights at a time. The same bytes read as
//     scalar `u32` reach only 1.4x f32 where these reach 3.6x.
//   * Nothing is dynamically indexed and nothing is kept for a row that does not
//     exist. Holding the eight unpacked lanes in an `array<vec4<f32>, 8>` and
//     looping over them costs 3.1x (it lands slower than plain f16), and even
//     carrying the four-row blocking of `qgemm_q8.wgsl` costs 1.8x here because
//     the extra live registers cut occupancy whether or not the rows are used.
//
// See quantization.rs for why the weights are split into separate quant and
// scale buffers rather than read as ggml's 34-byte blocks.
struct Params { m: u32, n: u32, k: u32, has_bias: u32, scale_off: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;
// Activations, vec4 view. k is a multiple of 32 for q8_0, so the row starts
// 16-byte aligned and there is no scalar tail.
@group(0) @binding(1) var<storage, read_write> lhs4: array<S4>;
@group(0) @binding(2) var<storage, read_write> q: array<vec4<u32>>;
// Same buffer as `q`, viewed as f32: the scales follow the quants, starting at
// word `pc.scale_off`. One buffer instead of two keeps the number of distinct
// buffers a compute pass references down, which measurably dominates.
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> bias: array<S>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }

    let nb = pc.k >> 5u;         // 32-value blocks per row
    let qrow = j * (pc.k >> 4u); // vec4<u32> per row (16 weights each)
    let srow = pc.scale_off + j * nb;

    var tot = 0.0;
    for (var b = 0u; b < nb; b = b + 1u) {
        let lo = q[qrow + b * 2u];
        let hi = q[qrow + b * 2u + 1u];
        let l0 = b * 8u;
        // 32 quants -> eight lanes of four, sign-extended from packed bytes.
        var acc = vec4<f32>(unpack4xI8(lo.x)) * vec4<f32>(lhs4[l0]);
        acc = acc + vec4<f32>(unpack4xI8(lo.y)) * vec4<f32>(lhs4[l0 + 1u]);
        acc = acc + vec4<f32>(unpack4xI8(lo.z)) * vec4<f32>(lhs4[l0 + 2u]);
        acc = acc + vec4<f32>(unpack4xI8(lo.w)) * vec4<f32>(lhs4[l0 + 3u]);
        acc = acc + vec4<f32>(unpack4xI8(hi.x)) * vec4<f32>(lhs4[l0 + 4u]);
        acc = acc + vec4<f32>(unpack4xI8(hi.y)) * vec4<f32>(lhs4[l0 + 5u]);
        acc = acc + vec4<f32>(unpack4xI8(hi.z)) * vec4<f32>(lhs4[l0 + 6u]);
        acc = acc + vec4<f32>(unpack4xI8(hi.w)) * vec4<f32>(lhs4[l0 + 7u]);
        // The scale is constant across the block, applied once per block.
        tot = tot + scales[srow + b] * (acc.x + acc.y + acc.z + acc.w);
    }
    if pc.has_bias != 0u { tot = tot + f32(bias[j]); }
    dst[j] = S(tot);
}
