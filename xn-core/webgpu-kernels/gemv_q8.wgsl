// q8_0 dequantize-fused GEMV: the m == 1 decode path with 8-bit weights.
//   dst[0, j] = sum_l lhs[0, l] * W[j, l],  W held as q8_0 blocks.
// Grid: (ceil(n/TN), 1, 1); local size 64.
//
// Structure follows gemv.wgsl -- 64 threads, TN columns in per-thread scalar
// accumulators, one reduction amortized across them -- so the only new thing
// here is where the weights come from. See that file for why the accumulators
// are scalars rather than an indexed array.
//
// The weight is split into two buffers rather than kept as ggml's interleaved
// 34-byte block (`f16` scale followed by 32 `i8`). 34 is not a multiple of 4,
// so the packed form costs an unaligned load per block and cannot be read as
// `u32`; splitting it makes both streams naturally aligned and lets the quants
// come in four at a time. The split happens once, on upload.
//
//   qs:     row j at j * (k/4),  each u32 holding 4 int8 weights
//   scales: row j at j * (k/32), one f32 per 32-weight block
//
// Activations stay f32. At m == 1 this kernel is bound by the weight stream --
// 1 byte per weight against f32's 4 -- so unpacking to f32 and using the FMA
// path costs nothing that a packed integer dot (`dot4I8Packed`) would save.
// That trade flips once the activation is reused across rows; see gemm_q8.wgsl.
struct Params {
    m: u32, n: u32, k: u32,
};
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> lhs: array<f32>;
@group(0) @binding(2) var<storage, read> qs: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;

const TPB: u32 = 64u;   // threads per workgroup
const TN: u32 = 4u;     // output columns per workgroup

var<workgroup> sh: array<f32, 256>;

// Four int8 weights out of one packed word. `extractBits` on a signed integer
// sign-extends, which is what the stored quants need.
fn unpack4(w: u32) -> vec4<f32> {
    let q = bitcast<i32>(w);
    return vec4<f32>(
        f32(extractBits(q, 0u, 8u)),
        f32(extractBits(q, 8u, 8u)),
        f32(extractBits(q, 16u, 8u)),
        f32(extractBits(q, 24u, 8u)),
    );
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let j0 = wid.x * TN;
    let tid = lid.x;

    let kw = pc.k >> 2u;   // packed words per row
    let kb = pc.k >> 5u;   // blocks per row

    // Column bases. Columns past `n` are read unguarded so the guard stays out
    // of the k loop, and are discarded at store time -- not neutralised by the
    // read. Buffers are pooled, rounded up to a size class and bound whole, so
    // such a read returns stale bytes rather than zero; only the `jj < pc.n`
    // guard on the store keeps them out of dst.
    let q0 = (j0 + 0u) * kw;
    let q1 = (j0 + 1u) * kw;
    let q2 = (j0 + 2u) * kw;
    let q3 = (j0 + 3u) * kw;
    let s0 = (j0 + 0u) * kb;
    let s1 = (j0 + 1u) * kb;
    let s2 = (j0 + 2u) * kb;
    let s3 = (j0 + 3u) * kb;

    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;

    for (var w = tid; w < kw; w = w + TPB) {
        let l = w << 2u;
        let a = vec4<f32>(lhs[l], lhs[l + 1u], lhs[l + 2u], lhs[l + 3u]);
        // Eight words to a block, so the scale index is the word index >> 3.
        let bi = w >> 3u;
        acc0 = fma(scales[s0 + bi], dot(unpack4(qs[q0 + w]), a), acc0);
        acc1 = fma(scales[s1 + bi], dot(unpack4(qs[q1 + w]), a), acc1);
        acc2 = fma(scales[s2 + bi], dot(unpack4(qs[q2 + w]), a), acc2);
        acc3 = fma(scales[s3 + bi], dot(unpack4(qs[q3 + w]), a), acc3);
    }

    sh[tid * TN + 0u] = acc0;
    sh[tid * TN + 1u] = acc1;
    sh[tid * TN + 2u] = acc2;
    sh[tid * TN + 3u] = acc3;
    workgroupBarrier();
    for (var s = TPB >> 1u; s > 0u; s = s >> 1u) {
        if tid < s {
            sh[tid * TN + 0u] = sh[tid * TN + 0u] + sh[(tid + s) * TN + 0u];
            sh[tid * TN + 1u] = sh[tid * TN + 1u] + sh[(tid + s) * TN + 1u];
            sh[tid * TN + 2u] = sh[tid * TN + 2u] + sh[(tid + s) * TN + 2u];
            sh[tid * TN + 3u] = sh[tid * TN + 3u] + sh[(tid + s) * TN + 3u];
        }
        workgroupBarrier();
    }
    if tid < TN {
        let jj = j0 + tid;
        if jj < pc.n {
            dst[jj] = sh[tid];
        }
    }
}
