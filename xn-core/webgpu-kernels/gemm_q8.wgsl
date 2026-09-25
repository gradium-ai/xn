// q8_0 dequantize-fused GEMM for small m: prefill and the streaming-conv
// shapes, with 8-bit weights.
//   dst[i, j] = sum_l lhs[i, l] * W[j, l],  W held as q8_0 blocks.
// Grid: (ceil(n/TN), ceil(m/MR), 1); local size 64.
//
// Same weight layout and the same reduction shape as gemv_q8.wgsl; the
// difference is that a workgroup covers MR rows as well as TN columns, so one
// pass over the weight stream serves MR outputs instead of one. That is the
// whole point of a separate kernel: at m == 1 the weight stream sets the time,
// but by m == 4 it is amortized and the activation loads start to matter.
//
// MR x TN accumulators, all scalars. naga spills a dynamically indexed
// function-scope array to thread-private memory instead of keeping it in
// registers, and on this backend that difference measured an order of
// magnitude on the f32 tiled GEMM -- so the tile is unrolled by hand.
//
// `m == 1` is deliberately NOT routed here. The 32x32 f32 tile wasting its
// rows on a 3-row problem is what motivated this kernel, but the mirror image
// is just as real: sending single-row work through a 4-row tile pays four
// times the reduction for one useful result.
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
const MR: u32 = 4u;     // output rows per workgroup
const ACC: u32 = 16u;   // MR * TN

var<workgroup> sh: array<f32, 1024>;  // TPB * ACC

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
    let i0 = wid.y * MR;
    let tid = lid.x;

    let kw = pc.k >> 2u;   // packed words per row
    let kb = pc.k >> 5u;   // blocks per row

    let q0 = (j0 + 0u) * kw;
    let q1 = (j0 + 1u) * kw;
    let q2 = (j0 + 2u) * kw;
    let q3 = (j0 + 3u) * kw;
    let s0 = (j0 + 0u) * kb;
    let s1 = (j0 + 1u) * kb;
    let s2 = (j0 + 2u) * kb;
    let s3 = (j0 + 3u) * kb;

    // Row bases into lhs. Rows past `m` are read unguarded, but NOT because
    // they come back zero -- buffers are pooled, rounded up to a size class
    // and bound whole, so such a read usually lands inside the binding and
    // returns stale bytes from a previously freed tensor. What makes it
    // correct is the store: accumulator rr*TN + cc maps to exactly one
    // output, the reduction sums each slot without crossing into another, and
    // the store writes only if both ii < m and jj < n -- so the junk is
    // discarded, never accumulated into a live result. Any future fast path
    // that writes a full tile unguarded would put that stale data in dst.
    let r0 = (i0 + 0u) * pc.k;
    let r1 = (i0 + 1u) * pc.k;
    let r2 = (i0 + 2u) * pc.k;
    let r3 = (i0 + 3u) * pc.k;

    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0;

    for (var w = tid; w < kw; w = w + TPB) {
        let l = w << 2u;
        let bi = w >> 3u;

        let w0 = unpack4(qs[q0 + w]);
        let w1 = unpack4(qs[q1 + w]);
        let w2 = unpack4(qs[q2 + w]);
        let w3 = unpack4(qs[q3 + w]);
        let d0 = scales[s0 + bi];
        let d1 = scales[s1 + bi];
        let d2 = scales[s2 + bi];
        let d3 = scales[s3 + bi];

        let x0 = vec4<f32>(lhs[r0 + l], lhs[r0 + l + 1u], lhs[r0 + l + 2u], lhs[r0 + l + 3u]);
        a00 = fma(d0, dot(w0, x0), a00);
        a01 = fma(d1, dot(w1, x0), a01);
        a02 = fma(d2, dot(w2, x0), a02);
        a03 = fma(d3, dot(w3, x0), a03);

        let x1 = vec4<f32>(lhs[r1 + l], lhs[r1 + l + 1u], lhs[r1 + l + 2u], lhs[r1 + l + 3u]);
        a10 = fma(d0, dot(w0, x1), a10);
        a11 = fma(d1, dot(w1, x1), a11);
        a12 = fma(d2, dot(w2, x1), a12);
        a13 = fma(d3, dot(w3, x1), a13);

        let x2 = vec4<f32>(lhs[r2 + l], lhs[r2 + l + 1u], lhs[r2 + l + 2u], lhs[r2 + l + 3u]);
        a20 = fma(d0, dot(w0, x2), a20);
        a21 = fma(d1, dot(w1, x2), a21);
        a22 = fma(d2, dot(w2, x2), a22);
        a23 = fma(d3, dot(w3, x2), a23);

        let x3 = vec4<f32>(lhs[r3 + l], lhs[r3 + l + 1u], lhs[r3 + l + 2u], lhs[r3 + l + 3u]);
        a30 = fma(d0, dot(w0, x3), a30);
        a31 = fma(d1, dot(w1, x3), a31);
        a32 = fma(d2, dot(w2, x3), a32);
        a33 = fma(d3, dot(w3, x3), a33);
    }

    let o = tid * ACC;
    sh[o +  0u] = a00; sh[o +  1u] = a01; sh[o +  2u] = a02; sh[o +  3u] = a03;
    sh[o +  4u] = a10; sh[o +  5u] = a11; sh[o +  6u] = a12; sh[o +  7u] = a13;
    sh[o +  8u] = a20; sh[o +  9u] = a21; sh[o + 10u] = a22; sh[o + 11u] = a23;
    sh[o + 12u] = a30; sh[o + 13u] = a31; sh[o + 14u] = a32; sh[o + 15u] = a33;
    workgroupBarrier();
    for (var s = TPB >> 1u; s > 0u; s = s >> 1u) {
        if tid < s {
            let a = tid * ACC;
            let b = (tid + s) * ACC;
            for (var c = 0u; c < ACC; c = c + 1u) {
                sh[a + c] = sh[a + c] + sh[b + c];
            }
        }
        workgroupBarrier();
    }
    // ACC results live in sh[0..16); one thread per (row, column) stores.
    if tid < ACC {
        let rr = tid / TN;
        let cc = tid % TN;
        let ii = i0 + rr;
        let jj = j0 + cc;
        if ii < pc.m && jj < pc.n {
            dst[ii * pc.n + jj] = sh[tid];
        }
    }
}
