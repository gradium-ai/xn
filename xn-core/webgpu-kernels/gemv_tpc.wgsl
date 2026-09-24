// GEMV, thread-per-column: the m == 1 case where each *thread* owns one output
// column and walks that column's whole weight row. No cross-lane reduction, so
// none of the barrier-tree cost of gemv.wgsl -- and one workgroup covers 64
// columns instead of one, so the launch count drops by 64x.
//   dst[b, 0, j] = sum_l lhs[b, 0, l] * rhs[b, l, j]
// Grid: (ceil(n / 64), batch, 1).
//
// Measured against gemv.wgsl on the phonon decode shapes (Apple M5): faster up
// to k = 768 (152 vs 63 GB/s at n=2304 k=768), slower from k = 1024 up, where a
// thread's serial walk over a long row loses locality -- 64 threads then span
// 64*k*4 bytes at once. The caller picks between the two on `k`; this kernel
// requires the contiguous-weight-row layout (`rhs_rs == 1`) that `matmul_t`
// produces, and reads lhs scalar because that single tiny row is shared by
// every thread and stays in cache.
struct Params {
    m: u32, n: u32, k: u32, batch: u32,
    lhs_b_stride: u32, rhs_b_stride: u32,
    lhs_cs: u32, lhs_rs: u32, rhs_cs: u32, rhs_rs: u32,
    dst_rs: u32, dst_cs: u32, lhs_o: u32, rhs_o: u32,
};
// WebGPU has no push constants; parameters arrive in a uniform windowed to
// this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;
@group(0) @binding(1) var<storage, read> lhs: array<S>;
@group(0) @binding(2) var<storage, read> rhs: array<S>;
@group(0) @binding(3) var<storage, read> rhs4: array<S4>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }
    let b = gid.y;

    let lbase = pc.lhs_o + b * pc.lhs_b_stride;      // row i = 0
    let rbase = pc.rhs_o + b * pc.rhs_b_stride + j * pc.rhs_cs;

    var acc = 0.0;
    var l = 0u;
    // 16-byte-aligned start lets the whole bulk of the row load as vec4. The
    // row is contiguous (rhs_rs == 1, checked by the caller), so only the start
    // offset and a k % 4 tail can break it.
    if (rbase & 3u) == 0u {
        let k4 = pc.k >> 2u;
        let base4 = rbase >> 2u;
        for (var g = 0u; g < k4; g = g + 1u) {
            let rv = vec4<f32>(rhs4[base4 + g]);
            let o = lbase + g * 4u * pc.lhs_cs;
            let s = pc.lhs_cs;
            acc = acc + rv.x * f32(lhs[o])
                      + rv.y * f32(lhs[o + s])
                      + rv.z * f32(lhs[o + 2u * s])
                      + rv.w * f32(lhs[o + 3u * s]);
        }
        l = k4 << 2u;
    }
    for (; l < pc.k; l = l + 1u) {
        acc = acc + f32(lhs[lbase + l * pc.lhs_cs]) * f32(rhs[rbase + l]);
    }
    dst[b * pc.n + j * pc.dst_cs] = S(acc);
}
