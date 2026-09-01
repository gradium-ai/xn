// GEMV: the m == 1 case of the batched GEMM (the LLM-decode hot path).
//   dst[b, 0, j] = sum_l lhs[b, 0, l] * rhs[b, l, j]
// One workgroup per output column j (per batch b); the workgroup's threads
// cooperatively reduce over the k dimension in f32. Push constants match the
// GEMM kernel. Grid: (n, batch, 1).
//
// `rhs` (the dominant, bandwidth-bound stream) is bound a second time as a
// 4-wide vector view (`rhs4`, binding 3, same underlying buffer): when a weight
// row is stored contiguously (`rhs_rs == 1`, the matmul_t case) and its start is
// 4-element (16-byte) aligned, each thread issues one 128-bit load instead of
// four scalar loads, quartering load instructions on the hot stream without
// changing the coalescing pattern. `lhs` stays scalar (a single tiny row reused
// by every workgroup, so it lives in cache). Any k % 4 remainder and the
// unaligned / strided cases fall back to the scalar loop.
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

var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let j = wid.x;
    let b = wid.y;
    let tid = lid.x;

    let lbase = pc.lhs_o + b * pc.lhs_b_stride; // row i = 0
    let rbase = pc.rhs_o + b * pc.rhs_b_stride + j * pc.rhs_cs;

    var acc = 0.0;
    if pc.rhs_rs == 1u && pc.k >= 4u && (rbase & 3u) == 0u {
        // Vectorized fast path: contiguous, 16-byte-aligned weight row.
        let k4 = pc.k >> 2u;
        let kbulk = k4 << 2u;
        let base4 = rbase >> 2u;
        for (var g = tid; g < k4; g = g + 256u) {
            let rv = vec4<f32>(rhs4[base4 + g]);
            let l = g * 4u;
            acc = acc + rv.x * f32(lhs[lbase + (l + 0u) * pc.lhs_cs]);
            acc = acc + rv.y * f32(lhs[lbase + (l + 1u) * pc.lhs_cs]);
            acc = acc + rv.z * f32(lhs[lbase + (l + 2u) * pc.lhs_cs]);
            acc = acc + rv.w * f32(lhs[lbase + (l + 3u) * pc.lhs_cs]);
        }
        // k % 4 remainder, scalar.
        for (var l = kbulk + tid; l < pc.k; l = l + 256u) {
            acc = acc + f32(lhs[lbase + l * pc.lhs_cs]) * f32(rhs[rbase + l]);
        }
    } else {
        for (var l = tid; l < pc.k; l = l + 256u) {
            acc = acc + f32(lhs[lbase + l * pc.lhs_cs]) * f32(rhs[rbase + l * pc.rhs_rs]);
        }
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    if tid == 0u {
        dst[b * pc.n + j * pc.dst_cs] = S(sh[0]);
    }
}
