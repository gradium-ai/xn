// GEMV: the m == 1 case of the batched GEMM, i.e. the decode hot path.
//   dst[b, 0, j] = sum_l lhs[b, 0, l] * rhs[b, l, j]
// Grid: (ceil(n/TN), batch, 1); local size 64.
//
// Shape follows TVM dlight's GEMV schedule, read out of WebLLM's compiled
// artifact (`NT_matmul_kernel_1`): a 64-thread workgroup producing TN columns
// held in per-thread registers, with a single reduction amortized across them.
// The kernel this replaced gave each column its own 256-thread workgroup, which
// at k = 576 is ~2 elements per thread under an eight-step barrier tree.
// Written from that structure, not copied: dlight specializes and fully unrolls
// per shape, this stays general over n/k and arbitrary strides.
//
// Origin, Apache-2.0:
//   https://github.com/apache/tvm/blob/main/python/tvm/s_tir/dlight/gpu/gemv.py
//   https://github.com/mlc-ai/binary-mlc-llm-libs (web-llm-models/v0_2_84/base/
//   SmolLM2-135M-Instruct-q0f32_cs1k-webgpu.wasm)
//
// `rhs` is bound twice: scalar, plus a vec4 view (`rhs4`) taken when a weight
// row is contiguous and 16-byte aligned.
struct Params {
    m: u32, n: u32, k: u32, batch: u32,
    lhs_b_stride: u32, rhs_b_stride: u32,
    lhs_cs: u32, lhs_rs: u32, rhs_cs: u32, rhs_rs: u32,
    dst_rs: u32, dst_cs: u32, lhs_o: u32, rhs_o: u32,
};
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read_write> lhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(3) var<storage, read_write> rhs4: array<vec4<f32>>;

const TPB: u32 = 64u;   // threads per workgroup
const TN: u32 = 4u;     // output columns per workgroup

// One slot per (thread, column): the whole tile reduces in six steps.
var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let j0 = wid.x * TN;
    let b = wid.y;
    let tid = lid.x;

    let lbase = pc.lhs_o + b * pc.lhs_b_stride; // row i = 0
    let rbase = pc.rhs_o + b * pc.rhs_b_stride;

    // Scalars, not `array<f32, TN>` with a loop index: naga cannot keep a
    // dynamically indexed function-scope array in registers and spills it to
    // thread-private memory.
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;

    // Bases for the TN columns. Columns past `n` are read and discarded at
    // store time -- WebGPU bounds-checks out-of-range storage reads, so the
    // guard can stay out of the k loop.
    let rb0 = rbase + (j0 + 0u) * pc.rhs_cs;
    let rb1 = rbase + (j0 + 1u) * pc.rhs_cs;
    let rb2 = rbase + (j0 + 2u) * pc.rhs_cs;
    let rb3 = rbase + (j0 + 3u) * pc.rhs_cs;

    // Vectorized path: contiguous, 16-byte-aligned weight rows. One `rhs_cs`
    // alignment check covers the whole group. The activation row is tiny and
    // reused by every workgroup, so it stays scalar.
    let vec_ok = pc.rhs_rs == 1u && pc.k >= 4u && (rbase & 3u) == 0u && (pc.rhs_cs & 3u) == 0u;

    if vec_ok {
        let k4 = pc.k >> 2u;
        let kbulk = k4 << 2u;
        let v0 = rb0 >> 2u;
        let v1 = rb1 >> 2u;
        let v2 = rb2 >> 2u;
        let v3 = rb3 >> 2u;
        for (var g = tid; g < k4; g = g + TPB) {
            let l = g * 4u;
            let a0 = lhs[lbase + (l + 0u) * pc.lhs_cs];
            let a1 = lhs[lbase + (l + 1u) * pc.lhs_cs];
            let a2 = lhs[lbase + (l + 2u) * pc.lhs_cs];
            let a3 = lhs[lbase + (l + 3u) * pc.lhs_cs];
            let w0 = rhs4[v0 + g];
            let w1 = rhs4[v1 + g];
            let w2 = rhs4[v2 + g];
            let w3 = rhs4[v3 + g];
            acc0 = fma(w0.x, a0, acc0);
            acc0 = fma(w0.y, a1, acc0);
            acc0 = fma(w0.z, a2, acc0);
            acc0 = fma(w0.w, a3, acc0);
            acc1 = fma(w1.x, a0, acc1);
            acc1 = fma(w1.y, a1, acc1);
            acc1 = fma(w1.z, a2, acc1);
            acc1 = fma(w1.w, a3, acc1);
            acc2 = fma(w2.x, a0, acc2);
            acc2 = fma(w2.y, a1, acc2);
            acc2 = fma(w2.z, a2, acc2);
            acc2 = fma(w2.w, a3, acc2);
            acc3 = fma(w3.x, a0, acc3);
            acc3 = fma(w3.y, a1, acc3);
            acc3 = fma(w3.z, a2, acc3);
            acc3 = fma(w3.w, a3, acc3);
        }
        // k % 4 remainder, scalar.
        for (var l = kbulk + tid; l < pc.k; l = l + TPB) {
            let a = lhs[lbase + l * pc.lhs_cs];
            acc0 = fma(rhs[rb0 + l], a, acc0);
            acc1 = fma(rhs[rb1 + l], a, acc1);
            acc2 = fma(rhs[rb2 + l], a, acc2);
            acc3 = fma(rhs[rb3 + l], a, acc3);
        }
    } else {
        for (var l = tid; l < pc.k; l = l + TPB) {
            let a = lhs[lbase + l * pc.lhs_cs];
            let lo = l * pc.rhs_rs;
            acc0 = fma(rhs[rb0 + lo], a, acc0);
            acc1 = fma(rhs[rb1 + lo], a, acc1);
            acc2 = fma(rhs[rb2 + lo], a, acc2);
            acc3 = fma(rhs[rb3 + lo], a, acc3);
        }
    }

    sh[tid * TN + 0u] = acc0;
    sh[tid * TN + 1u] = acc1;
    sh[tid * TN + 2u] = acc2;
    sh[tid * TN + 3u] = acc3;
    workgroupBarrier();
    // Six steps, all TN columns at once.
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
            dst[b * pc.n + jj * pc.dst_cs] = sh[tid];
        }
    }
}
