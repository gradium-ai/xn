// GEMV: the m == 1 case of the batched GEMM (the LLM-decode hot path).
//   dst[b, 0, j] = sum_l lhs[b, 0, l] * rhs[b, l, j]
// Grid: (ceil(n/TN), batch, 1); local size 64.
//
// ---------------------------------------------------------------------------
// Structure ported from MLC-LLM / WebLLM's generated decode matmul.
//
// The previous version gave each output column its own 256-thread workgroup and
// reduced over k through shared memory. At k = 576 that is ~2 elements per
// thread under an eight-step barrier tree per output: almost all of the time
// went into threads coordinating rather than computing.
//
// This follows the shape TVM's dlight GEMV schedule emits for WebGPU, read out
// of WebLLM's own compiled artifact (`NT_matmul_kernel_1`): a 64-thread
// workgroup, several outputs per workgroup held in per-thread registers
// (`array<f32, 8>` there for 2 columns x 4 batch rows; TN columns here), and a
// single reduction over 64 threads amortized across all of them. At k = 576
// each thread now accumulates 9 elements per column instead of 2.25, and the
// tree is six steps for TN outputs instead of eight steps for one.
//
// Origin, Apache-2.0:
//   schedule  https://github.com/apache/tvm/blob/main/python/tvm/s_tir/dlight/gpu/gemv.py
//             (was python/tvm/dlight/gpu/gemv.py through TVM v0.21.0)
//   generated https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_84/base/SmolLM2-135M-Instruct-q0f32_cs1k-webgpu.wasm
// Written from that structure rather than copied: dlight specializes and fully
// unrolls per shape, this stays general over n/k and arbitrary strides.
//
// Kept from the previous kernel: `rhs` is bound a second time as a 4-wide
// vector view (`rhs4`, binding 3, same underlying buffer). When a weight row is
// contiguous (`rhs_rs == 1`, the matmul_t case) and 16-byte aligned, each thread
// issues one 128-bit load instead of four scalar loads. Any k % 4 remainder and
// the unaligned / strided cases fall back to the scalar loop.
// ---------------------------------------------------------------------------
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

    // One scalar per output column, NOT `array<f32, TN>` with a loop index:
    // naga cannot keep a dynamically indexed function-scope array in registers
    // and spills it to thread-private memory (the same trap that made the
    // register-tiled GEMM 2.3x slower than a one-output-per-thread kernel until
    // it was unrolled). dlight avoids it by fully unrolling every shape.
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;

    // Weight-row bases for the TN columns this workgroup owns. Rows past `n`
    // are read anyway and discarded at store time: WebGPU guarantees an
    // out-of-range storage read is bounds-checked rather than undefined, so
    // hoisting the guard out of the k loop is safe and keeps the inner loop
    // branch-free.
    let rb0 = rbase + (j0 + 0u) * pc.rhs_cs;
    let rb1 = rbase + (j0 + 1u) * pc.rhs_cs;
    let rb2 = rbase + (j0 + 2u) * pc.rhs_cs;
    let rb3 = rbase + (j0 + 3u) * pc.rhs_cs;

    // Vectorized fast path: contiguous, 16-byte-aligned weight rows. Checking
    // `rhs_cs` alignment once covers every row in the group. The activation row
    // is tiny and reused by every workgroup, so it stays scalar and in cache.
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
    // Six steps, all TN columns at once -- one tree for four outputs, against
    // eight steps for one output before.
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
