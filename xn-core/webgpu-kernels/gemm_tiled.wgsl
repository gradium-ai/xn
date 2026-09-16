// Tiled GEMM with a 4x4 register tile (f32 accumulation), general strides and
// batch.
//   dst[b, i, j] = sum_l lhs[b, i, l] * rhs[b, l, j]
// Offsets: lhs_o + b*lhs_b_stride + i*lhs_rs + l*lhs_cs (rhs likewise; dst is
// b*m*n + i*dst_rs + j*dst_cs).
// Grid: (ceil(n/32), ceil(m/32), batch); local size (8, 8, 1).
//
// Shape follows TVM dlight's matmul schedule, read out of WebLLM's compiled
// artifact (`NT_matmul_kernel_2`): a 64-thread workgroup, a 32x32 output tile,
// 16 outputs per thread, operand tiles staged 8 deep. The kernel this replaced
// was a 16x16 tile computing one output per thread, so two global loads bought
// sixteen multiply-adds -- memory-bound on a compute-bound problem.
// Written from that structure, not copied: dlight specializes and fully unrolls
// per shape, this stays general over m/n/k and arbitrary strides.
//
// Origin, Apache-2.0:
//   https://github.com/apache/tvm/blob/main/python/tvm/s_tir/dlight/gpu/matmul.py
//   https://github.com/mlc-ai/binary-mlc-llm-libs (web-llm-models/v0_2_84/base/
//   SmolLM2-135M-Instruct-q0f32_cs1k-webgpu.wasm)
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

// 32x32 output tile, 8-deep k stage, 4x4 outputs per thread (dlight's shape).
const TILE: u32 = 32u;
const KSTEP: u32 = 8u;
const TPB: u32 = 64u;   // threads per workgroup (8x8)
const RT: u32 = 4u;     // register tile edge

// lhs tile: [32 rows][8 k]   rhs tile: [8 k][32 cols]
var<workgroup> at: array<f32, 256>;
var<workgroup> bt: array<f32, 256>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wid.z;
    let row0 = wid.y * TILE;
    let col0 = wid.x * TILE;
    let tid = lid.y * 8u + lid.x;

    let lhs_base = pc.lhs_o + b * pc.lhs_b_stride;
    let rhs_base = pc.rhs_o + b * pc.rhs_b_stride;

    // The thread's 4x4 patch of the output tile, one scalar per output.
    // Not `array<f32, 16>` with loop indices: naga cannot keep a dynamically
    // indexed function-scope array in registers and spills it to
    // thread-private memory, which measured 2.3x slower than the kernel this
    // replaced.
    var c00 = 0.0;
    var c01 = 0.0;
    var c02 = 0.0;
    var c03 = 0.0;
    var c10 = 0.0;
    var c11 = 0.0;
    var c12 = 0.0;
    var c13 = 0.0;
    var c20 = 0.0;
    var c21 = 0.0;
    var c22 = 0.0;
    var c23 = 0.0;
    var c30 = 0.0;
    var c31 = 0.0;
    var c32 = 0.0;
    var c33 = 0.0;

    // Whole-tile-in-range test, hoisted out of the staging loop.
    let tile_interior = (row0 + TILE) <= pc.m && (col0 + TILE) <= pc.n;

    let nk = (pc.k + KSTEP - 1u) / KSTEP;
    for (var kt = 0u; kt < nk; kt = kt + 1u) {
        let k0 = kt * KSTEP;
        let interior = tile_interior && (k0 + KSTEP) <= pc.k;

        // Stage both operand tiles: 256 elements each, 4 per thread.
        // `interior` drops the per-element bounds check for tiles fully in
        // range, which is every tile but the last row/column/k-slice.
        if interior {
            for (var s = 0u; s < 4u; s = s + 1u) {
                let idx = tid + s * TPB;
                let ar = idx / KSTEP;
                let ak = idx % KSTEP;
                at[idx] = lhs[lhs_base + (row0 + ar) * pc.lhs_rs + (k0 + ak) * pc.lhs_cs];
                var bk: u32;
                var bc: u32;
                if pc.rhs_rs == 1u {
                    bc = idx / KSTEP;
                    bk = idx % KSTEP;
                } else {
                    bk = idx / TILE;
                    bc = idx % TILE;
                }
                bt[bk * TILE + bc] =
                    rhs[rhs_base + (k0 + bk) * pc.rhs_rs + (col0 + bc) * pc.rhs_cs];
            }
        } else {
            for (var s = 0u; s < 4u; s = s + 1u) {
                let idx = tid + s * TPB;

                // at[r][kk]: consecutive threads walk kk, contiguous when
                // lhs is row-major.
                let ar = idx / KSTEP;
                let ak = idx % KSTEP;
                let grow = row0 + ar;
                let gk = k0 + ak;
                if grow < pc.m && gk < pc.k {
                    at[idx] = lhs[lhs_base + grow * pc.lhs_rs + gk * pc.lhs_cs];
                } else {
                    at[idx] = 0.0;
                }

                // bt is laid out [kk][c] either way; consecutive threads must
                // walk whichever axis is contiguous in memory -- k when
                // rhs_rs == 1 (matmul_t, every linear layer), else columns.
                var bk: u32;
                var bc: u32;
                if pc.rhs_rs == 1u {
                    bc = idx / KSTEP;
                    bk = idx % KSTEP;
                } else {
                    bk = idx / TILE;
                    bc = idx % TILE;
                }
                let gcol = col0 + bc;
                let gk2 = k0 + bk;
                if gcol < pc.n && gk2 < pc.k {
                    bt[bk * TILE + bc] = rhs[rhs_base + gk2 * pc.rhs_rs + gcol * pc.rhs_cs];
                } else {
                    bt[bk * TILE + bc] = 0.0;
                }
            }
        }
        workgroupBarrier();

        // Eight loads buy sixteen multiply-adds.
        for (var kk = 0u; kk < KSTEP; kk = kk + 1u) {
            let a0 = at[(lid.y * RT + 0u) * KSTEP + kk];
            let a1 = at[(lid.y * RT + 1u) * KSTEP + kk];
            let a2 = at[(lid.y * RT + 2u) * KSTEP + kk];
            let a3 = at[(lid.y * RT + 3u) * KSTEP + kk];
            let b0 = bt[kk * TILE + lid.x * RT + 0u];
            let b1 = bt[kk * TILE + lid.x * RT + 1u];
            let b2 = bt[kk * TILE + lid.x * RT + 2u];
            let b3 = bt[kk * TILE + lid.x * RT + 3u];
            c00 = fma(a0, b0, c00);
            c01 = fma(a0, b1, c01);
            c02 = fma(a0, b2, c02);
            c03 = fma(a0, b3, c03);
            c10 = fma(a1, b0, c10);
            c11 = fma(a1, b1, c11);
            c12 = fma(a1, b2, c12);
            c13 = fma(a1, b3, c13);
            c20 = fma(a2, b0, c20);
            c21 = fma(a2, b1, c21);
            c22 = fma(a2, b2, c22);
            c23 = fma(a2, b3, c23);
            c30 = fma(a3, b0, c30);
            c31 = fma(a3, b1, c31);
            c32 = fma(a3, b2, c32);
            c33 = fma(a3, b3, c33);
        }
        workgroupBarrier();
    }

    let dst_base = b * pc.m * pc.n;
    if tile_interior {
        let ir0 = row0 + lid.y * RT + 0u;
        let ir1 = row0 + lid.y * RT + 1u;
        let ir2 = row0 + lid.y * RT + 2u;
        let ir3 = row0 + lid.y * RT + 3u;
        dst[dst_base + ir0 * pc.dst_rs + (col0 + lid.x * RT + 0u) * pc.dst_cs] = c00;
        dst[dst_base + ir0 * pc.dst_rs + (col0 + lid.x * RT + 1u) * pc.dst_cs] = c01;
        dst[dst_base + ir0 * pc.dst_rs + (col0 + lid.x * RT + 2u) * pc.dst_cs] = c02;
        dst[dst_base + ir0 * pc.dst_rs + (col0 + lid.x * RT + 3u) * pc.dst_cs] = c03;
        dst[dst_base + ir1 * pc.dst_rs + (col0 + lid.x * RT + 0u) * pc.dst_cs] = c10;
        dst[dst_base + ir1 * pc.dst_rs + (col0 + lid.x * RT + 1u) * pc.dst_cs] = c11;
        dst[dst_base + ir1 * pc.dst_rs + (col0 + lid.x * RT + 2u) * pc.dst_cs] = c12;
        dst[dst_base + ir1 * pc.dst_rs + (col0 + lid.x * RT + 3u) * pc.dst_cs] = c13;
        dst[dst_base + ir2 * pc.dst_rs + (col0 + lid.x * RT + 0u) * pc.dst_cs] = c20;
        dst[dst_base + ir2 * pc.dst_rs + (col0 + lid.x * RT + 1u) * pc.dst_cs] = c21;
        dst[dst_base + ir2 * pc.dst_rs + (col0 + lid.x * RT + 2u) * pc.dst_cs] = c22;
        dst[dst_base + ir2 * pc.dst_rs + (col0 + lid.x * RT + 3u) * pc.dst_cs] = c23;
        dst[dst_base + ir3 * pc.dst_rs + (col0 + lid.x * RT + 0u) * pc.dst_cs] = c30;
        dst[dst_base + ir3 * pc.dst_rs + (col0 + lid.x * RT + 1u) * pc.dst_cs] = c31;
        dst[dst_base + ir3 * pc.dst_rs + (col0 + lid.x * RT + 2u) * pc.dst_cs] = c32;
        dst[dst_base + ir3 * pc.dst_rs + (col0 + lid.x * RT + 3u) * pc.dst_cs] = c33;
        return;
    }
    let r0 = row0 + lid.y * RT + 0u;
    let r1 = row0 + lid.y * RT + 1u;
    let r2 = row0 + lid.y * RT + 2u;
    let r3 = row0 + lid.y * RT + 3u;
    if r0 < pc.m { let cc = col0 + lid.x * RT + 0u; if cc < pc.n { dst[dst_base + r0 * pc.dst_rs + cc * pc.dst_cs] = c00; } }
    if r0 < pc.m { let cc = col0 + lid.x * RT + 1u; if cc < pc.n { dst[dst_base + r0 * pc.dst_rs + cc * pc.dst_cs] = c01; } }
    if r0 < pc.m { let cc = col0 + lid.x * RT + 2u; if cc < pc.n { dst[dst_base + r0 * pc.dst_rs + cc * pc.dst_cs] = c02; } }
    if r0 < pc.m { let cc = col0 + lid.x * RT + 3u; if cc < pc.n { dst[dst_base + r0 * pc.dst_rs + cc * pc.dst_cs] = c03; } }
    if r1 < pc.m { let cc = col0 + lid.x * RT + 0u; if cc < pc.n { dst[dst_base + r1 * pc.dst_rs + cc * pc.dst_cs] = c10; } }
    if r1 < pc.m { let cc = col0 + lid.x * RT + 1u; if cc < pc.n { dst[dst_base + r1 * pc.dst_rs + cc * pc.dst_cs] = c11; } }
    if r1 < pc.m { let cc = col0 + lid.x * RT + 2u; if cc < pc.n { dst[dst_base + r1 * pc.dst_rs + cc * pc.dst_cs] = c12; } }
    if r1 < pc.m { let cc = col0 + lid.x * RT + 3u; if cc < pc.n { dst[dst_base + r1 * pc.dst_rs + cc * pc.dst_cs] = c13; } }
    if r2 < pc.m { let cc = col0 + lid.x * RT + 0u; if cc < pc.n { dst[dst_base + r2 * pc.dst_rs + cc * pc.dst_cs] = c20; } }
    if r2 < pc.m { let cc = col0 + lid.x * RT + 1u; if cc < pc.n { dst[dst_base + r2 * pc.dst_rs + cc * pc.dst_cs] = c21; } }
    if r2 < pc.m { let cc = col0 + lid.x * RT + 2u; if cc < pc.n { dst[dst_base + r2 * pc.dst_rs + cc * pc.dst_cs] = c22; } }
    if r2 < pc.m { let cc = col0 + lid.x * RT + 3u; if cc < pc.n { dst[dst_base + r2 * pc.dst_rs + cc * pc.dst_cs] = c23; } }
    if r3 < pc.m { let cc = col0 + lid.x * RT + 0u; if cc < pc.n { dst[dst_base + r3 * pc.dst_rs + cc * pc.dst_cs] = c30; } }
    if r3 < pc.m { let cc = col0 + lid.x * RT + 1u; if cc < pc.n { dst[dst_base + r3 * pc.dst_rs + cc * pc.dst_cs] = c31; } }
    if r3 < pc.m { let cc = col0 + lid.x * RT + 2u; if cc < pc.n { dst[dst_base + r3 * pc.dst_rs + cc * pc.dst_cs] = c32; } }
    if r3 < pc.m { let cc = col0 + lid.x * RT + 3u; if cc < pc.n { dst[dst_base + r3 * pc.dst_rs + cc * pc.dst_cs] = c33; } }
}
