// Tiled GEMM with a 2x4 register tile, general strides + batch.
//   dst[b, i, j] = sum_l lhs[b, i, l] * rhs[b, l, j]
// Element offsets (same convention as the naive gemm):
//   lhs: lhs_o + b*lhs_b_stride + i*lhs_rs + l*lhs_cs
//   rhs: rhs_o + b*rhs_b_stride + l*rhs_rs + j*rhs_cs
//   dst:              b*m*n     + i*dst_rs + j*dst_cs
// Grid: (ceil(n/TN), ceil(m/TM), batch); local size (8, 8, 1).
//
// The tile is 16 rows x 32 columns, not square, because this model's GEMMs are
// not square. Per-kernel GPU timing puts 45% of a Phonon q8 frame in this
// kernel, and an (m, n, k) histogram of its calls is dominated by m = 16 -- the
// Mimi transformer's window -- with m = 96 and m = 480 from the SEANet convs.
// A 32-row tile leaves half its rows idle on the common case; a 16-row one
// fills exactly.
//
// Each thread holds 2 rows x 4 columns in registers, so one k step reads 2 + 4
// staged values for 8 multiply-adds. The kernel this replaced computed one
// output per thread, reading 2 staged values for 1 -- memory-bound on a
// compute-bound problem. The accumulators are named scalars rather than an
// array because naga spills a dynamically indexed function-scope array to
// thread-private memory instead of keeping it in registers.
//
// Shape follows TVM dlight's matmul schedule (read out of WebLLM's compiled
// artifact), retiled for m = 16.
//
// Origin, Apache-2.0:
//   https://github.com/apache/tvm/blob/main/python/tvm/s_tir/dlight/gpu/matmul.py
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

const TM: u32 = 16u;     // output rows per workgroup
const TN: u32 = 32u;     // output columns per workgroup
const KSTEP: u32 = 8u;   // k staged per iteration
const TPB: u32 = 64u;    // threads per workgroup (8x8)
const RM: u32 = 2u;      // register rows per thread
const RN: u32 = 4u;      // register columns per thread

// lhs tile: [16 rows][8 k]   rhs tile: [8 k][32 cols]
var<workgroup> at: array<f32, 128>;
var<workgroup> bt: array<f32, 256>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wid.z;
    let row0 = wid.y * TM;
    let col0 = wid.x * TN;
    let tid = lid.y * 8u + lid.x;

    let lhs_base = pc.lhs_o + b * pc.lhs_b_stride;
    let rhs_base = pc.rhs_o + b * pc.rhs_b_stride;

    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0; var c03 = 0.0;
    var c10 = 0.0; var c11 = 0.0; var c12 = 0.0; var c13 = 0.0;

    let tile_interior = (row0 + TM) <= pc.m && (col0 + TN) <= pc.n;

    let nk = (pc.k + KSTEP - 1u) / KSTEP;
    for (var kt = 0u; kt < nk; kt = kt + 1u) {
        let k0 = kt * KSTEP;
        let interior = tile_interior && (k0 + KSTEP) <= pc.k;

        // Stage both tiles: 128 + 256 elements over 64 threads, so 2 of `at`
        // and 4 of `bt` each. `interior` drops the per-element bounds check
        // for tiles fully in range, which is every tile but the last
        // row/column/k slice.
        if interior {
            for (var s = 0u; s < 2u; s = s + 1u) {
                let idx = tid + s * TPB;
                let ar = idx / KSTEP;
                let ak = idx % KSTEP;
                at[idx] = f32(lhs[lhs_base + (row0 + ar) * pc.lhs_rs + (k0 + ak) * pc.lhs_cs]);
            }
            for (var s = 0u; s < 4u; s = s + 1u) {
                let idx = tid + s * TPB;
                // Consecutive threads must walk whichever axis is contiguous:
                // k when rhs_rs == 1 (matmul_t, every linear layer and every
                // im2col conv), else columns.
                var bk: u32;
                var bc: u32;
                if pc.rhs_rs == 1u {
                    bc = idx / KSTEP;
                    bk = idx % KSTEP;
                } else {
                    bk = idx / TN;
                    bc = idx % TN;
                }
                bt[bk * TN + bc] =
                    f32(rhs[rhs_base + (k0 + bk) * pc.rhs_rs + (col0 + bc) * pc.rhs_cs]);
            }
        } else {
            for (var s = 0u; s < 2u; s = s + 1u) {
                let idx = tid + s * TPB;
                let ar = idx / KSTEP;
                let ak = idx % KSTEP;
                let grow = row0 + ar;
                let gk = k0 + ak;
                if grow < pc.m && gk < pc.k {
                    at[idx] = f32(lhs[lhs_base + grow * pc.lhs_rs + gk * pc.lhs_cs]);
                } else {
                    at[idx] = 0.0;
                }
            }
            for (var s = 0u; s < 4u; s = s + 1u) {
                let idx = tid + s * TPB;
                var bk: u32;
                var bc: u32;
                if pc.rhs_rs == 1u {
                    bc = idx / KSTEP;
                    bk = idx % KSTEP;
                } else {
                    bk = idx / TN;
                    bc = idx % TN;
                }
                let gcol = col0 + bc;
                let gk2 = k0 + bk;
                if gcol < pc.n && gk2 < pc.k {
                    bt[bk * TN + bc] = f32(rhs[rhs_base + gk2 * pc.rhs_rs + gcol * pc.rhs_cs]);
                } else {
                    bt[bk * TN + bc] = 0.0;
                }
            }
        }
        workgroupBarrier();

        // Six staged reads buy eight multiply-adds.
        for (var kk = 0u; kk < KSTEP; kk = kk + 1u) {
            let a0 = at[(lid.y * RM + 0u) * KSTEP + kk];
            let a1 = at[(lid.y * RM + 1u) * KSTEP + kk];
            let b0 = bt[kk * TN + lid.x * RN + 0u];
            let b1 = bt[kk * TN + lid.x * RN + 1u];
            let b2 = bt[kk * TN + lid.x * RN + 2u];
            let b3 = bt[kk * TN + lid.x * RN + 3u];
            c00 = fma(a0, b0, c00);
            c01 = fma(a0, b1, c01);
            c02 = fma(a0, b2, c02);
            c03 = fma(a0, b3, c03);
            c10 = fma(a1, b0, c10);
            c11 = fma(a1, b1, c11);
            c12 = fma(a1, b2, c12);
            c13 = fma(a1, b3, c13);
        }
        workgroupBarrier();
    }

    let dst_base = b * pc.m * pc.n;
    let r0 = row0 + lid.y * RM + 0u;
    let r1 = row0 + lid.y * RM + 1u;
    let q0 = col0 + lid.x * RN + 0u;
    let q1 = col0 + lid.x * RN + 1u;
    let q2 = col0 + lid.x * RN + 2u;
    let q3 = col0 + lid.x * RN + 3u;
    if tile_interior {
        dst[dst_base + r0 * pc.dst_rs + q0 * pc.dst_cs] = S(c00);
        dst[dst_base + r0 * pc.dst_rs + q1 * pc.dst_cs] = S(c01);
        dst[dst_base + r0 * pc.dst_rs + q2 * pc.dst_cs] = S(c02);
        dst[dst_base + r0 * pc.dst_rs + q3 * pc.dst_cs] = S(c03);
        dst[dst_base + r1 * pc.dst_rs + q0 * pc.dst_cs] = S(c10);
        dst[dst_base + r1 * pc.dst_rs + q1 * pc.dst_cs] = S(c11);
        dst[dst_base + r1 * pc.dst_rs + q2 * pc.dst_cs] = S(c12);
        dst[dst_base + r1 * pc.dst_rs + q3 * pc.dst_cs] = S(c13);
        return;
    }
    if r0 < pc.m {
        if q0 < pc.n { dst[dst_base + r0 * pc.dst_rs + q0 * pc.dst_cs] = S(c00); }
        if q1 < pc.n { dst[dst_base + r0 * pc.dst_rs + q1 * pc.dst_cs] = S(c01); }
        if q2 < pc.n { dst[dst_base + r0 * pc.dst_rs + q2 * pc.dst_cs] = S(c02); }
        if q3 < pc.n { dst[dst_base + r0 * pc.dst_rs + q3 * pc.dst_cs] = S(c03); }
    }
    if r1 < pc.m {
        if q0 < pc.n { dst[dst_base + r1 * pc.dst_rs + q0 * pc.dst_cs] = S(c10); }
        if q1 < pc.n { dst[dst_base + r1 * pc.dst_rs + q1 * pc.dst_cs] = S(c11); }
        if q2 < pc.n { dst[dst_base + r1 * pc.dst_rs + q2 * pc.dst_cs] = S(c12); }
        if q3 < pc.n { dst[dst_base + r1 * pc.dst_rs + q3 * pc.dst_cs] = S(c13); }
    }
}
