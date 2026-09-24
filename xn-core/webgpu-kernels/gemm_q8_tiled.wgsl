// q8_0 dequantize-fused tiled GEMM, for m past the point where a row-block
// kernel stops paying.
//   dst[i, j] = sum_l lhs[i, l] * W[j, l],  W held as q8_0 blocks.
// Grid: (ceil(n/32), ceil(m/32), 1); local size (8, 8, 1).
//
// Same 32x32 tile, 8-deep k stage and 4x4 register tile as gemm_tiled.wgsl --
// see that file for why the accumulators are scalars. The only change is where
// the rhs tile comes from: instead of reading f32 weights it unpacks q8_0 on
// the way into workgroup memory, so the inner product loop is untouched and
// the weight crosses the bus at one byte per value.
//
// This kernel exists because gemm_q8.wgsl re-reads the whole weight for every
// four rows. That is the right trade at m = 4 and the wrong one by m = 30,
// where it moves eight passes of weights against this kernel's one -- measured
// at 0.58x of the f32 path before this existed, against roughly 2x after.
// Staging in workgroup memory is what decouples weight traffic from m.
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

const TILE: u32 = 32u;
const KSTEP: u32 = 8u;
const TPB: u32 = 64u;
const RT: u32 = 4u;

var<workgroup> at: array<f32, 256>;  // [32 rows][8 k]
var<workgroup> bt: array<f32, 256>;  // [8 k][32 cols]

// One dequantized weight, W[j, l]. The quants are four to a word and the
// scales one per 32, so a value needs one word load, one byte extract
// (`extractBits` on a signed integer sign-extends) and one scale.
fn wq(j: u32, l: u32, kw: u32, kb: u32) -> f32 {
    let word = qs[j * kw + (l >> 2u)];
    let q = bitcast<i32>(word);
    let v = f32(extractBits(q, (l & 3u) * 8u, 8u));
    return v * scales[j * kb + (l >> 5u)];
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row0 = wid.y * TILE;
    let col0 = wid.x * TILE;
    let tid = lid.y * 8u + lid.x;

    let kw = pc.k >> 2u;
    let kb = pc.k >> 5u;

    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0; var c03 = 0.0;
    var c10 = 0.0; var c11 = 0.0; var c12 = 0.0; var c13 = 0.0;
    var c20 = 0.0; var c21 = 0.0; var c22 = 0.0; var c23 = 0.0;
    var c30 = 0.0; var c31 = 0.0; var c32 = 0.0; var c33 = 0.0;

    let tile_interior = (row0 + TILE) <= pc.m && (col0 + TILE) <= pc.n;

    let nk = (pc.k + KSTEP - 1u) / KSTEP;
    for (var kt = 0u; kt < nk; kt = kt + 1u) {
        let k0 = kt * KSTEP;
        let interior = tile_interior && (k0 + KSTEP) <= pc.k;

        // 256 elements per tile, 4 per thread. `bc = idx / KSTEP` walks k on
        // consecutive threads, which is the contiguous axis of a row-major
        // weight -- the same choice gemm_tiled makes for `rhs_rs == 1`.
        if interior {
            for (var s = 0u; s < 4u; s = s + 1u) {
                let idx = tid + s * TPB;
                let ar = idx / KSTEP;
                let ak = idx % KSTEP;
                at[idx] = lhs[(row0 + ar) * pc.k + k0 + ak];
                let bc = idx / KSTEP;
                let bk = idx % KSTEP;
                bt[bk * TILE + bc] = wq(col0 + bc, k0 + bk, kw, kb);
            }
        } else {
            for (var s = 0u; s < 4u; s = s + 1u) {
                let idx = tid + s * TPB;
                let ar = idx / KSTEP;
                let ak = idx % KSTEP;
                let grow = row0 + ar;
                let gk = k0 + ak;
                if grow < pc.m && gk < pc.k {
                    at[idx] = lhs[grow * pc.k + gk];
                } else {
                    at[idx] = 0.0;
                }
                let bc = idx / KSTEP;
                let bk = idx % KSTEP;
                let gcol = col0 + bc;
                let gk2 = k0 + bk;
                if gcol < pc.n && gk2 < pc.k {
                    bt[bk * TILE + bc] = wq(gcol, gk2, kw, kb);
                } else {
                    bt[bk * TILE + bc] = 0.0;
                }
            }
        }
        workgroupBarrier();

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

    let r0 = row0 + lid.y * RT + 0u;
    let r1 = row0 + lid.y * RT + 1u;
    let r2 = row0 + lid.y * RT + 2u;
    let r3 = row0 + lid.y * RT + 3u;
    let q0 = col0 + lid.x * RT + 0u;
    let q1 = col0 + lid.x * RT + 1u;
    let q2 = col0 + lid.x * RT + 2u;
    let q3 = col0 + lid.x * RT + 3u;
    if tile_interior {
        dst[r0 * pc.n + q0] = c00;
        dst[r0 * pc.n + q1] = c01;
        dst[r0 * pc.n + q2] = c02;
        dst[r0 * pc.n + q3] = c03;
        dst[r1 * pc.n + q0] = c10;
        dst[r1 * pc.n + q1] = c11;
        dst[r1 * pc.n + q2] = c12;
        dst[r1 * pc.n + q3] = c13;
        dst[r2 * pc.n + q0] = c20;
        dst[r2 * pc.n + q1] = c21;
        dst[r2 * pc.n + q2] = c22;
        dst[r2 * pc.n + q3] = c23;
        dst[r3 * pc.n + q0] = c30;
        dst[r3 * pc.n + q1] = c31;
        dst[r3 * pc.n + q2] = c32;
        dst[r3 * pc.n + q3] = c33;
        return;
    }
    if r0 < pc.m { if q0 < pc.n { dst[r0 * pc.n + q0] = c00; } if q1 < pc.n { dst[r0 * pc.n + q1] = c01; } if q2 < pc.n { dst[r0 * pc.n + q2] = c02; } if q3 < pc.n { dst[r0 * pc.n + q3] = c03; } }
    if r1 < pc.m { if q0 < pc.n { dst[r1 * pc.n + q0] = c10; } if q1 < pc.n { dst[r1 * pc.n + q1] = c11; } if q2 < pc.n { dst[r1 * pc.n + q2] = c12; } if q3 < pc.n { dst[r1 * pc.n + q3] = c13; } }
    if r2 < pc.m { if q0 < pc.n { dst[r2 * pc.n + q0] = c20; } if q1 < pc.n { dst[r2 * pc.n + q1] = c21; } if q2 < pc.n { dst[r2 * pc.n + q2] = c22; } if q3 < pc.n { dst[r2 * pc.n + q3] = c23; } }
    if r3 < pc.m { if q0 < pc.n { dst[r3 * pc.n + q0] = c30; } if q1 < pc.n { dst[r3 * pc.n + q1] = c31; } if q2 < pc.n { dst[r3 * pc.n + q2] = c32; } if q3 < pc.n { dst[r3 * pc.n + q3] = c33; } }
}
