// Skinny GEMM: 1 < m <= 16, i.e. a vocoder frame.
//   dst[b, i, j] = sum_l lhs[b, i, l] * rhs[b, l, j]
// Grid: (ceil(n/NT), ceil(m/MT), batch); local size 64.
//
// `gemm_tiled` takes its parallelism from the output: a 32x32 tile, so
// ceil(n/32) * ceil(m/32) workgroups. At m = 16 that is one row of tiles --
// sixteen workgroups at n = 512 -- and half of every tile's rows are computed
// and then discarded. An Apple GPU is idle at that width: the kernel measures
// 66 GFLOP/s at m = 16 against 590 for the same kernel at m = 128, and 8x the
// arithmetic at m = 128 finishes in less wall-clock than m = 16 does.
//
// Mimi decodes 16 positions at a time, so nearly every matmul the vocoder
// issues is that shape. This kernel takes its parallelism from k instead, the
// way gemv.wgsl does: each 64-thread workgroup owns an MT x NT patch of the
// output, every thread walks a disjoint slice of k, and a single workgroup
// reduction sums the partials at the end. The grid becomes
// ceil(n/NT) * ceil(m/MT) -- 512 workgroups at (16, 512, k) rather than 16 --
// and no thread computes a row that will be thrown away.
//
// MT x NT = 8 x 2 keeps sixteen accumulators, the same register pressure
// gemm_tiled carries, and was the best of five shapes measured across the
// vocoder's matmuls. Each k step loads NT weights and MT activations for
// MT * NT * 4 multiply-adds via the vec4 path.
//
// Only worth it where gemm_tiled is actually starved; see SKINNY_MAX_GROUPS in
// mod.rs for the gate. Summing k in a different order than gemm_tiled is
// within float reassociation -- measured against the CPU backend it is
// slightly more accurate, not less, since a reduction tree beats a long
// sequential sum.
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

const TPB: u32 = 64u;
const MT: u32 = 8u;
const NT: u32 = 2u;
var<workgroup> sh: array<f32, 1024u>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col0 = wid.x * NT;
    let row0 = wid.y * MT;
    let b = wid.z;
    let tid = lid.x;
    let lbase = pc.lhs_o + b * pc.lhs_b_stride;
    let rbase = pc.rhs_o + b * pc.rhs_b_stride;
    let dst_base = b * pc.m * pc.n;
    let mlast = pc.m - 1u;
    let l0 = lbase + min(row0 + 0u, mlast) * pc.lhs_rs;
    let l1 = lbase + min(row0 + 1u, mlast) * pc.lhs_rs;
    let l2 = lbase + min(row0 + 2u, mlast) * pc.lhs_rs;
    let l3 = lbase + min(row0 + 3u, mlast) * pc.lhs_rs;
    let l4 = lbase + min(row0 + 4u, mlast) * pc.lhs_rs;
    let l5 = lbase + min(row0 + 5u, mlast) * pc.lhs_rs;
    let l6 = lbase + min(row0 + 6u, mlast) * pc.lhs_rs;
    let l7 = lbase + min(row0 + 7u, mlast) * pc.lhs_rs;
    let rb0 = rbase + (col0 + 0u) * pc.rhs_cs;
    let rb1 = rbase + (col0 + 1u) * pc.rhs_cs;
    var c0_0 = 0.0;
    var c0_1 = 0.0;
    var c1_0 = 0.0;
    var c1_1 = 0.0;
    var c2_0 = 0.0;
    var c2_1 = 0.0;
    var c3_0 = 0.0;
    var c3_1 = 0.0;
    var c4_0 = 0.0;
    var c4_1 = 0.0;
    var c5_0 = 0.0;
    var c5_1 = 0.0;
    var c6_0 = 0.0;
    var c6_1 = 0.0;
    var c7_0 = 0.0;
    var c7_1 = 0.0;

    let vec_ok = pc.rhs_rs == 1u && pc.lhs_cs == 1u && pc.k >= 4u
        && (rbase & 3u) == 0u && (pc.rhs_cs & 3u) == 0u;

    if vec_ok {
        let k4 = pc.k >> 2u;
        let kbulk = k4 << 2u;
        let v0 = rb0 >> 2u;
        let v1 = rb1 >> 2u;
        for (var g = tid; g < k4; g = g + TPB) {
            let l = g * 4u;
            let a0_0 = lhs[l0 + l + 0u];
            let a0_1 = lhs[l0 + l + 1u];
            let a0_2 = lhs[l0 + l + 2u];
            let a0_3 = lhs[l0 + l + 3u];
            let a1_0 = lhs[l1 + l + 0u];
            let a1_1 = lhs[l1 + l + 1u];
            let a1_2 = lhs[l1 + l + 2u];
            let a1_3 = lhs[l1 + l + 3u];
            let a2_0 = lhs[l2 + l + 0u];
            let a2_1 = lhs[l2 + l + 1u];
            let a2_2 = lhs[l2 + l + 2u];
            let a2_3 = lhs[l2 + l + 3u];
            let a3_0 = lhs[l3 + l + 0u];
            let a3_1 = lhs[l3 + l + 1u];
            let a3_2 = lhs[l3 + l + 2u];
            let a3_3 = lhs[l3 + l + 3u];
            let a4_0 = lhs[l4 + l + 0u];
            let a4_1 = lhs[l4 + l + 1u];
            let a4_2 = lhs[l4 + l + 2u];
            let a4_3 = lhs[l4 + l + 3u];
            let a5_0 = lhs[l5 + l + 0u];
            let a5_1 = lhs[l5 + l + 1u];
            let a5_2 = lhs[l5 + l + 2u];
            let a5_3 = lhs[l5 + l + 3u];
            let a6_0 = lhs[l6 + l + 0u];
            let a6_1 = lhs[l6 + l + 1u];
            let a6_2 = lhs[l6 + l + 2u];
            let a6_3 = lhs[l6 + l + 3u];
            let a7_0 = lhs[l7 + l + 0u];
            let a7_1 = lhs[l7 + l + 1u];
            let a7_2 = lhs[l7 + l + 2u];
            let a7_3 = lhs[l7 + l + 3u];
            let w0 = rhs4[v0 + g];
            let w1 = rhs4[v1 + g];
            c0_0 = fma(w0.x, a0_0, c0_0);
            c0_0 = fma(w0.y, a0_1, c0_0);
            c0_0 = fma(w0.z, a0_2, c0_0);
            c0_0 = fma(w0.w, a0_3, c0_0);
            c0_1 = fma(w1.x, a0_0, c0_1);
            c0_1 = fma(w1.y, a0_1, c0_1);
            c0_1 = fma(w1.z, a0_2, c0_1);
            c0_1 = fma(w1.w, a0_3, c0_1);
            c1_0 = fma(w0.x, a1_0, c1_0);
            c1_0 = fma(w0.y, a1_1, c1_0);
            c1_0 = fma(w0.z, a1_2, c1_0);
            c1_0 = fma(w0.w, a1_3, c1_0);
            c1_1 = fma(w1.x, a1_0, c1_1);
            c1_1 = fma(w1.y, a1_1, c1_1);
            c1_1 = fma(w1.z, a1_2, c1_1);
            c1_1 = fma(w1.w, a1_3, c1_1);
            c2_0 = fma(w0.x, a2_0, c2_0);
            c2_0 = fma(w0.y, a2_1, c2_0);
            c2_0 = fma(w0.z, a2_2, c2_0);
            c2_0 = fma(w0.w, a2_3, c2_0);
            c2_1 = fma(w1.x, a2_0, c2_1);
            c2_1 = fma(w1.y, a2_1, c2_1);
            c2_1 = fma(w1.z, a2_2, c2_1);
            c2_1 = fma(w1.w, a2_3, c2_1);
            c3_0 = fma(w0.x, a3_0, c3_0);
            c3_0 = fma(w0.y, a3_1, c3_0);
            c3_0 = fma(w0.z, a3_2, c3_0);
            c3_0 = fma(w0.w, a3_3, c3_0);
            c3_1 = fma(w1.x, a3_0, c3_1);
            c3_1 = fma(w1.y, a3_1, c3_1);
            c3_1 = fma(w1.z, a3_2, c3_1);
            c3_1 = fma(w1.w, a3_3, c3_1);
            c4_0 = fma(w0.x, a4_0, c4_0);
            c4_0 = fma(w0.y, a4_1, c4_0);
            c4_0 = fma(w0.z, a4_2, c4_0);
            c4_0 = fma(w0.w, a4_3, c4_0);
            c4_1 = fma(w1.x, a4_0, c4_1);
            c4_1 = fma(w1.y, a4_1, c4_1);
            c4_1 = fma(w1.z, a4_2, c4_1);
            c4_1 = fma(w1.w, a4_3, c4_1);
            c5_0 = fma(w0.x, a5_0, c5_0);
            c5_0 = fma(w0.y, a5_1, c5_0);
            c5_0 = fma(w0.z, a5_2, c5_0);
            c5_0 = fma(w0.w, a5_3, c5_0);
            c5_1 = fma(w1.x, a5_0, c5_1);
            c5_1 = fma(w1.y, a5_1, c5_1);
            c5_1 = fma(w1.z, a5_2, c5_1);
            c5_1 = fma(w1.w, a5_3, c5_1);
            c6_0 = fma(w0.x, a6_0, c6_0);
            c6_0 = fma(w0.y, a6_1, c6_0);
            c6_0 = fma(w0.z, a6_2, c6_0);
            c6_0 = fma(w0.w, a6_3, c6_0);
            c6_1 = fma(w1.x, a6_0, c6_1);
            c6_1 = fma(w1.y, a6_1, c6_1);
            c6_1 = fma(w1.z, a6_2, c6_1);
            c6_1 = fma(w1.w, a6_3, c6_1);
            c7_0 = fma(w0.x, a7_0, c7_0);
            c7_0 = fma(w0.y, a7_1, c7_0);
            c7_0 = fma(w0.z, a7_2, c7_0);
            c7_0 = fma(w0.w, a7_3, c7_0);
            c7_1 = fma(w1.x, a7_0, c7_1);
            c7_1 = fma(w1.y, a7_1, c7_1);
            c7_1 = fma(w1.z, a7_2, c7_1);
            c7_1 = fma(w1.w, a7_3, c7_1);
        }
        for (var l = kbulk + tid; l < pc.k; l = l + TPB) {
            let a0 = lhs[l0 + l];
            let a1 = lhs[l1 + l];
            let a2 = lhs[l2 + l];
            let a3 = lhs[l3 + l];
            let a4 = lhs[l4 + l];
            let a5 = lhs[l5 + l];
            let a6 = lhs[l6 + l];
            let a7 = lhs[l7 + l];
            let w0 = rhs[rb0 + l];
            let w1 = rhs[rb1 + l];
            c0_0 = fma(w0, a0, c0_0);
            c0_1 = fma(w1, a0, c0_1);
            c1_0 = fma(w0, a1, c1_0);
            c1_1 = fma(w1, a1, c1_1);
            c2_0 = fma(w0, a2, c2_0);
            c2_1 = fma(w1, a2, c2_1);
            c3_0 = fma(w0, a3, c3_0);
            c3_1 = fma(w1, a3, c3_1);
            c4_0 = fma(w0, a4, c4_0);
            c4_1 = fma(w1, a4, c4_1);
            c5_0 = fma(w0, a5, c5_0);
            c5_1 = fma(w1, a5, c5_1);
            c6_0 = fma(w0, a6, c6_0);
            c6_1 = fma(w1, a6, c6_1);
            c7_0 = fma(w0, a7, c7_0);
            c7_1 = fma(w1, a7, c7_1);
        }
    } else {
        for (var l = tid; l < pc.k; l = l + TPB) {
            let lo = l * pc.rhs_rs;
            let a0 = lhs[l0 + l * pc.lhs_cs];
            let a1 = lhs[l1 + l * pc.lhs_cs];
            let a2 = lhs[l2 + l * pc.lhs_cs];
            let a3 = lhs[l3 + l * pc.lhs_cs];
            let a4 = lhs[l4 + l * pc.lhs_cs];
            let a5 = lhs[l5 + l * pc.lhs_cs];
            let a6 = lhs[l6 + l * pc.lhs_cs];
            let a7 = lhs[l7 + l * pc.lhs_cs];
            let w0 = rhs[rb0 + lo];
            let w1 = rhs[rb1 + lo];
            c0_0 = fma(w0, a0, c0_0);
            c0_1 = fma(w1, a0, c0_1);
            c1_0 = fma(w0, a1, c1_0);
            c1_1 = fma(w1, a1, c1_1);
            c2_0 = fma(w0, a2, c2_0);
            c2_1 = fma(w1, a2, c2_1);
            c3_0 = fma(w0, a3, c3_0);
            c3_1 = fma(w1, a3, c3_1);
            c4_0 = fma(w0, a4, c4_0);
            c4_1 = fma(w1, a4, c4_1);
            c5_0 = fma(w0, a5, c5_0);
            c5_1 = fma(w1, a5, c5_1);
            c6_0 = fma(w0, a6, c6_0);
            c6_1 = fma(w1, a6, c6_1);
            c7_0 = fma(w0, a7, c7_0);
            c7_1 = fma(w1, a7, c7_1);
        }
    }

    let o = tid * (MT * NT);
    sh[o + 0u] = c0_0;
    sh[o + 1u] = c0_1;
    sh[o + 2u] = c1_0;
    sh[o + 3u] = c1_1;
    sh[o + 4u] = c2_0;
    sh[o + 5u] = c2_1;
    sh[o + 6u] = c3_0;
    sh[o + 7u] = c3_1;
    sh[o + 8u] = c4_0;
    sh[o + 9u] = c4_1;
    sh[o + 10u] = c5_0;
    sh[o + 11u] = c5_1;
    sh[o + 12u] = c6_0;
    sh[o + 13u] = c6_1;
    sh[o + 14u] = c7_0;
    sh[o + 15u] = c7_1;
    workgroupBarrier();
    for (var st = TPB >> 1u; st > 0u; st = st >> 1u) {
        if tid < st {
            let p = tid * (MT * NT);
            let q = (tid + st) * (MT * NT);
            for (var i = 0u; i < MT * NT; i = i + 1u) {
                sh[p + i] = sh[p + i] + sh[q + i];
            }
        }
        workgroupBarrier();
    }
    if tid < MT * NT {
        let r = row0 + tid / NT;
        let c = col0 + tid % NT;
        if r < pc.m && c < pc.n {
            dst[dst_base + r * pc.dst_rs + c * pc.dst_cs] = sh[tid];
        }
    }
}
