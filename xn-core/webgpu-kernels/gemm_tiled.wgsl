// Tiled shared-memory GEMM (f32 accumulation), general strides + batch.
//   dst[b, i, j] = sum_l lhs[b, i, l] * rhs[b, l, j]
// Element offsets (same convention as the naive gemm):
//   lhs: lhs_o + b*lhs_b_stride + i*lhs_rs + l*lhs_cs
//   rhs: rhs_o + b*rhs_b_stride + l*rhs_rs + j*rhs_cs
//   dst:              b*m*n     + i*dst_rs + j*dst_cs
// Grid: (ceil(n/TILE), ceil(m/TILE), batch); local size (TILE, TILE, 1).
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

const TILE: u32 = 16u;
var<workgroup> lt: array<array<f32, 16>, 16>;
var<workgroup> rt: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wid.z;
    let row = gid.y;
    let col = gid.x;
    let ly = lid.y;
    let lx = lid.x;

    let lhs_base = pc.lhs_o + b * pc.lhs_b_stride;
    let rhs_base = pc.rhs_o + b * pc.rhs_b_stride;

    var acc = 0.0;
    let num_tiles = (pc.k + TILE - 1u) / TILE;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let l_lhs = t * TILE + lx;
        let l_rhs = t * TILE + ly;
        if row < pc.m && l_lhs < pc.k {
            lt[ly][lx] = f32(lhs[lhs_base + row * pc.lhs_rs + l_lhs * pc.lhs_cs]);
        } else {
            lt[ly][lx] = 0.0;
        }
        if col < pc.n && l_rhs < pc.k {
            rt[ly][lx] = f32(rhs[rhs_base + l_rhs * pc.rhs_rs + col * pc.rhs_cs]);
        } else {
            rt[ly][lx] = 0.0;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < TILE; kk = kk + 1u) {
            acc = acc + lt[ly][kk] * rt[kk][lx];
        }
        workgroupBarrier();
    }

    if row < pc.m && col < pc.n {
        dst[b * pc.m * pc.n + row * pc.dst_rs + col * pc.dst_cs] = S(acc);
    }
}
