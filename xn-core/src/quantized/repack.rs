//! Column-interleaved `q8_0` weights and the matmul kernels that consume them.
//!
//! Single-token decode is bandwidth-bound: the whole weight matrix is streamed once per token
//! and almost no arithmetic is reused. The `sgemm` path in [`super::k_quants`] reads four
//! output rows at a time straight out of the stored layout, which means four pointers walking
//! four separate regions `lda * 34` bytes apart, with every 34-byte block unaligned. Four
//! concurrent strided streams are much harder on the prefetcher than one contiguous one.
//!
//! This module stores the weights the way ggml's CPU repack buffer does (`q8_0_4x8`): four
//! adjacent output columns interleaved into one block, in runs of `BLOCKLEN` values. A tile of
//! four output columns is then one sequential, aligned walk over memory, and the `sdot` pairs
//! fall out of the layout for free -- a 16-byte load covers eight values of two columns, so one
//! `sdot` against a duplicated 8-value run of the activation produces partial sums for both.
//!
//! The block layout is byte-identical to ggml's `block_q8_0x4`, so the kernels here are a port
//! of `ggml_gemv_q8_0_4x8_q8_0` (see `ggml/src/ggml-cpu/arch/arm/repack.cpp`). Interleaving is
//! purely a storage decision: [`Q8_0x4::unpack`] recovers the original `q8_0` block stream, so
//! dequantization and GGUF writing still see the on-disk format.

use super::GgmlDType;
use super::k_quants::{BlockQ8_0, GgmlType, QK8_0};
use crate::Result;
use half::f16;
use std::borrow::Cow;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use core::arch::aarch64::*;

/// Output columns fused into one block.
pub const NCOLS: usize = 4;
/// Values of one column stored contiguously before moving to the next.
pub const BLOCKLEN: usize = 8;

/// Four `q8_0` columns interleaved, matching ggml's `block_q8_0x4` with an interleave of 8.
///
/// `qs` holds `NCOLS * QK8_0` values as 16 runs of `BLOCKLEN`, cycling through the four columns:
/// `c0[0..8] c1[0..8] c2[0..8] c3[0..8] c0[8..16] ...`.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ8_0x4 {
    pub d: [f16; NCOLS],
    pub qs: [i8; QK8_0 * NCOLS],
}

const _: () =
    assert!(std::mem::size_of::<BlockQ8_0x4>() == NCOLS * std::mem::size_of::<BlockQ8_0>());

impl BlockQ8_0x4 {
    fn zeros() -> Self {
        Self { d: [f16::ZERO; NCOLS], qs: [0i8; QK8_0 * NCOLS] }
    }

    /// Fuse four single-column blocks. Mirrors ggml's `make_block_q8_0x4`.
    fn interleave(src: [&BlockQ8_0; NCOLS]) -> Self {
        let mut out = Self::zeros();
        for (d, blk) in out.d.iter_mut().zip(src.iter()) {
            *d = blk.d;
        }
        // `runs` runs of BLOCKLEN, round-robin over the four columns.
        let runs = QK8_0 * NCOLS / BLOCKLEN;
        for i in 0..runs {
            let col = i % NCOLS;
            let src_off = (i / NCOLS) * BLOCKLEN;
            let dst_off = i * BLOCKLEN;
            out.qs[dst_off..dst_off + BLOCKLEN]
                .copy_from_slice(&src[col].qs[src_off..src_off + BLOCKLEN]);
        }
        out
    }

    /// Inverse of [`Self::interleave`], writing the four columns back out.
    fn deinterleave(&self, dst: &mut [BlockQ8_0; NCOLS]) {
        for (blk, d) in dst.iter_mut().zip(self.d.iter()) {
            blk.d = *d;
        }
        let runs = QK8_0 * NCOLS / BLOCKLEN;
        for i in 0..runs {
            let col = i % NCOLS;
            let dst_off = (i / NCOLS) * BLOCKLEN;
            let src_off = i * BLOCKLEN;
            dst[col].qs[dst_off..dst_off + BLOCKLEN]
                .copy_from_slice(&self.qs[src_off..src_off + BLOCKLEN]);
        }
    }
}

/// A `q8_0` weight matrix held in the interleaved layout.
///
/// Rows of the original `[n, k]` matrix become the "columns" the kernels iterate over, since
/// `matmul_t` contracts against the last axis of both operands.
pub struct Q8_0x4 {
    blocks: Vec<BlockQ8_0x4>,
    /// Output width, i.e. rows of the stored `[n, k]` weight matrix. Always a multiple of `NCOLS`.
    n: usize,
    /// Blocks per row, `k / QK8_0`.
    kb: usize,
}

/// Whether a `[n, k]` `q8_0` tensor can be held interleaved.
pub fn is_eligible(dims: &[usize]) -> bool {
    // Only the 2-D weight of a linear layer benefits: anything else is either tiny or is read
    // by a path that wants the plain layout.
    matches!(dims, [n, k] if n % NCOLS == 0 && k % QK8_0 == 0 && *n > 0 && *k > 0)
}

impl Q8_0x4 {
    /// Interleave a row-major stream of `q8_0` blocks. `src` is `n` rows of `k / QK8_0` blocks.
    pub fn from_q8_0(src: &[BlockQ8_0], n: usize, k: usize) -> Result<Self> {
        if !is_eligible(&[n, k]) {
            crate::bail!("shape [{n}, {k}] cannot be interleaved as q8_0_4x8")
        }
        let kb = k / QK8_0;
        if src.len() != n * kb {
            crate::bail!("expected {} q8_0 blocks for [{n}, {k}], got {}", n * kb, src.len())
        }
        let mut blocks = Vec::with_capacity(n / NCOLS * kb);
        for row0 in (0..n).step_by(NCOLS) {
            for b in 0..kb {
                blocks.push(BlockQ8_0x4::interleave([
                    &src[(row0) * kb + b],
                    &src[(row0 + 1) * kb + b],
                    &src[(row0 + 2) * kb + b],
                    &src[(row0 + 3) * kb + b],
                ]));
            }
        }
        Ok(Self { blocks, n, kb })
    }

    /// Recover the plain row-major `q8_0` block stream.
    pub fn unpack(&self) -> Vec<BlockQ8_0> {
        let mut out = vec![BlockQ8_0::zeros(); self.n * self.kb];
        let mut tmp =
            [BlockQ8_0::zeros(), BlockQ8_0::zeros(), BlockQ8_0::zeros(), BlockQ8_0::zeros()];
        for (g, row0) in (0..self.n).step_by(NCOLS).enumerate() {
            for b in 0..self.kb {
                self.blocks[g * self.kb + b].deinterleave(&mut tmp);
                for (c, t) in tmp.iter().enumerate() {
                    out[(row0 + c) * self.kb + b] = t.clone();
                }
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

impl super::QuantizedType for Q8_0x4 {
    fn dtype(&self) -> GgmlDType {
        // The interleaving is a storage detail; callers still see a q8_0 tensor.
        GgmlDType::Q8_0
    }

    fn block_size(&self) -> usize {
        QK8_0
    }

    fn size(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockQ8_0x4>()
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.size()
    }

    fn as_ptr(&self) -> *const u8 {
        self.blocks.as_ptr() as *const u8
    }

    fn raw_data(&self) -> Result<Cow<'_, [u8]>> {
        // De-interleave so anything that persists the tensor writes real q8_0 bytes.
        let plain = self.unpack();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                plain.as_ptr() as *const u8,
                std::mem::size_of_val(plain.as_slice()),
            )
        };
        Ok(Cow::Owned(bytes.to_vec()))
    }

    fn dequantize(&self, elem_count: usize) -> Result<Vec<f32>> {
        let plain = self.unpack();
        let mut ys = vec![0.0f32; elem_count];
        BlockQ8_0::to_float(&plain, &mut ys)?;
        Ok(ys)
    }

    fn from_float(&mut self, xs: &[f32]) -> Result<()> {
        let mut plain = vec![BlockQ8_0::zeros(); self.n * self.kb];
        BlockQ8_0::from_float(xs, &mut plain)?;
        *self = Self::from_q8_0(&plain, self.n, self.kb * QK8_0)?;
        Ok(())
    }

    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        let (m, k, n) = mkn;
        if n != self.n {
            crate::bail!("matmul_t: n mismatch, weights hold {} but got {n}", self.n)
        }
        if k != self.kb * QK8_0 {
            crate::bail!("matmul_t: k mismatch, weights hold {} but got {k}", self.kb * QK8_0)
        }
        if lhs.len() != m * k {
            crate::bail!("matmul_t: expected {} lhs elements, got {}", m * k, lhs.len())
        }
        if dst.len() < m * n {
            crate::bail!("matmul_t: dst too small ({} < {})", dst.len(), m * n)
        }
        if m == 0 {
            return Ok(());
        }
        let lhs_q = quantize_lhs(m, k, lhs)?;
        matmul_interleaved(&self.blocks, &lhs_q, dst, m, n, self.kb);
        Ok(())
    }
}

/// Quantize the activation rows into plain `q8_0` blocks. The interleaved kernels take the
/// activation in the stored layout; only the weights are repacked.
#[tracing::instrument(name = "q-matmul-quantize-lhs-4x8", skip_all, fields(m = m, k = k))]
fn quantize_lhs(m: usize, k: usize, lhs: &[f32]) -> Result<Vec<BlockQ8_0>> {
    let kb = k / QK8_0;
    let mut out = vec![BlockQ8_0::zeros(); m * kb];
    for row in 0..m {
        BlockQ8_0::from_float(&lhs[row * k..(row + 1) * k], &mut out[row * kb..(row + 1) * kb])?;
    }
    Ok(out)
}

/// Rows of the activation handled by one register tile. Beyond four the accumulators spill.
const MR: usize = 4;

#[tracing::instrument(name = "q-matmul-q8-0-4x8", skip_all, fields(m = m, n = n, k = kb * QK8_0))]
fn matmul_interleaved(
    w: &[BlockQ8_0x4],
    lhs: &[BlockQ8_0],
    dst: &mut [f32],
    m: usize,
    n: usize,
    kb: usize,
) {
    let ngroups = n / NCOLS;

    // Each worker owns a contiguous run of column groups, so it walks one unbroken span of the
    // weight buffer and writes a disjoint set of dst columns.
    // Pointers travel as `usize` so the closure stays `Send + Sync` without a wrapper type.
    let job = Job {
        w: w.as_ptr() as usize,
        a: lhs.as_ptr() as usize,
        c: dst.as_mut_ptr() as usize,
        m,
        n,
        kb,
    };

    if ngroups == 1 {
        // SAFETY: one range, covering the output exactly once.
        unsafe { job.run(0, ngroups) };
        return;
    }
    // A single-participant pool runs this inline, so a one-worker build pays no join.
    crate::threadpool::dispatch(|ith, nth| {
        let duty = ngroups.div_ceil(nth);
        let g0 = (duty * ith).min(ngroups);
        let g1 = (g0 + duty).min(ngroups);
        if g0 == g1 {
            return;
        }
        // SAFETY: the `[g0, g1)` ranges are disjoint across `ith`, so the columns written
        // through `job.c` never alias. Bounds were checked by the caller.
        unsafe { job.run(g0, g1) };
    });
}

/// One matmul's operands, as addresses so the dispatched closure needs no wrapper type.
#[derive(Clone, Copy)]
struct Job {
    w: usize,
    a: usize,
    c: usize,
    m: usize,
    n: usize,
    kb: usize,
}

impl Job {
    /// Compute output column groups `[g0, g1)` for all `m` rows.
    ///
    /// # Safety
    /// The addresses must point at buffers of at least `ngroups * kb` interleaved blocks,
    /// `m * kb` activation blocks and `m * n` floats, and no two concurrent calls may overlap
    /// in `[g0, g1)`.
    unsafe fn run(&self, g0: usize, g1: usize) {
        let w = self.w as *const BlockQ8_0x4;
        let a = self.a as *const BlockQ8_0;
        let c = self.c as *mut f32;
        for g in g0..g1 {
            let wg = unsafe { w.add(g * self.kb) };
            let col = g * NCOLS;
            let mut row = 0;
            while row + MR <= self.m {
                unsafe {
                    tile::<MR>(wg, a.add(row * self.kb), self.kb, c.add(row * self.n + col), self.n)
                };
                row += MR;
            }
            while row < self.m {
                unsafe {
                    tile::<1>(wg, a.add(row * self.kb), self.kb, c.add(row * self.n + col), self.n)
                };
                row += 1;
            }
        }
    }
}

/// f16x4 -> f32x4 in one `fcvtl`. Rust's `vcvt_f32_f16` needs the still-unstable `f16` scalar
/// type to load its operand, and converting the four scales one at a time would cost more than
/// the dot products they scale.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn load_f16x4(p: *const f16) -> float32x4_t {
    let h = unsafe { vld1_u16(p as *const u16) };
    let out: float32x4_t;
    unsafe {
        core::arch::asm!(
            "fcvtl {o:v}.4s, {i:v}.4h",
            o = out(vreg) out,
            i = in(vreg) h,
            options(pure, nomem, nostack),
        )
    };
    out
}

/// `acc + dot(a, b)` as a single `sdot`. `vdotq_s32` is nightly-only, matching the treatment in
/// [`super::neon`].
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut acc = acc;
    unsafe {
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        )
    };
    acc
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", not(target_feature = "dotprod")))]
#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let p0 = unsafe { vmull_s8(vget_low_s8(a), vget_low_s8(b)) };
    let p1 = unsafe { vmull_s8(vget_high_s8(a), vget_high_s8(b)) };
    unsafe { vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1))) }
}

/// One `RM x NCOLS` output tile.
///
/// The weight block is loaded once and reused across all `RM` activation rows, which is the
/// whole point of tiling: at `RM == 1` this is the decode gemv and every byte of weight is read
/// exactly once for one output; at `RM == 4` prefill amortizes that read four ways.
///
/// # Safety
/// `w` must address `kb` interleaved blocks, `a` must address `RM * kb` activation blocks with
/// row stride `kb`, and `out` must address `RM` rows of at least `NCOLS` floats with row stride
/// `out_stride`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn tile<const RM: usize>(
    w: *const BlockQ8_0x4,
    a: *const BlockQ8_0,
    kb: usize,
    out: *mut f32,
    out_stride: usize,
) {
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); RM];
        for b in 0..kb {
            let wb = &*w.add(b);
            // 128 contiguous bytes: the four columns' values for this block.
            let b_lo = vld1q_s8_x4(wb.qs.as_ptr());
            let b_hi = vld1q_s8_x4(wb.qs.as_ptr().add(64));
            let bd = load_f16x4(wb.d.as_ptr());

            for (r, acc) in acc.iter_mut().enumerate() {
                let ab = &*a.add(r * kb + b);
                // Each 8-value run of the activation is duplicated to 16 bytes so one `sdot`
                // covers the two columns sharing that 16-byte weight lane.
                let ac = vld1_s8_x4(ab.qs.as_ptr());
                let a0 = vcombine_s8(ac.0, ac.0);
                let a1 = vcombine_s8(ac.1, ac.1);
                let a2 = vcombine_s8(ac.2, ac.2);
                let a3 = vcombine_s8(ac.3, ac.3);

                let zero = vdupq_n_s32(0);
                // ret0 accumulates columns 0,1 in lane pairs; ret1 columns 2,3.
                let mut ret0 = sdot(zero, b_lo.0, a0);
                let mut ret1 = sdot(zero, b_lo.1, a0);
                ret0 = sdot(ret0, b_lo.2, a1);
                ret1 = sdot(ret1, b_lo.3, a1);
                ret0 = sdot(ret0, b_hi.0, a2);
                ret1 = sdot(ret1, b_hi.1, a2);
                ret0 = sdot(ret0, b_hi.2, a3);
                ret1 = sdot(ret1, b_hi.3, a3);

                // Pairwise-add folds the eight half-sums into one value per column.
                let ret = vpaddq_s32(ret0, ret1);
                let ad = vdupq_n_f32(ab.d.to_f32());
                *acc = vfmaq_f32(*acc, vcvtq_f32_s32(ret), vmulq_f32(ad, bd));
            }
        }
        for (r, acc) in acc.iter().enumerate() {
            vst1q_f32(out.add(r * out_stride), *acc);
        }
    }
}

/// Portable fallback with the same contract as the NEON tile.
///
/// # Safety
/// Same as the NEON implementation.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
unsafe fn tile<const RM: usize>(
    w: *const BlockQ8_0x4,
    a: *const BlockQ8_0,
    kb: usize,
    out: *mut f32,
    out_stride: usize,
) {
    unsafe {
        let mut acc = [[0f32; NCOLS]; RM];
        for b in 0..kb {
            let wb = &*w.add(b);
            for (r, acc) in acc.iter_mut().enumerate() {
                let ab = &*a.add(r * kb + b);
                let ad = ab.d.to_f32();
                let mut sums = [0i32; NCOLS];
                for i in 0..(QK8_0 * NCOLS / BLOCKLEN) {
                    let col = i % NCOLS;
                    let src_off = (i / NCOLS) * BLOCKLEN;
                    for j in 0..BLOCKLEN {
                        sums[col] += wb.qs[i * BLOCKLEN + j] as i32 * ab.qs[src_off + j] as i32;
                    }
                }
                for c in 0..NCOLS {
                    acc[c] += sums[c] as f32 * ad * wb.d[c].to_f32();
                }
            }
        }
        for (r, acc) in acc.iter().enumerate() {
            for (c, v) in acc.iter().enumerate() {
                *out.add(r * out_stride + c) = *v;
            }
        }
    }
}

/// Whether this build has a vectorized [`tile`].
///
/// Everywhere else the tile is a scalar loop, which would lose badly to the AVX/simd128
/// `sgemm` kernels the plain layout dispatches to, so interleaving is not the default there.
/// Must track the `cfg` on the NEON [`tile`].
const HAS_SIMD_TILE: bool = cfg!(all(target_arch = "aarch64", target_feature = "neon"));

/// Whether to interleave at all.
///
/// `XN_Q8_REPACK=0` keeps the stored layout, which is the escape hatch if a platform ever
/// turns out to prefer the row-tiled `sgemm` path. `XN_Q8_REPACK=1` forces interleaving on
/// even without a vectorized tile, which is how the portable path gets exercised end to end.
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("XN_Q8_REPACK").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => HAS_SIMD_TILE,
    })
}

/// Build the storage for a freshly read `q8_0` tensor, interleaving when the shape allows.
///
/// `src` is the block stream exactly as it appears on disk, so this is the one point where
/// the layout is chosen -- it takes plain blocks rather than a [`super::QTensor`] precisely
/// because an interleaved storage still reports `dtype() == Q8_0` and would be indistinguishable
/// from a plain one at the type level. Anything that does not qualify gets a plain copy, so
/// this is safe to call for every `q8_0` tensor a file contains.
pub fn q8_0_storage(src: &[BlockQ8_0], dims: &[usize]) -> super::QStorage {
    interleaved(src, dims).unwrap_or_else(|| super::QStorage::Cpu(Box::new(src.to_vec())))
}

/// Same layout decision as [`q8_0_storage`], for a block stream this call owns.
///
/// Quantizing in process produces the blocks rather than borrowing them from a mapped file, so
/// the plain path can move the vector instead of copying the whole weight.
pub fn q8_0_storage_owned(src: Vec<BlockQ8_0>, dims: &[usize]) -> super::QStorage {
    interleaved(&src, dims).unwrap_or_else(|| super::QStorage::Cpu(Box::new(src)))
}

fn interleaved(src: &[BlockQ8_0], dims: &[usize]) -> Option<super::QStorage> {
    // KleidiAI's packed layout comes first when it is built in and switched on: its kernels
    // cover the same shapes and more (any `n`), so the interleave is its fallback.
    #[cfg(all(feature = "kleidiai", target_arch = "aarch64"))]
    if let Some(storage) = super::kleidiai::q8_0_storage(src, dims) {
        return Some(storage);
    }
    if enabled() && is_eligible(dims) {
        // Interleaving is an optimization; a shape we mis-judged just keeps the plain layout.
        if let Ok(packed) = Q8_0x4::from_q8_0(src, dims[0], dims[1]) {
            return Some(super::QStorage::Cpu(Box::new(packed)));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::QuantizedType;

    fn ref_matmul(w: &[BlockQ8_0], lhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut dst = vec![0f32; m * n];
        crate::quantized::k_quants::matmul((m, k, n), lhs, w, &mut dst).unwrap();
        dst
    }

    /// The storage's bytes as actually held in memory, which is what distinguishes the
    /// interleaved layout from the plain one.
    fn stored_bytes(storage: &dyn QuantizedType) -> &[u8] {
        unsafe { std::slice::from_raw_parts(storage.as_ptr(), storage.storage_size_in_bytes()) }
    }

    /// Whether `q8_0_storage` is handing tensors to `kleidiai` instead, whose layout is lossy
    /// and whose in-memory bytes are not the canonical ones. The byte-level checks below
    /// are about the interleave and do not apply then.
    fn kleidiai_active() -> bool {
        #[cfg(all(feature = "kleidiai", target_arch = "aarch64"))]
        {
            crate::quantized::kleidiai::active()
        }
        #[cfg(not(all(feature = "kleidiai", target_arch = "aarch64")))]
        {
            false
        }
    }

    fn weights(n: usize, k: usize) -> Vec<BlockQ8_0> {
        let raw: Vec<f32> = (0..n * k).map(|i| ((i * 37 % 211) as f32 - 105.0) / 64.0).collect();
        let mut blocks = vec![BlockQ8_0::zeros(); n * k / QK8_0];
        BlockQ8_0::from_float(&raw, &mut blocks).unwrap();
        blocks
    }

    #[test]
    fn interleave_round_trips() {
        let (n, k) = (8, 64);
        let plain = weights(n, k);
        let packed = Q8_0x4::from_q8_0(&plain, n, k).unwrap();
        let back = packed.unpack();
        assert_eq!(plain.len(), back.len());
        for (a, b) in plain.iter().zip(back.iter()) {
            assert_eq!(a.d, b.d);
            assert_eq!(a.qs, b.qs);
        }
    }

    #[test]
    fn matches_the_plain_kernel() {
        // Cover the decode gemv, a full MR tile, and an m that needs the remainder path.
        for (m, k, n) in [(1, 768, 3072), (1, 64, 8), (4, 128, 16), (7, 96, 12), (13, 64, 4)] {
            let plain = weights(n, k);
            let packed = Q8_0x4::from_q8_0(&plain, n, k).unwrap();
            let lhs: Vec<f32> = (0..m * k).map(|i| ((i * 53 % 173) as f32 - 86.0) / 32.0).collect();

            let want = ref_matmul(&plain, &lhs, m, k, n);
            let mut got = vec![0f32; m * n];
            packed.matmul_t((m, k, n), &lhs, &mut got).unwrap();

            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                // Both paths quantize the activation identically, so the only spread is
                // float accumulation order.
                let tol = 1e-3 * w.abs().max(1.0);
                assert!((g - w).abs() <= tol, "m={m} k={k} n={n} idx={i}: got {g}, want {w}");
            }
        }
    }

    #[test]
    fn dequantize_matches_plain() {
        let (n, k) = (8, 64);
        let plain = weights(n, k);
        let packed = Q8_0x4::from_q8_0(&plain, n, k).unwrap();
        let want = plain.dequantize(n * k).unwrap();
        let got = packed.dequantize(n * k).unwrap();
        assert_eq!(want, got);
    }

    #[test]
    fn raw_data_is_the_stored_layout() {
        let (n, k) = (8, 64);
        let plain = weights(n, k);
        let packed = Q8_0x4::from_q8_0(&plain, n, k).unwrap();
        let want = unsafe {
            std::slice::from_raw_parts(
                plain.as_ptr() as *const u8,
                std::mem::size_of_val(plain.as_slice()),
            )
        };
        assert_eq!(packed.raw_data().unwrap().as_ref(), want);
    }

    /// `q8_0_storage` picks the layout, so whichever branch a target takes it must still
    /// answer matmuls correctly and hand back canonical bytes. Runs against whatever
    /// `enabled()` decided for this build, which is the point -- both branches are wired.
    #[test]
    fn q8_0_storage_is_layout_agnostic() {
        if kleidiai_active() {
            return;
        }
        let (m, n, k) = (3, 8, 128);
        let plain = weights(n, k);
        let storage = q8_0_storage(&plain, &[n, k]);
        let crate::quantized::QStorage::Cpu(storage) = &storage;

        let want_bytes = unsafe {
            std::slice::from_raw_parts(
                plain.as_ptr() as *const u8,
                std::mem::size_of_val(plain.as_slice()),
            )
        };
        assert_eq!(storage.raw_data().unwrap().as_ref(), want_bytes, "canonical bytes");
        assert_eq!(storage.dtype(), GgmlDType::Q8_0);
        assert_eq!(storage.storage_size_in_bytes(), want_bytes.len());

        // Wiring check: the layout actually tracks `enabled()`. This does not judge whether
        // the gate's default is *right* for this target -- that is a perf call, and what makes
        // it safe to flip either way is the equivalence asserted below. Compare the *in-memory*
        // bytes: `raw_data` un-interleaves and both layouts are the same length, so neither
        // tells the two apart.
        assert_eq!(stored_bytes(&**storage) != want_bytes, enabled(), "layout follows the gate");

        let lhs: Vec<f32> = (0..m * k).map(|i| ((i * 53 % 173) as f32 - 86.0) / 32.0).collect();
        let want = ref_matmul(&plain, &lhs, m, k, n);
        let mut got = vec![0f32; m * n];
        storage.matmul_t((m, k, n), &lhs, &mut got).unwrap();
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = 1e-3 * w.abs().max(1.0);
            assert!((g - w).abs() <= tol, "idx={i}: got {g}, want {w}");
        }
    }

    /// A shape the interleaver rejects must fall back to the plain layout rather than error,
    /// whatever the gate says.
    #[test]
    fn q8_0_storage_falls_back_on_ineligible_shapes() {
        if kleidiai_active() {
            return;
        }
        let (n, k) = (6, 64); // n % NCOLS != 0
        let plain = weights(n, k);
        let crate::quantized::QStorage::Cpu(storage) = &q8_0_storage(&plain, &[n, k]);
        let want = unsafe {
            std::slice::from_raw_parts(
                plain.as_ptr() as *const u8,
                std::mem::size_of_val(plain.as_slice()),
            )
        };
        assert_eq!(stored_bytes(&**storage), want, "ineligible shape must be stored verbatim");
    }

    /// The point of the in-process hook: a weight quantized by `QTensor::quantize_f32` must be
    /// byte-for-byte the storage the GGUF loader would have built for the same weight, so a
    /// model has the same layout (and so the same kernel) whichever way it was loaded.
    #[test]
    fn quantize_f32_matches_the_gguf_layout() {
        if kleidiai_active() {
            return;
        }
        for (n, k) in [(8, 128), (6, 64) /* ineligible: n % NCOLS != 0 */] {
            let raw: Vec<f32> =
                (0..n * k).map(|i| ((i * 37 % 211) as f32 - 105.0) / 64.0).collect();
            let mut plain = vec![BlockQ8_0::zeros(); n * k / QK8_0];
            BlockQ8_0::from_float(&raw, &mut plain).unwrap();

            let from_file = q8_0_storage(&plain, &[n, k]);
            let in_process =
                crate::quantized::QTensor::quantize_f32(&raw, &vec![n, k].into(), GgmlDType::Q8_0)
                    .unwrap();
            let crate::quantized::QStorage::Cpu(from_file) = &from_file;
            let crate::quantized::QStorage::Cpu(in_process_storage) = &in_process.storage;
            assert_eq!(
                stored_bytes(&**in_process_storage),
                stored_bytes(&**from_file),
                "[{n}, {k}]: in-process layout must match the loader's"
            );

            // And it still answers matmuls and hands back canonical bytes.
            let m = 3;
            let lhs: Vec<f32> = (0..m * k).map(|i| ((i * 53 % 173) as f32 - 86.0) / 32.0).collect();
            let want = ref_matmul(&plain, &lhs, m, k, n);
            let mut got = vec![0f32; m * n];
            in_process.matmul_t((m, k, n), &lhs, &mut got).unwrap();
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let tol = 1e-3 * w.abs().max(1.0);
                assert!((g - w).abs() <= tol, "[{n}, {k}] idx={i}: got {g}, want {w}");
            }
            let canonical = unsafe {
                std::slice::from_raw_parts(
                    plain.as_ptr() as *const u8,
                    std::mem::size_of_val(plain.as_slice()),
                )
            };
            assert_eq!(in_process.data().unwrap().as_ref(), canonical, "[{n}, {k}] canonical");
        }
    }

    #[test]
    fn eligibility() {
        assert!(is_eligible(&[3072, 768]));
        assert!(!is_eligible(&[3072]));
        assert!(!is_eligible(&[3, 768]), "n must be a multiple of 4");
        assert!(!is_eligible(&[4, 48]), "k must be a multiple of the block size");
    }
}
