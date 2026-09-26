//! `q8_0` weights in KleidiAI's packed int8 layout, run by its SME2 and NEON micro-kernels.
//!
//! [KleidiAI] is Arm's library of matmul micro-kernels: one hand-written routine per ISA tier,
//! each with packing routines that lay the operands out the way that kernel reads them. What it
//! adds over [`super::repack`] is SME2. Apple's M4 and M5 (and Arm's Cortex-X925 onwards) carry
//! a Scalable Matrix Extension unit whose outer-product `mopa` instruction multiplies a whole
//! tile per instruction, and it is only reachable from assembly: Rust has no stable SME
//! intrinsics, and streaming mode's calling convention rules out inline `asm!` in practice. The
//! kernels are compiled from KleidiAI's sources by `build.rs` under the `kleidiai` feature, and
//! this module is the Rust side: the packed storage, the dispatch, and the threading.
//!
//! [KleidiAI]: https://github.com/ARM-software/kleidiai
//!
//! # What is stored
//!
//! KleidiAI has no int8 kernel with per-block scales. Its int8 weight format, `qsi8cx`, is
//! symmetric int8 with one f32 scale per output channel, so a `q8_0` tensor is requantized on
//! load: each row is dequantized and rescaled to `max|w| / 127`. That costs about a bit in the
//! blocks of a row that are much smaller than its largest one, and is exactly what llama.cpp's
//! KleidiAI backend does for `Q8_0` (`ggml-cpu/kleidiai/kleidiai.cpp`). Activations are
//! quantized per row as well, to asymmetric int8 with a zero point (`qai8dx`), by KleidiAI's
//! own packing routine; the kernel folds the zero point in through per-column sums it stores
//! next to the weights. A weight that is still f32 when it gets here (quantized in process
//! rather than read from a GGUF) is requantized from the f32 values directly, so it pays one
//! rounding rather than two.
//!
//! The requantization is lossy, so [`raw_data`](super::QuantizedType::raw_data) cannot return
//! the bytes the tensor was loaded from. It quantizes the stored weights back to `q8_0`, which
//! is what the GGUF writer then persists: a model round-tripped through this storage is close
//! to, but not bit-identical with, the file it came from.
//!
//! # Which kernels
//!
//! A kernel [`Family`] is chosen once per process from the CPU's features and every weight is
//! packed for it. The packed layout depends on the kernel's `nr`/`kr` tile constants, so each
//! family pairs a gemm kernel with a gemv kernel that agree on those:
//!
//! | family    | `m > 1` (prefill)                             | `m == 1` (decode)                        |
//! |-----------|-----------------------------------------------|------------------------------------------|
//! | `Sme2`    | `qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa` | `qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot` |
//! | `I8mm`    | `qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm`        | `qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod` |
//! | `Dotprod` | `qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod`     | `qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod` |
//!
//! Environment knobs, all read once:
//!
//! * `XN_KLEIDIAI=0` turns the storage off, so `q8_0` weights take the [`super::repack`] path.
//!   `XN_KLEIDIAI=sme2|i8mm|dotprod` pins a family, for comparing them on a CPU that has more
//!   than one.
//! * `XN_KLEIDIAI_THREADS=N` caps how many pool threads a matmul here spreads over. An SME
//!   unit is shared by the cores of a cluster, so past a point more threads only add
//!   contention; see `thread_cap` for the default.
//!
//! # Measured on an Apple M5 (4P + 6E)
//!
//! Against the interleaved `q8_0_4x8` kernel of [`super::repack`], on Phonon's four linears
//! (`d_model` 512, `dim_feedforward` 2048), best of five interleaved rounds:
//!
//! | `m`  | threads | interleaved | SME2 | NEON i8mm |
//! |------|---------|-------------|------|-----------|
//! | 1    | 1       | 37.4 us     | 14.5 | 47.9      |
//! | 1    | 4       | 14.2        | 11.3 | 15.1      |
//! | 16   | 4       | 180         | 77   | 77        |
//! | 64   | 4       | 648         | 218  | 281       |
//! | 256  | 4       | 2589        | 860  | 1034      |
//!
//! The SME2 gemm gives the same throughput from one thread as from four (the unit is shared
//! by the cluster), and letting the efficiency cores join at 8 threads made it *slower*
//! (118 us at `m = 16`), hence the cap. The price is accuracy: per-row requantization of
//! Phonon's `q8_0` weights moves the per-frame latents by 1.4% relative L2 on average
//! (teacher-forced, 60 frames; median 0.9%, worst 9%) and the EOS logit by 0.04.

use super::GgmlDType;
use super::k_quants::{BlockQ8_0, GgmlType, QK8_0};
use crate::Result;
use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------------------------

/// Mirrors `kai_rhs_pack_qsi8cx_params`.
#[repr(C)]
struct RhsPackParams {
    lhs_zero_point: i32,
    scale_multiplier: f32,
}

unsafe extern "C" {
    fn xn_kleidiai_cpu_features() -> u32;
    fn xn_kleidiai_fast_cluster_cores() -> u32;

    fn kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
        m: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
    ) -> usize;
    fn kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(
        m_idx: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
    ) -> usize;
    fn kai_run_lhs_quant_pack_qai8dxp_f32(
        m: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
        m_idx_start: usize,
        lhs: *const f32,
        lhs_stride: usize,
        lhs_packed: *mut c_void,
    );

    fn kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        n: usize,
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
    ) -> usize;
    fn kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
    ) -> usize;
    fn kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        num_groups: usize,
        n: usize,
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
        rhs: *const i8,
        bias: *const f32,
        scale: *const f32,
        rhs_packed: *mut c_void,
        extra_bytes: usize,
        params: *const RhsPackParams,
    );
}

/// The entry points of one matmul micro-kernel, plus the tile constants its packed operands
/// are laid out for. Read once through the kernel's getters: for the SME kernels they depend
/// on the streaming vector length of the machine this runs on.
#[derive(Clone, Copy)]
struct Kernel {
    m_step: usize,
    n_step: usize,
    mr: usize,
    nr: usize,
    kr: usize,
    sr: usize,
    lhs_packed_offset: unsafe extern "C" fn(usize, usize) -> usize,
    rhs_packed_offset: unsafe extern "C" fn(usize, usize) -> usize,
    dst_offset: unsafe extern "C" fn(usize, usize, usize) -> usize,
    #[allow(clippy::type_complexity)]
    run: unsafe extern "C" fn(
        usize,
        usize,
        usize,
        *const c_void,
        *const c_void,
        *mut f32,
        usize,
        usize,
        f32,
        f32,
    ),
}

/// Declares the ten entry points of a KleidiAI matmul kernel and a constructor that reads
/// its tile constants.
macro_rules! declare_kernel {
    (
        $ctor:ident:
        $m_step:ident, $n_step:ident, $mr:ident, $nr:ident, $kr:ident, $sr:ident,
        $lhs_off:ident, $rhs_off:ident, $dst_off:ident, $run:ident $(,)?
    ) => {
        unsafe extern "C" {
            fn $m_step() -> usize;
            fn $n_step() -> usize;
            fn $mr() -> usize;
            fn $nr() -> usize;
            fn $kr() -> usize;
            fn $sr() -> usize;
            fn $lhs_off(m_idx: usize, k: usize) -> usize;
            fn $rhs_off(n_idx: usize, k: usize) -> usize;
            fn $dst_off(m_idx: usize, n_idx: usize, dst_stride: usize) -> usize;
            fn $run(
                m: usize,
                n: usize,
                k: usize,
                lhs_packed: *const c_void,
                rhs_packed: *const c_void,
                dst: *mut f32,
                dst_stride_row: usize,
                dst_stride_col: usize,
                scalar_min: f32,
                scalar_max: f32,
            );
        }

        /// # Safety
        /// The CPU must implement the kernel's ISA: the tile getters of an SME kernel read the
        /// streaming vector length with `rdsvl`, which traps on a core without SME.
        unsafe fn $ctor() -> Kernel {
            unsafe {
                Kernel {
                    m_step: $m_step(),
                    n_step: $n_step(),
                    mr: $mr(),
                    nr: $nr(),
                    kr: $kr(),
                    sr: $sr(),
                    lhs_packed_offset: $lhs_off,
                    rhs_packed_offset: $rhs_off,
                    dst_offset: $dst_off,
                    run: $run,
                }
            }
        }
    };
}

declare_kernel!(
    sme2_gemm:
    kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
);
declare_kernel!(
    sme2_gemv:
    kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
    kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
);
declare_kernel!(
    i8mm_gemm:
    kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
);
declare_kernel!(
    i8mm_gemv:
    kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
    kai_run_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
);
declare_kernel!(
    dotprod_gemm:
    kai_get_m_step_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    kai_get_n_step_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
    kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
);
declare_kernel!(
    dotprod_gemv:
    kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
);

// ---------------------------------------------------------------------------------------------
// CPU features and kernel families
// ---------------------------------------------------------------------------------------------

// Bit values of `xn_kleidiai_cpu_features`, mirrored from `csrc/xn_kleidiai.c`.
const FEAT_DOTPROD: u32 = 1;
const FEAT_I8MM: u32 = 2;
const FEAT_SME2: u32 = 8;

fn cpu_features() -> u32 {
    static F: OnceLock<u32> = OnceLock::new();
    // SAFETY: a pure probe with no preconditions.
    *F.get_or_init(|| unsafe { xn_kleidiai_cpu_features() })
}

/// A gemm/gemv kernel pair sharing one packed weight layout. See the module docs for which
/// KleidiAI kernels each one names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// SME2 outer products: Apple M4/M5, Cortex-X925 and later.
    Sme2,
    /// NEON `smmla` for gemm, `sdot` for gemv: Apple M4/M5, Cortex-A710 and later, Neoverse N2.
    I8mm,
    /// NEON `sdot` only: Apple M1-M3, Cortex-A75 and later, Raspberry Pi 5.
    Dotprod,
}

impl Family {
    /// Preference order when the CPU offers more than one.
    const ALL: [Family; 3] = [Family::Sme2, Family::I8mm, Family::Dotprod];

    /// Whether this CPU can run the family's kernels.
    pub fn available(self) -> bool {
        let f = cpu_features();
        match self {
            Family::Sme2 => f & FEAT_SME2 != 0,
            Family::I8mm => f & (FEAT_I8MM | FEAT_DOTPROD) == (FEAT_I8MM | FEAT_DOTPROD),
            Family::Dotprod => f & FEAT_DOTPROD != 0,
        }
    }

    /// The fastest family this CPU can run, if any.
    pub fn best() -> Option<Family> {
        Family::ALL.into_iter().find(|f| f.available())
    }

    pub fn name(self) -> &'static str {
        match self {
            Family::Sme2 => "sme2",
            Family::I8mm => "i8mm",
            Family::Dotprod => "dotprod",
        }
    }

    fn parse(name: &str) -> Option<Family> {
        Family::ALL.into_iter().find(|f| f.name() == name)
    }

    fn kernels(self) -> Result<&'static Kernels> {
        if !self.available() {
            crate::bail!("kleidiai: this CPU cannot run the {} kernels", self.name())
        }
        static TABLES: [OnceLock<Kernels>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
        Ok(TABLES[self as usize].get_or_init(|| {
            // SAFETY: availability was just checked, so the getters do not trap.
            let (gemm, gemv) = unsafe {
                match self {
                    Family::Sme2 => (sme2_gemm(), sme2_gemv()),
                    Family::I8mm => (i8mm_gemm(), i8mm_gemv()),
                    Family::Dotprod => (dotprod_gemm(), dotprod_gemv()),
                }
            };
            // The weight is packed once, for the gemm kernel's tile; the gemv kernel has to
            // read that same layout.
            assert_eq!(
                (gemm.nr, gemm.kr, gemm.sr),
                (gemv.nr, gemv.kr, gemv.sr),
                "kleidiai: the {} gemm and gemv kernels disagree on the packed layout",
                self.name()
            );
            Kernels { family: self, gemm, gemv }
        }))
    }
}

struct Kernels {
    family: Family,
    gemm: Kernel,
    gemv: Kernel,
}

/// The family this process packs `q8_0` weights for, or `None` if the storage is off
/// (`XN_KLEIDIAI=0`) or the CPU has no family it can run.
pub fn family() -> Option<Family> {
    static F: OnceLock<Option<Family>> = OnceLock::new();
    *F.get_or_init(|| match std::env::var("XN_KLEIDIAI").as_deref() {
        Ok("0") => None,
        Ok("") | Ok("1") | Err(_) => Family::best(),
        Ok(name) => match Family::parse(name) {
            Some(f) if f.available() => Some(f),
            Some(f) => {
                tracing::warn!(
                    "XN_KLEIDIAI={name}: this CPU lacks {}, kleidiai storage off",
                    f.name()
                );
                None
            }
            None => {
                tracing::warn!("XN_KLEIDIAI={name} is not a kernel family, using the default");
                Family::best()
            }
        },
    })
}

/// Whether `q8_0` tensors are being given this storage.
pub fn active() -> bool {
    family().is_some()
}

/// Most pool threads one matmul of `family` spreads over.
///
/// `XN_KLEIDIAI_THREADS` overrides for every family. Otherwise the NEON families are
/// uncapped -- they scale like any other kernel on the pool -- and the SME2 family is capped
/// at the size of the CPU's fastest cluster where the OS reports one: an SME unit is shared by
/// the cores of a cluster, so more threads than that add nothing, and slower cores joining a
/// static split cost the wait for the slowest. On the M5 this was tuned on that cap is 4;
/// where nothing is known the family runs uncapped.
fn thread_cap(family: Family) -> usize {
    static C: OnceLock<Option<usize>> = OnceLock::new();
    let forced = *C.get_or_init(|| {
        std::env::var("XN_KLEIDIAI_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
    });
    match (forced, family) {
        (Some(n), _) => n,
        (None, Family::Sme2) => {
            static S: OnceLock<usize> = OnceLock::new();
            // SAFETY: a pure probe with no preconditions.
            *S.get_or_init(|| match unsafe { xn_kleidiai_fast_cluster_cores() } {
                0 => usize::MAX,
                n => n as usize,
            })
        }
        (None, _) => usize::MAX,
    }
}

// ---------------------------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------------------------

/// A `[n, k]` weight requantized to per-row int8 and packed for one kernel [`Family`].
pub struct Q8_0Kai {
    /// KleidiAI's `qsi8cxp` layout: groups of `nr` rows, each group `kr`-interleaved int8
    /// values followed by per-row sums, scales and (zero) biases.
    packed: Vec<u8>,
    n: usize,
    k: usize,
    kernels: &'static Kernels,
}

impl Q8_0Kai {
    /// Pack a row-major stream of `q8_0` blocks (`n` rows of `k / QK8_0` blocks) for the
    /// process-wide [`family`].
    pub fn from_q8_0(src: &[BlockQ8_0], n: usize, k: usize) -> Result<Self> {
        let Some(family) = family() else { crate::bail!("kleidiai storage is not active") };
        Self::from_q8_0_for(family, src, n, k)
    }

    /// [`Self::from_q8_0`] for an explicit family, which is how a benchmark compares them.
    pub fn from_q8_0_for(family: Family, src: &[BlockQ8_0], n: usize, k: usize) -> Result<Self> {
        if n == 0 || k == 0 || !k.is_multiple_of(QK8_0) {
            crate::bail!("kleidiai: shape [{n}, {k}] cannot be packed")
        }
        let kb = k / QK8_0;
        if src.len() != n * kb {
            crate::bail!(
                "kleidiai: expected {} q8_0 blocks for [{n}, {k}], got {}",
                n * kb,
                src.len()
            )
        }
        let kernels = family.kernels()?;
        let (qdata, scales) = requantize_rows(n, k, |row, out| {
            for (b, blk) in src[row * kb..(row + 1) * kb].iter().enumerate() {
                let d = blk.d.to_f32();
                for (o, &q) in out[b * QK8_0..(b + 1) * QK8_0].iter_mut().zip(blk.qs.iter()) {
                    *o = q as f32 * d;
                }
            }
        });
        Self::pack(kernels, n, k, &qdata, &scales)
    }

    /// Quantize an f32 `[n, k]` weight straight to the packed layout, skipping the `q8_0`
    /// intermediate a file would have gone through.
    pub fn from_f32_for(family: Family, src: &[f32], n: usize, k: usize) -> Result<Self> {
        if n == 0 || k == 0 {
            crate::bail!("kleidiai: shape [{n}, {k}] cannot be packed")
        }
        if src.len() != n * k {
            crate::bail!("kleidiai: expected {} values for [{n}, {k}], got {}", n * k, src.len())
        }
        let kernels = family.kernels()?;
        let (qdata, scales) =
            requantize_rows(n, k, |row, out| out.copy_from_slice(&src[row * k..(row + 1) * k]));
        Self::pack(kernels, n, k, &qdata, &scales)
    }

    fn pack(
        kernels: &'static Kernels,
        n: usize,
        k: usize,
        qdata: &[i8],
        scales: &[f32],
    ) -> Result<Self> {
        let g = &kernels.gemm;
        // SAFETY: `qdata` holds `n * k` values and `scales` `n` of them, as checked by the
        // callers; the packed buffer is sized by the routine's own query.
        let packed = unsafe {
            let size =
                kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, g.nr, g.kr, g.sr);
            let mut packed = vec![0u8; size];
            // `lhs_zero_point = 1` stores plain row sums; the kernel scales them by the actual
            // activation zero point at run time.
            let params = RhsPackParams { lhs_zero_point: 1, scale_multiplier: 1.0 };
            kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
                1,
                n,
                k,
                g.nr,
                g.kr,
                g.sr,
                qdata.as_ptr(),
                std::ptr::null(),
                scales.as_ptr(),
                packed.as_mut_ptr() as *mut c_void,
                0,
                &params,
            );
            packed
        };
        Ok(Self { packed, n, k, kernels })
    }

    pub fn family(&self) -> Family {
        self.kernels.family
    }

    /// Read row `r`'s int8 values back out of the packed layout, returning its scale.
    ///
    /// Inverts `kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon`: within a group of `nr` rows the
    /// values are interleaved `kr` at a time, so `(row, k0 + j)` sits at
    /// `(k0 / kr * nr + row) * kr + j`, and the group's `nr` scales follow its values and
    /// `nr` reduction sums.
    fn row(&self, r: usize, out: &mut [i8]) -> f32 {
        let g = &self.kernels.gemm;
        let (nr, kr) = (g.nr, g.kr);
        let k_internal = self.k.div_ceil(32) * 32;
        // SAFETY: plain size arithmetic.
        let stride = unsafe {
            kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(self.k, nr, kr, g.sr)
        };
        let base = (r / nr) * stride;
        let i = r % nr;
        for (b, chunk) in out.chunks_mut(kr).enumerate() {
            let at = base + (b * nr + i) * kr;
            let len = chunk.len();
            for (o, &v) in chunk.iter_mut().zip(&self.packed[at..at + len]) {
                *o = v as i8;
            }
        }
        let scale_at = base + nr * k_internal + nr * 4 + i * 4;
        f32::from_ne_bytes(self.packed[scale_at..scale_at + 4].try_into().unwrap())
    }

    /// The stored weights as f32, i.e. what the kernels actually multiply by.
    fn to_f32(&self) -> Vec<f32> {
        let (n, k) = (self.n, self.k);
        let mut out = vec![0f32; n * k];
        crate::threadpool::par_chunks_mut(&mut out, k, |r, row| {
            let mut q = vec![0i8; k];
            let scale = self.row(r, &mut q);
            for (o, &v) in row.iter_mut().zip(&q) {
                *o = v as f32 * scale;
            }
        });
        out
    }

    /// `[n, k] x [k]^T` for one activation row, split over the pool by output column.
    #[tracing::instrument(name = "q-matmul-kai-gemv", skip_all, fields(n = self.n, k = self.k))]
    fn gemv(&self, lhs: &[f32], dst: &mut [f32]) {
        let kern = &self.kernels.gemv;
        let lhs_packed = pack_lhs(kern, 1, self.k, lhs);
        let job = Job {
            lhs: lhs_packed.as_ptr() as usize,
            rhs: self.packed.as_ptr() as usize,
            dst: dst.as_mut_ptr() as usize,
            m: 1,
            n: self.n,
            k: self.k,
        };
        run_split(kern, self.kernels.family, &job);
    }

    /// `[m, k] x [n, k]^T`: pack the activation rows in parallel, then split the output.
    #[tracing::instrument(name = "q-matmul-kai-gemm", skip_all, fields(m, n = self.n, k = self.k))]
    fn gemm(&self, m: usize, lhs: &[f32], dst: &mut [f32]) {
        let kern = &self.kernels.gemm;
        let lhs_packed = pack_lhs(kern, m, self.k, lhs);
        let job = Job {
            lhs: lhs_packed.as_ptr() as usize,
            rhs: self.packed.as_ptr() as usize,
            dst: dst.as_mut_ptr() as usize,
            m,
            n: self.n,
            k: self.k,
        };
        run_split(kern, self.kernels.family, &job);
    }
}

impl super::QuantizedType for Q8_0Kai {
    fn dtype(&self) -> GgmlDType {
        // The packing is a storage detail; callers still see a q8_0 tensor.
        GgmlDType::Q8_0
    }

    fn block_size(&self) -> usize {
        QK8_0
    }

    /// Bytes held in memory, i.e. the packed layout.
    fn size(&self) -> usize {
        self.packed.len()
    }

    /// Bytes of the *canonical* `q8_0` form, which is what `raw_data` returns and what the
    /// GGUF writer sizes the tensor by. Not the in-memory footprint: see `size`.
    fn storage_size_in_bytes(&self) -> usize {
        self.n * self.k / QK8_0 * std::mem::size_of::<BlockQ8_0>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.packed.as_ptr()
    }

    fn raw_data(&self) -> Result<Cow<'_, [u8]>> {
        // Lossy: the per-row requantization cannot be undone, so this is the stored weight
        // quantized to q8_0 afresh.
        let mut blocks = vec![BlockQ8_0::zeros(); self.n * self.k / QK8_0];
        BlockQ8_0::from_float(&self.to_f32(), &mut blocks)?;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                std::mem::size_of_val(blocks.as_slice()),
            )
        };
        Ok(Cow::Owned(bytes.to_vec()))
    }

    fn dequantize(&self, elem_count: usize) -> Result<Vec<f32>> {
        if elem_count != self.n * self.k {
            crate::bail!(
                "kleidiai: dequantize wants {elem_count} values, tensor holds {}",
                self.n * self.k
            )
        }
        Ok(self.to_f32())
    }

    fn from_float(&mut self, xs: &[f32]) -> Result<()> {
        *self = Self::from_f32_for(self.kernels.family, xs, self.n, self.k)?;
        Ok(())
    }

    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        let (m, k, n) = mkn;
        if n != self.n {
            crate::bail!("matmul_t: n mismatch, weights hold {} but got {n}", self.n)
        }
        if k != self.k {
            crate::bail!("matmul_t: k mismatch, weights hold {} but got {k}", self.k)
        }
        if lhs.len() != m * k {
            crate::bail!("matmul_t: expected {} lhs elements, got {}", m * k, lhs.len())
        }
        if dst.len() < m * n {
            crate::bail!("matmul_t: dst too small ({} < {})", dst.len(), m * n)
        }
        match m {
            0 => {}
            1 => self.gemv(lhs, dst),
            _ => self.gemm(m, lhs, dst),
        }
        Ok(())
    }
}

/// Symmetric per-row int8: `q = round(w / scale)`, `scale = max|w| / 127`. Rows run over the
/// pool since a checkpoint's worth of them adds up.
fn requantize_rows<F>(n: usize, k: usize, row: F) -> (Vec<i8>, Vec<f32>)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let mut qdata = vec![0i8; n * k];
    let mut scales = vec![0f32; n];
    let scales_ptr = scales.as_mut_ptr() as usize;
    crate::threadpool::par_chunks_mut(&mut qdata, k, |r, q| {
        let mut w = vec![0f32; k];
        row(r, &mut w);
        let max_abs = w.iter().fold(0f32, |acc, v| acc.max(v.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
        let inv = if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 };
        for (o, v) in q.iter_mut().zip(&w) {
            *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        // SAFETY: each chunk index `r` is visited by exactly one worker, so the writes to
        // `scales` are disjoint, and `scales` outlives the dispatch.
        unsafe { *(scales_ptr as *mut f32).add(r) = scale };
    });
    (qdata, scales)
}

/// Quantize and pack `m` activation rows for `kern`, `mr` rows per packed group, groups in
/// parallel.
fn pack_lhs(kern: &Kernel, m: usize, k: usize, lhs: &[f32]) -> Vec<u8> {
    let (mr, kr, sr) = (kern.mr, kern.kr, kern.sr);
    // SAFETY: a size query.
    let size = unsafe { kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr) };
    // Zeroed so the padding rows of a partial last group hold something harmless: the kernel
    // computes them and predicates the store, so their contents only have to be finite.
    let mut packed = vec![0u8; size];
    let job = PackJob {
        lhs: lhs.as_ptr() as usize,
        packed: packed.as_mut_ptr() as usize,
        m,
        k,
        mr,
        kr,
        sr,
    };
    let groups = m.div_ceil(mr);
    if groups == 1 {
        // SAFETY: one group, covering every row once.
        unsafe { job.run(0) };
    } else {
        // SAFETY: groups are disjoint `mr`-row ranges of the input and the output.
        crate::threadpool::par_units(groups, |g| unsafe { job.run(g) });
    }
    packed
}

/// Operands of one activation packing, as addresses so the closure needs no wrapper type.
#[derive(Clone, Copy)]
struct PackJob {
    lhs: usize,
    packed: usize,
    m: usize,
    k: usize,
    mr: usize,
    kr: usize,
    sr: usize,
}

impl PackJob {
    /// Pack the rows of group `g`, i.e. `[g * mr, min(m, (g + 1) * mr))`.
    ///
    /// # Safety
    /// `lhs` must address `m * k` floats and `packed` the buffer sized for them; no two
    /// concurrent calls may share `g`.
    unsafe fn run(&self, g: usize) {
        let r0 = g * self.mr;
        let rows = (self.m - r0).min(self.mr);
        unsafe {
            let off = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(
                r0, self.k, self.mr, self.kr, self.sr,
            );
            kai_run_lhs_quant_pack_qai8dxp_f32(
                rows,
                self.k,
                self.mr,
                self.kr,
                self.sr,
                r0,
                (self.lhs as *const f32).add(r0 * self.k),
                self.k * std::mem::size_of::<f32>(),
                (self.packed as *mut u8).add(off) as *mut c_void,
            );
        }
    }
}

/// Operands of one matmul, as addresses so the dispatched closure needs no wrapper type.
#[derive(Clone, Copy)]
struct Job {
    lhs: usize,
    rhs: usize,
    dst: usize,
    m: usize,
    n: usize,
    k: usize,
}

impl Job {
    /// Compute output rows `[m0, m1)` by columns `[n0, n1)` in one kernel call.
    ///
    /// # Safety
    /// `m0` must be a multiple of the kernel's `m_step` and `n0` of its `n_step`; the packed
    /// operands must cover the tensor and `dst` must hold `m * n` floats; no two concurrent
    /// calls may overlap in `[m0, m1) x [n0, n1)`.
    unsafe fn run(&self, kern: &Kernel, m0: usize, m1: usize, n0: usize, n1: usize) {
        let dst_stride_row = self.n * std::mem::size_of::<f32>();
        unsafe {
            (kern.run)(
                m1 - m0,
                n1 - n0,
                self.k,
                (self.lhs as *const u8).add((kern.lhs_packed_offset)(m0, self.k)) as *const c_void,
                (self.rhs as *const u8).add((kern.rhs_packed_offset)(n0, self.k)) as *const c_void,
                (self.dst as *mut u8).add((kern.dst_offset)(m0, n0, dst_stride_row)) as *mut f32,
                dst_stride_row,
                std::mem::size_of::<f32>(),
                f32::NEG_INFINITY,
                f32::INFINITY,
            )
        }
    }
}

/// Split the output over the pool and run one kernel call per participant.
///
/// Columns are split first, in multiples of `n_step`, so each participant streams its own
/// panel of the weight and writes a disjoint set of `dst` columns. Only when there are fewer
/// column blocks than participants are rows split too, in multiples of `m_step`.
fn run_split(kern: &Kernel, family: Family, job: &Job) {
    let nblocks = job.n.div_ceil(kern.n_step);
    let mblocks = job.m.div_ceil(kern.m_step);
    let cap = thread_cap(family);
    if cap <= 1 || (nblocks == 1 && mblocks == 1) {
        // SAFETY: one call covering the whole output.
        unsafe { job.run(kern, 0, job.m, 0, job.n) };
        return;
    }
    crate::threadpool::dispatch(|ith, nth| {
        let nth = nth.min(cap);
        let nsplit = nblocks.min(nth);
        let msplit = (nth / nsplit).min(mblocks).max(1);
        if ith >= nsplit * msplit {
            return;
        }
        let (ni, mi) = (ith % nsplit, ith / nsplit);
        let per_n = nblocks.div_ceil(nsplit);
        let per_m = mblocks.div_ceil(msplit);
        let n0 = (ni * per_n * kern.n_step).min(job.n);
        let n1 = ((ni + 1) * per_n * kern.n_step).min(job.n);
        let m0 = (mi * per_m * kern.m_step).min(job.m);
        let m1 = ((mi + 1) * per_m * kern.m_step).min(job.m);
        if n0 == n1 || m0 == m1 {
            return;
        }
        // SAFETY: the `(ni, mi)` ranges are disjoint across `ith` and aligned to the kernel's
        // steps by construction; bounds were checked by the caller.
        unsafe { job.run(kern, m0, m1, n0, n1) };
    });
}

/// Build this storage for a freshly read `q8_0` tensor, when the gate and the CPU allow.
///
/// `None` hands the decision back to [`super::repack`]; a shape that fails to pack is logged
/// and also falls through, since the plain layout answers the same matmuls.
pub fn q8_0_storage(src: &[BlockQ8_0], dims: &[usize]) -> Option<super::QStorage> {
    let family = family()?;
    let &[n, k] = dims else { return None };
    match Q8_0Kai::from_q8_0_for(family, src, n, k) {
        Ok(packed) => Some(super::QStorage::Cpu(Box::new(packed))),
        Err(e) => {
            tracing::warn!("kleidiai: keeping the plain layout for [{n}, {k}]: {e}");
            None
        }
    }
}

/// [`q8_0_storage`] for a weight that is still f32, skipping the `q8_0` intermediate.
pub fn q8_0_storage_f32(src: &[f32], dims: &[usize]) -> Option<super::QStorage> {
    let family = family()?;
    let &[n, k] = dims else { return None };
    match Q8_0Kai::from_f32_for(family, src, n, k) {
        Ok(packed) => Some(super::QStorage::Cpu(Box::new(packed))),
        Err(e) => {
            tracing::warn!("kleidiai: keeping the plain layout for [{n}, {k}]: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::QuantizedType;

    /// Uniform in `[-1, 1)` from a xorshift stream. Arithmetic sequences modulo a prime,
    /// which the other storages' tests use, are fine for exact comparisons but make the int8
    /// rounding errors of activations correlate across `k`, which overstates the kernel's error.
    fn uniform(seed: u64, len: usize) -> Vec<f32> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn weights(n: usize, k: usize) -> Vec<f32> {
        uniform(0x9E37_79B9_7F4A_7C15, n * k).into_iter().map(|v| v * 1.5).collect()
    }

    fn activations(m: usize, k: usize) -> Vec<f32> {
        uniform(0xD1B5_4A32_D192_ED03, m * k).into_iter().map(|v| v * 2.5).collect()
    }

    fn blocks(w: &[f32]) -> Vec<BlockQ8_0> {
        let mut blocks = vec![BlockQ8_0::zeros(); w.len() / QK8_0];
        BlockQ8_0::from_float(w, &mut blocks).unwrap();
        blocks
    }

    fn ref_matmul(w: &[f32], lhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut dst = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                dst[i * n + j] = (0..k).map(|l| lhs[i * k + l] * w[j * k + l]).sum();
            }
        }
        dst
    }

    /// Relative L2 distance, the right yardstick for int8 activations: their error is spread
    /// over every element rather than concentrated in one.
    fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
        let num: f32 = got.iter().zip(want).map(|(g, w)| (g - w).powi(2)).sum();
        let den: f32 = want.iter().map(|w| w.powi(2)).sum();
        (num / den.max(1e-20)).sqrt()
    }

    fn families() -> Vec<Family> {
        Family::ALL.into_iter().filter(|f| f.available()).collect()
    }

    #[test]
    fn some_family_is_available() {
        // Every AArch64 CPU this crate targets has dotprod, so the storage must be usable.
        assert!(!families().is_empty(), "no KleidiAI family for this CPU");
    }

    #[test]
    fn dequantize_is_the_per_row_requantization() {
        let (n, k) = (12, 64);
        let raw = weights(n, k);
        for family in families() {
            let packed = Q8_0Kai::from_f32_for(family, &raw, n, k).unwrap();
            let got = packed.dequantize(n * k).unwrap();
            for r in 0..n {
                let row = &raw[r * k..(r + 1) * k];
                let scale = row.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0;
                for (j, &w) in row.iter().enumerate() {
                    // Within half a step of the original and on the row's grid. Not the
                    // formula itself: a value on a rounding tie may legitimately go either way.
                    let g = got[r * k + j];
                    let steps = g / scale;
                    assert!(
                        (g - w).abs() <= scale / 2.0 + 1e-6,
                        "{}: row {r} col {j}: {g} vs {w}",
                        family.name()
                    );
                    assert!(
                        (steps - steps.round()).abs() < 1e-2 && steps.round().abs() <= 127.0,
                        "{}: row {r} col {j}: {g} is not on the grid",
                        family.name()
                    );
                }
            }
        }
    }

    #[test]
    fn matches_the_f32_reference() {
        // The decode gemv, a full tile, partial tiles, an `n` below any family's `nr`, and a
        // prefill-sized call.
        let mut failures = Vec::new();
        for (m, k, n) in
            [(1, 512, 512), (1, 64, 12), (4, 128, 16), (7, 96, 20), (33, 256, 68), (200, 512, 256)]
        {
            let raw = weights(n, k);
            let lhs = activations(m, k);
            for family in families() {
                let packed = Q8_0Kai::from_f32_for(family, &raw, n, k).unwrap();
                // Compare against what the kernel actually holds, so the only error left is the
                // activation quantization.
                let want = ref_matmul(&packed.dequantize(n * k).unwrap(), &lhs, m, k, n);
                let mut got = vec![0f32; m * n];
                packed.matmul_t((m, k, n), &lhs, &mut got).unwrap();
                let err = rel_l2(&got, &want);
                eprintln!("{} m={m} k={k} n={n}: rel l2 {err:.2e}", family.name());
                if err >= 5e-3 {
                    failures.push(format!("{} m={m} k={k} n={n}: rel l2 {err}", family.name()));
                }
                // Elementwise, relative to the output's scale rather than to each value: an
                // output near zero is the difference of large terms and carries their error.
                let tol = 2e-2 * want.iter().fold(0f32, |a, w| a.max(w.abs()));
                if let Some((i, (g, w))) = got
                    .iter()
                    .zip(&want)
                    .enumerate()
                    .find(|(_, (g, w))| !(g.is_finite() && (*g - *w).abs() <= tol))
                {
                    failures.push(format!(
                        "{} m={m} k={k} n={n} idx={i}: got {g}, want {w}",
                        family.name()
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn families_agree() {
        // Same activation quantization, same int8 weights, exact int32 accumulation: the
        // families can only differ by f32 rounding in the final scale.
        let fams = families();
        for (m, k, n) in [(1, 256, 64), (5, 128, 20), (64, 512, 128)] {
            let raw = weights(n, k);
            let lhs = activations(m, k);
            let outs: Vec<Vec<f32>> = fams
                .iter()
                .map(|&f| {
                    let packed = Q8_0Kai::from_f32_for(f, &raw, n, k).unwrap();
                    let mut got = vec![0f32; m * n];
                    packed.matmul_t((m, k, n), &lhs, &mut got).unwrap();
                    got
                })
                .collect();
            for (f, out) in fams.iter().zip(&outs).skip(1) {
                let err = rel_l2(out, &outs[0]);
                assert!(
                    err < 1e-5,
                    "{} vs {} m={m} k={k} n={n}: rel l2 {err}",
                    f.name(),
                    fams[0].name()
                );
            }
        }
    }

    #[test]
    fn from_q8_0_matches_from_f32_within_q8_0_error() {
        let (n, k) = (8, 96);
        let raw = weights(n, k);
        for family in families() {
            let from_blocks = Q8_0Kai::from_q8_0_for(family, &blocks(&raw), n, k).unwrap();
            let from_f32 = Q8_0Kai::from_f32_for(family, &raw, n, k).unwrap();
            let err = rel_l2(
                &from_blocks.dequantize(n * k).unwrap(),
                &from_f32.dequantize(n * k).unwrap(),
            );
            assert!(err < 1e-2, "{}: rel l2 {err}", family.name());
        }
    }

    #[test]
    fn raw_data_is_q8_0_of_the_stored_weights() {
        let (n, k) = (8, 64);
        let raw = weights(n, k);
        for family in families() {
            let packed = Q8_0Kai::from_f32_for(family, &raw, n, k).unwrap();
            let bytes = packed.raw_data().unwrap();
            assert_eq!(bytes.len(), packed.storage_size_in_bytes());
            assert_eq!(bytes.len(), n * k / QK8_0 * std::mem::size_of::<BlockQ8_0>());
            let blocks = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const BlockQ8_0,
                    bytes.len() / std::mem::size_of::<BlockQ8_0>(),
                )
            };
            let mut back = vec![0f32; n * k];
            BlockQ8_0::to_float(blocks, &mut back).unwrap();
            let err = rel_l2(&back, &packed.dequantize(n * k).unwrap());
            assert!(err < 1e-2, "{}: rel l2 {err}", family.name());
        }
    }

    #[test]
    fn from_float_replaces_the_weights() {
        let (n, k) = (4, 64);
        for family in families() {
            let mut packed = Q8_0Kai::from_f32_for(family, &weights(n, k), n, k).unwrap();
            let other: Vec<f32> = weights(n, k).iter().map(|v| -v * 0.5).collect();
            packed.from_float(&other).unwrap();
            let err = rel_l2(&packed.dequantize(n * k).unwrap(), &other);
            assert!(err < 1e-2, "{}: rel l2 {err}", family.name());
        }
    }

    #[test]
    fn storage_hooks_follow_the_gate() {
        let (n, k) = (8, 64);
        let raw = weights(n, k);
        let storage = q8_0_storage(&blocks(&raw), &[n, k]);
        assert_eq!(storage.is_some(), active());
        assert!(q8_0_storage(&blocks(&raw), &[n * k]).is_none(), "1-D tensors stay plain");
        let direct = q8_0_storage_f32(&raw, &[n, k]);
        assert_eq!(direct.is_some(), active());
        assert!(q8_0_storage_f32(&raw, &[n * k]).is_none(), "1-D tensors stay plain");
        if let Some(super::super::QStorage::Cpu(direct)) = direct {
            // One rounding: every stored value is within half a row step of the original.
            let got = direct.dequantize(n * k).unwrap();
            for r in 0..n {
                let row = &raw[r * k..(r + 1) * k];
                let step = row.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0;
                for (g, w) in got[r * k..(r + 1) * k].iter().zip(row) {
                    assert!((g - w).abs() <= step / 2.0 + 1e-6, "row {r}: {g} vs {w}");
                }
            }
        }
    }
}
