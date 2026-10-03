//! Arm KleidiAI micro-kernels for the quantized linears, behind the `kai` feature.
//!
//! [KleidiAI] is Arm's library of matmul micro-kernels: one hand-written routine per
//! instruction-set tier, each with packing routines that lay the operands out the way that
//! kernel reads them. What it adds over the kernels in [`super::repack`] is SME2. Apple's M4
//! and M5, and Arm's Cortex-X925 onwards, carry a Scalable Matrix Extension unit whose
//! outer-product `mopa` instruction multiplies a whole tile per instruction, and it is only
//! reachable from assembly: Rust has no stable SME intrinsics, and streaming mode's calling
//! convention rules out inline `asm!` in practice.
//!
//! `build.rs` compiles the kernels from a KleidiAI source tree under the `kai` feature. This
//! module is the Rust side. It picks one kernel [`Family`] per process from the CPU's
//! features, which is what everything else here is packed and dispatched for.
//!
//! [KleidiAI]: https://github.com/ARM-software/kleidiai
//!
//! | family    | instructions | hardware |
//! |-----------|--------------|----------|
//! | `Sme2`    | SME2 outer product, SME2 dot | Apple M4/M5, Cortex-X925 and later |
//! | `I8mm`    | `smmla` for gemm, `sdot` for gemv | Apple M4/M5, Cortex-A710, Neoverse N2 |
//! | `Dotprod` | `sdot` | Apple M1-M3, Cortex-A75 and later, Raspberry Pi 5 |
//!
//! `XN_KAI=0` turns the whole path off; `XN_KAI=sme2|i8mm|dotprod` pins a family, for
//! comparing them on a CPU that has more than one. Both are read once per process.

use std::sync::OnceLock;

unsafe extern "C" {
    fn xn_kai_cpu_features() -> u32;
}

// Bit values of `xn_kai_cpu_features`, mirrored from `csrc/xn_kai.c`.
const FEAT_DOTPROD: u32 = 1;
const FEAT_I8MM: u32 = 2;
const FEAT_SME2: u32 = 8;

fn cpu_features() -> u32 {
    static F: OnceLock<u32> = OnceLock::new();
    // SAFETY: a pure probe with no preconditions.
    *F.get_or_init(|| unsafe { xn_kai_cpu_features() })
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
}

/// The kernel family this process uses, or `None` if the path is off (`XN_KAI=0`) or the
/// CPU has no family it can run. Decided once, because weights are packed for it.
pub fn family() -> Option<Family> {
    static F: OnceLock<Option<Family>> = OnceLock::new();
    *F.get_or_init(|| match std::env::var("XN_KAI").as_deref() {
        Ok("0") => None,
        Ok("") | Ok("1") | Err(_) => Family::best(),
        Ok(name) => match Family::parse(name) {
            Some(f) if f.available() => Some(f),
            Some(f) => {
                tracing::warn!("XN_KAI={name}: this CPU lacks {}, KleidiAI off", f.name());
                None
            }
            None => {
                tracing::warn!("XN_KAI={name} is not a kernel family, using the default");
                Family::best()
            }
        },
    })
}

/// Whether the KleidiAI kernels are in use.
pub fn active() -> bool {
    family().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every AArch64 CPU this crate targets has dotprod, so some family must be usable.
    #[test]
    fn some_family_is_available() {
        assert!(Family::best().is_some(), "no KleidiAI family for this CPU");
        assert!(Family::Dotprod.available(), "dotprod is the AArch64 baseline");
    }

    /// The families are ordered strongest first, and a CPU that can run one can run every
    /// weaker one: SME2 and i8mm both imply dotprod.
    #[test]
    fn availability_is_monotonic() {
        let avail: Vec<bool> = Family::ALL.iter().map(|f| f.available()).collect();
        if avail[0] || avail[1] {
            assert!(avail[2], "a CPU with SME2 or i8mm must also have dotprod");
        }
        assert_eq!(Family::best(), Family::ALL.into_iter().find(|f| f.available()));
    }

    #[test]
    fn names_round_trip() {
        for f in Family::ALL {
            assert_eq!(Family::parse(f.name()), Some(f), "{}", f.name());
        }
        assert_eq!(Family::parse("avx512"), None);
    }

    /// The gate answers consistently, and agrees with `active`.
    #[test]
    fn the_gate_is_stable() {
        assert_eq!(family(), family());
        assert_eq!(active(), family().is_some());
        if let Some(f) = family() {
            assert!(f.available(), "the gate picked a family this CPU cannot run");
        }
    }
}
