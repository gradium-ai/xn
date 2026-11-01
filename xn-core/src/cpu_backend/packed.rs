//! Explicit, model-owned packing for repeated CPU f32 matrix products.
//!
//! Construction snapshots `[outputs, inputs]` weights (and optional bias). Mutating
//! the original slices cannot change this operator. There is no global weight cache:
//! dropping the operator releases its allocation. The caller owns reusable scratch,
//! so shared operators can run concurrently with separate workspaces.
//!
//! This is an opt-in alternative to ordinary GEMM, not an automatic dispatch policy.
//! `try_new` returns `None` on CPUs/builds without the supported kernel, allowing the
//! caller to retain its ordinary operator. Packing need not benefit every shape.

/// Reusable input-packing workspace. Capacity follows the largest product used by
/// this workspace and is released on drop. Do not share mutable scratch across calls.
#[derive(Default)]
pub struct PackedF32Scratch {
    #[cfg(xn_kai)]
    pub(crate) lhs: Vec<f32>,
}

pub(crate) fn validate(
    weights: &[f32],
    outputs: usize,
    inputs: usize,
    bias: Option<&[f32]>,
) -> crate::Result<()> {
    if outputs == 0
        || inputs == 0
        || outputs > i32::MAX as usize
        || inputs > i32::MAX as usize
        || outputs.checked_mul(inputs) != Some(weights.len())
        || bias.is_some_and(|b| b.len() != outputs)
    {
        crate::bail!("invalid packed f32 weight or bias dimensions");
    }
    Ok(())
}

#[cfg(xn_kai)]
pub use crate::kai_f32::PackedF32;

/// A snapshot of constant f32 weights for `X @ W^T + bias`.
#[cfg(not(xn_kai))]
pub struct PackedF32;

#[cfg(not(xn_kai))]
impl PackedF32 {
    /// Validate dimensions; this build has no supported packed kernel.
    pub fn try_new(
        weights: &[f32],
        outputs: usize,
        inputs: usize,
        bias: Option<&[f32]>,
    ) -> crate::Result<Option<Self>> {
        validate(weights, outputs, inputs, bias)?;
        Ok(None)
    }
    /// Bytes retained by the packed representation.
    pub fn packed_bytes(&self) -> usize {
        0
    }
    /// Compute a dense row-major product with caller-owned output and scratch.
    pub fn matmul(
        &self,
        _: &[f32],
        _: usize,
        _: &mut [f32],
        _: &mut PackedF32Scratch,
    ) -> crate::Result<()> {
        crate::bail!("packed f32 kernel is unavailable");
    }
}
