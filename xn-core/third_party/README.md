# Third-party code

## KleidiAI

`kleidiai/` holds files copied from Arm's [KleidiAI](https://github.com/ARM-software/kleidiai),
release tag **v1.24.0** (commit `0b2ee513`), unchanged and at their paths in that repository.
They are licensed under Apache-2.0 (`kleidiai/LICENSES/Apache-2.0.txt`), and each file keeps
its own copyright header.

The `kai` feature of `xn-core` uses these files and nothing else:

- the SME2 int8 matmul kernels, one gemm (`..._sme2_mopa`) and one gemv (`..._sme2_dot`)
- the SME2 f32 kernels, one gemm (`..._f32p2vlx1biasf32_sme2_mopa`) and one gemv
  (`..._f32p2vlx1b_1x16vl_sme2_mla`)
- the routines that pack their operands (`pack/`)
- the shared header and the SME helper assembly (`kai_common.h`, `kai_common_sme_asm.S`)

`build.rs` compiles them only when the feature is on and the target can run them. They are
ordinary files in this repository, so every checkout and every published crate has them.

To update KleidiAI, copy the same files from a newer tag over these, and change the tag
above. If a file was renamed or a new one is needed, change `build.rs` too. The kernel
interfaces are declared by hand in `src/quantized/kai.rs` and `src/kai_f32.rs`, so check them
against the new headers.
