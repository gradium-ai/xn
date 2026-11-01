# Explicit packed f32 CPU matrices

`cpu_backend::packed::PackedF32` snapshots constant `[outputs, inputs]` weights
for repeated `X @ W^T + bias` products. It is opt-in: ordinary `Linear`, GEMM and
convolution dispatch are unchanged. The current implementation uses the vendored
KleidiAI SME2 f32 kernel when compiled with `kai` and supported by the CPU.

```rust
use xn::cpu_backend::packed::{PackedF32, PackedF32Scratch};

fn example() -> xn::Result<()> {
    let weights = vec![1f32, 2., 3., 4., 5., 6.]; // [2 outputs, 3 inputs]
    let Some(matrix) = PackedF32::try_new(&weights, 2, 3, Some(&[0.5, -0.5]))? else {
        // Use the application's ordinary operator on an unsupported CPU/build.
        return Ok(());
    };
    let mut scratch = PackedF32Scratch::default();
    let input = vec![1., 0., 1., 0., 1., 0.]; // [2 rows, 3 inputs]
    let mut output = vec![0.; 4];            // [2 rows, 2 outputs]
    matrix.matmul(&input, 2, &mut output, &mut scratch)?;
    assert_eq!(output, vec![4.5, 9.5, 2.5, 4.5]);
    Ok(())
}
```

The input and output slices are dense row-major arrays with exact checked lengths.
Zero input rows are supported; zero-sized weights are rejected. Invalid dimensions
return errors before native access. Construction validates arguments even on an
unsupported build, then returns `None` if the kernel is unavailable or `XN_KAI=0`.

The operator owns its packed representation. It does not retain a pointer to the
original weights or bias: changing those slices after construction has no effect.
Dropping the operator frees its packed weights; `packed_bytes()` reports their
size. The application decides whether to retain the original weights for fallback.
There is no global packed-weight cache or implicit lookup by allocation address.

Scratch is separate and owned by the caller. Its input-packing buffer grows to the
largest product run through that workspace and is freed on drop. Share immutable
matrices across threads, with a separate mutable workspace and output per call.
The kernel's vector-length-dependent tile sizes are checked before each call;
a thread with an incompatible SME vector length receives an error rather than
using a different packed layout. This is not a CPU affinity or scheduling API.

All arithmetic stays f32, but reduction order can differ from other GEMM kernels.
Validate tolerances for the application. Packing is not guaranteed to beat
Accelerate or another GEMM for all shapes; include input packing, layout conversion,
memory cost, model loading and full-workload latency in comparisons. In particular,
convolution lowering can outweigh any matrix-kernel saving.

Tests:

```sh
cargo test -p xn --release --no-default-features --features accelerate,kai --test packed_f32_tests
cargo test -p xn --release --no-default-features --test packed_f32_tests
XN_KAI=0 cargo test -p xn --release --no-default-features --features accelerate,kai --test packed_f32_tests
```

The native test exercises padded/odd dimensions, biases, mutation of original
weights, changing row counts, empty input, invalid lengths and simultaneous calls
with independent workspaces against a scalar reference. Hardware without SME2
validates construction/fallback but cannot validate native arithmetic.
