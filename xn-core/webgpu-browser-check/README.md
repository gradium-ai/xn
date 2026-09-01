# Browser WGSL validation

Compiles every WebGPU kernel with a real browser's WGSL implementation and
reports anything it rejects.

Browsers validate WGSL more strictly than Naga, which is what wgpu uses on
native, and the backend compiles kernels lazily on first dispatch. So a kernel a
browser rejects will not surface natively at all, and in the browser only when
some particular op first runs. This check compiles all of them up front.

It found a real one on its first run: `const S_NEG_BIG: S = -3.4028235e38;` --
the shortest round-trip form Rust prints for `f32::MIN` -- is rejected by Chrome
because the literal is evaluated as an abstract float and range-checked before
rounding, and its magnitude exceeds `f32::MAX`. Naga accepts it. That was a bug
in all 29 f32 kernels.

## Usage

```
cargo run --release --features webgpu --example webgpu_dump_shaders -- /tmp/xn-wgsl
cp /tmp/xn-wgsl/*.wgsl /tmp/xn-wgsl/index.json xn-core/webgpu-browser-check/
cd xn-core/webgpu-browser-check && node server.js &
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --enable-unsafe-webgpu --no-first-run \
  --user-data-dir=/tmp/xn-wgsl-cd http://localhost:8732/
```

The server prints a JSON report and exits. `failed: 0` is the pass condition.
Each shader is both compiled (`createShaderModule`, checking
`getCompilationInfo`) and turned into a pipeline with an `auto` layout, which is
what forces full validation of the entry point and its bindings.

Headless Chrome does WebGPU on macOS with `--enable-unsafe-webgpu`; the
`CVDisplayLinkCreateWithCGDisplay` errors it logs are harmless.
