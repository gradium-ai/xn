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

# Backend harness

`harness/` runs the backend itself in a browser -- device creation, the uniform
parameter ring, batching, the buffer pool, the async readback -- against expected
values computed on the spot. The shader check above only proves the WGSL compiles;
this proves the orchestration works.

```
cd harness
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir pkg \
  target/wasm32-unknown-unknown/release/xn_webgpu_browser_harness.wasm
cd pkg && node server.js &
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --enable-unsafe-webgpu --no-first-run \
  --user-data-dir=/tmp/xn-harness-cd http://localhost:8733/
```

`"failed": 0` is the pass condition. Ops run through the ordinary synchronous
`Backend` trait, because they only record into the batch; only readbacks are
awaited, via `Device::tensor_to_vec`.

The harness is a standalone crate (its own `[workspace]`), since the xn workspace
only ever builds for native.

# Capability check (any device)

`caps.html` is standalone -- no wasm, no build -- and answers whether a given
device can run this backend. It checks the limits the kernels actually need, then
does a minimal dispatch in the same binding shape they use (one read_write and
one read-only storage buffer, plus a uniform at a non-zero dynamic offset).

```
node serve.js 8734      # serves this directory on 0.0.0.0:8734
```

The required limits are all below WebGPU's guaranteed minimums, so a conformant
implementation should pass. What varies by device is `shader-f16` (optional; f32
otherwise, at twice the weight bytes) and whether the driver is blocklisted.

## Testing on a phone

WebGPU needs a secure context, so `http://<LAN-IP>:8734` will report "not a
secure context" and `navigator.gpu` will be missing. Two ways round it:

- USB, no certificates: `adb reverse tcp:8734 tcp:8734`, then open
  `http://localhost:8734` on the phone. localhost is a secure context.
- An https tunnel (cloudflared, ngrok) pointed at port 8734.

Chrome enabled WebGPU by default on Android 12+ with Qualcomm and ARM GPUs in
Chrome 121; older Chrome or an unsupported driver shows up as a missing
`navigator.gpu` or a null adapter, which the page reports distinctly.
