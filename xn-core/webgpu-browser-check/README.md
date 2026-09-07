# Browser checks for the WebGPU backend

Three things live here, all served by one HTTPS server so they can be run on a
phone as easily as on the machine that built them.

| page | what it needs | what it tells you |
| --- | --- | --- |
| `caps.html` | nothing, pure JS | whether this device can run the backend at all |
| `harness.html` | the wasm build below | the real backend's op set, correct or not, and a benchmark |
| shader dump | `webgpu_dump_shaders` | whether every kernel compiles under a browser's WGSL validator |

## Serving

```
node serve.js            # https on 8734, http on 8735
```

It prints every URL it is reachable on. A self-signed certificate is generated
into `.tls/` on first run, with the machine's current addresses as
subject-alt-names so it survives a DHCP change.

**WebGPU requires a secure context.** Over plain `http://` from a LAN address
`navigator.gpu` is simply absent, however capable the device is. `localhost`
counts as secure, so the http port is fine on the machine itself; from a phone
use the https URL and accept the warning (Advanced, then Proceed). Binding is on
all interfaces, which is the point, but anything on the network can reach it
while it runs.

## Building the wasm harness

```
cd harness
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir pkg \
  target/wasm32-unknown-unknown/release/xn_webgpu_browser_harness.wasm
```

Needs `rustup target add wasm32-unknown-unknown` and `cargo install
wasm-bindgen-cli`. `harness/` is a standalone crate (its own `[workspace]`),
since the xn workspace only ever builds for native. `pkg/` is build output;
`harness.html` lives outside it.

`harness.html` has buttons, or `?auto=run` / `?auto=bench` to start immediately
so the same page can be driven headlessly. Results are also POSTed back, which
the server prints — that is how the numbers below were collected.

Ops run through the ordinary synchronous `Backend` trait, because they only
record into the batch. Only readbacks are awaited, via `Device::tensor_to_vec`.

## Shader validation

Browsers validate WGSL more strictly than Naga, and the backend compiles kernels
lazily on first dispatch, so a kernel a browser rejects will not surface natively
at all.

```
cargo run --release --features webgpu --example webgpu_dump_shaders -- ./shaders
```

then point a page at the dumped `index.json`. It found a real one on its first
run: `const S_NEG_BIG: S = -3.4028235e38;` — the shortest round-trip form Rust
prints for `f32::MIN` — is rejected by Chrome, which evaluates the literal as an
abstract float and range-checks before rounding, and its magnitude exceeds
`f32::MAX`. Naga accepts it. That was a bug in all 29 f32 kernels.

## Headless, for CI

```
node serve.js &
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --enable-unsafe-webgpu --no-first-run \
  --user-data-dir=/tmp/xn-cd "http://localhost:8735/harness.html?auto=run"
```

`"failed": 0` in the POSTed report is the pass condition. Use the http port for
this: headless Chrome does not reliably accept the self-signed certificate, and
localhost is a secure context anyway. The `CVDisplayLinkCreateWithCGDisplay`
errors it logs on macOS are harmless.
