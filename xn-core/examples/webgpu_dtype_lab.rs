//! Measures f32 / f16 / q8_0 decode-GEMV on phonon's real weight shapes.
//!
//! Companion to `webgpu_gemv_lab`, which established that the launch shape that
//! wins on these shapes is one thread per output column (no cross-lane
//! reduction), reaching ~150 GB/s of weight traffic. Decode is bound by exactly
//! that traffic, so narrower weights should buy close to their byte ratio.
//!
//! WGSL cannot load ggml's `q8_0` block (`{f16 d; i8 qs[32]}`, 34 B) directly:
//! storage buffers address 4-byte words and 34 does not divide, so a block's
//! scale and quants straddle word boundaries differently in every block. The
//! layout tested here therefore splits the tensor into two buffers -- one scale
//! per 32-value block, and the quants packed 4-per-u32 -- which is what a
//! load-time repack would produce.
//!
//! Run with: cargo run --release --features webgpu --example webgpu_dtype_lab
// Reading a byte buffer back as fixed-width scalars: the constant chunk size *is*
// the element width, which reads better here than `as_chunks`.
#![allow(clippy::chunks_exact_to_as_chunks)]

use half::f16;
use std::time::Instant;

const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("flow_lm in_proj      ", 2304, 768, 12),
    ("flow_lm out_proj     ", 768, 768, 12),
    ("flow_lm linear1      ", 3072, 768, 12),
    ("flow_lm linear2      ", 768, 3072, 12),
    ("flow_net resblk ada  ", 1536, 512, 6),
    ("flow_net resblk mlp  ", 512, 512, 12),
    ("flow_net cond_embed  ", 512, 768, 1),
    ("flow_net time embed  ", 512, 256, 2),
    ("flow_net final ada   ", 1024, 512, 1),
    ("mimi tf in_proj      ", 1536, 512, 2),
    ("mimi tf linear1      ", 2048, 512, 2),
    ("mimi tf linear2      ", 512, 2048, 2),
];

/// One thread per output column, f32 weights, vec4 loads.
const F32_TPC: &str = r#"
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> lhs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }
    let k4 = pc.k >> 2u;
    let base = j * k4;
    var acc = vec4<f32>(0.0);
    for (var g = 0u; g < k4; g = g + 1u) { acc = acc + w[base + g] * lhs[g]; }
    dst[j] = acc.x + acc.y + acc.z + acc.w;
}
"#;

/// Same launch shape, f16 weights (8 B per vec4 load), f32 accumulation.
const F16_TPC: &str = r#"
enable f16;
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> lhs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f16>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }
    let k4 = pc.k >> 2u;
    let base = j * k4;
    var acc = vec4<f32>(0.0);
    for (var g = 0u; g < k4; g = g + 1u) {
        acc = acc + vec4<f32>(w[base + g]) * lhs[g];
    }
    dst[j] = acc.x + acc.y + acc.z + acc.w;
}
"#;

/// q8_0, split layout, quants read as scalar u32 (4 weights per load).
/// `unpack4xI8` is a WGSL builtin; the manual variant below is the fallback.
const Q8_TPC_UNPACK: &str = r#"
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> lhs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> q: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }
    let nb = pc.k >> 5u;          // 32-value blocks in a row
    let qbase = j * (pc.k >> 2u); // u32 words in a row (4 weights each)
    let sbase = j * nb;
    var total = 0.0;
    for (var b = 0u; b < nb; b = b + 1u) {
        var acc = vec4<f32>(0.0);
        let w0 = qbase + b * 8u;
        let l0 = b * 8u;
        for (var t = 0u; t < 8u; t = t + 1u) {
            acc = acc + vec4<f32>(unpack4xI8(q[w0 + t])) * lhs[l0 + t];
        }
        total = total + scales[sbase + b] * (acc.x + acc.y + acc.z + acc.w);
    }
    dst[j] = total;
}
"#;

/// q8_0, split layout, quants read as `vec4<u32>` (16 weights per load).
const Q8_TPC_VEC: &str = r#"
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> lhs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> q: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }
    let nb = pc.k >> 5u;
    let qbase = j * (pc.k >> 4u); // vec4<u32> per row (16 weights each)
    let sbase = j * nb;
    var total = 0.0;
    for (var b = 0u; b < nb; b = b + 1u) {
        var acc = vec4<f32>(0.0);
        let v0 = qbase + b * 2u;
        let l0 = b * 8u;
        let a = q[v0];
        let c = q[v0 + 1u];
        acc = acc + vec4<f32>(unpack4xI8(a.x)) * lhs[l0];
        acc = acc + vec4<f32>(unpack4xI8(a.y)) * lhs[l0 + 1u];
        acc = acc + vec4<f32>(unpack4xI8(a.z)) * lhs[l0 + 2u];
        acc = acc + vec4<f32>(unpack4xI8(a.w)) * lhs[l0 + 3u];
        acc = acc + vec4<f32>(unpack4xI8(c.x)) * lhs[l0 + 4u];
        acc = acc + vec4<f32>(unpack4xI8(c.y)) * lhs[l0 + 5u];
        acc = acc + vec4<f32>(unpack4xI8(c.z)) * lhs[l0 + 6u];
        acc = acc + vec4<f32>(unpack4xI8(c.w)) * lhs[l0 + 7u];
        total = total + scales[sbase + b] * (acc.x + acc.y + acc.z + acc.w);
    }
    dst[j] = total;
}
"#;

/// q8_0 with the scale folded per block but quants unpacked by hand, in case
/// `unpack4xI8` is unavailable or slow.
const Q8_TPC_MANUAL: &str = r#"
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> lhs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> q: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
fn i8x4(v: u32) -> vec4<f32> {
    let s = vec4<u32>(v << 24u, v << 16u, v << 8u, v);
    return vec4<f32>(vec4<i32>(s) >> vec4<u32>(24u));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }
    let nb = pc.k >> 5u;
    let qbase = j * (pc.k >> 4u);
    let sbase = j * nb;
    var total = 0.0;
    for (var b = 0u; b < nb; b = b + 1u) {
        var acc = vec4<f32>(0.0);
        let v0 = qbase + b * 2u;
        let l0 = b * 8u;
        let a = q[v0];
        let c = q[v0 + 1u];
        acc = acc + i8x4(a.x) * lhs[l0];
        acc = acc + i8x4(a.y) * lhs[l0 + 1u];
        acc = acc + i8x4(a.z) * lhs[l0 + 2u];
        acc = acc + i8x4(a.w) * lhs[l0 + 3u];
        acc = acc + i8x4(c.x) * lhs[l0 + 4u];
        acc = acc + i8x4(c.y) * lhs[l0 + 5u];
        acc = acc + i8x4(c.z) * lhs[l0 + 6u];
        acc = acc + i8x4(c.w) * lhs[l0 + 7u];
        total = total + scales[sbase + b] * (acc.x + acc.y + acc.z + acc.w);
    }
    dst[j] = total;
}
"#;

/// The structure the backend kernel first shipped with: the block's 32 unpacked
/// weights held in an `array<vec4<f32>, 8>` and consumed through a dynamically
/// indexed inner loop, wrapped in a row-blocking loop for m > 1. Dynamic
/// indexing can force that array out of registers into thread-local memory,
/// which would cost far more than the packing saves -- this measures whether it
/// does.
const Q8_ARRAY_MLOOP: &str = r#"
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> lhs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> q: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }
    let nb = pc.k >> 5u;
    let qrow = j * (pc.k >> 4u);
    let srow = j * nb;
    let l4 = pc.k >> 2u;
    let m = 1u;
    var m0 = 0u;
    while m0 < m {
        let rows = min(4u, m - m0);
        var tot = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
        for (var b = 0u; b < nb; b = b + 1u) {
            let lo = q[qrow + b * 2u];
            let hi = q[qrow + b * 2u + 1u];
            var wv = array<vec4<f32>, 8>(
                vec4<f32>(unpack4xI8(lo.x)), vec4<f32>(unpack4xI8(lo.y)),
                vec4<f32>(unpack4xI8(lo.z)), vec4<f32>(unpack4xI8(lo.w)),
                vec4<f32>(unpack4xI8(hi.x)), vec4<f32>(unpack4xI8(hi.y)),
                vec4<f32>(unpack4xI8(hi.z)), vec4<f32>(unpack4xI8(hi.w)),
            );
            let sc = scales[srow + b];
            for (var r = 0u; r < rows; r = r + 1u) {
                let lbase = (m0 + r) * l4 + b * 8u;
                var acc = vec4<f32>(0.0);
                for (var t = 0u; t < 8u; t = t + 1u) {
                    acc = acc + wv[t] * lhs[lbase + t];
                }
                tot[r] = tot[r] + sc * (acc.x + acc.y + acc.z + acc.w);
            }
        }
        for (var r = 0u; r < rows; r = r + 1u) {
            dst[(m0 + r) * pc.n + j] = tot[r];
        }
        m0 = m0 + 4u;
    }
}
"#;

/// Same row blocking, but the block's weights stay in named locals and the eight
/// lanes are written out, so nothing is dynamically indexed.
const Q8_UNROLLED_MLOOP: &str = r#"
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> lhs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> q: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }
    let nb = pc.k >> 5u;
    let qrow = j * (pc.k >> 4u);
    let srow = j * nb;
    var tot = 0.0;
    for (var b = 0u; b < nb; b = b + 1u) {
        let lo = q[qrow + b * 2u];
        let hi = q[qrow + b * 2u + 1u];
        let l0 = b * 8u;
        var acc = vec4<f32>(unpack4xI8(lo.x)) * lhs[l0];
        acc = acc + vec4<f32>(unpack4xI8(lo.y)) * lhs[l0 + 1u];
        acc = acc + vec4<f32>(unpack4xI8(lo.z)) * lhs[l0 + 2u];
        acc = acc + vec4<f32>(unpack4xI8(lo.w)) * lhs[l0 + 3u];
        acc = acc + vec4<f32>(unpack4xI8(hi.x)) * lhs[l0 + 4u];
        acc = acc + vec4<f32>(unpack4xI8(hi.y)) * lhs[l0 + 5u];
        acc = acc + vec4<f32>(unpack4xI8(hi.z)) * lhs[l0 + 6u];
        acc = acc + vec4<f32>(unpack4xI8(hi.w)) * lhs[l0 + 7u];
        tot = tot + scales[srow + b] * (acc.x + acc.y + acc.z + acc.w);
    }
    dst[j] = tot;
}
"#;

enum W {
    F32,
    F16,
    Q8,
}

struct Variant {
    label: &'static str,
    src: &'static str,
    w: W,
}

const VARIANTS: &[Variant] = &[
    Variant { label: "f32 thread-per-col", src: F32_TPC, w: W::F32 },
    Variant { label: "f16 thread-per-col", src: F16_TPC, w: W::F16 },
    Variant { label: "q8_0 u32 + unpack4xI8", src: Q8_TPC_UNPACK, w: W::Q8 },
    Variant { label: "q8_0 vec4<u32> + unpack", src: Q8_TPC_VEC, w: W::Q8 },
    Variant { label: "q8_0 vec4<u32> + manual", src: Q8_TPC_MANUAL, w: W::Q8 },
    Variant { label: "q8_0 array + m-loop", src: Q8_ARRAY_MLOOP, w: W::Q8 },
    Variant { label: "q8_0 unrolled + m-loop", src: Q8_UNROLLED_MLOOP, w: W::Q8 },
];

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    pl: wgpu::PipelineLayout,
}

impl Gpu {
    fn new() -> Self {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = instance
                .enumerate_adapters(wgpu::Backends::all())
                .into_iter()
                .max_by_key(|a| match a.get_info().device_type {
                    wgpu::DeviceType::DiscreteGpu => 4,
                    wgpu::DeviceType::IntegratedGpu => 3,
                    _ => 1,
                })
                .expect("no adapter");
            let info = adapter.get_info();
            println!("device: {} ({:?})", info.name, info.backend);
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::PUSH_CONSTANTS | wgpu::Features::SHADER_F16,
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 128,
                        ..adapter.limits()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("request_device");
            let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..4)
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: i != 0 },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..128,
                }],
            });
            Gpu { device, queue, bgl, pl }
        })
    }

    fn pipeline(&self, src: &str) -> Result<wgpu::ComputePipeline, String> {
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&self.pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        match pollster::block_on(self.device.pop_error_scope()) {
            Some(e) => Err(format!("{e}").lines().take(6).collect::<Vec<_>>().join(" | ")),
            None => Ok(p),
        }
    }

    fn buf(&self, bytes: &[u8]) -> wgpu::Buffer {
        let b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes.len().max(4) as u64).div_ceil(4) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&b, 0, bytes);
        b
    }

    fn read_f32(&self, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let bytes = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::Wait).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let out: Vec<f32> =
            mapped.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(mapped);
        staging.unmap();
        out
    }

    fn time(
        &self,
        p: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        push: &[u8],
        groups: u32,
        iters: usize,
    ) -> f64 {
        let run = |n: usize| {
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(p);
                cp.set_bind_group(0, bg, &[]);
                cp.set_push_constants(0, push);
                for _ in 0..n {
                    cp.dispatch_workgroups(groups, 1, 1);
                }
            }
            self.queue.submit(Some(enc.finish()));
            self.device.poll(wgpu::PollType::Wait).unwrap();
        };
        run(16);
        let t = Instant::now();
        run(iters);
        t.elapsed().as_secs_f64() * 1e6 / iters as f64
    }
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Quantize one row to q8_0 in the split layout: one f32 scale per 32 values,
/// quants packed 4-per-u32. Returns the dequantized row too, so the reference
/// result carries the same quantization error and the check stays exact.
fn quant_row(row: &[f32], scales: &mut Vec<f32>, quants: &mut Vec<u32>, deq: &mut Vec<f32>) {
    for blk in row.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        scales.push(d);
        let qs: Vec<i8> =
            blk.iter().map(|&v| (v * id).round().clamp(-127.0, 127.0) as i8).collect();
        for q in &qs {
            deq.push(*q as f32 * d);
        }
        for w in qs.chunks(4) {
            quants.push(u32::from_le_bytes([w[0] as u8, w[1] as u8, w[2] as u8, w[3] as u8]));
        }
    }
}

fn main() {
    let gpu = Gpu::new();

    let mut pipes = Vec::new();
    for v in VARIANTS {
        match gpu.pipeline(v.src) {
            Ok(p) => pipes.push((v, p)),
            Err(e) => println!("  !! {} failed to compile: {e}", v.label),
        }
    }
    println!();

    let mut totals = vec![0f64; pipes.len()];

    for &(label, n, k, per_frame) in SHAPES {
        let lhs: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
        let w32: Vec<f32> = (0..n * k).map(|i| ((i % 251) as f32 - 125.0) / 256.0).collect();
        let w16: Vec<f16> = w32.iter().map(|&v| f16::from_f32(v)).collect();

        let (mut qs, mut qq, mut qdeq) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            quant_row(&w32[j * k..(j + 1) * k], &mut qs, &mut qq, &mut qdeq);
        }

        // One reference per weight representation, so relerr measures the
        // kernel and not the format.
        let refr = |wv: &dyn Fn(usize, usize) -> f32| -> Vec<f32> {
            (0..n).map(|j| (0..k).map(|l| lhs[l] * wv(j, l)).sum::<f32>()).collect::<Vec<f32>>()
        };
        let ref32 = refr(&|j, l| w32[j * k + l]);
        let ref16 = refr(&|j, l| w16[j * k + l].to_f32());
        let refq8 = refr(&|j, l| qdeq[j * k + l]);

        let dst = gpu.buf(as_bytes(&vec![0f32; n]));
        let lb = gpu.buf(as_bytes(&lhs));
        let b32 = gpu.buf(as_bytes(&w32));
        let b16 = gpu.buf(as_bytes(&w16));
        let bq = gpu.buf(as_bytes(&qq));
        let bs = gpu.buf(as_bytes(&qs));

        let mut push = Vec::new();
        push.extend_from_slice(&(n as u32).to_le_bytes());
        push.extend_from_slice(&(k as u32).to_le_bytes());

        println!("{label} n={n:<5} k={k:<5}  x{per_frame}/frame");
        for (vi, (v, p)) in pipes.iter().enumerate() {
            let (wbuf, sbuf, wbytes, reference) = match v.w {
                W::F32 => (&b32, &b32, n * k * 4, &ref32),
                W::F16 => (&b16, &b16, n * k * 2, &ref16),
                W::Q8 => (&bq, &bs, n * k + n * (k / 32) * 4, &refq8),
            };
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &gpu.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: dst.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: lb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: sbuf.as_entire_binding() },
                ],
            });
            let iters = if n * k > 1 << 21 { 400 } else { 1200 };
            let us = gpu.time(p, &bg, &push, n.div_ceil(64) as u32, iters);
            let got = gpu.read_f32(&dst, n);
            let maxerr = got
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                .fold(0f32, f32::max);
            let ok = if maxerr < 5e-3 { "ok " } else { "BAD" };
            totals[vi] += us * per_frame as f64;
            println!(
                "    {:<26} {us:>8.2} us  {:>6.0} GB/s  {:>5.2} MB  {ok} (relerr {maxerr:.1e})",
                v.label,
                wbytes as f64 / (us * 1e3),
                wbytes as f64 / 1e6
            );
        }
        println!();
    }

    println!("=== weighted per-frame GEMV total across these shapes ===");
    let mut rows: Vec<_> =
        pipes.iter().map(|p| p.0.label).zip(totals.iter().copied()).collect::<Vec<_>>();
    let base = rows[0].1;
    rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (label, us) in rows {
        println!("  {label:<26} {:>7.3} ms/frame  {:>5.2}x vs f32", us / 1e3, base / us);
    }
}
