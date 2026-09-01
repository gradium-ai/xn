//! Head-to-head of q8_0 decode-matvec designs on phonon's shapes.
//!
//! Ours dequantizes the weight to f32 in registers and multiplies against f16
//! activations, one thread per output column. llama.cpp's WebGPU backend instead
//! quantizes the *activations* to int8 too and uses `dot4I8Packed` -- one
//! instruction for four int8 x int8 products -- spreading each 32-value block
//! across 4 threads, 4 output rows per 256-thread workgroup, finishing with a
//! workgroup reduction (they use a subgroup reduction where available; Naga has
//! no subgroup support, so this ports the workgroup path).
//!
//! Two changes are bundled there, so this separates them:
//!   A  ours: unpack to f32, f16 activations, thread per column
//!   B  llama.cpp: dot4I8Packed, int8 activations, cooperative over k
//!   C  dot4I8Packed and int8 activations, but *our* thread-per-column shape
//!   D  ours' arithmetic, but llama.cpp's cooperative launch shape
//!
//! A vs C isolates the instruction; A vs D isolates the launch shape.
//!
//! Run with: cargo run --release --features webgpu --example webgpu_q8_shootout
// Reading a byte buffer back as fixed-width scalars: the constant chunk size *is*
// the element width, which reads better here than `as_chunks`.
#![allow(clippy::chunks_exact_to_as_chunks)]

use half::f16;
use std::time::Instant;

// (label, n, k, dispatches per decode frame)
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("flow_lm in_proj ", 2304, 768, 12),
    ("flow_lm out_proj", 768, 768, 12),
    ("flow_lm linear1 ", 3072, 768, 12),
    ("flow_lm linear2 ", 768, 3072, 12),
    ("mimi tf linear1 ", 2048, 512, 2),
    ("mimi tf linear2 ", 512, 2048, 2),
];

/// A -- ours. Weight quants as vec4<u32>, unpacked to f32, f16 activations.
const A_OURS: &str = r#"
enable f16;
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> ws: array<f32>;
@group(0) @binding(3) var<storage, read> xf: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> xq: array<u32>;
@group(0) @binding(5) var<storage, read> xs: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }
    let nb = pc.k >> 5u;
    let qrow = j * (pc.k >> 4u);
    let srow = j * nb;
    var tot = 0.0;
    for (var b = 0u; b < nb; b = b + 1u) {
        let lo = wq[qrow + b * 2u];
        let hi = wq[qrow + b * 2u + 1u];
        let l0 = b * 8u;
        var acc = vec4<f32>(unpack4xI8(lo.x)) * vec4<f32>(xf[l0]);
        acc = acc + vec4<f32>(unpack4xI8(lo.y)) * vec4<f32>(xf[l0 + 1u]);
        acc = acc + vec4<f32>(unpack4xI8(lo.z)) * vec4<f32>(xf[l0 + 2u]);
        acc = acc + vec4<f32>(unpack4xI8(lo.w)) * vec4<f32>(xf[l0 + 3u]);
        acc = acc + vec4<f32>(unpack4xI8(hi.x)) * vec4<f32>(xf[l0 + 4u]);
        acc = acc + vec4<f32>(unpack4xI8(hi.y)) * vec4<f32>(xf[l0 + 5u]);
        acc = acc + vec4<f32>(unpack4xI8(hi.z)) * vec4<f32>(xf[l0 + 6u]);
        acc = acc + vec4<f32>(unpack4xI8(hi.w)) * vec4<f32>(xf[l0 + 7u]);
        tot = tot + ws[srow + b] * (acc.x + acc.y + acc.z + acc.w);
    }
    dst[j] = tot;
}
"#;

/// C -- dot4I8Packed and int8 activations, kept in our thread-per-column shape.
const C_DP4A_TPC: &str = r#"
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> ws: array<f32>;
@group(0) @binding(3) var<storage, read> xf: array<u32>;
@group(0) @binding(4) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> xs: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }
    let nb = pc.k >> 5u;
    let qrow = j * (pc.k >> 4u);
    let srow = j * nb;
    var tot = 0.0;
    for (var b = 0u; b < nb; b = b + 1u) {
        let wlo = wq[qrow + b * 2u];
        let whi = wq[qrow + b * 2u + 1u];
        let alo = xq[b * 2u];
        let ahi = xq[b * 2u + 1u];
        var s = dot4I8Packed(wlo.x, alo.x);
        s = s + dot4I8Packed(wlo.y, alo.y);
        s = s + dot4I8Packed(wlo.z, alo.z);
        s = s + dot4I8Packed(wlo.w, alo.w);
        s = s + dot4I8Packed(whi.x, ahi.x);
        s = s + dot4I8Packed(whi.y, ahi.y);
        s = s + dot4I8Packed(whi.z, ahi.z);
        s = s + dot4I8Packed(whi.w, ahi.w);
        tot = tot + f32(s) * ws[srow + b] * xs[b];
    }
    dst[j] = tot;
}
"#;

/// B -- llama.cpp's shape: 4 threads per 32-value block, OUTPUTS_PER_WG rows per
/// 256-thread workgroup, workgroup-tree reduction. Grid is ceil(n / 4).
const B_LLAMACPP: &str = r#"
const WG_SIZE: u32 = 256u;
const OUTPUTS_PER_WG: u32 = 4u;
const THREADS_PER_BLOCK: u32 = 4u;
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> wq: array<u32>;
@group(0) @binding(2) var<storage, read> ws: array<f32>;
@group(0) @binding(3) var<storage, read> xf: array<u32>;
@group(0) @binding(4) var<storage, read> xq: array<u32>;
@group(0) @binding(5) var<storage, read> xs: array<f32>;

var<workgroup> partial: array<f32, OUTPUTS_PER_WG * WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let row_base = wid.x * OUTPUTS_PER_WG;
    let nb = pc.k >> 5u;
    let inner = tid % THREADS_PER_BLOCK;   // which 8 of the block's 32 values
    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);

    for (var b = tid / THREADS_PER_BLOCK; b < nb; b = b + WG_SIZE / THREADS_PER_BLOCK) {
        // This thread's two activation words and the block scale.
        let a0 = xq[b * 8u + inner * 2u];
        let a1 = xq[b * 8u + inner * 2u + 1u];
        let a_s = xs[b];
        for (var r = 0u; r < OUTPUTS_PER_WG; r = r + 1u) {
            let row = row_base + r;
            if row < pc.n {
                let base = row * (pc.k >> 2u) + b * 8u + inner * 2u;
                let s = dot4I8Packed(wq[base], a0) + dot4I8Packed(wq[base + 1u], a1);
                acc[r] = acc[r] + f32(s) * ws[row * nb + b] * a_s;
            }
        }
    }

    for (var r = 0u; r < OUTPUTS_PER_WG; r = r + 1u) {
        partial[r * WG_SIZE + tid] = acc[r];
    }
    workgroupBarrier();
    var stride = WG_SIZE / 2u;
    while stride > 0u {
        if tid < stride {
            for (var r = 0u; r < OUTPUTS_PER_WG; r = r + 1u) {
                partial[r * WG_SIZE + tid] = partial[r * WG_SIZE + tid] + partial[r * WG_SIZE + tid + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if tid < OUTPUTS_PER_WG {
        let row = row_base + tid;
        if row < pc.n { dst[row] = partial[tid * WG_SIZE]; }
    }
}
"#;

/// D -- our f32 arithmetic in llama.cpp's cooperative launch shape, to separate
/// the launch shape from the instruction.
const D_F32_COOP: &str = r#"
enable f16;
const WG_SIZE: u32 = 256u;
const OUTPUTS_PER_WG: u32 = 4u;
const THREADS_PER_BLOCK: u32 = 4u;
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> wq: array<u32>;
@group(0) @binding(2) var<storage, read> ws: array<f32>;
@group(0) @binding(3) var<storage, read> xf: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> xq: array<u32>;
@group(0) @binding(5) var<storage, read> xs: array<f32>;

var<workgroup> partial: array<f32, OUTPUTS_PER_WG * WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let row_base = wid.x * OUTPUTS_PER_WG;
    let nb = pc.k >> 5u;
    let inner = tid % THREADS_PER_BLOCK;
    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);

    for (var b = tid / THREADS_PER_BLOCK; b < nb; b = b + WG_SIZE / THREADS_PER_BLOCK) {
        let xbase = b * 8u + inner * 2u;
        let x0 = vec4<f32>(xf[xbase]);
        let x1 = vec4<f32>(xf[xbase + 1u]);
        for (var r = 0u; r < OUTPUTS_PER_WG; r = r + 1u) {
            let row = row_base + r;
            if row < pc.n {
                let base = row * (pc.k >> 2u) + xbase;
                var a = vec4<f32>(unpack4xI8(wq[base])) * x0;
                a = a + vec4<f32>(unpack4xI8(wq[base + 1u])) * x1;
                acc[r] = acc[r] + ws[row * nb + b] * (a.x + a.y + a.z + a.w);
            }
        }
    }

    for (var r = 0u; r < OUTPUTS_PER_WG; r = r + 1u) {
        partial[r * WG_SIZE + tid] = acc[r];
    }
    workgroupBarrier();
    var stride = WG_SIZE / 2u;
    while stride > 0u {
        if tid < stride {
            for (var r = 0u; r < OUTPUTS_PER_WG; r = r + 1u) {
                partial[r * WG_SIZE + tid] = partial[r * WG_SIZE + tid] + partial[r * WG_SIZE + tid + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if tid < OUTPUTS_PER_WG {
        let row = row_base + tid;
        if row < pc.n { dst[row] = partial[tid * WG_SIZE]; }
    }
}
"#;

struct Variant {
    label: &'static str,
    src: &'static str,
    /// Workgroups for `n` outputs.
    grid: fn(usize) -> u32,
    /// Whether the reference should use int8 activations (vs f16 ones).
    int8_acts: bool,
}

const VARIANTS: &[Variant] = &[
    Variant {
        label: "A ours (unpack+f32, tpc)",
        src: A_OURS,
        grid: |n| n.div_ceil(64) as u32,
        int8_acts: false,
    },
    Variant {
        label: "B llama.cpp (dp4a, coop)",
        src: B_LLAMACPP,
        grid: |n| n.div_ceil(4) as u32,
        int8_acts: true,
    },
    Variant {
        label: "C dp4a, our tpc shape",
        src: C_DP4A_TPC,
        grid: |n| n.div_ceil(64) as u32,
        int8_acts: true,
    },
    Variant {
        label: "D unpack+f32, coop shape",
        src: D_F32_COOP,
        grid: |n| n.div_ceil(4) as u32,
        int8_acts: false,
    },
];

/// q8_0 quantize one row: f32 scale per 32 values, quants packed 4 per u32.
/// Returns (scales, packed quants, dequantized values).
fn quant(row: &[f32]) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
    let (mut s, mut q, mut d) = (Vec::new(), Vec::new(), Vec::new());
    for blk in row.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let sc = amax / 127.0;
        let inv = if sc != 0.0 { 1.0 / sc } else { 0.0 };
        s.push(sc);
        let qs: Vec<i8> =
            blk.iter().map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8).collect();
        for &v in &qs {
            d.push(v as f32 * sc);
        }
        for w in qs.chunks(4) {
            q.push(u32::from_le_bytes([w[0] as u8, w[1] as u8, w[2] as u8, w[3] as u8]));
        }
    }
    (s, q, d)
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
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
    println!("device: {} ({:?})\n", adapter.get_info().name, adapter.get_info().backend);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::PUSH_CONSTANTS | wgpu::Features::SHADER_F16,
            required_limits: wgpu::Limits { max_push_constant_size: 128, ..adapter.limits() },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("request_device");

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &(0..6)
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
            .collect::<Vec<_>>(),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..128,
        }],
    });

    let mut pipes = Vec::new();
    for v in VARIANTS {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(v.src.into()),
        });
        let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &m,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        match pollster::block_on(device.pop_error_scope()) {
            Some(e) => println!(
                "  !! {} failed: {}",
                v.label,
                format!("{e}").lines().take(4).collect::<Vec<_>>().join(" | ")
            ),
            None => pipes.push((v, p)),
        }
    }
    println!();

    let mk = |bytes: &[u8]| {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes.len().max(4) as u64).div_ceil(4) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytes);
        buf
    };

    let mut totals = vec![0f64; pipes.len()];

    for &(label, n, k, per_frame) in SHAPES {
        let wf: Vec<f32> = (0..n * k).map(|i| ((i % 251) as f32 - 125.0) / 256.0).collect();
        let xfv: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();

        // Weight: q8_0 per row.
        let (mut wsv, mut wqv, mut wdq) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            let (s, q, d) = quant(&wf[j * k..(j + 1) * k]);
            wsv.extend(s);
            wqv.extend(q);
            wdq.extend(d);
        }
        // Activation, both representations.
        let (xsv, xqv, xdq) = quant(&xfv);
        let xf16: Vec<f16> = xfv.iter().map(|&v| f16::from_f32(v)).collect();
        let xf16_deq: Vec<f32> = xf16.iter().map(|v| v.to_f32()).collect();

        // One reference per activation representation: the int8 path is exactly
        // dequantized-weight . dequantized-activation.
        let refr = |acts: &[f32]| -> Vec<f32> {
            (0..n).map(|j| (0..k).map(|l| wdq[j * k + l] * acts[l]).sum::<f32>()).collect()
        };
        let ref_int8 = refr(&xdq);
        let ref_f16 = refr(&xf16_deq);

        let dst = mk(as_bytes(&vec![0f32; n]));
        let bufs = [
            &dst,
            &mk(as_bytes(&wqv)),
            &mk(as_bytes(&wsv)),
            &mk(as_bytes(&xf16)),
            &mk(as_bytes(&xqv)),
            &mk(as_bytes(&xsv)),
        ];
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &(0..6)
                .map(|i| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: bufs[i].as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut push = Vec::new();
        push.extend_from_slice(&(n as u32).to_le_bytes());
        push.extend_from_slice(&(k as u32).to_le_bytes());

        // Bytes of weight read: one byte per value plus a 4-byte scale per 32.
        let wbytes = (n * k) as f64 * 1.125;
        println!("{label} n={n:<5} k={k:<5}  weights {:.2} MB  x{per_frame}/frame", wbytes / 1e6);

        for (vi, (v, p)) in pipes.iter().enumerate() {
            let groups = (v.grid)(n);
            let iters = if n * k > 1 << 21 { 400 } else { 1200 };
            let run = |count: usize| {
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut cp = enc.begin_compute_pass(&Default::default());
                    cp.set_pipeline(p);
                    cp.set_bind_group(0, &bg, &[]);
                    cp.set_push_constants(0, &push);
                    for _ in 0..count {
                        cp.dispatch_workgroups(groups, 1, 1);
                    }
                }
                queue.submit(Some(enc.finish()));
                device.poll(wgpu::PollType::Wait).unwrap();
            };
            run(16);
            let t = Instant::now();
            run(iters);
            let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&dst, 0, &staging, 0, (n * 4) as u64);
            queue.submit(Some(enc.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device.poll(wgpu::PollType::Wait).unwrap();
            rx.recv().unwrap().unwrap();
            let got: Vec<f32> = slice
                .get_mapped_range()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            staging.unmap();

            let reference = if v.int8_acts { &ref_int8 } else { &ref_f16 };
            let maxerr = got
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                .fold(0f32, f32::max);
            let ok = if maxerr < 5e-3 { "ok " } else { "BAD" };
            totals[vi] += us * per_frame as f64;
            println!(
                "    {:<26} {us:>8.2} us  {:>6.0} GB/s  {ok} (relerr {maxerr:.1e}, {groups} wg)",
                v.label,
                wbytes / (us * 1e3)
            );
        }
        println!();
    }

    println!("=== weighted per-frame total over these shapes ===");
    let mut rows: Vec<_> = pipes.iter().map(|p| p.0.label).zip(totals.iter().copied()).collect();
    let base = rows[0].1;
    rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (label, us) in rows {
        println!("  {label:<26} {:>7.3} ms/frame  {:>5.2}x vs ours", us / 1e3, base / us);
    }
}
