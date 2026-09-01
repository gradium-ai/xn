//! Picks a decode-GEMV kernel design for the phonon TTS model, empirically.
//!
//! The backend's current gemv launches one 256-thread workgroup per output
//! column and reduces over k with an 8-step shared-memory barrier tree. On the
//! model's shapes that lands at 20-60 GFLOP/s (see `webgpu_dispatch_cost`),
//! which is roughly a tenth of what the memory system can sustain -- the kernel
//! is launch- and reduction-bound, not bandwidth-bound.
//!
//! This runs several alternative designs over the exact (n, k) pairs the model
//! uses, in f32 / f16 / q8_0, checks each against a CPU reference and reports
//! GB/s of weight traffic (the quantity that actually bounds decode).
//!
//! Run with: cargo run --release --features webgpu --example webgpu_gemv_lab
// Reading a byte buffer back as fixed-width scalars: the constant chunk size *is*
// the element width, which reads better here than `as_chunks`.
#![allow(clippy::chunks_exact_to_as_chunks)]

use std::time::Instant;

// (label, n, k, how many times one frame runs it)
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
    ("flow_net input_proj  ", 512, 32, 1),
    // The k-threshold probes: where does cooperative overtake thread-per-col?
    ("k-probe n=512 k=1024 ", 512, 1024, 0),
    ("mimi tf linear2      ", 512, 2048, 2),
    ("k-probe n=2048 k=2048", 2048, 2048, 0),
];

const WG_MEM_LIMIT: usize = 32768;

/// Shared preamble: push constants + bindings. `dst = lhs (1 x k) * rhs^T`,
/// rhs is [n, k] row-major so each output column reads one contiguous row.
const PREAMBLE: &str = r#"
struct Params { n: u32, k: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> lhs: array<f32>;
@group(0) @binding(2) var<storage, read> rhs: array<f32>;
@group(0) @binding(3) var<storage, read> rhs4: array<vec4<f32>>;
"#;

/// v0 -- what the backend does today: 256 threads per output column, vec4 loads
/// on the weight row, shared-memory barrier tree to finish the reduction.
const V0_TREE256: &str = r#"
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = wid.x; let tid = lid.x;
    let rbase = j * pc.k;
    var acc = 0.0;
    let k4 = pc.k >> 2u;
    let base4 = rbase >> 2u;
    for (var g = tid; g < k4; g = g + 256u) {
        let rv = rhs4[base4 + g]; let l = g * 4u;
        acc = acc + rv.x * lhs[l] + rv.y * lhs[l+1u] + rv.z * lhs[l+2u] + rv.w * lhs[l+3u];
    }
    for (var l = (k4 << 2u) + tid; l < pc.k; l = l + 256u) { acc = acc + lhs[l] * rhs[rbase + l]; }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if tid < s { sh[tid] = sh[tid] + sh[tid + s]; }
        workgroupBarrier();
    }
    if tid == 0u { dst[j] = sh[0]; }
}
"#;

/// v1 -- same launch shape as v0, but the reduction is one `subgroupAdd` per
/// subgroup plus a short tree over the 8 partials. Isolates how much of v0's
/// cost is the barrier tree rather than the launch.
const V1_SG256: &str = r#"
var<workgroup> sh: array<f32, 32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sg_size: u32, @builtin(subgroup_invocation_id) sg_id: u32) {
    let j = wid.x; let tid = lid.x;
    let rbase = j * pc.k;
    var acc = 0.0;
    let k4 = pc.k >> 2u;
    let base4 = rbase >> 2u;
    for (var g = tid; g < k4; g = g + 256u) {
        let rv = rhs4[base4 + g]; let l = g * 4u;
        acc = acc + rv.x * lhs[l] + rv.y * lhs[l+1u] + rv.z * lhs[l+2u] + rv.w * lhs[l+3u];
    }
    for (var l = (k4 << 2u) + tid; l < pc.k; l = l + 256u) { acc = acc + lhs[l] * rhs[rbase + l]; }
    let s = subgroupAdd(acc);
    let n_sg = 256u / sg_size;
    if sg_id == 0u { sh[tid / sg_size] = s; }
    workgroupBarrier();
    if tid == 0u {
        var t = 0.0;
        for (var i = 0u; i < n_sg; i = i + 1u) { t = t + sh[i]; }
        dst[j] = t;
    }
}
"#;

/// v2 -- one subgroup per output column, `COLS` columns per workgroup, and the
/// lhs row staged into workgroup memory once and reused by every column. Cuts
/// the workgroup count by COLS and removes the cross-subgroup barrier entirely.
fn v2_sg_cols(cols: usize, sg: usize, kmax: usize) -> String {
    format!(
        r#"
var<workgroup> ls: array<f32, {kmax}>;
@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tid = lid.x;
    for (var i = tid; i < pc.k; i = i + {wg}u) {{ ls[i] = lhs[i]; }}
    workgroupBarrier();
    let sg_idx = tid / {sg}u;
    let lane = tid % {sg}u;
    let j = wid.x * {cols}u + sg_idx;
    if j >= pc.n {{ return; }}
    let rbase = j * pc.k;
    var acc = 0.0;
    let k4 = pc.k >> 2u;
    let base4 = rbase >> 2u;
    for (var g = lane; g < k4; g = g + {sg}u) {{
        let rv = rhs4[base4 + g]; let l = g * 4u;
        acc = acc + rv.x * ls[l] + rv.y * ls[l+1u] + rv.z * ls[l+2u] + rv.w * ls[l+3u];
    }}
    for (var l = (k4 << 2u) + lane; l < pc.k; l = l + {sg}u) {{ acc = acc + ls[l] * rhs[rbase + l]; }}
    let t = subgroupAdd(acc);
    if lane == 0u {{ dst[j] = t; }}
}}
"#,
        wg = cols * sg,
        cols = cols,
        sg = sg,
        kmax = kmax
    )
}

/// v3 -- no cross-lane reduction at all: one thread owns one output column and
/// walks its whole weight row. Loads are strided by k between neighbouring
/// threads, so this trades coalescing for zero reduction cost.
const V3_THREAD_PER_COL: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= pc.n { return; }
    let base4 = (j * pc.k) >> 2u;
    let k4 = pc.k >> 2u;
    var acc = 0.0;
    for (var g = 0u; g < k4; g = g + 1u) {
        let rv = rhs4[base4 + g]; let l = g * 4u;
        acc = acc + rv.x * lhs[l] + rv.y * lhs[l+1u] + rv.z * lhs[l+2u] + rv.w * lhs[l+3u];
    }
    dst[j] = acc;
}
"#;

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
            println!("device: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);
            let feats = wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::SHADER_F16
                | wgpu::Features::SUBGROUP;
            let limits = wgpu::Limits { max_push_constant_size: 128, ..adapter.limits() };
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: feats,
                    required_limits: limits,
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

    /// Assembles `enable` directives (which must lead the module), the shared
    /// binding preamble, and the variant body into one shader.
    fn pipeline(&self, body: &str) -> Result<wgpu::ComputePipeline, String> {
        let needs_sg = body.contains("subgroupAdd") || body.contains("subgroup_");
        let src =
            format!("{}{}{}", if needs_sg { "enable subgroups;\n" } else { "" }, PREAMBLE, body);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
        });
        let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&self.pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(self.device.pop_error_scope()) {
            return Err(format!("{e}"));
        }
        Ok(p)
    }

    fn buf_f32(&self, data: &[f32]) -> wgpu::Buffer {
        let bytes: &[u8] = bytemuck_cast(data);
        let b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes.len().max(4) as u64,
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

    /// Time `iters` back-to-back dispatches inside one pass and one submit, so
    /// the number is the marginal cost of the kernel rather than of submission.
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

fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() {
    let gpu = Gpu::new();
    let sg = 32usize; // Apple GPU subgroup width
    println!("assuming subgroup width {sg}\n");

    let kmax = SHAPES.iter().map(|s| s.2).max().unwrap();
    println!("{:<22} {:>7} {:>9}  us/op and GB/s of weights", "shape (n x k)", "n", "k");

    // (label, source, workgroup-count fn)
    #[allow(clippy::type_complexity)]
    let variants: Vec<(String, String, Box<dyn Fn(usize) -> u32>)> = vec![
        ("v0 tree256 (current)".into(), V0_TREE256.to_string(), Box::new(|n: usize| n as u32)),
        ("v1 subgroup256".into(), V1_SG256.to_string(), Box::new(|n: usize| n as u32)),
        (
            "v2 8sg x 32, lhs shared".into(),
            v2_sg_cols(8, sg, kmax),
            Box::new(|n: usize| n.div_ceil(8) as u32),
        ),
        (
            "v2 4sg x 32, lhs shared".into(),
            v2_sg_cols(4, sg, kmax),
            Box::new(|n: usize| n.div_ceil(4) as u32),
        ),
        (
            "v2 16sg x 32, lhs shared".into(),
            v2_sg_cols(16, sg, kmax),
            Box::new(|n: usize| n.div_ceil(16) as u32),
        ),
        (
            "v3 thread per col".into(),
            V3_THREAD_PER_COL.to_string(),
            Box::new(|n: usize| n.div_ceil(64) as u32),
        ),
    ];

    let mut pipes = Vec::new();
    for (label, src, grid) in &variants {
        match gpu.pipeline(src) {
            Ok(p) => pipes.push((label.clone(), p, grid)),
            Err(e) => println!("  !! {label} failed to compile:\n{e}\n"),
        }
    }
    if kmax * 4 > WG_MEM_LIMIT {
        println!("  (note: k={kmax} needs {} B of workgroup memory)", kmax * 4);
    }
    println!();

    // Totals weighted by how often a frame runs each shape, so the last column
    // answers "what would a frame's GEMV time be with this variant".
    let mut totals = vec![0f64; pipes.len()];

    for &(label, n, k, per_frame) in SHAPES {
        let lhs: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
        let rhs: Vec<f32> = (0..n * k).map(|i| ((i % 31) as f32 - 15.0) / 32.0).collect();
        let mut reference = vec![0f32; n];
        for j in 0..n {
            let mut acc = 0f32;
            for l in 0..k {
                acc += lhs[l] * rhs[j * k + l];
            }
            reference[j] = acc;
        }

        let dst = gpu.buf_f32(&vec![0f32; n]);
        let lb = gpu.buf_f32(&lhs);
        let rb = gpu.buf_f32(&rhs);
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &gpu.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: rb.as_entire_binding() },
            ],
        });
        let mut push = Vec::new();
        push.extend_from_slice(&(n as u32).to_le_bytes());
        push.extend_from_slice(&(k as u32).to_le_bytes());

        let bytes = (n * k * 4) as f64;
        println!("{label} n={n:<5} k={k:<5} weights {:.2} MB  x{per_frame}/frame", bytes / 1e6);
        for (vi, (vlabel, p, grid)) in pipes.iter().enumerate() {
            let groups = grid(n);
            let iters = if n * k > 1 << 21 { 300 } else { 1000 };
            let us = gpu.time(p, &bg, &push, groups, iters);
            let got = gpu.read_f32(&dst, n);
            let maxerr = got
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                .fold(0f32, f32::max);
            let ok = if maxerr < 2e-3 { "ok " } else { "BAD" };
            totals[vi] += us * per_frame as f64;
            println!(
                "    {vlabel:<26} {us:>8.2} us  {:>6.0} GB/s  {ok} (relerr {maxerr:.1e}, {groups} wg)",
                bytes / (us * 1e3)
            );
        }
        println!();
    }

    println!("=== weighted per-frame GEMV total (all shapes x their frame count) ===");
    let mut rows: Vec<_> = pipes.iter().map(|p| p.0.clone()).zip(totals.iter().copied()).collect();
    if rows.is_empty() {
        println!("  (nothing compiled)");
        return;
    }
    let base = rows[0].1;
    rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (label, us) in rows {
        println!("  {label:<26} {:>8.3} ms/frame  {:>5.2}x vs current", us / 1e3, base / us);
    }
}
