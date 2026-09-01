//! Why does a dispatch in the real backend cost ~38 us when the floor is ~9 us?
//!
//! The model runs ~583 dispatches per frame and the profile attributes ~22 ms of
//! each 31 ms frame to `poll(Wait)`. That is 38 us per dispatch, four times the
//! cost of an isolated tiny dispatch. The candidates this separates:
//!
//!   * pipeline switching -- the real stream alternates kernels every dispatch,
//!     the floor measurement repeats one
//!   * bind group creation -- the backend builds a fresh one per dispatch
//!   * hazard barriers -- every binding in the backend is declared
//!     `read_write`, so wgpu must assume a write-after-write hazard between any
//!     two dispatches and cannot let them overlap. Declaring inputs `read_only`
//!     is what would let independent ops run concurrently.
//!
//! Run with: cargo run --release --features webgpu --example webgpu_barrier_lab

use std::time::Instant;

const SRC_RW: &str = r#"
struct Params { n: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = src[i] * 1.000001 + 0.5;
}
"#;

const SRC_RO: &str = r#"
struct Params { n: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = src[i] * 1.000001 + 0.5;
}
"#;

/// A second kernel body so alternating pipelines is a real state change.
const SRC_RO_B: &str = r#"
struct Params { n: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = src[i] * 0.999999 - 0.25;
}
"#;

const SRC_RW_B: &str = r#"
struct Params { n: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = src[i] * 0.999999 - 0.25;
}
"#;

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
            required_features: wgpu::Features::PUSH_CONSTANTS,
            required_limits: wgpu::Limits { max_push_constant_size: 128, ..adapter.limits() },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("request_device");

    // Two bind group layouts differing only in whether binding 1 is read_only.
    let mk_bgl = |read_only: bool| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[0u32, 1]
                .iter()
                .map(|&i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: i == 1 && read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        })
    };
    let bgl_rw = mk_bgl(false);
    let bgl_ro = mk_bgl(true);

    let mk_pl = |bgl: &wgpu::BindGroupLayout| {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..128,
            }],
        })
    };
    let pl_rw = mk_pl(&bgl_rw);
    let pl_ro = mk_pl(&bgl_ro);

    let mk_pipe = |src: &str, pl: &wgpu::PipelineLayout| {
        let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(pl),
            module: &m,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let p_rw_a = mk_pipe(SRC_RW, &pl_rw);
    let p_rw_b = mk_pipe(SRC_RW_B, &pl_rw);
    let p_ro_a = mk_pipe(SRC_RO, &pl_ro);
    let p_ro_b = mk_pipe(SRC_RO_B, &pl_ro);

    // 16 independent buffer pairs, so a run can either hammer one pair (every
    // dispatch depends on the previous) or rotate through all of them (no
    // dependency at all).
    const PAIRS: usize = 16;
    const N: usize = 4096; // 16 workgroups -- small, like the model's ops
    let bufs: Vec<wgpu::Buffer> = (0..PAIRS * 2)
        .map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (N * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();

    let mk_bg = |bgl: &wgpu::BindGroupLayout, i: usize| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs[i * 2].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs[i * 2 + 1].as_entire_binding() },
            ],
        })
    };
    let bgs_rw: Vec<_> = (0..PAIRS).map(|i| mk_bg(&bgl_rw, i)).collect();
    let bgs_ro: Vec<_> = (0..PAIRS).map(|i| mk_bg(&bgl_ro, i)).collect();

    let push = (N as u32).to_le_bytes();

    // `rotate_bg`: cycle bind groups (independent dispatches) or reuse one
    // (dependent chain). `alternate_pipe`: switch pipeline every dispatch.
    // `fresh_bg`: build a new bind group per dispatch, as the backend does.
    let bench = |label: &str,
                 pipes: (&wgpu::ComputePipeline, &wgpu::ComputePipeline),
                 bgs: &Vec<wgpu::BindGroup>,
                 bgl: &wgpu::BindGroupLayout,
                 rotate_bg: bool,
                 alternate_pipe: bool,
                 fresh_bg: bool| {
        let run = |iters: usize| {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                for i in 0..iters {
                    let p = if alternate_pipe && i % 2 == 1 { pipes.1 } else { pipes.0 };
                    cp.set_pipeline(p);
                    let idx = if rotate_bg { i % PAIRS } else { 0 };
                    if fresh_bg {
                        let bg = mk_bg(bgl, idx);
                        cp.set_bind_group(0, &bg, &[]);
                        cp.set_push_constants(0, &push);
                        cp.dispatch_workgroups((N / 256) as u32, 1, 1);
                    } else {
                        cp.set_bind_group(0, &bgs[idx], &[]);
                        cp.set_push_constants(0, &push);
                        cp.dispatch_workgroups((N / 256) as u32, 1, 1);
                    }
                }
            }
            queue.submit(Some(enc.finish()));
            device.poll(wgpu::PollType::Wait).unwrap();
        };
        eprint!("    ...{label}\r");
        run(32);
        let iters = 400;
        let t = Instant::now();
        run(iters);
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("  {label:<52} {us:>7.2} us/dispatch");
    };

    println!("4096-element kernel (16 workgroups), 3000 dispatches in one pass/submit:");
    println!("\n-- binding 1 declared read_write (what the backend does today)");
    bench(
        "same pipe, same buffers (dependent chain)",
        (&p_rw_a, &p_rw_b),
        &bgs_rw,
        &bgl_rw,
        false,
        false,
        false,
    );
    bench(
        "same pipe, rotating buffers (independent)",
        (&p_rw_a, &p_rw_b),
        &bgs_rw,
        &bgl_rw,
        true,
        false,
        false,
    );
    bench(
        "alternating pipes, rotating buffers",
        (&p_rw_a, &p_rw_b),
        &bgs_rw,
        &bgl_rw,
        true,
        true,
        false,
    );
    bench(
        "alternating pipes, rotating, fresh bind group",
        (&p_rw_a, &p_rw_b),
        &bgs_rw,
        &bgl_rw,
        true,
        true,
        true,
    );

    println!("\n-- binding 1 declared read_only (inputs immutable)");
    bench(
        "same pipe, same buffers (dependent chain)",
        (&p_ro_a, &p_ro_b),
        &bgs_ro,
        &bgl_ro,
        false,
        false,
        false,
    );
    bench(
        "same pipe, rotating buffers (independent)",
        (&p_ro_a, &p_ro_b),
        &bgs_ro,
        &bgl_ro,
        true,
        false,
        false,
    );
    bench(
        "alternating pipes, rotating buffers",
        (&p_ro_a, &p_ro_b),
        &bgs_ro,
        &bgl_ro,
        true,
        true,
        false,
    );
    bench(
        "alternating pipes, rotating, fresh bind group",
        (&p_ro_a, &p_ro_b),
        &bgs_ro,
        &bgl_ro,
        true,
        true,
        true,
    );

    // How much does splitting one pass into many cost? The backend used to end a
    // pass per op; the uncommitted branch holds one open.
    println!("\n-- pass granularity (read_only, alternating pipes, rotating buffers)");
    for per_pass in [1usize, 4, 16, 64, 1024] {
        let iters = 400;
        let run = |n: usize| {
            let mut enc = device.create_command_encoder(&Default::default());
            let mut done = 0;
            while done < n {
                let this = per_pass.min(n - done);
                {
                    let mut cp = enc.begin_compute_pass(&Default::default());
                    for i in done..done + this {
                        cp.set_pipeline(if i % 2 == 1 { &p_ro_b } else { &p_ro_a });
                        cp.set_bind_group(0, &bgs_ro[i % PAIRS], &[]);
                        cp.set_push_constants(0, &push);
                        cp.dispatch_workgroups((N / 256) as u32, 1, 1);
                    }
                }
                done += this;
            }
            queue.submit(Some(enc.finish()));
            device.poll(wgpu::PollType::Wait).unwrap();
        };
        run(32);
        let t = Instant::now();
        run(iters);
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "  {:<52} {us:>7.2} us/dispatch",
            format!("{per_pass} dispatch(es) per compute pass")
        );
    }
}
