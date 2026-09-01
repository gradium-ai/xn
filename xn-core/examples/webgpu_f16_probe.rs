//! Validates the WGSL mechanics an f16 compute path depends on, before the
//! kernels are rewritten around them.
//!
//! The plan is one source per kernel, parameterised by a Rust-emitted preamble
//! that aliases the *storage* scalar (`S`) while all arithmetic stays f32 --
//! the same shape as the Vulkan backend's `dtype.glsl`, since WGSL has no
//! preprocessor. That needs: `enable f16`, `alias`, a scalar `array<f16>`
//! storage binding (2-byte stride, not just `vec4<f16>`), and f32<->f16
//! conversions that are no-ops when S is already f32.
//!
//! Run with: cargo run --release --features webgpu --example webgpu_f16_probe
// Reading a byte buffer back as fixed-width scalars: the constant chunk size *is*
// the element width, which reads better here than `as_chunks`.
#![allow(clippy::chunks_exact_to_as_chunks)]

use half::f16;

/// The preamble the backend would emit, plus a kernel written once against it.
fn src(f16_mode: bool) -> String {
    let preamble = if f16_mode {
        "enable f16;\nalias S = f16;\nalias S4 = vec4<f16>;\nconst S_NEG_BIG: S = -65504.0h;\n"
    } else {
        "alias S = f32;\nalias S4 = vec4<f32>;\nconst S_NEG_BIG: S = -3.4028235e38;\n"
    };
    format!(
        r#"{preamble}
struct Params {{ n: u32 }};
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;
@group(0) @binding(1) var<storage, read> src: array<S>;
@group(0) @binding(2) var<storage, read> src4: array<S4>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if i >= pc.n {{ return; }}
    // Scalar load, f32 arithmetic, converted store. `f32(x)` and `S(x)` are
    // both no-ops when S is f32, so one source serves both dtypes.
    var acc = f32(src[i]) * 2.0 + 1.0;
    // vec4 view over the same buffer, to confirm both strides bind at once.
    if i == 0u {{
        let v = vec4<f32>(src4[0]);
        acc = acc + v.x + v.y + v.z + v.w;
    }}
    // Confirm the sentinel constant participates in arithmetic.
    if acc < f32(S_NEG_BIG) {{ acc = 0.0; }}
    dst[i] = S(acc);
}}
"#
    )
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
    println!("device: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);
    let has_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
    println!("adapter SHADER_F16: {has_f16}\n");

    let mut feats = wgpu::Features::PUSH_CONSTANTS;
    if has_f16 {
        feats |= wgpu::Features::SHADER_F16;
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: feats,
            required_limits: wgpu::Limits { max_push_constant_size: 128, ..adapter.limits() },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("request_device");

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &(0..3)
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

    for f16_mode in [false, true] {
        let tag = if f16_mode { "f16" } else { "f32" };
        if f16_mode && !has_f16 {
            println!("{tag}: adapter lacks SHADER_F16, skipping");
            continue;
        }
        let s = src(f16_mode);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(s.as_str().into()),
        });
        let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &m,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            println!("{tag}: COMPILE FAILED\n{e}\n");
            continue;
        }

        // n deliberately odd, so the f16 case ends on a 2-byte boundary and the
        // 4-byte copy/write padding is exercised.
        let n = 65usize;
        let esz = if f16_mode { 2 } else { 4 };
        let host: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 4.0).collect();
        let mut bytes: Vec<u8> = if f16_mode {
            host.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect()
        } else {
            host.iter().flat_map(|&v| v.to_le_bytes()).collect()
        };
        // An odd f16 count is not a multiple of 4 bytes, and every WGSL buffer
        // write/copy must be. The backend rounds up the same way (`round4`).
        bytes.resize(bytes.len().div_ceil(4) * 4, 0);
        let mk = |sz: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (sz as u64).div_ceil(4) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let dst = mk(n * esz);
        let srcb = mk(n * esz);
        queue.write_buffer(&srcb, 0, &bytes);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: srcb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: srcb.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&p);
            cp.set_bind_group(0, &bg, &[]);
            cp.set_push_constants(0, &(n as u32).to_le_bytes());
            cp.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
        }
        let readback_bytes = ((n * esz) as u64).div_ceil(4) * 4;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: readback_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&dst, 0, &staging, 0, readback_bytes);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::Wait).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let got: Vec<f32> = if f16_mode {
            mapped
                .chunks_exact(2)
                .take(n)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        } else {
            mapped
                .chunks_exact(4)
                .take(n)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        drop(mapped);
        staging.unmap();

        // Reference, including the i == 0 vec4 term.
        let mut want: Vec<f32> = host.iter().map(|&v| v * 2.0 + 1.0).collect();
        let quant = |v: f32| if f16_mode { f16::from_f32(v).to_f32() } else { v };
        want[0] += (0..4).map(|i| quant(host[i])).sum::<f32>();
        let tol = if f16_mode { 2e-2 } else { 1e-6 };
        let maxerr = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0f32, f32::max);
        println!(
            "{tag}: compiled, {} elems, element stride {esz} B, max rel err {maxerr:.2e}  {}",
            n,
            if maxerr < tol { "OK" } else { "MISMATCH" }
        );
    }
}
