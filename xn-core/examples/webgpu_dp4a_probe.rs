//! Does this wgpu/Naga build support the WGSL packed integer dot product?
//!
//! llama.cpp's WebGPU q8_0 matvec is built on `dot4I8Packed` -- one instruction
//! for four int8 x int8 products -- with the activations quantized to int8 as
//! well, so the inner loop is integer throughout. Our kernel instead unpacks to
//! f32 and does vec4 FMAs against f16 activations. If Naga can emit
//! `dot4I8Packed`, that whole design is available to us; if not, it is not, and
//! the gap is structural rather than something to tune.
//!
//! Run with: cargo run --release --features webgpu --example webgpu_dp4a_probe
// Reading a byte buffer back as fixed-width scalars: the constant chunk size *is*
// the element width, which reads better here than `as_chunks`.
#![allow(clippy::chunks_exact_to_as_chunks)]

/// Variants to try: with and without the `requires` directive llama.cpp uses,
/// plus the unsigned form and a scalar-shift baseline for comparison.
fn variants() -> Vec<(&'static str, String)> {
    let body = r#"
struct Params { n: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<i32>;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = OP(a[i], b[i]);
}
"#;
    vec![
        (
            "dot4I8Packed, with `requires`",
            format!(
                "requires packed_4x8_integer_dot_product;\n{}",
                body.replace("OP", "dot4I8Packed")
            ),
        ),
        ("dot4I8Packed, no directive", body.replace("OP", "dot4I8Packed")),
        (
            "dot4U8Packed, no directive",
            body.replace("OP", "i32(dot4U8Packed").replace("a[i], b[i])", "a[i], b[i]))"),
        ),
    ]
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
    println!("wgpu {}\n", env!("CARGO_PKG_VERSION"));

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

    // Four signed bytes each; the reference is the plain int8 dot product.
    let n = 64usize;
    let av: Vec<u32> = (0..n)
        .map(|i| u32::from_le_bytes([i as u8, (i * 3) as u8, (200 + i) as u8, 0xFF]))
        .collect();
    let bv: Vec<u32> =
        (0..n).map(|i| u32::from_le_bytes([0x01, (250 - i) as u8, (i * 7) as u8, 0x80])).collect();
    let want: Vec<i32> = av
        .iter()
        .zip(&bv)
        .map(|(&x, &y)| {
            (0..4)
                .map(|k| {
                    let p = ((x >> (k * 8)) & 0xFF) as i8 as i32;
                    let q = ((y >> (k * 8)) & 0xFF) as i8 as i32;
                    p * q
                })
                .sum()
        })
        .collect();

    let mk = |data: &[u32]| {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytes);
        buf
    };
    let dst = mk(&vec![0u32; n]);
    let (ab, bb) = (mk(&av), mk(&bv));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: ab.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bb.as_entire_binding() },
        ],
    });

    for (label, src) in variants() {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
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
            let msg = format!("{e}");
            let line = msg
                .lines()
                .find(|l| l.contains("not") || l.contains("unknown") || l.contains("no "))
                .unwrap_or_else(|| msg.lines().next().unwrap_or(""));
            println!("{label:<32} FAILED: {}", line.trim());
            continue;
        }

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&p);
            cp.set_bind_group(0, &bg, &[]);
            cp.set_push_constants(0, &(n as u32).to_le_bytes());
            cp.dispatch_workgroups(1, 1, 1);
        }
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&dst, 0, &staging, 0, (n * 4) as u64);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::Wait).unwrap();
        rx.recv().unwrap().unwrap();
        let got: Vec<i32> = slice
            .get_mapped_range()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        staging.unmap();
        let ok = got == want;
        println!(
            "{label:<32} compiled, results {}",
            if ok {
                "CORRECT".to_string()
            } else {
                format!("WRONG (got {:?} want {:?})", &got[..3], &want[..3])
            }
        );
    }
}
