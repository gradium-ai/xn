fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        let i = a.get_info();
        let f = a.features();
        let l = a.limits();
        println!("=== {} ({:?}) {:?}", i.name, i.backend, i.device_type);
        for (name, feat) in [
            ("SHADER_F16", wgpu::Features::SHADER_F16),
            ("SUBGROUP", wgpu::Features::SUBGROUP),
            ("SUBGROUP_BARRIER", wgpu::Features::SUBGROUP_BARRIER),
            ("PUSH_CONSTANTS", wgpu::Features::PUSH_CONSTANTS),
            ("TIMESTAMP_QUERY", wgpu::Features::TIMESTAMP_QUERY),
            ("TIMESTAMP_QUERY_INSIDE_PASSES", wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            ("MAPPABLE_PRIMARY_BUFFERS", wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            ("BUFFER_BINDING_ARRAY", wgpu::Features::BUFFER_BINDING_ARRAY),
        ] {
            println!("  {:34} {}", name, f.contains(feat));
        }
        println!(
            "  max_compute_invocations_per_workgroup   {}",
            l.max_compute_invocations_per_workgroup
        );
        println!(
            "  max_compute_workgroup_size_x/y/z       {} {} {}",
            l.max_compute_workgroup_size_x,
            l.max_compute_workgroup_size_y,
            l.max_compute_workgroup_size_z
        );
        println!(
            "  max_compute_workgroup_storage_size     {} B",
            l.max_compute_workgroup_storage_size
        );
        println!(
            "  max_compute_workgroups_per_dimension   {}",
            l.max_compute_workgroups_per_dimension
        );
        println!(
            "  max_storage_buffers_per_shader_stage   {}",
            l.max_storage_buffers_per_shader_stage
        );
        println!(
            "  max_storage_buffer_binding_size        {} MB",
            l.max_storage_buffer_binding_size / (1 << 20)
        );
        println!("  max_push_constant_size                 {} B", l.max_push_constant_size);
        println!(
            "  min_subgroup_size / max                {} / {}",
            l.min_subgroup_size, l.max_subgroup_size
        );
        println!("  max_buffer_size                        {} MB", l.max_buffer_size / (1 << 20));
    }
}
