// Direct 1D transposed convolution (f32), gather form. One thread per output.
//   src:    [batch, in_channels, length]
//   kernel: [in_channels, out_channels/groups, kernel_size]
//   dst:    [batch, out_channels, out_length]
struct Params {
    batch: u32, in_channels: u32, out_channels: u32, in_len: u32,
    out_length: u32, kernel_size: u32, stride: u32, padding: u32, groups: u32,
};
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;
@group(0) @binding(1) var<storage, read_write> src: array<S>;
@group(0) @binding(2) var<storage, read_write> kern: array<S>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let total = pc.batch * pc.out_channels * pc.out_length;
    if g >= total { return; }

    let out_pos = g % pc.out_length;
    let tmp = g / pc.out_length;
    let oc = tmp % pc.out_channels;
    let b = tmp / pc.out_channels;

    let in_c_per_group = pc.in_channels / pc.groups;
    let out_c_per_group = pc.out_channels / pc.groups;
    let grp = oc / out_c_per_group;
    let oc_in_group = oc % out_c_per_group;
    let in_c_start = grp * in_c_per_group;

    let out_pos_raw = out_pos + pc.padding;
    var acc = 0.0;
    for (var ko = 0u; ko < pc.kernel_size; ko = ko + 1u) {
        if out_pos_raw < ko { continue; }
        let num = out_pos_raw - ko;
        if num % pc.stride != 0u { continue; }
        let il = num / pc.stride;
        if il >= pc.in_len { continue; }
        for (var ic = 0u; ic < in_c_per_group; ic = ic + 1u) {
            let in_c = in_c_start + ic;
            let sv = f32(src[(b * pc.in_channels + in_c) * pc.in_len + il]);
            let kv = f32(kern[(in_c * out_c_per_group + oc_in_group) * pc.kernel_size + ko]);
            acc = acc + sv * kv;
        }
    }
    dst[(b * pc.out_channels + oc) * pc.out_length + out_pos] = S(acc);
}
