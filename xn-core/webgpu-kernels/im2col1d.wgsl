// Im2Col for 1D convolution (f32), groups == 1 only.
//   src: [batch, in_channels, in_len]
//   dst (col): [batch, out_length, in_channels * kernel_size]
// One thread per output element; dst is written in its natural flat order so
// gid doubles as the destination index directly.
struct Params {
    batch: u32, in_channels: u32, in_len: u32, out_length: u32,
    kernel_size: u32, stride: u32, padding: u32, dilation: u32,
};
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;
@group(0) @binding(1) var<storage, read_write> src: array<S>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let ck = pc.in_channels * pc.kernel_size;
    let total = pc.batch * pc.out_length * ck;
    if g >= total { return; }

    let ck_idx = g % ck;
    let tmp = g / ck;
    let l = tmp % pc.out_length;
    let b = tmp / pc.out_length;

    let k_idx = ck_idx % pc.kernel_size;
    let c_idx = ck_idx / pc.kernel_size;

    let src_l_raw = l * pc.stride + k_idx * pc.dilation;
    var v = S(0.0);
    if src_l_raw >= pc.padding && src_l_raw < pc.padding + pc.in_len {
        let src_l = src_l_raw - pc.padding;
        v = src[(b * pc.in_channels + c_idx) * pc.in_len + src_l];
    }
    dst[g] = v;
}
