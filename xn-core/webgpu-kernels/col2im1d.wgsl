// Col2Im for 1D transposed convolution (f32), gather form. Only used for the
// groups == 1, padding == 0, output_padding == 0, dilation == 1 case.
//   src (col): [batch, l_in, out_channels, kernel_size]
//   dst: [batch, out_channels, out_length]
// One thread per output element; dst is written in its natural flat order so
// gid doubles as the destination index directly.
struct Params {
    batch: u32, l_in: u32, out_channels: u32, out_length: u32,
    kernel_size: u32, stride: u32,
};
// WebGPU has no push constants; parameters arrive in a uniform windowed
// to this dispatch's slot by a dynamic offset.
@group(0) @binding(8) var<uniform> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let total = pc.batch * pc.out_channels * pc.out_length;
    if g >= total { return; }

    let l_out_idx = g % pc.out_length;
    let tmp = g / pc.out_length;
    let c_idx = tmp % pc.out_channels;
    let b = tmp / pc.out_channels;

    let src_s1 = pc.out_channels * pc.kernel_size;
    let src_batch_base = b * pc.l_in * src_s1;

    // out_l = in_l * stride + k  =>  in_l = (out_l - k) / stride for k in
    // [0, stride) so that the numerator is non-negative and divisible.
    var l_in_idx = i32(l_out_idx / pc.stride);
    var k = i32(l_out_idx) - l_in_idx * i32(pc.stride);

    var sum = 0.0;
    loop {
        if !(k < i32(pc.kernel_size) && l_in_idx >= 0) { break; }
        if l_in_idx < i32(pc.l_in) {
            let src_idx = src_batch_base + u32(l_in_idx) * src_s1 + c_idx * pc.kernel_size + u32(k);
            sum = sum + src[src_idx];
        }
        k = k + i32(pc.stride);
        l_in_idx = l_in_idx - 1;
    }
    dst[g] = sum;
}
