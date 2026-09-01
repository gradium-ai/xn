// Elementwise dtype conversion between two float storage types.
//
// Unlike every other kernel this one needs two storage scalars, so its preamble
// aliases the source as `SRC` alongside the usual destination `S`. Both sides go
// via f32, which is exact for f16 -> f32 and a single rounding for f32 -> f16.
// Integer dtypes have no WGSL equivalent (there is no i64) and stay on the host.
struct Params { n: u32 };
var<push_constant> pc: Params;
@group(0) @binding(0) var<storage, read_write> dst: array<S>;
@group(0) @binding(1) var<storage, read_write> src: array<SRC>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pc.n { return; }
    dst[i] = S(f32(src[i]));
}
