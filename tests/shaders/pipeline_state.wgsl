// Fixed-function and binding state that a render pipeline carries, checked
// with the least shading a backend can get away with, so that a failure points
// at the state rather than at the pixels.

struct QuadParams {
    color: vec4<f32>,
    depth: f32,
}
var<uniform> quad_params: QuadParams;

// A full-screen triangle at a depth the host picks, in a color the host picks.
@vertex
fn quad_vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = 4.0 * f32(vi & 1u) - 1.0;
    let y = 4.0 * f32(vi >> 1u) - 1.0;
    return vec4<f32>(x, y, quad_params.depth, 1.0);
}

@fragment
fn quad_fs() -> @location(0) vec4<f32> {
    return quad_params.color;
}

var left_texture: texture_2d<f32>;
var right_texture: texture_2d<f32>;
var samp: sampler;

struct SplitOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn split_vs(@builtin(vertex_index) vi: u32) -> SplitOutput {
    let uv = vec2<f32>(2.0 * f32(vi & 1u), 2.0 * f32(vi >> 1u));
    var so: SplitOutput;
    so.position = vec4<f32>(2.0 * uv - 1.0, 0.0, 1.0);
    so.uv = uv;
    return so;
}

// One texture on each half of the screen. Both are sampled unconditionally:
// two textures that ended up sharing a slot still produce a plausible looking
// image, just the same one twice, so the halves have to be compared.
@fragment
fn split_fs(so: SplitOutput) -> @location(0) vec4<f32> {
    let left = textureSampleLevel(left_texture, samp, vec2<f32>(0.5), 0.0);
    let right = textureSampleLevel(right_texture, samp, vec2<f32>(0.5), 0.0);
    return select(left, right, so.uv.x >= 0.5);
}
