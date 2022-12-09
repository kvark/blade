struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

struct Locals {
    screen_size: vec2<f32>,
    convert_to_linear: f32,
    padding: f32,
};
var<uniform> r_locals: Locals;

@vertex
fn vs_main(
    @location(0) a_pos: vec2<f32>,
    @location(1) a_tex_coord: vec2<f32>,
    @location(2) a_color: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coord = a_tex_coord;
    out.color = unpack4x8unorm(a_color);
    out.position = vec4<f32>(
        2.0 * a_pos.x / r_locals.screen_size.x - 1.0,
        1.0 - 2.0 * a_pos.y / r_locals.screen_size.y,
        0.0,
        1.0,
    );
    return out;
}

var r_tex_color: texture_2d<f32>;
var r_tex_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color * textureSample(r_tex_color, r_tex_sampler, in.tex_coord);
}
