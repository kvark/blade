struct Globals {
    mvp_transform: mat4x4<f32>,
    sprite_size: vec2<f32>,
};
var<uniform> globals: Globals;

struct Locals {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: u32,
};
var<uniform> locals: Locals;

var sprite_texture: texture_2d<f32>;
var sprite_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let tc = vec2<f32>(f32(vi & 1u), 0.5 * f32(vi & 2u));
    let offset = tc * globals.sprite_size;
    let pos = globals.mvp_transform * vec4<f32>(locals.position + offset, 0.0, 1.0);
    let color = unpack4x8unorm(locals.color);
    return VertexOutput(pos, tc, color);
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return vertex.color * textureSampleLevel(sprite_texture, sprite_sampler, vertex.tex_coords, 0.0);
}
