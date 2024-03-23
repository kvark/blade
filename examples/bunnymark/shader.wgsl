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

struct Vertex {
    pos: vec2<f32>,
};
var<storage, read> vertices: array<Vertex>;

var sprite_texture: texture_2d<f32>;
var sprite_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
}

fn unpack_color(raw: u32) -> vec4<f32> {
    //TODO: https://github.com/gfx-rs/naga/issues/2188
    //return unpack4x8unorm(raw);
    return vec4<f32>((vec4<u32>(raw) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(0xFFu)) / 255.0;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    //let tc = vec2<f32>(f32(vi & 1u), 0.5 * f32(vi & 2u));
    let tc = vertices[vi].pos;
    let offset = tc * globals.sprite_size;
    let pos = globals.mvp_transform * vec4<f32>(locals.position + offset, 0.0, 1.0);
    let color = unpack_color(locals.color);
    return VertexOutput(pos, tc, color);
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return vertex.color * textureSampleLevel(sprite_texture, sprite_sampler, vertex.tex_coords, 0.0);
}
