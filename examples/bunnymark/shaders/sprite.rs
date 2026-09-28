use synaga_shader::*;

#[derive(Clone, Copy, Default)]
struct Globals {
    pub mvp_transform: Mat4,
    pub sprite_size: Vec2,
}

#[derive(Clone, Copy, Default)]
struct Locals {
    pub position: Vec2,
    pub velocity: Vec2,
    pub color: u32,
}

#[derive(Clone, Copy, Default)]
struct Vertex {
    pub pos: Vec2,
}

#[derive(Clone, Copy, Debug, Default, Io)]
struct VertexOutput {
    #[builtin(position)]
    position: Vec4,
    #[location(0)]
    tex_coords: Vec2,
    #[location(1)]
    color: Vec4,
}

static globals: Uniform<Globals> = binding();
static locals: Uniform<Locals> = binding();
static sprite_texture: Texture2D<f32> = binding();
static sprite_sampler: Sampler = binding();

fn unpack_color(raw: u32) -> Vec4 {
    //TODO: https://github.com/gfx-rs/naga/issues/2188
    //return unpack4x8unorm(raw);
    return Vec4::from(
        (Vec4::<u32>::splat(raw) >> vec4::<u32>(0u32, 8u32, 16u32, 24u32))
            & Vec4::<u32>::splat(0xFFu32),
    ) / 255.0;
}

#[entry_point(fragment)]
#[output(location(0))]
fn fs_main(vertex: VertexOutput) -> Vec4 {
    return vertex.color * sprite_texture.sample_level(&sprite_sampler, vertex.tex_coords, 0.0);
}

#[entry_point(vertex)]
fn vs_main(vertex: Vertex) -> VertexOutput {
    let tc = vertex.pos;
    let offset = tc * globals.sprite_size;
    let pos = globals.mvp_transform * ((locals.position + offset).extend(0.0)).extend(1.0);
    let color = unpack_color(locals.color);
    return VertexOutput {
        position: pos,
        tex_coords: tc,
        color: color,
    };
}
