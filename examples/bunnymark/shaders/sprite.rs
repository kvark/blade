use synaga_shader::*;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Globals {
    pub mvp_transform: Mat4,
    pub sprite_size: Vec2,
    pub _pad: Vec2,
}

/// One bunny, as the host moves it; only the host reads `velocity`.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Locals {
    pub position: Vec2,
    pub velocity: Vec2,
    pub color: u32,
    pub _pad: u32,
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
    ((Vec4::splat(raw) >> vec4(0, 8, 16, 24)) & Vec4::splat(0xFF)).cast::<f32>() / 255.0
}

#[entry_point(fragment)]
fn fs_main(vertex: VertexOutput) -> Vec4 {
    vertex.color * sprite_texture.sample_level(&sprite_sampler, vertex.tex_coords, 0.0)
}

#[entry_point(vertex)]
fn vs_main(vertex: Vertex) -> VertexOutput {
    let tc = vertex.pos;
    let offset = tc * globals.sprite_size;
    let pos = globals.mvp_transform * (locals.position + offset).extend(0.0).extend(1.0);
    VertexOutput {
        position: pos,
        tex_coords: tc,
        color: unpack_color(locals.color),
    }
}
