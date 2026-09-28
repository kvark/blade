use synaga_shader::*;

#[derive(Clone, Copy, Debug, Default, Io)]
struct VertexOutput {
    #[location(0)]
    tex_coord: Vec2,
    #[location(1)]
    color: Vec4,
    #[builtin(position)]
    position: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Uniforms {
    pub screen_size: Vec2,
    pub _pad: Vec2,
}

#[derive(Clone, Copy, Default)]
struct Vertex {
    pub pos: Vec2,
    pub uv: Vec2,
    pub color: u32,
}

static r_uniforms: Uniform<Uniforms> = binding();
static r_texture: Texture2D<f32> = binding();
static r_sampler: Sampler = binding();

fn linear_from_gamma(srgb: Vec3) -> Vec3 {
    let cutoff = srgb.cmplt(Vec3::splat(0.04045));
    let lower = srgb / Vec3::splat(12.92);
    let higher = ((srgb + Vec3::splat(0.055)) / Vec3::splat(1.055)).powf(2.4);
    select(higher, lower, cutoff)
}

#[entry_point(vertex)]
fn vs_main(input: Vertex) -> VertexOutput {
    VertexOutput {
        tex_coord: input.uv,
        color: unpack4x8unorm(input.color),
        position: vec4(
            2.0 * input.pos.x / r_uniforms.screen_size.x - 1.0,
            1.0 - 2.0 * input.pos.y / r_uniforms.screen_size.y,
            0.0,
            1.0,
        ),
    }
}

#[entry_point(fragment)]
fn fs_main(input: VertexOutput) -> Vec4 {
    //Note: we always assume rendering to linear color space,
    // but Egui wants to blend in gamma space, see
    // https://github.com/emilk/egui/pull/2071
    let blended = input.color * r_texture.sample(&r_sampler, input.tex_coord);
    linear_from_gamma(blended.xyz()).extend(blended.a())
}
