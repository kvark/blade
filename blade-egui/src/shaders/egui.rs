use synaga_shader::*;

#[io]
pub struct VertexOutput {
    #[location(0)]
    tex_coord: vec2,
    #[location(1)]
    color: vec4,
    #[builtin(position)]
    position: vec4,
}

#[derive(Clone, Copy, Default)]
pub struct Uniforms {
    pub screen_size: vec2,
    pub convert_to_linear: f32,
    pub padding: f32,
}

#[derive(Clone, Copy, Default)]
pub struct Vertex {
    pub pos: vec2,
    pub uv: vec2,
    pub color: u32,
}

pub static r_uniforms: Uniform<Uniforms> = binding();

pub static r_texture: texture_2d<f32> = binding();

pub static r_sampler: sampler = binding();

#[shader]
pub fn linear_from_gamma(srgb: vec3) -> vec3 {
    let cutoff = srgb.cmplt(vec3::splat(0.04045));
    let lower = srgb / vec3::splat(12.92);
    let higher = pow(
        (srgb + vec3::splat(0.055)) / vec3::splat(1.055),
        vec3::splat(2.4),
    );
    return select(higher, lower, cutoff);
}

#[vertex]
pub fn vs_main(input: Vertex) -> VertexOutput {
    let mut out = VertexOutput::default();
    out.tex_coord = input.uv;
    out.color = unpack4x8unorm(input.color);
    out.position = vec4(
        2.0 * input.pos.x / r_uniforms.screen_size.x - 1.0,
        1.0 - 2.0 * input.pos.y / r_uniforms.screen_size.y,
        0.0,
        1.0,
    );
    return out;
}

#[fragment]
#[output(location(0))]
pub fn fs_main(input: VertexOutput) -> vec4 {
    //Note: we always assume rendering to linear color space,
    // but Egui wants to blend in gamma space, see
    // https://github.com/emilk/egui/pull/2071
    let blended = input.color * textureSample(&r_texture, &r_sampler, input.tex_coord);
    return (linear_from_gamma(blended.xyz())).extend(blended.a());
}
