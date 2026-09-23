use super::brdf::PI;
use super::random::*;
use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct EnvImportantSample {
    pub pixel: vec2i,
    pub pdf: f32,
}

pub static env_weights: texture_2d<f32> = binding();

#[shader]
pub fn compute_latitude_area_bounds(texel_y: i32, dim: u32) -> vec2 {
    return cos(vec2::from(vec2i(texel_y, texel_y + 1)) / (dim) as f32 * PI);
}

#[shader]
pub fn compute_texel_solid_angle(itc: vec2i, dim: vec2u) -> f32 {
    //Note: this has to agree with `map_equirect_uv_to_dir`
    let meridian_solid_angle = 4.0 * PI / (dim.x) as f32;
    let bounds = compute_latitude_area_bounds(itc.y, dim.y);
    let meridian_part = 0.5 * (bounds.x - bounds.y);
    return meridian_solid_angle * meridian_part;
}

#[shader]
pub fn generate_environment_sample(rng: &mut RandomState, dim: vec2u) -> EnvImportantSample {
    let mut es = EnvImportantSample::default();
    es.pdf = 1.0;
    let mut mip = (textureNumLevels(&env_weights)) as i32;
    let mut itc = vec2i::splat(0);
    // descend through the mip chain to find a concrete pixel
    while (mip != 0) {
        mip -= 1;
        let weights = textureLoad(&env_weights, itc, mip);
        let sum = dot(vec4::splat(1.0), weights);
        let r = random_gen(rng) * sum;
        let mut weight = f32::default();
        itc *= 2;
        if (r >= weights.x + weights.y) {
            itc.y += 1;
            if (r >= weights.x + weights.y + weights.z) {
                weight = weights.w;
                itc.x += 1;
            } else {
                weight = weights.z;
            }
        } else {
            if (r >= weights.x) {
                weight = weights.y;
                itc.x += 1;
            } else {
                weight = weights.x;
            }
        }
        es.pdf *= weight / sum;
    }

    // adjust for the texel's solid angle
    es.pdf /= compute_texel_solid_angle(itc, dim);
    es.pixel = itc;
    return es;
}

#[shader]
pub fn compute_environment_sample_pdf(pixel: vec2i, dim: vec2u) -> f32 {
    let mut itc = pixel;
    let mut pdf = 1.0 / compute_texel_solid_angle(itc, dim);
    let mip_count = (textureNumLevels(&env_weights)) as i32;
    for mip in (0)..(mip_count) {
        let rem = itc & vec2i::splat(1);
        itc = itc >> vec2u::splat(1u32);
        let weights = textureLoad(&env_weights, itc, mip);
        let sum = dot(vec4::splat(1.0), weights);
        let w2 = select(weights.xy(), weights.zw(), rem.y != 0);
        let weight = select(w2.x, w2.y, rem.x != 0);
        pdf *= weight / sum;
    }
    return pdf;
}
