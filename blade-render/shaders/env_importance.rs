use super::brdf::PI;
use super::random::{RandomState, random_gen};
use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct EnvImportantSample {
    pub pixel: Vec2<i32>,
    pub pdf: f32,
}

static env_weights: Texture2D<f32> = binding();

pub fn compute_latitude_area_bounds(texel_y: i32, dim: u32) -> Vec2 {
    return cos(Vec2::from(vec2::<i32>(texel_y, texel_y + 1)) / (dim) as f32 * PI);
}

fn compute_texel_solid_angle(itc: Vec2<i32>, dim: Vec2<u32>) -> f32 {
    //Note: this has to agree with `map_equirect_uv_to_dir`
    let meridian_solid_angle = 4.0 * PI / (dim.x) as f32;
    let bounds = compute_latitude_area_bounds(itc.y, dim.y);
    let meridian_part = 0.5 * (bounds.x - bounds.y);
    return meridian_solid_angle * meridian_part;
}

pub fn generate_environment_sample(rng: &mut RandomState, dim: Vec2<u32>) -> EnvImportantSample {
    let mut es = EnvImportantSample::default();
    es.pdf = 1.0;
    let mut mip = (env_weights.num_levels()) as i32;
    let mut itc = Vec2::<i32>::splat(0);
    // descend through the mip chain to find a concrete pixel
    while mip != 0 {
        mip -= 1;
        let weights = env_weights.load(itc, mip);
        let sum = dot(Vec4::splat(1.0), weights);
        let r = random_gen(rng) * sum;
        let mut weight = f32::default();
        itc *= 2;
        if r >= weights.x + weights.y {
            itc.y += 1;
            if r >= weights.x + weights.y + weights.z {
                weight = weights.w;
                itc.x += 1;
            } else {
                weight = weights.z;
            }
        } else {
            if r >= weights.x {
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

pub fn compute_environment_sample_pdf(pixel: Vec2<i32>, dim: Vec2<u32>) -> f32 {
    let mut itc = pixel;
    let mut pdf = 1.0 / compute_texel_solid_angle(itc, dim);
    let mip_count = (env_weights.num_levels()) as i32;
    for mip in (0)..(mip_count) {
        let rem = itc & Vec2::<i32>::splat(1);
        itc = itc >> Vec2::<u32>::splat(1u32);
        let weights = env_weights.load(itc, mip);
        let sum = dot(Vec4::splat(1.0), weights);
        let w2 = select(weights.xy(), weights.zw(), rem.y != 0);
        let weight = select(w2.x, w2.y, rem.x != 0);
        pdf *= weight / sum;
    }
    return pdf;
}
