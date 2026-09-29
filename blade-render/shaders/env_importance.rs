use super::random::{RandomState, random_gen};
use core::f32::consts::PI;
use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct EnvImportantSample {
    pub pixel: Vec2<i32>,
    pub pdf: f32,
}

static env_weights: Texture2D<f32> = binding();

pub fn compute_latitude_area_bounds(texel_y: i32, dim: u32) -> Vec2 {
    (vec2(texel_y, texel_y + 1).cast::<f32>() / dim as f32 * PI).cos()
}

fn compute_texel_solid_angle(itc: Vec2<i32>, dim: Vec2<u32>) -> f32 {
    //Note: this has to agree with `map_equirect_uv_to_dir`
    let meridian_solid_angle = 4.0 * PI / dim.x as f32;
    let bounds = compute_latitude_area_bounds(itc.y, dim.y);
    let meridian_part = 0.5 * (bounds.x - bounds.y);
    meridian_solid_angle * meridian_part
}

pub fn generate_environment_sample(rng: &mut RandomState, dim: Vec2<u32>) -> EnvImportantSample {
    let mut es = EnvImportantSample {
        pdf: 1.0,
        ..Default::default()
    };
    let mut mip = env_weights.num_levels() as i32;
    let mut itc: Vec2<i32> = Vec2::ZERO;
    // descend through the mip chain to find a concrete pixel
    while mip != 0 {
        mip -= 1;
        let weights = env_weights.load(itc, mip);
        let sum = weights.element_sum();
        let r = random_gen(rng) * sum;
        itc *= 2;
        let weight = if r >= weights.x + weights.y {
            itc.y += 1;
            if r >= weights.x + weights.y + weights.z {
                itc.x += 1;
                weights.w
            } else {
                weights.z
            }
        } else if r >= weights.x {
            itc.x += 1;
            weights.y
        } else {
            weights.x
        };
        es.pdf *= weight / sum;
    }

    // adjust for the texel's solid angle
    es.pdf /= compute_texel_solid_angle(itc, dim);
    es.pixel = itc;
    es
}

pub fn compute_environment_sample_pdf(pixel: Vec2<i32>, dim: Vec2<u32>) -> f32 {
    let mut itc = pixel;
    let mut pdf = 1.0 / compute_texel_solid_angle(itc, dim);
    let mip_count = env_weights.num_levels() as i32;
    for mip in 0..mip_count {
        let rem = itc & 1;
        itc >>= 1;
        let weights = env_weights.load(itc, mip);
        let sum = weights.element_sum();
        let w2 = select(weights.xy(), weights.zw(), rem.y != 0);
        let weight = select(w2.x, w2.y, rem.x != 0);
        pdf *= weight / sum;
    }
    pdf
}
