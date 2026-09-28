use super::brdf::PI;
use super::env_importance::{
    compute_environment_sample_pdf, compute_latitude_area_bounds, generate_environment_sample,
};
use super::hit::sampler_linear;
use super::random::{RandomState, random_gen};
use super::sampling::sample_circle_uniform;
use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct LightSample {
    pub radiance: vec3,
    // Solid angle density of drawing this sample.
    pub pdf: f32,
    pub uv: vec2,
}

static env_map: texture_2d<f32> = binding();
static sampler_nearest: sampler = binding();

pub fn map_equirect_dir_to_uv(dir: vec3) -> vec2 {
    //Note: Y axis is up
    let yaw = dir.y.asin();
    let pitch = dir.x.atan2(dir.z);
    return vec2(pitch + PI, -2.0 * yaw + PI) / (2.0 * PI);
}

pub fn map_equirect_uv_to_dir(uv: vec2) -> vec3 {
    let yaw = PI * (0.5 - uv.y);
    let pitch = 2.0 * PI * (uv.x - 0.5);
    return vec3(yaw.cos() * pitch.sin(), yaw.sin(), yaw.cos() * pitch.cos());
}

fn sample_light_from_environment(rng: &mut RandomState) -> LightSample {
    let dim = env_map.level_dimensions(0);
    let es = generate_environment_sample(rng, dim);
    let mut ls = LightSample::default();
    ls.pdf = es.pdf;
    // sample the incoming radiance
    ls.radiance = env_map.load(es.pixel, 0).xyz();
    // for determining direction - offset randomly within the texel
    // this offset has to be uniformly distributed across the surface of the texel
    let u = ((es.pixel.x) as f32 + random_gen(rng)) / (dim.x) as f32;
    let bounds = compute_latitude_area_bounds(es.pixel.y, dim.y);
    let v = mix(bounds.x, bounds.y, random_gen(rng)).acos() / PI;
    ls.uv = vec2(u, v);
    return ls;
}

pub fn compute_light_pdf(uv: vec2, importance: bool) -> f32 {
    if !importance {
        return 1.0 / (4.0 * PI);
    }
    let dim = env_map.level_dimensions(0);
    let pixel = clamp(
        vec2i::from(uv * vec2::from(dim)),
        vec2i::splat(0),
        vec2i::from(dim) - vec2i::splat(1),
    );
    return compute_environment_sample_pdf(pixel, dim);
}

pub fn evaluate_environment(dir: vec3) -> vec3 {
    let uv = map_equirect_dir_to_uv(dir);
    return env_map.sample_level(&sampler_nearest, uv, 0.0).xyz();
}

pub fn evaluate_environment_background(dir: vec3) -> vec3 {
    let uv = map_equirect_dir_to_uv(dir);
    return env_map.sample_level(&sampler_linear, uv, 0.0).xyz();
}

fn sample_light_from_sphere(rng: &mut RandomState) -> LightSample {
    let a = random_gen(rng);
    let h = 1.0 - 2.0 * random_gen(rng); // make sure to allow h==1
    let tangential = sqrt(max(0.0, 1.0 - h * h)) * sample_circle_uniform(a);
    let dir = vec3(tangential.x, h, tangential.y);
    let mut ls = LightSample::default();
    ls.uv = map_equirect_dir_to_uv(dir);
    ls.pdf = 1.0 / (4.0 * PI);
    ls.radiance = env_map.sample_level(&sampler_nearest, ls.uv, 0.0).xyz();
    return ls;
}

pub fn sample_light(importance: bool, rng: &mut RandomState) -> LightSample {
    if importance {
        return sample_light_from_environment(rng);
    } else {
        return sample_light_from_sphere(rng);
    }
}
