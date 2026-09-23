use super::brdf::*;
use super::env_importance::*;
use super::hit::*;
use super::random::*;
use super::sampling::sample_circle_uniform;
use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct LightSample {
    pub radiance: vec3,
    // Solid angle density of drawing this sample.
    pub pdf: f32,
    pub uv: vec2,
}

pub static env_map: texture_2d<f32> = binding();

pub static sampler_nearest: sampler = binding();

#[shader]
pub fn map_equirect_dir_to_uv(dir: vec3) -> vec2 {
    //Note: Y axis is up
    let yaw = asin(dir.y);
    let pitch = atan2(dir.x, dir.z);
    return vec2(pitch + PI, -2.0 * yaw + PI) / (2.0 * PI);
}

#[shader]
pub fn map_equirect_uv_to_dir(uv: vec2) -> vec3 {
    let yaw = PI * (0.5 - uv.y);
    let pitch = 2.0 * PI * (uv.x - 0.5);
    return vec3(cos(yaw) * sin(pitch), sin(yaw), cos(yaw) * cos(pitch));
}

#[shader]
pub fn sample_light_from_environment(rng: &mut RandomState) -> LightSample {
    let dim = textureDimensionsLevel(&env_map, 0);
    let es = generate_environment_sample(rng, dim);
    let mut ls = LightSample::default();
    ls.pdf = es.pdf;
    // sample the incoming radiance
    ls.radiance = textureLoad(&env_map, es.pixel, 0).xyz();
    // for determining direction - offset randomly within the texel
    // this offset has to be uniformly distributed across the surface of the texel
    let u = ((es.pixel.x) as f32 + random_gen(rng)) / (dim.x) as f32;
    let bounds = compute_latitude_area_bounds(es.pixel.y, dim.y);
    let v = acos(mix(bounds.x, bounds.y, random_gen(rng))) / PI;
    ls.uv = vec2(u, v);
    return ls;
}

#[shader]
pub fn compute_light_pdf(uv: vec2, importance: bool) -> f32 {
    if (!importance) {
        return 1.0 / (4.0 * PI);
    }
    let dim = textureDimensionsLevel(&env_map, 0);
    let pixel = clamp(
        vec2i::from(uv * vec2::from(dim)),
        vec2i::splat(0),
        vec2i::from(dim) - vec2i::splat(1),
    );
    return compute_environment_sample_pdf(pixel, dim);
}

#[shader]
pub fn evaluate_environment(dir: vec3) -> vec3 {
    let uv = map_equirect_dir_to_uv(dir);
    return textureSampleLevel(&env_map, &sampler_nearest, uv, 0.0).xyz();
}

#[shader]
pub fn evaluate_environment_background(dir: vec3) -> vec3 {
    let uv = map_equirect_dir_to_uv(dir);
    return textureSampleLevel(&env_map, &sampler_linear, uv, 0.0).xyz();
}

#[shader]
pub fn sample_light_from_sphere(rng: &mut RandomState) -> LightSample {
    let a = random_gen(rng);
    let h = 1.0 - 2.0 * random_gen(rng); // make sure to allow h==1
    let tangential = sqrt(max(0.0, 1.0 - h * h)) * sample_circle_uniform(a);
    let dir = vec3(tangential.x, h, tangential.y);
    let mut ls = LightSample::default();
    ls.uv = map_equirect_dir_to_uv(dir);
    ls.pdf = 1.0 / (4.0 * PI);
    ls.radiance = textureSampleLevel(&env_map, &sampler_nearest, ls.uv, 0.0).xyz();
    return ls;
}

#[shader]
pub fn sample_light(importance: bool, rng: &mut RandomState) -> LightSample {
    if (importance) {
        return sample_light_from_environment(rng);
    } else {
        return sample_light_from_sphere(rng);
    }
}
