// Sampling of the environment map, which is our only light source.
//
// Requires "brdf.inc.wgsl", "random.inc.wgsl", "env-importance.inc.wgsl",
// as well as the `env_map` texture with the `sampler_nearest`
// and `sampler_linear` samplers.

struct LightSample {
    radiance: vec3<f32>,
    // Solid angle density of drawing this sample.
    pdf: f32,
    uv: vec2<f32>,
}

fn map_equirect_dir_to_uv(dir: vec3<f32>) -> vec2<f32> {
    //Note: Y axis is up
    let yaw = asin(dir.y);
    let pitch = atan2(dir.x, dir.z);
    return vec2<f32>(pitch + PI, -2.0 * yaw + PI) / (2.0 * PI);
}
fn map_equirect_uv_to_dir(uv: vec2<f32>) -> vec3<f32> {
    let yaw = PI * (0.5 - uv.y);
    let pitch = 2.0 * PI * (uv.x - 0.5);
    return vec3<f32>(cos(yaw) * sin(pitch), sin(yaw), cos(yaw) * cos(pitch));
}

// Radiance arriving from the environment.
//
// Note: sampled without filtering, so that it matches the density
// of `sample_light` exactly, as MIS requires.
fn evaluate_environment(dir: vec3<f32>) -> vec3<f32> {
    let uv = map_equirect_dir_to_uv(dir);
    return textureSampleLevel(env_map, sampler_nearest, uv, 0.0).xyz;
}

// Same, but filtered, for when the camera looks at the environment directly.
fn evaluate_environment_background(dir: vec3<f32>) -> vec3<f32> {
    let uv = map_equirect_dir_to_uv(dir);
    return textureSampleLevel(env_map, sampler_linear, uv, 0.0).xyz;
}

fn sample_light_from_sphere(rng: ptr<function, RandomState>) -> LightSample {
    let a = random_gen(rng);
    let h = 1.0 - 2.0 * random_gen(rng); // make sure to allow h==1
    let tangential = sqrt(max(0.0, 1.0 - h * h)) * sample_circle_uniform(a);
    let dir = vec3<f32>(tangential.x, h, tangential.y);
    var ls = LightSample();
    ls.uv = map_equirect_dir_to_uv(dir);
    ls.pdf = 1.0 / (4.0 * PI);
    ls.radiance = textureSampleLevel(env_map, sampler_nearest, ls.uv, 0.0).xyz;
    return ls;
}

fn sample_light_from_environment(rng: ptr<function, RandomState>) -> LightSample {
    let dim = textureDimensions(env_map, 0);
    let es = generate_environment_sample(rng, dim);
    var ls = LightSample();
    ls.pdf = es.pdf;
    // sample the incoming radiance
    ls.radiance = textureLoad(env_map, es.pixel, 0).xyz;
    // for determining direction - offset randomly within the texel
    // this offset has to be uniformly distributed across the surface of the texel
    let u = (f32(es.pixel.x) + random_gen(rng)) / f32(dim.x);
    let bounds = compute_latitude_area_bounds(es.pixel.y, dim.y);
    let v = acos(mix(bounds.x, bounds.y, random_gen(rng))) / PI;
    ls.uv = vec2<f32>(u, v);
    return ls;
}

fn sample_light(importance: bool, rng: ptr<function, RandomState>) -> LightSample {
    if (importance) {
        return sample_light_from_environment(rng);
    } else {
        return sample_light_from_sphere(rng);
    }
}

// Solid angle density of `sample_light` for a given direction.
fn compute_light_pdf(uv: vec2<f32>, importance: bool) -> f32 {
    if (!importance) {
        return 1.0 / (4.0 * PI);
    }
    let dim = textureDimensions(env_map, 0);
    let pixel = clamp(vec2<i32>(uv * vec2<f32>(dim)), vec2<i32>(0), vec2<i32>(dim) - vec2<i32>(1));
    return compute_environment_sample_pdf(pixel, dim);
}
