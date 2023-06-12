#include "../../blade-render/code/random.inc.wgsl"
#include "../../blade-render/code/env-importance.inc.wgsl"

const PI: f32 = 3.1415926;
const BUMP: f32 = 0.025;

var env_main: texture_2d<f32>;

@vertex
fn vs_accum(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var rng = random_init(vi, 0u);
    let dim = textureDimensions(env_main);
    let es = generate_environment_sample(&rng, dim);
    let extent = textureDimensions(env_weights, 0);
    let relative = (vec2<f32>(es.pixel) + vec2<f32>(0.5)) / vec2<f32>(extent);
    return vec4<f32>(relative.x - 1.0, 1.0 - relative.y, 0.0, 1.0);
}

@fragment
fn fs_accum() -> @location(0) vec4<f32> {
    return vec4<f32>(BUMP);
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

struct UvOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_init(@builtin(vertex_index) vi: u32) -> UvOutput {
    var vo: UvOutput;
    let uv = vec2<f32>(2.0 * f32(vi & 1u), f32(vi & 2u));
    vo.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    vo.uv = uv;
    return vo;
}

@fragment
fn fs_init(input: UvOutput) -> @location(0) vec4<f32> {
    let dir = map_equirect_uv_to_dir(input.uv);
    let uv = map_equirect_dir_to_uv(dir);
    let dim = textureDimensions(env_main);
    let pixel = vec2<i32>(uv * vec2<f32>(dim));
    let pdf = compute_environment_sample_pdf(pixel, dim);
    return vec4<f32>(0.0, pdf, length(uv - input.uv), 0.0);
}
