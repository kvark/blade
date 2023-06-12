#include "../../blade-render/code/random.inc.wgsl"

const PI: f32 = 3.1415926;
const BUMP: f32 = 0.025;

var env_main: texture_2d<f32>;
var env_weights: texture_2d<f32>;

fn compute_texel_solid_angle(itc: vec2<i32>, dim: vec2<u32>) -> f32 {
    let meridian_solid_angle = 4.0 * PI / f32(dim.x);
    let meridian_part = 0.5 * (cos(PI * f32(itc.y) / f32(dim.y)) - cos(PI * f32(itc.y + 1) / f32(dim.y)));
    return meridian_solid_angle * meridian_part;
}

struct EnvSample {
    pixel: vec2<i32>,
    pdf: f32,
}

fn sample_light_from_environment(rng: ptr<function, RandomState>) -> EnvSample {
    var es = EnvSample();
    es.pdf = 1.0;
    var mip = i32(textureNumLevels(env_weights));
    var itc = vec2<i32>(0);
    // descend through the mip chain to find a concrete pixel
    while (mip != 0) {
        mip -= 1;
        let weights = textureLoad(env_weights, itc, mip);
        let sum = dot(vec4<f32>(1.0), weights);
        let r = random_gen(rng) * sum;
        var weight: f32;
        itc *= 2;
        if (r >= weights.x+weights.y) {
            itc.y += 1;
            if (r >= weights.x+weights.y+weights.z) {
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
    let dim = textureDimensions(env_main, 0);
    es.pdf /= compute_texel_solid_angle(itc, dim);
    es.pixel = itc;
    return es;
}

fn compute_environment_sample_pdf(pixel: vec2<i32>) -> f32 {
    var itc = pixel;
    let dim = textureDimensions(env_main, 0);
    var pdf = 1.0 / compute_texel_solid_angle(itc, dim);
    let mip_count = i32(textureNumLevels(env_weights));
    for (var mip = 0; mip < mip_count; mip += 1) {
        let rem = itc & vec2<i32>(1);
        itc >>= vec2<u32>(1u);
        let weights = textureLoad(env_weights, itc, mip);
        let sum = dot(vec4<f32>(1.0), weights);
        let w2 = select(weights.xy, weights.zw, rem.y != 0);
        let weight = select(w2.x, w2.y, rem.x != 0);
        pdf *= weight / sum;
    }
    return pdf;
}

@vertex
fn vs_accum(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var rng = random_init(vi, 0u);
    let es = sample_light_from_environment(&rng);
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
    let pdf = compute_environment_sample_pdf(pixel);
    return vec4<f32>(0.0, pdf, length(uv - input.uv), 0.0);
}
