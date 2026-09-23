use synaga_shader::*;

pub const PI: f32 = 3.1415926;

pub const LUMA: vec3 = vec3(0.299, 0.587, 0.114);

pub const MAX_FP16: f32 = 65504.0;

pub const SUM: vec4 = vec4(0.25, 0.25, 0.25, 0.25);

#[derive(Clone, Copy, Default)]
pub struct EnvPreprocParams {
    pub target_level: u32,
}

pub static source: texture_2d<f32> = binding();

pub static destination: texture_storage_2d<Rgba16Float, Write> = binding();

pub static params: Uniform<EnvPreprocParams> = binding();

#[shader]
pub fn get_pixel_weight(pixel: vec2u, src_size: vec2u) -> f32 {
    if (any(pixel.cmpge(src_size))) {
        return 0.0;
    }
    let color = textureLoad(&source, vec2i::from(pixel), 0);
    if (params.target_level == 0u32) {
        let luma = max(0.0, dot(LUMA, color.xyz()));
        let elevation = (((pixel.y) as f32 + 0.5) / (src_size.y) as f32 - 0.5) * PI;
        let relative_solid_angle = cos(elevation);
        return clamp(luma * relative_solid_angle, 0.0, MAX_FP16);
    } else {
        return dot(SUM, color);
    }
}

#[compute]
#[workgroup_size(8, 8)]
pub fn downsample(#[builtin(global_invocation_id)] global_id: vec3u) {
    let dst_size = textureDimensions(&destination);
    if (any(global_id.xy().cmpge(dst_size))) {
        return;
    }

    let src_size = textureDimensions(&source);
    let value = vec4(
        get_pixel_weight(global_id.xy() * 2u32 + vec2u(0u32, 0u32), src_size),
        get_pixel_weight(global_id.xy() * 2u32 + vec2u(1u32, 0u32), src_size),
        get_pixel_weight(global_id.xy() * 2u32 + vec2u(0u32, 1u32), src_size),
        get_pixel_weight(global_id.xy() * 2u32 + vec2u(1u32, 1u32), src_size),
    );

    textureStore(&destination, vec2i::from(global_id.xy()), value);
}
