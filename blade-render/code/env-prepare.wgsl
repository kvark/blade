var source: texture_2d<f32>;
var destination: texture_storage_2d<rgba16float, write>;
struct EnvPreprocParams {
    target_level: u32,
}
var<uniform> params: EnvPreprocParams;

const PI: f32 = 3.1415926;
const LUMA: vec3<f32> = vec3<f32>(0.299, 0.587, 0.114);
const MAX_FP16: f32 = 65504.0;
const SUM: vec4<f32> = vec4<f32>(0.25, 0.25, 0.25, 0.25);

// Returns the weight of a pixel at the specified coordinates.
// The weight is the area of a pixel multiplied by its luminance, not normalized.
fn get_pixel_weight(pixel: vec2<u32>, src_size: vec2<u32>) -> f32 {
    if (any(pixel >= src_size)) {
        return 0.0;
    }
    let color = textureLoad(source, vec2<i32>(pixel), 0);
    if (params.target_level == 0u) {
        let luma = max(0.0, dot(LUMA, color.xyz));
        let elevation = ((f32(pixel.y) + 0.5) / f32(src_size.y) - 0.5) * PI;
        let relative_solid_angle = cos(elevation);
        return clamp(luma * relative_solid_angle, 0.0, MAX_FP16);
    } else {
        return dot(SUM, color);
    }
}

@compute
@workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_size = textureDimensions(destination);
    if (any(global_id.xy >= dst_size)) {
        return;
    }

    let src_size = textureDimensions(source);
    let value = vec4<f32>(
        get_pixel_weight(global_id.xy * 2u + vec2<u32>(0u, 0u), src_size),
        get_pixel_weight(global_id.xy * 2u + vec2<u32>(1u, 0u), src_size),
        get_pixel_weight(global_id.xy * 2u + vec2<u32>(0u, 1u), src_size),
        get_pixel_weight(global_id.xy * 2u + vec2<u32>(1u, 1u), src_size)
    );

    textureStore(destination, vec2<i32>(global_id.xy), value);
}
