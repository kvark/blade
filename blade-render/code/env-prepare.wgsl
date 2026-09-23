struct EnvPreprocParams {
    target_level: u32,
}

const PI: f32 = 3.1415925f;
const LUMA: vec3<f32> = vec3<f32>(0.299f, 0.587f, 0.114f);
const MAX_FP16: f32 = 65504f;
const SUM: vec4<f32> = vec4<f32>(0.25f, 0.25f, 0.25f, 0.25f);

var source: texture_2d<f32>;
var destination: texture_storage_2d<rgba16float,write>;
var<uniform> params: EnvPreprocParams;

fn get_pixel_weight(pixel: vec2<u32>, src_size_1: vec2<u32>) -> f32 {
    var color: vec4<f32>;
    var luma: f32;
    var elevation: f32;
    var relative_solid_angle: f32;

    if any((pixel >= src_size_1)) {
        return 0f;
    }
    let _e10 = textureLoad(source, vec2<i32>(pixel), 0i);
    color = _e10;
    let _e13 = params.target_level;
    if (_e13 == 0u) {
        let _e18 = color;
        luma = max(0f, dot(LUMA, _e18.xyz));
        elevation = ((((f32(pixel.y) + 0.5f) / f32(src_size_1.y)) - 0.5f) * PI);
        let _e35 = elevation;
        relative_solid_angle = cos(_e35);
        let _e38 = luma;
        let _e39 = relative_solid_angle;
        return clamp((_e38 * _e39), 0f, MAX_FP16);
    } else {
        let _e45 = color;
        return dot(SUM, _e45);
    }
}

@compute @workgroup_size(8, 8, 1) 
fn downsample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var dst_size: vec2<u32>;
    var src_size: vec2<u32>;
    var value: vec4<f32>;

    let _e4 = textureDimensions(destination);
    dst_size = _e4;
    let _e7 = dst_size;
    if any((global_id.xy >= _e7)) {
        return;
    }
    let _e10 = textureDimensions(source);
    src_size = _e10;
    let _e20 = src_size;
    let _e21 = get_pixel_weight(((global_id.xy * vec2(2u)) + vec2<u32>(0u, 0u)), _e20);
    let _e30 = src_size;
    let _e31 = get_pixel_weight(((global_id.xy * vec2(2u)) + vec2<u32>(1u, 0u)), _e30);
    let _e40 = src_size;
    let _e41 = get_pixel_weight(((global_id.xy * vec2(2u)) + vec2<u32>(0u, 1u)), _e40);
    let _e50 = src_size;
    let _e51 = get_pixel_weight(((global_id.xy * vec2(2u)) + vec2<u32>(1u, 1u)), _e50);
    value = vec4<f32>(_e21, _e31, _e41, _e51);
    let _e56 = value;
    textureStore(destination, vec2<i32>(global_id.xy), _e56);
}
