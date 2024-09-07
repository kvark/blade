const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
const MOTION_FACTOR: f32 = 0.1;

var inout_diffuse: texture_storage_2d<rgba16float, read_write>;

fn accumulate_temporal(
    pixel: vec2<i32>, cur_illumination: vec3<f32>,
    temporal_weight: f32, prev_pixel: vec2<i32>,
    motion_sqr: f32,
) {
    var illumination = cur_illumination;
    if (prev_pixel.x >= 0 && temporal_weight < 1.0) {
        let factor = mix(temporal_weight, 1.0, min(pow(motion_sqr, 0.25) * MOTION_FACTOR, 1.0));
        let prev_illumination = textureLoad(inout_diffuse, prev_pixel).xyz;
        illumination = mix(prev_illumination, illumination, factor);
    }

    let luminocity = dot(illumination, LUMA);
    let ilm = vec4<f32>(illumination, luminocity * luminocity);
    textureStore(inout_diffuse, pixel, ilm);
}
