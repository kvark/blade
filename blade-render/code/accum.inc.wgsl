const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

var inout_diffuse: texture_storage_2d<rgba16float, read_write>;

fn accumulate_temporal(
    surface: Surface, position: vec3<f32>, pixel_coord: vec2<i32>,
    cur_illumination: vec3<f32>, temporal_weight: f32,
    prev_surface: Surface, prev_pixel: vec2<i32>, prev_valid: bool,
) {
    let cur_luminocity = dot(cur_illumination, LUMA);
    var ilm = vec4<f32>(cur_illumination, cur_luminocity * cur_luminocity);
    if (prev_valid && temporal_weight < 1.0) {
        let illumination = textureLoad(inout_diffuse, prev_pixel).xyz;
        let luminocity = dot(illumination, LUMA);
        let prev_ilm = vec4<f32>(illumination, luminocity * luminocity);
        ilm = mix(prev_ilm, ilm, temporal_weight);
    }
    textureStore(inout_diffuse, pixel_coord, ilm);
}
