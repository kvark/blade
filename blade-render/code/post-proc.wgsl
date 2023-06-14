struct ToneMapParams {
    enabled: u32,
    average_lum: f32,
    key_value: f32,
    // minimum value of the pixels mapped to white brightness
    white_level: f32,
}

var input: texture_2d<f32>;
var<uniform> tone_map_params: ToneMapParams;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) input_size: vec2<u32>,
}

@vertex
fn blit_vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo: VertexOutput;
    vo.clip_pos = vec4<f32>(f32(vi & 1u) * 4.0 - 1.0, f32(vi & 2u) * 2.0 - 1.0, 0.0, 1.0);
    vo.input_size = textureDimensions(input, 0);
    return vo;
}

@fragment
fn blit_fs(vo: VertexOutput) -> @location(0) vec4<f32> {
    let tc = vec2<i32>(i32(vo.clip_pos.x), i32(vo.input_size.y) - i32(vo.clip_pos.y));
    let color = textureLoad(input, tc, 0);
    if (tone_map_params.enabled != 0u) {
        // Following https://blog.en.uwa4d.com/2022/07/19/physically-based-renderingg-hdr-tone-mapping/
        let l_adjusted = tone_map_params.key_value / tone_map_params.average_lum * color.xyz;
        let l_white = tone_map_params.white_level;
        let l_ldr = l_adjusted * (1.0 + l_adjusted / (l_white*l_white)) / (1.0 + l_adjusted);
        return vec4<f32>(l_ldr, 1.0);
    } else {
        return color;
    }
}
