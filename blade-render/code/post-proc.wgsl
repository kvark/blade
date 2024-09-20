#include "debug.inc.wgsl"
#include "debug-param.inc.wgsl"

struct ToneMapParams {
    enabled: u32,
    average_lum: f32,
    key_value: f32,
    // minimum value of the pixels mapped to white brightness
    white_level: f32,
}

var t_albedo: texture_2d<f32>;
var light_diffuse: texture_2d<f32>;
var t_debug: texture_2d<f32>;
var<uniform> tone_map_params: ToneMapParams;
var<uniform> debug_params: DebugParams;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) input_size: vec2<u32>,
}

@vertex
fn postfx_vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo: VertexOutput;
    vo.clip_pos = vec4<f32>(f32(vi & 1u) * 4.0 - 1.0, f32(vi & 2u) * 2.0 - 1.0, 0.0, 1.0);
    vo.input_size = textureDimensions(light_diffuse, 0);
    return vo;
}

@fragment
fn postfx_fs(vo: VertexOutput) -> @location(0) vec4<f32> {
    let tc = vec2<i32>(i32(vo.clip_pos.x), i32(vo.clip_pos.y));
    let illumunation = textureLoad(light_diffuse, tc, 0);
    if (debug_params.view_mode == DebugMode_Final) {
        let albedo = textureLoad(t_albedo, tc, 0).xyz;
        let color = albedo.xyz * illumunation.xyz;
        if (tone_map_params.enabled != 0u) {
            // Following https://blog.en.uwa4d.com/2022/07/19/physically-based-renderingg-hdr-tone-mapping/
            let l_adjusted = tone_map_params.key_value / tone_map_params.average_lum * color;
            let l_white = tone_map_params.white_level;
            let l_ldr = l_adjusted * (1.0 + l_adjusted / (l_white*l_white)) / (1.0 + l_adjusted);
            return vec4<f32>(l_ldr, 1.0);
        } else {
            return vec4<f32>(color, 1.0);
        }
    } else if (debug_params.view_mode == DebugMode_Variance) {
        return vec4<f32>(illumunation.w);
    } else {
        return textureLoad(t_debug, tc, 0);
    }
}
