#include "debug.inc.wgsl"
#include "color.inc.wgsl"
#include "debug-param.inc.wgsl"

struct PostProcParams {
    tone_map_enabled: u32,
    average_lum: f32,
    key_value: f32,
    // minimum value of the pixels mapped to white brightness
    white_level: f32,
    // when set, the color comes from the path traced accumulator
    accumulated: u32,
    // when set, the surface needs the values encoded for the display
    encode_srgb: u32,
}

var t_diffuse_albedo: texture_2d<f32>;
var t_emissive: texture_2d<f32>;
var light_diffuse: texture_2d<f32>;
var light_specular: texture_2d<f32>;
// RGB is the sum of the radiance, alpha is the number of samples in it
var t_accumulation: texture_2d<f32>;
var t_debug: texture_2d<f32>;
var<uniform> post_proc_params: PostProcParams;
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
    let illumination = textureLoad(light_diffuse, tc, 0);
    if (debug_params.view_mode == DebugMode_Final) {
        var color: vec3<f32>;
        if (post_proc_params.accumulated != 0u) {
            // The canonical renderer produces the final radiance directly.
            let total = textureLoad(t_accumulation, tc, 0);
            color = total.xyz / max(total.w, 1.0);
        } else {
            // The diffuse light is demodulated by the albedo, while the specular
            // one is not, since it's tinted by the Fresnel reflectance.
            let diffuse_albedo = textureLoad(t_diffuse_albedo, tc, 0).xyz;
            let specular = textureLoad(light_specular, tc, 0).xyz;
            let emissive = textureLoad(t_emissive, tc, 0).xyz;
            color = diffuse_albedo * illumination.xyz + specular + emissive;
        }
        if (post_proc_params.tone_map_enabled == 0u) {
            // Hand back the composed radiance untouched. A display transfer
            // function is only defined over the display range, so a value
            // that was never brought into it doesn't get encoded.
            return vec4<f32>(color, 1.0);
        }
        // Following https://blog.en.uwa4d.com/2022/07/19/physically-based-renderingg-hdr-tone-mapping/
        let l_adjusted = post_proc_params.key_value / post_proc_params.average_lum * color;
        let l_white = post_proc_params.white_level;
        let mapped = l_adjusted * (1.0 + l_adjusted / (l_white*l_white)) / (1.0 + l_adjusted);
        let encode = post_proc_params.encode_srgb != 0u;
        return vec4<f32>(encode_surface_color(mapped, encode), 1.0);
    } else if (debug_params.view_mode == DebugMode_Variance) {
        return vec4<f32>(illumination.w);
    } else {
        return textureLoad(t_debug, tc, 0);
    }
}
