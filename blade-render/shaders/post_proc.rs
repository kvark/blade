use super::color::*;
use super::config::*;
use super::debug::*;
use super::debug_param::*;
use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct PostProcParams {
    pub tone_map_enabled: u32,
    pub average_lum: f32,
    pub key_value: f32,
    // minimum value of the pixels mapped to white brightness
    pub white_level: f32,
    // when set, the color comes from the path traced accumulator
    pub accumulated: u32,
    // when set, the surface needs the values encoded for the display
    pub encode_srgb: u32,
    // when set, final color comes from t_external
    pub external_input: u32,
    pub _pad: u32,
}

#[io]
pub struct VertexOutput {
    #[builtin(position)]
    clip_pos: vec4,
    #[location(0)]
    #[flat]
    input_size: vec2u,
}

pub static t_diffuse_albedo: texture_2d<f32> = binding();

pub static t_emissive: texture_2d<f32> = binding();

pub static light_diffuse: texture_2d<f32> = binding();

pub static light_specular: texture_2d<f32> = binding();

pub static t_accumulation: texture_2d<f32> = binding();

pub static t_debug: texture_2d<f32> = binding();

pub static t_external: texture_2d<f32> = binding();

pub static post_proc_params: Uniform<PostProcParams> = binding();

pub static debug_params: Uniform<DebugParams> = binding();

#[vertex]
pub fn postfx_vs(#[builtin(vertex_index)] vi: u32) -> VertexOutput {
    let mut vo = VertexOutput::default();
    vo.clip_pos = vec4(
        (vi & 1u32) as f32 * 4.0 - 1.0,
        (vi & 2u32) as f32 * 2.0 - 1.0,
        0.0,
        1.0,
    );
    vo.input_size = textureDimensionsLevel(&light_diffuse, 0);
    return vo;
}

#[fragment]
#[output(location(0))]
pub fn postfx_fs(vo: VertexOutput) -> vec4 {
    let tc = vec2i((vo.clip_pos.x) as i32, (vo.clip_pos.y) as i32);
    let illumination = textureLoad(&light_diffuse, tc, 0);
    if (debug_params.view_mode == DebugMode_Final) {
        let mut color = vec3::default();
        if (post_proc_params.external_input != 0u32) {
            color = textureLoad(&t_external, tc, 0).xyz();
        } else if (post_proc_params.accumulated != 0u32) {
            // The canonical renderer produces the final radiance directly.
            let total = textureLoad(&t_accumulation, tc, 0);
            color = total.xyz() / max(total.w, 1.0);
        } else {
            // The diffuse light is demodulated by the albedo, while the specular
            // one is not, since it's tinted by the Fresnel reflectance.
            let diffuse_albedo = textureLoad(&t_diffuse_albedo, tc, 0).xyz();
            let specular = textureLoad(&light_specular, tc, 0).xyz();
            let emissive = textureLoad(&t_emissive, tc, 0).xyz();
            color = diffuse_albedo * illumination.xyz() + specular + emissive;
        }
        if (post_proc_params.tone_map_enabled == 0u32) {
            // Hand back the composed radiance untouched. A display transfer
            // function is only defined over the display range, so a value
            // that was never brought into it doesn't get encoded.
            return (color).extend(1.0);
        }
        // Following https://blog.en.uwa4d.com/2022/07/19/physically-based-renderingg-hdr-tone-mapping/
        let l_adjusted = post_proc_params.key_value / post_proc_params.average_lum * color;
        let l_white = post_proc_params.white_level;
        let mapped = l_adjusted * (1.0 + l_adjusted / (l_white * l_white)) / (1.0 + l_adjusted);
        let encode = post_proc_params.encode_srgb != 0u32;
        return (encode_surface_color(mapped, encode)).extend(1.0);
    } else if (debug_params.view_mode == DebugMode_Variance) {
        return vec4::splat(illumination.w);
    } else {
        return textureLoad(&t_debug, tc, 0);
    }
}
