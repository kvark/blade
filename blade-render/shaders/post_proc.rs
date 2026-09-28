use super::color::{encode_srgb, encode_surface_color};
use super::config::DebugMode;

use super::debug_param::DebugParams;
use synaga_shader::*;

#[derive(Clone, Copy, Default)]
struct PostProcParams {
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

#[derive(Clone, Copy, Debug, Default, Io)]
struct VertexOutput {
    #[builtin(position)]
    clip_pos: Vec4,
    #[location(0)]
    #[flat]
    input_size: Vec2<u32>,
}

static t_diffuse_albedo: Texture2D<f32> = binding();
static t_emissive: Texture2D<f32> = binding();
static light_diffuse: Texture2D<f32> = binding();
static light_specular: Texture2D<f32> = binding();
static t_accumulation: Texture2D<f32> = binding();
static t_debug: Texture2D<f32> = binding();
static t_external: Texture2D<f32> = binding();
static post_proc_params: Uniform<PostProcParams> = binding();
static debug_params: Uniform<DebugParams> = binding();

#[entry_point(vertex)]
fn postfx_vs(#[builtin(vertex_index)] vi: u32) -> VertexOutput {
    let mut vo = VertexOutput::default();
    vo.clip_pos = vec4(
        (vi & 1u32) as f32 * 4.0 - 1.0,
        (vi & 2u32) as f32 * 2.0 - 1.0,
        0.0,
        1.0,
    );
    vo.input_size = light_diffuse.level_dimensions(0);
    return vo;
}

#[entry_point(fragment)]
#[output(location(0))]
fn postfx_fs(vo: VertexOutput) -> Vec4 {
    let tc = vec2::<i32>((vo.clip_pos.x) as i32, (vo.clip_pos.y) as i32);
    let illumination = light_diffuse.load(tc, 0);
    if debug_params.view_mode == DebugMode::Final as u32 {
        let mut color = Vec3::default();
        if post_proc_params.external_input != 0u32 {
            color = t_external.load(tc, 0).xyz();
        } else if post_proc_params.accumulated != 0u32 {
            // The canonical renderer produces the final radiance directly.
            let total = t_accumulation.load(tc, 0);
            color = total.xyz() / max(total.w, 1.0);
        } else {
            // The diffuse light is demodulated by the albedo, while the specular
            // one is not, since it's tinted by the Fresnel reflectance.
            let diffuse_albedo = t_diffuse_albedo.load(tc, 0).xyz();
            let specular = light_specular.load(tc, 0).xyz();
            let emissive = t_emissive.load(tc, 0).xyz();
            color = diffuse_albedo * illumination.xyz() + specular + emissive;
        }
        if post_proc_params.tone_map_enabled == 0u32 {
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
    } else if debug_params.view_mode == DebugMode::Variance as u32 {
        return Vec4::splat(illumination.w);
    } else {
        return t_debug.load(tc, 0);
    }
}
