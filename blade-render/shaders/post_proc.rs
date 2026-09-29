use super::color::encode_surface_color;
use super::config::DebugMode;
use super::debug_param::DebugParams;
use synaga_shader::*;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
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
fn postfx_vs(vertex_index: u32) -> VertexOutput {
    VertexOutput {
        clip_pos: vec4(
            (vertex_index & 1) as f32 * 4.0 - 1.0,
            (vertex_index & 2) as f32 * 2.0 - 1.0,
            0.0,
            1.0,
        ),
        input_size: light_diffuse.level_dimensions(0),
    }
}

#[entry_point(fragment)]
fn postfx_fs(vo: VertexOutput) -> Vec4 {
    let tc = vo.clip_pos.xy().cast::<i32>();
    let illumination = light_diffuse.load(tc, 0);
    if debug_params.view_mode == DebugMode::Final {
        let color = if post_proc_params.external_input != 0 {
            t_external.load(tc, 0).xyz()
        } else if post_proc_params.accumulated != 0 {
            // The canonical renderer produces the final radiance directly.
            let total = t_accumulation.load(tc, 0);
            total.xyz() / total.w.max(1.0)
        } else {
            // The diffuse light is demodulated by the albedo, while the specular
            // one is not, since it's tinted by the Fresnel reflectance.
            let diffuse_albedo = t_diffuse_albedo.load(tc, 0).xyz();
            let specular = light_specular.load(tc, 0).xyz();
            let emissive = t_emissive.load(tc, 0).xyz();
            diffuse_albedo * illumination.xyz() + specular + emissive
        };
        if post_proc_params.tone_map_enabled == 0 {
            // Hand back the composed radiance untouched. A display transfer
            // function is only defined over the display range, so a value
            // that was never brought into it doesn't get encoded.
            return color.extend(1.0);
        }
        // Following https://blog.en.uwa4d.com/2022/07/19/physically-based-renderingg-hdr-tone-mapping/
        let l_adjusted = post_proc_params.key_value / post_proc_params.average_lum * color;
        let l_white = post_proc_params.white_level;
        let mapped = l_adjusted * (1.0 + l_adjusted / (l_white * l_white)) / (1.0 + l_adjusted);
        let encode = post_proc_params.encode_srgb != 0;
        encode_surface_color(mapped, encode).extend(1.0)
    } else if debug_params.view_mode == DebugMode::Variance {
        Vec4::splat(illumination.w)
    } else {
        t_debug.load(tc, 0)
    }
}
