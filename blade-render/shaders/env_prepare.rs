use core::f32::consts::PI;
use synaga_shader::*;

const LUMA: Vec3 = vec3(0.299, 0.587, 0.114);
const MAX_FP16: f32 = 65504.0;
const SUM: Vec4 = vec4(0.25, 0.25, 0.25, 0.25);

#[derive(Clone, Copy, Default)]
struct EnvPreprocParams {
    pub target_level: u32,
}

static source: Texture2D<f32> = binding();
static destination: TextureStorage2D<Rgba16Float, Write> = binding();
static params: Uniform<EnvPreprocParams> = binding();

fn get_pixel_weight(pixel: Vec2<u32>, src_size: Vec2<u32>) -> f32 {
    if pixel.cmpge(src_size).any() {
        return 0.0;
    }
    let color = source.load(pixel.cast::<i32>(), 0);
    if params.target_level == 0 {
        let luma = LUMA.dot(color.xyz()).max(0.0);
        let elevation = ((pixel.y as f32 + 0.5) / src_size.y as f32 - 0.5) * PI;
        let relative_solid_angle = elevation.cos();
        (luma * relative_solid_angle).clamp(0.0, MAX_FP16)
    } else {
        SUM.dot(color)
    }
}

#[entry_point(compute, threads(8, 8))]
fn downsample(global_invocation_id: Vec3<u32>) {
    let dst_size = destination.dimensions();
    if global_invocation_id.xy().cmpge(dst_size).any() {
        return;
    }

    let src_size = source.dimensions();
    let value = vec4(
        get_pixel_weight(global_invocation_id.xy() * 2 + vec2(0, 0), src_size),
        get_pixel_weight(global_invocation_id.xy() * 2 + vec2(1, 0), src_size),
        get_pixel_weight(global_invocation_id.xy() * 2 + vec2(0, 1), src_size),
        get_pixel_weight(global_invocation_id.xy() * 2 + vec2(1, 1), src_size),
    );

    destination.store(global_invocation_id.xy().cast::<i32>(), value);
}
