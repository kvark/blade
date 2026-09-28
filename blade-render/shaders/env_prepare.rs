use synaga_shader::*;

const PI: f32 = 3.1415926;

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
    if any(pixel.cmpge(src_size)) {
        return 0.0;
    }
    let color = source.load(Vec2::<i32>::from(pixel), 0);
    if params.target_level == 0u32 {
        let luma = max(0.0, dot(LUMA, color.xyz()));
        let elevation = (((pixel.y) as f32 + 0.5) / (src_size.y) as f32 - 0.5) * PI;
        let relative_solid_angle = cos(elevation);
        return clamp(luma * relative_solid_angle, 0.0, MAX_FP16);
    } else {
        return dot(SUM, color);
    }
}

#[entry_point(compute, threads(8, 8))]
fn downsample(#[builtin(global_invocation_id)] global_id: Vec3<u32>) {
    let dst_size = destination.dimensions();
    if any(global_id.xy().cmpge(dst_size)) {
        return;
    }

    let src_size = source.dimensions();
    let value = vec4(
        get_pixel_weight(global_id.xy() * 2u32 + vec2::<u32>(0u32, 0u32), src_size),
        get_pixel_weight(global_id.xy() * 2u32 + vec2::<u32>(1u32, 0u32), src_size),
        get_pixel_weight(global_id.xy() * 2u32 + vec2::<u32>(0u32, 1u32), src_size),
        get_pixel_weight(global_id.xy() * 2u32 + vec2::<u32>(1u32, 1u32), src_size),
    );

    destination.store(Vec2::<i32>::from(global_id.xy()), value);
}
