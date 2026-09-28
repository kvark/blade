use synaga_shader::*;

#[derive(Clone, Copy, Default)]
struct DebugBlitParams {
    pub target_offset: Vec2,
    pub target_size: Vec2,
    pub mip_level: f32,
}

#[derive(Clone, Copy, Debug, Default, Io)]
struct VertexOutput {
    #[builtin(position)]
    clip_pos: Vec4,
    #[location(0)]
    tc: Vec2,
}

static params: Uniform<DebugBlitParams> = binding();
static input: Texture2D<f32> = binding();
static samp: Sampler = binding();

#[entry_point(vertex)]
fn blit_vs(#[builtin(vertex_index)] vi: u32) -> VertexOutput {
    let tc = Vec2::from(vec2::<u32>(vi & 1u32, (vi & 2u32) >> 1u32));
    let transformed = params.target_offset + params.target_size * vec2(tc.x, 1.0 - tc.y);
    let mut vo = VertexOutput::default();
    vo.tc = tc;
    vo.clip_pos = ((2.0 * transformed - 1.0).extend(0.0)).extend(1.0);
    return vo;
}

#[entry_point(fragment)]
#[output(location(0))]
fn blit_fs(vo: VertexOutput) -> Vec4 {
    return input.sample_level(&samp, vo.tc, params.mip_level);
}
