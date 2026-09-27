use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct DebugBlitParams {
    pub target_offset: vec2,
    pub target_size: vec2,
    pub mip_level: f32,
}

#[derive(Clone, Copy, Debug, Default, Io)]
pub struct VertexOutput {
    #[builtin(position)]
    clip_pos: vec4,
    #[location(0)]
    tc: vec2,
}

pub static params: Uniform<DebugBlitParams> = binding();

pub static input: texture_2d<f32> = binding();

pub static samp: sampler = binding();

#[entry_point(vertex)]
pub fn blit_vs(#[builtin(vertex_index)] vi: u32) -> VertexOutput {
    let tc = vec2::from(vec2u(vi & 1u32, (vi & 2u32) >> 1u32));
    let transformed = params.target_offset + params.target_size * vec2(tc.x, 1.0 - tc.y);
    let mut vo = VertexOutput::default();
    vo.tc = tc;
    vo.clip_pos = ((2.0 * transformed - 1.0).extend(0.0)).extend(1.0);
    return vo;
}

#[entry_point(fragment)]
#[output(location(0))]
pub fn blit_fs(vo: VertexOutput) -> vec4 {
    return input.sample_level(&samp, vo.tc, params.mip_level);
}
