use synaga_shader::*;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct DebugBlitParams {
    pub target_offset: Vec2,
    pub target_size: Vec2,
    pub mip_level: f32,
    pub _pad: u32,
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
fn blit_vs(vertex_index: u32) -> VertexOutput {
    let tc = vec2(vertex_index & 1, (vertex_index & 2) >> 1).cast::<f32>();
    let transformed = params.target_offset + params.target_size * vec2(tc.x, 1.0 - tc.y);
    VertexOutput {
        tc,
        clip_pos: (2.0 * transformed - 1.0).extend(0.0).extend(1.0),
    }
}

#[entry_point(fragment)]
fn blit_fs(vo: VertexOutput) -> Vec4 {
    input.sample_level(&samp, vo.tc, params.mip_level)
}
