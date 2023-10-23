struct DebugBlitParams {
    target_offset: vec2<f32>,
    target_size: vec2<f32>,
    mip_level: f32,
}
var<uniform> params: DebugBlitParams;
var input: texture_2d<f32>;
var samp: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) tc: vec2<f32>,
}

@vertex
fn blit_vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let tc = vec2<f32>(vec2<u32>(vi & 1u, (vi & 2u) >> 1u));
    let transformed = params.target_offset + params.target_size * vec2<f32>(tc.x, 1.0 - tc.y);
    var vo: VertexOutput;
    vo.tc = tc;
    vo.clip_pos = vec4<f32>(2.0 * transformed - 1.0, 0.0, 1.0);
    return vo;
}

@fragment
fn blit_fs(vo: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(input, samp, vo.tc, params.mip_level);
}
