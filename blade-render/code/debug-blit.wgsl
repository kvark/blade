struct DebugBlitParams {
    target_offset: vec2<f32>,
    target_size: vec2<f32>,
    mip_level: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) tc: vec2<f32>,
}

var<uniform> params: DebugBlitParams;
var input: texture_2d<f32>;
var samp: sampler;

@vertex 
fn blit_vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var tc: vec2<f32>;
    var transformed: vec2<f32>;
    var vo: VertexOutput;

    tc = vec2<f32>(vec2<u32>((vi & 1u), ((vi & 2u) >> 1u)));
    let _e14 = params.target_offset;
    let _e16 = params.target_size;
    let _e18 = tc.x;
    let _e21 = tc.y;
    transformed = (_e14 + (_e16 * vec2<f32>(_e18, (1f - _e21))));
    vo = VertexOutput();
    let _e30 = tc;
    vo.tc = _e30;
    let _e33 = transformed;
    vo.clip_pos = vec4<f32>(vec3<f32>(((vec2(2f) * _e33) - vec2(1f)), 0f), 1f);
    let _e43 = vo;
    return _e43;
}

@fragment 
fn blit_fs(vo_1: VertexOutput) -> @location(0) vec4<f32> {
    let _e6 = params.mip_level;
    let _e7 = textureSampleLevel(input, samp, vo_1.tc, _e6);
    return _e7;
}
