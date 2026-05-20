struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

struct Uniforms {
    screen_size: vec2<f32>,
    convert_to_linear: f32,
    padding: f32,
};
var<uniform> r_uniforms: Uniforms;

//Note: avoiding `vec2<f32>` in order to keep the scalar alignment
struct VertexInput {
    a_pos: vec2<f32>,
    a_tex_coord: vec2<f32>,
    a_color: u32,
}

fn linear_from_gamma(srgb: vec3<f32>) -> vec3<f32> {
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let is_higher = step(vec3<f32>(0.04045), srgb);
    return mix(lower, higher, is_higher);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coord = input.a_tex_coord;
    out.color = unpack4x8unorm(input.a_color);
    out.position = vec4<f32>(
        2.0 * input.a_pos.x / r_uniforms.screen_size.x - 1.0,
        1.0 - 2.0 * input.a_pos.y / r_uniforms.screen_size.y,
        0.0,
        1.0,
    );
    return out;
}

var r_texture: texture_2d<f32>;
var r_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    //Note: we output sRGB directly because our swapchain is configured as Srgb 
    // which gives us a Unorm format (no hardware conversion).
    let blended = in.color * textureSample(r_texture, r_sampler, in.tex_coord);
    return vec4f(blended.xyz, blended.a);
}
