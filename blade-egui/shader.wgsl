// NOTE: Borrows heavily from egui-wgpu:s wgsl shaders, used here under the MIT license

struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

struct Uniforms {
    screen_size: vec2<f32>,
    dithering: u32,
    padding: u32,
};
var<uniform> r_uniforms: Uniforms;

//Note: avoiding `vec2<f32>` in order to keep the scalar alignment
struct Vertex {
    pos_x: f32,
    pos_y: f32,
    tex_coord_x: f32,
    tex_coord_y: f32,
    color: u32,
}
var<storage, read> r_vertex_data: array<Vertex>;

@vertex
fn vs_main(
    @builtin(vertex_index) v_index: u32,
) -> VertexOutput {
    let input = r_vertex_data[v_index];
    var out: VertexOutput;
    out.tex_coord = vec2<f32>(input.tex_coord_x, input.tex_coord_y);
    // let color = unpack4x8unorm(input.color);
    let color = unpack_color(input.color);
    out.color = color;
    out.position = vec4<f32>(
        2.0 * input.pos_x / r_uniforms.screen_size.x - 1.0,
        1.0 - 2.0 * input.pos_y / r_uniforms.screen_size.y,
        0.0,
        1.0,
    );
    return out;
}

var r_texture: texture_2d<f32>;
var r_sampler: sampler;

@fragment
fn fs_main_linear_framebuffer(in: VertexOutput) -> @location(0) vec4<f32> {
    // We always have an sRGB aware texture at the moment.
    let tex_linear = textureSample(r_texture, r_sampler, in.tex_coord);
    let tex_gamma = gamma_from_linear_rgba(tex_linear);
    var out_color_gamma = in.color * tex_gamma;
    // Dither the float color down to eight bits to reduce banding.
    // This step is optional for egui backends.
    // Note that dithering is performed on the gamma encoded values,
    // because this function is used together with a srgb converting target.
    if r_uniforms.dithering == 1 {
        let out_color_gamma_rgb = dither_interleaved(out_color_gamma.rgb, 256.0, in.position);
        out_color_gamma = vec4<f32>(out_color_gamma_rgb, out_color_gamma.a);
    }
    let out_color_linear = linear_from_gamma_rgb(out_color_gamma.rgb);
    return vec4<f32>(out_color_linear, out_color_gamma.a);
}

@fragment
fn fs_main_gamma_framebuffer(in: VertexOutput) -> @location(0) vec4<f32> {
    // We always have an sRGB aware texture at the moment.
    let tex_linear = textureSample(r_texture, r_sampler, in.tex_coord);
    let tex_gamma = gamma_from_linear_rgba(tex_linear);
    var out_color_gamma = in.color * tex_gamma;
    // Dither the float color down to eight bits to reduce banding.
    // This step is optional for egui backends.
    if r_uniforms.dithering == 1 {
        let out_color_gamma_rgb = dither_interleaved(out_color_gamma.rgb, 256.0, in.position);
        out_color_gamma = vec4<f32>(out_color_gamma_rgb, out_color_gamma.a);
    }
    return out_color_gamma;
}




// -----------------------------------------------
// Adapted from
// https://www.shadertoy.com/view/llVGzG
// Originally presented in:
// Jimenez 2014, "Next Generation Post-Processing in Call of Duty"
//
// A good overview can be found in
// https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence/
// via https://github.com/rerun-io/rerun/
fn interleaved_gradient_noise(n: vec2<f32>) -> f32 {
    let f = 0.06711056 * n.x + 0.00583715 * n.y;
    return fract(52.9829189 * fract(f));
}

fn dither_interleaved(rgb: vec3<f32>, levels: f32, frag_coord: vec4<f32>) -> vec3<f32> {
    var noise = interleaved_gradient_noise(frag_coord.xy);
    // scale down the noise slightly to ensure flat colors aren't getting dithered
    noise = (noise - 0.5) * 0.95;
    return rgb + noise / (levels - 1.0);
}

// 0-1 linear  from  0-1 sRGB gamma
fn linear_from_gamma_rgb(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

// 0-1 sRGB gamma  from  0-1 linear
fn gamma_from_linear_rgb(rgb: vec3<f32>) -> vec3<f32> {
    let cutoff = rgb < vec3<f32>(0.0031308);
    let lower = rgb * vec3<f32>(12.92);
    let higher = vec3<f32>(1.055) * pow(rgb, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(higher, lower, cutoff);
}

// 0-1 sRGBA gamma  from  0-1 linear
fn gamma_from_linear_rgba(linear_rgba: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(gamma_from_linear_rgb(linear_rgba.rgb), linear_rgba.a);
}

// [u8; 4] SRGB as u32 -> [r, g, b, a] in 0.-1
fn unpack_color(color: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(color & 255u),
        f32((color >> 8u) & 255u),
        f32((color >> 16u) & 255u),
        f32((color >> 24u) & 255u),
    ) / 255.0;
}

fn position_from_screen(screen_pos: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(
        2.0 * screen_pos.x / r_uniforms.screen_size.x - 1.0,
        1.0 - 2.0 * screen_pos.y / r_uniforms.screen_size.y,
        0.0,
        1.0,
    );
}
