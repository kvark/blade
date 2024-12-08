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
struct Vertex {
    pos_x: f32,
    pos_y: f32,
    tex_coord_x: f32,
    tex_coord_y: f32,
    color: u32,
}
var<storage, read> r_vertex_data: array<Vertex>;

fn linear_from_gamma(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

@vertex
fn vs_main(
    @builtin(vertex_index) v_index: u32,
) -> VertexOutput {
    let input = r_vertex_data[v_index];
    var out: VertexOutput;
    out.tex_coord = vec2<f32>(input.tex_coord_x, input.tex_coord_y);
    out.color = unpack4x8unorm(input.color);
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
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    //Note: we always assume rendering to linear color space,
    // but Egui wants to blend in gamma space, see
    // https://github.com/emilk/egui/pull/2071
    let blended = in.color * textureSample(r_texture, r_sampler, in.tex_coord);
    return vec4f(linear_from_gamma(blended.xyz), blended.a);
}
