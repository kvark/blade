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
    return in.color * textureSample(r_texture, r_sampler, in.tex_coord);
}
