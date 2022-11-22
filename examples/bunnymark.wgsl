#header

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let tc = vec2<f32>(f32(vi & 1u), 0.5 * f32(vi & 2u));
    let offset = tc.x * vec2<f32>($sprite_size);
    let pos = $mvp_transform * vec4<f32>($sprite_pos + offset, 0.0, 1.0);
    let color = vec4<f32>((vec4<u32>($sprite_color) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(255u)) / 255.0;
    return VertexOutput(pos, tc, color);
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return vertex.color * textureSampleLevel($sprite_texture, $sprite_sampler, vertex.tex_coords, 0.0);
}
