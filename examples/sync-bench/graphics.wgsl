struct GraphicsParams {
    rounds: u32,
    seed: u32,
    _padding_a: u32,
    _padding_b: u32,
};

var<uniform> graphics_params: GraphicsParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[index];
    return VertexOutput(
        vec4<f32>(position, 0.0, 1.0),
        position * 0.5 + vec2<f32>(0.5),
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let coordinate = vec2<u32>(abs(input.uv) * vec2<f32>(65535.0));
    var value = coordinate.x ^ (coordinate.y * 747796405u) ^ graphics_params.seed;
    for (var round = 0u; round < graphics_params.rounds; round += 1u) {
        value = value * 1664525u + 1013904223u;
        value = value ^ (value >> 16u);
    }

    let red = f32(value & 255u) / 255.0;
    let green = f32((value >> 8u) & 255u) / 255.0;
    let blue = f32((value >> 16u) & 255u) / 255.0;
    return vec4<f32>(red, green, blue, 1.0);
}
