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
    // Avoid a function-local array here. naga 30.0.0 decorates that array
    // with ArrayStride in SPIR-V, which fails Vulkan validation under
    // VUID-StandaloneSpirv-None-10684 even though the tested drivers accept it.
    let position = select(
        select(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>(3.0, -1.0),
            index == 1u,
        ),
        vec2<f32>(-1.0, 3.0),
        index == 2u,
    );
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

    // Validation reads the first target row after all passes. Full-range
    // additive RGBA8 output saturates that row to 0xff after a short chain,
    // hiding missing late passes. Keep that one row to exact 2-bit UNORM
    // increments (and alpha to one increment), which cannot saturate in the
    // benchmark's 64-pass sweep. The other 1023 rows retain the measured
    // full-range workload.
    let validation_row = input.position.y < 1.0;
    let channel_mask = select(255u, 3u, validation_row);
    let alpha = select(1.0, 1.0 / 255.0, validation_row);
    let red = f32(value & channel_mask) / 255.0;
    let green = f32((value >> 8u) & channel_mask) / 255.0;
    let blue = f32((value >> 16u) & channel_mask) / 255.0;
    return vec4<f32>(red, green, blue, alpha);
}
