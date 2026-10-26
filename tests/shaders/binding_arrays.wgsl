enable wgpu_binding_array;

var textures: binding_array<texture_2d<f32>, 4>;
var samp: sampler;

struct Input {
    values: vec4<f32>,
}
var<storage, read> buffers: binding_array<Input, 4>;
var<storage, read_write> output: array<vec4<f32>, 4>;

@compute @workgroup_size(4)
fn cs_main(@builtin(local_invocation_index) index: u32) {
    output[index] = textureLoad(textures[index], vec2<i32>(0), 0)
        + vec4<f32>(buffers[index].values.x);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) bias: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var result: VertexOutput;
    result.position = vec4<f32>(
        4.0 * f32(index & 1u) - 1.0,
        4.0 * f32(index >> 1u) - 1.0, 0.0, 1.0);
    result.bias = textureLoad(textures[1], vec2<i32>(0), 0);
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(textures[u32(vertex.position.x)], samp, vec2<f32>(0.5), 0.0)
        + 0.25 * vertex.bias;
}
