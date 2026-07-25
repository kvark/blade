struct ComputeParams {
    element_count: u32,
    rounds: u32,
    seed: u32,
    _padding: u32,
};

var<storage, read> input_data: array<u32>;
var<storage, read_write> output_data: array<u32>;
var<uniform> compute_params: ComputeParams;

@compute
@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= compute_params.element_count {
        return;
    }

    var value = input_data[index] ^ compute_params.seed;
    for (var round = 0u; round < compute_params.rounds; round += 1u) {
        value = value * 1664525u + 1013904223u;
        value = value ^ (value >> 16u);
    }
    output_data[index] = value;
}
