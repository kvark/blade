@group(0) @binding(0)
var<storage, read> input: array<f32>;
@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn doubleMe(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let array_length = arrayLength(&input);
    if (global_id.x >= array_length) {
        return;
    }
    output[global_id.x] = input[global_id.x] * 2.0;
}
