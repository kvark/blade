var<storage, read> input: array<u32>;
var<storage, read_write> output: array<u32>;

@compute
@workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output[idx] = input[idx] * 2u + 1u;
}
