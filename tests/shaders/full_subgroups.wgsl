var<storage, read> input: array<u32>;
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(subgroup_size) width: u32) {
    let active_count = subgroupAdd(1u);
    output[gid.x] = input[gid.x] * 2u + 1u + 10000u * (active_count + width);
}
