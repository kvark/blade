var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let transforms = array<mat3x3<f32>, 3>(
        mat3x3<f32>(vec3<f32>(2.0, 0.0, 0.0), vec3<f32>(0.0, 3.0, 0.0), vec3<f32>(0.0, 0.0, 4.0)),
        mat3x3<f32>(vec3<f32>(-2.0, 0.0, 0.0), vec3<f32>(0.0, 3.0, 0.0), vec3<f32>(0.0, 0.0, 4.0)),
        mat3x3<f32>(vec3<f32>(0.0, 2.0, 0.0), vec3<f32>(-3.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 4.0)),
    );
    output[id.x] = vec4<f32>(transform_normal(transforms[id.x], vec3<f32>(1.0, 2.0, 3.0)), 0.0);
}
