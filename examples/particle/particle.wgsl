struct Particle {
    pos: vec2<f32>,
    pos_vel: vec2<f32>,
    rot: vec2<f32>,
    rot_vel: vec2<f32>,
    scale: f32,
    scale_vel: f32,
    color: u32,
    age: f32,
}
var<storage,read_write> particles: array<Particle>;

struct Uniforms {
    time_delta: f32,
}
var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64, 1, 1)
fn update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    //todo
}
