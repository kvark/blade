struct Particle {
    pos: vec2<f32>,
    pos_vel: vec2<f32>,
    rot: f32,
    rot_vel: f32,
    scale: f32,
    color: u32,
    age: u32,
}
var<storage,read_write> particles: array<Particle>;

struct FreeList {
    count: atomic<i32>,
    data: array<u32>,
}
var<storage,read_write> free_list: FreeList;

struct Uniforms {
    time_delta: f32,
    max_age: u32,
}
var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64, 1, 1)
fn emit(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let start_offset = atomicSub(&free_list.count, 64);
    if (start_offset < 64) {
        atomicAdd(&free_list.count, 64 - start_offset);
    }
    workgroupBarrier();
    let list_index = start_offset - i32(global_id.x);
    var p: Particle;
    if (list_index >= 0) {
        let p_index = free_list.data[list_index];
        particles[p_index] = p;
    }
}

@compute @workgroup_size(64, 1, 1)
fn update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let p = &particles[global_id.x];
    if ((*p).scale != 0.0) {
        (*p).pos += (*p).pos_vel * uniforms.time_delta;
        (*p).rot += (*p).rot_vel * uniforms.time_delta;
        (*p).age += 1u;
        if ((*p).age >= uniforms.max_age) {
            let list_index = atomicAdd(&free_list.count, 1);
            free_list.data[list_index] = global_id.x;
            (*p).scale = 0.0;
        }
    }
}
