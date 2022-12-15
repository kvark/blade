struct Particle {
    pos: vec2<f32>,
    rot: f32,
    scale: f32,
    color: u32,
    pos_vel: vec2<f32>,
    rot_vel: f32,
    age: u32,
}
var<storage,read_write> particles: array<Particle>;

struct FreeList {
    count: atomic<i32>,
    data: array<u32>,
}
var<storage,read_write> free_list: FreeList;

@compute @workgroup_size(64, 1, 1)
fn reset(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>,
) {
    let total = num_groups.x * 64u;
    let global_index = total - 1u - global_id.x * 64u - local_index;
    free_list.data[global_index] = global_index;
    var p: Particle;
    particles[global_index] = p;
    if (local_index == 0u) {
        atomicStore(&free_list.count, i32(total));
    }
}

struct UpdateParams {
    time_delta: f32,
    max_age: u32,
}
var<uniform> update_params: UpdateParams;

@compute @workgroup_size(64, 1, 1)
fn emit(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let start_offset = atomicSub(&free_list.count, 64);
    if (start_offset < 64) {
        atomicAdd(&free_list.count, 64 - max(0, start_offset));
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
        (*p).pos += (*p).pos_vel * update_params.time_delta;
        (*p).rot += (*p).rot_vel * update_params.time_delta;
        (*p).age += 1u;
        if ((*p).age >= update_params.max_age) {
            let list_index = atomicAdd(&free_list.count, 1);
            free_list.data[list_index] = global_id.x;
            (*p).scale = 0.0;
        }
    }
}

struct Transform2D {
    pos: vec2<f32>,
    scale: f32,
    rot: f32,
}

struct DrawParams {
    t_emitter: Transform2D,
    screen_center: vec2<f32>,
    screen_extent: vec2<f32>, 
}
var<uniform> draw_params: DrawParams;

struct VertexOutput {
    @builtin(position) proj_pos: vec4<f32>,
    @location(0) color: u32,
}

fn transform(t: Transform2D, pos: vec2<f32>) -> vec2<f32> {
    let rc = cos(t.rot);
    let rs = sin(t.rot);
    return t.scale * vec2<f32>(rc*pos.x - rs*pos.y, rs*pos.x + rc*pos.y) + t.pos;
}

@vertex
fn draw_vs(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let particle = particles[instance_index];
    var out: VertexOutput;
    let zero_one_pos = vec2<f32>(vec2<u32>(vertex_index&1u, vertex_index>>1u));
    let emitter_pos = particle.scale * (2.0 * zero_one_pos - vec2<f32>(1.0)) + particle.pos;
    let world_pos = transform(draw_params.t_emitter, emitter_pos);
    out.proj_pos = vec4<f32>((world_pos - draw_params.screen_center) / draw_params.screen_extent, 0.0, 1.0);
    out.color = particle.color;
    return out;
}

@fragment
fn draw_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    //TODO: texture fetch
    return unpack4x8unorm(in.color);
}
