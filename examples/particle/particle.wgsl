struct Particle {
    pos: vec2<f32>,
    rot: f32,
    scale: f32,
    color: u32,
    pos_vel: vec2<f32>,
    rot_vel: f32,
    life: f32,
    generation: u32,
}
var<storage,read_write> particles: array<Particle>;

struct FreeList {
    count: atomic<i32>,
    data: array<u32>,
}
var<storage,read_write> free_list: FreeList;

@compute @workgroup_size(64, 1, 1)
fn reset(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>,
) {
    let total = num_groups.x * 64u;
    // reversing the order because it works like a stack
    free_list.data[global_id.x] = total - 1u - global_id.x;
    var p: Particle;
    particles[global_id.x] = p;
    if (global_id.x == 0u) {
        atomicStore(&free_list.count, i32(total));
    }
}

struct Parameters {
    life: f32,
    velocity: f32,
    scale: f32,
}
var<uniform> parameters: Parameters;
struct UpdateParams {
    time_delta: f32,
}
var<uniform> update_params: UpdateParams;

var<workgroup> emit_end: i32;

@compute @workgroup_size(64, 1, 1)
fn emit(@builtin(local_invocation_index) local_index: u32) {
    if (local_index == 0u) {
        emit_end = atomicSub(&free_list.count, 64);
        if (emit_end < 64) {
            atomicAdd(&free_list.count, 64 - max(0, emit_end));
        }
    }
    workgroupBarrier();

    let list_index = emit_end - 1 - i32(local_index);
    if (list_index >= 0) {
        var p: Particle;
        let p_index = free_list.data[list_index];
        p.generation += 1u;
        let random = i32(p_index * p.generation);
        p.life = max(1.0, parameters.life * f32((random + 17) % 20) / 20.0);
        p.scale = max(0.1, parameters.scale * f32((random + 13) % 10) / 10.0);
        p.color = ((p_index * 12345678u + p.generation * 912123u) << 8u) | 0xFFu;
        let angle = f32(random) * 0.3;
        let a_sin = sin(angle);
        let a_cos = cos(angle);
        p.pos_vel = parameters.velocity * vec2<f32>(a_cos, a_sin);
        p.rot_vel = f32((list_index + 189) % 10);
        particles[p_index] = p;
    }
}

@compute @workgroup_size(64, 1, 1)
fn update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let p = &particles[global_id.x];
    if ((*p).scale != 0.0) {
        (*p).pos += (*p).pos_vel * update_params.time_delta;
        (*p).rot += (*p).rot_vel * update_params.time_delta;
        (*p).life -= update_params.time_delta;
        if ((*p).life < 0.0) {
            let list_index = atomicAdd(&free_list.count, 1);
            free_list.data[list_index] = global_id.x;
            (*p).scale = 0.0;
        }
    }
}

var<storage,read> draw_particles: array<Particle>;

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
    let particle = draw_particles[instance_index];
    var out: VertexOutput;
    let zero_one_pos = vec2<f32>(vec2<u32>(vertex_index&1u, vertex_index>>1u));
    let pt = Transform2D(particle.pos, particle.scale, particle.rot);
    let emitter_pos = transform(pt, 2.0 * zero_one_pos - vec2<f32>(1.0));
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
