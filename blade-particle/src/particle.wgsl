struct Particle {
    pos: vec3<f32>,
    scale: f32,
    color: u32,
    vel: vec3<f32>,
    life: f32,
    max_life: f32,
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

struct EmitParams {
    origin: vec3<f32>,
    emitter_radius: f32,
    direction: vec3<f32>,
    cone_half_angle_cos: f32,
    colors: vec4<u32>,
    color_count: u32,
    emit_count: u32,
    life_min: f32,
    life_max: f32,
    speed_min: f32,
    speed_max: f32,
    scale_min: f32,
    scale_max: f32,
}
var<uniform> emit_params: EmitParams;

struct UpdateParams {
    time_delta: f32,
}
var<uniform> update_params: UpdateParams;

fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x45d9f3bu;
    h = h ^ (h >> 16u);
    h = h * 0x45d9f3bu;
    h = h ^ (h >> 16u);
    return h;
}

fn rand01(seed: u32) -> f32 {
    return f32(hash_u32(seed) & 0xFFFFu) / 65535.0;
}

/// Rotate a vector from +Z axis to the given axis direction.
fn rotate_to(to: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    // d = dot(+Z, to) = to.z
    let d = to.z;
    if (d > 0.9999) {
        return v;
    }
    if (d < -0.9999) {
        return vec3<f32>(v.x, -v.y, -v.z);
    }
    // cross(+Z, to) = (-to.y, to.x, 0)
    let a = normalize(vec3<f32>(-to.y, to.x, 0.0));
    let s = sqrt(1.0 - d * d);
    // Rodrigues rotation
    return v * d + cross(a, v) * s + a * dot(a, v) * (1.0 - d);
}

var<workgroup> emit_end: i32;

@compute @workgroup_size(64, 1, 1)
fn emit(@builtin(local_invocation_index) local_index: u32) {
    let count = i32(emit_params.emit_count);
    if (local_index == 0u) {
        emit_end = atomicSub(&free_list.count, count);
        if (emit_end < count) {
            atomicAdd(&free_list.count, count - max(0, emit_end));
        }
    }
    workgroupBarrier();

    let my_index = i32(local_index);
    let list_index = emit_end - 1 - my_index;
    if (my_index >= count || list_index < 0) {
        return;
    }

    let p_index = free_list.data[list_index];
    var p: Particle;
    p.generation += 1u;

    let seed = p_index * 1337u + p.generation * 7919u;
    let r0 = rand01(seed);
    let r1 = rand01(seed + 1u);
    let r2 = rand01(seed + 2u);
    let r3 = rand01(seed + 3u);
    let r4 = rand01(seed + 4u);
    let r5 = rand01(seed + 5u);

    p.life = mix(emit_params.life_min, emit_params.life_max, r0);
    p.max_life = p.life;
    p.scale = mix(emit_params.scale_min, emit_params.scale_max, r1);
    let speed = mix(emit_params.speed_min, emit_params.speed_max, r2);

    // Random direction in a cone around emit_params.direction.
    // cos_phi is uniformly distributed in [cone_half_angle_cos, 1].
    let theta = r3 * 6.283185;
    let cos_phi = mix(1.0, emit_params.cone_half_angle_cos, r4);
    let sin_phi = sqrt(1.0 - cos_phi * cos_phi);
    // Local direction with cone axis = +Z
    let local_dir = vec3<f32>(sin_phi * cos(theta), sin_phi * sin(theta), cos_phi);
    // Rotate from +Z to emit_params.direction
    let dir = rotate_to(emit_params.direction, local_dir);
    p.vel = speed * dir;

    // Position: origin + shape offset
    p.pos = emit_params.origin + dir * emit_params.emitter_radius;

    // Pick color from palette
    let ci = u32(r5 * f32(emit_params.color_count)) % emit_params.color_count;
    p.color = emit_params.colors[ci];

    particles[p_index] = p;
}

@compute @workgroup_size(64, 1, 1)
fn update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let p = &particles[global_id.x];
    if ((*p).scale != 0.0) {
        (*p).pos += (*p).vel * update_params.time_delta;
        (*p).life -= update_params.time_delta;
        if ((*p).life < 0.0) {
            let list_index = atomicAdd(&free_list.count, 1);
            free_list.data[list_index] = global_id.x;
            (*p).scale = 0.0;
        }
    }
}

// Draw pass

var<storage,read> draw_particles: array<Particle>;

struct CameraParams {
    view_proj: mat4x4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
}
var<uniform> camera: CameraParams;

struct VertexOutput {
    @builtin(position) proj_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn draw_vs(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let particle = draw_particles[instance_index];
    var out: VertexOutput;

    if (particle.scale == 0.0) {
        out.proj_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        out.color = vec4<f32>(0.0);
        out.uv = vec2<f32>(0.0);
        return out;
    }

    // Billboard: offset particle position in world space along camera axes
    let zero_one = vec2<f32>(vec2<u32>(vertex_index & 1u, vertex_index >> 1u));
    let offset = 2.0 * zero_one - vec2<f32>(1.0);
    let world_pos = particle.pos
        + camera.camera_right.xyz * (offset.x * particle.scale)
        + camera.camera_up.xyz * (offset.y * particle.scale);

    // Project to clip space
    out.proj_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);

    // Unpack base color and apply lifetime fade
    let base_color = unpack4x8unorm(particle.color);
    let age = 1.0 - particle.life / particle.max_life;
    // Fade out alpha over lifetime
    let alpha = base_color.a * (1.0 - age * age);
    out.color = vec4<f32>(base_color.rgb, alpha);
    out.uv = 2.0 * zero_one - vec2<f32>(1.0);
    return out;
}

@fragment
fn draw_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    // Soft circular particle: smooth falloff from center
    let dist_sq = dot(in.uv, in.uv);
    if (dist_sq > 1.0) {
        discard;
    }
    let softness = 1.0 - dist_sq;
    return vec4<f32>(in.color.rgb, in.color.a * softness);
}
