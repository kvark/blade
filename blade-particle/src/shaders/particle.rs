use synaga_shader::*;

#[derive(Clone, Copy, Default)]
struct Particle {
    pub pos: Vec3,
    pub scale: f32,
    pub color: u32,
    pub vel: Vec3,
    pub life: f32,
    pub max_life: f32,
    pub generation: u32,
}

struct FreeList {
    pub count: AtomicI32,
    pub data: [u32],
}

#[derive(Clone, Copy, Default)]
struct EmitParams {
    pub origin: Vec3,
    pub emitter_radius: f32,
    pub direction: Vec3,
    pub cone_half_angle_cos: f32,
    pub colors: Vec4<u32>,
    pub color_count: u32,
    pub emit_count: u32,
    pub life_min: f32,
    pub life_max: f32,
    pub speed_min: f32,
    pub speed_max: f32,
    pub scale_min: f32,
    pub scale_max: f32,
}

#[derive(Clone, Copy, Default)]
struct UpdateParams {
    pub time_delta: f32,
}

#[derive(Clone, Copy, Default)]
struct CameraParams {
    pub view_proj: Mat4,
    pub camera_right: Vec4,
    pub camera_up: Vec4,
}

#[derive(Clone, Copy, Debug, Default, Io)]
struct VertexOutput {
    #[builtin(position)]
    proj_pos: Vec4,
    #[location(0)]
    color: Vec4,
    #[location(1)]
    uv: Vec2,
}

static particles: StorageMut<[Particle]> = binding();
static free_list: StorageMut<FreeList> = binding();
static emit_params: Uniform<EmitParams> = binding();
static update_params: Uniform<UpdateParams> = binding();
static emit_end: Workgroup<i32> = binding();
static draw_particles: Storage<[Particle]> = binding();
static camera: Uniform<CameraParams> = binding();

#[entry_point(compute, threads(64, 1, 1))]
fn reset(
    #[builtin(global_invocation_id)] global_id: Vec3<u32>,
    #[builtin(num_workgroups)] num_groups: Vec3<u32>,
) {
    let total = num_groups.x * 64u32;
    // reversing the order because it works like a stack
    let mut p = Particle::default();
    unsafe {
        free_list.get_mut().data[(global_id.x) as usize] = total - 1u32 - global_id.x;
        particles.get_mut()[(global_id.x) as usize] = p;
    }
    if global_id.x == 0u32 {
        free_list.count.store((total) as i32);
    }
}

fn hash_u32(x: u32) -> u32 {
    let mut h = x;
    h = h ^ (h >> 16u32);
    h = h * 0x45d9f3bu32;
    h = h ^ (h >> 16u32);
    h = h * 0x45d9f3bu32;
    h = h ^ (h >> 16u32);
    return h;
}

fn rotate_to(to: Vec3, v: Vec3) -> Vec3 {
    // d = dot(+Z, to) = to.z
    let d = to.z;
    if d > 0.9999 {
        return v;
    }
    if d < -0.9999 {
        return vec3(v.x, -v.y, -v.z);
    }
    // cross(+Z, to) = (-to.y, to.x, 0)
    let a = normalize(vec3(-to.y, to.x, 0.0));
    let s = sqrt(1.0 - d * d);
    // Rodrigues rotation
    return v * d + cross(a, v) * s + a * dot(a, v) * (1.0 - d);
}

#[entry_point(compute, threads(64, 1, 1))]
fn update(#[builtin(global_invocation_id)] global_id: Vec3<u32>) {
    if (particles[(global_id.x) as usize]).scale != 0.0 {
        let index = (global_id.x) as usize;
        unsafe {
            particles.get_mut()[index].pos += particles[index].vel * update_params.time_delta;
            particles.get_mut()[index].life -= update_params.time_delta;
            if particles[index].life < 0.0 {
                let list_index = free_list.count.fetch_add(1);
                free_list.get_mut().data[(list_index) as usize] = global_id.x;
                particles.get_mut()[index].scale = 0.0;
            }
        }
    }
}

#[entry_point(vertex)]
fn draw_vs(
    #[builtin(vertex_index)] vertex_index: u32,
    #[builtin(instance_index)] instance_index: u32,
) -> VertexOutput {
    let particle = draw_particles[(instance_index) as usize];
    let mut out = VertexOutput::default();

    if particle.scale == 0.0 {
        out.proj_pos = vec4(0.0, 0.0, -1.0, 1.0);
        out.color = Vec4::splat(0.0);
        out.uv = Vec2::splat(0.0);
        return out;
    }

    // Billboard: offset particle position in world space along camera axes
    let zero_one = Vec2::from(vec2::<u32>(vertex_index & 1u32, vertex_index >> 1u32));
    let offset = 2.0 * zero_one - Vec2::splat(1.0);
    let world_pos = particle.pos
        + camera.camera_right.xyz() * (offset.x * particle.scale)
        + camera.camera_up.xyz() * (offset.y * particle.scale);

    // Project to clip space
    out.proj_pos = camera.view_proj * (world_pos).extend(1.0);

    // Unpack base color and apply lifetime fade
    let base_color = unpack4x8unorm(particle.color);
    let age = 1.0 - particle.life / particle.max_life;
    // Fade out alpha over lifetime
    let alpha = base_color.a() * (1.0 - age * age);
    out.color = (base_color.rgb()).extend(alpha);
    out.uv = 2.0 * zero_one - Vec2::splat(1.0);
    return out;
}

#[entry_point(fragment)]
#[output(location(0))]
fn draw_fs(input: VertexOutput) -> Vec4 {
    // Soft circular particle: smooth falloff from center
    let dist_sq = dot(input.uv, input.uv);
    if dist_sq > 1.0 {
        discard();
    }
    let softness = 1.0 - dist_sq;
    return (input.color.rgb()).extend(input.color.a() * softness);
}

fn rand01(seed: u32) -> f32 {
    return (hash_u32(seed) & 0xFFFFu32) as f32 / 65535.0;
}

#[entry_point(compute, threads(64, 1, 1))]
fn emit(#[builtin(local_invocation_index)] local_index: u32) {
    let count = (emit_params.emit_count) as i32;
    if local_index == 0u32 {
        unsafe {
            *emit_end.get_mut() = free_list.count.fetch_sub(count);
        }
        if *emit_end < count {
            free_list.count.fetch_add(count - max(0, *emit_end));
        }
    }
    workgroup_barrier();

    let my_index = (local_index) as i32;
    let list_index = *emit_end - 1 - my_index;
    if my_index >= count || list_index < 0 {
        return;
    }

    let p_index = free_list.data[(list_index) as usize];
    let mut p = Particle::default();
    p.generation += 1u32;

    let seed = p_index * 1337u32 + p.generation * 7919u32;
    let r0 = rand01(seed);
    let r1 = rand01(seed + 1u32);
    let r2 = rand01(seed + 2u32);
    let r3 = rand01(seed + 3u32);
    let r4 = rand01(seed + 4u32);
    let r5 = rand01(seed + 5u32);

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
    let local_dir = vec3(sin_phi * cos(theta), sin_phi * sin(theta), cos_phi);
    // Rotate from +Z to emit_params.direction
    let dir = rotate_to(emit_params.direction, local_dir);
    p.vel = speed * dir;

    // Position: origin + shape offset
    p.pos = emit_params.origin + dir * emit_params.emitter_radius;

    // Pick color from palette
    let ci = (r5 * (emit_params.color_count) as f32) as u32 % emit_params.color_count;
    p.color = emit_params.colors[(ci) as usize];

    unsafe {
        particles.get_mut()[(p_index) as usize] = p;
    }
}
