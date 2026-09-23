use super::brdf::*;
use super::camera::*;
use super::config::*;
use super::debug::*;
use super::debug_param::*;
use super::env_importance::*;
use super::env_light::*;
use super::gbuf::*;
use super::hit::*;
use super::quaternion::*;
use super::random::*;
use super::sampling::*;
use super::surface::*;
use super::vertex::*;
use synaga_shader::*;

pub const MAX_RESERVOIRS: u32 = 4u32;

pub const DECOUPLED_SHADING: bool = false;

pub const FACTOR_CANDIDATES: u32 = 3u32;

#[derive(Clone, Copy, Default)]
pub struct MainParams {
    pub frame_index: u32,
    pub num_environment_samples: u32,
    pub num_brdf_samples: u32,
    pub environment_importance_sampling: u32,
    pub tap_count: u32,
    pub tap_radius: f32,
    pub tap_confidence_near: f32,
    pub tap_confidence_far: f32,
    pub t_start: f32,
    pub use_pairwise_mis: u32,
    pub defensive_mis: f32,
    pub use_motion_vectors: u32,
}

#[derive(Clone, Copy, Default)]
pub struct StoredReservoir {
    pub light_uv: vec2,
    pub light_index: u32,
    pub target_score: f32,
    pub contribution_weight: f32,
    pub confidence: f32,
}

#[derive(Clone, Copy, Default)]
pub struct Radiance {
    pub diffuse: vec3,
    pub specular: vec3,
}

#[derive(Clone, Copy, Default)]
pub struct LiveReservoir {
    pub selected_uv: vec2,
    pub selected_light_index: u32,
    pub selected_target_score: f32,
    pub selected_radiance: Radiance,
    pub weight_sum: f32,
    pub history: f32,
}

#[derive(Clone, Copy, Default)]
pub struct TargetScore {
    pub radiance: Radiance,
    pub score: f32,
}

#[derive(Clone, Copy, Default)]
pub struct RestirOutput {
    pub radiance: Radiance,
}

pub static camera: Uniform<CameraParams> = binding();

pub static prev_camera: Uniform<CameraParams> = binding();

pub static parameters: Uniform<MainParams> = binding();

pub static debug: Uniform<DebugParams> = binding();

pub static acc_struct: acceleration_structure = binding();

pub static prev_acc_struct: acceleration_structure = binding();

pub static mut reservoirs: StorageMut<[StoredReservoir]> = binding();

pub static prev_reservoirs: Storage<[StoredReservoir]> = binding();

pub static t_depth: texture_2d<f32> = binding();

pub static t_prev_depth: texture_2d<f32> = binding();

pub static t_basis: texture_2d<f32> = binding();

pub static t_prev_basis: texture_2d<f32> = binding();

pub static t_flat_normal: texture_2d<f32> = binding();

pub static t_prev_flat_normal: texture_2d<f32> = binding();

pub static t_diffuse_albedo: texture_2d<f32> = binding();

pub static t_prev_diffuse_albedo: texture_2d<f32> = binding();

pub static t_specular_f0: texture_2d<f32> = binding();

pub static t_prev_specular_f0: texture_2d<f32> = binding();

pub static t_motion: texture_2d<f32> = binding();

pub static out_diffuse: texture_storage_2d<Rgba16Float, Write> = binding();

pub static out_specular: texture_storage_2d<Rgba16Float, Write> = binding();

pub static out_debug: texture_storage_2d<Rgba8Unorm, Write> = binding();

pub static mut debug_len: Private<f32> = binding();

#[shader]
pub fn zero_radiance() -> Radiance {
    return Radiance {
        diffuse: vec3::splat(0.0),
        specular: vec3::splat(0.0),
    };
}

#[shader]
pub fn reflect_light(brdf: BrdfLobes, light: vec3) -> Radiance {
    return Radiance {
        diffuse: brdf.diffuse * light,
        specular: brdf.specular * light,
    };
}

#[shader]
pub fn compute_target_score(radiance: Radiance, diffuse_albedo: vec3) -> f32 {
    return compute_luminocity(diffuse_albedo * radiance.diffuse + radiance.specular);
}

#[shader]
pub fn get_reservoir_index(pixel: vec2i, cam: CameraParams) -> i32 {
    if (all(vec2u::from(pixel).cmplt(cam.target_size))) {
        return pixel.y * (cam.target_size.x) as i32 + pixel.x;
    } else {
        return -1;
    }
}

#[shader]
pub fn get_pixel_from_reservoir_index(index: i32, cam: CameraParams) -> vec2i {
    let y = index / (cam.target_size.x) as i32;
    let x = index - y * (cam.target_size.x) as i32;
    return vec2i(x, y);
}

#[shader]
pub fn bump_reservoir(r: &mut LiveReservoir, history: f32) {
    r.history += history;
}

#[shader]
pub fn merge_reservoir(r: &mut LiveReservoir, other: LiveReservoir, random: f32) -> bool {
    r.weight_sum += other.weight_sum;
    r.history += other.history;
    if (r.weight_sum * random < other.weight_sum) {
        r.selected_light_index = other.selected_light_index;
        r.selected_uv = other.selected_uv;
        r.selected_target_score = other.selected_target_score;
        r.selected_radiance = other.selected_radiance;
        return true;
    } else {
        return false;
    }
}

#[shader]
pub fn normalize_reservoir(r: &mut LiveReservoir, history: f32) {
    let h = r.history;
    if (h > 0.0) {
        r.weight_sum *= history / h;
        r.history = history;
    }
}

#[shader]
pub fn unpack_reservoir(
    f: StoredReservoir,
    max_confidence: f32,
    radiance: Radiance,
) -> LiveReservoir {
    let mut r = LiveReservoir::default();
    r.selected_light_index = f.light_index;
    r.selected_uv = f.light_uv;
    r.selected_target_score = f.target_score;
    r.selected_radiance = radiance;
    let history = min(f.confidence, max_confidence);
    r.weight_sum = f.contribution_weight * f.target_score * history;
    r.history = history;
    return r;
}

#[shader]
pub fn pack_reservoir_detail(r: LiveReservoir, denom_factor: f32) -> StoredReservoir {
    let mut f = StoredReservoir::default();
    f.light_index = r.selected_light_index;
    f.light_uv = r.selected_uv;
    f.target_score = r.selected_target_score;
    f.confidence = r.history;
    let denom = f.target_score * denom_factor;
    f.contribution_weight = select(0.0, r.weight_sum / denom, denom > 0.0);
    return f;
}

#[shader]
pub fn read_surface(pixel: vec2i) -> Surface {
    let mut surface = Surface::default();
    surface.basis = normalize(textureLoad(&t_basis, pixel, 0));
    surface.flat_normal = normalize(textureLoad(&t_flat_normal, pixel, 0).xyz());
    surface.depth = textureLoad(&t_depth, pixel, 0).x;
    surface.view_dir = -get_ray_direction(*camera, pixel);
    surface.diffuse_albedo = textureLoad(&t_diffuse_albedo, pixel, 0).xyz();
    let specular = textureLoad(&t_specular_f0, pixel, 0);
    surface.specular_f0 = specular.xyz();
    surface.roughness = specular.w;
    return surface;
}

#[shader]
pub fn read_prev_surface(pixel: vec2i) -> Surface {
    let mut surface = Surface::default();
    surface.basis = normalize(textureLoad(&t_prev_basis, pixel, 0));
    surface.flat_normal = normalize(textureLoad(&t_prev_flat_normal, pixel, 0).xyz());
    surface.depth = textureLoad(&t_prev_depth, pixel, 0).x;
    surface.view_dir = -get_ray_direction(*prev_camera, pixel);
    surface.diffuse_albedo = textureLoad(&t_prev_diffuse_albedo, pixel, 0).xyz();
    let specular = textureLoad(&t_prev_specular_f0, pixel, 0);
    surface.specular_f0 = specular.xyz();
    surface.roughness = specular.w;
    return surface;
}

#[shader]
pub fn surface_normal(surface: Surface) -> vec3 {
    return qrot(surface.basis, vec3(0.0, 0.0, 1.0));
}

#[shader]
pub fn surface_material(surface: Surface) -> Material {
    return Material {
        diffuse_albedo: surface.diffuse_albedo,
        specular_f0: surface.specular_f0,
        roughness: surface.roughness,
    };
}

#[shader]
pub fn evaluate_incoming_radiance(
    acs: acceleration_structure,
    position: vec3,
    direction: vec3,
    ray_len: f32,
    debug_color: u32,
) -> vec3 {
    let mut rq = ray_query::default();
    rayQueryInitialize(
        rq,
        &acs,
        RayDesc {
            flags: RAY_FLAG_CULL_NO_OPAQUE,
            cull_mask: 0xFFu32,
            tmin: parameters.t_start,
            tmax: camera.depth,
            origin: position,
            dir: direction,
        },
    );
    rayQueryProceed(rq);
    let intersection = rayQueryGetCommittedIntersection(rq);

    if (DEBUG_MODE && ray_len > 0.0) {
        let hit = intersection.kind != RAY_QUERY_INTERSECTION_NONE;
        let color = select(0xFFFFFFu32, 0x808080u32, hit) & debug_color;
        debug_line(position, position + ray_len * direction, color);
    }

    if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
        return evaluate_environment(direction);
    }

    let entry =
        hit_entries[(intersection.instance_custom_data + intersection.geometry_index) as usize];
    let indices = fetch_triangle_indices(entry, intersection.primitive_index);

    let barycentrics = make_barycentrics(intersection.barycentrics);
    let tex_coords = mat3x2(
        (vertex_buffers[(entry.vertex_buf) as usize].data)[(indices.x) as usize].tex_coords,
        (vertex_buffers[(entry.vertex_buf) as usize].data)[(indices.y) as usize].tex_coords,
        (vertex_buffers[(entry.vertex_buf) as usize].data)[(indices.z) as usize].tex_coords,
    ) * barycentrics;
    return sample_hit_emissive(entry, tex_coords, 0.0, 0u32);
}

#[shader]
pub fn get_prev_pixel(pixel: vec2i, pos_world: vec3) -> vec2 {
    if (USE_MOTION_VECTORS && parameters.use_motion_vectors != 0u32) {
        let motion = textureLoad(&t_motion, pixel, 0).xy() / MOTION_SCALE;
        return vec2::from(pixel) + 0.5 + motion;
    } else {
        return get_projected_pixel_float(*prev_camera, pos_world);
    }
}

#[shader]
pub fn ratio(a: f32, b: f32) -> f32 {
    return select(0.0, a / (a + b), a + b > 0.0);
}

#[shader]
pub fn zero_target_score() -> TargetScore {
    return TargetScore {
        radiance: zero_radiance(),
        score: 0.0,
    };
}

#[shader]
pub fn make_reservoir(
    ls: LightSample,
    light_index: u32,
    brdf: BrdfLobes,
    diffuse_albedo: vec3,
) -> LiveReservoir {
    let mut r = LiveReservoir::default();
    r.selected_radiance = reflect_light(brdf, ls.radiance);
    r.selected_uv = ls.uv;
    r.selected_light_index = light_index;
    r.selected_target_score = compute_target_score(r.selected_radiance, diffuse_albedo);
    r.weight_sum = select(0.0, r.selected_target_score / ls.pdf, ls.pdf > 0.0);
    r.history = 1.0;
    return r;
}

#[shader]
pub fn make_target_score(radiance: Radiance, diffuse_albedo: vec3) -> TargetScore {
    return TargetScore {
        radiance: radiance,
        score: compute_target_score(radiance, diffuse_albedo),
    };
}

#[shader]
pub fn pack_reservoir(r: LiveReservoir) -> StoredReservoir {
    return pack_reservoir_detail(r, r.history);
}

#[shader]
pub fn evaluate_surface_brdf(surface: Surface, dir: vec3) -> BrdfLobes {
    return evaluate_brdf(
        surface_material(surface),
        surface_normal(surface),
        surface.view_dir,
        dir,
    );
}

#[shader]
pub fn sample_incoming_light(
    surface: Surface,
    from_light: bool,
    rng: &mut RandomState,
) -> LightSample {
    let importance = parameters.environment_importance_sampling != 0u32;
    let mat = surface_material(surface);
    let normal = surface_normal(surface);

    let mut ls = LightSample::default();
    if (from_light) {
        ls = sample_light(importance, rng);
    } else {
        let bs = sample_bsdf(mat, normal, surface.view_dir, rng);
        ls.uv = map_equirect_dir_to_uv(bs.dir);
        ls.radiance = evaluate_environment(bs.dir);
    }

    let dir = map_equirect_uv_to_dir(ls.uv);
    let num_light = (parameters.num_environment_samples) as f32;
    let num_brdf = (parameters.num_brdf_samples) as f32;
    ls.pdf = (num_light * compute_light_pdf(ls.uv, importance)
        + num_brdf * compute_bsdf_pdf(mat, normal, surface.view_dir, dir))
        / max(num_light + num_brdf, 1.0);
    return ls;
}

#[shader]
pub fn estimate_target_score_with_occlusion(
    surface: Surface,
    position: vec3,
    light_index: u32,
    light_uv: vec2,
    acs: acceleration_structure,
    ray_len: f32,
    debug_color: u32,
) -> TargetScore {
    if (light_index != 0u32) {
        return zero_target_score();
    }
    let direction = map_equirect_uv_to_dir(light_uv);
    if (dot(direction, surface.flat_normal) <= 0.0) {
        return zero_target_score();
    }
    let brdf = evaluate_surface_brdf(surface, direction);
    if (is_brdf_black(brdf)) {
        return zero_target_score();
    }

    let radiance = evaluate_incoming_radiance(acs, position, direction, ray_len, debug_color);
    return make_target_score(reflect_light(brdf, radiance), surface.diffuse_albedo);
}

#[shader]
pub fn evaluate_sample(
    ls: &mut LightSample,
    surface: Surface,
    start_pos: vec3,
    ray_len: f32,
    debug_color: u32,
) -> BrdfLobes {
    let dir = map_equirect_uv_to_dir(ls.uv);
    if (dot(dir, surface.flat_normal) <= 0.0) {
        return zero_brdf();
    }

    let brdf = evaluate_surface_brdf(surface, dir);
    if (is_brdf_black(brdf)) {
        return zero_brdf();
    }

    // Evaluate the actual first-hit radiance.  Besides supporting emissive
    // geometry, this deliberately avoids the old absolute contribution
    // cutoff, which biased dim surfaces toward black.
    ls.radiance = evaluate_incoming_radiance(acc_struct, start_pos, dir, ray_len, debug_color);
    if (all(ls.radiance.cmple(vec3::splat(0.0)))) {
        return zero_brdf();
    }

    return brdf;
}

#[shader]
pub fn compute_restir(
    surface: Surface,
    pixel: vec2i,
    rng: &mut RandomState,
    enable_debug: bool,
) -> RestirOutput {
    let ray_dir = get_ray_direction(*camera, pixel);
    let pixel_index = get_reservoir_index(pixel, *camera);
    if (surface.depth == 0.0) {
        reservoirs[(pixel_index) as usize] = StoredReservoir::default();
        // Note: the diffuse albedo of the sky is 1.0, so the environment
        // survives the modulation in the post-processing.
        let env = evaluate_environment_background(ray_dir);
        return RestirOutput {
            radiance: Radiance {
                diffuse: env,
                specular: vec3::splat(0.0),
            },
        };
    }

    if (WRITE_DEBUG_IMAGE && debug.view_mode == DebugMode_Depth) {
        textureStore(&out_debug, pixel, vec4::splat(1.0 / surface.depth));
    }
    let position = camera.position + surface.depth * ray_dir;
    let ray_len = select(0.0, surface.depth * 0.2, enable_debug);

    let mut canonical = LiveReservoir::default();
    let num_initial = parameters.num_environment_samples + parameters.num_brdf_samples;
    for i in (0u32)..(num_initial) {
        let mut ls = sample_incoming_light(surface, i < parameters.num_environment_samples, rng);
        let brdf = evaluate_sample(&mut ls, surface, position, ray_len, 0x00FF00u32);
        if (is_brdf_black(brdf)) {
            bump_reservoir(&mut canonical, 1.0);
        } else {
            let other = make_reservoir(ls, 0u32, brdf, surface.diffuse_albedo);
            merge_reservoir(&mut canonical, other, random_gen(rng));
        }
    }

    let center_coord = get_prev_pixel(pixel, position);

    // First, gather the list of reservoirs to merge with
    let mut accepted_reservoir_indices = [0i32, 0i32, 0i32, 0i32];
    let mut accepted_count = 0u32;
    let max_samples = min(MAX_RESERVOIRS, parameters.tap_count);
    let num_candidates = max_samples * FACTOR_CANDIDATES;

    for tap in (0u32)..(num_candidates) {
        if (accepted_count >= max_samples) {
            break;
        }
        let radius = parameters.tap_radius * random_gen(rng);
        let offset = radius * sample_circle_uniform(random_gen(rng));
        let other_pixel = vec2i::from(center_coord + offset);

        let other_index = get_reservoir_index(other_pixel, *prev_camera);
        if (other_index < 0) {
            continue;
        }
        if (prev_reservoirs[(other_index) as usize].confidence == 0.0) {
            continue;
        }

        let other_surface = read_prev_surface(other_pixel);
        let compatibility = compare_surfaces(surface, other_surface);
        if (compatibility < 0.1) {
            // if the surfaces are too different, there is no trust in this sample
            continue;
        }

        accepted_reservoir_indices[(accepted_count) as usize] = other_index;
        accepted_count += 1u32;
    }

    if (WRITE_DEBUG_IMAGE && debug.view_mode == DebugMode_SampleReuse) {
        let mut color = vec4::splat(0.0);
        for i in (0u32)..(min(3u32, accepted_count)) {
            color[(i) as usize] = 1.0;
        }
        textureStore(&out_debug, pixel, color);
    }

    // Next, evaluate the MIS of each of the samples versus the canonical one.
    let mut reservoir = LiveReservoir::default();
    let mut shaded = zero_radiance();
    let mut shaded_weight = 0.0;
    let mis_scale = 1.0 / ((accepted_count) as f32 + parameters.defensive_mis);
    let mut mis_canonical = select(
        mis_scale * parameters.defensive_mis,
        1.0,
        accepted_count == 0u32 || parameters.use_pairwise_mis == 0u32,
    );
    let inv_count = 1.0 / (accepted_count) as f32;

    for rid in (0u32)..(accepted_count) {
        let neighbor_index = accepted_reservoir_indices[(rid) as usize];
        let neighbor = prev_reservoirs[(neighbor_index) as usize];
        let neighbor_pixel = get_pixel_from_reservoir_index(neighbor_index, *prev_camera);

        let offset = vec2::from(neighbor_pixel) - center_coord;
        let max_confidence = mix(
            parameters.tap_confidence_near,
            parameters.tap_confidence_far,
            length(offset) / parameters.tap_radius,
        );
        let mut other = LiveReservoir::default();
        if (parameters.use_pairwise_mis != 0u32) {
            let neighbor_history = min(neighbor.confidence, max_confidence);
            {
                // scoping this to hint the register allocation
                let neighbor_surface = read_prev_surface(neighbor_pixel);
                let neighbor_dir = get_ray_direction(*prev_camera, neighbor_pixel);
                let neighbor_position =
                    prev_camera.position + neighbor_surface.depth * neighbor_dir;

                let t_canonical_at_neighbor = estimate_target_score_with_occlusion(
                    neighbor_surface,
                    neighbor_position,
                    canonical.selected_light_index,
                    canonical.selected_uv,
                    prev_acc_struct,
                    ray_len,
                    0xFF0000u32,
                );
                let r_canonical = ratio(
                    canonical.history * canonical.selected_target_score * inv_count,
                    neighbor_history * t_canonical_at_neighbor.score,
                );
                mis_canonical += mis_scale * r_canonical;
            }

            let t_neighbor_at_canonical = estimate_target_score_with_occlusion(
                surface,
                position,
                neighbor.light_index,
                neighbor.light_uv,
                acc_struct,
                ray_len,
                0x0000FFu32,
            );
            let r_neighbor = ratio(
                neighbor_history * neighbor.target_score,
                canonical.history * t_neighbor_at_canonical.score * inv_count,
            );
            let mis_neighbor = mis_scale * r_neighbor;

            other.history = neighbor_history;
            other.selected_light_index = neighbor.light_index;
            other.selected_uv = neighbor.light_uv;
            other.selected_target_score = t_neighbor_at_canonical.score;
            other.selected_radiance = t_neighbor_at_canonical.radiance;
            other.weight_sum =
                t_neighbor_at_canonical.score * neighbor.contribution_weight * mis_neighbor;
        } else {
            // A reservoir stores the target score at the surface that created
            // it. Reusing that score at this surface gives a sample non-zero
            // selection weight even when it is occluded or its BRDF is black
            // here; if selected, that sample then shades to zero and creates
            // the persistent dark patches temporal reuse used to exhibit.
            // Re-evaluate both visibility and the target at the recipient.
            let evaluated = estimate_target_score_with_occlusion(
                surface,
                position,
                neighbor.light_index,
                neighbor.light_uv,
                acc_struct,
                ray_len,
                0x0000FFu32,
            );
            let history = min(neighbor.confidence, max_confidence);
            other.selected_light_index = neighbor.light_index;
            other.selected_uv = neighbor.light_uv;
            other.selected_target_score = evaluated.score;
            other.selected_radiance = evaluated.radiance;
            other.weight_sum = neighbor.contribution_weight * evaluated.score * history;
            other.history = history;
        }

        if (DECOUPLED_SHADING) {
            let scale = other.weight_sum * neighbor.contribution_weight;
            shaded.diffuse += scale * other.selected_radiance.diffuse;
            shaded.specular += scale * other.selected_radiance.specular;
            shaded_weight += other.weight_sum;
        }
        if (other.weight_sum <= 0.0) {
            bump_reservoir(&mut reservoir, other.history);
        } else {
            merge_reservoir(&mut reservoir, other, random_gen(rng));
        }
    }

    // Finally, merge in the canonical sample
    if (parameters.use_pairwise_mis != 0u32) {
        normalize_reservoir(&mut canonical, mis_canonical);
    }
    if (DECOUPLED_SHADING) {
        let cw = canonical.weight_sum / max(canonical.selected_target_score, 0.1);
        let scale = canonical.weight_sum * cw;
        shaded.diffuse += scale * canonical.selected_radiance.diffuse;
        shaded.specular += scale * canonical.selected_radiance.specular;
        shaded_weight += canonical.weight_sum;
    }
    merge_reservoir(&mut reservoir, canonical, random_gen(rng));

    let effective_history = select(reservoir.history, 1.0, parameters.use_pairwise_mis != 0u32);
    let stored = pack_reservoir_detail(reservoir, effective_history);
    reservoirs[(pixel_index) as usize] = stored;
    let mut ro = RestirOutput::default();
    if (DECOUPLED_SHADING) {
        let denom = max(shaded_weight, 0.001);
        ro.radiance = Radiance {
            diffuse: shaded.diffuse / denom,
            specular: shaded.specular / denom,
        };
    } else {
        let cw = stored.contribution_weight;
        ro.radiance = Radiance {
            diffuse: cw * reservoir.selected_radiance.diffuse,
            specular: cw * reservoir.selected_radiance.specular,
        };
    }
    return ro;
}

#[compute]
#[workgroup_size(8, 4)]
pub fn main(#[builtin(global_invocation_id)] global_id: vec3u) {
    if (any(global_id.xy().cmpge(camera.target_size))) {
        return;
    }

    let global_index = global_id.y * camera.target_size.x + global_id.x;
    let mut rng = random_init(global_index, parameters.frame_index);

    let surface = read_surface(vec2i::from(global_id.xy()));
    let enable_debug = DEBUG_MODE && all(global_id.xy().cmpeq(debug.mouse_pos));
    let enable_restir_debug = (debug.draw_flags & DebugDrawFlags_RESTIR) != 0u32 && enable_debug;
    let ro = compute_restir(
        surface,
        vec2i::from(global_id.xy()),
        &mut rng,
        enable_restir_debug,
    );

    if (enable_debug) {
        // Note: the variance is tracked on the fully modulated color
        let color = surface.diffuse_albedo * ro.radiance.diffuse + ro.radiance.specular;
        debug_buf.variance.color_sum += color;
        debug_buf.variance.color2_sum += color * color;
        debug_buf.variance.count += 1u32;
    }
    textureStore(
        &out_diffuse,
        global_id.xy(),
        (ro.radiance.diffuse).extend(1.0),
    );
    textureStore(
        &out_specular,
        global_id.xy(),
        (ro.radiance.specular).extend(1.0),
    );
}
