use super::brdf::{BrdfLobes, Material, compute_luminocity};
use super::camera::{CameraParams, get_ray_direction};
use super::config::{DEBUG_MODE, DebugDrawFlags, DebugMode, DebugTextureFlags};
use super::debug::{debug_buf, debug_line};
use super::debug_param::DebugParams;
use super::env_light::{
    LightSample, compute_light_pdf, evaluate_environment, evaluate_environment_background,
    map_equirect_dir_to_uv, map_equirect_uv_to_dir, sample_light,
};
use super::gbuf::{WRITE_DEBUG_IMAGE, get_prev_pixel, prev_camera};
use super::hit::{
    fetch_triangle_indices, hit_entries, make_barycentrics, sample_hit_emissive, vertex_buffers,
};
use super::quaternion::Quaternion;
use super::random::{RandomState, random_gen, random_init};
use super::sampling::{compute_bsdf_pdf, sample_bsdf, sample_circle_uniform};
use super::surface::{Surface, compare_surfaces};
use synaga_shader::*;

const MAX_RESERVOIRS: u32 = 4;
const DECOUPLED_SHADING: bool = false;
const FACTOR_CANDIDATES: u32 = 3;

#[repr(C)]
#[derive(Shared)]
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
struct StoredReservoir {
    pub light_uv: Vec2,
    pub light_index: u32,
    pub target_score: f32,
    pub contribution_weight: f32,
    pub confidence: f32,
}

#[derive(Clone, Copy, Default)]
struct Radiance {
    pub diffuse: Vec3,
    pub specular: Vec3,
}

#[derive(Clone, Copy, Default)]
struct LiveReservoir {
    pub selected_uv: Vec2,
    pub selected_light_index: u32,
    pub selected_target_score: f32,
    pub selected_radiance: Radiance,
    pub weight_sum: f32,
    pub history: f32,
}

#[derive(Clone, Copy, Default)]
struct TargetScore {
    pub radiance: Radiance,
    pub score: f32,
}

#[derive(Clone, Copy, Default)]
struct RestirOutput {
    pub radiance: Radiance,
}

static camera: Uniform<CameraParams> = binding();
static parameters: Uniform<MainParams> = binding();
static debug: Uniform<DebugParams> = binding();
static acc_struct: AccelerationStructure = binding();
static prev_acc_struct: AccelerationStructure = binding();
static reservoirs: StorageMut<[StoredReservoir]> = binding();
static prev_reservoirs: Storage<[StoredReservoir]> = binding();
static t_depth: Texture2D<f32> = binding();
static t_prev_depth: Texture2D<f32> = binding();
static t_basis: Texture2D<f32> = binding();
static t_prev_basis: Texture2D<f32> = binding();
static t_flat_normal: Texture2D<f32> = binding();
static t_prev_flat_normal: Texture2D<f32> = binding();
static t_diffuse_albedo: Texture2D<f32> = binding();
static t_prev_diffuse_albedo: Texture2D<f32> = binding();
static t_specular_f0: Texture2D<f32> = binding();
static t_prev_specular_f0: Texture2D<f32> = binding();
static out_diffuse: TextureStorage2D<Rgba16Float, Write> = binding();
static out_specular: TextureStorage2D<Rgba16Float, Write> = binding();
static out_debug: TextureStorage2D<Rgba8Unorm, Write> = binding();

fn divide_if_positive(numer: f32, denom: f32) -> f32 {
    if denom > 0.0 { numer / denom } else { 0.0 }
}

fn normalize_nonzero(v: Vec4) -> Vec4 {
    if v.dot(v) > 0.0 {
        v.normalize()
    } else {
        Vec4::ZERO
    }
}

fn normalize_nonzero3(v: Vec3) -> Vec3 {
    if v.dot(v) > 0.0 {
        v.normalize()
    } else {
        Vec3::ZERO
    }
}

fn zero_radiance() -> Radiance {
    Radiance {
        diffuse: Vec3::ZERO,
        specular: Vec3::ZERO,
    }
}

fn reflect_light(brdf: BrdfLobes, light: Vec3) -> Radiance {
    Radiance {
        diffuse: brdf.diffuse * light,
        specular: brdf.specular * light,
    }
}

fn compute_target_score(radiance: Radiance, diffuse_albedo: Vec3) -> f32 {
    compute_luminocity(diffuse_albedo * radiance.diffuse + radiance.specular)
}

fn get_reservoir_index(pixel: Vec2<i32>, cam: CameraParams) -> i32 {
    if pixel.cast::<u32>() < cam.target_size {
        pixel.y * cam.target_size.x as i32 + pixel.x
    } else {
        -1
    }
}

fn get_pixel_from_reservoir_index(index: i32, cam: CameraParams) -> Vec2<i32> {
    let y = index / cam.target_size.x as i32;
    let x = index - y * cam.target_size.x as i32;
    vec2(x, y)
}

fn bump_reservoir(r: &mut LiveReservoir, history: f32) {
    r.history += history;
}

fn merge_reservoir(r: &mut LiveReservoir, other: LiveReservoir, random: f32) -> bool {
    r.weight_sum += other.weight_sum;
    r.history += other.history;
    if r.weight_sum * random < other.weight_sum {
        r.selected_light_index = other.selected_light_index;
        r.selected_uv = other.selected_uv;
        r.selected_target_score = other.selected_target_score;
        r.selected_radiance = other.selected_radiance;
        true
    } else {
        false
    }
}

fn normalize_reservoir(r: &mut LiveReservoir, history: f32) {
    let h = r.history;
    if h > 0.0 {
        r.weight_sum *= history / h;
        r.history = history;
    }
}

fn pack_reservoir_detail(r: LiveReservoir, denom_factor: f32) -> StoredReservoir {
    let denom = r.selected_target_score * denom_factor;
    StoredReservoir {
        light_index: r.selected_light_index,
        light_uv: r.selected_uv,
        target_score: r.selected_target_score,
        confidence: r.history,
        // `select` evaluates both arms. A zero denominator is 0/0, and lavapipe
        // built with LLVM 22 keeps that NaN instead of the selected zero.
        contribution_weight: divide_if_positive(r.weight_sum, denom),
    }
}

fn read_surface(pixel: Vec2<i32>) -> Surface {
    let specular = t_specular_f0.load(pixel, 0);
    Surface {
        basis: normalize_nonzero(t_basis.load(pixel, 0)),
        flat_normal: normalize_nonzero3(t_flat_normal.load(pixel, 0).xyz()),
        depth: t_depth.load(pixel, 0).x,
        view_dir: -get_ray_direction(*camera, pixel),
        diffuse_albedo: t_diffuse_albedo.load(pixel, 0).xyz(),
        specular_f0: specular.xyz(),
        roughness: specular.w,
    }
}

fn read_prev_surface(pixel: Vec2<i32>) -> Surface {
    let specular = t_prev_specular_f0.load(pixel, 0);
    Surface {
        basis: normalize_nonzero(t_prev_basis.load(pixel, 0)),
        flat_normal: normalize_nonzero3(t_prev_flat_normal.load(pixel, 0).xyz()),
        depth: t_prev_depth.load(pixel, 0).x,
        view_dir: -get_ray_direction(*prev_camera, pixel),
        diffuse_albedo: t_prev_diffuse_albedo.load(pixel, 0).xyz(),
        specular_f0: specular.xyz(),
        roughness: specular.w,
    }
}

fn surface_normal(surface: Surface) -> Vec3 {
    surface.basis.rotate(vec3(0.0, 0.0, 1.0))
}

fn surface_material(surface: Surface) -> Material {
    Material {
        diffuse_albedo: surface.diffuse_albedo,
        specular_f0: surface.specular_f0,
        roughness: surface.roughness,
    }
}

fn evaluate_incoming_radiance(
    acs: AccelerationStructure,
    position: Vec3,
    direction: Vec3,
    ray_len: f32,
    debug_color: u32,
) -> Vec3 {
    let mut rq = RayQuery::default();
    rq.initialize(
        &acs,
        RayDesc {
            flags: RayFlag::CULL_NO_OPAQUE,
            cull_mask: 0xFF,
            tmin: parameters.t_start,
            tmax: camera.depth,
            origin: position,
            dir: direction,
        },
    );
    rq.proceed();
    let intersection = rq.committed_intersection();

    if DEBUG_MODE && ray_len > 0.0 {
        let hit = intersection.kind != RayQueryIntersection::None;
        let color = select(0xFFFFFFu32, 0x808080, hit) & debug_color;
        debug_line(position, position + ray_len * direction, color);
    }

    if intersection.kind == RayQueryIntersection::None {
        return evaluate_environment(direction);
    }

    let entry =
        hit_entries[(intersection.instance_custom_data + intersection.geometry_index) as usize];
    let indices = fetch_triangle_indices(entry, intersection.primitive_index);

    let barycentrics = make_barycentrics(intersection.barycentrics);
    let tex_coords = mat3x2(
        vertex_buffers[entry.vertex_buf as usize].data[indices.x as usize].tex_coords,
        vertex_buffers[entry.vertex_buf as usize].data[indices.y as usize].tex_coords,
        vertex_buffers[entry.vertex_buf as usize].data[indices.z as usize].tex_coords,
    ) * barycentrics;
    sample_hit_emissive(entry, tex_coords, 0.0, DebugTextureFlags::empty())
}

fn ratio(a: f32, b: f32) -> f32 {
    divide_if_positive(a, a + b)
}

fn zero_target_score() -> TargetScore {
    TargetScore {
        radiance: zero_radiance(),
        score: 0.0,
    }
}

fn make_reservoir(
    ls: LightSample,
    light_index: u32,
    brdf: BrdfLobes,
    diffuse_albedo: Vec3,
) -> LiveReservoir {
    let mut r = LiveReservoir::default();
    r.selected_radiance = reflect_light(brdf, ls.radiance);
    r.selected_uv = ls.uv;
    r.selected_light_index = light_index;
    r.selected_target_score = compute_target_score(r.selected_radiance, diffuse_albedo);
    r.weight_sum = divide_if_positive(r.selected_target_score, ls.pdf);
    r.history = 1.0;
    r
}

fn make_target_score(radiance: Radiance, diffuse_albedo: Vec3) -> TargetScore {
    TargetScore {
        radiance,
        score: compute_target_score(radiance, diffuse_albedo),
    }
}

fn evaluate_surface_brdf(surface: Surface, dir: Vec3) -> BrdfLobes {
    surface_material(surface).evaluate_brdf(surface_normal(surface), surface.view_dir, dir)
}

fn sample_incoming_light(surface: Surface, from_light: bool, rng: &mut RandomState) -> LightSample {
    let importance = parameters.environment_importance_sampling != 0;
    let mat = surface_material(surface);
    let normal = surface_normal(surface);

    let mut ls = LightSample::default();
    if from_light {
        ls = sample_light(importance, rng);
    } else {
        let bs = sample_bsdf(mat, normal, surface.view_dir, rng);
        ls.uv = map_equirect_dir_to_uv(bs.dir);
        ls.radiance = evaluate_environment(bs.dir);
    }

    let dir = map_equirect_uv_to_dir(ls.uv);
    let num_light = parameters.num_environment_samples as f32;
    let num_brdf = parameters.num_brdf_samples as f32;
    ls.pdf = (num_light * compute_light_pdf(ls.uv, importance)
        + num_brdf * compute_bsdf_pdf(mat, normal, surface.view_dir, dir))
        / (num_light + num_brdf).max(1.0);
    ls
}

fn estimate_target_score_with_occlusion(
    surface: Surface,
    position: Vec3,
    light_index: u32,
    light_uv: Vec2,
    acs: AccelerationStructure,
    ray_len: f32,
    debug_color: u32,
) -> TargetScore {
    if light_index != 0 {
        return zero_target_score();
    }
    let direction = map_equirect_uv_to_dir(light_uv);
    if direction.dot(surface.flat_normal) <= 0.0 {
        return zero_target_score();
    }
    let brdf = evaluate_surface_brdf(surface, direction);
    if brdf.is_black() {
        return zero_target_score();
    }

    let radiance = evaluate_incoming_radiance(acs, position, direction, ray_len, debug_color);
    make_target_score(reflect_light(brdf, radiance), surface.diffuse_albedo)
}

fn evaluate_sample(
    ls: &mut LightSample,
    surface: Surface,
    start_pos: Vec3,
    ray_len: f32,
    debug_color: u32,
) -> BrdfLobes {
    let dir = map_equirect_uv_to_dir(ls.uv);
    if dir.dot(surface.flat_normal) <= 0.0 {
        return BrdfLobes::default();
    }

    let brdf = evaluate_surface_brdf(surface, dir);
    if brdf.is_black() {
        return BrdfLobes::default();
    }

    // Evaluate the actual first-hit radiance.  Besides supporting emissive
    // geometry, this deliberately avoids the old absolute contribution
    // cutoff, which biased dim surfaces toward black.
    ls.radiance = evaluate_incoming_radiance(acc_struct, start_pos, dir, ray_len, debug_color);
    if ls.radiance <= Vec3::ZERO {
        return BrdfLobes::default();
    }

    brdf
}

fn compute_restir(
    surface: Surface,
    pixel: Vec2<i32>,
    rng: &mut RandomState,
    enable_debug: bool,
) -> RestirOutput {
    let ray_dir = get_ray_direction(*camera, pixel);
    let pixel_index = get_reservoir_index(pixel, *camera);
    if surface.depth == 0.0 {
        reservoirs.get_mut()[pixel_index as usize] = StoredReservoir::default();
        // Note: the diffuse albedo of the sky is 1.0, so the environment
        // survives the modulation in the post-processing.
        let env = evaluate_environment_background(ray_dir);
        return RestirOutput {
            radiance: Radiance {
                diffuse: env,
                specular: Vec3::ZERO,
            },
        };
    }

    if WRITE_DEBUG_IMAGE && debug.view_mode == DebugMode::Depth {
        out_debug.store(pixel, Vec4::splat(1.0 / surface.depth));
    }
    let position = camera.position + surface.depth * ray_dir;
    let ray_len = select(0.0, surface.depth * 0.2, enable_debug);

    let mut canonical = LiveReservoir::default();
    let num_initial = parameters.num_environment_samples + parameters.num_brdf_samples;
    for i in 0..num_initial {
        let mut ls = sample_incoming_light(surface, i < parameters.num_environment_samples, rng);
        let brdf = evaluate_sample(&mut ls, surface, position, ray_len, 0x00FF00);
        if brdf.is_black() {
            bump_reservoir(&mut canonical, 1.0);
        } else {
            let other = make_reservoir(ls, 0, brdf, surface.diffuse_albedo);
            merge_reservoir(&mut canonical, other, random_gen(rng));
        }
    }

    let center_coord = get_prev_pixel(pixel, position, parameters.use_motion_vectors != 0);

    // First, gather the list of reservoirs to merge with
    let mut accepted_reservoir_indices = [0i32, 0, 0, 0];
    let mut accepted_count = 0u32;
    let max_samples = MAX_RESERVOIRS.min(parameters.tap_count);
    let num_candidates = max_samples * FACTOR_CANDIDATES;

    for _ in 0..num_candidates {
        if accepted_count >= max_samples {
            break;
        }
        let radius = parameters.tap_radius * random_gen(rng);
        let offset = radius * sample_circle_uniform(random_gen(rng));
        let other_pixel = (center_coord + offset).cast::<i32>();

        let other_index = get_reservoir_index(other_pixel, *prev_camera);
        if other_index < 0 {
            continue;
        }
        if prev_reservoirs[other_index as usize].confidence == 0.0 {
            continue;
        }

        let other_surface = read_prev_surface(other_pixel);
        let compatibility = compare_surfaces(surface, other_surface);
        if compatibility < 0.1 {
            // if the surfaces are too different, there is no trust in this sample
            continue;
        }

        accepted_reservoir_indices[accepted_count as usize] = other_index;
        accepted_count += 1;
    }

    if WRITE_DEBUG_IMAGE && debug.view_mode == DebugMode::SampleReuse {
        let mut color = Vec4::ZERO;
        for i in 0..accepted_count.min(3) {
            color[i as usize] = 1.0;
        }
        out_debug.store(pixel, color);
    }

    // Next, evaluate the MIS of each of the samples versus the canonical one.
    let mut reservoir = LiveReservoir::default();
    let mut shaded = zero_radiance();
    let mut shaded_weight = 0.0;
    let mis_scale = 1.0 / (accepted_count as f32 + parameters.defensive_mis);
    let mut mis_canonical = select(
        mis_scale * parameters.defensive_mis,
        1.0,
        accepted_count == 0 || parameters.use_pairwise_mis == 0,
    );
    // No accepted neighbors means this factor is unused. Dividing by zero
    // here still produces an infinity that LLVM 22 folds into later results.
    let mut inv_count = 0.0;
    if accepted_count != 0 {
        inv_count = 1.0 / accepted_count as f32;
    }

    for rid in 0..accepted_count {
        let neighbor_index = accepted_reservoir_indices[rid as usize];
        let neighbor = prev_reservoirs[neighbor_index as usize];
        let neighbor_pixel = get_pixel_from_reservoir_index(neighbor_index, *prev_camera);

        let offset = Vec2::from(neighbor_pixel) - center_coord;
        let max_confidence = mix(
            parameters.tap_confidence_near,
            parameters.tap_confidence_far,
            offset.length() / parameters.tap_radius,
        );
        let mut other = LiveReservoir::default();
        if parameters.use_pairwise_mis != 0 {
            let neighbor_history = neighbor.confidence.min(max_confidence);
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
                    0xFF0000,
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
                0x0000FF,
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
                0x0000FF,
            );
            let history = neighbor.confidence.min(max_confidence);
            other.selected_light_index = neighbor.light_index;
            other.selected_uv = neighbor.light_uv;
            other.selected_target_score = evaluated.score;
            other.selected_radiance = evaluated.radiance;
            other.weight_sum = neighbor.contribution_weight * evaluated.score * history;
            other.history = history;
        }

        if DECOUPLED_SHADING {
            let scale = other.weight_sum * neighbor.contribution_weight;
            shaded.diffuse += scale * other.selected_radiance.diffuse;
            shaded.specular += scale * other.selected_radiance.specular;
            shaded_weight += other.weight_sum;
        }
        if other.weight_sum <= 0.0 {
            bump_reservoir(&mut reservoir, other.history);
        } else {
            merge_reservoir(&mut reservoir, other, random_gen(rng));
        }
    }

    // Finally, merge in the canonical sample
    if parameters.use_pairwise_mis != 0 {
        normalize_reservoir(&mut canonical, mis_canonical);
    }
    if DECOUPLED_SHADING {
        let cw = canonical.weight_sum / canonical.selected_target_score.max(0.1);
        let scale = canonical.weight_sum * cw;
        shaded.diffuse += scale * canonical.selected_radiance.diffuse;
        shaded.specular += scale * canonical.selected_radiance.specular;
        shaded_weight += canonical.weight_sum;
    }
    merge_reservoir(&mut reservoir, canonical, random_gen(rng));

    let effective_history = select(reservoir.history, 1.0, parameters.use_pairwise_mis != 0);
    let stored = pack_reservoir_detail(reservoir, effective_history);
    reservoirs.get_mut()[pixel_index as usize] = stored;
    let mut ro = RestirOutput::default();
    if DECOUPLED_SHADING {
        let denom = shaded_weight.max(0.001);
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
    ro
}

#[entry_point(compute, threads(8, 4))]
fn main(global_invocation_id: Vec3<u32>) {
    if global_invocation_id.xy().cmpge(camera.target_size).any() {
        return;
    }

    let global_index = global_invocation_id.y * camera.target_size.x + global_invocation_id.x;
    let mut rng = random_init(global_index, parameters.frame_index);

    let surface = read_surface(global_invocation_id.xy().cast::<i32>());
    let enable_debug = DEBUG_MODE && global_invocation_id.xy() == debug.mouse_pos;
    let enable_restir_debug = debug.draw_flags.contains(DebugDrawFlags::RESTIR) && enable_debug;
    let ro = compute_restir(
        surface,
        global_invocation_id.xy().cast::<i32>(),
        &mut rng,
        enable_restir_debug,
    );

    if enable_debug {
        // Note: the variance is tracked on the fully modulated color
        let color = surface.diffuse_albedo * ro.radiance.diffuse + ro.radiance.specular;
        debug_buf.get_mut().variance.color_sum += color;
        debug_buf.get_mut().variance.color2_sum += color * color;
        debug_buf.get_mut().variance.count += 1;
    }
    out_diffuse.store(global_invocation_id.xy(), ro.radiance.diffuse.extend(1.0));
    out_specular.store(global_invocation_id.xy(), ro.radiance.specular.extend(1.0));
}
