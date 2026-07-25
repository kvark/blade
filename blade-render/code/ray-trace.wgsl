enable wgpu_ray_query;
#include "quaternion.inc.wgsl"
#include "random.inc.wgsl"
#include "env-importance.inc.wgsl"
#include "debug.inc.wgsl"
#include "debug-param.inc.wgsl"
#include "camera.inc.wgsl"
#include "brdf.inc.wgsl"
#include "sampling.inc.wgsl"
#include "env-light.inc.wgsl"
#include "surface.inc.wgsl"
#include "gbuf.inc.wgsl"

const MAX_RESERVOIRS: u32 = 4u;
// See "DECOUPLING SHADING AND REUSE" in
// "Rearchitecting Spatiotemporal Resampling for Production"
const DECOUPLED_SHADING: bool = false;

// How many more candidates to consder than the taps we need
const FACTOR_CANDIDATES: u32 = 3u;

struct MainParams {
    frame_index: u32,
    num_environment_samples: u32,
    environment_importance_sampling: u32,
    tap_count: u32,
    tap_radius: f32,
    tap_confidence_near: f32,
    tap_confidence_far: f32,
    t_start: f32,
    use_pairwise_mis: u32,
    defensive_mis: f32,
    use_motion_vectors: u32,
};

var<uniform> camera: CameraParams;
var<uniform> prev_camera: CameraParams;
var<uniform> parameters: MainParams;
var<uniform> debug: DebugParams;
var acc_struct: acceleration_structure;
var prev_acc_struct: acceleration_structure;
var env_map: texture_2d<f32>;
var sampler_linear: sampler;
var sampler_nearest: sampler;

struct StoredReservoir {
    light_uv: vec2<f32>,
    light_index: u32,
    target_score: f32,
    contribution_weight: f32,
    confidence: f32,
}
var<storage, read_write> reservoirs: array<StoredReservoir>;
var<storage, read> prev_reservoirs: array<StoredReservoir>;

// Reflected light, separated into the lobes that we estimate
// and denoise independently.
//
// Note: the diffuse part is demodulated, it has to be multiplied
// by the diffuse albedo of the surface.
struct Radiance {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

fn zero_radiance() -> Radiance {
    return Radiance(vec3<f32>(0.0), vec3<f32>(0.0));
}
fn reflect_light(brdf: BrdfLobes, light: vec3<f32>) -> Radiance {
    return Radiance(brdf.diffuse * light, brdf.specular * light);
}

struct LiveReservoir {
    selected_uv: vec2<f32>,
    selected_light_index: u32,
    selected_target_score: f32,
    selected_radiance: Radiance,
    weight_sum: f32,
    history: f32,
}

// Note: the target function includes both of the lobes, so the diffuse albedo
// of the surface is needed to bring the diffuse one back into radiance.
fn compute_target_score(radiance: Radiance, diffuse_albedo: vec3<f32>) -> f32 {
    return compute_luminocity(diffuse_albedo * radiance.diffuse + radiance.specular);
}

fn get_reservoir_index(pixel: vec2<i32>, camera: CameraParams) -> i32 {
    if (all(vec2<u32>(pixel) < camera.target_size)) {
        return pixel.y * i32(camera.target_size.x) + pixel.x;
    } else {
        return -1;
    }
}

fn get_pixel_from_reservoir_index(index: i32, camera: CameraParams) -> vec2<i32> {
    let y = index / i32(camera.target_size.x);
    let x = index - y * i32(camera.target_size.x);
    return vec2<i32>(x, y);
}

fn bump_reservoir(r: ptr<function, LiveReservoir>, history: f32) {
    (*r).history += history;
}
fn make_reservoir(ls: LightSample, light_index: u32, brdf: BrdfLobes, diffuse_albedo: vec3<f32>) -> LiveReservoir {
    var r: LiveReservoir;
    r.selected_radiance = reflect_light(brdf, ls.radiance);
    r.selected_uv = ls.uv;
    r.selected_light_index = light_index;
    r.selected_target_score = compute_target_score(r.selected_radiance, diffuse_albedo);
    r.weight_sum = select(0.0, r.selected_target_score / ls.pdf, ls.pdf > 0.0);
    r.history = 1.0;
    return r;
}
fn merge_reservoir(r: ptr<function, LiveReservoir>, other: LiveReservoir, random: f32) -> bool {
    (*r).weight_sum += other.weight_sum;
    (*r).history += other.history;
    if ((*r).weight_sum * random < other.weight_sum) {
        (*r).selected_light_index = other.selected_light_index;
        (*r).selected_uv = other.selected_uv;
        (*r).selected_target_score = other.selected_target_score;
        (*r).selected_radiance = other.selected_radiance;
        return true;
    } else {
        return false;
    }
}
fn normalize_reservoir(r: ptr<function, LiveReservoir>, history: f32) {
    let h = (*r).history;
    if (h > 0.0) {
        (*r).weight_sum *= history / h;
        (*r).history = history;
    }
}
fn unpack_reservoir(f: StoredReservoir, max_confidence: f32, radiance: Radiance) -> LiveReservoir {
    var r: LiveReservoir;
    r.selected_light_index = f.light_index;
    r.selected_uv = f.light_uv;
    r.selected_target_score = f.target_score;
    r.selected_radiance = radiance;
    let history = min(f.confidence, max_confidence);
    r.weight_sum = f.contribution_weight * f.target_score * history;
    r.history = history;
    return r;
}
fn pack_reservoir_detail(r: LiveReservoir, denom_factor: f32) -> StoredReservoir {
    var f: StoredReservoir;
    f.light_index = r.selected_light_index;
    f.light_uv = r.selected_uv;
    f.target_score = r.selected_target_score;
    f.confidence = r.history;
    let denom = f.target_score * denom_factor;
    f.contribution_weight = select(0.0, r.weight_sum / denom, denom > 0.0);
    return f;
}

fn pack_reservoir(r: LiveReservoir) -> StoredReservoir {
    return pack_reservoir_detail(r, r.history);
}

var t_depth: texture_2d<f32>;
var t_prev_depth: texture_2d<f32>;
var t_basis: texture_2d<f32>;
var t_prev_basis: texture_2d<f32>;
var t_flat_normal: texture_2d<f32>;
var t_prev_flat_normal: texture_2d<f32>;
var t_diffuse_albedo: texture_2d<f32>;
var t_prev_diffuse_albedo: texture_2d<f32>;
var t_specular_f0: texture_2d<f32>;
var t_prev_specular_f0: texture_2d<f32>;
var t_motion: texture_2d<f32>;
var out_diffuse: texture_storage_2d<rgba16float, write>;
var out_specular: texture_storage_2d<rgba16float, write>;
var out_debug: texture_storage_2d<rgba8unorm, write>;

fn read_surface(pixel: vec2<i32>) -> Surface {
    var surface: Surface;
    surface.basis = normalize(textureLoad(t_basis, pixel, 0));
    surface.flat_normal = normalize(textureLoad(t_flat_normal, pixel, 0).xyz);
    surface.depth = textureLoad(t_depth, pixel, 0).x;
    surface.view_dir = -get_ray_direction(camera, pixel);
    surface.diffuse_albedo = textureLoad(t_diffuse_albedo, pixel, 0).xyz;
    let specular = textureLoad(t_specular_f0, pixel, 0);
    surface.specular_f0 = specular.xyz;
    surface.roughness = specular.w;
    return surface;
}

fn read_prev_surface(pixel: vec2<i32>) -> Surface {
    var surface: Surface;
    surface.basis = normalize(textureLoad(t_prev_basis, pixel, 0));
    surface.flat_normal = normalize(textureLoad(t_prev_flat_normal, pixel, 0).xyz);
    surface.depth = textureLoad(t_prev_depth, pixel, 0).x;
    surface.view_dir = -get_ray_direction(prev_camera, pixel);
    surface.diffuse_albedo = textureLoad(t_prev_diffuse_albedo, pixel, 0).xyz;
    let specular = textureLoad(t_prev_specular_f0, pixel, 0);
    surface.specular_f0 = specular.xyz;
    surface.roughness = specular.w;
    return surface;
}

fn surface_normal(surface: Surface) -> vec3<f32> {
    return qrot(surface.basis, vec3<f32>(0.0, 0.0, 1.0));
}

fn surface_material(surface: Surface) -> Material {
    return Material(surface.diffuse_albedo, surface.specular_f0, surface.roughness);
}

// Note: the diffuse lobe isn't modulated by the albedo here,
// see `Radiance` for the reasoning.
fn evaluate_surface_brdf(surface: Surface, dir: vec3<f32>) -> BrdfLobes {
    return evaluate_brdf(surface_material(surface), surface_normal(surface), surface.view_dir, dir);
}

// Portion of the candidates that follow the BRDF instead of the light.
//
// Rough diffuse surfaces are served well by sampling the light, while
// a narrow or dominant specular lobe needs to be sampled directly.
fn compute_brdf_sampling_ratio(surface: Surface) -> f32 {
    let mat = surface_material(surface);
    let smoothness = 1.0 - clamp(mat.roughness, 0.0, 1.0);
    return clamp(max(specular_sampling_ratio(mat), smoothness * smoothness), 0.1, 0.9);
}

// Draw a candidate following either the light distribution or the BRDF.
//
// The returned density is the one of the mixture of both strategies,
// which is the balance heuristic MIS weight for a single sample.
fn sample_incoming_light(surface: Surface, rng: ptr<function, RandomState>) -> LightSample {
    let importance = parameters.environment_importance_sampling != 0u;
    let mat = surface_material(surface);
    let normal = surface_normal(surface);
    let brdf_ratio = compute_brdf_sampling_ratio(surface);

    var ls: LightSample;
    if (random_gen(rng) < brdf_ratio) {
        let bs = sample_bsdf(mat, normal, surface.view_dir, rng);
        ls.uv = map_equirect_dir_to_uv(bs.dir);
        ls.radiance = evaluate_environment(bs.dir);
    } else {
        ls = sample_light(importance, rng);
    }

    let dir = map_equirect_uv_to_dir(ls.uv);
    ls.pdf = mix(
        compute_light_pdf(ls.uv, importance),
        compute_bsdf_pdf(mat, normal, surface.view_dir, dir),
        brdf_ratio,
    );
    return ls;
}

var<private> debug_len: f32;

fn check_ray_occluded(acs: acceleration_structure, position: vec3<f32>, direction: vec3<f32>, debug_len: f32, debug_color: u32) -> bool {
    var rq: ray_query;
    let flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_CULL_NO_OPAQUE;
    rayQueryInitialize(&rq, acs,
        RayDesc(flags, 0xFFu, parameters.t_start, camera.depth, position, direction)
    );
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);

    let occluded = intersection.kind != RAY_QUERY_INTERSECTION_NONE;
    if (DEBUG_MODE && debug_len > 0.0) {
        let color = select(0xFFFFFFu, 0x808080u, occluded) & debug_color;
        debug_line(position, position + debug_len * direction, color);
    }
    return occluded;
}

fn evaluate_reflected_light(surface: Surface, light_index: u32, light_uv: vec2<f32>) -> Radiance {
    if (light_index != 0u) {
        return zero_radiance();
    }
    let direction = map_equirect_uv_to_dir(light_uv);
    let brdf = evaluate_surface_brdf(surface, direction);
    if (is_brdf_black(brdf)) {
        return zero_radiance();
    }
    let radiance = textureSampleLevel(env_map, sampler_nearest, light_uv, 0.0).xyz;
    return reflect_light(brdf, radiance);
}

fn get_prev_pixel(pixel: vec2<i32>, pos_world: vec3<f32>) -> vec2<f32> {
    if (USE_MOTION_VECTORS && parameters.use_motion_vectors != 0u) {
        let motion = textureLoad(t_motion, pixel, 0).xy / MOTION_SCALE;
        return vec2<f32>(pixel) + 0.5 + motion;
    } else {
        return get_projected_pixel_float(prev_camera, pos_world);
    }
}

struct TargetScore {
    radiance: Radiance,
    score: f32,
}

fn zero_target_score() -> TargetScore {
    return TargetScore(zero_radiance(), 0.0);
}

fn make_target_score(radiance: Radiance, diffuse_albedo: vec3<f32>) -> TargetScore {
    return TargetScore(radiance, compute_target_score(radiance, diffuse_albedo));
}

fn estimate_target_score_with_occlusion(
    surface: Surface, position: vec3<f32>, light_index: u32, light_uv: vec2<f32>, acs: acceleration_structure,
    debug_len: f32, debug_color: u32,
) -> TargetScore {
    if (light_index != 0u) {
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

    if (check_ray_occluded(acs, position, direction, debug_len, debug_color)) {
        return zero_target_score();
    } else {
        //Note: same as `evaluate_reflected_light`
        let radiance = textureSampleLevel(env_map, sampler_nearest, light_uv, 0.0).xyz;
        return make_target_score(reflect_light(brdf, radiance), surface.diffuse_albedo);
    }
}

fn evaluate_sample(ls: LightSample, surface: Surface, start_pos: vec3<f32>, debug_len: f32, debug_color: u32) -> BrdfLobes {
    let dir = map_equirect_uv_to_dir(ls.uv);
    if (dot(dir, surface.flat_normal) <= 0.0) {
        return zero_brdf();
    }

    let brdf = evaluate_surface_brdf(surface, dir);
    if (is_brdf_black(brdf)) {
        return zero_brdf();
    }

    // Don't spend a ray on the samples that can't contribute much.
    // Note: this is the weight the sample would get in the reservoir.
    let target_score = compute_target_score(reflect_light(brdf, ls.radiance), surface.diffuse_albedo);
    if (target_score < 0.01 * ls.pdf) {
        return zero_brdf();
    }

    if (check_ray_occluded(acc_struct, start_pos, dir, debug_len, debug_color)) {
        return zero_brdf();
    }

    return brdf;
}

fn ratio(a: f32, b: f32) -> f32 {
    return select(0.0, a / (a+b), a+b > 0.0);
}

struct RestirOutput {
    radiance: Radiance,
}

fn compute_restir(surface: Surface, pixel: vec2<i32>, rng: ptr<function, RandomState>, enable_debug: bool) -> RestirOutput {
    let ray_dir = get_ray_direction(camera, pixel);
    let pixel_index = get_reservoir_index(pixel, camera);
    if (surface.depth == 0.0) {
        reservoirs[pixel_index] = StoredReservoir();
        // Note: the diffuse albedo of the sky is 1.0, so the environment
        // survives the modulation in the post-processing.
        let env = evaluate_environment_background(ray_dir);
        return RestirOutput(Radiance(env, vec3<f32>(0.0)));
    }

    if (WRITE_DEBUG_IMAGE && debug.view_mode == DebugMode_Depth) {
        textureStore(out_debug, pixel, vec4<f32>(1.0 / surface.depth));
    }
    let position = camera.position + surface.depth * ray_dir;
    let debug_len = select(0.0, surface.depth * 0.2, enable_debug);

    var canonical = LiveReservoir();
    for (var i = 0u; i < parameters.num_environment_samples; i += 1u) {
        let ls = sample_incoming_light(surface, rng);
        let brdf = evaluate_sample(ls, surface, position, debug_len, 0x00FF00u);
        if (is_brdf_black(brdf)) {
            bump_reservoir(&canonical, 1.0);
        } else {
            let other = make_reservoir(ls, 0u, brdf, surface.diffuse_albedo);
            merge_reservoir(&canonical, other, random_gen(rng));
        }
    }

    let center_coord = get_prev_pixel(pixel, position);

    // First, gather the list of reservoirs to merge with
    var accepted_reservoir_indices = array<i32, MAX_RESERVOIRS>();
    var accepted_count = 0u;
    let max_samples = min(MAX_RESERVOIRS, parameters.tap_count);
    let num_candidates = max_samples * FACTOR_CANDIDATES;

    for (var tap = 0u; tap < num_candidates && accepted_count < max_samples; tap += 1u) {
        let radius = parameters.tap_radius * random_gen(rng);
        let offset = radius * sample_circle_uniform(random_gen(rng));
        let other_pixel = vec2<i32>(center_coord + offset);

        let other_index = get_reservoir_index(other_pixel, prev_camera);
        if (other_index < 0) {
            continue;
        }
        if (prev_reservoirs[other_index].confidence == 0.0) {
            continue;
        }

        let other_surface = read_prev_surface(other_pixel);
        let compatibility = compare_surfaces(surface, other_surface);
        if (compatibility < 0.1) {
            // if the surfaces are too different, there is no trust in this sample
            continue;
        }

        accepted_reservoir_indices[accepted_count] = other_index;
        accepted_count += 1u;
    }

    if (WRITE_DEBUG_IMAGE && debug.view_mode == DebugMode_SampleReuse) {
        var color = vec4<f32>(0.0);
        for (var i = 0u; i < min(3u, accepted_count); i += 1u) {
            color[i] = 1.0;
        }
        textureStore(out_debug, pixel, color);
    }

    // Next, evaluate the MIS of each of the samples versus the canonical one.
    var reservoir = LiveReservoir();
    var shaded = zero_radiance();
    var shaded_weight = 0.0;
    let mis_scale = 1.0 / (f32(accepted_count) + parameters.defensive_mis);
    var mis_canonical = select(mis_scale * parameters.defensive_mis, 1.0, accepted_count == 0u || parameters.use_pairwise_mis == 0u);
    let inv_count = 1.0 / f32(accepted_count);

    for (var rid = 0u; rid < accepted_count; rid += 1u) {
        let neighbor_index = accepted_reservoir_indices[rid];
        let neighbor = prev_reservoirs[neighbor_index];
        let neighbor_pixel = get_pixel_from_reservoir_index(neighbor_index, prev_camera);

        let offset = vec2<f32>(neighbor_pixel) - center_coord;
        let max_confidence = mix(parameters.tap_confidence_near, parameters.tap_confidence_far, length(offset) / parameters.tap_radius);
        var other: LiveReservoir;
        if (parameters.use_pairwise_mis != 0u) {
            let neighbor_history = min(neighbor.confidence, max_confidence);
            {   // scoping this to hint the register allocation
                let neighbor_surface = read_prev_surface(neighbor_pixel);
                let neighbor_dir = get_ray_direction(prev_camera, neighbor_pixel);
                let neighbor_position = prev_camera.position + neighbor_surface.depth * neighbor_dir;

                let t_canonical_at_neighbor = estimate_target_score_with_occlusion(
                    neighbor_surface, neighbor_position, canonical.selected_light_index, canonical.selected_uv, prev_acc_struct, debug_len, 0xFF0000u);
                let r_canonical = ratio(canonical.history * canonical.selected_target_score * inv_count, neighbor_history * t_canonical_at_neighbor.score);
                mis_canonical += mis_scale * r_canonical;
            }

            let t_neighbor_at_canonical = estimate_target_score_with_occlusion(
                surface, position, neighbor.light_index, neighbor.light_uv, acc_struct, debug_len, 0x0000FFu);
            let r_neighbor = ratio(neighbor_history * neighbor.target_score, canonical.history * t_neighbor_at_canonical.score * inv_count);
            let mis_neighbor = mis_scale * r_neighbor;

            other.history = neighbor_history;
            other.selected_light_index = neighbor.light_index;
            other.selected_uv = neighbor.light_uv;
            other.selected_target_score = t_neighbor_at_canonical.score;
            other.selected_radiance = t_neighbor_at_canonical.radiance;
            other.weight_sum = t_neighbor_at_canonical.score * neighbor.contribution_weight * mis_neighbor;
        } else {
            let radiance = evaluate_reflected_light(surface, neighbor.light_index, neighbor.light_uv);
            other = unpack_reservoir(neighbor, max_confidence, radiance);
        }

        if (DECOUPLED_SHADING) {
            let scale = other.weight_sum * neighbor.contribution_weight;
            shaded.diffuse += scale * other.selected_radiance.diffuse;
            shaded.specular += scale * other.selected_radiance.specular;
            shaded_weight += other.weight_sum;
        }
        if (other.weight_sum <= 0.0) {
            bump_reservoir(&reservoir, other.history);
        } else {
            merge_reservoir(&reservoir, other, random_gen(rng));
        }
    }

    // Finally, merge in the canonical sample
    if (parameters.use_pairwise_mis != 0) {
        normalize_reservoir(&canonical, mis_canonical);
    }
    if (DECOUPLED_SHADING) {
        let cw = canonical.weight_sum / max(canonical.selected_target_score, 0.1);
        let scale = canonical.weight_sum * cw;
        shaded.diffuse += scale * canonical.selected_radiance.diffuse;
        shaded.specular += scale * canonical.selected_radiance.specular;
        shaded_weight += canonical.weight_sum;
    }
    merge_reservoir(&reservoir, canonical, random_gen(rng));

    let effective_history = select(reservoir.history, 1.0, parameters.use_pairwise_mis != 0);
    let stored = pack_reservoir_detail(reservoir, effective_history);
    reservoirs[pixel_index] = stored;
    var ro = RestirOutput();
    if (DECOUPLED_SHADING) {
        let denom = max(shaded_weight, 0.001);
        ro.radiance = Radiance(shaded.diffuse / denom, shaded.specular / denom);
    } else {
        let cw = stored.contribution_weight;
        ro.radiance = Radiance(cw * reservoir.selected_radiance.diffuse, cw * reservoir.selected_radiance.specular);
    }
    return ro;
}

@compute @workgroup_size(8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id.xy >= camera.target_size)) {
        return;
    }

    let global_index = global_id.y * camera.target_size.x + global_id.x;
    var rng = random_init(global_index, parameters.frame_index);

    let surface = read_surface(vec2<i32>(global_id.xy));
    let enable_debug = DEBUG_MODE && all(global_id.xy == debug.mouse_pos);
    let enable_restir_debug = (debug.draw_flags & DebugDrawFlags_RESTIR) != 0u && enable_debug;
    let ro = compute_restir(surface, vec2<i32>(global_id.xy), &rng, enable_restir_debug);

    if (enable_debug) {
        // Note: the variance is tracked on the fully modulated color
        let color = surface.diffuse_albedo * ro.radiance.diffuse + ro.radiance.specular;
        debug_buf.variance.color_sum += color;
        debug_buf.variance.color2_sum += color * color;
        debug_buf.variance.count += 1u;
    }
    textureStore(out_diffuse, global_id.xy, vec4<f32>(ro.radiance.diffuse, 1.0));
    textureStore(out_specular, global_id.xy, vec4<f32>(ro.radiance.specular, 1.0));
}
