#include "color.inc.wgsl"
#include "quaternion.inc.wgsl"
#include "random.inc.wgsl"
#include "env-importance.inc.wgsl"
#include "debug.inc.wgsl"
#include "debug-param.inc.wgsl"
#include "camera.inc.wgsl"
#include "surface.inc.wgsl"
#include "geometry.inc.wgsl"
#include "motion.inc.wgsl"
#include "accum.inc.wgsl"

const PI: f32 = 3.1415926;
const MAX_RESAMPLE: u32 = 4u;

const DRAW_DEBUG: bool = false;
// See "DECOUPLING SHADING AND REUSE" in
// "Rearchitecting Spatiotemporal Resampling for Production"
const DECOUPLED_SHADING: bool = false;
const WRITE_DEBUG_IMAGE: bool = false;
//TODO: currently unused
const WRITE_MOTION_VECTORS: bool = false;

//TODO: crashes on AMD 6850U if `GROUP_SIZE_TOTAL` > 32
const GROUP_SIZE: vec2<u32> = vec2<u32>(8, 4);
const GROUP_SIZE_TOTAL: u32 = GROUP_SIZE.x * GROUP_SIZE.y;

var<private> p_debug_len: f32;
var<private> p_rng: RandomState;

struct MainParams {
    frame_index: u32,
    num_environment_samples: u32,
    environment_importance_sampling: u32,
    temporal_tap: u32,
    temporal_tap_confidence: f32,
    spatial_taps: u32,
    spatial_tap_confidence: f32,
    spatial_min_distance: i32,
    t_start: f32,
    use_pairwise_mis: u32,
    use_motion_vectors: u32,
    temporal_accumulation_weight: f32,
    grid_scale: vec2<u32>,
}

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

struct PixelCache {
    surface: Surface,
    reservoir: StoredReservoir,
    //Note: we could store direction XY in local camera space instead
    world_pos: vec3<f32>,
}
var<workgroup> pixel_cache: array<PixelCache, GROUP_SIZE_TOTAL>;

struct LightSample {
    radiance: vec3<f32>,
    pdf: f32,
    uv: vec2<f32>,
}

struct LiveReservoir {
    selected_uv: vec2<f32>,
    selected_light_index: u32,
    selected_target_score: f32,
    radiance: vec3<f32>,
    weight_sum: f32,
    history: f32,
}

fn compute_target_score(radiance: vec3<f32>) -> f32 {
    return dot(radiance, vec3<f32>(0.3, 0.4, 0.3));
}

fn get_reservoir_index(pixel: vec2<i32>, camera: CameraParams) -> i32 {
    if (all(vec2<u32>(pixel) < camera.target_size)) {
        return pixel.y * i32(camera.target_size.x) + pixel.x;
    } else {
        return -1;
    }
}

fn bump_reservoir(r: ptr<function, LiveReservoir>, history: f32) {
    (*r).history += history;
}
fn make_reservoir(ls: LightSample, light_index: u32, brdf: vec3<f32>) -> LiveReservoir {
    var r: LiveReservoir;
    r.radiance = ls.radiance * brdf;
    r.selected_uv = ls.uv;
    r.selected_light_index = light_index;
    r.selected_target_score = compute_target_score(r.radiance);
    r.weight_sum = r.selected_target_score / ls.pdf;
    r.history = 1.0;
    return r;
}

fn merge_reservoir(r: ptr<function, LiveReservoir>, other: LiveReservoir) -> bool {
    (*r).weight_sum += other.weight_sum;
    (*r).history += other.history;
    let random = random_gen(&p_rng);
    if ((*r).weight_sum * random < other.weight_sum) {
        (*r).selected_light_index = other.selected_light_index;
        (*r).selected_uv = other.selected_uv;
        (*r).selected_target_score = other.selected_target_score;
        (*r).radiance = other.radiance;
        return true;
    } else {
        return false;
    }
}
fn unpack_reservoir(f: StoredReservoir, max_confidence: f32, radiance: vec3<f32>) -> LiveReservoir {
    var r: LiveReservoir;
    r.selected_light_index = f.light_index;
    r.selected_uv = f.light_uv;
    r.selected_target_score = f.target_score;
    r.radiance = radiance;
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

var inout_depth: texture_storage_2d<r32float, read_write>;
var inout_basis: texture_storage_2d<rgba8snorm, read_write>;
var inout_flat_normal: texture_storage_2d<rgba8snorm, read_write>;
var out_albedo: texture_storage_2d<rgba8unorm, write>;
var out_motion: texture_storage_2d<rg8snorm, write>;
var out_debug: texture_storage_2d<rgba8unorm, write>;

fn sample_circle(random: f32) -> vec2<f32> {
    let angle = 2.0 * PI * random;
    return vec2<f32>(cos(angle), sin(angle));
}

fn square(v: f32) -> f32 {
    return v * v;
}

fn map_equirect_dir_to_uv(dir: vec3<f32>) -> vec2<f32> {
    //Note: Y axis is up
    let yaw = asin(dir.y);
    let pitch = atan2(dir.x, dir.z);
    return vec2<f32>(pitch + PI, -2.0 * yaw + PI) / (2.0 * PI);
}
fn map_equirect_uv_to_dir(uv: vec2<f32>) -> vec3<f32> {
    let yaw = PI * (0.5 - uv.y);
    let pitch = 2.0 * PI * (uv.x - 0.5);
    return vec3<f32>(cos(yaw) * sin(pitch), sin(yaw), cos(yaw) * cos(pitch));
}

fn evaluate_environment(dir: vec3<f32>) -> vec3<f32> {
    let uv = map_equirect_dir_to_uv(dir);
    return textureSampleLevel(env_map, sampler_linear, uv, 0.0).xyz;
}

fn sample_light_from_sphere() -> LightSample {
    let a = random_gen(&p_rng);
    let h = 1.0 - 2.0 * random_gen(&p_rng); // make sure to allow h==1
    let tangential = sqrt(1.0 - square(h)) * sample_circle(a);
    let dir = vec3<f32>(tangential.x, h, tangential.y);
    var ls = LightSample();
    ls.uv = map_equirect_dir_to_uv(dir);
    ls.pdf = 1.0 / (4.0 * PI);
    ls.radiance = textureSampleLevel(env_map, sampler_linear, ls.uv, 0.0).xyz;
    return ls;
}

fn sample_light_from_environment() -> LightSample {
    let dim = textureDimensions(env_map, 0);
    let es = generate_environment_sample(&p_rng, dim);
    var ls = LightSample();
    ls.pdf = es.pdf;
    // sample the incoming radiance
    ls.radiance = textureLoad(env_map, es.pixel, 0).xyz;
    // for determining direction - offset randomly within the texel
    // Note: this only works if the texels are sufficiently small
    ls.uv = (vec2<f32>(es.pixel) + vec2<f32>(random_gen(&p_rng), random_gen(&p_rng))) / vec2<f32>(dim);
    return ls;
}

fn read_prev_surface(pixel: vec2<i32>) -> Surface {
    var surface: Surface;
    surface.basis = normalize(textureLoad(inout_basis, pixel));
    surface.flat_normal = normalize(textureLoad(inout_flat_normal, pixel).xyz);
    surface.depth = textureLoad(inout_depth, pixel).x;
    return surface;
}

fn thread_index_to_coord(thread_index: u32, group_id: vec3<u32>) -> vec2<i32> {
    let cluster_id = group_id.xy / parameters.grid_scale;
    let cluster_offset = group_id.xy - cluster_id * parameters.grid_scale;
    let local_id = vec2<u32>(thread_index % GROUP_SIZE.x, thread_index / GROUP_SIZE.x);
    let global_id = (cluster_id * GROUP_SIZE + local_id) * parameters.grid_scale + cluster_offset;
    return vec2<i32>(global_id);
}

fn evaluate_brdf(surface: Surface, dir: vec3<f32>) -> f32 {
    let lambert_brdf = 1.0 / PI;
    let lambert_term = qrot(qinv(surface.basis), dir).z;
    //Note: albedo not modulated
    return lambert_brdf * max(0.0, lambert_term);
}

fn check_ray_occluded(prev_frame: bool, position: vec3<f32>, direction: vec3<f32>) -> bool {
    var rq: ray_query;
    let flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_CULL_NO_OPAQUE;
    let desc = RayDesc(flags, 0xFFu, parameters.t_start, camera.depth, position, direction);
    if (prev_frame) {
        rayQueryInitialize(&rq, prev_acc_struct, desc);
    } else {
        rayQueryInitialize(&rq, acc_struct, desc);
    }
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);

    let occluded = intersection.kind != RAY_QUERY_INTERSECTION_NONE;
    if (DRAW_DEBUG && p_debug_len != 0.0) {
        let color = select(0xFFFFFFu, 0x0000FFu, occluded);
        debug_line(position, position + p_debug_len * direction, color);
    }
    return occluded;
}

fn evaluate_reflected_light(surface: Surface, light_index: u32, light_uv: vec2<f32>) -> vec3<f32> {
    if (light_index != 0u) {
        return vec3<f32>(0.0);
    }
    let direction = map_equirect_uv_to_dir(light_uv);
    let brdf = evaluate_brdf(surface, direction);
    if (brdf <= 0.0) {
        return vec3<f32>(0.0);
    }
    // Note: returns radiance not modulated by albedo
    let radiance = textureSampleLevel(env_map, sampler_nearest, light_uv, 0.0).xyz;
    return radiance * brdf;
}

struct TargetScore {
    color: vec3<f32>,
    score: f32,
}

fn make_target_score(color: vec3<f32>) -> TargetScore {
    return TargetScore(color, compute_target_score(color));
}

fn estimate_target_score_with_occlusion(
    surface: Surface, position: vec3<f32>, light_index: u32, light_uv: vec2<f32>, prev_frame: bool,
) -> TargetScore {
    if (light_index != 0u) {
        return TargetScore();
    }
    let direction = map_equirect_uv_to_dir(light_uv);
    if (dot(direction, surface.flat_normal) <= 0.0) {
        return TargetScore();
    }
    let brdf = evaluate_brdf(surface, direction);
    if (brdf <= 0.0) {
        return TargetScore();
    }

    if (check_ray_occluded(prev_frame, position, direction)) {
        return TargetScore();
    }

    //Note: same as `evaluate_reflected_light`
    let radiance = textureSampleLevel(env_map, sampler_nearest, light_uv, 0.0).xyz;
    return make_target_score(brdf * radiance);
}

fn evaluate_sample(ls: LightSample, surface: Surface, start_pos: vec3<f32>) -> f32 {
    let dir = map_equirect_uv_to_dir(ls.uv);
    if (dot(dir, surface.flat_normal) <= 0.0) {
        return 0.0;
    }

    let brdf = evaluate_brdf(surface, dir);
    if (brdf <= 0.0) {
        return 0.0;
    }

    let target_score = compute_target_score(ls.radiance);
    if (target_score < 0.01 * ls.pdf) {
        return 0.0;
    }

    if (check_ray_occluded(false, start_pos, dir)) {
        return 0.0;
    }

    return brdf;
}

fn produce_canonical(
    surface: Surface, position: vec3<f32>,
) -> LiveReservoir {
    var reservoir = LiveReservoir();
    for (var i = 0u; i < parameters.num_environment_samples; i += 1u) {
        var ls: LightSample;
        if (parameters.environment_importance_sampling != 0u) {
            ls = sample_light_from_environment();
        } else {
            ls = sample_light_from_sphere();
        }

        let brdf = evaluate_sample(ls, surface, position);
        if (brdf > 0.0) {
            let other = make_reservoir(ls, 0u, vec3<f32>(brdf));
            merge_reservoir(&reservoir, other);
        } else {
            bump_reservoir(&reservoir, 1.0);
        }
    }
    return reservoir;
}

struct TemporalReprojection {
    is_valid: bool,
    pixel: vec2<i32>,
    surface: Surface,
    reservoir: StoredReservoir,
}

fn find_temporal(surface: Surface, pixel: vec2<i32>, center_coord: vec2<f32>) -> TemporalReprojection {
    var tr = TemporalReprojection();
    tr.is_valid = false;
    if (surface.depth == 0.0) {
        return tr;
    }

    // Find best match in a 2x2 grid
    let center_pixel = vec2<i32>(center_coord);
    // Trick to start with closer pixels
    let center_sum = vec2<i32>(center_coord - 0.5) + vec2<i32>(center_coord + 0.5);
    var prev_pixels = array<vec2<i32>, 4>(
        center_pixel.xy,
        vec2<i32>(center_sum.x - center_pixel.x, center_pixel.y),
        center_sum - center_pixel,
        vec2<i32>(center_pixel.x, center_sum.y - center_pixel.y),
    );

    for (var i = 0; i < 4 && !tr.is_valid; i += 1) {
        tr.pixel = prev_pixels[i];
        let prev_reservoir_index = get_reservoir_index(tr.pixel, prev_camera);
        if (prev_reservoir_index < 0) {
            continue;
        }
        tr.reservoir = reservoirs[prev_reservoir_index];
        if (tr.reservoir.confidence == 0.0) {
            continue;
        }
        tr.surface = read_prev_surface(tr.pixel);
        if (compare_surfaces(surface, tr.surface) < 0.1) {
            continue;
        }
        tr.is_valid = true;

        if (WRITE_DEBUG_IMAGE && debug.view_mode == DebugMode_Reprojection) {
            var colors = array<vec3<f32>, 4>(
                vec3<f32>(1.0, 1.0, 1.0),
                vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(0.0, 1.0, 0.0),
                vec3<f32>(0.0, 0.0, 1.0),
            );
            textureStore(out_debug, pixel, vec4<f32>(colors[i], 1.0));
        }
    }
    return tr;
}

struct ResampleBase {
    surface: Surface,
    canonical: LiveReservoir,
    world_pos: vec3<f32>,
    accepted_count: f32,
}

struct ShiftSample {
    reservoir: LiveReservoir,
    mis_canonical: f32,
    mis_sample: f32,
}

// Resample following Algorithm 8 in section 9.1 of Bitterli thesis
fn shift_sample(
    base: ResampleBase, other: PixelCache, other_prev_frame: bool,
    max_confidence: f32,
) -> ShiftSample {
    var ss = ShiftSample();
    let neighbor = other.reservoir;
    if (parameters.use_pairwise_mis != 0u) {
        let canonical = base.canonical;
        let neighbor_history = min(neighbor.confidence, max_confidence);
        {   // scoping this to hint the register allocation
            let t_canonical_at_neighbor = estimate_target_score_with_occlusion(
                other.surface, other.world_pos, canonical.selected_light_index, canonical.selected_uv, other_prev_frame);
            let nom = canonical.selected_target_score * canonical.history / base.accepted_count;
            let denom = t_canonical_at_neighbor.score * neighbor_history + nom;
            ss.mis_canonical = select(0.0, nom / denom, denom > 0.0);
        }

        let canonical_prev_frame = false;
        let t_neighbor_at_canonical = estimate_target_score_with_occlusion(
            base.surface, base.world_pos, neighbor.light_index, neighbor.light_uv, canonical_prev_frame);
        let nom = neighbor.target_score * neighbor_history;
        let denom = nom + t_neighbor_at_canonical.score * canonical.history / base.accepted_count;
        let mis_neighbor = select(0.0, nom / denom, denom > 0.0);
        ss.mis_sample  = mis_neighbor;

        var src: LiveReservoir;
        src.history = neighbor_history;
        src.selected_light_index = neighbor.light_index;
        src.selected_uv = neighbor.light_uv;
        src.selected_target_score = t_neighbor_at_canonical.score;
        src.weight_sum = t_neighbor_at_canonical.score * neighbor.contribution_weight * mis_neighbor;
        src.radiance = t_neighbor_at_canonical.color;
        ss.reservoir = src;
    } else {
        ss.mis_canonical = 0.5;
        ss.mis_sample = 0.5;
        let radiance = evaluate_reflected_light(base.surface, neighbor.light_index, neighbor.light_uv);
        ss.reservoir = unpack_reservoir(neighbor, max_confidence, radiance);
    }
    return ss;
}

struct ResampleOutput {
    reservoir: StoredReservoir,
    color: vec3<f32>,
}

fn revive_canonical(ro: ResampleOutput) -> LiveReservoir {
    let radiance = select(vec3<f32>(0.0), ro.color / ro.reservoir.contribution_weight, ro.reservoir.contribution_weight > 0.0);
    return unpack_reservoir(ro.reservoir, 100.0, radiance);
}

fn finalize_canonical(reservoir: LiveReservoir) -> ResampleOutput {
    var ro = ResampleOutput();
    ro.reservoir = pack_reservoir(reservoir);
    ro.color = ro.reservoir.contribution_weight * reservoir.radiance;
    return ro;
}

fn finalize_resampling(
    reservoir: ptr<function, LiveReservoir>, color_and_weight: ptr<function, vec4<f32>>,
    base: ResampleBase, mis_canonical: f32,
) -> ResampleOutput {
    var canonical = base.canonical;
    if (parameters.use_pairwise_mis != 0u) {
        canonical.weight_sum *= mis_canonical / canonical.history;
    }
    merge_reservoir(reservoir, canonical);

    let effective_history = select((*reservoir).history, 1.0 + base.accepted_count, parameters.use_pairwise_mis != 0u);
    var ro = ResampleOutput();
    ro.reservoir = pack_reservoir_detail(*reservoir, effective_history);

    if (DECOUPLED_SHADING) {
        if (canonical.selected_target_score > 0.0) {
            let contribution_weight = canonical.weight_sum / canonical.selected_target_score;
            *color_and_weight += canonical.weight_sum * vec4<f32>(contribution_weight * canonical.radiance, 1.0);
        }
        ro.color = (*color_and_weight).xyz / max((*color_and_weight).w, 0.001);
    } else {
        ro.color = ro.reservoir.contribution_weight * (*reservoir).radiance;
    }
    return ro;
}

struct Pass {
    is_temporal: bool,
    confidence: f32,
    taps: u32,
    candidates: u32,
}

fn compute_restir(
    rs: RichSurface, pixel: vec2<i32>, local_index: u32, group_id: vec3<u32>,
) -> vec3<f32> {
    let center_coord = vec2<f32>(pixel) + 0.5 + select(vec2<f32>(0.0), rs.motion, parameters.use_motion_vectors != 0u);
    //TODO: recompute this at the end?
    let tr = find_temporal(rs.inner, pixel, center_coord);
    var prev_pixel = select(vec2<i32>(-1), tr.pixel, tr.is_valid);
    let motion_sqr = dot(rs.motion, rs.motion);

    var result = ResampleOutput();
    if (rs.inner.depth == 0.0) {
        let dir = normalize(rs.position - camera.position);
        result.color = evaluate_environment(dir);
    } else {
        let canonical = produce_canonical(rs.inner, rs.position);
        result = finalize_canonical(canonical);

        var num_passes = 0u;
        var passes = array<Pass, 2>();
        if (parameters.temporal_tap != 0u) {
            passes[num_passes] = Pass(true, parameters.temporal_tap_confidence, 1, 0);
            num_passes += 1u;
        }
        if (parameters.spatial_taps > 0) {
            passes[num_passes] = Pass(false, parameters.spatial_tap_confidence, parameters.spatial_taps, parameters.spatial_taps * 4u);
            num_passes += 1u;
        }

        for(var pass_i = 0u; pass_i < num_passes; pass_i += 1u) {
            let ps = passes[pass_i];
            var reservoir = LiveReservoir();
            var color_and_weight = vec4<f32>(0.0);
            var mis_canonical = 0.0;
            var accepted_count = 0u;
            var accepted_local_indices = array<u32, MAX_RESAMPLE>();

            if (ps.is_temporal) {
                if (tr.is_valid) {
                    let prev_dir = get_ray_direction(prev_camera, tr.pixel);
                    let prev_world_pos = prev_camera.position + tr.surface.depth * prev_dir;
                    pixel_cache[local_index] = PixelCache(tr.surface, tr.reservoir, prev_world_pos);
                    accepted_local_indices[0] = local_index;
                    accepted_count += 1u;
                }
            } else {
                pixel_cache[local_index] = PixelCache(rs.inner, result.reservoir, rs.position);
                // sync with the workgroup to ensure all reservoirs are available.
                workgroupBarrier();

                // gather the list of neighbors (within the workgroup) to resample.
                let max_accepted = min(MAX_RESAMPLE, ps.taps);
                for (var i = 0u; i < ps.candidates && accepted_count < max_accepted; i += 1u) {
                    let other_cache_index = random_u32(&p_rng) % GROUP_SIZE_TOTAL;
                    let diff = thread_index_to_coord(other_cache_index, group_id) - pixel;
                    if (dot(diff, diff) < parameters.spatial_min_distance * parameters.spatial_min_distance) {
                        continue;
                    }
                    let other = pixel_cache[other_cache_index];
                    // if the surfaces are too different, there is no trust in this sample
                    if (other.reservoir.confidence > 0.0 && compare_surfaces(rs.inner, other.surface) > 0.1) {
                        accepted_local_indices[accepted_count] = other_cache_index;
                        accepted_count += 1u;
                    }
                }
            }

            if (accepted_count == 0u) {
                continue;
            }

            let input = revive_canonical(result);
            let base = ResampleBase(rs.inner, input, rs.position, f32(accepted_count));

            mis_canonical = 1.0;
            // evaluate the MIS of each of the samples versus the canonical one.
            for (var lid = 0u; lid < accepted_count; lid += 1u) {
                let other = pixel_cache[accepted_local_indices[lid]];

                let ss = shift_sample(base, other, ps.is_temporal, ps.confidence);
                mis_canonical += ss.mis_canonical;

                if (DECOUPLED_SHADING) {
                    let stored = pack_reservoir(ss.reservoir);
                    color_and_weight += ss.reservoir.weight_sum * vec4<f32>(stored.contribution_weight * ss.reservoir.radiance, 1.0);
                }
                if (ss.reservoir.weight_sum <= 0.0) {
                    bump_reservoir(&reservoir, ss.reservoir.history);
                } else {
                    merge_reservoir(&reservoir, ss.reservoir);
                }
            }

            if (WRITE_DEBUG_IMAGE && pass_i == debug.pass_index) {
                if (debug.view_mode == DebugMode_PassMatch) {
                    textureStore(out_debug, pixel, vec4<f32>(1.0));
                }
                if (debug.view_mode == DebugMode_PassMisCanonical) {
                    let mis = mis_canonical / f32(1u + accepted_count);
                    textureStore(out_debug, pixel, vec4<f32>(mis));
                }
            }
            result = finalize_resampling(&reservoir, &color_and_weight, base, mis_canonical);
        }
    }

    let pixel_index = get_reservoir_index(pixel, camera);
    reservoirs[pixel_index] = result.reservoir;

    accumulate_temporal(pixel, result.color, parameters.temporal_accumulation_weight, prev_pixel, motion_sqr);
    return result.color;
}

@compute @workgroup_size(GROUP_SIZE.x, GROUP_SIZE.y)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    pixel_cache[local_index] = PixelCache();
    let pixel_coord = thread_index_to_coord(local_index, group_id);
    if (any(vec2<u32>(pixel_coord) >= camera.target_size)) {
        return;
    }

    if (WRITE_DEBUG_IMAGE) {
        var default_color = vec3<f32>(0.0);
        if (debug.view_mode == DebugMode_Grouping) {
            p_rng = random_init(group_id.y * 1000u + group_id.x, 0u);
            let h = random_gen(&p_rng) * 360.0;
            default_color = hsv_to_rgb(h, 1.0, 1.0);
        }
        textureStore(out_debug, pixel_coord, vec4<f32>(default_color, 0.0));
    }

    let enable_debug = DRAW_DEBUG && all(pixel_coord == vec2<i32>(debug.mouse_pos));
    let rs = fetch_geometry(pixel_coord, true, enable_debug);

    let global_index = u32(pixel_coord.y) * camera.target_size.x + u32(pixel_coord.x);
    p_rng = random_init(global_index, parameters.frame_index);

    let enable_restir_debug = (debug.draw_flags & DebugDrawFlags_RESTIR) != 0u && enable_debug;
    p_debug_len = select(0.0, rs.inner.depth * 0.2, enable_restir_debug);
    let color = compute_restir(rs, pixel_coord, local_index, group_id);

    //Note: important to do this after the temporal pass specifically
    // TODO: option to avoid writing data for the sky
    textureStore(inout_depth, pixel_coord, vec4<f32>(rs.inner.depth, 0.0, 0.0, 0.0));
    textureStore(inout_basis, pixel_coord, rs.inner.basis);
    textureStore(inout_flat_normal, pixel_coord, vec4<f32>(rs.inner.flat_normal, 0.0));
    textureStore(out_albedo, pixel_coord, vec4<f32>(rs.albedo, 0.0));
    if (WRITE_MOTION_VECTORS) {
        textureStore(out_motion, pixel_coord, vec4<f32>(rs.motion * MOTION_SCALE, 0.0, 0.0));
    }

    if (enable_debug) {
        debug_buf.variance.color_sum += color;
        debug_buf.variance.color2_sum += color * color;
        debug_buf.variance.count += 1u;
    }
}
