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
// See "9.1 pairwise mis for robust reservoir reuse"
// "Correlations and Reuse for Fast and Accurate Physically Based Light Transport"
const PAIRWISE_MIS: bool = false; //TODO
// See "DECOUPLING SHADING AND REUSE" in
// "Rearchitecting Spatiotemporal Resampling for Production"
const DECOUPLED_SHADING: bool = false;

//TODO: crashes on AMD 6850U if `GROUP_SIZE_TOTAL` > 32
const GROUP_SIZE: vec2<u32> = vec2<u32>(8, 4);
const GROUP_SIZE_TOTAL: u32 = GROUP_SIZE.x * GROUP_SIZE.y;

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
    use_motion_vectors: u32,
    grid_scale: vec2<u32>,
    temporal_accumulation_weight: f32,
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

fn merge_reservoir(r: ptr<function, LiveReservoir>, other: LiveReservoir, random: f32) -> bool {
    (*r).weight_sum += other.weight_sum;
    (*r).history += other.history;
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

fn sample_light_from_sphere(rng: ptr<function, RandomState>) -> LightSample {
    let a = random_gen(rng);
    let h = 1.0 - 2.0 * random_gen(rng); // make sure to allow h==1
    let tangential = sqrt(1.0 - square(h)) * sample_circle(a);
    let dir = vec3<f32>(tangential.x, h, tangential.y);
    var ls = LightSample();
    ls.uv = map_equirect_dir_to_uv(dir);
    ls.pdf = 1.0 / (4.0 * PI);
    ls.radiance = textureSampleLevel(env_map, sampler_linear, ls.uv, 0.0).xyz;
    return ls;
}

fn sample_light_from_environment(rng: ptr<function, RandomState>) -> LightSample {
    let dim = textureDimensions(env_map, 0);
    let es = generate_environment_sample(rng, dim);
    var ls = LightSample();
    ls.pdf = es.pdf;
    // sample the incoming radiance
    ls.radiance = textureLoad(env_map, es.pixel, 0).xyz;
    // for determining direction - offset randomly within the texel
    // Note: this only works if the texels are sufficiently small
    ls.uv = (vec2<f32>(es.pixel) + vec2<f32>(random_gen(rng), random_gen(rng))) / vec2<f32>(dim);
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

fn check_ray_occluded(acs: acceleration_structure, position: vec3<f32>, direction: vec3<f32>, debug_len: f32) -> bool {
    var rq: ray_query;
    let flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_CULL_NO_OPAQUE;
    rayQueryInitialize(&rq, acs,
        RayDesc(flags, 0xFFu, parameters.t_start, camera.depth, position, direction)
    );
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);

    let occluded = intersection.kind != RAY_QUERY_INTERSECTION_NONE;
    if (debug_len != 0.0) {
        let color = select(0xFFFFFFu, 0x0000FFu, occluded);
        debug_line(position, position + debug_len * direction, color);
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

fn get_prev_pixel(pixel: vec2<i32>, pos_world: vec3<f32>, motion: vec2<f32>) -> vec2<f32> {
    if (USE_MOTION_VECTORS && parameters.use_motion_vectors != 0u) {
        return vec2<f32>(pixel) + 0.5 + motion;
    } else {
        return get_projected_pixel_float(prev_camera, pos_world);
    }
}

struct TargetScore {
    color: vec3<f32>,
    score: f32,
}

fn make_target_score(color: vec3<f32>) -> TargetScore {
    return TargetScore(color, compute_target_score(color));
}

fn estimate_target_score_with_occlusion(
    surface: Surface, position: vec3<f32>, light_index: u32, light_uv: vec2<f32>, acs: acceleration_structure, debug_len: f32
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

    if (check_ray_occluded(acs, position, direction, debug_len)) {
        return TargetScore();
    }

    //Note: same as `evaluate_reflected_light`
    let radiance = textureSampleLevel(env_map, sampler_nearest, light_uv, 0.0).xyz;
    return make_target_score(brdf * radiance);
}

fn evaluate_sample(ls: LightSample, surface: Surface, start_pos: vec3<f32>, debug_len: f32) -> f32 {
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

    if (check_ray_occluded(acc_struct, start_pos, dir, debug_len)) {
        return 0.0;
    }

    return brdf;
}

fn produce_canonical(
    surface: Surface, position: vec3<f32>,
    rng: ptr<function, RandomState>, debug_len: f32,
) -> LiveReservoir {
    var reservoir = LiveReservoir();
    for (var i = 0u; i < parameters.num_environment_samples; i += 1u) {
        var ls: LightSample;
        if (parameters.environment_importance_sampling != 0u) {
            ls = sample_light_from_environment(rng);
        } else {
            ls = sample_light_from_sphere(rng);
        }

        let brdf = evaluate_sample(ls, surface, position, debug_len);
        if (brdf > 0.0) {
            let other = make_reservoir(ls, 0u, vec3<f32>(brdf));
            merge_reservoir(&reservoir, other, random_gen(rng));
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

        if (debug.view_mode == DebugMode_Reprojection) {
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
struct ResampleResult {
    selected: bool,
    mis_canonical: f32,
    mis_sample: f32,
}

// Resample following Algorithm 8 in section 9.1 of Bitterli thesis
fn resample(
    dst: ptr<function, LiveReservoir>, color_and_weight: ptr<function, vec4<f32>>,
    base: ResampleBase, other: PixelCache, other_acs: acceleration_structure,
    max_confidence: f32, rng: ptr<function, RandomState>, debug_len: f32,
) -> ResampleResult {
    var src: LiveReservoir;
    let neighbor = other.reservoir;
    var rr = ResampleResult();
    if (PAIRWISE_MIS) {
        let canonical = base.canonical;
        let neighbor_history = min(neighbor.confidence, max_confidence);
        {   // scoping this to hint the register allocation
            let t_canonical_at_neighbor = estimate_target_score_with_occlusion(
                other.surface, other.world_pos, canonical.selected_light_index, canonical.selected_uv, other_acs, debug_len);
            let nom = canonical.selected_target_score * canonical.history / base.accepted_count;
            let denom = t_canonical_at_neighbor.score * neighbor_history + nom;
            rr.mis_canonical = select(0.0, nom / denom, denom > 0.0);
        }

        // Notes about t_neighbor_at_neighbor:
        // 1. we assume lights aren't moving. Technically we should check if the
        //   target light has moved, and re-evaluate the occlusion.
        // 2. we can use the cached target score, and there is no use of the target color
        //let t_neighbor_at_neighbor = estimate_target_pdf(neighbor_surface, neighbor_position, neighbor.selected_dir);
        let t_neighbor_at_canonical = estimate_target_score_with_occlusion(
            base.surface, base.world_pos, neighbor.light_index, neighbor.light_uv, acc_struct, debug_len);
        let nom = neighbor.target_score * neighbor_history;
        let denom = nom + t_neighbor_at_canonical.score * canonical.history / base.accepted_count;
        let mis_neighbor = select(0.0, nom / denom, denom > 0.0);
        rr.mis_sample  = mis_neighbor;

        src.history = neighbor_history;
        src.selected_light_index = neighbor.light_index;
        src.selected_uv = neighbor.light_uv;
        src.selected_target_score = t_neighbor_at_canonical.score;
        src.weight_sum = t_neighbor_at_canonical.score * neighbor.contribution_weight * mis_neighbor;
        src.radiance = t_neighbor_at_canonical.color;
    } else {
        rr.mis_canonical = 0.0;
        rr.mis_sample = 1.0;
        let radiance = evaluate_reflected_light(base.surface, neighbor.light_index, neighbor.light_uv);
        src = unpack_reservoir(neighbor, max_confidence, radiance);
    }

    if (DECOUPLED_SHADING) {
        //TODO: use `mis_neighbor`O
        *color_and_weight += src.weight_sum * vec4<f32>(neighbor.contribution_weight * src.radiance, 1.0);
    }
    if (src.weight_sum <= 0.0) {
        bump_reservoir(dst, src.history);
    } else {
        merge_reservoir(dst, src, random_gen(rng));
        rr.selected = true;
    }
    return rr;
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
    base: ResampleBase, mis_canonical: f32, rng: ptr<function, RandomState>,
) -> ResampleOutput {
    var ro = ResampleOutput();
    var canonical = base.canonical;
    canonical.weight_sum *= mis_canonical;
    merge_reservoir(reservoir, canonical, random_gen(rng));

    if (base.accepted_count > 0.0) {
        let effective_history = select((*reservoir).history, 1.0 + base.accepted_count, PAIRWISE_MIS);
        ro.reservoir = pack_reservoir_detail(*reservoir, effective_history);
    } else {
        ro.reservoir = pack_reservoir(canonical);
    }

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

fn resample_temporal(
    surface: Surface, cur_pixel: vec2<i32>, position: vec3<f32>,
    local_index: u32, tr: TemporalReprojection,
    rng: ptr<function, RandomState>, debug_len: f32,
) -> ResampleOutput {
    if (surface.depth == 0.0) {
        return ResampleOutput();
    }

    let canonical = produce_canonical(surface, position, rng, debug_len);
    if (parameters.temporal_tap == 0u || !tr.is_valid) {
        return finalize_canonical(canonical);
    }

    var reservoir = LiveReservoir();
    var color_and_weight = vec4<f32>(0.0);
    let base = ResampleBase(surface, canonical, position, 1.0);

    let prev_dir = get_ray_direction(prev_camera, tr.pixel);
    let prev_world_pos = prev_camera.position + tr.surface.depth * prev_dir;
    let other = PixelCache(tr.surface, tr.reservoir, prev_world_pos);
    let rr = resample(&reservoir, &color_and_weight, base, other, prev_acc_struct, parameters.temporal_tap_confidence, rng, debug_len);
    let mis_canonical = 1.0 + rr.mis_canonical;

    if (debug.view_mode == DebugMode_TemporalMatch) {
        textureStore(out_debug, cur_pixel, vec4<f32>(1.0));
    }
    if (debug.view_mode == DebugMode_TemporalMisCanonical) {
        let mis = mis_canonical / (1.0 + base.accepted_count);
        textureStore(out_debug, cur_pixel, vec4<f32>(mis));
    }

    return finalize_resampling(&reservoir, &color_and_weight, base, mis_canonical, rng);
}

fn resample_spatial(
    surface: Surface, cur_pixel: vec2<i32>, position: vec3<f32>,
    group_id: vec3<u32>, canonical: LiveReservoir,
    rng: ptr<function, RandomState>, debug_len: f32,
) -> ResampleOutput {
    if (surface.depth == 0.0) {
        let dir = normalize(position - camera.position);
        var ro = ResampleOutput();
        ro.color = evaluate_environment(dir);
        return ro;
    }

    // gather the list of neighbors (within the workgroup) to resample.
    var accepted_count = 0u;
    var accepted_local_indices = array<u32, MAX_RESAMPLE>();
    let max_accepted = min(MAX_RESAMPLE, parameters.spatial_taps);
    let num_candidates = parameters.spatial_taps * 3u;
    for (var i = 0u; i < num_candidates && accepted_count < max_accepted; i += 1u) {
        let other_cache_index = random_u32(rng) % GROUP_SIZE_TOTAL;
        let diff = thread_index_to_coord(other_cache_index, group_id) - cur_pixel;
        if (dot(diff, diff) < parameters.spatial_min_distance * parameters.spatial_min_distance) {
            continue;
        }
        let other = pixel_cache[other_cache_index];
        // if the surfaces are too different, there is no trust in this sample
        if (other.reservoir.confidence > 0.0 && compare_surfaces(surface, other.surface) > 0.1) {
            accepted_local_indices[accepted_count] = other_cache_index;
            accepted_count += 1u;
        }
    }

    var reservoir = LiveReservoir();
    var color_and_weight = vec4<f32>(0.0);
    let base = ResampleBase(surface, canonical, position, f32(accepted_count));
    var mis_canonical = 1.0;

    // evaluate the MIS of each of the samples versus the canonical one.
    for (var lid = 0u; lid < accepted_count; lid += 1u) {
        let other = pixel_cache[accepted_local_indices[lid]];
        let rr = resample(&reservoir, &color_and_weight, base, other, acc_struct, parameters.spatial_tap_confidence, rng, debug_len);
        mis_canonical += rr.mis_canonical;
    }

    if (debug.view_mode == DebugMode_SpatialMatch) {
        let value = base.accepted_count / max(1.0, f32(parameters.spatial_taps));
        textureStore(out_debug, cur_pixel, vec4<f32>(value));
    }
    if (debug.view_mode == DebugMode_SpatialMisCanonical) {
        let mis = mis_canonical / (1.0 + base.accepted_count);
        textureStore(out_debug, cur_pixel, vec4<f32>(mis));
    }
    return finalize_resampling(&reservoir, &color_and_weight, base, mis_canonical, rng);
}

fn compute_restir(
    rs: RichSurface,
    pixel: vec2<i32>, local_index: u32, group_id: vec3<u32>,
    rng: ptr<function, RandomState>, enable_debug: bool,
) -> vec3<f32> {
    let debug_len = select(0.0, rs.inner.depth * 0.2, enable_debug);

    let center_coord = get_prev_pixel(pixel, rs.position, rs.motion);
    let tr = find_temporal(rs.inner, pixel, center_coord);

    let temporal = resample_temporal(rs.inner, pixel, rs.position, local_index, tr, rng, debug_len);
    pixel_cache[local_index] = PixelCache(rs.inner, temporal.reservoir, rs.position);
    var prev_pixel = select(vec2<i32>(-1), tr.pixel, tr.is_valid);

    // sync with the workgroup to ensure all reservoirs are available.
    workgroupBarrier();

    let temporal_live = revive_canonical(temporal);
    let spatial = resample_spatial(rs.inner, pixel, rs.position, group_id, temporal_live, rng, debug_len);

    let pixel_index = get_reservoir_index(pixel, camera);
    reservoirs[pixel_index] = spatial.reservoir;

    accumulate_temporal(pixel, spatial.color, parameters.temporal_accumulation_weight, prev_pixel);
    return spatial.color;
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

    if (debug.view_mode == DebugMode_Grouping) {
        var rng = random_init(group_id.y * 1000u + group_id.x, 0u);
        let h = random_gen(&rng) * 360.0;
        let color = hsv_to_rgb(h, 1.0, 1.0);
        textureStore(out_debug, pixel_coord, vec4<f32>(color, 1.0));
    } else if (debug.view_mode != DebugMode_Final) {
        textureStore(out_debug, pixel_coord, vec4<f32>(0.0));
    }

    let enable_debug = all(pixel_coord == vec2<i32>(debug.mouse_pos));
    let rs = fetch_geometry(pixel_coord, true, enable_debug);

    let global_index = u32(pixel_coord.y) * camera.target_size.x + u32(pixel_coord.x);
    var rng = random_init(global_index, parameters.frame_index);

    let enable_restir_debug = (debug.draw_flags & DebugDrawFlags_RESTIR) != 0u && enable_debug;
    let color = compute_restir(rs, pixel_coord, local_index, group_id, &rng, enable_restir_debug);

    //Note: important to do this after the temporal pass specifically
    // TODO: option to avoid writing data for the sky
    textureStore(inout_depth, pixel_coord, vec4<f32>(rs.inner.depth, 0.0, 0.0, 0.0));
    textureStore(inout_basis, pixel_coord, rs.inner.basis);
    textureStore(inout_flat_normal, pixel_coord, vec4<f32>(rs.inner.flat_normal, 0.0));
    textureStore(out_albedo, pixel_coord, vec4<f32>(rs.albedo, 0.0));
    textureStore(out_motion, pixel_coord, vec4<f32>(rs.motion * MOTION_SCALE, 0.0, 0.0));

    if (enable_debug) {
        debug_buf.variance.color_sum += color;
        debug_buf.variance.color2_sum += color * color;
        debug_buf.variance.count += 1u;
    }
}
