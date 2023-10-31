#include "quaternion.inc.wgsl"
#include "random.inc.wgsl"
#include "env-importance.inc.wgsl"
#include "debug.inc.wgsl"
#include "debug-param.inc.wgsl"
#include "camera.inc.wgsl"
#include "surface.inc.wgsl"

//TODO: use proper WGSL
const RAY_FLAG_CULL_NO_OPAQUE: u32 = 0x80u;

const PI: f32 = 3.1415926;
const MAX_RESERVOIRS: u32 = 2u;
// See "9.1 pairwise mis for robust reservoir reuse"
// "Correlations and Reuse for Fast and Accurate Physically Based Light Transport"
const PAIRWISE_MIS: bool = true;
// Base MIS for canonical samples. The constant isolates a critical difference between
// Bitterli's pseudocode (where it's 1) and NVidia's RTXDI implementation (where it's 0).
// With Bitterli's 1 we have MIS not respecting the prior history enough.
const BASE_CANONICAL_MIS: f32 = 0.05;
// See "DECOUPLING SHADING AND REUSE" in
// "Rearchitecting Spatiotemporal Resampling for Production"
const DECOUPLED_SHADING: bool = false;

struct MainParams {
    frame_index: u32,
    num_environment_samples: u32,
    environment_importance_sampling: u32,
    temporal_history: u32,
    spatial_taps: u32,
    spatial_tap_history: u32,
    spatial_radius: i32,
};

var<uniform> camera: CameraParams;
var<uniform> prev_camera: CameraParams;
var<uniform> parameters: MainParams;
var<uniform> debug: DebugParams;
var acc_struct: acceleration_structure;
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

fn get_pixel_from_reservoir_index(index: i32, camera: CameraParams) -> vec2<i32> {
    let y = index / i32(camera.target_size.x);
    let x = index - y * i32(camera.target_size.x);
    return vec2<i32>(x, y);
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
fn unpack_reservoir(f: StoredReservoir, max_history: u32) -> LiveReservoir {
    var r: LiveReservoir;
    r.selected_light_index = f.light_index;
    r.selected_uv = f.light_uv;
    r.selected_target_score = f.target_score;
    r.radiance = vec3<f32>(0.0); // to be continued...
    let history = min(f.confidence, f32(max_history));
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
var out_diffuse: texture_storage_2d<rgba16float, write>;
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

fn read_surface(pixel: vec2<i32>) -> Surface {
    var surface: Surface;
    surface.basis = normalize(textureLoad(t_basis, pixel, 0));
    surface.flat_normal = normalize(textureLoad(t_flat_normal, pixel, 0).xyz);
    surface.depth = textureLoad(t_depth, pixel, 0).x;
    return surface;
}

fn read_prev_surface(pixel: vec2<i32>) -> Surface {
    var surface: Surface;
    surface.basis = normalize(textureLoad(t_prev_basis, pixel, 0));
    surface.flat_normal = normalize(textureLoad(t_prev_flat_normal, pixel, 0).xyz);
    surface.depth = textureLoad(t_prev_depth, pixel, 0).x;
    return surface;
}

fn evaluate_brdf(surface: Surface, dir: vec3<f32>) -> f32 {
    let lambert_brdf = 1.0 / PI;
    let lambert_term = qrot(qinv(surface.basis), dir).z;
    //Note: albedo not modulated
    return lambert_brdf * max(0.0, lambert_term);
}

fn check_ray_occluded(position: vec3<f32>, direction: vec3<f32>, debug_len: f32) -> bool {
    let start_t = 0.5; // some offset required to avoid self-shadowing
    var rq: ray_query;
    let flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_CULL_NO_OPAQUE;
    rayQueryInitialize(&rq, acc_struct,
        RayDesc(flags, 0xFFu, start_t, camera.depth, position, direction)
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

struct TargetScore {
    color: vec3<f32>,
    score: f32,
}

fn make_target_score(color: vec3<f32>) -> TargetScore {
    return TargetScore(color, compute_target_score(color));
}

fn estimate_target_score_with_occlusion(
    surface: Surface, position: vec3<f32>, light_index: u32, light_uv: vec2<f32>, debug_len: f32
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

    if (check_ray_occluded(position, direction, debug_len)) {
        return TargetScore();
    } else {
        //Note: same as `evaluate_reflected_light`
        let radiance = textureSampleLevel(env_map, sampler_nearest, light_uv, 0.0).xyz;
        return make_target_score(brdf * radiance);
    }
}

fn evaluate_sample(ls: LightSample, surface: Surface, start_pos: vec3<f32>, debug_len: f32) -> f32 {
    let dir = map_equirect_uv_to_dir(ls.uv);
    if (dot(dir, surface.flat_normal) <= 0.0)
    {
        return 0.0;
    }

    let brdf = evaluate_brdf(surface, dir);
    if (brdf <= 0.0)
    {
        return 0.0;
    }

    let target_score = compute_target_score(ls.radiance);
    if (target_score < 0.01 * ls.pdf) {
        return 0.0;
    }

    if (check_ray_occluded(start_pos, dir, debug_len)) {
        return 0.0;
    }

    return brdf;
}

struct HeuristicFactors {
    weight: f32,
    //history: f32,
}

fn balance_heuristic(w0: f32, w1: f32, h0: f32, h1: f32) -> HeuristicFactors {
    var hf: HeuristicFactors;
    let balance_denom = h0 * w0 + h1 * w1;
    hf.weight = select(h0 * w0 / balance_denom, 0.0, balance_denom <= 0.0);
    //hf.history = select(pow(clamp(w1 / w0, 0.0, 1.0), 8.0), 1.0, w0 <= 0.0);
    return hf;
}

struct RestirOutput {
    radiance: vec3<f32>,
}

fn compute_restir(surface: Surface, pixel: vec2<i32>, rng: ptr<function, RandomState>, enable_debug: bool) -> RestirOutput {
    if (debug.view_mode == DebugMode_Depth) {
        textureStore(out_debug, pixel, vec4<f32>(surface.depth / camera.depth));
    }
    let ray_dir = get_ray_direction(camera, pixel);
    let pixel_index = get_reservoir_index(pixel, camera);
    if (surface.depth == 0.0) {
        reservoirs[pixel_index] = StoredReservoir();
        let env = evaluate_environment(ray_dir);
        return RestirOutput(env);
    }

    let debug_len = select(0.0, surface.depth * 0.2, enable_debug);
    let position = camera.position + surface.depth * ray_dir;
    let normal = qrot(surface.basis, vec3<f32>(0.0, 0.0, 1.0));
    if (debug.view_mode == DebugMode_Normal) {
        textureStore(out_debug, pixel, vec4<f32>(normal, 0.0));
    }

    var canonical = LiveReservoir();
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
            merge_reservoir(&canonical, other, random_gen(rng));
        } else {
            bump_reservoir(&canonical, 1.0);
        }
    }

    let prev_pixel = get_projected_pixel(prev_camera, position);

    // First, gather the list of reservoirs to merge with
    var accepted_reservoir_indices = array<i32, MAX_RESERVOIRS>();
    var accepted_count = 0u;
    var temporal_index = ~0u;
    for (var tap = 0u; tap <= parameters.spatial_taps; tap += 1u) {
        var other_pixel = prev_pixel;
        if (tap != 0u) {
            let r0 = max(prev_pixel - vec2<i32>(parameters.spatial_radius), vec2<i32>(0));
            let r1 = min(prev_pixel + vec2<i32>(parameters.spatial_radius + 1), vec2<i32>(prev_camera.target_size));
            other_pixel = vec2<i32>(mix(vec2<f32>(r0), vec2<f32>(r1), vec2<f32>(random_gen(rng), random_gen(rng))));
        } else if (parameters.temporal_history == 0u)
        {
            continue;
        }

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

        if (tap == 0u) {
            temporal_index = accepted_count;
        }
        accepted_reservoir_indices[accepted_count] = other_index;
        if (accepted_count < MAX_RESERVOIRS) {
            accepted_count += 1u;
        }
    }

    // Next, evaluate the MIS of each of the samples versus the canonical one.
    var reservoir = LiveReservoir();
    var shaded_color = vec3<f32>(0.0);
    var mis_canonical = BASE_CANONICAL_MIS;
    var color_and_weight = vec4<f32>(0.0);
    for (var rid = 0u; rid < accepted_count; rid += 1u) {
        let neighbor_index = accepted_reservoir_indices[rid];
        let neighbor = prev_reservoirs[neighbor_index];

        let max_history = select(parameters.spatial_tap_history, parameters.temporal_history, rid == temporal_index);
        var other: LiveReservoir;
        if (PAIRWISE_MIS) {
            let neighbor_pixel = get_pixel_from_reservoir_index(neighbor_index, prev_camera);
            let neighbor_history = min(neighbor.confidence, f32(max_history));
            {   // scoping this to hint the register allocation
                let neighbor_surface = read_prev_surface(neighbor_pixel);
                let neighbor_dir = get_ray_direction(prev_camera, neighbor_pixel);
                let neighbor_position = prev_camera.position + neighbor_surface.depth * neighbor_dir;

                let t_canonical_at_neighbor = estimate_target_score_with_occlusion(
                    neighbor_surface, neighbor_position, canonical.selected_light_index, canonical.selected_uv, debug_len);
                let mis_sub_canonical = balance_heuristic(
                    t_canonical_at_neighbor.score, canonical.selected_target_score,
                    neighbor_history * f32(accepted_count), canonical.history);
                mis_canonical += 1.0 - mis_sub_canonical.weight;
            }

            // Notes about t_neighbor_at_neighbor:
            // 1. we assume lights aren't moving. Technically we should check if the
            //   target light has moved, and re-evaluate the occlusion.
            // 2. we can use the cached target score, and there is no use of the target color
            //let t_neighbor_at_neighbor = estimate_target_pdf(neighbor_surface, neighbor_position, neighbor.selected_dir);
            let t_neighbor_at_canonical = estimate_target_score_with_occlusion(
                surface, position, neighbor.light_index, neighbor.light_uv, debug_len);
            let mis_neighbor = balance_heuristic(
                neighbor.target_score, t_neighbor_at_canonical.score,
                neighbor_history * f32(accepted_count), canonical.history);

            other.history = neighbor_history;
            other.selected_light_index = neighbor.light_index;
            other.selected_uv = neighbor.light_uv;
            other.selected_target_score = t_neighbor_at_canonical.score;
            other.weight_sum = t_neighbor_at_canonical.score * neighbor.contribution_weight * mis_neighbor.weight;
            //Note: should be needed according to the paper
            // other.history *= min(mis_neighbor.history, mis_sub_canonical.history);
            other.radiance = t_neighbor_at_canonical.color;
        } else {
            other = unpack_reservoir(neighbor, max_history);
            other.radiance = evaluate_reflected_light(surface, other.selected_light_index, other.selected_uv);
        }

        if (DECOUPLED_SHADING) {
            color_and_weight += other.weight_sum * vec4<f32>(neighbor.contribution_weight * other.radiance, 1.0);
        }
        if (other.weight_sum <= 0.0) {
            bump_reservoir(&reservoir, other.history);
        } else {
            merge_reservoir(&reservoir, other, random_gen(rng));
        }
    }

    // Finally, merge in the canonical sample
    if (PAIRWISE_MIS) {
        canonical.weight_sum *= mis_canonical / canonical.history;
    }
    if (DECOUPLED_SHADING) {
        //FIXME: issue with near zero denominator. Do we need do use BASE_CANONICAL_MIS?
        let cw = canonical.weight_sum / max(canonical.selected_target_score * mis_canonical, 0.1);
        color_and_weight += canonical.weight_sum * vec4<f32>(cw * canonical.radiance, 1.0);
    }
    merge_reservoir(&reservoir, canonical, random_gen(rng));

    let effective_history = select(reservoir.history, BASE_CANONICAL_MIS + f32(accepted_count), PAIRWISE_MIS);
    let stored = pack_reservoir_detail(reservoir, effective_history);
    reservoirs[pixel_index] = stored;
    var ro = RestirOutput();
    if (DECOUPLED_SHADING) {
        ro.radiance = color_and_weight.xyz / max(color_and_weight.w, 0.001);
    } else {
        ro.radiance = stored.contribution_weight * reservoir.radiance;
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
    let enable_debug = all(global_id.xy == debug.mouse_pos);
    let enable_restir_debug = (debug.draw_flags & DebugDrawFlags_RESTIR) != 0u && enable_debug;
    let ro = compute_restir(surface, vec2<i32>(global_id.xy), &rng, enable_restir_debug);
    let color = ro.radiance;
    if (enable_debug) {
        debug_buf.variance.color_sum += color;
        debug_buf.variance.color2_sum += color * color;
        debug_buf.variance.count += 1u;
    }
    textureStore(out_diffuse, global_id.xy, vec4<f32>(color, 1.0));
}
