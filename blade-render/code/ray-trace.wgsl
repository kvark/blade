#include "quaternion.inc.wgsl"
#include "random.inc.wgsl"
#include "env-importance.inc.wgsl"
#include "debug.inc.wgsl"
#include "debug-param.inc.wgsl"
#include "camera.inc.wgsl"

const PI: f32 = 3.1415926;

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
var acc_struct: acceleration_structure;
var env_map: texture_2d<f32>;
var sampler_linear: sampler;

struct StoredReservoir {
    light_dir: vec3<f32>,
    target_score: f32,
    contribution_weight: f32,
    confidence: f32,
}
var<storage, read_write> reservoirs: array<StoredReservoir>;
var<storage, read> prev_reservoirs: array<StoredReservoir>;

struct LightSample {
    dir: vec3<f32>,
    radiance: vec3<f32>,
    pdf: f32,
}

struct LiveReservoir {
    selected_dir: vec3<f32>,
    selected_target_score: f32,
    weight_sum: f32,
    count: u32,
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

fn bump_reservoir(r: ptr<function, LiveReservoir>, history: u32) {
    (*r).count += history;
}
fn make_reservoir(ls: LightSample) -> LiveReservoir {
    var r: LiveReservoir;
    r.selected_dir = ls.dir;
    r.selected_target_score = compute_target_score(ls.radiance);
    r.weight_sum = r.selected_target_score / ls.pdf;
    r.count = 1u;
    return r;
}
fn merge_reservoir(r: ptr<function, LiveReservoir>, other: LiveReservoir, random: f32) -> bool {
    (*r).weight_sum += other.weight_sum;
    (*r).count += other.count;
    if ((*r).weight_sum * random < other.weight_sum) {
        (*r).selected_dir = other.selected_dir;
        (*r).selected_target_score = other.selected_target_score;
        return true;
    } else {
        return false;
    }
}
fn update_reservoir(r: ptr<function, LiveReservoir>, ls: LightSample, random: f32) -> bool {
    let other = make_reservoir(ls);
    return merge_reservoir(r, other, random);
}
fn unpack_reservoir(f: StoredReservoir, max_count: u32) -> LiveReservoir {
    var r: LiveReservoir;
    r.selected_dir = f.light_dir;
    r.selected_target_score = f.target_score;
    let count = min(u32(f.confidence), max_count);
    r.weight_sum = f.contribution_weight * f.target_score * f32(count);
    r.count = count;
    return r;
}
fn pack_reservoir(r: LiveReservoir) -> StoredReservoir {
    var f: StoredReservoir;
    f.light_dir = r.selected_dir;
    f.target_score = r.selected_target_score;
    f.confidence = f32(r.count);
    let denom = f.target_score * f.confidence;
    f.contribution_weight = select(0.0, r.weight_sum / denom, denom > 0.0);
    return f;
}

var in_depth: texture_2d<f32>;
var in_basis: texture_2d<f32>;
var in_albedo: texture_2d<f32>;
var output: texture_storage_2d<rgba16float, write>;

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
    var ls = LightSample();
    ls.pdf = 1.0 / (4.0 * PI);
    let a = random_gen(rng);
    let h = 1.0 - 2.0 * random_gen(rng); // make sure to allow h==1
    let tangential = sqrt(1.0 - square(h)) * sample_circle(a);
    ls.dir = vec3<f32>(tangential.x, h, tangential.y);
    ls.radiance = evaluate_environment(ls.dir);
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
    let uv = (vec2<f32>(es.pixel) + vec2<f32>(random_gen(rng), random_gen(rng))) / vec2<f32>(dim);
    ls.dir = map_equirect_uv_to_dir(uv);
    return ls;
}

struct Surface {
    basis: vec4<f32>,
    albedo: vec3<f32>,
    depth: f32,
}

fn read_surface(pixel: vec2<i32>) -> Surface {
    var surface: Surface;
    surface.basis = normalize(textureLoad(in_basis, pixel, 0));
    surface.albedo = textureLoad(in_albedo, pixel, 0).xyz;
    surface.depth = textureLoad(in_depth, pixel, 0).x;
    return surface;
}

// Return the compatibility rating, where
// 1.0 means fully compatible, and
// 0.0 means totally incompatible.
fn compare_surfaces(a: Surface, b: Surface) -> f32 {
    let r_normal = smoothstep(0.4, 0.9, dot(a.basis, b.basis));
    let r_depth = 1.0 - smoothstep(0.0, 100.0, abs(a.depth - b.depth));
    return r_normal * r_depth;
}

fn evaluate_color(surface: Surface, dir: vec3<f32>) -> vec3<f32> {
    let lambert_brdf = 1.0 / PI;
    let lambert_term = qrot(qinv(surface.basis), dir).z;
    if (lambert_term <= 0.0) {
        return vec3<f32>(0.0);
    }
    return surface.albedo * lambert_brdf;
}

fn check_ray_occluded(position: vec3<f32>, direction: vec3<f32>, debug_len: f32) -> bool {
    let start_t = 0.5; // some offset required to avoid self-shadowing
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct,
        RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT | 0x80u, 0xFFu, start_t, camera.depth, position, direction)
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

const REJECT_NO = 0u;
const REJECT_BACKFACING = 1u;
const REJECT_ZERO_TARGET_SCORE = 2u;
const REJECT_OCCLUDED = 3u;

fn evaluate_sample(ls: LightSample, surface: Surface, start_pos: vec3<f32>, debug_len: f32) -> u32 {
    let up = qrot(surface.basis, vec3<f32>(0.0, 0.0, 1.0));
    if (dot(ls.dir, up) <= 0.0)
    {
        return REJECT_BACKFACING;
    }

    let target_score = compute_target_score(ls.radiance);
    if (target_score < 0.01 * ls.pdf) {
        return REJECT_ZERO_TARGET_SCORE;
    }

    if (check_ray_occluded(start_pos, ls.dir, debug_len)) {
        return REJECT_OCCLUDED;
    }

    return REJECT_NO;
}

fn compute_restir(surface: Surface, pixel: vec2<i32>, rng: ptr<function, RandomState>, enable_debug: bool) -> vec3<f32> {
    if (debug.view_mode == DEBUG_MODE_DEPTH) {
        return vec3<f32>(surface.depth / camera.depth);
    }
    let ray_dir = get_ray_direction(camera, pixel);
    let pixel_index = get_reservoir_index(pixel, camera);
    if (surface.depth == 0.0) {
        reservoirs[pixel_index] = StoredReservoir();
        return evaluate_environment(ray_dir);
    }

    let position = camera.position + surface.depth * ray_dir;
    let normal = qrot(surface.basis, vec3<f32>(0.0, 0.0, 1.0));
    if (debug.view_mode == DEBUG_MODE_NORMAL) {
        return normal;
    }

    var reservoir = LiveReservoir();
    var radiance = vec3<f32>(0.0);
    let debug_len = select(0.0, surface.depth * 0.2, enable_debug);
    for (var i = 0u; i < parameters.num_environment_samples; i += 1u) {
        var ls: LightSample;
        if (parameters.environment_importance_sampling != 0u) {
            ls = sample_light_from_environment(rng);
        } else {
            ls = sample_light_from_sphere(rng);
        }

        let reject = evaluate_sample(ls, surface, position, debug_len);
        if (reject == 0u) {
            if (update_reservoir(&reservoir, ls, random_gen(rng))) {
                radiance = ls.radiance;
            }
        } else {
            bump_reservoir(&reservoir, 1u);
        }
    }

    var radiance_dirty = false;
    if (parameters.temporal_history != 0u) {
        let prev_pixel = get_projected_pixel(prev_camera, position);
        let prev_index = get_reservoir_index(prev_pixel, prev_camera);
        if (prev_index >= 0) {
            let prev = unpack_reservoir(prev_reservoirs[prev_index], parameters.temporal_history);
            if (check_ray_occluded(position, prev.selected_dir, debug_len)) {
                bump_reservoir(&reservoir, prev.count);
            } else {
                radiance_dirty |= merge_reservoir(&reservoir, prev, random_gen(rng));
            }
        }
    }
    for (var tap = 0u; tap < parameters.spatial_taps; tap += 1u) {
        let r0 = max(pixel - vec2<i32>(parameters.spatial_radius), vec2<i32>(0));
        let r1 = min(pixel + vec2<i32>(parameters.spatial_radius + 1), vec2<i32>(camera.target_size));
        let other_pixel = vec2<i32>(mix(vec2<f32>(r0), vec2<f32>(r1), vec2<f32>(random_gen(rng), random_gen(rng))));
        let other_surface = read_surface(other_pixel);
        let compatibility = compare_surfaces(surface, other_surface);
        let history = u32(f32(parameters.spatial_tap_history) * compatibility);
        if (history == 0u) {
            // if the surfaces are too different, there is no trust in this sample
            continue;
        }
        let other_dir = get_ray_direction(camera, other_pixel);
        let other_position = camera.position + other_surface.depth * other_dir;
        let prev_pixel = get_projected_pixel(prev_camera, other_position);
        let prev_index = get_reservoir_index(prev_pixel, prev_camera);
        if (prev_index < 0) {
            continue;
        }
        let prev = unpack_reservoir(prev_reservoirs[prev_index], history);
        if (check_ray_occluded(position, prev.selected_dir, debug_len)) {
            bump_reservoir(&reservoir, prev.count);
        } else {
            radiance_dirty |= merge_reservoir(&reservoir, prev, random_gen(rng));
        }
    }

    if (radiance_dirty) {
        radiance = evaluate_environment(reservoir.selected_dir);
    }
    let stored = pack_reservoir(reservoir);
    reservoirs[pixel_index] = stored;
    let color = evaluate_color(surface, stored.light_dir);
    return stored.contribution_weight * radiance * color;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id.xy > camera.target_size)) {
        return;
    }

    let global_index = global_id.y * camera.target_size.x + global_id.x;
    var rng = random_init(global_index, parameters.frame_index);

    let surface = read_surface(vec2<i32>(global_id.xy));
    let enable_debug = all(global_id.xy == debug.mouse_pos);
    let enable_restir_debug = (debug.flags & DEBUG_FLAGS_RESTIR) != 0u && enable_debug;
    let color = compute_restir(surface, vec2<i32>(global_id.xy), &rng, enable_restir_debug);
    if (enable_debug) {
        debug_buf.variance.color_sum += color;
        debug_buf.variance.color2_sum += color * color;
        debug_buf.variance.count += 1u;
    }
    textureStore(output, global_id.xy, vec4<f32>(color, 1.0));
}
