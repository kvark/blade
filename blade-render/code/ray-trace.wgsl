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

fn bump_reservoir(r: ptr<function, LiveReservoir>) {
    (*r).count += 1u;
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
}

fn evaluate_color(surface: Surface, dir: vec3<f32>) -> vec3<f32> {
    let lambert_brdf = 1.0 / PI;
    let lambert_term = qrot(qinv(surface.basis), dir).z;
    if (lambert_term <= 0.0) {
        return vec3<f32>(0.0);
    }
    return surface.albedo * lambert_brdf;
}

const REJECT_NO = 0u;
const REJECT_BACKFACING = 1u;
const REJECT_ZERO_TARGET_SCORE = 2u;
const REJECT_OCCLUDED = 3u;

fn evaluate_sample(ls: LightSample, surface: Surface, start_pos: vec3<f32>) -> u32 {
    let up = qrot(surface.basis, vec3<f32>(0.0, 0.0, 1.0));
    if (dot(ls.dir, up) <= 0.0)
    {
        return REJECT_BACKFACING;
    }

    let target_score = compute_target_score(ls.radiance);
    if (target_score < 0.01 * ls.pdf) {
        return REJECT_ZERO_TARGET_SCORE;
    }

    let start_t = 0.5; // some offset required to avoid self-shadowing
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct,
        RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT | 0x80u, 0xFFu, start_t, camera.depth, start_pos, ls.dir)
    );
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        return REJECT_OCCLUDED;
    }

    return REJECT_NO;
}

fn compute_restir(ray_dir: vec3<f32>, depth: f32, surface: Surface, pixel_index: u32, rng: ptr<function, RandomState>, enable_debug: bool) -> vec3<f32> {
    if (debug.view_mode == DEBUG_MODE_DEPTH) {
        return vec3<f32>(depth / camera.depth);
    }
    if (depth == 0.0) {
        reservoirs[pixel_index] = StoredReservoir();
        return evaluate_environment(ray_dir);
    }

    let position = camera.position + depth * ray_dir;
    let normal = qrot(surface.basis, vec3<f32>(0.0, 0.0, 1.0));
    if (debug.view_mode == DEBUG_MODE_NORMAL) {
        return normal;
    }

    var reservoir = LiveReservoir();
    var radiance = vec3<f32>(0.0);
    let num_env_samples = select(parameters.num_environment_samples, 100u, enable_debug);
    for (var i = 0u; i < num_env_samples; i += 1u) {
        var ls: LightSample;
        if (parameters.environment_importance_sampling != 0u) {
            ls = sample_light_from_environment(rng);
        } else {
            ls = sample_light_from_sphere(rng);
        }

        let reject = evaluate_sample(ls, surface, position);
        let debug_len = depth * 0.2;
        if (reject == 0u) {
            if (enable_debug) {
                debug_line(position, position + debug_len * ls.dir, 0xFFFFFFu);
            }
            if (update_reservoir(&reservoir, ls, random_gen(rng))) {
                radiance = ls.radiance;
            }
        } else {
            if (enable_debug) {
                debug_line(position, position + debug_len * ls.dir, 0xFF0000u);
            }
            bump_reservoir(&reservoir);
        }
    }

    if (parameters.temporal_history != 0u) {
        let prev_pixel = get_projected_pixel(prev_camera, position);
        if (all(vec2<u32>(prev_pixel) < prev_camera.target_size)) {
            let prev_pixel_index = prev_pixel.y * i32(prev_camera.target_size.x) + prev_pixel.x;
            let prev = unpack_reservoir(prev_reservoirs[prev_pixel_index], parameters.temporal_history);
            if (merge_reservoir(&reservoir, prev, random_gen(rng))) {
                radiance = evaluate_environment(prev.selected_dir);
            }
        }
    }
    let stored = pack_reservoir(reservoir);
    reservoirs[pixel_index] = stored;
    return stored.contribution_weight * radiance * evaluate_color(surface, stored.light_dir);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id.xy > camera.target_size)) {
        return;
    }

    let global_index = global_id.y * camera.target_size.x + global_id.x;
    var rng = random_init(global_index, parameters.frame_index);

    var surface: Surface;
    let ray_dir = get_ray_direction(camera, global_id.xy);
    let depth = textureLoad(in_depth, global_id.xy, 0).x;
    surface.basis = normalize(textureLoad(in_basis, global_id.xy, 0));
    surface.albedo = textureLoad(in_albedo, global_id.xy, 0).xyz;
    let enable_debug = all(global_id.xy == debug.mouse_pos);
    let enable_restir_debug = (debug.flags & DEBUG_FLAGS_RESTIR) != 0u && enable_debug;
    let color = compute_restir(ray_dir, depth, surface, global_index, &rng, enable_restir_debug);
    if (enable_debug) {
        debug_buf.variance.color_sum += color;
        debug_buf.variance.color2_sum += color * color;
        debug_buf.variance.count += 1u;
    }
    textureStore(output, global_id.xy, vec4<f32>(color, 1.0));
}
