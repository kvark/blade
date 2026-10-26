#include "camera.inc.wgsl"
#include "gbuf.inc.wgsl"
#include "quaternion.inc.wgsl"
#include "surface.inc.wgsl"

// Spatio-temporal variance-guided filtering
// https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf

// RGB is illumination; alpha is luminance variance. Unfiltered temporal RGB
// and variance are retained in the current-frame target as next frame's history.

struct Params {
    extent: vec2<i32>,
    temporal_weight: f32,
    iteration: u32,
    use_motion_vectors: u32,
    _pad: u32,
}

var<uniform> camera: CameraParams;
var<uniform> prev_camera: CameraParams;
var<uniform> params: Params;
var t_depth: texture_2d<f32>;
var t_prev_depth: texture_2d<f32>;
var t_flat_normal: texture_2d<f32>;
var t_prev_flat_normal: texture_2d<f32>;
var t_basis: texture_2d<f32>;
var t_prev_basis: texture_2d<f32>;
var t_specular_f0: texture_2d<f32>;
var t_prev_specular_f0: texture_2d<f32>;
var t_motion: texture_2d<f32>;
var input: texture_2d<f32>;
var current_diffuse: texture_2d<f32>;
var current_specular: texture_2d<f32>;
var output: texture_storage_2d<rgba16float, write>;
// Both filters share guide reconstruction across the lighting lobes.
var input_specular: texture_2d<f32>;
var output_specular: texture_storage_2d<rgba16float, write>;

const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
const MIN_WEIGHT: f32 = 0.01;

fn read_surface(pixel: vec2<i32>) -> Surface {
    var surface = Surface();
    surface.depth = textureLoad(t_depth, pixel, 0).x;
    if (surface.depth > 0.0) {
        surface.flat_normal = normalize(textureLoad(t_flat_normal, pixel, 0).xyz);
        surface.basis = normalize(textureLoad(t_basis, pixel, 0));
        let specular = textureLoad(t_specular_f0, pixel, 0);
        surface.roughness = specular.w;
        surface.specular_f0 = specular.xyz;
    }
    return surface;
}
fn read_prev_surface(pixel: vec2<i32>) -> Surface {
    var surface = Surface();
    surface.depth = textureLoad(t_prev_depth, pixel, 0).x;
    if (surface.depth > 0.0) {
        surface.flat_normal = normalize(textureLoad(t_prev_flat_normal, pixel, 0).xyz);
        surface.basis = normalize(textureLoad(t_prev_basis, pixel, 0));
        let specular = textureLoad(t_prev_specular_f0, pixel, 0);
        surface.roughness = specular.w;
        surface.specular_f0 = specular.xyz;
    }
    return surface;
}

fn get_prev_pixel(pixel: vec2<i32>, pos_world: vec3<f32>) -> vec2<f32> {
    if (USE_MOTION_VECTORS && params.use_motion_vectors != 0u) {
        let motion = textureLoad(t_motion, pixel, 0).xy / MOTION_SCALE;
        return vec2<f32>(pixel) + 0.5 + motion;
    } else {
        return get_projected_pixel_float(prev_camera, pos_world);
    }
}

// XY are diffuse/specular normal weights; Z is specular material rejection.
fn normal_weights(a: Surface, b: Surface) -> vec3<f32> {
    let na = qrot(a.basis, vec3<f32>(0.0, 0.0, 1.0));
    let nb = qrot(b.basis, vec3<f32>(0.0, 0.0, 1.0));
    let cosine = max(0.0, dot(na, nb));
    let cosine2 = cosine * cosine;
    let roughness = min(a.roughness, b.roughness);
    let flat_weight = compare_flat_normals(a.flat_normal, b.flat_normal);
    return vec3<f32>(
        flat_weight * cosine2 * cosine2,
        flat_weight * pow(cosine, mix(256.0, 4.0, roughness * roughness)),
        8.0 * (abs(a.roughness - b.roughness) + length(a.specular_f0 - b.specular_f0)),
    );
}

fn history_moments(history: vec4<f32>) -> vec4<f32> {
    let luminance = dot(history.xyz, LUMA);
    return vec4<f32>(history.xyz, max(history.w, 0.0) + luminance * luminance);
}

fn accumulate_history(current: vec3<f32>, history: vec4<f32>, weight: f32) -> vec4<f32> {
    let luminance = dot(current, LUMA);
    var result = vec4<f32>(current, luminance * luminance);
    if (weight > MIN_WEIGHT) {
        result = mix(result, history / weight, min(weight, 1.0) * (1.0 - params.temporal_weight));
    }
    let mean_luminance = dot(result.xyz, LUMA);
    result.w = max(0.0, result.w - mean_luminance * mean_luminance);
    return result;
}

// Disoccluded pixels have no temporal moments yet. Estimate their variance
// from raw lighting, not from the in-flight temporal outputs (which would race
// other invocations). Both lobes share the same local surface guide loads.
fn bootstrap_variance(pixel: vec2<i32>, surface: Surface) -> vec2<f32> {
    var moments = vec4<f32>(0.0);
    var total_weight = vec2<f32>(0.0);
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let p = pixel + vec2<i32>(x, y);
            if (any(p < vec2<i32>(0)) || any(p >= params.extent)) {
                continue;
            }
            let neighbor = read_surface(p);
            if (neighbor.depth == 0.0 || abs(neighbor.depth - surface.depth) > 0.01 * surface.depth) {
                continue;
            }
            let guides = normal_weights(surface, neighbor);
            let weights = vec2<f32>(guides.x, guides.y * exp(-guides.z));
            let light = vec2<f32>(
                dot(textureLoad(current_diffuse, p, 0).xyz, LUMA),
                dot(textureLoad(current_specular, p, 0).xyz, LUMA),
            );
            moments += vec4<f32>(weights * light, weights * light * light);
            total_weight += weights;
        }
    }
    let mean = moments.xy / max(total_weight, vec2<f32>(MIN_WEIGHT));
    return max(vec2<f32>(0.0), moments.zw / max(total_weight, vec2<f32>(MIN_WEIGHT)) - mean * mean);
}

@compute @workgroup_size(8, 8)
fn temporal_accum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = vec2<i32>(global_id.xy);
    if (any(pixel >= params.extent)) {
        return;
    }

    let surface = read_surface(pixel);
    let cur_illumination = textureLoad(current_diffuse, pixel, 0).xyz;
    let cur_specular = textureLoad(current_specular, pixel, 0).xyz;
    if (surface.depth == 0.0) {
        textureStore(output, pixel, vec4<f32>(cur_illumination, 0.0));
        textureStore(output_specular, pixel, vec4<f32>(cur_specular, 0.0));
        return;
    }
    if (params.temporal_weight >= 1.0) {
        let variance = bootstrap_variance(pixel, surface);
        textureStore(output, pixel, vec4<f32>(cur_illumination, variance.x));
        textureStore(output_specular, pixel, vec4<f32>(cur_specular, variance.y));
        return;
    }
    let pos_world = camera.position + surface.depth * get_ray_direction(camera, pixel);
    // considering all samples in 2x2 quad, to help with edges
    let center_pixel = get_prev_pixel(pixel, pos_world);
    let base = vec2<i32>(floor(center_pixel - vec2<f32>(0.5)));
    var prev_pixels = array<vec2<i32>, 4>(
        base, base + vec2<i32>(1, 0), base + vec2<i32>(1, 1), base + vec2<i32>(0, 1),
    );
    //Note: careful about the pixel center when there is a perfect match
    let w_bot_right = fract(center_pixel + vec2<f32>(0.5));
    var prev_weights = vec4<f32>(
        (1.0 - w_bot_right.x) * (1.0 - w_bot_right.y),
        w_bot_right.x * (1.0 - w_bot_right.y),
        w_bot_right.x * w_bot_right.y,
        (1.0 - w_bot_right.x) * w_bot_right.y,
    );

    var sum_weight = vec2<f32>(0.0);
    var sum_ilm = vec4<f32>(0.0);
    var sum_specular = vec4<f32>(0.0);
    var projected_distance = length(pos_world - prev_camera.position);
    if (USE_MOTION_VECTORS && params.use_motion_vectors != 0u) {
        projected_distance = textureLoad(t_motion, pixel, 0).z;
    }
    if (params.temporal_weight != 1.0) {
        //TODO: optimize depth load with a gather operation
        for (var i = 0; i < 4; i += 1) {
            let w = prev_weights[i];
            if (w == 0.0) {
                continue;
            }
            let prev_pixel = prev_pixels[i];
            if (all(prev_pixel >= vec2<i32>(0)) && all(prev_pixel < params.extent)) {
                let prev_surface = read_prev_surface(prev_pixel);
                if (prev_surface.depth == 0.0) {
                    continue;
                }
                if (abs(prev_surface.depth - projected_distance) > max(0.001, 0.01 * projected_distance)) {
                    continue;
                }
                let weights = normal_weights(surface, prev_surface);
                if (weights.x >= 0.5) {
                    sum_weight.x += w;
                    sum_ilm += w * history_moments(textureLoad(input, prev_pixel, 0));
                }
                if (weights.y * exp(-weights.z) >= 0.5) {
                    sum_weight.y += w;
                    sum_specular += w * history_moments(textureLoad(input_specular, prev_pixel, 0));
                }
            }
        }
    }

    var diffuse = accumulate_history(cur_illumination, sum_ilm, sum_weight.x);
    var specular = accumulate_history(cur_specular, sum_specular, sum_weight.y);
    if (any(sum_weight <= vec2<f32>(MIN_WEIGHT))) {
        let variance = bootstrap_variance(pixel, surface);
        if (sum_weight.x <= MIN_WEIGHT) { diffuse.w = variance.x; }
        if (sum_weight.y <= MIN_WEIGHT) { specular.w = variance.y; }
    }
    textureStore(output, pixel, diffuse);
    textureStore(output_specular, pixel, specular);
}

const GAUSSIAN_WEIGHTS = vec2<f32>(0.44198, 0.27901);
const SIGMA_L: f32 = 4.0;
const EPSILON: f32 = 0.001;

fn luminance_rejection(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let variance = sqrt(max(a.w, b.w));
    return abs(dot(a.xyz - b.xyz, LUMA)) / (SIGMA_L * variance + EPSILON);
}

@compute @workgroup_size(8, 8)
fn atrous3x3(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let center = vec2<i32>(global_id.xy);
    if (any(center >= params.extent)) {
        return;
    }

    let center_ilm = textureLoad(input, center, 0);
    let center_specular = textureLoad(input_specular, center, 0);
    let center_suf = read_surface(center);
    if (center_suf.depth == 0.0) {
        textureStore(output, global_id.xy, center_ilm);
        textureStore(output_specular, global_id.xy, center_specular);
        return;
    }
    let center_position = camera.position + center_suf.depth * get_ray_direction(camera, center);
    var filtered_ilm = center_ilm;
    var neighbor_variance = 0.0;
    var neighbor_weight = 0.0;
    var filtered_specular = center_specular;
    var specular_variance = 0.0;
    var specular_weight_sum = 0.0;

    for (var yy=-1; yy<=1; yy+=1) {
        for (var xx=-1; xx<=1; xx+=1) {
            let p = center + vec2<i32>(xx, yy) * (1i << params.iteration);
            if (all(p == center) || any(p < vec2<i32>(0)) || any(p >= params.extent)) {
                continue;
            }

            //TODO: store in group-shared memory
            let surface = read_surface(p);
            if (surface.depth == 0.0) {
                continue;
            }
            let gaussian = GAUSSIAN_WEIGHTS[abs(xx)] * GAUSSIAN_WEIGHTS[abs(yy)];
            let weights = normal_weights(surface, center_suf);
            // Ray distance is not planar depth: even a flat wall's distance
            // changes across the image. Use a symmetric point-to-plane error
            // to preserve both sloped surfaces and nearby depth discontinuities.
            let position = camera.position + surface.depth * get_ray_direction(camera, p);
            let delta = position - center_position;
            let plane_error = max(abs(dot(delta, surface.flat_normal)), abs(dot(delta, center_suf.flat_normal)));
            // A smooth curved surface bends away from both tangent planes.
            // Account for that bend, but not for a gap between parallel faces.
            let curvature = abs(dot(delta, surface.flat_normal - center_suf.flat_normal));
            let depth_scale = max(0.0001, 0.001 * min(surface.depth, center_suf.depth) + curvature);
            let depth_rejection = plane_error / depth_scale;
            let other_ilm = textureLoad(input, p, 0);
            let other_specular = textureLoad(input_specular, p, 0);
            // The luminance gate must be symmetric. Using only the centre's
            // variance lets a noisy bright pixel accept a dark neighbour while
            // the dark pixel rejects the bright one, which systematically
            // moves radiance out of highlights and shadowed geometry.
            // Combine depth, luminance and material exponentials. This is the
            // same symmetric gate, with one exp instead of up to four.
            let weight = gaussian * weights.x
                * exp(-depth_rejection - luminance_rejection(center_ilm, other_ilm));
            let specular_weight = gaussian * weights.y
                * exp(-depth_rejection - weights.z - luminance_rejection(center_specular, other_specular));

            // Rejected neighbour weight stays on the centre instead of
            // renormalising the surviving neighbours. Every pair then applies
            // equal and opposite RGB deltas, so an A-trous pass conserves
            // linear radiance over the frame. The Gaussian neighbour weights
            // sum to less than one, keeping this a convex update.
            filtered_ilm = vec4<f32>(filtered_ilm.xyz + weight * (other_ilm.xyz - center_ilm.xyz), 0.0);
            neighbor_variance += weight * weight * max(other_ilm.w, 0.0);
            neighbor_weight += weight;
            filtered_specular = vec4<f32>(filtered_specular.xyz + specular_weight * (other_specular.xyz - center_specular.xyz), 0.0);
            specular_variance += specular_weight * specular_weight * max(other_specular.w, 0.0);
            specular_weight_sum += specular_weight;
        }
    }

    filtered_ilm.w = (1.0 - neighbor_weight) * (1.0 - neighbor_weight) * max(center_ilm.w, 0.0) + neighbor_variance;
    textureStore(output, global_id.xy, filtered_ilm);
    filtered_specular.w = (1.0 - specular_weight_sum) * (1.0 - specular_weight_sum) * max(center_specular.w, 0.0) + specular_variance;
    textureStore(output_specular, global_id.xy, filtered_specular);
}
