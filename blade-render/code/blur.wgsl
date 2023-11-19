#include "camera.inc.wgsl"
#include "quaternion.inc.wgsl"
#include "surface.inc.wgsl"

// Spatio-temporal variance-guided filtering
// https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf

// Note: using "ilm" in place of "illumination and the 2nd moment of its luminance"

struct Params {
    extent: vec2<i32>,
    temporal_weight: f32,
    iteration: u32,
}

var<uniform> camera: CameraParams;
var<uniform> prev_camera: CameraParams;
var<uniform> params: Params;
var t_depth: texture_2d<f32>;
var t_prev_depth: texture_2d<f32>;
var t_flat_normal: texture_2d<f32>;
var t_prev_flat_normal: texture_2d<f32>;
var input: texture_2d<f32>;
var prev_input: texture_2d<f32>;
var output: texture_storage_2d<rgba16float, write>;

const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
const MIN_WEIGHT: f32 = 0.01;

fn read_surface(pixel: vec2<i32>) -> Surface {
    var surface = Surface();
    surface.flat_normal = normalize(textureLoad(t_flat_normal, pixel, 0).xyz);
    surface.depth = textureLoad(t_depth, pixel, 0).x;
    return surface;
}
fn read_prev_surface(pixel: vec2<i32>) -> Surface {
    var surface = Surface();
    surface.flat_normal = normalize(textureLoad(t_prev_flat_normal, pixel, 0).xyz);
    surface.depth = textureLoad(t_prev_depth, pixel, 0).x;
    return surface;
}

@compute @workgroup_size(8, 8)
fn temporal_accum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = vec2<i32>(global_id.xy);
    if (any(pixel >= params.extent)) {
        return;
    }

    //TODO: use motion vectors
    let surface = read_surface(pixel);
    let pos_world = camera.position + surface.depth * get_ray_direction(camera, pixel);
    // considering all samples in 2x2 quad, to help with edges
    let center_pixel = get_projected_pixel_float(prev_camera, pos_world);
    var prev_pixels = array<vec2<i32>, 4>(
        vec2<i32>(vec2<f32>(center_pixel.x - 0.5, center_pixel.y - 0.5)),
        vec2<i32>(vec2<f32>(center_pixel.x + 0.5, center_pixel.y - 0.5)),
        vec2<i32>(vec2<f32>(center_pixel.x + 0.5, center_pixel.y + 0.5)),
        vec2<i32>(vec2<f32>(center_pixel.x - 0.5, center_pixel.y + 0.5)),
    );
    //Note: careful about the pixel center when there is a perfect match
    let w_bot_right = fract(center_pixel + vec2<f32>(0.5));
    var prev_weights = vec4<f32>(
        (1.0 - w_bot_right.x) * (1.0 - w_bot_right.y),
        w_bot_right.x * (1.0 - w_bot_right.y),
        w_bot_right.x * w_bot_right.y,
        (1.0 - w_bot_right.x) * w_bot_right.y,
    );

    var sum_weight = 0.0;
    var sum_ilm = vec4<f32>(0.0);
    //TODO: optimize depth load with a gather operation
    for (var i = 0; i < 4; i += 1) {
        let prev_pixel = prev_pixels[i];
        if (all(prev_pixel >= vec2<i32>(0)) && all(prev_pixel < params.extent)) {
            let prev_surface = read_prev_surface(prev_pixel);
            if (compare_flat_normals(surface.flat_normal, prev_surface.flat_normal) < 0.5) {
                continue;
            }
            let projected_distance = length(pos_world - prev_camera.position);
            if (compare_depths(prev_surface.depth, projected_distance) < 0.5) {
                continue;
            }
            let w = prev_weights[i];
            sum_weight += w;
            let illumination = w * textureLoad(prev_input, prev_pixel, 0).xyz;
            let luminocity = dot(illumination, LUMA);
            sum_ilm += vec4<f32>(illumination, luminocity * luminocity);
        }
    }

    let cur_illumination = textureLoad(input, pixel, 0).xyz;
    let cur_luminocity = dot(cur_illumination, LUMA);
    var mixed_ilm = vec4<f32>(cur_illumination, cur_luminocity * cur_luminocity);
    if (sum_weight > MIN_WEIGHT) {
        let prev_ilm = sum_ilm / w4(sum_weight);
        mixed_ilm = mix(mixed_ilm, prev_ilm, sum_weight * (1.0 - params.temporal_weight));
    }
    textureStore(output, global_id.xy, mixed_ilm);
}

const GAUSSIAN_WEIGHTS = vec2<f32>(0.44198, 0.27901);
const SIGMA_L: f32 = 4.0;
const EPSILON: f32 = 0.001;

fn compare_luminance(a_lum: f32, b_lum: f32, variance: f32) -> f32 {
    return exp(-abs(a_lum - b_lum) / (SIGMA_L * variance + EPSILON));
}

fn w4(w: f32) -> vec4<f32> {
    return vec4<f32>(vec3<f32>(w), w * w);
}

@compute @workgroup_size(8, 8)
fn atrous3x3(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let center = vec2<i32>(global_id.xy);
    if (any(center >= params.extent)) {
        return;
    }

    let center_ilm = textureLoad(input, center, 0);
    let center_luma = dot(center_ilm.xyz, LUMA);
    let variance = sqrt(center_ilm.w);
    let center_suf = read_surface(center);
    var sum_weight = GAUSSIAN_WEIGHTS[0] * GAUSSIAN_WEIGHTS[0];
    var sum_ilm = w4(sum_weight) * center_ilm;

    for (var yy=-1; yy<=1; yy+=1) {
        for (var xx=-1; xx<=1; xx+=1) {
            let p = center + vec2<i32>(xx, yy) * (1 << params.iteration);
            if (all(p == center) || any(p < vec2<i32>(0)) || any(p >= params.extent)) {
                continue;
            }

            //TODO: store in group-shared memory
            let surface = read_surface(p);
            var weight = GAUSSIAN_WEIGHTS[abs(xx)] * GAUSSIAN_WEIGHTS[abs(yy)];
            //TODO: make it stricter on higher iterations
            weight *= compare_flat_normals(surface.flat_normal, center_suf.flat_normal);
            //Note: should we use a projected depth instead of the surface one?
            weight *= compare_depths(surface.depth, center_suf.depth);
            let other_ilm = textureLoad(input, p, 0);
            weight *= compare_luminance(center_luma, dot(other_ilm.xyz, LUMA), variance);
            sum_ilm += w4(weight) * other_ilm;
            sum_weight += weight;
        }
    }

    let filtered_ilm = select(center_ilm, sum_ilm / w4(sum_weight), sum_weight > MIN_WEIGHT);
    textureStore(output, global_id.xy, filtered_ilm);
}
