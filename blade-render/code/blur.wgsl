#include "camera.inc.wgsl"
#include "quaternion.inc.wgsl"
#include "surface.inc.wgsl"

// Spatio-temporal variance-guided filtering
// https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf

// Note: using "ilm" in place of "illumination and the 2nd moment of its luminanc"

struct Params {
    extent: vec2<i32>,
    temporal_weight: f32,
    iteration: u32,
}

var<uniform> camera: CameraParams;
var<uniform> prev_camera: CameraParams;
var<uniform> params: Params;
var t_flat_normal: texture_2d<f32>;
var t_prev_flat_normal: texture_2d<f32>;
var t_depth: texture_2d<f32>;
var t_prev_depth: texture_2d<f32>;
var input: texture_2d<f32>;
var prev_input: texture_2d<f32>;
var output: texture_storage_2d<rgba16float, write>;

const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

fn get_projected_pixel_quad(cp: CameraParams, point: vec3<f32>) -> array<vec2<i32>, 4> {
    let pixel = get_projected_pixel_float(cp, point);
    return array<vec2<i32>, 4>(
        vec2<i32>(vec2<f32>(pixel.x - 0.5, pixel.y - 0.5)),
        vec2<i32>(vec2<f32>(pixel.x + 0.5, pixel.y - 0.5)),
        vec2<i32>(vec2<f32>(pixel.x + 0.5, pixel.y + 0.5)),
        vec2<i32>(vec2<f32>(pixel.x - 0.5, pixel.y + 0.5)),
    );
}

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
    let cur_illumination = textureLoad(input, pixel, 0).xyz;
    let surface = read_surface(pixel);
    let pos_world = camera.position + surface.depth * get_ray_direction(camera, pixel);
    // considering all samples in 2x2 quad, to help with edges
    var prev_pixels = get_projected_pixel_quad(prev_camera, pos_world);
    var best_index = 0;
    var best_weight = 0.0;
    //TODO: optimize depth load with a gather operation
    for (var i = 0; i < 4; i += 1) {
        let prev_pixel = prev_pixels[i];
        if (all(prev_pixel >= vec2<i32>(0)) && all(prev_pixel < params.extent)) {
            let prev_surface = read_prev_surface(prev_pixel);
            let projected_distance = length(pos_world - prev_camera.position);
            let weight = compare_flat_normals(surface.flat_normal, prev_surface.flat_normal)
                * compare_depths(surface.depth, projected_distance);
            if (weight > best_weight) {
                best_index = i;
                best_weight = weight;
            }
        }
    }

    let luminocity = dot(cur_illumination, LUMA);
    var mixed_ilm = vec4<f32>(cur_illumination, luminocity * luminocity);
    if (best_weight > 0.01) {
        let prev_ilm = textureLoad(prev_input, prev_pixels[best_index], 0);
        mixed_ilm = mix(mixed_ilm, prev_ilm, best_weight * (1.0 - params.temporal_weight));
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

    let filtered_ilm = sum_ilm / w4(sum_weight);
    textureStore(output, global_id.xy, filtered_ilm);
}
