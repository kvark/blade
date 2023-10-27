#include "camera.inc.wgsl"
#include "quaternion.inc.wgsl"
#include "surface.inc.wgsl"

// Spatio-temporal variance-guided filtering
// https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf

struct Params {
    extent: vec2<i32>,
    temporal_weight: f32,
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
    let cur_radiance = textureLoad(input, pixel, 0).xyz;
    let surface = read_surface(pixel);
    let pos_world = camera.position + surface.depth * get_ray_direction(camera, pixel);
    let prev_pixel = get_projected_pixel(prev_camera, pos_world);
    var prev_radiance = cur_radiance;
    var history_weight = 1.0 - params.temporal_weight;
    if (all(prev_pixel >= vec2<i32>(0)) && all(prev_pixel < params.extent)) {
        prev_radiance = textureLoad(prev_input, prev_pixel, 0).xyz;
        let prev_surface = read_prev_surface(prev_pixel);
        let projected_distance = length(pos_world - prev_camera.position);
        history_weight *= compare_flat_normals(surface.flat_normal, prev_surface.flat_normal);
        history_weight *= compare_depths(surface.depth, projected_distance);
    }
    let radiance = mix(cur_radiance, prev_radiance, history_weight);
    textureStore(output, global_id.xy, vec4<f32>(radiance, 0.0));
}

const gaussian_weights = vec2<f32>(0.44198, 0.27901);

@compute @workgroup_size(8, 8)
fn atrous(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let center = vec2<i32>(global_id.xy);
    if (any(center >= params.extent)) {
        return;
    }

    let center_radiance = textureLoad(input, center, 0).xyz;
    let center_suf = read_surface(center);
    var sum_weight = gaussian_weights[0] * gaussian_weights[0];
    var sum_radiance = center_radiance * sum_weight;

    for (var yy=-1; yy<=1; yy+=1) {
        for (var xx=-1; xx<=1; xx+=1) {
            let p = center + vec2<i32>(xx, yy);
            if (all(p == center) || any(p < vec2<i32>(0)) || any(p >= params.extent)) {
                continue;
            }

            //TODO: store in group-shared memory
            let surface = read_surface(p);
            var weight = gaussian_weights[abs(xx)] * gaussian_weights[abs(yy)];
            //weight *= compare_surfaces(center_suf, surface);
            let radiance = textureLoad(input, p, 0).xyz;
            sum_radiance += weight * radiance;
            sum_weight += weight;
        }
    }

    let radiance = sum_radiance / sum_weight;
    textureStore(output, global_id.xy, vec4<f32>(radiance, 0.0));
}
