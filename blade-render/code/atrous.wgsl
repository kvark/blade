#include "surface.inc.wgsl"

struct Params {
    extent: vec2<i32>,
}

var<uniform> params: Params;
var t_flat_normal: texture_2d<f32>;
var t_depth: texture_2d<f32>;
var input: texture_2d<f32>;
var output: texture_storage_2d<rgba16float, write>;

fn read_surface(pixel: vec2<i32>) -> Surface {
    var surface = Surface();
    surface.flat_normal = normalize(textureLoad(t_flat_normal, pixel, 0).xyz);
    surface.depth = textureLoad(t_depth, pixel, 0).x;
    return surface;
}

const gaussian_weights = vec2<f32>(0.44198, 0.27901);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
            weight *= compare_surfaces(center_suf, surface);
            let radiance = textureLoad(input, p, 0).xyz;
            sum_radiance += weight * radiance;
            sum_weight += weight;
        }
    }

    let radiance = sum_radiance / sum_weight;
    textureStore(output, global_id.xy, vec4<f32>(radiance, 0.0));
}
