struct CameraParams {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>,
    fov: vec2<f32>,
    film_offset: vec2<f32>,
    target_size: vec2<u32>,
}

struct Surface {
    basis: vec4<f32>,
    flat_normal: vec3<f32>,
    depth: f32,
    view_dir: vec3<f32>,
    diffuse_albedo: vec3<f32>,
    specular_f0: vec3<f32>,
    roughness: f32,
}

struct Params {
    extent: vec2<i32>,
    temporal_weight: f32,
    iteration: u32,
    use_motion_vectors: u32,
}

const DEBUG_MODE: bool = true;
const MAX_LOCAL_LIGHTS: u32 = 8u;
const MAX_LOCAL_LIGHTS_LEN: u32 = 8u;
const MAX_JOINTS_PER_DRAW: u32 = 64u;
const MAX_JOINTS_PER_DRAW_LEN: u32 = 64u;
const DebugMode_Final: u32 = 0u;
const DebugMode_Depth: u32 = 1u;
const DebugMode_DiffuseAlbedoTexture: u32 = 2u;
const DebugMode_DiffuseAlbedoFactor: u32 = 3u;
const DebugMode_NormalTexture: u32 = 4u;
const DebugMode_NormalScale: u32 = 5u;
const DebugMode_GeometryNormal: u32 = 6u;
const DebugMode_ShadingNormal: u32 = 7u;
const DebugMode_Motion: u32 = 8u;
const DebugMode_HitConsistency: u32 = 9u;
const DebugMode_SampleReuse: u32 = 10u;
const DebugMode_Roughness: u32 = 11u;
const DebugMode_SpecularF0: u32 = 12u;
const DebugMode_Emissive: u32 = 13u;
const DebugMode_Variance: u32 = 15u;
const DebugDrawFlags_SPACE: u32 = 1u;
const DebugDrawFlags_GEOMETRY: u32 = 2u;
const DebugDrawFlags_RESTIR: u32 = 4u;
const DebugTextureFlags_ALBEDO: u32 = 1u;
const DebugTextureFlags_NORMAL: u32 = 2u;
const DebugTextureFlags_METALLIC_ROUGHNESS: u32 = 4u;
const DebugTextureFlags_EMISSIVE: u32 = 8u;
const VFLIP: vec2<f32> = vec2<f32>(1f, -1f);
const MOTION_SCALE: f32 = 0.02f;
const USE_MOTION_VECTORS: bool = true;
const WRITE_DEBUG_IMAGE: bool = DEBUG_MODE;
const SIGMA_N: f32 = 4f;
const LUMA: vec3<f32> = vec3<f32>(0.2126f, 0.7152f, 0.0722f);
const MIN_WEIGHT: f32 = 0.01f;
const GAUSSIAN_WEIGHTS: vec2<f32> = vec2<f32>(0.44198f, 0.27901f);
const SIGMA_L: f32 = 4f;
const EPSILON: f32 = 0.001f;

var<uniform> camera: CameraParams;
var<uniform> prev_camera: CameraParams;
var<uniform> params: Params;
var t_depth: texture_2d<f32>;
var t_prev_depth: texture_2d<f32>;
var t_flat_normal: texture_2d<f32>;
var t_prev_flat_normal: texture_2d<f32>;
var t_motion: texture_2d<f32>;
var input: texture_2d<f32>;
var output: texture_storage_2d<rgba16float,read_write>;

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return (v + (vec3(2f) * cross(q.xyz, (cross(q.xyz, v) + (vec3(q.w) * v)))));
}

fn qinv(q_1: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-(q_1.xyz), q_1.w);
}

fn make_quat(m: mat3x3<f32>) -> vec4<f32> {
    var q_2: vec4<f32>;
    var dif10: f32;
    var omm22: f32;
    var sum10: f32;
    var opm22: f32;

    q_2 = vec4<f32>();
    if (m[2].z < 0f) {
        dif10 = (m[1].y - m[0].x);
        omm22 = (1f - m[2].z);
        let _e18 = dif10;
        if (_e18 < 0f) {
            let _e21 = omm22;
            let _e22 = dif10;
            q_2 = vec4<f32>((_e21 - _e22), (m[0].y + m[1].x), (m[0].z + m[2].x), (m[1].z - m[2].y));
        } else {
            let _e45 = omm22;
            let _e46 = dif10;
            q_2 = vec4<f32>((m[0].y + m[1].x), (_e45 + _e46), (m[1].z + m[2].y), (m[2].x - m[0].z));
        }
    } else {
        sum10 = (m[1].y + m[0].x);
        opm22 = (1f + m[2].z);
        let _e70 = sum10;
        if (_e70 < 0f) {
            let _e83 = opm22;
            let _e84 = sum10;
            q_2 = vec4<f32>((m[0].z + m[2].x), (m[1].z + m[2].y), (_e83 - _e84), (m[0].y - m[1].x));
        } else {
            let _e107 = opm22;
            let _e108 = sum10;
            q_2 = vec4<f32>((m[1].z - m[2].y), (m[2].x - m[0].z), (m[0].y - m[1].x), (_e107 + _e108));
        }
    }
    let _e111 = q_2;
    return normalize(_e111);
}

fn shortest_arc_quat(a: vec3<f32>, b: vec3<f32>) -> vec4<f32> {
    if (dot(a, b) < -(0.99999f)) {
        return select(vec4<f32>(1f, 0f, 0f, 0f), vec4<f32>(0f, 1f, 0f, 0f), (abs(a.x) > abs(a.y)));
    } else {
        return normalize(vec4<f32>(cross(a, b), (1f + dot(a, b))));
    }
}

fn get_ray_direction_at(cp: CameraParams, film_pos: vec2<f32>) -> vec3<f32> {
    var half_size: vec2<f32>;
    var ndc: vec2<f32>;
    var local_dir: vec3<f32>;

    half_size = (vec2(0.5f) * vec2<f32>(cp.target_size));
    let _e8 = half_size;
    let _e10 = half_size;
    ndc = ((film_pos - _e8) / _e10);
    let _e15 = ndc;
    local_dir = vec3<f32>((cp.film_offset + ((VFLIP * _e15) * tan((vec2(0.5f) * cp.fov)))), -(1f));
    let _e29 = local_dir;
    let _e30 = qrot(cp.orientation, _e29);
    return normalize(_e30);
}

fn get_projected_pixel_float(cp_1: CameraParams, point: vec3<f32>) -> vec2<f32> {
    var local_dir_1: vec3<f32>;
    var slope: vec2<f32>;
    var ndc_1: vec2<f32>;
    var half_size_1: vec2<f32>;

    let _e3 = qinv(cp_1.orientation);
    let _e6 = qrot(_e3, (point - cp_1.position));
    local_dir_1 = _e6;
    let _e9 = local_dir_1.z;
    if (_e9 >= 0f) {
        return vec2(-(1f));
    }
    let _e15 = local_dir_1;
    let _e18 = local_dir_1.z;
    slope = (_e15.xy / vec2(-(_e18)));
    let _e24 = slope;
    ndc_1 = ((VFLIP * (_e24 - cp_1.film_offset)) / tan((vec2(0.5f) * cp_1.fov)));
    half_size_1 = (vec2(0.5f) * vec2<f32>(cp_1.target_size));
    let _e41 = ndc_1;
    let _e45 = half_size_1;
    return ((_e41 + vec2(1f)) * _e45);
}

fn get_ray_direction(cp_2: CameraParams, pixel_1: vec2<i32>) -> vec3<f32> {
    let _e6 = get_ray_direction_at(cp_2, (vec2<f32>(pixel_1) + vec2(0.5f)));
    return _e6;
}

fn get_projected_pixel(cp_3: CameraParams, point_1: vec3<f32>) -> vec2<i32> {
    let _e2 = get_projected_pixel_float(cp_3, point_1);
    return vec2<i32>(_e2);
}

fn compare_flat_normals(a_1: vec3<f32>, b_1: vec3<f32>) -> f32 {
    return pow(max(0f, dot(a_1, b_1)), SIGMA_N);
}

fn compare_depths(a_2: f32, b_2: f32) -> f32 {
    return (1f - smoothstep(0f, 100f, abs((a_2 - b_2))));
}

fn compare_surfaces(a_3: Surface, b_3: Surface) -> f32 {
    var r_normal: f32;
    var r_depth: f32;

    let _e4 = compare_flat_normals(a_3.flat_normal, b_3.flat_normal);
    r_normal = _e4;
    let _e8 = compare_depths(a_3.depth, b_3.depth);
    r_depth = _e8;
    let _e10 = r_normal;
    let _e11 = r_depth;
    return (_e10 * _e11);
}

fn read_surface(pixel_2: vec2<i32>) -> Surface {
    var surface_2: Surface;

    surface_2 = Surface();
    let _e15 = textureLoad(t_flat_normal, pixel_2, 0i);
    surface_2.flat_normal = normalize(_e15.xyz);
    let _e20 = textureLoad(t_depth, pixel_2, 0i);
    surface_2.depth = _e20.x;
    let _e22 = surface_2;
    return _e22;
}

fn read_prev_surface(pixel_3: vec2<i32>) -> Surface {
    var surface_3: Surface;

    surface_3 = Surface();
    let _e15 = textureLoad(t_prev_flat_normal, pixel_3, 0i);
    surface_3.flat_normal = normalize(_e15.xyz);
    let _e20 = textureLoad(t_prev_depth, pixel_3, 0i);
    surface_3.depth = _e20.x;
    let _e22 = surface_3;
    return _e22;
}

fn get_prev_pixel(pixel_4: vec2<i32>, pos_world_1: vec3<f32>) -> vec2<f32> {
    var motion: vec2<f32>;

    let _e14 = params.use_motion_vectors;
    if (USE_MOTION_VECTORS && (_e14 != 0u)) {
        let _e19 = textureLoad(t_motion, pixel_4, 0i);
        motion = (_e19.xy / vec2(MOTION_SCALE));
        let _e29 = motion;
        return ((vec2<f32>(pixel_4) + vec2(0.5f)) + _e29);
    } else {
        let _e31 = prev_camera;
        let _e32 = get_projected_pixel_float(_e31, pos_world_1);
        return _e32;
    }
}

fn compare_luminance(a_lum: f32, b_lum: f32, variance_1: f32) -> f32 {
    return exp((-(abs((a_lum - b_lum))) / ((SIGMA_L * variance_1) + EPSILON)));
}

fn w4(w_1: f32) -> vec4<f32> {
    return vec4<f32>(vec3(w_1), (w_1 * w_1));
}

@compute @workgroup_size(8, 8, 1) 
fn temporal_accum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var pixel: vec2<i32>;
    var surface: Surface;
    var pos_world: vec3<f32>;
    var center_pixel: vec2<f32>;
    var prev_pixels: array<vec2<i32>, 4>;
    var w_bot_right: vec2<f32>;
    var prev_weights: vec4<f32>;
    var sum_weight: f32;
    var sum_ilm: vec4<f32>;
    var i: i32;
    var prev_pixel: vec2<i32>;
    var prev_surface: Surface;
    var projected_distance: f32;
    var w: f32;
    var illumination: vec3<f32>;
    var luminocity: f32;
    var cur_illumination: vec3<f32>;
    var cur_luminocity: f32;
    var mixed_ilm: vec4<f32>;
    var prev_ilm: vec4<f32>;

    pixel = vec2<i32>(global_id.xy);
    let _e14 = pixel;
    let _e16 = params.extent;
    if any((_e14 >= _e16)) {
        return;
    }
    let _e19 = pixel;
    let _e20 = read_surface(_e19);
    surface = _e20;
    let _e23 = camera.position;
    let _e25 = surface.depth;
    let _e26 = camera;
    let _e27 = pixel;
    let _e28 = get_ray_direction(_e26, _e27);
    pos_world = (_e23 + (vec3(_e25) * _e28));
    let _e33 = pixel;
    let _e34 = pos_world;
    let _e35 = get_prev_pixel(_e33, _e34);
    center_pixel = _e35;
    let _e38 = center_pixel.x;
    let _e42 = center_pixel.y;
    let _e48 = center_pixel.x;
    let _e52 = center_pixel.y;
    let _e58 = center_pixel.x;
    let _e62 = center_pixel.y;
    let _e68 = center_pixel.x;
    let _e72 = center_pixel.y;
    prev_pixels = array<vec2<i32>, 4>(vec2<i32>(vec2<f32>((_e38 - 0.5f), (_e42 - 0.5f))), vec2<i32>(vec2<f32>((_e48 + 0.5f), (_e52 - 0.5f))), vec2<i32>(vec2<f32>((_e58 + 0.5f), (_e62 + 0.5f))), vec2<i32>(vec2<f32>((_e68 - 0.5f), (_e72 + 0.5f))));
    let _e79 = center_pixel;
    w_bot_right = fract((_e79 + vec2(0.5f)));
    let _e87 = w_bot_right.x;
    let _e91 = w_bot_right.y;
    let _e95 = w_bot_right.x;
    let _e98 = w_bot_right.y;
    let _e102 = w_bot_right.x;
    let _e104 = w_bot_right.y;
    let _e108 = w_bot_right.x;
    let _e111 = w_bot_right.y;
    prev_weights = vec4<f32>(((1f - _e87) * (1f - _e91)), (_e95 * (1f - _e98)), (_e102 * _e104), ((1f - _e108) * _e111));
    sum_weight = 0f;
    sum_ilm = vec4(0f);
    let _e121 = params.temporal_weight;
    if (_e121 != 1f) {
        i = 0i;
        loop {
            let _e127 = i;
            if (_e127 < 4i) {
            } else {
                break;
            }
            let _e129 = i;
            let _e132 = prev_pixels[u32(_e129)];
            prev_pixel = _e132;
            let _e134 = prev_pixel;
            let _e139 = prev_pixel;
            let _e141 = params.extent;
            if (all((_e134 >= vec2(0i))) && all((_e139 < _e141))) {
                let _e145 = prev_pixel;
                let _e146 = read_prev_surface(_e145);
                prev_surface = _e146;
                let _e149 = surface.flat_normal;
                let _e151 = prev_surface.flat_normal;
                let _e152 = compare_flat_normals(_e149, _e151);
                if (_e152 < 0.5f) {
                    continue;
                }
                let _e155 = pos_world;
                let _e157 = prev_camera.position;
                projected_distance = length((_e155 - _e157));
                let _e162 = prev_surface.depth;
                let _e163 = projected_distance;
                let _e164 = compare_depths(_e162, _e163);
                if (_e164 < 0.5f) {
                    continue;
                }
                let _e167 = i;
                let _e170 = prev_weights[u32(_e167)];
                w = _e170;
                let _e172 = w;
                let _e173 = sum_weight;
                sum_weight = (_e173 + _e172);
                let _e175 = w;
                let _e176 = prev_pixel;
                let _e178 = textureLoad(input, _e176, 0i);
                illumination = (vec3(_e175) * _e178.xyz);
                let _e183 = illumination;
                luminocity = dot(_e183, LUMA);
                let _e187 = illumination;
                let _e188 = luminocity;
                let _e189 = luminocity;
                let _e192 = sum_ilm;
                sum_ilm = (_e192 + vec4<f32>(_e187, (_e188 * _e189)));
            }
            continuing {
                let _e194 = i;
                i = (_e194 + 1i);
            }
        }
    }
    let _e197 = pixel;
    let _e198 = textureLoad(output, _e197);
    cur_illumination = _e198.xyz;
    let _e201 = cur_illumination;
    cur_luminocity = dot(_e201, LUMA);
    let _e205 = cur_illumination;
    let _e206 = cur_luminocity;
    let _e207 = cur_luminocity;
    mixed_ilm = vec4<f32>(_e205, (_e206 * _e207));
    let _e211 = sum_weight;
    if (_e211 > MIN_WEIGHT) {
        let _e214 = sum_ilm;
        let _e215 = sum_weight;
        let _e218 = sum_weight;
        let _e219 = sum_weight;
        prev_ilm = (_e214 / vec4<f32>(vec3(_e215), max(0.001f, (_e218 * _e219))));
        let _e225 = mixed_ilm;
        let _e226 = prev_ilm;
        let _e227 = sum_weight;
        let _e230 = params.temporal_weight;
        mixed_ilm = mix(_e225, _e226, (_e227 * (1f - _e230)));
    }
    let _e234 = pixel;
    let _e235 = mixed_ilm;
    textureStore(output, _e234, _e235);
}

@compute @workgroup_size(8, 8, 1) 
fn atrous_filter(@builtin(global_invocation_id) global_id_1: vec3<u32>) {
    var center: vec2<i32>;
    var center_ilm: vec4<f32>;
    var center_luma: f32;
    var center_suf: Surface;
    var filtered_ilm: vec4<f32>;
    var yy: i32;
    var xx: i32;
    var p: vec2<i32>;
    var surface_1: Surface;
    var weight: f32;
    var other_ilm: vec4<f32>;
    var variance: f32;

    center = vec2<i32>(global_id_1.xy);
    let _e14 = center;
    let _e16 = params.extent;
    if any((_e14 >= _e16)) {
        return;
    }
    let _e19 = center;
    let _e21 = textureLoad(input, _e19, 0i);
    center_ilm = _e21;
    let _e23 = center_ilm;
    center_luma = dot(_e23.xyz, LUMA);
    let _e28 = center;
    let _e29 = read_surface(_e28);
    center_suf = _e29;
    let _e31 = center_ilm;
    filtered_ilm = _e31;
    yy = -(1i);
    loop {
        let _e37 = yy;
        if (_e37 <= 1i) {
        } else {
            break;
        }
        xx = -(1i);
        loop {
            let _e43 = xx;
            if (_e43 <= 1i) {
            } else {
                break;
            }
            let _e45 = center;
            let _e46 = xx;
            let _e47 = yy;
            let _e51 = params.iteration;
            p = (_e45 + (vec2<i32>(_e46, _e47) * vec2((1i << _e51))));
            let _e57 = p;
            let _e58 = center;
            let _e61 = p;
            let _e67 = p;
            let _e69 = params.extent;
            if ((all((_e57 == _e58)) || any((_e61 < vec2(0i)))) || any((_e67 >= _e69))) {
                continue;
            }
            let _e73 = p;
            let _e74 = read_surface(_e73);
            surface_1 = _e74;
            let _e77 = xx;
            let _e82 = yy;
            weight = (GAUSSIAN_WEIGHTS[u32(abs(_e77))] * GAUSSIAN_WEIGHTS[u32(abs(_e82))]);
            let _e89 = surface_1.flat_normal;
            let _e91 = center_suf.flat_normal;
            let _e92 = compare_flat_normals(_e89, _e91);
            let _e93 = weight;
            weight = (_e93 * _e92);
            let _e96 = surface_1.depth;
            let _e98 = center_suf.depth;
            let _e99 = compare_depths(_e96, _e98);
            let _e100 = weight;
            weight = (_e100 * _e99);
            let _e102 = p;
            let _e104 = textureLoad(input, _e102, 0i);
            other_ilm = _e104;
            let _e107 = center_ilm.w;
            let _e109 = other_ilm.w;
            variance = sqrt(max(_e107, _e109));
            let _e113 = center_luma;
            let _e114 = other_ilm;
            let _e118 = variance;
            let _e119 = compare_luminance(_e113, dot(_e114.xyz, LUMA), _e118);
            let _e120 = weight;
            weight = (_e120 * _e119);
            let _e122 = weight;
            let _e123 = w4(_e122);
            let _e124 = other_ilm;
            let _e125 = center_ilm;
            let _e128 = filtered_ilm;
            filtered_ilm = (_e128 + (_e123 * (_e124 - _e125)));
            continuing {
                let _e130 = xx;
                xx = (_e130 + 1i);
            }
        }
        continuing {
            let _e133 = yy;
            yy = (_e133 + 1i);
        }
    }
    let _e137 = filtered_ilm;
    textureStore(output, global_id_1.xy, _e137);
}
