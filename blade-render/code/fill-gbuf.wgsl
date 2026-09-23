enable wgpu_ray_query;
enable wgpu_binding_array;
enable wgpu_ray_tracing_pipeline;

struct Vertex {
    position: vec3<f32>,
    bitangent_sign: f32,
    tex_coords: vec2<f32>,
    normal: u32,
    tangent: u32,
}

struct CameraParams {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>,
    fov: vec2<f32>,
    film_offset: vec2<f32>,
    target_size: vec2<u32>,
}

struct DebugParams {
    view_mode: u32,
    draw_flags: u32,
    texture_flags: u32,
    pad: u32,
    mouse_pos: vec2<u32>,
}

struct DebugPoint {
    pos: vec3<f32>,
    color: u32,
}

struct DebugLine {
    a: DebugPoint,
    b: DebugPoint,
}

struct DebugVariance {
    color_sum: vec3<f32>,
    color2_sum: vec3<f32>,
    count: u32,
}

struct DebugEntry {
    custom_index: u32,
    depth: f32,
    tex_coords: vec2<f32>,
    base_color_texture: u32,
    normal_texture: u32,
    pad: vec2<u32>,
    position: vec3<f32>,
    flat_normal: vec3<f32>,
}

struct DebugBuffer {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
    capacity: u32,
    open: u32,
    variance: DebugVariance,
    entry: DebugEntry,
    lines: array<DebugLine>,
}

struct Material {
    diffuse_albedo: vec3<f32>,
    specular_f0: vec3<f32>,
    roughness: f32,
}

struct BrdfLobes {
    diffuse: f32,
    specular: vec3<f32>,
}

struct VertexBuffer {
    data: array<Vertex>,
}

struct IndexBuffer {
    data: array<u32>,
}

struct HitEntry {
    index_buf: u32,
    vertex_buf: u32,
    prev_vertex_buf: u32,
    flags: u32,
    geometry_to_object: mat4x3<f32>,
    prev_geometry_to_object: mat4x3<f32>,
    prev_object_to_world: mat4x3<f32>,
    base_color_texture: u32,
    base_color_factor: u32,
    normal_texture: u32,
    normal_scale: f32,
    metallic_roughness_texture: u32,
    metalness: f32,
    roughness: f32,
    emissive_texture: u32,
    emissive_factor: vec4<f32>,
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
const PI: f32 = 3.1415925f;
const DIELECTRIC_F0: f32 = 0.04f;
const MIN_ROUGHNESS: f32 = 0.05f;
const LUMINOCITY_WEIGHTS: vec3<f32> = vec3<f32>(0.3f, 0.4f, 0.3f);
const MOTION_SCALE: f32 = 0.02f;
const USE_MOTION_VECTORS: bool = true;
const WRITE_DEBUG_IMAGE: bool = DEBUG_MODE;

var<storage, read_write> debug_buf: DebugBuffer;
var<storage> vertex_buffers: binding_array<VertexBuffer>;
var<storage> index_buffers: binding_array<IndexBuffer>;
var<storage> hit_entries: array<HitEntry>;
var textures: binding_array<texture_2d<f32>>;
var sampler_linear: sampler;
var<uniform> camera: CameraParams;
var<uniform> prev_camera: CameraParams;
var<uniform> debug: DebugParams;
var acc_struct: acceleration_structure;
var out_depth: texture_storage_2d<r32float,write>;
var out_flat_normal: texture_storage_2d<rgba8snorm,write>;
var out_basis: texture_storage_2d<rgba8snorm,write>;
var out_diffuse_albedo: texture_storage_2d<rgba8unorm,write>;
var out_specular_f0: texture_storage_2d<rgba8unorm,write>;
var out_emissive: texture_storage_2d<rgba16float,write>;
var out_motion: texture_storage_2d<rg16float,write>;
var out_debug: texture_storage_2d<rgba8unorm,write>;















fn decode_normal(raw: u32) -> vec3<f32> {
    return unpack4x8snorm(raw).xyz;
}

fn tangent_basis(n: vec3<f32>, transformed_tangent: vec3<f32>, bitangent_sign: f32, linear_sign: f32) -> mat3x3<f32> {
    var t: vec3<f32>;
    var b: vec3<f32>;

    t = normalize((transformed_tangent - (n * vec3(dot(n, transformed_tangent)))));
    let _e10 = t;
    b = ((normalize(cross(n, _e10)) * vec3(bitangent_sign)) * vec3(linear_sign));
    let _e18 = t;
    let _e19 = b;
    return mat3x3<f32>(_e18, _e19, n);
}

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

fn shortest_arc_quat(a: vec3<f32>, b_1: vec3<f32>) -> vec4<f32> {
    if (dot(a, b_1) < -(0.99999f)) {
        return select(vec4<f32>(1f, 0f, 0f, 0f), vec4<f32>(0f, 1f, 0f, 0f), (abs(a.x) > abs(a.y)));
    } else {
        return normalize(vec4<f32>(cross(a, b_1), (1f + dot(a, b_1))));
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

fn get_ray_direction(cp_2: CameraParams, pixel: vec2<i32>) -> vec3<f32> {
    let _e6 = get_ray_direction_at(cp_2, (vec2<f32>(pixel) + vec2(0.5f)));
    return _e6;
}

fn get_projected_pixel(cp_3: CameraParams, point_1: vec3<f32>) -> vec2<i32> {
    let _e2 = get_projected_pixel_float(cp_3, point_1);
    return vec2<i32>(_e2);
}

fn debug_line(a_1: vec3<f32>, b_2: vec3<f32>, color: u32) {
    var index: u32;

    let _e5 = debug_buf.open;
    if (_e5 != 0u) {
        let _e10 = atomicAdd((&debug_buf.instance_count), 1u);
        index = _e10;
        let _e12 = index;
        let _e14 = debug_buf.capacity;
        if (_e12 < _e14) {
            let _e17 = index;
            debug_buf.lines[_e17] = DebugLine(DebugPoint(a_1, color), DebugPoint(b_2, color));
        } else {
            let _e24 = atomicSub((&debug_buf.instance_count), 1u);
        }
    }
}

fn compute_luminocity(color_1: vec3<f32>) -> f32 {
    return dot(color_1, LUMINOCITY_WEIGHTS);
}

fn material_from_metallic_roughness(base_color: vec3<f32>, metalness: f32, roughness: f32) -> Material {
    var mat: Material;

    mat = Material();
    mat.diffuse_albedo = (base_color * vec3((1f - metalness)));
    mat.specular_f0 = mix(vec3(DIELECTRIC_F0), base_color, metalness);
    mat.roughness = roughness;
    let _e16 = mat;
    return _e16;
}

fn material_alpha(mat_1: Material) -> f32 {
    var r: f32;

    r = clamp(mat_1.roughness, MIN_ROUGHNESS, 1f);
    let _e7 = r;
    let _e8 = r;
    return (_e7 * _e8);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return (f0 + ((vec3(1f) - f0) * vec3(pow((1f - cos_theta), 5f))));
}

fn fresnel_schlick_scalar(cos_theta_1: f32, f0_1: f32) -> f32 {
    return (f0_1 + ((1f - f0_1) * pow((1f - cos_theta_1), 5f)));
}

fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    var a2: f32;
    var denom: f32;

    a2 = (alpha * alpha);
    let _e6 = a2;
    denom = (((n_dot_h * n_dot_h) * (_e6 - 1f)) + 1f);
    let _e13 = a2;
    let _e15 = denom;
    let _e17 = denom;
    return (_e13 / max(((PI * _e15) * _e17), 0.0000001f));
}

fn visibility_smith(n_dot_v: f32, n_dot_l: f32, alpha_1: f32) -> f32 {
    var a2_1: f32;
    var lambda_v: f32;
    var lambda_l: f32;

    a2_1 = (alpha_1 * alpha_1);
    let _e8 = a2_1;
    let _e11 = a2_1;
    lambda_v = (n_dot_l * sqrt((((n_dot_v * n_dot_v) * (1f - _e8)) + _e11)));
    let _e18 = a2_1;
    let _e21 = a2_1;
    lambda_l = (n_dot_v * sqrt((((n_dot_l * n_dot_l) * (1f - _e18)) + _e21)));
    let _e27 = lambda_v;
    let _e28 = lambda_l;
    return (0.5f / max((_e27 + _e28), 0.0000001f));
}

fn zero_brdf() -> BrdfLobes {
    return BrdfLobes(0f, vec3(0f));
}

fn is_brdf_black(lobes: BrdfLobes) -> bool {
    return ((lobes.diffuse <= 0f) && all((lobes.specular <= vec3(0f))));
}

fn evaluate_ambient(mat_2: Material) -> vec3<f32> {
    return ((mat_2.diffuse_albedo * (vec3(1f) - mat_2.specular_f0)) + mat_2.specular_f0);
}

fn specular_sampling_ratio(mat_3: Material) -> f32 {
    var diffuse: f32;
    var specular: f32;

    let _e3 = compute_luminocity(mat_3.diffuse_albedo);
    diffuse = _e3;
    let _e6 = compute_luminocity(mat_3.specular_f0);
    specular = _e6;
    let _e8 = specular;
    let _e9 = diffuse;
    let _e10 = specular;
    return clamp((_e8 / max((_e9 + _e10), 0.00001f)), 0.1f, 0.9f);
}

fn evaluate_brdf(mat_4: Material, normal_1: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> BrdfLobes {
    var n_dot_l_1: f32;
    var n_dot_v_1: f32;
    var half_dir: vec3<f32>;
    var n_dot_h_1: f32;
    var v_dot_h: f32;
    var alpha_2: f32;
    var fresnel: vec3<f32>;
    var specular_1: vec3<f32>;
    var k_diffuse: f32;

    n_dot_l_1 = dot(normal_1, light_dir);
    n_dot_v_1 = dot(normal_1, view_dir);
    let _e9 = n_dot_l_1;
    let _e12 = n_dot_v_1;
    if ((_e9 <= 0f) || (_e12 <= 0f)) {
        let _e16 = zero_brdf();
        return _e16;
    }
    half_dir = normalize((view_dir + light_dir));
    let _e20 = half_dir;
    n_dot_h_1 = max(dot(normal_1, _e20), 0f);
    let _e25 = half_dir;
    v_dot_h = max(dot(view_dir, _e25), 0f);
    let _e30 = material_alpha(mat_4);
    alpha_2 = _e30;
    let _e32 = v_dot_h;
    let _e34 = fresnel_schlick(_e32, mat_4.specular_f0);
    fresnel = _e34;
    let _e36 = n_dot_h_1;
    let _e37 = alpha_2;
    let _e38 = distribution_ggx(_e36, _e37);
    let _e39 = n_dot_v_1;
    let _e40 = n_dot_l_1;
    let _e41 = alpha_2;
    let _e42 = visibility_smith(_e39, _e40, _e41);
    let _e44 = fresnel;
    specular_1 = (vec3((_e38 * _e42)) * _e44);
    let _e49 = v_dot_h;
    let _e51 = fresnel_schlick_scalar(_e49, DIELECTRIC_F0);
    k_diffuse = (1f - _e51);
    let _e54 = k_diffuse;
    let _e55 = n_dot_l_1;
    let _e59 = specular_1;
    let _e60 = n_dot_l_1;
    return BrdfLobes(((_e54 * _e55) / PI), (_e59 * vec3(_e60)));
}

fn affine_linear(transform: mat4x3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(transform[0].xyz, transform[1].xyz, transform[2].xyz);
}

fn hit_winding(entry_1: HitEntry) -> f32 {
    return select(1f, -(1f), ((entry_1.flags & 1u) != 0u));
}

fn fetch_triangle_indices(entry_2: HitEntry, primitive_index: u32) -> vec3<u32> {
    var indices_1: vec3<u32>;

    indices_1 = (vec3((primitive_index * 3u)) + vec3<u32>(0u, 1u, 2u));
    if (entry_2.index_buf != ~(0u)) {
        let _e25 = indices_1.x;
        let _e27 = index_buffers[entry_2.index_buf].data[_e25];
        let _e32 = indices_1.y;
        let _e34 = index_buffers[entry_2.index_buf].data[_e32];
        let _e39 = indices_1.z;
        let _e41 = index_buffers[entry_2.index_buf].data[_e39];
        indices_1 = vec3<u32>(_e27, _e34, _e41);
    }
    let _e43 = indices_1;
    return _e43;
}

fn make_barycentrics(uv: vec2<f32>) -> vec3<f32> {
    var w: f32;

    w = ((1f - uv.x) - uv.y);
    let _e13 = w;
    return vec3<f32>(_e13, uv.x, uv.y);
}

fn sample_hit_material(entry_3: HitEntry, tex_coords_1: vec2<f32>, lod_1: f32, ignore_textures: u32) -> Material {
    var base_color_1: vec3<f32>;
    var metalness_1: f32;
    var roughness_1: f32;
    var mr: vec4<f32>;

    base_color_1 = unpack4x8unorm(entry_3.base_color_factor).xyz;
    if ((ignore_textures & DebugTextureFlags_ALBEDO) == 0u) {
        let _e20 = textureSampleLevel(textures[entry_3.base_color_texture], sampler_linear, tex_coords_1, lod_1);
        let _e22 = base_color_1;
        base_color_1 = (_e22 * _e20.xyz);
    }
    metalness_1 = entry_3.metalness;
    roughness_1 = entry_3.roughness;
    if ((ignore_textures & DebugTextureFlags_METALLIC_ROUGHNESS) == 0u) {
        let _e34 = textureSampleLevel(textures[entry_3.metallic_roughness_texture], sampler_linear, tex_coords_1, lod_1);
        mr = _e34;
        let _e37 = mr.y;
        let _e38 = roughness_1;
        roughness_1 = (_e38 * _e37);
        let _e41 = mr.z;
        let _e42 = metalness_1;
        metalness_1 = (_e42 * _e41);
    }
    let _e44 = base_color_1;
    let _e45 = metalness_1;
    let _e46 = roughness_1;
    let _e47 = material_from_metallic_roughness(_e44, _e45, _e46);
    return _e47;
}

fn sample_hit_emissive(entry_4: HitEntry, tex_coords_2: vec2<f32>, lod_2: f32, ignore_textures_1: u32) -> vec3<f32> {
    var emissive_1: vec3<f32>;

    emissive_1 = entry_4.emissive_factor.xyz;
    if ((ignore_textures_1 & DebugTextureFlags_EMISSIVE) == 0u) {
        let _e19 = textureSampleLevel(textures[entry_4.emissive_texture], sampler_linear, tex_coords_2, lod_2);
        let _e21 = emissive_1;
        emissive_1 = (_e21 * _e19.xyz);
    }
    let _e23 = emissive_1;
    return _e23;
}

fn sample_hit_normal_map(entry_5: HitEntry, tex_coords_3: vec2<f32>, lod_3: f32, ignore_textures_2: u32) -> vec3<f32> {
    var raw_unorm: vec2<f32>;
    var n_xy: vec2<f32>;

    if ((ignore_textures_2 & DebugTextureFlags_NORMAL) != 0u) {
        return vec3<f32>(0f, 0f, 1f);
    }
    let _e20 = textureSampleLevel(textures[entry_5.normal_texture], sampler_linear, tex_coords_3, lod_3);
    raw_unorm = _e20.xy;
    let _e25 = raw_unorm;
    n_xy = (vec2(entry_5.normal_scale) * ((vec2(2f) * _e25) - vec2(1f)));
    let _e34 = n_xy;
    let _e37 = n_xy;
    let _e38 = n_xy;
    return vec3<f32>(_e34, sqrt(max(0f, (1f - dot(_e37, _e38)))));
}

fn hit_normal(entry_6: HitEntry, object_to_world: mat4x3<f32>, normal_2: vec3<f32>) -> vec3<f32> {
    var linear: mat3x3<f32>;

    let _e9 = affine_linear(object_to_world);
    let _e11 = affine_linear(entry_6.geometry_to_object);
    linear = (_e9 * _e11);
    let _e14 = linear;
    return normalize((_e14 * normal_2));
}

fn hit_tangent_space(entry_7: HitEntry, object_to_world_1: mat4x3<f32>, normal_3: vec3<f32>, tangent: vec3<f32>, bitangent_sign_1: f32) -> mat3x3<f32> {
    var linear_1: mat3x3<f32>;
    var n_1: vec3<f32>;

    let _e11 = affine_linear(object_to_world_1);
    let _e13 = affine_linear(entry_7.geometry_to_object);
    linear_1 = (_e11 * _e13);
    let _e16 = hit_normal(entry_7, object_to_world_1, normal_3);
    n_1 = _e16;
    let _e18 = n_1;
    let _e19 = linear_1;
    let _e21 = linear_1;
    let _e24 = tangent_basis(_e18, (_e19 * tangent), bitangent_sign_1, sign(determinant(_e21)));
    return _e24;
}

fn debug_raw_normal(pos: vec3<f32>, normal_raw: u32, entry_8: HitEntry, object_to_world_2: mat4x3<f32>, debug_len_1: f32, color_2: u32) {
    var nw: vec3<f32>;

    let _e24 = decode_normal(normal_raw);
    let _e25 = hit_normal(entry_8, object_to_world_2, _e24);
    nw = _e25;
    let _e27 = nw;
    debug_line(pos, (pos + (vec3(debug_len_1) * _e27)), color_2);
}

@compute @workgroup_size(8, 4, 1) 
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var rq: ray_query;
    var ray_dir: vec3<f32>;
    var intersection: RayIntersection;
    var depth: f32;
    var basis: vec4<f32>;
    var flat_normal: vec3<f32>;
    var material: Material;
    var emissive: vec3<f32>;
    var motion: vec2<f32>;
    var enable_debug: bool;
    var entry: HitEntry;
    var indices: vec3<u32>;
    var vertices: array<Vertex, 3>;
    var prev_vertices: array<Vertex, 3>;
    var positions_object: mat3x3<f32>;
    var prev_positions_object: mat3x3<f32>;
    var positions: mat3x3<f32>;
    var barycentrics: vec3<f32>;
    var position_object: vec4<f32>;
    var tex_coords: vec2<f32>;
    var normal_geo: vec3<f32>;
    var tangent_geo: vec3<f32>;
    var lod: f32;
    var tangent_space_world: mat3x3<f32>;
    var normal_local: vec3<f32>;
    var normal: vec3<f32>;
    var hit_position: vec3<f32>;
    var normal_w: vec3<f32>;
    var tangent_w: vec3<f32>;
    var bitangent_w: vec3<f32>;
    var debug_len: f32;
    var poly_center: vec3<f32>;
    var reprojected: vec2<i32>;
    var barycentrics_pos_diff: vec3<f32>;
    var camera_projection_diff: vec2<f32>;
    var consistency: vec4<f32>;
    var prev_position_object: vec4<f32>;
    var prev_position: vec3<f32>;
    var prev_screen: vec2<f32>;

    let _e21 = camera.target_size;
    if any((global_id.xy >= _e21)) {
        return;
    }
    let _e26 = debug.view_mode;
    if (WRITE_DEBUG_IMAGE && (_e26 != DebugMode_Final)) {
        textureStore(out_debug, global_id.xy, vec4(0f));
    }
    let _e34 = camera;
    let _e37 = get_ray_direction(_e34, vec2<i32>(global_id.xy));
    ray_dir = _e37;
    let _e43 = camera.depth;
    let _e45 = camera.position;
    let _e46 = ray_dir;
    rayQueryInitialize((&rq), acc_struct, RayDesc(128u, 255u, 0f, _e43, _e45, _e46));
    let _e48 = rayQueryProceed((&rq));
    let _e49 = rayQueryGetCommittedIntersection((&rq));
    intersection = _e49;
    depth = 0f;
    basis = vec4(0f);
    flat_normal = vec3(0f);
    material = Material(vec3(1f), vec3(0f), 0f);
    emissive = vec3(0f);
    motion = vec2(0f);
    let _e74 = debug.mouse_pos;
    enable_debug = all((global_id.xy == _e74));
    let _e79 = intersection.kind;
    if (_e79 != 0u) {
        let _e83 = intersection.instance_custom_data;
        let _e85 = intersection.geometry_index;
        let _e88 = hit_entries[(_e83 + _e85)];
        entry = _e88;
        let _e91 = intersection.t;
        depth = _e91;
        let _e92 = entry;
        let _e94 = intersection.primitive_index;
        let _e95 = fetch_triangle_indices(_e92, _e94);
        indices = _e95;
        let _e98 = entry.vertex_buf;
        let _e102 = indices.x;
        let _e104 = vertex_buffers[_e98].data[_e102];
        let _e106 = entry.vertex_buf;
        let _e110 = indices.y;
        let _e112 = vertex_buffers[_e106].data[_e110];
        let _e114 = entry.vertex_buf;
        let _e118 = indices.z;
        let _e120 = vertex_buffers[_e114].data[_e118];
        vertices = array<Vertex, 3>(_e104, _e112, _e120);
        let _e124 = entry.prev_vertex_buf;
        let _e128 = indices.x;
        let _e130 = vertex_buffers[_e124].data[_e128];
        let _e132 = entry.prev_vertex_buf;
        let _e136 = indices.y;
        let _e138 = vertex_buffers[_e132].data[_e136];
        let _e140 = entry.prev_vertex_buf;
        let _e144 = indices.z;
        let _e146 = vertex_buffers[_e140].data[_e144];
        prev_vertices = array<Vertex, 3>(_e130, _e138, _e146);
        let _e150 = entry.geometry_to_object;
        let _e153 = vertices[0].position;
        let _e158 = vertices[1].position;
        let _e163 = vertices[2].position;
        positions_object = (_e150 * mat3x4<f32>(vec4<f32>(_e153, 1f), vec4<f32>(_e158, 1f), vec4<f32>(_e163, 1f)));
        let _e170 = entry.prev_geometry_to_object;
        let _e173 = prev_vertices[0].position;
        let _e178 = prev_vertices[1].position;
        let _e183 = prev_vertices[2].position;
        prev_positions_object = (_e170 * mat3x4<f32>(vec4<f32>(_e173, 1f), vec4<f32>(_e178, 1f), vec4<f32>(_e183, 1f)));
        let _e190 = intersection.object_to_world;
        let _e192 = positions_object[0];
        let _e196 = positions_object[1];
        let _e200 = positions_object[2];
        positions = (_e190 * mat3x4<f32>(vec4<f32>(_e192, 1f), vec4<f32>(_e196, 1f), vec4<f32>(_e200, 1f)));
        let _e206 = entry;
        let _e207 = hit_winding(_e206);
        let _e209 = positions[1];
        let _e212 = positions[0];
        let _e216 = positions[2];
        let _e219 = positions[0];
        flat_normal = (vec3(_e207) * normalize(cross((_e209.xyz - _e212.xyz), (_e216.xyz - _e219.xyz))));
        let _e227 = intersection.barycentrics;
        let _e228 = make_barycentrics(_e227);
        barycentrics = _e228;
        let _e230 = positions_object;
        let _e231 = barycentrics;
        position_object = vec4<f32>((_e230 * _e231), 1f);
        let _e238 = vertices[0].tex_coords;
        let _e241 = vertices[1].tex_coords;
        let _e244 = vertices[2].tex_coords;
        let _e246 = barycentrics;
        tex_coords = (mat3x2<f32>(_e238, _e241, _e244) * _e246);
        let _e251 = vertices[0].normal;
        let _e252 = decode_normal(_e251);
        let _e255 = vertices[1].normal;
        let _e256 = decode_normal(_e255);
        let _e259 = vertices[2].normal;
        let _e260 = decode_normal(_e259);
        let _e262 = barycentrics;
        normal_geo = normalize((mat3x3<f32>(_e252, _e256, _e260) * _e262));
        let _e268 = vertices[0].tangent;
        let _e269 = decode_normal(_e268);
        let _e272 = vertices[1].tangent;
        let _e273 = decode_normal(_e272);
        let _e276 = vertices[2].tangent;
        let _e277 = decode_normal(_e276);
        let _e279 = barycentrics;
        tangent_geo = normalize((mat3x3<f32>(_e269, _e273, _e277) * _e279));
        lod = 0f;
        let _e285 = entry;
        let _e287 = intersection.object_to_world;
        let _e288 = normal_geo;
        let _e289 = tangent_geo;
        let _e292 = vertices[0].bitangent_sign;
        let _e293 = hit_tangent_space(_e285, _e287, _e288, _e289, _e292);
        tangent_space_world = _e293;
        let _e295 = entry;
        let _e296 = tex_coords;
        let _e297 = lod;
        let _e299 = debug.texture_flags;
        let _e300 = sample_hit_normal_map(_e295, _e296, _e297, _e299);
        normal_local = _e300;
        let _e302 = tangent_space_world;
        let _e303 = normal_local;
        normal = (_e302 * _e303);
        let _e310 = normal;
        let _e312 = shortest_arc_quat(vec3<f32>(0f, 0f, 1f), normalize(_e310));
        basis = _e312;
        let _e314 = camera.position;
        let _e316 = intersection.t;
        let _e317 = ray_dir;
        hit_position = (_e314 + (vec3(_e316) * _e317));
        let _e322 = enable_debug;
        if _e322 {
            let _e326 = intersection.instance_custom_data;
            debug_buf.entry.custom_index = _e326;
            let _e330 = intersection.t;
            debug_buf.entry.depth = _e330;
            let _e333 = tex_coords;
            debug_buf.entry.tex_coords = _e333;
            let _e337 = entry.base_color_texture;
            debug_buf.entry.base_color_texture = _e337;
            let _e341 = entry.normal_texture;
            debug_buf.entry.normal_texture = _e341;
            let _e344 = hit_position;
            debug_buf.entry.position = _e344;
            let _e347 = flat_normal;
            debug_buf.entry.flat_normal = _e347;
        }
        let _e348 = enable_debug;
        let _e350 = debug.draw_flags;
        if (_e348 && ((_e350 & DebugDrawFlags_SPACE) != 0u)) {
            let _e358 = intersection.t;
            let _e361 = tangent_space_world[2];
            normal_w = (vec3((0.15f * _e358)) * _e361);
            let _e367 = intersection.t;
            let _e370 = tangent_space_world[0];
            tangent_w = (vec3((0.05f * _e367)) * _e370);
            let _e376 = intersection.t;
            let _e379 = tangent_space_world[1];
            bitangent_w = (vec3((0.05f * _e376)) * _e379);
            let _e383 = hit_position;
            let _e384 = hit_position;
            let _e385 = normal_w;
            debug_line(_e383, (_e384 + _e385), 16744448u);
            let _e388 = hit_position;
            let _e390 = tangent_w;
            let _e394 = hit_position;
            let _e395 = tangent_w;
            debug_line((_e388 - (vec3(0.5f) * _e390)), (_e394 + _e395), 8421631u);
            let _e398 = hit_position;
            let _e400 = bitangent_w;
            let _e404 = hit_position;
            let _e405 = bitangent_w;
            debug_line((_e398 - (vec3(0.5f) * _e400)), (_e404 + _e405), 8454016u);
        }
        let _e408 = enable_debug;
        let _e410 = debug.draw_flags;
        if (_e408 && ((_e410 & DebugDrawFlags_GEOMETRY) != 0u)) {
            let _e417 = intersection.t;
            debug_len = (_e417 * 0.2f);
            let _e422 = positions[0];
            let _e425 = positions[1];
            debug_line(_e422.xyz, _e425.xyz, 65535u);
            let _e429 = positions[1];
            let _e432 = positions[2];
            debug_line(_e429.xyz, _e432.xyz, 65535u);
            let _e436 = positions[2];
            let _e439 = positions[0];
            debug_line(_e436.xyz, _e439.xyz, 65535u);
            let _e443 = positions[0];
            let _e446 = positions[1];
            let _e450 = positions[2];
            poly_center = (((_e443.xyz + _e446.xyz) + _e450.xyz) / vec3(3f));
            let _e457 = poly_center;
            let _e458 = poly_center;
            let _e460 = debug_len;
            let _e462 = flat_normal;
            debug_line(_e457, (_e458 + (vec3((0.2f * _e460)) * _e462)), 16711935u);
            let _e468 = positions[0];
            let _e472 = vertices[0].normal;
            let _e473 = entry;
            let _e475 = intersection.object_to_world;
            let _e477 = debug_len;
            debug_raw_normal(_e468.xyz, _e472, _e473, _e475, (0.5f * _e477), 16776960u);
            let _e481 = positions[1];
            let _e485 = vertices[1].normal;
            let _e486 = entry;
            let _e488 = intersection.object_to_world;
            let _e490 = debug_len;
            debug_raw_normal(_e481.xyz, _e485, _e486, _e488, (0.5f * _e490), 16776960u);
            let _e494 = positions[2];
            let _e498 = vertices[2].normal;
            let _e499 = entry;
            let _e501 = intersection.object_to_world;
            let _e503 = debug_len;
            debug_raw_normal(_e494.xyz, _e498, _e499, _e501, (0.5f * _e503), 16776960u);
            let _e506 = hit_position;
            let _e507 = hit_position;
            let _e508 = debug_len;
            let _e509 = basis;
            let _e514 = qrot(_e509, vec3<f32>(1f, 0f, 0f));
            debug_line(_e506, (_e507 + (vec3(_e508) * _e514)), 255u);
            let _e519 = hit_position;
            let _e520 = hit_position;
            let _e521 = debug_len;
            let _e522 = basis;
            let _e527 = qrot(_e522, vec3<f32>(0f, 1f, 0f));
            debug_line(_e519, (_e520 + (vec3(_e521) * _e527)), 65280u);
            let _e532 = hit_position;
            let _e533 = hit_position;
            let _e534 = debug_len;
            let _e535 = basis;
            let _e540 = qrot(_e535, vec3<f32>(0f, 0f, 1f));
            debug_line(_e532, (_e533 + (vec3(_e534) * _e540)), 16711680u);
        }
        let _e545 = entry;
        let _e546 = tex_coords;
        let _e547 = lod;
        let _e549 = debug.texture_flags;
        let _e550 = sample_hit_material(_e545, _e546, _e547, _e549);
        material = _e550;
        let _e551 = entry;
        let _e552 = tex_coords;
        let _e553 = lod;
        let _e555 = debug.texture_flags;
        let _e556 = sample_hit_emissive(_e551, _e552, _e553, _e555);
        emissive = _e556;
        if WRITE_DEBUG_IMAGE {
            let _e559 = debug.view_mode;
            if (_e559 == DebugMode_DiffuseAlbedoTexture) {
                let _e564 = material.diffuse_albedo;
                textureStore(out_debug, global_id.xy, vec4<f32>(_e564, 0f));
            }
            let _e568 = debug.view_mode;
            if (_e568 == DebugMode_DiffuseAlbedoFactor) {
                let _e573 = entry.base_color_factor;
                textureStore(out_debug, global_id.xy, unpack4x8unorm(_e573));
            }
            let _e576 = debug.view_mode;
            if (_e576 == DebugMode_NormalTexture) {
                let _e580 = normal_local;
                textureStore(out_debug, global_id.xy, vec4<f32>(_e580, 0f));
            }
            let _e584 = debug.view_mode;
            if (_e584 == DebugMode_NormalScale) {
                let _e589 = entry.normal_scale;
                textureStore(out_debug, global_id.xy, vec4(_e589));
            }
            let _e592 = debug.view_mode;
            if (_e592 == DebugMode_Roughness) {
                let _e597 = material.roughness;
                textureStore(out_debug, global_id.xy, vec4(_e597));
            }
            let _e600 = debug.view_mode;
            if (_e600 == DebugMode_SpecularF0) {
                let _e605 = material.specular_f0;
                textureStore(out_debug, global_id.xy, vec4<f32>(_e605, 0f));
            }
            let _e609 = debug.view_mode;
            if (_e609 == DebugMode_Emissive) {
                let _e613 = emissive;
                textureStore(out_debug, global_id.xy, vec4<f32>(_e613, 0f));
            }
            let _e617 = debug.view_mode;
            if (_e617 == DebugMode_GeometryNormal) {
                let _e621 = normal_geo;
                textureStore(out_debug, global_id.xy, vec4<f32>(_e621, 0f));
            }
            let _e625 = debug.view_mode;
            if (_e625 == DebugMode_ShadingNormal) {
                let _e629 = normal;
                textureStore(out_debug, global_id.xy, vec4<f32>(_e629, 0f));
            }
            let _e633 = debug.view_mode;
            if (_e633 == DebugMode_HitConsistency) {
                let _e636 = camera;
                let _e637 = hit_position;
                let _e638 = get_projected_pixel(_e636, _e637);
                reprojected = _e638;
                let _e641 = intersection.object_to_world;
                let _e642 = position_object;
                let _e645 = hit_position;
                barycentrics_pos_diff = ((_e641 * _e642).xyz - _e645);
                let _e650 = reprojected;
                camera_projection_diff = (vec2<f32>(global_id.xy) - vec2<f32>(_e650));
                let _e654 = barycentrics_pos_diff;
                let _e656 = camera_projection_diff;
                consistency = vec4<f32>(length(_e654), length(_e656), 0f, 0f);
                let _e663 = consistency;
                textureStore(out_debug, global_id.xy, _e663);
            }
        }
        let _e664 = prev_positions_object;
        let _e665 = barycentrics;
        prev_position_object = vec4<f32>((_e664 * _e665), 1f);
        let _e671 = entry.prev_object_to_world;
        let _e672 = prev_position_object;
        prev_position = (_e671 * _e672).xyz;
        let _e676 = prev_camera;
        let _e677 = prev_position;
        let _e678 = get_projected_pixel_float(_e676, _e677);
        prev_screen = _e678;
        let _e680 = prev_screen;
        motion = ((_e680 - vec2<f32>(global_id.xy)) - vec2(0.5f));
        let _e689 = debug.view_mode;
        if (WRITE_DEBUG_IMAGE && (_e689 == DebugMode_Motion)) {
            let _e694 = motion;
            textureStore(out_debug, global_id.xy, vec4<f32>(vec3<f32>(((_e694 * vec2(MOTION_SCALE)) + vec2(0.5f)), 0f), 1f));
        }
    } else {
        let _e705 = enable_debug;
        if _e705 {
            debug_buf.entry = DebugEntry();
        }
    }
    let _e709 = depth;
    textureStore(out_depth, global_id.xy, vec4<f32>(_e709, 0f, 0f, 0f));
    let _e715 = basis;
    textureStore(out_basis, global_id.xy, _e715);
    let _e717 = flat_normal;
    textureStore(out_flat_normal, global_id.xy, vec4<f32>(_e717, 0f));
    let _e722 = material.diffuse_albedo;
    textureStore(out_diffuse_albedo, global_id.xy, vec4<f32>(_e722, 0f));
    let _e727 = material.specular_f0;
    let _e729 = material.roughness;
    textureStore(out_specular_f0, global_id.xy, vec4<f32>(_e727, _e729));
    let _e732 = emissive;
    textureStore(out_emissive, global_id.xy, vec4<f32>(_e732, 0f));
    let _e736 = motion;
    textureStore(out_motion, global_id.xy, vec4<f32>(vec3<f32>((_e736 * vec2(MOTION_SCALE)), 0f), 0f));
}
