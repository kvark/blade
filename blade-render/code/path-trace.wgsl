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

struct RandomState {
    seed: u32,
    index: u32,
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

struct Material {
    diffuse_albedo: vec3<f32>,
    specular_f0: vec3<f32>,
    roughness: f32,
}

struct BrdfLobes {
    diffuse: f32,
    specular: vec3<f32>,
}

struct BsdfSample {
    dir: vec3<f32>,
    pdf: f32,
}

struct EnvImportantSample {
    pixel: vec2<i32>,
    pdf: f32,
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

struct LightSample {
    radiance: vec3<f32>,
    pdf: f32,
    uv: vec2<f32>,
}

struct PathTraceParams {
    frame_index: u32,
    num_environment_samples: u32,
    num_brdf_samples: u32,
    max_bounces: u32,
    max_accumulated_samples: u32,
    t_start: f32,
    environment_importance_sampling: u32,
    reset_accumulation: u32,
    jitter_primary_rays: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct PathVertex {
    position: vec3<f32>,
    flat_normal: vec3<f32>,
    normal: vec3<f32>,
    material: Material,
    emissive: vec3<f32>,
}

struct PathRadiance {
    total: vec3<f32>,
    diffuse: vec3<f32>,
    specular: vec3<f32>,
    emissive: vec3<f32>,
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
const ROULETTE_START: u32 = 4u;
const MAX_RADIANCE: f32 = 1000000f;

var env_weights: texture_2d<f32>;
var<storage> vertex_buffers: binding_array<VertexBuffer>;
var<storage> index_buffers: binding_array<IndexBuffer>;
var<storage> hit_entries: array<HitEntry>;
var textures: binding_array<texture_2d<f32>>;
var sampler_linear: sampler;
var env_map: texture_2d<f32>;
var sampler_nearest: sampler;
var<uniform> camera: CameraParams;
var<uniform> parameters: PathTraceParams;
var acc_struct: acceleration_structure;
var accumulator: texture_storage_2d<rgba32float,read_write>;
var accumulator_diffuse: texture_storage_2d<rgba32float,read_write>;
var accumulator_specular: texture_storage_2d<rgba32float,read_write>;
var accumulator_emissive: texture_storage_2d<rgba32float,read_write>;















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

fn hash_jenkins(value: u32) -> u32 {
    var a_1: u32;

    a_1 = value;
    let _e2 = a_1;
    let _e5 = a_1;
    a_1 = ((_e2 + 2127912214u) + (_e5 << 12u));
    let _e9 = a_1;
    let _e12 = a_1;
    a_1 = ((_e9 ^ 3345072700u) ^ (_e12 >> 19u));
    let _e16 = a_1;
    let _e19 = a_1;
    a_1 = ((_e16 + 374761393u) + (_e19 << 5u));
    let _e23 = a_1;
    let _e26 = a_1;
    a_1 = ((_e23 + 3550635116u) ^ (_e26 << 9u));
    let _e30 = a_1;
    let _e33 = a_1;
    a_1 = ((_e30 + 4251993797u) + (_e33 << 3u));
    let _e37 = a_1;
    let _e40 = a_1;
    a_1 = ((_e37 ^ 3042594569u) ^ (_e40 >> 16u));
    let _e44 = a_1;
    return _e44;
}

fn rot32(x: u32, bits: u32) -> u32 {
    return ((x << bits) | (x >> (32u - bits)));
}

fn random_init(pixel_index: u32, frame_index: u32) -> RandomState {
    var rs: RandomState;

    rs = RandomState();
    let _e5 = hash_jenkins(pixel_index);
    rs.seed = (_e5 + frame_index);
    rs.index = 0u;
    let _e9 = rs;
    return _e9;
}

fn murmur3(rng_1: ptr<function, RandomState>) -> u32 {
    var c1: u32;
    var c2: u32;
    var r1: u32;
    var r2: u32;
    var m_1: u32;
    var n_1: u32;
    var hash: u32;
    var k: u32;

    c1 = 3432918353u;
    c2 = 461845907u;
    r1 = 15u;
    r2 = 13u;
    m_1 = 5u;
    n_1 = 3864292196u;
    let _e14 = (*rng_1).seed;
    hash = _e14;
    let _e18 = (*rng_1).index;
    (*rng_1).index = (_e18 + 1u);
    let _e21 = (*rng_1).index;
    k = _e21;
    let _e23 = c1;
    let _e24 = k;
    k = (_e24 * _e23);
    let _e26 = k;
    let _e27 = r1;
    let _e28 = rot32(_e26, _e27);
    k = _e28;
    let _e29 = c2;
    let _e30 = k;
    k = (_e30 * _e29);
    let _e32 = k;
    let _e33 = hash;
    hash = (_e33 ^ _e32);
    let _e35 = hash;
    let _e36 = r2;
    let _e37 = rot32(_e35, _e36);
    let _e38 = m_1;
    let _e40 = n_1;
    hash = ((_e37 * _e38) + _e40);
    let _e43 = hash;
    hash = (_e43 ^ 4u);
    let _e45 = hash;
    let _e48 = hash;
    hash = (_e48 ^ (_e45 >> 16u));
    let _e51 = hash;
    hash = (_e51 * 2246822507u);
    let _e53 = hash;
    let _e56 = hash;
    hash = (_e56 ^ (_e53 >> 13u));
    let _e59 = hash;
    hash = (_e59 * 3266489909u);
    let _e61 = hash;
    let _e64 = hash;
    hash = (_e64 ^ (_e61 >> 16u));
    let _e66 = hash;
    return _e66;
}

fn random_gen(rng_2: ptr<function, RandomState>) -> f32 {
    var v_1: u32;
    var one: u32;
    var mask: u32;

    let _e1 = murmur3(rng_2);
    v_1 = _e1;
    one = bitcast<u32>(1f);
    mask = ((1u << 23u) - 1u);
    let _e12 = mask;
    let _e13 = v_1;
    let _e15 = one;
    return (bitcast<f32>(((_e12 & _e13) | _e15)) - 1f);
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

fn compute_luminocity(color: vec3<f32>) -> f32 {
    return dot(color, LUMINOCITY_WEIGHTS);
}

fn material_from_metallic_roughness(base_color: vec3<f32>, metalness: f32, roughness: f32) -> Material {
    var mat: Material;

    mat = Material();
    mat.diffuse_albedo = (base_color * vec3((1f - metalness)));
    mat.specular_f0 = mix(vec3(DIELECTRIC_F0), base_color, metalness);
    mat.roughness = roughness;
    let _e15 = mat;
    return _e15;
}

fn material_alpha(mat_1: Material) -> f32 {
    var r: f32;

    r = clamp(mat_1.roughness, MIN_ROUGHNESS, 1f);
    let _e6 = r;
    let _e7 = r;
    return (_e6 * _e7);
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
    let _e5 = a2;
    denom = (((n_dot_h * n_dot_h) * (_e5 - 1f)) + 1f);
    let _e12 = a2;
    let _e14 = denom;
    let _e16 = denom;
    return (_e12 / max(((PI * _e14) * _e16), 0.0000001f));
}

fn visibility_smith(n_dot_v: f32, n_dot_l: f32, alpha_1: f32) -> f32 {
    var a2_1: f32;
    var lambda_v: f32;
    var lambda_l: f32;

    a2_1 = (alpha_1 * alpha_1);
    let _e7 = a2_1;
    let _e10 = a2_1;
    lambda_v = (n_dot_l * sqrt((((n_dot_v * n_dot_v) * (1f - _e7)) + _e10)));
    let _e17 = a2_1;
    let _e20 = a2_1;
    lambda_l = (n_dot_v * sqrt((((n_dot_l * n_dot_l) * (1f - _e17)) + _e20)));
    let _e26 = lambda_v;
    let _e27 = lambda_l;
    return (0.5f / max((_e26 + _e27), 0.0000001f));
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

    let _e2 = compute_luminocity(mat_3.diffuse_albedo);
    diffuse = _e2;
    let _e5 = compute_luminocity(mat_3.specular_f0);
    specular = _e5;
    let _e7 = specular;
    let _e8 = diffuse;
    let _e9 = specular;
    return clamp((_e7 / max((_e8 + _e9), 0.00001f)), 0.1f, 0.9f);
}

fn evaluate_brdf(mat_4: Material, normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> BrdfLobes {
    var n_dot_l_1: f32;
    var n_dot_v_1: f32;
    var half_dir: vec3<f32>;
    var n_dot_h_1: f32;
    var v_dot_h: f32;
    var alpha_2: f32;
    var fresnel: vec3<f32>;
    var specular_1: vec3<f32>;
    var k_diffuse: f32;

    n_dot_l_1 = dot(normal, light_dir);
    n_dot_v_1 = dot(normal, view_dir);
    let _e8 = n_dot_l_1;
    let _e11 = n_dot_v_1;
    if ((_e8 <= 0f) || (_e11 <= 0f)) {
        let _e15 = zero_brdf();
        return _e15;
    }
    half_dir = normalize((view_dir + light_dir));
    let _e19 = half_dir;
    n_dot_h_1 = max(dot(normal, _e19), 0f);
    let _e24 = half_dir;
    v_dot_h = max(dot(view_dir, _e24), 0f);
    let _e29 = material_alpha(mat_4);
    alpha_2 = _e29;
    let _e31 = v_dot_h;
    let _e33 = fresnel_schlick(_e31, mat_4.specular_f0);
    fresnel = _e33;
    let _e35 = n_dot_h_1;
    let _e36 = alpha_2;
    let _e37 = distribution_ggx(_e35, _e36);
    let _e38 = n_dot_v_1;
    let _e39 = n_dot_l_1;
    let _e40 = alpha_2;
    let _e41 = visibility_smith(_e38, _e39, _e40);
    let _e43 = fresnel;
    specular_1 = (vec3((_e37 * _e41)) * _e43);
    let _e48 = v_dot_h;
    let _e50 = fresnel_schlick_scalar(_e48, DIELECTRIC_F0);
    k_diffuse = (1f - _e50);
    let _e53 = k_diffuse;
    let _e54 = n_dot_l_1;
    let _e58 = specular_1;
    let _e59 = n_dot_l_1;
    return BrdfLobes(((_e53 * _e54) / PI), (_e58 * vec3(_e59)));
}

fn make_tangent_frame(normal_1: vec3<f32>) -> mat3x3<f32> {
    var s: f32;
    var a_2: f32;
    var b_2: f32;

    s = select(-(1f), 1f, (normal_1.z >= 0f));
    let _e11 = s;
    a_2 = (-(1f) / (_e11 + normal_1.z));
    let _e19 = a_2;
    b_2 = ((normal_1.x * normal_1.y) * _e19);
    let _e23 = s;
    let _e28 = a_2;
    let _e31 = s;
    let _e32 = b_2;
    let _e34 = s;
    let _e39 = b_2;
    let _e40 = s;
    let _e44 = a_2;
    return mat3x3<f32>(vec3<f32>((1f + (((_e23 * normal_1.x) * normal_1.x) * _e28)), (_e31 * _e32), (-(_e34) * normal_1.x)), vec3<f32>(_e39, (_e40 + ((normal_1.y * normal_1.y) * _e44)), -(normal_1.y)), normal_1);
}

fn sample_circle_uniform(random: f32) -> vec2<f32> {
    var angle: f32;

    angle = ((2f * PI) * random);
    let _e6 = angle;
    let _e8 = angle;
    return vec2<f32>(cos(_e6), sin(_e8));
}

fn compute_bsdf_pdf(mat_5: Material, normal_2: vec3<f32>, view_dir_1: vec3<f32>, light_dir_1: vec3<f32>) -> f32 {
    var n_dot_l_2: f32;
    var half_dir_1: vec3<f32>;
    var n_dot_h_2: f32;
    var v_dot_h_1: f32;
    var specular_pdf: f32;
    var diffuse_pdf: f32;

    n_dot_l_2 = dot(normal_2, light_dir_1);
    let _e6 = n_dot_l_2;
    if ((_e6 <= 0f) || (dot(normal_2, view_dir_1) <= 0f)) {
        return 0f;
    }
    half_dir_1 = normalize((view_dir_1 + light_dir_1));
    let _e17 = half_dir_1;
    n_dot_h_2 = max(dot(normal_2, _e17), 0f);
    let _e22 = half_dir_1;
    v_dot_h_1 = max(dot(view_dir_1, _e22), 0.00001f);
    let _e27 = n_dot_h_2;
    let _e28 = material_alpha(mat_5);
    let _e29 = distribution_ggx(_e27, _e28);
    let _e30 = n_dot_h_2;
    let _e33 = v_dot_h_1;
    specular_pdf = ((_e29 * _e30) / (4f * _e33));
    let _e37 = n_dot_l_2;
    diffuse_pdf = (_e37 / PI);
    let _e41 = diffuse_pdf;
    let _e42 = specular_pdf;
    let _e43 = specular_sampling_ratio(mat_5);
    return mix(_e41, _e42, _e43);
}

fn evaluate_bsdf(mat_6: Material, normal_3: vec3<f32>, view_dir_2: vec3<f32>, light_dir_2: vec3<f32>) -> vec3<f32> {
    var brdf: BrdfLobes;

    let _e4 = evaluate_brdf(mat_6, normal_3, view_dir_2, light_dir_2);
    brdf = _e4;
    let _e8 = brdf.diffuse;
    let _e12 = brdf.specular;
    return ((mat_6.diffuse_albedo * vec3(_e8)) + _e12);
}

fn sample_hemisphere_cosine(rng_3: ptr<function, RandomState>) -> vec3<f32> {
    var r_1: f32;
    var tangential: vec2<f32>;

    let _e1 = random_gen(rng_3);
    r_1 = _e1;
    let _e3 = r_1;
    let _e5 = random_gen(rng_3);
    let _e6 = sample_circle_uniform(_e5);
    tangential = (vec2(sqrt(_e3)) * _e6);
    let _e10 = tangential;
    let _e13 = r_1;
    return vec3<f32>(_e10, sqrt(max(0f, (1f - _e13))));
}

fn sample_ggx_half_dir(alpha_3: f32, rng_4: ptr<function, RandomState>) -> vec3<f32> {
    var a2_2: f32;
    var r_2: f32;
    var cos_theta_2: f32;
    var sin_theta: f32;

    a2_2 = (alpha_3 * alpha_3);
    let _e4 = random_gen(rng_4);
    r_2 = _e4;
    let _e7 = r_2;
    let _e10 = a2_2;
    let _e13 = r_2;
    cos_theta_2 = sqrt(((1f - _e7) / (1f + ((_e10 - 1f) * _e13))));
    let _e21 = cos_theta_2;
    let _e22 = cos_theta_2;
    sin_theta = sqrt(max(0f, (1f - (_e21 * _e22))));
    let _e28 = sin_theta;
    let _e29 = random_gen(rng_4);
    let _e30 = sample_circle_uniform(_e29);
    let _e33 = cos_theta_2;
    return vec3<f32>((vec2(_e28) * _e30), _e33);
}

fn sample_bsdf(mat_7: Material, normal_4: vec3<f32>, view_dir_3: vec3<f32>, rng_5: ptr<function, RandomState>) -> BsdfSample {
    var frame: mat3x3<f32>;
    var dir: vec3<f32>;
    var half_dir_2: vec3<f32>;

    let _e4 = make_tangent_frame(normal_4);
    frame = _e4;
    dir = vec3<f32>();
    let _e8 = random_gen(rng_5);
    let _e9 = specular_sampling_ratio(mat_7);
    if (_e8 < _e9) {
        let _e11 = frame;
        let _e12 = material_alpha(mat_7);
        let _e13 = sample_ggx_half_dir(_e12, rng_5);
        half_dir_2 = (_e11 * _e13);
        let _e17 = half_dir_2;
        let _e20 = half_dir_2;
        dir = ((vec3((2f * dot(view_dir_3, _e17))) * _e20) - view_dir_3);
    } else {
        let _e24 = frame;
        let _e25 = sample_hemisphere_cosine(rng_5);
        dir = (_e24 * _e25);
    }
    let _e27 = dir;
    dir = normalize(_e27);
    let _e29 = dir;
    let _e30 = dir;
    let _e31 = compute_bsdf_pdf(mat_7, normal_4, view_dir_3, _e30);
    return BsdfSample(_e29, _e31);
}

fn compute_latitude_area_bounds(texel_y: i32, dim: u32) -> vec2<f32> {
    return cos(((vec2<f32>(vec2<i32>(texel_y, (texel_y + 1i))) / vec2(f32(dim))) * vec2(PI)));
}

fn compute_texel_solid_angle(itc: vec2<i32>, dim_1: vec2<u32>) -> f32 {
    var meridian_solid_angle: f32;
    var bounds: vec2<f32>;
    var meridian_part: f32;

    meridian_solid_angle = ((4f * PI) / f32(dim_1.x));
    let _e12 = compute_latitude_area_bounds(itc.y, dim_1.y);
    bounds = _e12;
    let _e16 = bounds.x;
    let _e18 = bounds.y;
    meridian_part = (0.5f * (_e16 - _e18));
    let _e22 = meridian_solid_angle;
    let _e23 = meridian_part;
    return (_e22 * _e23);
}

fn generate_environment_sample(rng_6: ptr<function, RandomState>, dim_2: vec2<u32>) -> EnvImportantSample {
    var es: EnvImportantSample;
    var mip: i32;
    var itc_1: vec2<i32>;
    var weights: vec4<f32>;
    var sum_1: f32;
    var r_3: f32;
    var weight: f32;

    es = EnvImportantSample();
    es.pdf = 1f;
    let _e7 = textureNumLevels(env_weights);
    mip = i32(_e7);
    itc_1 = vec2(0i);
    loop {
        let _e13 = mip;
        if (_e13 != 0i) {
            let _e17 = mip;
            mip = (_e17 - 1i);
            let _e19 = itc_1;
            let _e20 = mip;
            let _e21 = textureLoad(env_weights, _e19, _e20);
            weights = _e21;
            let _e25 = weights;
            sum_1 = dot(vec4(1f), _e25);
            let _e28 = random_gen(rng_6);
            let _e29 = sum_1;
            r_3 = (_e28 * _e29);
            weight = f32();
            let _e35 = itc_1;
            itc_1 = (_e35 * vec2(2i));
            let _e38 = r_3;
            let _e40 = weights.x;
            let _e42 = weights.y;
            if (_e38 >= (_e40 + _e42)) {
                let _e47 = itc_1.y;
                itc_1.y = (_e47 + 1i);
                let _e49 = r_3;
                let _e51 = weights.x;
                let _e53 = weights.y;
                let _e56 = weights.z;
                if (_e49 >= ((_e51 + _e53) + _e56)) {
                    let _e60 = weights.w;
                    weight = _e60;
                    let _e63 = itc_1.x;
                    itc_1.x = (_e63 + 1i);
                } else {
                    let _e66 = weights.z;
                    weight = _e66;
                }
            } else {
                let _e67 = r_3;
                let _e69 = weights.x;
                if (_e67 >= _e69) {
                    let _e72 = weights.y;
                    weight = _e72;
                    let _e75 = itc_1.x;
                    itc_1.x = (_e75 + 1i);
                } else {
                    let _e78 = weights.x;
                    weight = _e78;
                }
            }
            let _e80 = weight;
            let _e81 = sum_1;
            let _e83 = es.pdf;
            es.pdf = (_e83 * (_e80 / _e81));
        } else {
            break;
        }
    }
    let _e86 = itc_1;
    let _e87 = compute_texel_solid_angle(_e86, dim_2);
    let _e88 = es.pdf;
    es.pdf = (_e88 / _e87);
    let _e91 = itc_1;
    es.pixel = _e91;
    let _e92 = es;
    return _e92;
}

fn compute_environment_sample_pdf(pixel_1: vec2<i32>, dim_3: vec2<u32>) -> f32 {
    var itc_2: vec2<i32>;
    var pdf: f32;
    var mip_count: i32;
    var mip_1: i32;
    var rem: vec2<i32>;
    var weights_1: vec4<f32>;
    var sum_2: f32;
    var w2: vec2<f32>;
    var weight_1: f32;

    itc_2 = pixel_1;
    let _e5 = itc_2;
    let _e6 = compute_texel_solid_angle(_e5, dim_3);
    pdf = (1f / _e6);
    let _e9 = textureNumLevels(env_weights);
    mip_count = i32(_e9);
    let _e12 = mip_count;
    mip_1 = 0i;
    loop {
        let _e15 = mip_1;
        if (_e15 < _e12) {
        } else {
            break;
        }
        let _e17 = itc_2;
        rem = (_e17 & vec2(1i));
        let _e22 = itc_2;
        itc_2 = (_e22 >> vec2(1u));
        let _e26 = itc_2;
        let _e27 = mip_1;
        let _e28 = textureLoad(env_weights, _e26, _e27);
        weights_1 = _e28;
        let _e32 = weights_1;
        sum_2 = dot(vec4(1f), _e32);
        let _e35 = weights_1;
        let _e37 = weights_1;
        let _e40 = rem.y;
        w2 = select(_e35.xy, _e37.zw, (_e40 != 0i));
        let _e46 = w2.x;
        let _e48 = w2.y;
        let _e50 = rem.x;
        weight_1 = select(_e46, _e48, (_e50 != 0i));
        let _e55 = weight_1;
        let _e56 = sum_2;
        let _e58 = pdf;
        pdf = (_e58 * (_e55 / _e56));
        continuing {
            let _e60 = mip_1;
            mip_1 = (_e60 + 1i);
        }
    }
    let _e63 = pdf;
    return _e63;
}

fn affine_linear(transform: mat4x3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(transform[0].xyz, transform[1].xyz, transform[2].xyz);
}

fn hit_winding(entry: HitEntry) -> f32 {
    return select(1f, -(1f), ((entry.flags & 1u) != 0u));
}

fn fetch_triangle_indices(entry_1: HitEntry, primitive_index: u32) -> vec3<u32> {
    var indices: vec3<u32>;

    indices = (vec3((primitive_index * 3u)) + vec3<u32>(0u, 1u, 2u));
    if (entry_1.index_buf != ~(0u)) {
        let _e25 = indices.x;
        let _e27 = index_buffers[entry_1.index_buf].data[_e25];
        let _e32 = indices.y;
        let _e34 = index_buffers[entry_1.index_buf].data[_e32];
        let _e39 = indices.z;
        let _e41 = index_buffers[entry_1.index_buf].data[_e39];
        indices = vec3<u32>(_e27, _e34, _e41);
    }
    let _e43 = indices;
    return _e43;
}

fn make_barycentrics(uv: vec2<f32>) -> vec3<f32> {
    var w: f32;

    w = ((1f - uv.x) - uv.y);
    let _e13 = w;
    return vec3<f32>(_e13, uv.x, uv.y);
}

fn sample_hit_material(entry_2: HitEntry, tex_coords: vec2<f32>, lod: f32, ignore_textures: u32) -> Material {
    var base_color_1: vec3<f32>;
    var metalness_1: f32;
    var roughness_1: f32;
    var mr: vec4<f32>;

    base_color_1 = unpack4x8unorm(entry_2.base_color_factor).xyz;
    if ((ignore_textures & DebugTextureFlags_ALBEDO) == 0u) {
        let _e20 = textureSampleLevel(textures[entry_2.base_color_texture], sampler_linear, tex_coords, lod);
        let _e22 = base_color_1;
        base_color_1 = (_e22 * _e20.xyz);
    }
    metalness_1 = entry_2.metalness;
    roughness_1 = entry_2.roughness;
    if ((ignore_textures & DebugTextureFlags_METALLIC_ROUGHNESS) == 0u) {
        let _e34 = textureSampleLevel(textures[entry_2.metallic_roughness_texture], sampler_linear, tex_coords, lod);
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

fn sample_hit_emissive(entry_3: HitEntry, tex_coords_1: vec2<f32>, lod_1: f32, ignore_textures_1: u32) -> vec3<f32> {
    var emissive: vec3<f32>;

    emissive = entry_3.emissive_factor.xyz;
    if ((ignore_textures_1 & DebugTextureFlags_EMISSIVE) == 0u) {
        let _e19 = textureSampleLevel(textures[entry_3.emissive_texture], sampler_linear, tex_coords_1, lod_1);
        let _e21 = emissive;
        emissive = (_e21 * _e19.xyz);
    }
    let _e23 = emissive;
    return _e23;
}

fn sample_hit_normal_map(entry_4: HitEntry, tex_coords_2: vec2<f32>, lod_2: f32, ignore_textures_2: u32) -> vec3<f32> {
    var raw_unorm: vec2<f32>;
    var n_xy: vec2<f32>;

    if ((ignore_textures_2 & DebugTextureFlags_NORMAL) != 0u) {
        return vec3<f32>(0f, 0f, 1f);
    }
    let _e20 = textureSampleLevel(textures[entry_4.normal_texture], sampler_linear, tex_coords_2, lod_2);
    raw_unorm = _e20.xy;
    let _e25 = raw_unorm;
    n_xy = (vec2(entry_4.normal_scale) * ((vec2(2f) * _e25) - vec2(1f)));
    let _e34 = n_xy;
    let _e37 = n_xy;
    let _e38 = n_xy;
    return vec3<f32>(_e34, sqrt(max(0f, (1f - dot(_e37, _e38)))));
}

fn hit_normal(entry_5: HitEntry, object_to_world: mat4x3<f32>, normal_5: vec3<f32>) -> vec3<f32> {
    var linear: mat3x3<f32>;

    let _e9 = affine_linear(object_to_world);
    let _e11 = affine_linear(entry_5.geometry_to_object);
    linear = (_e9 * _e11);
    let _e14 = linear;
    return normalize((_e14 * normal_5));
}

fn hit_tangent_space(entry_6: HitEntry, object_to_world_1: mat4x3<f32>, normal_6: vec3<f32>, tangent: vec3<f32>, bitangent_sign_1: f32) -> mat3x3<f32> {
    var linear_1: mat3x3<f32>;
    var n_2: vec3<f32>;

    let _e11 = affine_linear(object_to_world_1);
    let _e13 = affine_linear(entry_6.geometry_to_object);
    linear_1 = (_e11 * _e13);
    let _e16 = hit_normal(entry_6, object_to_world_1, normal_6);
    n_2 = _e16;
    let _e18 = n_2;
    let _e19 = linear_1;
    let _e21 = linear_1;
    let _e24 = tangent_basis(_e18, (_e19 * tangent), bitangent_sign_1, sign(determinant(_e21)));
    return _e24;
}

fn map_equirect_dir_to_uv(dir_1: vec3<f32>) -> vec2<f32> {
    var yaw: f32;
    var pitch: f32;

    yaw = asin(dir_1.y);
    pitch = atan2(dir_1.x, dir_1.z);
    let _e16 = pitch;
    let _e21 = yaw;
    return (vec2<f32>((_e16 + PI), ((-(2f) * _e21) + PI)) / vec2((2f * PI)));
}

fn map_equirect_uv_to_dir(uv_1: vec2<f32>) -> vec3<f32> {
    var yaw_1: f32;
    var pitch_1: f32;

    yaw_1 = (PI * (0.5f - uv_1.y));
    pitch_1 = ((2f * PI) * (uv_1.x - 0.5f));
    let _e23 = yaw_1;
    let _e25 = pitch_1;
    let _e28 = yaw_1;
    let _e30 = yaw_1;
    let _e32 = pitch_1;
    return vec3<f32>((cos(_e23) * sin(_e25)), sin(_e28), (cos(_e30) * cos(_e32)));
}

fn sample_light_from_environment(rng_7: ptr<function, RandomState>) -> LightSample {
    var dim_4: vec2<u32>;
    var es_1: EnvImportantSample;
    var ls: LightSample;
    var u: f32;
    var bounds_1: vec2<f32>;
    var v_2: f32;

    let _e10 = textureDimensions(env_map, 0i);
    dim_4 = _e10;
    let _e12 = dim_4;
    let _e13 = generate_environment_sample(rng_7, _e12);
    es_1 = _e13;
    ls = LightSample();
    let _e19 = es_1.pdf;
    ls.pdf = _e19;
    let _e22 = es_1.pixel;
    let _e24 = textureLoad(env_map, _e22, 0i);
    ls.radiance = _e24.xyz;
    let _e28 = es_1.pixel.x;
    let _e30 = random_gen(rng_7);
    let _e33 = dim_4.x;
    u = ((f32(_e28) + _e30) / f32(_e33));
    let _e39 = es_1.pixel.y;
    let _e41 = dim_4.y;
    let _e42 = compute_latitude_area_bounds(_e39, _e41);
    bounds_1 = _e42;
    let _e45 = bounds_1.x;
    let _e47 = bounds_1.y;
    let _e48 = random_gen(rng_7);
    v_2 = (acos(mix(_e45, _e47, _e48)) / PI);
    let _e55 = u;
    let _e56 = v_2;
    ls.uv = vec2<f32>(_e55, _e56);
    let _e58 = ls;
    return _e58;
}

fn compute_light_pdf(uv_2: vec2<f32>, importance: bool) -> f32 {
    var dim_5: vec2<u32>;
    var pixel_2: vec2<i32>;

    if !(importance) {
        return (1f / (4f * PI));
    }
    let _e17 = textureDimensions(env_map, 0i);
    dim_5 = _e17;
    let _e19 = dim_5;
    let _e25 = dim_5;
    pixel_2 = clamp(vec2<i32>((uv_2 * vec2<f32>(_e19))), vec2(0i), (vec2<i32>(_e25) - vec2(1i)));
    let _e32 = pixel_2;
    let _e33 = dim_5;
    let _e34 = compute_environment_sample_pdf(_e32, _e33);
    return _e34;
}

fn evaluate_environment(dir_2: vec3<f32>) -> vec3<f32> {
    var uv_3: vec2<f32>;

    let _e9 = map_equirect_dir_to_uv(dir_2);
    uv_3 = _e9;
    let _e11 = uv_3;
    let _e13 = textureSampleLevel(env_map, sampler_nearest, _e11, 0f);
    return _e13.xyz;
}

fn evaluate_environment_background(dir_3: vec3<f32>) -> vec3<f32> {
    var uv_4: vec2<f32>;

    let _e9 = map_equirect_dir_to_uv(dir_3);
    uv_4 = _e9;
    let _e11 = uv_4;
    let _e13 = textureSampleLevel(env_map, sampler_linear, _e11, 0f);
    return _e13.xyz;
}

fn sample_light_from_sphere(rng_8: ptr<function, RandomState>) -> LightSample {
    var a_3: f32;
    var h: f32;
    var tangential_1: vec2<f32>;
    var dir_4: vec3<f32>;
    var ls_1: LightSample;

    let _e9 = random_gen(rng_8);
    a_3 = _e9;
    let _e13 = random_gen(rng_8);
    h = (1f - (2f * _e13));
    let _e19 = h;
    let _e20 = h;
    let _e25 = a_3;
    let _e26 = sample_circle_uniform(_e25);
    tangential_1 = (vec2(sqrt(max(0f, (1f - (_e19 * _e20))))) * _e26);
    let _e31 = tangential_1.x;
    let _e32 = h;
    let _e34 = tangential_1.y;
    dir_4 = vec3<f32>(_e31, _e32, _e34);
    ls_1 = LightSample();
    let _e40 = dir_4;
    let _e41 = map_equirect_dir_to_uv(_e40);
    ls_1.uv = _e41;
    ls_1.pdf = (1f / (4f * PI));
    let _e50 = ls_1.uv;
    let _e52 = textureSampleLevel(env_map, sampler_nearest, _e50, 0f);
    ls_1.radiance = _e52.xyz;
    let _e54 = ls_1;
    return _e54;
}

fn sample_light(importance_1: bool, rng_9: ptr<function, RandomState>) -> LightSample {
    if importance_1 {
        let _e10 = sample_light_from_environment(rng_9);
        return _e10;
    } else {
        let _e11 = sample_light_from_sphere(rng_9);
        return _e11;
    }
}

fn trace_ray(position: vec3<f32>, direction: vec3<f32>, t_min: f32) -> RayIntersection {
    var rq: ray_query;

    let _e22 = camera.depth;
    rayQueryInitialize((&rq), acc_struct, RayDesc(128u, 255u, t_min, _e22, position, direction));
    let _e24 = rayQueryProceed((&rq));
    let _e25 = rayQueryGetCommittedIntersection((&rq));
    return _e25;
}

fn is_occluded(position_1: vec3<f32>, direction_1: vec3<f32>) -> bool {
    var rq_1: ray_query;
    var flags: u32;

    flags = (4u | 128u);
    let _e22 = flags;
    let _e25 = parameters.t_start;
    let _e27 = camera.depth;
    rayQueryInitialize((&rq_1), acc_struct, RayDesc(_e22, 255u, _e25, _e27, position_1, direction_1));
    let _e29 = rayQueryProceed((&rq_1));
    let _e30 = rayQueryGetCommittedIntersection((&rq_1));
    return (_e30.kind != 0u);
}

fn resolve_hit(intersection: RayIntersection) -> PathVertex {
    var entry_7: HitEntry;
    var indices_1: vec3<u32>;
    var vertices: array<Vertex, 3>;
    var positions_object: mat3x3<f32>;
    var positions: mat3x3<f32>;
    var barycentrics: vec3<f32>;
    var tex_coords_3: vec2<f32>;
    var normal_geo: vec3<f32>;
    var tangent_geo: vec3<f32>;
    var tangent_space_world: mat3x3<f32>;
    var lod_3: f32;
    var vertex: PathVertex;
    var normal_local: vec3<f32>;

    let _e20 = hit_entries[(intersection.instance_custom_data + intersection.geometry_index)];
    entry_7 = _e20;
    let _e22 = entry_7;
    let _e24 = fetch_triangle_indices(_e22, intersection.primitive_index);
    indices_1 = _e24;
    let _e27 = entry_7.vertex_buf;
    let _e31 = indices_1.x;
    let _e33 = vertex_buffers[_e27].data[_e31];
    let _e35 = entry_7.vertex_buf;
    let _e39 = indices_1.y;
    let _e41 = vertex_buffers[_e35].data[_e39];
    let _e43 = entry_7.vertex_buf;
    let _e47 = indices_1.z;
    let _e49 = vertex_buffers[_e43].data[_e47];
    vertices = array<Vertex, 3>(_e33, _e41, _e49);
    let _e53 = entry_7.geometry_to_object;
    let _e56 = vertices[0].position;
    let _e61 = vertices[1].position;
    let _e66 = vertices[2].position;
    positions_object = (_e53 * mat3x4<f32>(vec4<f32>(_e56, 1f), vec4<f32>(_e61, 1f), vec4<f32>(_e66, 1f)));
    let _e74 = positions_object[0];
    let _e78 = positions_object[1];
    let _e82 = positions_object[2];
    positions = (intersection.object_to_world * mat3x4<f32>(vec4<f32>(_e74, 1f), vec4<f32>(_e78, 1f), vec4<f32>(_e82, 1f)));
    let _e89 = make_barycentrics(intersection.barycentrics);
    barycentrics = _e89;
    let _e93 = vertices[0].tex_coords;
    let _e96 = vertices[1].tex_coords;
    let _e99 = vertices[2].tex_coords;
    let _e101 = barycentrics;
    tex_coords_3 = (mat3x2<f32>(_e93, _e96, _e99) * _e101);
    let _e106 = vertices[0].normal;
    let _e107 = decode_normal(_e106);
    let _e110 = vertices[1].normal;
    let _e111 = decode_normal(_e110);
    let _e114 = vertices[2].normal;
    let _e115 = decode_normal(_e114);
    let _e117 = barycentrics;
    normal_geo = normalize((mat3x3<f32>(_e107, _e111, _e115) * _e117));
    let _e123 = vertices[0].tangent;
    let _e124 = decode_normal(_e123);
    let _e127 = vertices[1].tangent;
    let _e128 = decode_normal(_e127);
    let _e131 = vertices[2].tangent;
    let _e132 = decode_normal(_e131);
    let _e134 = barycentrics;
    tangent_geo = normalize((mat3x3<f32>(_e124, _e128, _e132) * _e134));
    let _e138 = entry_7;
    let _e140 = normal_geo;
    let _e141 = tangent_geo;
    let _e144 = vertices[0].bitangent_sign;
    let _e145 = hit_tangent_space(_e138, intersection.object_to_world, _e140, _e141, _e144);
    tangent_space_world = _e145;
    lod_3 = 0f;
    vertex = PathVertex();
    let _e152 = positions;
    let _e153 = barycentrics;
    vertex.position = (_e152 * _e153);
    let _e156 = entry_7;
    let _e157 = hit_winding(_e156);
    let _e159 = positions[1];
    let _e162 = positions[0];
    let _e166 = positions[2];
    let _e169 = positions[0];
    vertex.flat_normal = (vec3(_e157) * normalize(cross((_e159.xyz - _e162.xyz), (_e166.xyz - _e169.xyz))));
    let _e176 = entry_7;
    let _e177 = tex_coords_3;
    let _e178 = lod_3;
    let _e180 = sample_hit_normal_map(_e176, _e177, _e178, 0u);
    normal_local = _e180;
    let _e183 = tangent_space_world;
    let _e184 = normal_local;
    vertex.normal = normalize((_e183 * _e184));
    let _e188 = entry_7;
    let _e189 = tex_coords_3;
    let _e190 = lod_3;
    let _e192 = sample_hit_material(_e188, _e189, _e190, 0u);
    vertex.material = _e192;
    let _e194 = entry_7;
    let _e195 = tex_coords_3;
    let _e196 = lod_3;
    let _e198 = sample_hit_emissive(_e194, _e195, _e196, 0u);
    vertex.emissive = _e198;
    let _e199 = vertex;
    return _e199;
}

fn mis_weight(count_1: f32, pdf_1: f32, other_count: f32, other_pdf: f32) -> f32 {
    var total_1: f32;

    total_1 = ((count_1 * pdf_1) + (other_count * other_pdf));
    let _e25 = total_1;
    let _e27 = total_1;
    return select(0f, ((count_1 * pdf_1) / _e25), (_e27 > 0f));
}

fn zero_path_radiance() -> PathRadiance {
    return PathRadiance(vec3(0f), vec3(0f), vec3(0f), vec3(0f));
}

fn trace_path(start_dir: vec3<f32>, rng_10: ptr<function, RandomState>) -> PathRadiance {
    var importance_2: bool;
    var num_light: f32;
    var radiance: PathRadiance;
    var primary_albedo: vec3<f32>;
    var diffuse_throughput: vec3<f32>;
    var specular_throughput: vec3<f32>;
    var position_2: vec3<f32>;
    var direction_2: vec3<f32>;
    var bsdf_pdf: f32;
    var t_min_1: f32;
    var bounce: u32;
    var intersection_1: RayIntersection;
    var light_pdf: f32;
    var weight_2: f32;
    var incoming: vec3<f32>;
    var vertex_1: PathVertex;
    var view_dir_4: vec3<f32>;
    var will_extend: bool;
    var bsdf_count: f32;
    var i_1: u32;
    var ls_2: LightSample;
    var light_dir_3: vec3<f32>;
    var lobes_1: BrdfLobes;
    var other_pdf_1: f32;
    var weight_3: f32;
    var incoming_1: vec3<f32>;
    var bsdf: vec3<f32>;
    var bs: BsdfSample;
    var lobes_2: BrdfLobes;
    var bsdf_1: vec3<f32>;
    var throughput: vec3<f32>;
    var probability: f32;
    var throughput_1: vec3<f32>;
    var is_finite: bool;
    var scale: vec3<f32>;

    let _e18 = parameters.environment_importance_sampling;
    importance_2 = (_e18 != 0u);
    let _e23 = parameters.num_environment_samples;
    num_light = f32(_e23);
    let _e26 = zero_path_radiance();
    radiance = _e26;
    primary_albedo = vec3(1f);
    diffuse_throughput = vec3(0f);
    specular_throughput = vec3(0f);
    let _e38 = camera.position;
    position_2 = _e38;
    direction_2 = start_dir;
    bsdf_pdf = -(1f);
    t_min_1 = 0f;
    let _e48 = parameters.max_bounces;
    bounce = 0u;
    loop {
        let _e50 = bounce;
        if (_e50 <= _e48) {
        } else {
            break;
        }
        let _e52 = position_2;
        let _e53 = direction_2;
        let _e54 = t_min_1;
        let _e55 = trace_ray(_e52, _e53, _e54);
        intersection_1 = _e55;
        let _e58 = intersection_1.kind;
        if (_e58 == 0u) {
            let _e61 = bsdf_pdf;
            if (_e61 < 0f) {
                let _e65 = direction_2;
                let _e66 = evaluate_environment_background(_e65);
                let _e67 = radiance.diffuse;
                radiance.diffuse = (_e67 + _e66);
            } else {
                let _e69 = direction_2;
                let _e70 = map_equirect_dir_to_uv(_e69);
                let _e71 = importance_2;
                let _e72 = compute_light_pdf(_e70, _e71);
                light_pdf = _e72;
                let _e75 = bsdf_pdf;
                let _e76 = num_light;
                let _e77 = light_pdf;
                let _e78 = mis_weight(1f, _e75, _e76, _e77);
                weight_2 = _e78;
                let _e80 = direction_2;
                let _e81 = evaluate_environment(_e80);
                let _e82 = weight_2;
                incoming = (_e81 * vec3(_e82));
                let _e87 = diffuse_throughput;
                let _e88 = incoming;
                let _e90 = radiance.diffuse;
                radiance.diffuse = (_e90 + (_e87 * _e88));
                let _e93 = specular_throughput;
                let _e94 = incoming;
                let _e96 = radiance.specular;
                radiance.specular = (_e96 + (_e93 * _e94));
            }
            break;
        }
        let _e98 = intersection_1;
        let _e99 = resolve_hit(_e98);
        vertex_1 = _e99;
        let _e101 = direction_2;
        view_dir_4 = -(_e101);
        let _e104 = bounce;
        if (_e104 == 0u) {
            let _e109 = vertex_1.material.diffuse_albedo;
            primary_albedo = _e109;
            let _e112 = vertex_1.emissive;
            let _e113 = radiance.emissive;
            radiance.emissive = (_e113 + _e112);
        } else {
            let _e116 = diffuse_throughput;
            let _e118 = vertex_1.emissive;
            let _e120 = radiance.diffuse;
            radiance.diffuse = (_e120 + (_e116 * _e118));
            let _e123 = specular_throughput;
            let _e125 = vertex_1.emissive;
            let _e127 = radiance.specular;
            radiance.specular = (_e127 + (_e123 * _e125));
        }
        let _e130 = vertex_1.position;
        position_2 = _e130;
        let _e132 = parameters.t_start;
        t_min_1 = _e132;
        let _e133 = bounce;
        let _e135 = parameters.max_bounces;
        let _e138 = parameters.num_brdf_samples;
        will_extend = ((_e133 < _e135) && (_e138 != 0u));
        let _e145 = will_extend;
        bsdf_count = select(0f, 1f, _e145);
        let _e150 = parameters.num_environment_samples;
        i_1 = 0u;
        loop {
            let _e152 = i_1;
            if (_e152 < _e150) {
            } else {
                break;
            }
            let _e154 = importance_2;
            let _e155 = sample_light(_e154, rng_10);
            ls_2 = _e155;
            let _e158 = ls_2.pdf;
            if (_e158 <= 0f) {
                continue;
            }
            let _e162 = ls_2.uv;
            let _e163 = map_equirect_uv_to_dir(_e162);
            light_dir_3 = _e163;
            let _e166 = vertex_1.material;
            let _e168 = vertex_1.normal;
            let _e169 = view_dir_4;
            let _e170 = light_dir_3;
            let _e171 = evaluate_brdf(_e166, _e168, _e169, _e170);
            lobes_1 = _e171;
            let _e173 = light_dir_3;
            let _e175 = vertex_1.flat_normal;
            let _e179 = lobes_1;
            let _e180 = is_brdf_black(_e179);
            let _e182 = position_2;
            let _e183 = light_dir_3;
            let _e184 = is_occluded(_e182, _e183);
            if (((dot(_e173, _e175) <= 0f) || _e180) || _e184) {
                continue;
            }
            let _e187 = vertex_1.material;
            let _e189 = vertex_1.normal;
            let _e190 = view_dir_4;
            let _e191 = light_dir_3;
            let _e192 = compute_bsdf_pdf(_e187, _e189, _e190, _e191);
            other_pdf_1 = _e192;
            let _e194 = num_light;
            let _e196 = ls_2.pdf;
            let _e197 = bsdf_count;
            let _e198 = other_pdf_1;
            let _e199 = mis_weight(_e194, _e196, _e197, _e198);
            let _e200 = num_light;
            let _e202 = ls_2.pdf;
            weight_3 = (_e199 / (_e200 * _e202));
            let _e207 = ls_2.radiance;
            let _e208 = weight_3;
            incoming_1 = (_e207 * vec3(_e208));
            let _e212 = bounce;
            if (_e212 == 0u) {
                let _e217 = lobes_1.diffuse;
                let _e218 = incoming_1;
                let _e221 = radiance.diffuse;
                radiance.diffuse = (_e221 + (vec3(_e217) * _e218));
                let _e225 = lobes_1.specular;
                let _e226 = incoming_1;
                let _e228 = radiance.specular;
                radiance.specular = (_e228 + (_e225 * _e226));
            } else {
                let _e232 = vertex_1.material.diffuse_albedo;
                let _e234 = lobes_1.diffuse;
                let _e238 = lobes_1.specular;
                bsdf = ((_e232 * vec3(_e234)) + _e238);
                let _e242 = diffuse_throughput;
                let _e243 = bsdf;
                let _e245 = incoming_1;
                let _e247 = radiance.diffuse;
                radiance.diffuse = (_e247 + ((_e242 * _e243) * _e245));
                let _e250 = specular_throughput;
                let _e251 = bsdf;
                let _e253 = incoming_1;
                let _e255 = radiance.specular;
                radiance.specular = (_e255 + ((_e250 * _e251) * _e253));
            }
            continuing {
                let _e257 = i_1;
                i_1 = (_e257 + 1u);
            }
        }
        let _e260 = will_extend;
        if !(_e260) {
            break;
        }
        let _e263 = vertex_1.material;
        let _e265 = vertex_1.normal;
        let _e266 = view_dir_4;
        let _e267 = sample_bsdf(_e263, _e265, _e266, rng_10);
        bs = _e267;
        let _e270 = bs.pdf;
        let _e274 = bs.dir;
        let _e276 = vertex_1.flat_normal;
        if ((_e270 <= 0f) || (dot(_e274, _e276) <= 0f)) {
            break;
        }
        let _e282 = vertex_1.material;
        let _e284 = vertex_1.normal;
        let _e285 = view_dir_4;
        let _e287 = bs.dir;
        let _e288 = evaluate_brdf(_e282, _e284, _e285, _e287);
        lobes_2 = _e288;
        let _e290 = bounce;
        if (_e290 == 0u) {
            let _e294 = lobes_2.diffuse;
            let _e296 = bs.pdf;
            diffuse_throughput = vec3((_e294 / _e296));
            let _e300 = lobes_2.specular;
            let _e302 = bs.pdf;
            specular_throughput = (_e300 / vec3(_e302));
        } else {
            let _e307 = vertex_1.material.diffuse_albedo;
            let _e309 = lobes_2.diffuse;
            let _e313 = lobes_2.specular;
            bsdf_1 = ((_e307 * vec3(_e309)) + _e313);
            let _e316 = bsdf_1;
            let _e318 = bs.pdf;
            let _e321 = diffuse_throughput;
            diffuse_throughput = (_e321 * (_e316 / vec3(_e318)));
            let _e323 = bsdf_1;
            let _e325 = bs.pdf;
            let _e328 = specular_throughput;
            specular_throughput = (_e328 * (_e323 / vec3(_e325)));
        }
        let _e331 = bs.pdf;
        bsdf_pdf = _e331;
        let _e333 = bs.dir;
        direction_2 = _e333;
        let _e334 = bounce;
        if (_e334 >= ROULETTE_START) {
            let _e337 = primary_albedo;
            let _e338 = diffuse_throughput;
            let _e340 = specular_throughput;
            throughput = ((_e337 * _e338) + _e340);
            let _e343 = throughput;
            let _e344 = compute_luminocity(_e343);
            probability = clamp(_e344, 0.05f, 1f);
            let _e349 = random_gen(rng_10);
            let _e350 = probability;
            if (_e349 >= _e350) {
                break;
            }
            let _e352 = probability;
            let _e353 = diffuse_throughput;
            diffuse_throughput = (_e353 / vec3(_e352));
            let _e356 = probability;
            let _e357 = specular_throughput;
            specular_throughput = (_e357 / vec3(_e356));
        }
        let _e360 = primary_albedo;
        let _e361 = diffuse_throughput;
        let _e363 = specular_throughput;
        throughput_1 = ((_e360 * _e361) + _e363);
        let _e366 = throughput_1;
        if all((_e366 <= vec3(0f))) {
            break;
        }
        continuing {
            let _e371 = bounce;
            bounce = (_e371 + 1u);
        }
    }
    let _e375 = primary_albedo;
    let _e377 = radiance.diffuse;
    let _e380 = radiance.specular;
    let _e383 = radiance.emissive;
    radiance.total = (((_e375 * _e377) + _e380) + _e383);
    let _e386 = radiance.total;
    let _e388 = radiance.total;
    let _e392 = radiance.diffuse;
    let _e394 = radiance.diffuse;
    let _e399 = radiance.specular;
    let _e401 = radiance.specular;
    let _e406 = radiance.emissive;
    let _e408 = radiance.emissive;
    is_finite = (((all((_e386 == _e388)) && all((_e392 == _e394))) && all((_e399 == _e401))) && all((_e406 == _e408)));
    let _e413 = is_finite;
    if !(_e413) {
        let _e415 = zero_path_radiance();
        return _e415;
    }
    let _e421 = radiance.total;
    scale = min(vec3(1f), (vec3(MAX_RADIANCE) / max(_e421, vec3(0.00000000000000000001f))));
    let _e429 = scale;
    let _e430 = radiance.total;
    radiance.total = (_e430 * _e429);
    let _e433 = scale;
    let _e434 = radiance.diffuse;
    radiance.diffuse = (_e434 * _e433);
    let _e437 = scale;
    let _e438 = radiance.specular;
    radiance.specular = (_e438 * _e437);
    let _e441 = scale;
    let _e442 = radiance.emissive;
    radiance.emissive = (_e442 * _e441);
    let _e444 = radiance;
    return _e444;
}

@compute @workgroup_size(8, 4, 1) 
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var total: vec4<f32>;
    var total_diffuse: vec4<f32>;
    var total_specular: vec4<f32>;
    var total_emissive: vec4<f32>;
    var global_index: u32;
    var rng: RandomState;
    var num_paths: u32;
    var sum: PathRadiance;
    var i: u32;
    var jitter: vec2<f32>;
    var ray_dir: vec3<f32>;
    var sample: PathRadiance;
    var count: f32;

    let _e18 = camera.target_size;
    if any((global_id.xy >= _e18)) {
        return;
    }
    total = vec4(0f);
    total_diffuse = vec4(0f);
    total_specular = vec4(0f);
    total_emissive = vec4(0f);
    let _e34 = parameters.reset_accumulation;
    if (_e34 == 0u) {
        let _e38 = textureLoad(accumulator, global_id.xy);
        total = _e38;
        let _e40 = parameters.max_accumulated_samples;
        let _e44 = total.w;
        let _e46 = parameters.max_accumulated_samples;
        if ((_e40 != 0u) && (_e44 >= f32(_e46))) {
            return;
        }
        let _e51 = textureLoad(accumulator_diffuse, global_id.xy);
        total_diffuse = _e51;
        let _e53 = textureLoad(accumulator_specular, global_id.xy);
        total_specular = _e53;
        let _e55 = textureLoad(accumulator_emissive, global_id.xy);
        total_emissive = _e55;
    }
    let _e59 = camera.target_size.x;
    global_index = ((global_id.y * _e59) + global_id.x);
    let _e64 = global_index;
    let _e66 = parameters.frame_index;
    let _e67 = random_init(_e64, _e66);
    rng = _e67;
    let _e70 = parameters.num_brdf_samples;
    num_paths = max(_e70, 1u);
    let _e74 = zero_path_radiance();
    sum = _e74;
    let _e77 = num_paths;
    i = 0u;
    loop {
        let _e79 = i;
        if (_e79 < _e77) {
        } else {
            break;
        }
        let _e83 = random_gen((&rng));
        let _e84 = random_gen((&rng));
        let _e87 = parameters.jitter_primary_rays;
        jitter = select(vec2(0.5f), vec2<f32>(_e83, _e84), (_e87 != 0u));
        let _e92 = camera;
        let _e95 = jitter;
        let _e97 = get_ray_direction_at(_e92, (vec2<f32>(global_id.xy) + _e95));
        ray_dir = _e97;
        let _e99 = ray_dir;
        let _e100 = trace_path(_e99, (&rng));
        sample = _e100;
        let _e104 = sample.total;
        let _e105 = sum.total;
        sum.total = (_e105 + _e104);
        let _e109 = sample.diffuse;
        let _e110 = sum.diffuse;
        sum.diffuse = (_e110 + _e109);
        let _e114 = sample.specular;
        let _e115 = sum.specular;
        sum.specular = (_e115 + _e114);
        let _e119 = sample.emissive;
        let _e120 = sum.emissive;
        sum.emissive = (_e120 + _e119);
        continuing {
            let _e122 = i;
            i = (_e122 + 1u);
        }
    }
    let _e125 = num_paths;
    count = f32(_e125);
    let _e129 = total;
    let _e131 = sum.total;
    let _e132 = count;
    textureStore(accumulator, global_id.xy, (_e129 + vec4<f32>(_e131, _e132)));
    let _e136 = total_diffuse;
    let _e138 = sum.diffuse;
    let _e139 = count;
    textureStore(accumulator_diffuse, global_id.xy, (_e136 + vec4<f32>(_e138, _e139)));
    let _e143 = total_specular;
    let _e145 = sum.specular;
    let _e146 = count;
    textureStore(accumulator_specular, global_id.xy, (_e143 + vec4<f32>(_e145, _e146)));
    let _e150 = total_emissive;
    let _e152 = sum.emissive;
    let _e153 = count;
    textureStore(accumulator_emissive, global_id.xy, (_e150 + vec4<f32>(_e152, _e153)));
}
