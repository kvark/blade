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

struct BsdfSample {
    dir: vec3<f32>,
    pdf: f32,
}

struct EnvImportantSample {
    pixel: vec2<i32>,
    pdf: f32,
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

struct MainParams {
    frame_index: u32,
    num_environment_samples: u32,
    num_brdf_samples: u32,
    environment_importance_sampling: u32,
    tap_count: u32,
    tap_radius: f32,
    tap_confidence_near: f32,
    tap_confidence_far: f32,
    t_start: f32,
    use_pairwise_mis: u32,
    defensive_mis: f32,
    use_motion_vectors: u32,
}

struct StoredReservoir {
    light_uv: vec2<f32>,
    light_index: u32,
    target_score: f32,
    contribution_weight: f32,
    confidence: f32,
}

struct Radiance {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

struct LiveReservoir {
    selected_uv: vec2<f32>,
    selected_light_index: u32,
    selected_target_score: f32,
    selected_radiance: Radiance,
    weight_sum: f32,
    history: f32,
}

struct TargetScore {
    radiance: Radiance,
    score: f32,
}

struct RestirOutput {
    radiance: Radiance,
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
const SIGMA_N: f32 = 4f;
const MOTION_SCALE: f32 = 0.02f;
const USE_MOTION_VECTORS: bool = true;
const WRITE_DEBUG_IMAGE: bool = DEBUG_MODE;
const MAX_RESERVOIRS: u32 = 4u;
const DECOUPLED_SHADING: bool = false;
const FACTOR_CANDIDATES: u32 = 3u;

var<storage, read_write> debug_buf: DebugBuffer;
var env_weights: texture_2d<f32>;
var<storage> vertex_buffers: binding_array<VertexBuffer>;
var<storage> index_buffers: binding_array<IndexBuffer>;
var<storage> hit_entries: array<HitEntry>;
var textures: binding_array<texture_2d<f32>>;
var sampler_linear: sampler;
var env_map: texture_2d<f32>;
var sampler_nearest: sampler;
var<uniform> camera: CameraParams;
var<uniform> prev_camera: CameraParams;
var<uniform> parameters: MainParams;
var<uniform> debug: DebugParams;
var acc_struct: acceleration_structure;
var prev_acc_struct: acceleration_structure;
var<storage, read_write> reservoirs: array<StoredReservoir>;
var<storage> prev_reservoirs: array<StoredReservoir>;
var t_depth: texture_2d<f32>;
var t_prev_depth: texture_2d<f32>;
var t_basis: texture_2d<f32>;
var t_prev_basis: texture_2d<f32>;
var t_flat_normal: texture_2d<f32>;
var t_prev_flat_normal: texture_2d<f32>;
var t_diffuse_albedo: texture_2d<f32>;
var t_prev_diffuse_albedo: texture_2d<f32>;
var t_specular_f0: texture_2d<f32>;
var t_prev_specular_f0: texture_2d<f32>;
var t_motion: texture_2d<f32>;
var out_diffuse: texture_storage_2d<rgba16float,write>;
var out_specular: texture_storage_2d<rgba16float,write>;
var out_debug: texture_storage_2d<rgba8unorm,write>;
var<private> debug_len: f32;















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

fn debug_line(a_2: vec3<f32>, b_2: vec3<f32>, color_1: u32) {
    var index: u32;

    let _e5 = debug_buf.open;
    if (_e5 != 0u) {
        let _e10 = atomicAdd((&debug_buf.instance_count), 1u);
        index = _e10;
        let _e12 = index;
        let _e14 = debug_buf.capacity;
        if (_e12 < _e14) {
            let _e17 = index;
            debug_buf.lines[_e17] = DebugLine(DebugPoint(a_2, color_1), DebugPoint(b_2, color_1));
        } else {
            let _e24 = atomicSub((&debug_buf.instance_count), 1u);
        }
    }
}

fn compute_luminocity(color_2: vec3<f32>) -> f32 {
    return dot(color_2, LUMINOCITY_WEIGHTS);
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
    let _e9 = n_dot_l_1;
    let _e12 = n_dot_v_1;
    if ((_e9 <= 0f) || (_e12 <= 0f)) {
        let _e16 = zero_brdf();
        return _e16;
    }
    half_dir = normalize((view_dir + light_dir));
    let _e20 = half_dir;
    n_dot_h_1 = max(dot(normal, _e20), 0f);
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

fn make_tangent_frame(normal_1: vec3<f32>) -> mat3x3<f32> {
    var s: f32;
    var a_3: f32;
    var b_3: f32;

    s = select(-(1f), 1f, (normal_1.z >= 0f));
    let _e12 = s;
    a_3 = (-(1f) / (_e12 + normal_1.z));
    let _e20 = a_3;
    b_3 = ((normal_1.x * normal_1.y) * _e20);
    let _e24 = s;
    let _e29 = a_3;
    let _e32 = s;
    let _e33 = b_3;
    let _e35 = s;
    let _e40 = b_3;
    let _e41 = s;
    let _e45 = a_3;
    return mat3x3<f32>(vec3<f32>((1f + (((_e24 * normal_1.x) * normal_1.x) * _e29)), (_e32 * _e33), (-(_e35) * normal_1.x)), vec3<f32>(_e40, (_e41 + ((normal_1.y * normal_1.y) * _e45)), -(normal_1.y)), normal_1);
}

fn sample_circle_uniform(random: f32) -> vec2<f32> {
    var angle: f32;

    angle = ((2f * PI) * random);
    let _e7 = angle;
    let _e9 = angle;
    return vec2<f32>(cos(_e7), sin(_e9));
}

fn compute_bsdf_pdf(mat_5: Material, normal_2: vec3<f32>, view_dir_1: vec3<f32>, light_dir_1: vec3<f32>) -> f32 {
    var n_dot_l_2: f32;
    var half_dir_1: vec3<f32>;
    var n_dot_h_2: f32;
    var v_dot_h_1: f32;
    var specular_pdf: f32;
    var diffuse_pdf: f32;

    n_dot_l_2 = dot(normal_2, light_dir_1);
    let _e7 = n_dot_l_2;
    if ((_e7 <= 0f) || (dot(normal_2, view_dir_1) <= 0f)) {
        return 0f;
    }
    half_dir_1 = normalize((view_dir_1 + light_dir_1));
    let _e18 = half_dir_1;
    n_dot_h_2 = max(dot(normal_2, _e18), 0f);
    let _e23 = half_dir_1;
    v_dot_h_1 = max(dot(view_dir_1, _e23), 0.00001f);
    let _e28 = n_dot_h_2;
    let _e29 = material_alpha(mat_5);
    let _e30 = distribution_ggx(_e28, _e29);
    let _e31 = n_dot_h_2;
    let _e34 = v_dot_h_1;
    specular_pdf = ((_e30 * _e31) / (4f * _e34));
    let _e38 = n_dot_l_2;
    diffuse_pdf = (_e38 / PI);
    let _e42 = diffuse_pdf;
    let _e43 = specular_pdf;
    let _e44 = specular_sampling_ratio(mat_5);
    return mix(_e42, _e43, _e44);
}

fn evaluate_bsdf(mat_6: Material, normal_3: vec3<f32>, view_dir_2: vec3<f32>, light_dir_2: vec3<f32>) -> vec3<f32> {
    var brdf: BrdfLobes;

    let _e5 = evaluate_brdf(mat_6, normal_3, view_dir_2, light_dir_2);
    brdf = _e5;
    let _e9 = brdf.diffuse;
    let _e13 = brdf.specular;
    return ((mat_6.diffuse_albedo * vec3(_e9)) + _e13);
}

fn sample_hemisphere_cosine(rng_3: ptr<function, RandomState>) -> vec3<f32> {
    var r_1: f32;
    var tangential: vec2<f32>;

    let _e2 = random_gen(rng_3);
    r_1 = _e2;
    let _e4 = r_1;
    let _e6 = random_gen(rng_3);
    let _e7 = sample_circle_uniform(_e6);
    tangential = (vec2(sqrt(_e4)) * _e7);
    let _e11 = tangential;
    let _e14 = r_1;
    return vec3<f32>(_e11, sqrt(max(0f, (1f - _e14))));
}

fn sample_ggx_half_dir(alpha_3: f32, rng_4: ptr<function, RandomState>) -> vec3<f32> {
    var a2_2: f32;
    var r_2: f32;
    var cos_theta_2: f32;
    var sin_theta: f32;

    a2_2 = (alpha_3 * alpha_3);
    let _e5 = random_gen(rng_4);
    r_2 = _e5;
    let _e8 = r_2;
    let _e11 = a2_2;
    let _e14 = r_2;
    cos_theta_2 = sqrt(((1f - _e8) / (1f + ((_e11 - 1f) * _e14))));
    let _e22 = cos_theta_2;
    let _e23 = cos_theta_2;
    sin_theta = sqrt(max(0f, (1f - (_e22 * _e23))));
    let _e29 = sin_theta;
    let _e30 = random_gen(rng_4);
    let _e31 = sample_circle_uniform(_e30);
    let _e34 = cos_theta_2;
    return vec3<f32>((vec2(_e29) * _e31), _e34);
}

fn sample_bsdf(mat_7: Material, normal_4: vec3<f32>, view_dir_3: vec3<f32>, rng_5: ptr<function, RandomState>) -> BsdfSample {
    var frame: mat3x3<f32>;
    var dir: vec3<f32>;
    var half_dir_2: vec3<f32>;

    let _e5 = make_tangent_frame(normal_4);
    frame = _e5;
    dir = vec3<f32>();
    let _e9 = random_gen(rng_5);
    let _e10 = specular_sampling_ratio(mat_7);
    if (_e9 < _e10) {
        let _e12 = frame;
        let _e13 = material_alpha(mat_7);
        let _e14 = sample_ggx_half_dir(_e13, rng_5);
        half_dir_2 = (_e12 * _e14);
        let _e18 = half_dir_2;
        let _e21 = half_dir_2;
        dir = ((vec3((2f * dot(view_dir_3, _e18))) * _e21) - view_dir_3);
    } else {
        let _e25 = frame;
        let _e26 = sample_hemisphere_cosine(rng_5);
        dir = (_e25 * _e26);
    }
    let _e28 = dir;
    dir = normalize(_e28);
    let _e30 = dir;
    let _e31 = dir;
    let _e32 = compute_bsdf_pdf(mat_7, normal_4, view_dir_3, _e31);
    return BsdfSample(_e30, _e32);
}

fn compute_latitude_area_bounds(texel_y: i32, dim: u32) -> vec2<f32> {
    return cos(((vec2<f32>(vec2<i32>(texel_y, (texel_y + 1i))) / vec2(f32(dim))) * vec2(PI)));
}

fn compute_texel_solid_angle(itc: vec2<i32>, dim_1: vec2<u32>) -> f32 {
    var meridian_solid_angle: f32;
    var bounds: vec2<f32>;
    var meridian_part: f32;

    meridian_solid_angle = ((4f * PI) / f32(dim_1.x));
    let _e13 = compute_latitude_area_bounds(itc.y, dim_1.y);
    bounds = _e13;
    let _e17 = bounds.x;
    let _e19 = bounds.y;
    meridian_part = (0.5f * (_e17 - _e19));
    let _e23 = meridian_solid_angle;
    let _e24 = meridian_part;
    return (_e23 * _e24);
}

fn generate_environment_sample(rng_6: ptr<function, RandomState>, dim_2: vec2<u32>) -> EnvImportantSample {
    var es: EnvImportantSample;
    var mip: i32;
    var itc_1: vec2<i32>;
    var weights: vec4<f32>;
    var sum: f32;
    var r_3: f32;
    var weight: f32;

    es = EnvImportantSample();
    es.pdf = 1f;
    let _e8 = textureNumLevels(env_weights);
    mip = i32(_e8);
    itc_1 = vec2(0i);
    loop {
        let _e14 = mip;
        if (_e14 != 0i) {
            let _e18 = mip;
            mip = (_e18 - 1i);
            let _e20 = itc_1;
            let _e21 = mip;
            let _e22 = textureLoad(env_weights, _e20, _e21);
            weights = _e22;
            let _e26 = weights;
            sum = dot(vec4(1f), _e26);
            let _e29 = random_gen(rng_6);
            let _e30 = sum;
            r_3 = (_e29 * _e30);
            weight = f32();
            let _e36 = itc_1;
            itc_1 = (_e36 * vec2(2i));
            let _e39 = r_3;
            let _e41 = weights.x;
            let _e43 = weights.y;
            if (_e39 >= (_e41 + _e43)) {
                let _e48 = itc_1.y;
                itc_1.y = (_e48 + 1i);
                let _e50 = r_3;
                let _e52 = weights.x;
                let _e54 = weights.y;
                let _e57 = weights.z;
                if (_e50 >= ((_e52 + _e54) + _e57)) {
                    let _e61 = weights.w;
                    weight = _e61;
                    let _e64 = itc_1.x;
                    itc_1.x = (_e64 + 1i);
                } else {
                    let _e67 = weights.z;
                    weight = _e67;
                }
            } else {
                let _e68 = r_3;
                let _e70 = weights.x;
                if (_e68 >= _e70) {
                    let _e73 = weights.y;
                    weight = _e73;
                    let _e76 = itc_1.x;
                    itc_1.x = (_e76 + 1i);
                } else {
                    let _e79 = weights.x;
                    weight = _e79;
                }
            }
            let _e81 = weight;
            let _e82 = sum;
            let _e84 = es.pdf;
            es.pdf = (_e84 * (_e81 / _e82));
        } else {
            break;
        }
    }
    let _e87 = itc_1;
    let _e88 = compute_texel_solid_angle(_e87, dim_2);
    let _e89 = es.pdf;
    es.pdf = (_e89 / _e88);
    let _e92 = itc_1;
    es.pixel = _e92;
    let _e93 = es;
    return _e93;
}

fn compute_environment_sample_pdf(pixel_1: vec2<i32>, dim_3: vec2<u32>) -> f32 {
    var itc_2: vec2<i32>;
    var pdf: f32;
    var mip_count: i32;
    var mip_1: i32;
    var rem: vec2<i32>;
    var weights_1: vec4<f32>;
    var sum_1: f32;
    var w2: vec2<f32>;
    var weight_1: f32;

    itc_2 = pixel_1;
    let _e6 = itc_2;
    let _e7 = compute_texel_solid_angle(_e6, dim_3);
    pdf = (1f / _e7);
    let _e10 = textureNumLevels(env_weights);
    mip_count = i32(_e10);
    let _e13 = mip_count;
    mip_1 = 0i;
    loop {
        let _e16 = mip_1;
        if (_e16 < _e13) {
        } else {
            break;
        }
        let _e18 = itc_2;
        rem = (_e18 & vec2(1i));
        let _e23 = itc_2;
        itc_2 = (_e23 >> vec2(1u));
        let _e27 = itc_2;
        let _e28 = mip_1;
        let _e29 = textureLoad(env_weights, _e27, _e28);
        weights_1 = _e29;
        let _e33 = weights_1;
        sum_1 = dot(vec4(1f), _e33);
        let _e36 = weights_1;
        let _e38 = weights_1;
        let _e41 = rem.y;
        w2 = select(_e36.xy, _e38.zw, (_e41 != 0i));
        let _e47 = w2.x;
        let _e49 = w2.y;
        let _e51 = rem.x;
        weight_1 = select(_e47, _e49, (_e51 != 0i));
        let _e56 = weight_1;
        let _e57 = sum_1;
        let _e59 = pdf;
        pdf = (_e59 * (_e56 / _e57));
        continuing {
            let _e61 = mip_1;
            mip_1 = (_e61 + 1i);
        }
    }
    let _e64 = pdf;
    return _e64;
}

fn compare_flat_normals(a_4: vec3<f32>, b_4: vec3<f32>) -> f32 {
    return pow(max(0f, dot(a_4, b_4)), SIGMA_N);
}

fn compare_depths(a_5: f32, b_5: f32) -> f32 {
    return (1f - smoothstep(0f, 100f, abs((a_5 - b_5))));
}

fn compare_surfaces(a_6: Surface, b_6: Surface) -> f32 {
    var r_normal: f32;
    var r_depth: f32;

    let _e6 = compare_flat_normals(a_6.flat_normal, b_6.flat_normal);
    r_normal = _e6;
    let _e10 = compare_depths(a_6.depth, b_6.depth);
    r_depth = _e10;
    let _e12 = r_normal;
    let _e13 = r_depth;
    return (_e12 * _e13);
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
        let _e26 = indices.x;
        let _e28 = index_buffers[entry_1.index_buf].data[_e26];
        let _e33 = indices.y;
        let _e35 = index_buffers[entry_1.index_buf].data[_e33];
        let _e40 = indices.z;
        let _e42 = index_buffers[entry_1.index_buf].data[_e40];
        indices = vec3<u32>(_e28, _e35, _e42);
    }
    let _e44 = indices;
    return _e44;
}

fn make_barycentrics(uv: vec2<f32>) -> vec3<f32> {
    var w: f32;

    w = ((1f - uv.x) - uv.y);
    let _e14 = w;
    return vec3<f32>(_e14, uv.x, uv.y);
}

fn sample_hit_material(entry_2: HitEntry, tex_coords: vec2<f32>, lod: f32, ignore_textures: u32) -> Material {
    var base_color_1: vec3<f32>;
    var metalness_1: f32;
    var roughness_1: f32;
    var mr: vec4<f32>;

    base_color_1 = unpack4x8unorm(entry_2.base_color_factor).xyz;
    if ((ignore_textures & DebugTextureFlags_ALBEDO) == 0u) {
        let _e21 = textureSampleLevel(textures[entry_2.base_color_texture], sampler_linear, tex_coords, lod);
        let _e23 = base_color_1;
        base_color_1 = (_e23 * _e21.xyz);
    }
    metalness_1 = entry_2.metalness;
    roughness_1 = entry_2.roughness;
    if ((ignore_textures & DebugTextureFlags_METALLIC_ROUGHNESS) == 0u) {
        let _e35 = textureSampleLevel(textures[entry_2.metallic_roughness_texture], sampler_linear, tex_coords, lod);
        mr = _e35;
        let _e38 = mr.y;
        let _e39 = roughness_1;
        roughness_1 = (_e39 * _e38);
        let _e42 = mr.z;
        let _e43 = metalness_1;
        metalness_1 = (_e43 * _e42);
    }
    let _e45 = base_color_1;
    let _e46 = metalness_1;
    let _e47 = roughness_1;
    let _e48 = material_from_metallic_roughness(_e45, _e46, _e47);
    return _e48;
}

fn sample_hit_emissive(entry_3: HitEntry, tex_coords_1: vec2<f32>, lod_1: f32, ignore_textures_1: u32) -> vec3<f32> {
    var emissive: vec3<f32>;

    emissive = entry_3.emissive_factor.xyz;
    if ((ignore_textures_1 & DebugTextureFlags_EMISSIVE) == 0u) {
        let _e20 = textureSampleLevel(textures[entry_3.emissive_texture], sampler_linear, tex_coords_1, lod_1);
        let _e22 = emissive;
        emissive = (_e22 * _e20.xyz);
    }
    let _e24 = emissive;
    return _e24;
}

fn sample_hit_normal_map(entry_4: HitEntry, tex_coords_2: vec2<f32>, lod_2: f32, ignore_textures_2: u32) -> vec3<f32> {
    var raw_unorm: vec2<f32>;
    var n_xy: vec2<f32>;

    if ((ignore_textures_2 & DebugTextureFlags_NORMAL) != 0u) {
        return vec3<f32>(0f, 0f, 1f);
    }
    let _e21 = textureSampleLevel(textures[entry_4.normal_texture], sampler_linear, tex_coords_2, lod_2);
    raw_unorm = _e21.xy;
    let _e26 = raw_unorm;
    n_xy = (vec2(entry_4.normal_scale) * ((vec2(2f) * _e26) - vec2(1f)));
    let _e35 = n_xy;
    let _e38 = n_xy;
    let _e39 = n_xy;
    return vec3<f32>(_e35, sqrt(max(0f, (1f - dot(_e38, _e39)))));
}

fn hit_normal(entry_5: HitEntry, object_to_world: mat4x3<f32>, normal_5: vec3<f32>) -> vec3<f32> {
    var linear: mat3x3<f32>;

    let _e10 = affine_linear(object_to_world);
    let _e12 = affine_linear(entry_5.geometry_to_object);
    linear = (_e10 * _e12);
    let _e15 = linear;
    return normalize((_e15 * normal_5));
}

fn hit_tangent_space(entry_6: HitEntry, object_to_world_1: mat4x3<f32>, normal_6: vec3<f32>, tangent: vec3<f32>, bitangent_sign_1: f32) -> mat3x3<f32> {
    var linear_1: mat3x3<f32>;
    var n_2: vec3<f32>;

    let _e12 = affine_linear(object_to_world_1);
    let _e14 = affine_linear(entry_6.geometry_to_object);
    linear_1 = (_e12 * _e14);
    let _e17 = hit_normal(entry_6, object_to_world_1, normal_6);
    n_2 = _e17;
    let _e19 = n_2;
    let _e20 = linear_1;
    let _e22 = linear_1;
    let _e25 = tangent_basis(_e19, (_e20 * tangent), bitangent_sign_1, sign(determinant(_e22)));
    return _e25;
}

fn map_equirect_dir_to_uv(dir_1: vec3<f32>) -> vec2<f32> {
    var yaw: f32;
    var pitch: f32;

    yaw = asin(dir_1.y);
    pitch = atan2(dir_1.x, dir_1.z);
    let _e17 = pitch;
    let _e22 = yaw;
    return (vec2<f32>((_e17 + PI), ((-(2f) * _e22) + PI)) / vec2((2f * PI)));
}

fn map_equirect_uv_to_dir(uv_1: vec2<f32>) -> vec3<f32> {
    var yaw_1: f32;
    var pitch_1: f32;

    yaw_1 = (PI * (0.5f - uv_1.y));
    pitch_1 = ((2f * PI) * (uv_1.x - 0.5f));
    let _e24 = yaw_1;
    let _e26 = pitch_1;
    let _e29 = yaw_1;
    let _e31 = yaw_1;
    let _e33 = pitch_1;
    return vec3<f32>((cos(_e24) * sin(_e26)), sin(_e29), (cos(_e31) * cos(_e33)));
}

fn sample_light_from_environment(rng_7: ptr<function, RandomState>) -> LightSample {
    var dim_4: vec2<u32>;
    var es_1: EnvImportantSample;
    var ls: LightSample;
    var u: f32;
    var bounds_1: vec2<f32>;
    var v_2: f32;

    let _e11 = textureDimensions(env_map, 0i);
    dim_4 = _e11;
    let _e13 = dim_4;
    let _e14 = generate_environment_sample(rng_7, _e13);
    es_1 = _e14;
    ls = LightSample();
    let _e20 = es_1.pdf;
    ls.pdf = _e20;
    let _e23 = es_1.pixel;
    let _e25 = textureLoad(env_map, _e23, 0i);
    ls.radiance = _e25.xyz;
    let _e29 = es_1.pixel.x;
    let _e31 = random_gen(rng_7);
    let _e34 = dim_4.x;
    u = ((f32(_e29) + _e31) / f32(_e34));
    let _e40 = es_1.pixel.y;
    let _e42 = dim_4.y;
    let _e43 = compute_latitude_area_bounds(_e40, _e42);
    bounds_1 = _e43;
    let _e46 = bounds_1.x;
    let _e48 = bounds_1.y;
    let _e49 = random_gen(rng_7);
    v_2 = (acos(mix(_e46, _e48, _e49)) / PI);
    let _e56 = u;
    let _e57 = v_2;
    ls.uv = vec2<f32>(_e56, _e57);
    let _e59 = ls;
    return _e59;
}

fn compute_light_pdf(uv_2: vec2<f32>, importance: bool) -> f32 {
    var dim_5: vec2<u32>;
    var pixel_2: vec2<i32>;

    if !(importance) {
        return (1f / (4f * PI));
    }
    let _e18 = textureDimensions(env_map, 0i);
    dim_5 = _e18;
    let _e20 = dim_5;
    let _e26 = dim_5;
    pixel_2 = clamp(vec2<i32>((uv_2 * vec2<f32>(_e20))), vec2(0i), (vec2<i32>(_e26) - vec2(1i)));
    let _e33 = pixel_2;
    let _e34 = dim_5;
    let _e35 = compute_environment_sample_pdf(_e33, _e34);
    return _e35;
}

fn evaluate_environment(dir_2: vec3<f32>) -> vec3<f32> {
    var uv_3: vec2<f32>;

    let _e10 = map_equirect_dir_to_uv(dir_2);
    uv_3 = _e10;
    let _e12 = uv_3;
    let _e14 = textureSampleLevel(env_map, sampler_nearest, _e12, 0f);
    return _e14.xyz;
}

fn evaluate_environment_background(dir_3: vec3<f32>) -> vec3<f32> {
    var uv_4: vec2<f32>;

    let _e10 = map_equirect_dir_to_uv(dir_3);
    uv_4 = _e10;
    let _e12 = uv_4;
    let _e14 = textureSampleLevel(env_map, sampler_linear, _e12, 0f);
    return _e14.xyz;
}

fn sample_light_from_sphere(rng_8: ptr<function, RandomState>) -> LightSample {
    var a_7: f32;
    var h: f32;
    var tangential_1: vec2<f32>;
    var dir_4: vec3<f32>;
    var ls_1: LightSample;

    let _e10 = random_gen(rng_8);
    a_7 = _e10;
    let _e14 = random_gen(rng_8);
    h = (1f - (2f * _e14));
    let _e20 = h;
    let _e21 = h;
    let _e26 = a_7;
    let _e27 = sample_circle_uniform(_e26);
    tangential_1 = (vec2(sqrt(max(0f, (1f - (_e20 * _e21))))) * _e27);
    let _e32 = tangential_1.x;
    let _e33 = h;
    let _e35 = tangential_1.y;
    dir_4 = vec3<f32>(_e32, _e33, _e35);
    ls_1 = LightSample();
    let _e41 = dir_4;
    let _e42 = map_equirect_dir_to_uv(_e41);
    ls_1.uv = _e42;
    ls_1.pdf = (1f / (4f * PI));
    let _e51 = ls_1.uv;
    let _e53 = textureSampleLevel(env_map, sampler_nearest, _e51, 0f);
    ls_1.radiance = _e53.xyz;
    let _e55 = ls_1;
    return _e55;
}

fn sample_light(importance_1: bool, rng_9: ptr<function, RandomState>) -> LightSample {
    if importance_1 {
        let _e11 = sample_light_from_environment(rng_9);
        return _e11;
    } else {
        let _e12 = sample_light_from_sphere(rng_9);
        return _e12;
    }
}

fn zero_radiance() -> Radiance {
    return Radiance(vec3(0f), vec3(0f));
}

fn reflect_light(brdf_1: BrdfLobes, light: vec3<f32>) -> Radiance {
    return Radiance((vec3(brdf_1.diffuse) * light), (brdf_1.specular * light));
}

fn compute_target_score(radiance: Radiance, diffuse_albedo: vec3<f32>) -> f32 {
    let _e38 = compute_luminocity(((diffuse_albedo * radiance.diffuse) + radiance.specular));
    return _e38;
}

fn get_reservoir_index(pixel_3: vec2<i32>, cam: CameraParams) -> i32 {
    if all((vec2<u32>(pixel_3) < cam.target_size)) {
        return ((pixel_3.y * i32(cam.target_size.x)) + pixel_3.x);
    } else {
        return -(1i);
    }
}

fn get_pixel_from_reservoir_index(index_1: i32, cam_1: CameraParams) -> vec2<i32> {
    var y: i32;
    var x_1: i32;

    y = (index_1 / i32(cam_1.target_size.x));
    let _e39 = y;
    x_1 = (index_1 - (_e39 * i32(cam_1.target_size.x)));
    let _e46 = x_1;
    let _e47 = y;
    return vec2<i32>(_e46, _e47);
}

fn bump_reservoir(r_4: ptr<function, LiveReservoir>, history: f32) {
    let _e35 = (*r_4).history;
    (*r_4).history = (_e35 + history);
}

fn merge_reservoir(r_5: ptr<function, LiveReservoir>, other: LiveReservoir, random_1: f32) -> bool {
    let _e37 = (*r_5).weight_sum;
    (*r_5).weight_sum = (_e37 + other.weight_sum);
    let _e41 = (*r_5).history;
    (*r_5).history = (_e41 + other.history);
    let _e44 = (*r_5).weight_sum;
    if ((_e44 * random_1) < other.weight_sum) {
        (*r_5).selected_light_index = other.selected_light_index;
        (*r_5).selected_uv = other.selected_uv;
        (*r_5).selected_target_score = other.selected_target_score;
        (*r_5).selected_radiance = other.selected_radiance;
        return true;
    } else {
        return false;
    }
}

fn normalize_reservoir(r_6: ptr<function, LiveReservoir>, history_1: f32) {
    var h_1: f32;

    let _e35 = (*r_6).history;
    h_1 = _e35;
    let _e37 = h_1;
    if (_e37 > 0f) {
        let _e41 = h_1;
        let _e43 = (*r_6).weight_sum;
        (*r_6).weight_sum = (_e43 * (history_1 / _e41));
        (*r_6).history = history_1;
    }
}

fn unpack_reservoir(f: StoredReservoir, max_confidence: f32, radiance_1: Radiance) -> LiveReservoir {
    var r_7: LiveReservoir;
    var history_2: f32;

    r_7 = LiveReservoir();
    r_7.selected_light_index = f.light_index;
    r_7.selected_uv = f.light_uv;
    r_7.selected_target_score = f.target_score;
    r_7.selected_radiance = radiance_1;
    history_2 = min(f.confidence, max_confidence);
    let _e51 = history_2;
    r_7.weight_sum = ((f.contribution_weight * f.target_score) * _e51);
    let _e54 = history_2;
    r_7.history = _e54;
    let _e55 = r_7;
    return _e55;
}

fn pack_reservoir_detail(r_8: LiveReservoir, denom_factor: f32) -> StoredReservoir {
    var f_1: StoredReservoir;
    var denom_1: f32;

    f_1 = StoredReservoir();
    f_1.light_index = r_8.selected_light_index;
    f_1.light_uv = r_8.selected_uv;
    f_1.target_score = r_8.selected_target_score;
    f_1.confidence = r_8.history;
    let _e45 = f_1.target_score;
    denom_1 = (_e45 * denom_factor);
    let _e51 = denom_1;
    let _e53 = denom_1;
    f_1.contribution_weight = select(0f, (r_8.weight_sum / _e51), (_e53 > 0f));
    let _e57 = f_1;
    return _e57;
}

fn read_surface(pixel_4: vec2<i32>) -> Surface {
    var surface_1: Surface;
    var specular_2: vec4<f32>;

    surface_1 = Surface();
    let _e37 = textureLoad(t_basis, pixel_4, 0i);
    surface_1.basis = normalize(_e37);
    let _e41 = textureLoad(t_flat_normal, pixel_4, 0i);
    surface_1.flat_normal = normalize(_e41.xyz);
    let _e46 = textureLoad(t_depth, pixel_4, 0i);
    surface_1.depth = _e46.x;
    let _e49 = camera;
    let _e50 = get_ray_direction(_e49, pixel_4);
    surface_1.view_dir = -(_e50);
    let _e54 = textureLoad(t_diffuse_albedo, pixel_4, 0i);
    surface_1.diffuse_albedo = _e54.xyz;
    let _e57 = textureLoad(t_specular_f0, pixel_4, 0i);
    specular_2 = _e57;
    let _e60 = specular_2;
    surface_1.specular_f0 = _e60.xyz;
    let _e64 = specular_2.w;
    surface_1.roughness = _e64;
    let _e65 = surface_1;
    return _e65;
}

fn read_prev_surface(pixel_5: vec2<i32>) -> Surface {
    var surface_2: Surface;
    var specular_3: vec4<f32>;

    surface_2 = Surface();
    let _e37 = textureLoad(t_prev_basis, pixel_5, 0i);
    surface_2.basis = normalize(_e37);
    let _e41 = textureLoad(t_prev_flat_normal, pixel_5, 0i);
    surface_2.flat_normal = normalize(_e41.xyz);
    let _e46 = textureLoad(t_prev_depth, pixel_5, 0i);
    surface_2.depth = _e46.x;
    let _e49 = prev_camera;
    let _e50 = get_ray_direction(_e49, pixel_5);
    surface_2.view_dir = -(_e50);
    let _e54 = textureLoad(t_prev_diffuse_albedo, pixel_5, 0i);
    surface_2.diffuse_albedo = _e54.xyz;
    let _e57 = textureLoad(t_prev_specular_f0, pixel_5, 0i);
    specular_3 = _e57;
    let _e60 = specular_3;
    surface_2.specular_f0 = _e60.xyz;
    let _e64 = specular_3.w;
    surface_2.roughness = _e64;
    let _e65 = surface_2;
    return _e65;
}

fn surface_normal(surface_3: Surface) -> vec3<f32> {
    let _e38 = qrot(surface_3.basis, vec3<f32>(0f, 0f, 1f));
    return _e38;
}

fn surface_material(surface_4: Surface) -> Material {
    return Material(surface_4.diffuse_albedo, surface_4.specular_f0, surface_4.roughness);
}

fn evaluate_incoming_radiance(acs: acceleration_structure, position: vec3<f32>, direction: vec3<f32>, ray_len: f32, debug_color: u32) -> vec3<f32> {
    var rq: ray_query;
    var intersection: RayIntersection;
    var hit: bool;
    var color_3: u32;
    var entry_7: HitEntry;
    var indices_1: vec3<u32>;
    var barycentrics: vec3<f32>;
    var tex_coords_3: vec2<f32>;

    let _e41 = parameters.t_start;
    let _e43 = camera.depth;
    rayQueryInitialize((&rq), acs, RayDesc(128u, 255u, _e41, _e43, position, direction));
    let _e45 = rayQueryProceed((&rq));
    let _e46 = rayQueryGetCommittedIntersection((&rq));
    intersection = _e46;
    if (DEBUG_MODE && (ray_len > 0f)) {
        let _e53 = intersection.kind;
        hit = (_e53 != 0u);
        let _e59 = hit;
        color_3 = (select(16777215u, 8421504u, _e59) & debug_color);
        let _e66 = color_3;
        debug_line(position, (position + (vec3(ray_len) * direction)), _e66);
    }
    let _e68 = intersection.kind;
    if (_e68 == 0u) {
        let _e71 = evaluate_environment(direction);
        return _e71;
    }
    let _e73 = intersection.instance_custom_data;
    let _e75 = intersection.geometry_index;
    let _e78 = hit_entries[(_e73 + _e75)];
    entry_7 = _e78;
    let _e80 = entry_7;
    let _e82 = intersection.primitive_index;
    let _e83 = fetch_triangle_indices(_e80, _e82);
    indices_1 = _e83;
    let _e86 = intersection.barycentrics;
    let _e87 = make_barycentrics(_e86);
    barycentrics = _e87;
    let _e90 = entry_7.vertex_buf;
    let _e94 = indices_1.x;
    let _e97 = vertex_buffers[_e90].data[_e94].tex_coords;
    let _e99 = entry_7.vertex_buf;
    let _e103 = indices_1.y;
    let _e106 = vertex_buffers[_e99].data[_e103].tex_coords;
    let _e108 = entry_7.vertex_buf;
    let _e112 = indices_1.z;
    let _e115 = vertex_buffers[_e108].data[_e112].tex_coords;
    let _e117 = barycentrics;
    tex_coords_3 = (mat3x2<f32>(_e97, _e106, _e115) * _e117);
    let _e120 = entry_7;
    let _e121 = tex_coords_3;
    let _e124 = sample_hit_emissive(_e120, _e121, 0f, 0u);
    return _e124;
}

fn get_prev_pixel(pixel_6: vec2<i32>, pos_world: vec3<f32>) -> vec2<f32> {
    var motion: vec2<f32>;

    let _e36 = parameters.use_motion_vectors;
    if (USE_MOTION_VECTORS && (_e36 != 0u)) {
        let _e41 = textureLoad(t_motion, pixel_6, 0i);
        motion = (_e41.xy / vec2(MOTION_SCALE));
        let _e51 = motion;
        return ((vec2<f32>(pixel_6) + vec2(0.5f)) + _e51);
    } else {
        let _e53 = prev_camera;
        let _e54 = get_projected_pixel_float(_e53, pos_world);
        return _e54;
    }
}

fn ratio(a_8: f32, b_7: f32) -> f32 {
    return select(0f, (a_8 / (a_8 + b_7)), ((a_8 + b_7) > 0f));
}

fn zero_target_score() -> TargetScore {
    let _e32 = zero_radiance();
    return TargetScore(_e32, 0f);
}

fn make_reservoir(ls_2: LightSample, light_index: u32, brdf_2: BrdfLobes, diffuse_albedo_1: vec3<f32>) -> LiveReservoir {
    var r_9: LiveReservoir;

    r_9 = LiveReservoir();
    let _e40 = reflect_light(brdf_2, ls_2.radiance);
    r_9.selected_radiance = _e40;
    r_9.selected_uv = ls_2.uv;
    r_9.selected_light_index = light_index;
    let _e46 = r_9.selected_radiance;
    let _e47 = compute_target_score(_e46, diffuse_albedo_1);
    r_9.selected_target_score = _e47;
    let _e51 = r_9.selected_target_score;
    r_9.weight_sum = select(0f, (_e51 / ls_2.pdf), (ls_2.pdf > 0f));
    r_9.history = 1f;
    let _e60 = r_9;
    return _e60;
}

fn make_target_score(radiance_2: Radiance, diffuse_albedo_2: vec3<f32>) -> TargetScore {
    let _e34 = compute_target_score(radiance_2, diffuse_albedo_2);
    return TargetScore(radiance_2, _e34);
}

fn pack_reservoir(r_10: LiveReservoir) -> StoredReservoir {
    let _e34 = pack_reservoir_detail(r_10, r_10.history);
    return _e34;
}

fn evaluate_surface_brdf(surface_5: Surface, dir_5: vec3<f32>) -> BrdfLobes {
    let _e34 = surface_material(surface_5);
    let _e35 = surface_normal(surface_5);
    let _e37 = evaluate_brdf(_e34, _e35, surface_5.view_dir, dir_5);
    return _e37;
}

fn sample_incoming_light(surface_6: Surface, from_light: bool, rng_10: ptr<function, RandomState>) -> LightSample {
    var importance_2: bool;
    var mat_8: Material;
    var normal_7: vec3<f32>;
    var ls_3: LightSample;
    var bs: BsdfSample;
    var dir_6: vec3<f32>;
    var num_light: f32;
    var num_brdf: f32;

    let _e36 = parameters.environment_importance_sampling;
    importance_2 = (_e36 != 0u);
    let _e40 = surface_material(surface_6);
    mat_8 = _e40;
    let _e42 = surface_normal(surface_6);
    normal_7 = _e42;
    ls_3 = LightSample();
    if from_light {
        let _e46 = importance_2;
        let _e47 = sample_light(_e46, rng_10);
        ls_3 = _e47;
    } else {
        let _e48 = mat_8;
        let _e49 = normal_7;
        let _e51 = sample_bsdf(_e48, _e49, surface_6.view_dir, rng_10);
        bs = _e51;
        let _e55 = bs.dir;
        let _e56 = map_equirect_dir_to_uv(_e55);
        ls_3.uv = _e56;
        let _e59 = bs.dir;
        let _e60 = evaluate_environment(_e59);
        ls_3.radiance = _e60;
    }
    let _e62 = ls_3.uv;
    let _e63 = map_equirect_uv_to_dir(_e62);
    dir_6 = _e63;
    let _e66 = parameters.num_environment_samples;
    num_light = f32(_e66);
    let _e70 = parameters.num_brdf_samples;
    num_brdf = f32(_e70);
    let _e74 = num_light;
    let _e76 = ls_3.uv;
    let _e77 = importance_2;
    let _e78 = compute_light_pdf(_e76, _e77);
    let _e80 = num_brdf;
    let _e81 = mat_8;
    let _e82 = normal_7;
    let _e84 = dir_6;
    let _e85 = compute_bsdf_pdf(_e81, _e82, surface_6.view_dir, _e84);
    let _e88 = num_light;
    let _e89 = num_brdf;
    ls_3.pdf = (((_e74 * _e78) + (_e80 * _e85)) / max((_e88 + _e89), 1f));
    let _e94 = ls_3;
    return _e94;
}

fn estimate_target_score_with_occlusion(surface_7: Surface, position_1: vec3<f32>, light_index_1: u32, light_uv: vec2<f32>, acs_1: acceleration_structure, ray_len_1: f32, debug_color_1: u32) -> TargetScore {
    var direction_1: vec3<f32>;
    var brdf_3: BrdfLobes;
    var radiance_3: vec3<f32>;

    if (light_index_1 != 0u) {
        let _e41 = zero_target_score();
        return _e41;
    }
    let _e42 = map_equirect_uv_to_dir(light_uv);
    direction_1 = _e42;
    let _e44 = direction_1;
    if (dot(_e44, surface_7.flat_normal) <= 0f) {
        let _e49 = zero_target_score();
        return _e49;
    }
    let _e50 = direction_1;
    let _e51 = evaluate_surface_brdf(surface_7, _e50);
    brdf_3 = _e51;
    let _e53 = brdf_3;
    let _e54 = is_brdf_black(_e53);
    if _e54 {
        let _e55 = zero_target_score();
        return _e55;
    }
    let _e56 = direction_1;
    let _e57 = evaluate_incoming_radiance(acs_1, position_1, _e56, ray_len_1, debug_color_1);
    radiance_3 = _e57;
    let _e59 = brdf_3;
    let _e60 = radiance_3;
    let _e61 = reflect_light(_e59, _e60);
    let _e63 = make_target_score(_e61, surface_7.diffuse_albedo);
    return _e63;
}

fn evaluate_sample(ls_4: ptr<function, LightSample>, surface_8: Surface, start_pos: vec3<f32>, ray_len_2: f32, debug_color_2: u32) -> BrdfLobes {
    var dir_7: vec3<f32>;
    var brdf_4: BrdfLobes;

    let _e38 = (*ls_4).uv;
    let _e39 = map_equirect_uv_to_dir(_e38);
    dir_7 = _e39;
    let _e41 = dir_7;
    if (dot(_e41, surface_8.flat_normal) <= 0f) {
        let _e46 = zero_brdf();
        return _e46;
    }
    let _e47 = dir_7;
    let _e48 = evaluate_surface_brdf(surface_8, _e47);
    brdf_4 = _e48;
    let _e50 = brdf_4;
    let _e51 = is_brdf_black(_e50);
    if _e51 {
        let _e52 = zero_brdf();
        return _e52;
    }
    let _e54 = dir_7;
    let _e55 = evaluate_incoming_radiance(acc_struct, start_pos, _e54, ray_len_2, debug_color_2);
    (*ls_4).radiance = _e55;
    let _e57 = (*ls_4).radiance;
    if all((_e57 <= vec3(0f))) {
        let _e62 = zero_brdf();
        return _e62;
    }
    let _e63 = brdf_4;
    return _e63;
}

fn compute_restir(surface_9: Surface, pixel_7: vec2<i32>, rng_11: ptr<function, RandomState>, enable_debug_1: bool) -> RestirOutput {
    var ray_dir: vec3<f32>;
    var pixel_index_1: i32;
    var env: vec3<f32>;
    var position_2: vec3<f32>;
    var ray_len_3: f32;
    var canonical: LiveReservoir;
    var num_initial: u32;
    var i: u32;
    var ls_5: LightSample;
    var brdf_5: BrdfLobes;
    var other_1: LiveReservoir;
    var center_coord: vec2<f32>;
    var accepted_reservoir_indices: array<i32, 4>;
    var accepted_count: u32;
    var max_samples: u32;
    var num_candidates: u32;
    var tap: u32;
    var radius: f32;
    var offset: vec2<f32>;
    var other_pixel: vec2<i32>;
    var other_index: i32;
    var other_surface: Surface;
    var compatibility: f32;
    var color_4: vec4<f32>;
    var i_1: u32;
    var reservoir: LiveReservoir;
    var shaded: Radiance;
    var shaded_weight: f32;
    var mis_scale: f32;
    var mis_canonical: f32;
    var inv_count: f32;
    var rid: u32;
    var neighbor_index: i32;
    var neighbor: StoredReservoir;
    var neighbor_pixel: vec2<i32>;
    var offset_1: vec2<f32>;
    var max_confidence_1: f32;
    var other_2: LiveReservoir;
    var neighbor_history: f32;
    var neighbor_surface: Surface;
    var neighbor_dir: vec3<f32>;
    var neighbor_position: vec3<f32>;
    var t_canonical_at_neighbor: TargetScore;
    var r_canonical: f32;
    var t_neighbor_at_canonical: TargetScore;
    var r_neighbor: f32;
    var mis_neighbor: f32;
    var evaluated: TargetScore;
    var history_3: f32;
    var scale: f32;
    var cw: f32;
    var scale_1: f32;
    var effective_history: f32;
    var stored: StoredReservoir;
    var ro_1: RestirOutput;
    var denom_2: f32;
    var cw_1: f32;

    let _e36 = camera;
    let _e37 = get_ray_direction(_e36, pixel_7);
    ray_dir = _e37;
    let _e39 = camera;
    let _e40 = get_reservoir_index(pixel_7, _e39);
    pixel_index_1 = _e40;
    if (surface_9.depth == 0f) {
        let _e45 = pixel_index_1;
        reservoirs[u32(_e45)] = StoredReservoir();
        let _e49 = ray_dir;
        let _e50 = evaluate_environment_background(_e49);
        env = _e50;
        let _e52 = env;
        return RestirOutput(Radiance(_e52, vec3(0f)));
    }
    let _e59 = debug.view_mode;
    if (WRITE_DEBUG_IMAGE && (_e59 == DebugMode_Depth)) {
        textureStore(out_debug, pixel_7, vec4((1f / surface_9.depth)));
    }
    let _e68 = camera.position;
    let _e70 = ray_dir;
    position_2 = (_e68 + (vec3(surface_9.depth) * _e70));
    ray_len_3 = select(0f, (surface_9.depth * 0.2f), enable_debug_1);
    canonical = LiveReservoir();
    let _e84 = parameters.num_environment_samples;
    let _e86 = parameters.num_brdf_samples;
    num_initial = (_e84 + _e86);
    let _e90 = num_initial;
    i = 0u;
    loop {
        let _e92 = i;
        if (_e92 < _e90) {
        } else {
            break;
        }
        let _e94 = i;
        let _e96 = parameters.num_environment_samples;
        let _e98 = sample_incoming_light(surface_9, (_e94 < _e96), rng_11);
        ls_5 = _e98;
        let _e100 = position_2;
        let _e101 = ray_len_3;
        let _e103 = evaluate_sample((&ls_5), surface_9, _e100, _e101, 65280u);
        brdf_5 = _e103;
        let _e105 = brdf_5;
        let _e106 = is_brdf_black(_e105);
        if _e106 {
            bump_reservoir((&canonical), 1f);
        } else {
            let _e108 = ls_5;
            let _e110 = brdf_5;
            let _e112 = make_reservoir(_e108, 0u, _e110, surface_9.diffuse_albedo);
            other_1 = _e112;
            let _e114 = other_1;
            let _e115 = random_gen(rng_11);
            let _e116 = merge_reservoir((&canonical), _e114, _e115);
        }
        continuing {
            let _e117 = i;
            i = (_e117 + 1u);
        }
    }
    let _e120 = position_2;
    let _e121 = get_prev_pixel(pixel_7, _e120);
    center_coord = _e121;
    accepted_reservoir_indices = array<i32, 4>(0i, 0i, 0i, 0i);
    accepted_count = 0u;
    let _e133 = parameters.tap_count;
    max_samples = min(MAX_RESERVOIRS, _e133);
    let _e136 = max_samples;
    num_candidates = (_e136 * FACTOR_CANDIDATES);
    let _e141 = num_candidates;
    tap = 0u;
    loop {
        let _e143 = tap;
        if (_e143 < _e141) {
        } else {
            break;
        }
        let _e145 = accepted_count;
        let _e146 = max_samples;
        if (_e145 >= _e146) {
            break;
        }
        let _e149 = parameters.tap_radius;
        let _e150 = random_gen(rng_11);
        radius = (_e149 * _e150);
        let _e153 = radius;
        let _e154 = random_gen(rng_11);
        let _e155 = sample_circle_uniform(_e154);
        offset = (vec2(_e153) * _e155);
        let _e159 = center_coord;
        let _e160 = offset;
        other_pixel = vec2<i32>((_e159 + _e160));
        let _e164 = other_pixel;
        let _e165 = prev_camera;
        let _e166 = get_reservoir_index(_e164, _e165);
        other_index = _e166;
        let _e168 = other_index;
        if (_e168 < 0i) {
            continue;
        }
        let _e171 = other_index;
        let _e175 = prev_reservoirs[u32(_e171)].confidence;
        if (_e175 == 0f) {
            continue;
        }
        let _e178 = other_pixel;
        let _e179 = read_prev_surface(_e178);
        other_surface = _e179;
        let _e181 = other_surface;
        let _e182 = compare_surfaces(surface_9, _e181);
        compatibility = _e182;
        let _e184 = compatibility;
        if (_e184 < 0.1f) {
            continue;
        }
        let _e187 = accepted_count;
        let _e189 = other_index;
        accepted_reservoir_indices[_e187] = _e189;
        let _e191 = accepted_count;
        accepted_count = (_e191 + 1u);
        continuing {
            let _e193 = tap;
            tap = (_e193 + 1u);
        }
    }
    let _e198 = debug.view_mode;
    if (WRITE_DEBUG_IMAGE && (_e198 == DebugMode_SampleReuse)) {
        color_4 = vec4(0f);
        let _e207 = accepted_count;
        i_1 = 0u;
        loop {
            let _e210 = i_1;
            if (_e210 < min(3u, _e207)) {
            } else {
                break;
            }
            let _e212 = i_1;
            color_4[_e212] = 1f;
            continuing {
                let _e215 = i_1;
                i_1 = (_e215 + 1u);
            }
        }
        let _e218 = color_4;
        textureStore(out_debug, pixel_7, _e218);
    }
    reservoir = LiveReservoir();
    let _e221 = zero_radiance();
    shaded = _e221;
    shaded_weight = 0f;
    let _e226 = accepted_count;
    let _e229 = parameters.defensive_mis;
    mis_scale = (1f / (f32(_e226) + _e229));
    let _e233 = mis_scale;
    let _e235 = parameters.defensive_mis;
    let _e238 = accepted_count;
    let _e242 = parameters.use_pairwise_mis;
    mis_canonical = select((_e233 * _e235), 1f, ((_e238 == 0u) || (_e242 == 0u)));
    let _e249 = accepted_count;
    inv_count = (1f / f32(_e249));
    let _e254 = accepted_count;
    rid = 0u;
    loop {
        let _e256 = rid;
        if (_e256 < _e254) {
        } else {
            break;
        }
        let _e258 = rid;
        let _e260 = accepted_reservoir_indices[_e258];
        neighbor_index = _e260;
        let _e262 = neighbor_index;
        let _e265 = prev_reservoirs[u32(_e262)];
        neighbor = _e265;
        let _e267 = neighbor_index;
        let _e268 = prev_camera;
        let _e269 = get_pixel_from_reservoir_index(_e267, _e268);
        neighbor_pixel = _e269;
        let _e271 = neighbor_pixel;
        let _e273 = center_coord;
        offset_1 = (vec2<f32>(_e271) - _e273);
        let _e277 = parameters.tap_confidence_near;
        let _e279 = parameters.tap_confidence_far;
        let _e280 = offset_1;
        let _e283 = parameters.tap_radius;
        max_confidence_1 = mix(_e277, _e279, (length(_e280) / _e283));
        other_2 = LiveReservoir();
        let _e290 = parameters.use_pairwise_mis;
        if (_e290 != 0u) {
            let _e294 = neighbor.confidence;
            let _e295 = max_confidence_1;
            neighbor_history = min(_e294, _e295);
            let _e298 = neighbor_pixel;
            let _e299 = read_prev_surface(_e298);
            neighbor_surface = _e299;
            let _e301 = prev_camera;
            let _e302 = neighbor_pixel;
            let _e303 = get_ray_direction(_e301, _e302);
            neighbor_dir = _e303;
            let _e306 = prev_camera.position;
            let _e308 = neighbor_surface.depth;
            let _e309 = neighbor_dir;
            neighbor_position = (_e306 + (vec3(_e308) * _e309));
            let _e314 = neighbor_surface;
            let _e315 = neighbor_position;
            let _e317 = canonical.selected_light_index;
            let _e319 = canonical.selected_uv;
            let _e320 = ray_len_3;
            let _e322 = estimate_target_score_with_occlusion(_e314, _e315, _e317, _e319, prev_acc_struct, _e320, 16711680u);
            t_canonical_at_neighbor = _e322;
            let _e325 = canonical.history;
            let _e327 = canonical.selected_target_score;
            let _e329 = inv_count;
            let _e331 = neighbor_history;
            let _e333 = t_canonical_at_neighbor.score;
            let _e335 = ratio(((_e325 * _e327) * _e329), (_e331 * _e333));
            r_canonical = _e335;
            let _e337 = mis_scale;
            let _e338 = r_canonical;
            let _e340 = mis_canonical;
            mis_canonical = (_e340 + (_e337 * _e338));
            let _e342 = position_2;
            let _e344 = neighbor.light_index;
            let _e346 = neighbor.light_uv;
            let _e347 = ray_len_3;
            let _e349 = estimate_target_score_with_occlusion(surface_9, _e342, _e344, _e346, acc_struct, _e347, 255u);
            t_neighbor_at_canonical = _e349;
            let _e351 = neighbor_history;
            let _e353 = neighbor.target_score;
            let _e356 = canonical.history;
            let _e358 = t_neighbor_at_canonical.score;
            let _e360 = inv_count;
            let _e362 = ratio((_e351 * _e353), ((_e356 * _e358) * _e360));
            r_neighbor = _e362;
            let _e364 = mis_scale;
            let _e365 = r_neighbor;
            mis_neighbor = (_e364 * _e365);
            let _e369 = neighbor_history;
            other_2.history = _e369;
            let _e372 = neighbor.light_index;
            other_2.selected_light_index = _e372;
            let _e375 = neighbor.light_uv;
            other_2.selected_uv = _e375;
            let _e378 = t_neighbor_at_canonical.score;
            other_2.selected_target_score = _e378;
            let _e381 = t_neighbor_at_canonical.radiance;
            other_2.selected_radiance = _e381;
            let _e384 = t_neighbor_at_canonical.score;
            let _e386 = neighbor.contribution_weight;
            let _e388 = mis_neighbor;
            other_2.weight_sum = ((_e384 * _e386) * _e388);
        } else {
            let _e390 = position_2;
            let _e392 = neighbor.light_index;
            let _e394 = neighbor.light_uv;
            let _e395 = ray_len_3;
            let _e397 = estimate_target_score_with_occlusion(surface_9, _e390, _e392, _e394, acc_struct, _e395, 255u);
            evaluated = _e397;
            let _e400 = neighbor.confidence;
            let _e401 = max_confidence_1;
            history_3 = min(_e400, _e401);
            let _e406 = neighbor.light_index;
            other_2.selected_light_index = _e406;
            let _e409 = neighbor.light_uv;
            other_2.selected_uv = _e409;
            let _e412 = evaluated.score;
            other_2.selected_target_score = _e412;
            let _e415 = evaluated.radiance;
            other_2.selected_radiance = _e415;
            let _e418 = neighbor.contribution_weight;
            let _e420 = evaluated.score;
            let _e422 = history_3;
            other_2.weight_sum = ((_e418 * _e420) * _e422);
            let _e425 = history_3;
            other_2.history = _e425;
        }
        if DECOUPLED_SHADING {
            let _e428 = other_2.weight_sum;
            let _e430 = neighbor.contribution_weight;
            scale = (_e428 * _e430);
            let _e434 = scale;
            let _e437 = other_2.selected_radiance.diffuse;
            let _e440 = shaded.diffuse;
            shaded.diffuse = (_e440 + (vec3(_e434) * _e437));
            let _e443 = scale;
            let _e446 = other_2.selected_radiance.specular;
            let _e449 = shaded.specular;
            shaded.specular = (_e449 + (vec3(_e443) * _e446));
            let _e452 = other_2.weight_sum;
            let _e453 = shaded_weight;
            shaded_weight = (_e453 + _e452);
        }
        let _e456 = other_2.weight_sum;
        if (_e456 <= 0f) {
            let _e460 = other_2.history;
            bump_reservoir((&reservoir), _e460);
        } else {
            let _e461 = other_2;
            let _e462 = random_gen(rng_11);
            let _e463 = merge_reservoir((&reservoir), _e461, _e462);
        }
        continuing {
            let _e464 = rid;
            rid = (_e464 + 1u);
        }
    }
    let _e468 = parameters.use_pairwise_mis;
    if (_e468 != 0u) {
        let _e471 = mis_canonical;
        normalize_reservoir((&canonical), _e471);
    }
    if DECOUPLED_SHADING {
        let _e474 = canonical.weight_sum;
        let _e476 = canonical.selected_target_score;
        cw = (_e474 / max(_e476, 0.1f));
        let _e482 = canonical.weight_sum;
        let _e483 = cw;
        scale_1 = (_e482 * _e483);
        let _e487 = scale_1;
        let _e490 = canonical.selected_radiance.diffuse;
        let _e493 = shaded.diffuse;
        shaded.diffuse = (_e493 + (vec3(_e487) * _e490));
        let _e496 = scale_1;
        let _e499 = canonical.selected_radiance.specular;
        let _e502 = shaded.specular;
        shaded.specular = (_e502 + (vec3(_e496) * _e499));
        let _e505 = canonical.weight_sum;
        let _e506 = shaded_weight;
        shaded_weight = (_e506 + _e505);
    }
    let _e508 = canonical;
    let _e509 = random_gen(rng_11);
    let _e510 = merge_reservoir((&reservoir), _e508, _e509);
    let _e512 = reservoir.history;
    let _e515 = parameters.use_pairwise_mis;
    effective_history = select(_e512, 1f, (_e515 != 0u));
    let _e520 = reservoir;
    let _e521 = effective_history;
    let _e522 = pack_reservoir_detail(_e520, _e521);
    stored = _e522;
    let _e524 = pixel_index_1;
    let _e527 = stored;
    reservoirs[u32(_e524)] = _e527;
    ro_1 = RestirOutput();
    if DECOUPLED_SHADING {
        let _e531 = shaded_weight;
        denom_2 = max(_e531, 0.001f);
        let _e537 = shaded.diffuse;
        let _e538 = denom_2;
        let _e542 = shaded.specular;
        let _e543 = denom_2;
        ro_1.radiance = Radiance((_e537 / vec3(_e538)), (_e542 / vec3(_e543)));
    } else {
        let _e548 = stored.contribution_weight;
        cw_1 = _e548;
        let _e551 = cw_1;
        let _e554 = reservoir.selected_radiance.diffuse;
        let _e557 = cw_1;
        let _e560 = reservoir.selected_radiance.specular;
        ro_1.radiance = Radiance((vec3(_e551) * _e554), (vec3(_e557) * _e560));
    }
    let _e564 = ro_1;
    return _e564;
}

@compute @workgroup_size(8, 4, 1) 
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var global_index: u32;
    var rng: RandomState;
    var surface: Surface;
    var enable_debug: bool;
    var enable_restir_debug: bool;
    var ro: RestirOutput;
    var color: vec3<f32>;

    let _e35 = camera.target_size;
    if any((global_id.xy >= _e35)) {
        return;
    }
    let _e41 = camera.target_size.x;
    global_index = ((global_id.y * _e41) + global_id.x);
    let _e46 = global_index;
    let _e48 = parameters.frame_index;
    let _e49 = random_init(_e46, _e48);
    rng = _e49;
    let _e53 = read_surface(vec2<i32>(global_id.xy));
    surface = _e53;
    let _e58 = debug.mouse_pos;
    enable_debug = (DEBUG_MODE && all((global_id.xy == _e58)));
    let _e64 = debug.draw_flags;
    let _e69 = enable_debug;
    enable_restir_debug = (((_e64 & DebugDrawFlags_RESTIR) != 0u) && _e69);
    let _e72 = surface;
    let _e75 = enable_restir_debug;
    let _e76 = compute_restir(_e72, vec2<i32>(global_id.xy), (&rng), _e75);
    ro = _e76;
    let _e78 = enable_debug;
    if _e78 {
        let _e80 = surface.diffuse_albedo;
        let _e83 = ro.radiance.diffuse;
        let _e87 = ro.radiance.specular;
        color = ((_e80 * _e83) + _e87);
        let _e92 = color;
        let _e93 = debug_buf.variance.color_sum;
        debug_buf.variance.color_sum = (_e93 + _e92);
        let _e97 = color;
        let _e98 = color;
        let _e100 = debug_buf.variance.color2_sum;
        debug_buf.variance.color2_sum = (_e100 + (_e97 * _e98));
        let _e105 = debug_buf.variance.count;
        debug_buf.variance.count = (_e105 + 1u);
    }
    let _e110 = ro.radiance.diffuse;
    textureStore(out_diffuse, global_id.xy, vec4<f32>(_e110, 1f));
    let _e116 = ro.radiance.specular;
    textureStore(out_specular, global_id.xy, vec4<f32>(_e116, 1f));
}
