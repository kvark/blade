struct Vertex {
    position: vec3<f32>,
    bitangent_sign: f32,
    tex_coords: vec2<f32>,
    normal: u32,
    tangent: u32,
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

struct SkinningParams {
    post_transform: mat3x4<f32>,
    joint_matrices: array<mat3x4<f32>, 64>,
}

struct SkinVertex {
    joints: u32,
    weights: u32,
}

struct LocalLight {
    position_range: vec4<f32>,
    intensity: vec4<f32>,
    direction: vec4<f32>,
    spot: vec4<f32>,
}

struct LocalLightParams {
    count_seed: vec4<f32>,
    lights: array<LocalLight, 8>,
}

struct RasterFrameParams {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    ambient_color: vec4<f32>,
    settings: vec4<f32>,
    shadow_params: vec4<f32>,
}

struct RasterDrawParams {
    model: mat4x4<f32>,
    normal_quat: vec4<f32>,
    base_color_factor: vec4<f32>,
    emissive_factor: vec4<f32>,
    material: vec4<f32>,
}

struct ShadowFrameParams {
    light_view_proj: mat4x4<f32>,
}

struct ShadowDrawParams {
    model: mat4x4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) uv: vec2<f32>,
}

struct SkyOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
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
const PI: f32 = 3.1415925f;
const DIELECTRIC_F0: f32 = 0.04f;
const MIN_ROUGHNESS: f32 = 0.05f;
const LUMINOCITY_WEIGHTS: vec3<f32> = vec3<f32>(0.3f, 0.4f, 0.3f);

var<uniform> skinning_params: SkinningParams;
var<uniform> frame_params: RasterFrameParams;
var<uniform> light_params: LocalLightParams;
var<uniform> draw_params: RasterDrawParams;
var samp: sampler;
var base_color_tex: texture_2d<f32>;
var normal_tex: texture_2d<f32>;
var metallic_roughness_tex: texture_2d<f32>;
var emissive_tex: texture_2d<f32>;
var shadow_samp: sampler_comparison;
var shadow_tex: texture_depth_2d;
var<uniform> shadow_frame_params: ShadowFrameParams;
var<uniform> shadow_draw_params: ShadowDrawParams;
var<uniform> sky_params: RasterFrameParams;
var env_map: texture_2d<f32>;

fn decode_normal(raw: u32) -> vec3<f32> {
    return unpack4x8snorm(raw).xyz;
}

fn tangent_basis(n_1: vec3<f32>, transformed_tangent: vec3<f32>, bitangent_sign: f32, linear_sign: f32) -> mat3x3<f32> {
    var t_1: vec3<f32>;
    var b_1: vec3<f32>;

    t_1 = normalize((transformed_tangent - (n_1 * vec3(dot(n_1, transformed_tangent)))));
    let _e10 = t_1;
    b_1 = ((normalize(cross(n_1, _e10)) * vec3(bitangent_sign)) * vec3(linear_sign));
    let _e18 = t_1;
    let _e19 = b_1;
    return mat3x3<f32>(_e18, _e19, n_1);
}

fn compute_luminocity(color_2: vec3<f32>) -> f32 {
    return dot(color_2, LUMINOCITY_WEIGHTS);
}

fn material_from_metallic_roughness(base_color_1: vec3<f32>, metalness: f32, roughness: f32) -> Material {
    var mat_1: Material;

    mat_1 = Material();
    mat_1.diffuse_albedo = (base_color_1 * vec3((1f - metalness)));
    mat_1.specular_f0 = mix(vec3(DIELECTRIC_F0), base_color_1, metalness);
    mat_1.roughness = roughness;
    let _e15 = mat_1;
    return _e15;
}

fn material_alpha(mat_2: Material) -> f32 {
    var r: f32;

    r = clamp(mat_2.roughness, MIN_ROUGHNESS, 1f);
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

fn evaluate_ambient(mat_3: Material) -> vec3<f32> {
    return ((mat_3.diffuse_albedo * (vec3(1f) - mat_3.specular_f0)) + mat_3.specular_f0);
}

fn specular_sampling_ratio(mat_4: Material) -> f32 {
    var diffuse: f32;
    var specular: f32;

    let _e2 = compute_luminocity(mat_4.diffuse_albedo);
    diffuse = _e2;
    let _e5 = compute_luminocity(mat_4.specular_f0);
    specular = _e5;
    let _e7 = specular;
    let _e8 = diffuse;
    let _e9 = specular;
    return clamp((_e7 / max((_e8 + _e9), 0.00001f)), 0.1f, 0.9f);
}

fn evaluate_brdf(mat_5: Material, normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> BrdfLobes {
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
    let _e29 = material_alpha(mat_5);
    alpha_2 = _e29;
    let _e31 = v_dot_h;
    let _e33 = fresnel_schlick(_e31, mat_5.specular_f0);
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

fn encode_srgb(linear_1: vec3<f32>) -> vec3<f32> {
    var low: vec3<f32>;
    var high: vec3<f32>;

    low = (vec3(12.92f) * linear_1);
    high = ((vec3(1.055f) * pow(max(linear_1, vec3(0f)), vec3((1f / 2.4f)))) - vec3(0.055f));
    let _e20 = high;
    let _e21 = low;
    return select(_e20, _e21, (linear_1 <= vec3(0.0031308f)));
}

fn encode_surface_color(color_3: vec3<f32>, needs_encoding: bool) -> vec3<f32> {
    let _e2 = encode_srgb(color_3);
    return select(color_3, _e2, needs_encoding);
}

fn unpack_joints(raw_1: u32) -> vec4<u32> {
    return ((vec4(raw_1) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4(255u));
}

fn apply_affine(m: mat3x4<f32>, p: vec3<f32>) -> vec3<f32> {
    var h_1: vec4<f32>;

    h_1 = vec4<f32>(p, 1f);
    let _e6 = h_1;
    return (_e6 * m);
}

fn skin_linear(skin_1: mat3x4<f32>) -> mat3x3<f32> {
    return transpose(mat3x3<f32>(skin_1[0].xyz, skin_1[1].xyz, skin_1[2].xyz));
}

fn skin_blend(skin_2: SkinVertex) -> mat3x4<f32> {
    var joints: vec4<u32>;
    var weights: vec4<f32>;

    let _e3 = unpack_joints(skin_2.joints);
    joints = _e3;
    weights = unpack4x8unorm(skin_2.weights);
    let _e10 = joints.x;
    let _e12 = skinning_params.joint_matrices[_e10];
    let _e14 = weights.x;
    let _e18 = joints.y;
    let _e20 = skinning_params.joint_matrices[_e18];
    let _e22 = weights.y;
    let _e27 = joints.z;
    let _e29 = skinning_params.joint_matrices[_e27];
    let _e31 = weights.z;
    let _e36 = joints.w;
    let _e38 = skinning_params.joint_matrices[_e36];
    let _e40 = weights.w;
    return ((((_e12 * _e14) + (_e20 * _e22)) + (_e29 * _e31)) + (_e38 * _e40));
}

fn quat_rotate(q: vec4<f32>, v_2: vec3<f32>) -> vec3<f32> {
    return (v_2 + (vec3(2f) * cross(q.xyz, (cross(q.xyz, v_2) + (vec3(q.w) * v_2)))));
}

fn map_equirect_dir_to_uv(dir_1: vec3<f32>) -> vec2<f32> {
    var yaw: f32;
    var pitch: f32;

    yaw = atan2(dir_1.x, dir_1.z);
    pitch = asin(clamp(dir_1.y, -(1f), 1f));
    let _e27 = yaw;
    let _e34 = pitch;
    return vec2<f32>((((_e27 / PI) + 1f) * 0.5f), ((_e34 / PI) + 0.5f));
}

fn directional_shadow(world_pos_1: vec3<f32>, n_2: vec3<f32>) -> f32 {
    var light_dir_1: vec3<f32>;
    var ndotl: f32;
    var normal_bias: f32;
    var depth_bias: f32;
    var receiver: vec3<f32>;
    var clip: vec4<f32>;
    var ndc_1: vec3<f32>;
    var uv_2: vec2<f32>;
    var texel: f32;
    var reference: f32;
    var visibility_1: f32;

    let _e19 = frame_params.shadow_params.x;
    if (_e19 < 0.5f) {
        return 1f;
    }
    let _e24 = frame_params.light_dir;
    light_dir_1 = normalize(_e24.xyz);
    let _e28 = light_dir_1;
    ndotl = max(dot(n_2, _e28), 0f);
    let _e35 = frame_params.shadow_params.z;
    let _e39 = ndotl;
    normal_bias = (_e35 * (1f + (0.75f * (1f - _e39))));
    let _e47 = frame_params.shadow_params.w;
    depth_bias = _e47;
    let _e49 = normal_bias;
    let _e53 = light_dir_1;
    let _e54 = depth_bias;
    receiver = ((world_pos_1 + (n_2 * vec3(_e49))) + (_e53 * vec3(_e54)));
    let _e60 = frame_params.light_view_proj;
    let _e61 = receiver;
    clip = (_e60 * vec4<f32>(_e61, 1f));
    let _e66 = clip;
    let _e69 = clip.w;
    ndc_1 = (_e66.xyz / vec3(_e69));
    let _e74 = ndc_1.x;
    let _e81 = ndc_1.y;
    uv_2 = vec2<f32>(((_e74 * 0.5f) + 0.5f), (0.5f - (_e81 * 0.5f)));
    let _e88 = ndc_1.z;
    let _e92 = ndc_1.z;
    let _e96 = uv_2;
    let _e102 = uv_2;
    if ((((_e88 <= 0f) || (_e92 >= 1f)) || any((_e96 < vec2(0f)))) || any((_e102 > vec2(1f)))) {
        return 1f;
    }
    let _e110 = textureDimensions(shadow_tex);
    texel = (1f / f32(_e110.x));
    let _e116 = ndc_1.z;
    reference = _e116;
    visibility_1 = 0f;
    let _e120 = uv_2;
    let _e126 = texel;
    let _e130 = reference;
    let _e131 = textureSampleCompare(shadow_tex, shadow_samp, (_e120 + (vec2<f32>(-(0.75f), -(0.75f)) * vec2(_e126))), _e130);
    let _e132 = visibility_1;
    visibility_1 = (_e132 + _e131);
    let _e134 = uv_2;
    let _e139 = texel;
    let _e143 = reference;
    let _e144 = textureSampleCompare(shadow_tex, shadow_samp, (_e134 + (vec2<f32>(0.75f, -(0.75f)) * vec2(_e139))), _e143);
    let _e145 = visibility_1;
    visibility_1 = (_e145 + _e144);
    let _e147 = uv_2;
    let _e152 = texel;
    let _e156 = reference;
    let _e157 = textureSampleCompare(shadow_tex, shadow_samp, (_e147 + (vec2<f32>(-(0.75f), 0.75f) * vec2(_e152))), _e156);
    let _e158 = visibility_1;
    visibility_1 = (_e158 + _e157);
    let _e160 = uv_2;
    let _e164 = texel;
    let _e168 = reference;
    let _e169 = textureSampleCompare(shadow_tex, shadow_samp, (_e160 + (vec2<f32>(0.75f, 0.75f) * vec2(_e164))), _e168);
    let _e170 = visibility_1;
    visibility_1 = (_e170 + _e169);
    let _e173 = visibility_1;
    visibility_1 = (_e173 * 0.25f);
    let _e176 = visibility_1;
    let _e179 = frame_params.shadow_params.y;
    return mix(1f, _e176, _e179);
}

fn hash31(p_1: vec3<f32>) -> f32 {
    return fract((sin(dot(p_1, vec3<f32>(127.1f, 311.7f, 74.7f))) * 43758.547f));
}

fn angular_attenuation(light_1: LocalLight, direction_to_light: vec3<f32>) -> f32 {
    var cosine: f32;
    var width: f32;
    var blend: f32;

    if (light_1.spot.w < 0.5f) {
        return 1f;
    }
    cosine = dot(light_1.direction.xyz, -(direction_to_light));
    width = (light_1.spot.x - light_1.spot.y);
    let _e33 = width;
    if (_e33 <= 0.00001f) {
        let _e38 = cosine;
        return select(0f, 1f, (_e38 >= light_1.spot.y));
    }
    let _e43 = cosine;
    let _e47 = width;
    blend = clamp(((_e43 - light_1.spot.y) / _e47), 0f, 1f);
    let _e53 = blend;
    return pow(_e53, light_1.spot.z);
}

fn raster_vertex(input_6: Vertex, position: vec3<f32>, normal_1: vec3<f32>, tangent: vec3<f32>, bitangent_sign_1: f32) -> VertexOutput {
    var out_1: VertexOutput;
    var pos_world: vec4<f32>;
    var n_3: vec3<f32>;
    var t_2: vec3<f32>;
    var b_2: vec3<f32>;

    out_1 = VertexOutput();
    let _e23 = draw_params.model;
    pos_world = (_e23 * vec4<f32>(position, 1f));
    let _e30 = frame_params.view_proj;
    let _e31 = pos_world;
    out_1.clip_pos = (_e30 * _e31);
    let _e34 = pos_world;
    out_1.world_pos = _e34.xyz;
    let _e40 = light_params.count_seed.x;
    let _e43 = out_1.world_pos.x;
    out_1.world_pos.x = (_e43 + (_e40 * 0f));
    let _e46 = draw_params.normal_quat;
    let _e47 = quat_rotate(_e46, normal_1);
    n_3 = normalize(_e47);
    let _e51 = draw_params.normal_quat;
    let _e52 = quat_rotate(_e51, tangent);
    t_2 = normalize(_e52);
    let _e55 = n_3;
    let _e56 = t_2;
    b_2 = (normalize(cross(_e55, _e56)) * vec3(bitangent_sign_1));
    let _e63 = n_3;
    out_1.normal = _e63;
    let _e65 = t_2;
    out_1.tangent = _e65;
    let _e67 = b_2;
    out_1.bitangent = _e67;
    out_1.uv = input_6.tex_coords;
    let _e70 = out_1;
    return _e70;
}

fn local_light_score(light_2: LocalLight, world_pos_2: vec3<f32>, n_4: vec3<f32>) -> f32 {
    var delta: vec3<f32>;
    var dist2: f32;
    var dist: f32;
    var range: f32;
    var falloff_1: f32;
    var ldir: vec3<f32>;
    var ndotl_1: f32;
    var intensity: f32;

    delta = (light_2.position_range.xyz - world_pos_2);
    let _e22 = delta;
    let _e23 = delta;
    dist2 = max(dot(_e22, _e23), 0.04f);
    let _e28 = dist2;
    dist = sqrt(_e28);
    range = max(light_2.position_range.w, 0.01f);
    let _e37 = dist;
    let _e38 = range;
    falloff_1 = max((1f - (_e37 / _e38)), 0f);
    let _e44 = delta;
    let _e45 = dist;
    ldir = (_e44 / vec3(_e45));
    let _e49 = ldir;
    ndotl_1 = max(dot(n_4, _e49), 0f);
    intensity = max(light_2.intensity.x, max(light_2.intensity.y, light_2.intensity.z));
    let _e63 = intensity;
    let _e64 = ldir;
    let _e65 = angular_attenuation(light_2, _e64);
    let _e67 = falloff_1;
    let _e69 = falloff_1;
    let _e73 = ndotl_1;
    return ((((_e63 * _e65) * _e67) * _e69) * (0.2f + (0.8f * _e73)));
}

fn shade_local_light(mat_6: Material, n_5: vec3<f32>, v_3: vec3<f32>, world_pos_3: vec3<f32>) -> vec3<f32> {
    var count: u32;
    var chosen: u32;
    var chosen_score: f32;
    var weight_sum: f32;
    var i: u32;
    var score: f32;
    var u: f32;
    var light_3: LocalLight;
    var delta_1: vec3<f32>;
    var dist2_1: f32;
    var dist_1: f32;
    var range_1: f32;
    var falloff_2: f32;
    var ldir_1: vec3<f32>;
    var brdf_1: BrdfLobes;
    var atten: f32;
    var inverse_probability: f32;

    let _e21 = light_params.count_seed.x;
    count = min(u32(_e21), MAX_LOCAL_LIGHTS);
    let _e26 = count;
    if (_e26 == 0u) {
        return vec3(0f);
    }
    chosen = 0u;
    chosen_score = 0f;
    weight_sum = 0f;
    i = 0u;
    loop {
        let _e40 = i;
        if (_e40 < MAX_LOCAL_LIGHTS) {
        } else {
            break;
        }
        let _e42 = i;
        let _e43 = count;
        if (_e42 >= _e43) {
            break;
        }
        let _e46 = i;
        let _e48 = light_params.lights[_e46];
        let _e49 = local_light_score(_e48, world_pos_3, n_5);
        score = _e49;
        let _e51 = score;
        if (_e51 <= 0f) {
            continue;
        }
        let _e54 = score;
        let _e55 = weight_sum;
        weight_sum = (_e55 + _e54);
        let _e57 = i;
        let _e61 = light_params.count_seed.y;
        let _e62 = score;
        let _e65 = hash31((world_pos_3 + vec3<f32>(f32(_e57), _e61, _e62)));
        u = _e65;
        let _e67 = u;
        let _e68 = weight_sum;
        let _e70 = score;
        if ((_e67 * _e68) < _e70) {
            let _e72 = i;
            chosen = _e72;
            let _e73 = score;
            chosen_score = _e73;
        }
        continuing {
            let _e74 = i;
            i = (_e74 + 1u);
        }
    }
    let _e77 = weight_sum;
    if (_e77 <= 0f) {
        return vec3(0f);
    }
    let _e83 = chosen;
    let _e85 = light_params.lights[_e83];
    light_3 = _e85;
    let _e88 = light_3.position_range;
    delta_1 = (_e88.xyz - world_pos_3);
    let _e92 = delta_1;
    let _e93 = delta_1;
    dist2_1 = max(dot(_e92, _e93), 0.04f);
    let _e98 = dist2_1;
    dist_1 = sqrt(_e98);
    let _e103 = light_3.position_range.w;
    range_1 = max(_e103, 0.01f);
    let _e108 = dist_1;
    let _e109 = range_1;
    falloff_2 = max((1f - (_e108 / _e109)), 0f);
    let _e115 = delta_1;
    let _e116 = dist_1;
    ldir_1 = (_e115 / vec3(_e116));
    let _e120 = ldir_1;
    let _e121 = evaluate_brdf(mat_6, n_5, v_3, _e120);
    brdf_1 = _e121;
    let _e123 = light_3;
    let _e124 = ldir_1;
    let _e125 = angular_attenuation(_e123, _e124);
    let _e126 = falloff_2;
    let _e128 = falloff_2;
    let _e130 = dist2_1;
    atten = (((_e125 * _e126) * _e128) / _e130);
    let _e133 = weight_sum;
    let _e134 = chosen_score;
    inverse_probability = (_e133 / max(_e134, 0.000001f));
    let _e141 = brdf_1.diffuse;
    let _e145 = brdf_1.specular;
    let _e148 = light_3.intensity;
    let _e151 = atten;
    let _e154 = inverse_probability;
    return (((((mat_6.diffuse_albedo * vec3(_e141)) + _e145) * _e148.xyz) * vec3(_e151)) * vec3(_e154));
}

@vertex 
fn raster_shadow_vs(input: Vertex) -> @builtin(position) vec4<f32> {
    var world: vec4<f32>;

    let _e17 = shadow_draw_params.model;
    world = (_e17 * vec4<f32>(input.position, 1f));
    let _e24 = shadow_frame_params.light_view_proj;
    let _e25 = world;
    return (_e24 * _e25);
}

@vertex 
fn raster_shadow_skinned_vs(input_1: Vertex, skin_input: SkinVertex) -> @builtin(position) vec4<f32> {
    var skinned: vec3<f32>;
    var world_1: vec4<f32>;

    let _e17 = skin_blend(skin_input);
    let _e19 = apply_affine(_e17, input_1.position);
    skinned = _e19;
    let _e22 = shadow_draw_params.model;
    let _e23 = skinned;
    world_1 = (_e22 * vec4<f32>(_e23, 1f));
    let _e29 = shadow_frame_params.light_view_proj;
    let _e30 = world_1;
    return (_e29 * _e30);
}

@fragment 
fn raster_shadow_fs() {
}

@vertex 
fn raster_sky_vs(@builtin(vertex_index) vertex_id: u32) -> SkyOutput {
    var positions: array<vec2<f32>, 3>;
    var pos: vec2<f32>;
    var out: SkyOutput;

    positions = array<vec2<f32>, 3>(vec2<f32>(-(1f), -(1f)), vec2<f32>(3f, -(1f)), vec2<f32>(-(1f), 3f));
    let _e32 = positions[vertex_id];
    pos = _e32;
    out = SkyOutput();
    let _e37 = pos;
    out.clip_pos = vec4<f32>(vec3<f32>(_e37, 1f), 1f);
    let _e43 = pos;
    out.ndc = _e43;
    let _e44 = out;
    return _e44;
}

@fragment 
fn raster_sky_fs(input_2: SkyOutput) -> @location(0) vec4<f32> {
    var ndc: vec4<f32>;
    var world_2: vec4<f32>;
    var world_pos: vec3<f32>;
    var dir: vec3<f32>;
    var env_enabled: bool;
    var color: vec3<f32>;
    var uv: vec2<f32>;
    var space_mode: bool;
    var theta: f32;
    var v: f32;
    var uv_1: vec2<f32>;
    var cell: vec2<f32>;
    var local: vec2<f32>;
    var p3: vec3<f32>;
    var h: f32;
    var h2: f32;
    var h3: f32;
    var star_pos: vec2<f32>;
    var d: f32;
    var falloff: f32;
    var b: f32;
    var temp: f32;
    var tint: vec3<f32>;
    var uv2: vec2<f32>;
    var cell2: vec2<f32>;
    var local2: vec2<f32>;
    var q3: vec3<f32>;
    var g: f32;
    var g2: f32;
    var g3: f32;
    var star_pos2: vec2<f32>;
    var d2: f32;
    var falloff2: f32;
    var b2: f32;
    var tint2: vec3<f32>;
    var t: f32;
    var horizon: vec3<f32>;
    var zenith: vec3<f32>;
    var mapped: vec3<f32>;

    ndc = vec4<f32>(vec3<f32>(input_2.ndc, 0f), 1f);
    let _e23 = sky_params.inv_view_proj;
    let _e24 = ndc;
    world_2 = (_e23 * _e24);
    let _e27 = world_2;
    let _e30 = world_2.w;
    world_pos = (_e27.xyz / vec3(_e30));
    let _e34 = world_pos;
    let _e36 = sky_params.camera_pos;
    dir = normalize((_e34 - _e36.xyz));
    let _e43 = sky_params.settings.x;
    env_enabled = (_e43 > 0.5f);
    color = vec3(0f);
    let _e50 = env_enabled;
    if _e50 {
        let _e51 = dir;
        let _e52 = map_equirect_dir_to_uv(_e51);
        uv = _e52;
        let _e54 = uv;
        let _e56 = textureSampleLevel(env_map, samp, _e54, 0f);
        color = _e56.xyz;
    } else {
        let _e60 = sky_params.ambient_color.w;
        space_mode = (_e60 > 0.5f);
        let _e64 = space_mode;
        if _e64 {
            let _e66 = dir.z;
            let _e68 = dir.x;
            theta = (atan2(_e66, _e68) + 10f);
            let _e74 = dir.y;
            v = (_e74 + 10f);
            let _e78 = theta;
            let _e79 = v;
            uv_1 = (vec2<f32>(_e78, _e79) * vec2(50f));
            let _e85 = uv_1;
            cell = floor(_e85);
            let _e88 = uv_1;
            local = (fract(_e88) - vec2(0.5f));
            let _e95 = cell.x;
            let _e97 = cell.y;
            let _e99 = cell.x;
            p3 = fract((vec3<f32>(_e95, _e97, _e99) * vec3<f32>(0.1031f, 0.103f, 0.0973f)));
            let _e108 = p3;
            let _e109 = p3;
            let _e111 = p3.y;
            let _e115 = p3.z;
            let _e119 = p3.x;
            p3 = (_e108 + vec3(dot(_e109, vec3<f32>((_e111 + 33.33f), (_e115 + 33.33f), (_e119 + 33.33f)))));
            let _e127 = p3.x;
            let _e129 = p3.y;
            let _e132 = p3.z;
            h = fract(((_e127 + _e129) * _e132));
            let _e137 = p3.y;
            let _e139 = p3.z;
            let _e142 = p3.x;
            h2 = fract(((_e137 + _e139) * _e142));
            let _e147 = p3.z;
            let _e149 = p3.x;
            let _e152 = p3.y;
            h3 = fract(((_e147 + _e149) * _e152));
            let _e156 = h;
            let _e159 = h2;
            star_pos = (vec2<f32>((_e156 - 0.5f), (_e159 - 0.5f)) * vec2(0.8f));
            let _e167 = local;
            let _e168 = star_pos;
            d = length((_e167 - _e168));
            let _e173 = d;
            falloff = clamp((1f - (_e173 / 0.08f)), 0f, 1f);
            let _e181 = falloff;
            let _e182 = falloff;
            let _e187 = h3;
            b = (((_e181 * _e182) * 0.8f) * step(0.92f, _e187));
            let _e191 = h;
            temp = (_e191 * 3f);
            tint = vec3<f32>(0.4f, 0.55f, 1f);
            let _e200 = temp;
            if (_e200 > 2f) {
                tint = vec3<f32>(1f, 0.4f, 0.15f);
            } else {
                let _e207 = temp;
                if (_e207 > 1f) {
                    tint = vec3<f32>(1f, 0.9f, 0.7f);
                }
            }
            let _e214 = color;
            let _e215 = tint;
            let _e216 = b;
            color = (_e214 + (_e215 * vec3(_e216)));
            let _e220 = theta;
            let _e221 = v;
            uv2 = (vec2<f32>(_e220, _e221) * vec2(150f));
            let _e227 = uv2;
            cell2 = floor(_e227);
            let _e230 = uv2;
            local2 = (fract(_e230) - vec2(0.5f));
            let _e237 = cell2.x;
            let _e239 = cell2.y;
            let _e241 = cell2.x;
            q3 = fract((vec3<f32>(_e237, _e239, _e241) * vec3<f32>(0.1031f, 0.103f, 0.0973f)));
            let _e250 = q3;
            let _e251 = q3;
            let _e253 = q3.y;
            let _e257 = q3.z;
            let _e261 = q3.x;
            q3 = (_e250 + vec3(dot(_e251, vec3<f32>((_e253 + 33.33f), (_e257 + 33.33f), (_e261 + 33.33f)))));
            let _e269 = q3.x;
            let _e271 = q3.y;
            let _e274 = q3.z;
            g = fract(((_e269 + _e271) * _e274));
            let _e279 = q3.y;
            let _e281 = q3.z;
            let _e284 = q3.x;
            g2 = fract(((_e279 + _e281) * _e284));
            let _e289 = q3.z;
            let _e291 = q3.x;
            let _e294 = q3.y;
            g3 = fract(((_e289 + _e291) * _e294));
            let _e298 = g;
            let _e301 = g2;
            star_pos2 = (vec2<f32>((_e298 - 0.5f), (_e301 - 0.5f)) * vec2(0.8f));
            let _e309 = local2;
            let _e310 = star_pos2;
            d2 = length((_e309 - _e310));
            let _e315 = d2;
            falloff2 = clamp((1f - (_e315 / 0.06f)), 0f, 1f);
            let _e323 = falloff2;
            let _e324 = falloff2;
            let _e329 = g3;
            b2 = (((_e323 * _e324) * 0.3f) * step(0.94f, _e329));
            let _e341 = g;
            tint2 = mix(vec3<f32>(0.5f, 0.65f, 1f), vec3<f32>(1f, 0.7f, 0.5f), _e341);
            let _e344 = color;
            let _e345 = tint2;
            let _e346 = b2;
            color = (_e344 + (_e345 * vec3(_e346)));
        } else {
            let _e351 = dir.y;
            t = clamp(((_e351 * 0.5f) + 0.5f), 0f, 1f);
            horizon = vec3<f32>(0.6f, 0.7f, 0.9f);
            zenith = vec3<f32>(0.2f, 0.35f, 0.6f);
            let _e370 = horizon;
            let _e371 = zenith;
            let _e372 = t;
            color = mix(_e370, _e371, _e372);
        }
    }
    let _e374 = color;
    let _e375 = color;
    mapped = (_e374 / (_e375 + vec3(1f)));
    let _e381 = mapped;
    let _e384 = sky_params.settings.y;
    let _e387 = encode_surface_color(_e381, (_e384 > 0.5f));
    return vec4<f32>(_e387, 1f);
}

@vertex 
fn raster_vs(input_3: Vertex) -> VertexOutput {
    let _e18 = decode_normal(input_3.normal);
    let _e20 = decode_normal(input_3.tangent);
    let _e22 = raster_vertex(input_3, input_3.position, _e18, _e20, input_3.bitangent_sign);
    return _e22;
}

@vertex 
fn raster_skinned_vs(input_4: Vertex, skin_input_1: SkinVertex) -> VertexOutput {
    var skin: mat3x4<f32>;
    var linear: mat3x3<f32>;

    let _e17 = skin_blend(skin_input_1);
    skin = _e17;
    let _e19 = skin;
    let _e20 = skin_linear(_e19);
    linear = _e20;
    let _e22 = skin;
    let _e24 = apply_affine(_e22, input_4.position);
    let _e25 = linear;
    let _e27 = decode_normal(input_4.normal);
    let _e29 = linear;
    let _e31 = decode_normal(input_4.tangent);
    let _e34 = linear;
    let _e38 = raster_vertex(input_4, _e24, (_e25 * _e27), (_e29 * _e31), (input_4.bitangent_sign * sign(determinant(_e34))));
    return _e38;
}

@fragment 
fn raster_fs(input_5: VertexOutput) -> @location(0) vec4<f32> {
    var mr_sample: vec4<f32>;
    var base_color: vec3<f32>;
    var mat: Material;
    var n: vec3<f32>;
    var normal_scale: f32;
    var raw_unorm: vec2<f32>;
    var n_xy: vec2<f32>;
    var n_z: f32;
    var n_tangent: vec3<f32>;
    var tbn: mat3x3<f32>;
    var v_1: vec3<f32>;
    var l: vec3<f32>;
    var brdf: BrdfLobes;
    var visibility: f32;
    var light: vec3<f32>;
    var ambient: vec3<f32>;
    var emissive: vec3<f32>;
    var local_1: vec3<f32>;
    var color_1: vec3<f32>;
    var mapped_1: vec3<f32>;

    let _e17 = textureSample(metallic_roughness_tex, samp, input_5.uv);
    mr_sample = _e17;
    let _e20 = textureSample(base_color_tex, samp, input_5.uv);
    let _e23 = draw_params.base_color_factor;
    base_color = (_e20.xyz * _e23.xyz);
    let _e27 = base_color;
    let _e30 = draw_params.material.y;
    let _e32 = mr_sample.z;
    let _e39 = draw_params.material.z;
    let _e41 = mr_sample.y;
    let _e46 = material_from_metallic_roughness(_e27, clamp((_e30 * _e32), 0f, 1f), clamp((_e39 * _e41), 0f, 1f));
    mat = _e46;
    n = normalize(input_5.normal);
    let _e53 = draw_params.material.x;
    normal_scale = _e53;
    let _e55 = normal_scale;
    if (_e55 > 0f) {
        let _e59 = textureSample(normal_tex, samp, input_5.uv);
        raw_unorm = _e59.xy;
        let _e62 = normal_scale;
        let _e64 = raw_unorm;
        n_xy = (vec2(_e62) * ((vec2(2f) * _e64) - vec2(1f)));
        let _e75 = n_xy;
        let _e76 = n_xy;
        n_z = sqrt(max(0f, (1f - dot(_e75, _e76))));
        let _e82 = n_xy;
        let _e83 = n_z;
        n_tangent = normalize(vec3<f32>(_e82, _e83));
        let _e91 = n;
        tbn = mat3x3<f32>(normalize(input_5.tangent), normalize(input_5.bitangent), _e91);
        let _e94 = tbn;
        let _e95 = n_tangent;
        n = normalize((_e94 * _e95));
    }
    let _e99 = frame_params.camera_pos;
    v_1 = normalize((_e99.xyz - input_5.world_pos));
    let _e106 = frame_params.light_dir;
    l = normalize(_e106.xyz);
    let _e110 = mat;
    let _e111 = n;
    let _e112 = v_1;
    let _e113 = l;
    let _e114 = evaluate_brdf(_e110, _e111, _e112, _e113);
    brdf = _e114;
    let _e117 = n;
    let _e118 = directional_shadow(input_5.world_pos, _e117);
    visibility = _e118;
    let _e121 = mat.diffuse_albedo;
    let _e123 = brdf.diffuse;
    let _e127 = brdf.specular;
    let _e130 = frame_params.light_color;
    let _e133 = visibility;
    light = ((((_e121 * vec3(_e123)) + _e127) * _e130.xyz) * vec3(_e133));
    let _e137 = mat;
    let _e138 = evaluate_ambient(_e137);
    let _e140 = frame_params.ambient_color;
    ambient = (_e138 * _e140.xyz);
    let _e145 = draw_params.emissive_factor;
    let _e148 = textureSample(emissive_tex, samp, input_5.uv);
    emissive = (_e145.xyz * _e148.xyz);
    let _e152 = mat;
    let _e153 = n;
    let _e154 = v_1;
    let _e156 = shade_local_light(_e152, _e153, _e154, input_5.world_pos);
    local_1 = _e156;
    let _e158 = ambient;
    let _e159 = light;
    let _e161 = local_1;
    let _e163 = emissive;
    color_1 = (((_e158 + _e159) + _e161) + _e163);
    let _e166 = color_1;
    let _e167 = color_1;
    mapped_1 = (_e166 / (_e167 + vec3(1f)));
    let _e173 = mapped_1;
    let _e176 = frame_params.settings.y;
    let _e179 = encode_surface_color(_e173, (_e176 > 0.5f));
    return vec4<f32>(_e179, 1f);
}
