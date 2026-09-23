struct Vertex {
    position: vec3<f32>,
    bitangent_sign: f32,
    tex_coords: vec2<f32>,
    normal: u32,
    tangent: u32,
}

struct SkinningParams {
    post_transform: mat3x4<f32>,
    joint_matrices: array<mat3x4<f32>, 64>,
}

struct SkinVertex {
    joints: u32,
    weights: u32,
}

struct SkinDispatch {
    vertex_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

var<uniform> skinning_params: SkinningParams;
var<uniform> skin_dispatch: SkinDispatch;
var<storage> source: array<Vertex>;
var<storage> skin_source: array<SkinVertex>;
var<storage, read_write> destination: array<Vertex>;

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

fn unpack_joints(raw_1: u32) -> vec4<u32> {
    return ((vec4(raw_1) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4(255u));
}

fn apply_affine(m: mat3x4<f32>, p: vec3<f32>) -> vec3<f32> {
    var h: vec4<f32>;

    h = vec4<f32>(p, 1f);
    let _e6 = h;
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

fn encode_normal(n_1: vec3<f32>) -> u32 {
    return pack4x8snorm(vec4<f32>(n_1, 0f));
}

fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    var len2: f32;

    len2 = dot(v, v);
    let _e8 = len2;
    if (_e8 < 0.00000000000000000001f) {
        return vec3(0f);
    }
    let _e13 = len2;
    return (v * vec3(inverseSqrt(_e13)));
}

fn skin_stored_vertex(input: Vertex, skin_3: SkinVertex) -> Vertex {
    var out: Vertex;
    var blended: mat3x4<f32>;
    var skinned_position: vec3<f32>;
    var linear: mat3x3<f32>;

    out = input;
    let _e8 = skin_blend(skin_3);
    blended = _e8;
    let _e10 = blended;
    let _e12 = apply_affine(_e10, input.position);
    skinned_position = _e12;
    let _e15 = skinning_params.post_transform;
    let _e16 = skin_linear(_e15);
    let _e17 = blended;
    let _e18 = skin_linear(_e17);
    linear = (_e16 * _e18);
    let _e23 = skinning_params.post_transform;
    let _e24 = skinned_position;
    let _e25 = apply_affine(_e23, _e24);
    out.position = _e25;
    let _e27 = linear;
    let _e29 = decode_normal(input.normal);
    let _e31 = normalize_or_zero((_e27 * _e29));
    let _e32 = encode_normal(_e31);
    out.normal = _e32;
    let _e34 = linear;
    let _e36 = decode_normal(input.tangent);
    let _e38 = normalize_or_zero((_e34 * _e36));
    let _e39 = encode_normal(_e38);
    out.tangent = _e39;
    let _e41 = linear;
    let _e44 = out.bitangent_sign;
    out.bitangent_sign = (_e44 * sign(determinant(_e41)));
    let _e46 = out;
    return _e46;
}

@compute @workgroup_size(64, 1, 1) 
fn skin(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var i: u32;

    i = global_id.x;
    let _e8 = i;
    let _e10 = skin_dispatch.vertex_count;
    if (_e8 >= _e10) {
        return;
    }
    let _e12 = i;
    let _e14 = i;
    let _e16 = source[_e14];
    let _e17 = i;
    let _e19 = skin_source[_e17];
    let _e20 = skin_stored_vertex(_e16, _e19);
    destination[_e12] = _e20;
}
