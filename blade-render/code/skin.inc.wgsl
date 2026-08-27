// Shared by the raster vertex shaders and the compute skin pass.
// Must match `blade_render::SkinVertex`.
#include "vertex.inc.wgsl"

#use MAX_JOINTS_PER_DRAW

struct SkinningParams {
    post_transform: mat3x4<f32>,
    joint_matrices: array<mat3x4<f32>, MAX_JOINTS_PER_DRAW>,
}

// Has to match `blade_render::SkinVertex`!
struct SkinVertex {
    joints: u32,
    weights: u32,
}

var<uniform> skinning_params: SkinningParams;

fn unpack_joints(raw: u32) -> vec4<u32> {
    return (vec4<u32>(raw) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(0xFFu);
}

fn apply_affine(m: mat3x4<f32>, p: vec3<f32>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return h * m;
}

fn skin_blend(skin: SkinVertex) -> mat3x4<f32> {
    let joints = unpack_joints(skin.joints);
    let weights = unpack4x8unorm(skin.weights);
    return skinning_params.joint_matrices[joints.x] * weights.x
        + skinning_params.joint_matrices[joints.y] * weights.y
        + skinning_params.joint_matrices[joints.z] * weights.z
        + skinning_params.joint_matrices[joints.w] * weights.w;
}

fn skin_linear(skin: mat3x4<f32>) -> mat3x3<f32> {
    return transpose(mat3x3<f32>(skin[0].xyz, skin[1].xyz, skin[2].xyz));
}
