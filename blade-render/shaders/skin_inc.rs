use super::config::*;
use super::vertex::*;
use synaga_shader::*;

#[derive(Clone, Copy)]
pub struct SkinningParams {
    pub post_transform: mat3x4,
    pub joint_matrices: [mat3x4; MAX_JOINTS_PER_DRAW_LEN],
}

#[derive(Clone, Copy, Default)]
pub struct SkinVertex {
    pub joints: u32,
    pub weights: u32,
}

pub static skinning_params: Uniform<SkinningParams> = binding();

#[shader]
pub fn unpack_joints(raw: u32) -> vec4u {
    return (vec4u::splat(raw) >> vec4u(0u32, 8u32, 16u32, 24u32)) & vec4u::splat(0xFFu32);
}

#[shader]
pub fn apply_affine(m: mat3x4, p: vec3) -> vec3 {
    let h = (p).extend(1.0);
    return h * m;
}

#[shader]
pub fn skin_linear(skin: mat3x4) -> mat3x3 {
    return transpose(mat3x3(skin[0].xyz(), skin[1].xyz(), skin[2].xyz()));
}

#[shader]
pub fn skin_blend(skin: SkinVertex) -> mat3x4 {
    let joints = unpack_joints(skin.joints);
    let weights = unpack4x8unorm(skin.weights);
    return skinning_params.joint_matrices[(joints.x) as usize] * weights.x
        + skinning_params.joint_matrices[(joints.y) as usize] * weights.y
        + skinning_params.joint_matrices[(joints.z) as usize] * weights.z
        + skinning_params.joint_matrices[(joints.w) as usize] * weights.w;
}
