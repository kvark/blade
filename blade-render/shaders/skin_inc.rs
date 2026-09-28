use super::config::MAX_JOINTS_PER_DRAW_LEN;

use synaga_shader::*;

#[derive(Clone, Copy)]
pub struct SkinningParams {
    pub post_transform: Mat3x4,
    pub joint_matrices: [Mat3x4; MAX_JOINTS_PER_DRAW_LEN],
}

#[derive(Clone, Copy, Default)]
pub struct SkinVertex {
    pub joints: u32,
    pub weights: u32,
}

pub static skinning_params: Uniform<SkinningParams> = binding();

fn unpack_joints(raw: u32) -> Vec4<u32> {
    return (Vec4::<u32>::splat(raw) >> vec4::<u32>(0u32, 8u32, 16u32, 24u32))
        & Vec4::<u32>::splat(0xFFu32);
}

pub fn apply_affine(m: Mat3x4, p: Vec3) -> Vec3 {
    let h = (p).extend(1.0);
    return h * m;
}

pub fn skin_linear(skin: Mat3x4) -> Mat3 {
    return transpose(mat3(skin[0].xyz(), skin[1].xyz(), skin[2].xyz()));
}

pub fn skin_blend(skin: SkinVertex) -> Mat3x4 {
    let joints = unpack_joints(skin.joints);
    let weights = unpack4x8unorm(skin.weights);
    return skinning_params.joint_matrices[(joints.x) as usize] * weights.x
        + skinning_params.joint_matrices[(joints.y) as usize] * weights.y
        + skinning_params.joint_matrices[(joints.z) as usize] * weights.z
        + skinning_params.joint_matrices[(joints.w) as usize] * weights.w;
}
