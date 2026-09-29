use super::config::MAX_JOINTS_PER_DRAW;
use synaga_shader::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct SkinningParams {
    pub post_transform: Mat3x4,
    pub joint_matrices: [Mat3x4; MAX_JOINTS_PER_DRAW],
}

/// Per-vertex skinning data, kept in a separate buffer so that the base
/// vertex layout is identical for skinned and rigid models.
///
/// Its `Default`, in the host, puts all the weight on the first joint.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Zeroable, bytemuck::Pod)]
pub struct SkinVertex {
    /// Four 8-bit indices into the geometry's compact joint palette, the
    /// first in the lowest byte.
    pub joints: u32,
    /// Four unorm8 linear-blend skinning weights, one for each of `joints`.
    pub weights: u32,
}

pub static skinning_params: Uniform<SkinningParams> = binding();

fn unpack_joints(raw: u32) -> Vec4<u32> {
    (Vec4::splat(raw) >> vec4(0, 8, 16, 24)) & 0xFF
}

pub fn apply_affine(m: Mat3x4, p: Vec3) -> Vec3 {
    let h = p.extend(1.0);
    h * m
}

pub fn skin_linear(skin: Mat3x4) -> Mat3 {
    transpose(mat3(skin[0].xyz(), skin[1].xyz(), skin[2].xyz()))
}

pub fn skin_blend(skin: SkinVertex) -> Mat3x4 {
    let joints = unpack_joints(skin.joints);
    let weights = unpack4x8unorm(skin.weights);
    skinning_params.joint_matrices[joints.x as usize] * weights.x
        + skinning_params.joint_matrices[joints.y as usize] * weights.y
        + skinning_params.joint_matrices[joints.z as usize] * weights.z
        + skinning_params.joint_matrices[joints.w as usize] * weights.w
}
