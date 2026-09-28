use super::skin_inc::{SkinVertex, apply_affine, skin_blend, skin_linear, skinning_params};
use super::vertex::{Vertex, decode_normal};
use synaga_shader::*;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct SkinDispatch {
    pub vertex_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

static skin_dispatch: Uniform<SkinDispatch> = binding();
static source: Storage<[Vertex]> = binding();
static skin_source: Storage<[SkinVertex]> = binding();
static destination: StorageMut<[Vertex]> = binding();

fn encode_normal(n: Vec3) -> u32 {
    pack4x8snorm(n.extend(0.0))
}

fn normalize_or_zero(v: Vec3) -> Vec3 {
    let len2 = v.dot(v);
    if len2 < 1.0e-20 {
        return Vec3::splat(0.0);
    }
    v * inverse_sqrt(len2)
}

fn skin_stored_vertex(input: Vertex, skin: SkinVertex) -> Vertex {
    let mut out = input;
    let blended = skin_blend(skin);
    let skinned_position = apply_affine(blended, input.position);
    let linear = skin_linear(skinning_params.post_transform) * skin_linear(blended);
    out.position = apply_affine(skinning_params.post_transform, skinned_position);
    // Uniform scale is assumed, so the linear part rotates the normals
    // after normalization (non-uniform scale logs a load-time warning).
    out.normal = encode_normal(normalize_or_zero(linear * decode_normal(input.normal)));
    out.tangent = encode_normal(normalize_or_zero(linear * decode_normal(input.tangent)));
    out.bitangent_sign *= sign(determinant(linear));
    out
}

#[entry_point(compute, threads(64, 1, 1))]
fn skin(global_invocation_id: Vec3<u32>) {
    let i = global_invocation_id.x;
    if i >= skin_dispatch.vertex_count {
        return;
    }
    destination.get_mut()[i as usize] =
        skin_stored_vertex(source[i as usize], skin_source[i as usize]);
}
