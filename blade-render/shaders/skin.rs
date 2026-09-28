use super::skin_inc::{SkinVertex, apply_affine, skin_blend, skin_linear, skinning_params};
use super::vertex::{Vertex, decode_normal};
use synaga_shader::*;

#[derive(Clone, Copy, Default)]
struct SkinDispatch {
    pub vertex_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

static skin_dispatch: Uniform<SkinDispatch> = binding();
static source: Storage<[Vertex]> = binding();
static skin_source: Storage<[SkinVertex]> = binding();
static destination: StorageMut<[Vertex]> = binding();

fn encode_normal(n: vec3) -> u32 {
    return pack4x8snorm((n).extend(0.0));
}

fn normalize_or_zero(v: vec3) -> vec3 {
    let len2 = dot(v, v);
    if len2 < 1.0e-20 {
        return vec3::splat(0.0);
    }
    return v * inverseSqrt(len2);
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
    return out;
}

#[entry_point(compute, threads(64, 1, 1))]
fn skin(#[builtin(global_invocation_id)] global_id: vec3u) {
    let i = global_id.x;
    if i >= skin_dispatch.vertex_count {
        return;
    }
    unsafe {
        destination.get_mut()[(i) as usize] =
            skin_stored_vertex(source[(i) as usize], skin_source[(i) as usize]);
    }
}
