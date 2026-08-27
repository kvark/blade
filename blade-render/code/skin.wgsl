#include "skin.inc.wgsl"

struct SkinDispatch {
    vertex_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<uniform> skin_dispatch: SkinDispatch;
var<storage, read> source: array<Vertex>;
var<storage, read> skin_source: array<SkinVertex>;
var<storage, read_write> destination: array<Vertex>;

fn encode_normal(n: vec3<f32>) -> u32 {
    return pack4x8snorm(vec4<f32>(n, 0.0));
}

fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if (len2 < 1.0e-20) {
        return vec3<f32>(0.0);
    }
    return v * inverseSqrt(len2);
}

fn skin_stored_vertex(input: Vertex, skin: SkinVertex) -> Vertex {
    var out = input;
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

@compute
@workgroup_size(64, 1, 1)
fn skin(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= skin_dispatch.vertex_count) {
        return;
    }
    destination[i] = skin_stored_vertex(source[i], skin_source[i]);
}
