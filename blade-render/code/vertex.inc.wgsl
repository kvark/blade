// Shared vertex definition and tangent-space helpers.
// Used by the raster vertex shaders, the skinning compute pass,
// and the ray-tracing geometry fetch.

// Has to match `blade_render::Vertex`!
struct Vertex {
    position: vec3<f32>,
    bitangent_sign: f32,
    tex_coords: vec2<f32>,
    normal: u32,
    tangent: u32,
}

fn decode_normal(raw: u32) -> vec3<f32> {
    return unpack4x8snorm(raw).xyz;
}

// Orthonormalize a linearly transformed tangent against the transformed
// normal, returning the tangent-frame basis columns (tangent, bitangent,
// normal). `linear_sign` flips the bitangent for mirrored transforms.
fn tangent_basis(
    n: vec3<f32>,
    transformed_tangent: vec3<f32>,
    bitangent_sign: f32,
    linear_sign: f32,
) -> mat3x3<f32> {
    let t = normalize(transformed_tangent - n * dot(n, transformed_tangent));
    let b = normalize(cross(n, t)) * bitangent_sign * linear_sign;
    return mat3x3<f32>(t, b, n);
}
