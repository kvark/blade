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

// Inverse-transpose direction, without dividing by the determinant before
// normalization. Unlike a position transform, this preserves perpendicularity
// under non-uniform scale and shear. The determinant sign handles reflections.
fn transform_normal(linear: mat3x3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let cofactor = mat3x3<f32>(
        cross(linear[1], linear[2]), cross(linear[2], linear[0]), cross(linear[0], linear[1]),
    );
    let n = (cofactor * normal) * sign(dot(linear[0], cofactor[0]));
    return n * inverseSqrt(max(dot(n, n), 1.0e-20));
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
