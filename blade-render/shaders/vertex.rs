use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct Vertex {
    pub position: vec3,
    pub bitangent_sign: f32,
    pub tex_coords: vec2,
    pub normal: u32,
    pub tangent: u32,
}

#[shader]
pub fn decode_normal(raw: u32) -> vec3 {
    return unpack4x8snorm(raw).xyz();
}

#[shader]
pub fn tangent_basis(
    n: vec3,
    transformed_tangent: vec3,
    bitangent_sign: f32,
    linear_sign: f32,
) -> mat3x3 {
    let t = normalize(transformed_tangent - n * dot(n, transformed_tangent));
    let b = normalize(cross(n, t)) * bitangent_sign * linear_sign;
    return mat3x3(t, b, n);
}
