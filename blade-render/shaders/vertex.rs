use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct Vertex {
    pub position: Vec3,
    pub bitangent_sign: f32,
    pub tex_coords: Vec2,
    pub normal: u32,
    pub tangent: u32,
}

pub fn decode_normal(raw: u32) -> Vec3 {
    return unpack4x8snorm(raw).xyz();
}

pub fn tangent_basis(
    n: Vec3,
    transformed_tangent: Vec3,
    bitangent_sign: f32,
    linear_sign: f32,
) -> Mat3 {
    let t = normalize(transformed_tangent - n * dot(n, transformed_tangent));
    let b = normalize(cross(n, t)) * bitangent_sign * linear_sign;
    return mat3(t, b, n);
}
