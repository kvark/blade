use synaga_shader::*;

/// A vertex as the host uploads it, and as the vertex shaders and the ray
/// tracer read it. Its layout for a pipeline is derived from the fields.
#[repr(C)]
#[derive(Debug, Shared, blade_macros::Vertex)]
pub struct Vertex {
    pub position: Vec3,
    pub bitangent_sign: f32,
    pub tex_coords: Vec2,
    pub normal: u32,
    pub tangent: u32,
}

pub fn decode_normal(raw: u32) -> Vec3 {
    unpack4x8snorm(raw).xyz()
}

pub fn tangent_basis(
    n: Vec3,
    transformed_tangent: Vec3,
    bitangent_sign: f32,
    linear_sign: f32,
) -> Mat3 {
    let t = (transformed_tangent - n * n.dot(transformed_tangent)).normalize();
    let b = n.cross(t).normalize() * bitangent_sign * linear_sign;
    mat3(t, b, n)
}
