use std::mem;

use super::{ResourceIndex, ShaderBinding, VertexFormat};

pub trait HasShaderBinding {
    const TYPE: ShaderBinding;
}
impl<T: bytemuck::NoUninit> HasShaderBinding for T {
    const TYPE: ShaderBinding = ShaderBinding::Plain {
        size: mem::size_of::<T>() as u32,
    };
}
impl HasShaderBinding for super::TextureView {
    const TYPE: ShaderBinding = ShaderBinding::Texture;
}
impl HasShaderBinding for super::Sampler {
    const TYPE: ShaderBinding = ShaderBinding::Sampler;
}
impl HasShaderBinding for super::BufferPiece {
    const TYPE: ShaderBinding = ShaderBinding::Buffer;
}
impl<'a, const N: ResourceIndex> HasShaderBinding for &'a super::BufferArray<N> {
    const TYPE: ShaderBinding = ShaderBinding::BufferArray { count: N };
}
impl<'a, const N: ResourceIndex> HasShaderBinding for &'a super::TextureArray<N> {
    const TYPE: ShaderBinding = ShaderBinding::TextureArray { count: N };
}
impl HasShaderBinding for super::AccelerationStructure {
    const TYPE: ShaderBinding = ShaderBinding::AccelerationStructure;
}
impl<'a, const N: ResourceIndex> HasShaderBinding for &'a super::AccelerationStructureArray<N> {
    const TYPE: ShaderBinding = ShaderBinding::AccelerationStructureArray { count: N };
}

pub trait HasVertexAttribute {
    const FORMAT: VertexFormat;
}

impl HasVertexAttribute for f32 {
    const FORMAT: VertexFormat = VertexFormat::F32;
}
impl HasVertexAttribute for [f32; 2] {
    const FORMAT: VertexFormat = VertexFormat::F32Vec2;
}
impl HasVertexAttribute for [f32; 3] {
    const FORMAT: VertexFormat = VertexFormat::F32Vec3;
}
impl HasVertexAttribute for [f32; 4] {
    const FORMAT: VertexFormat = VertexFormat::F32Vec4;
}
impl HasVertexAttribute for u32 {
    const FORMAT: VertexFormat = VertexFormat::U32;
}
impl HasVertexAttribute for [u32; 2] {
    const FORMAT: VertexFormat = VertexFormat::U32Vec2;
}
impl HasVertexAttribute for [u32; 3] {
    const FORMAT: VertexFormat = VertexFormat::U32Vec3;
}
impl HasVertexAttribute for [u32; 4] {
    const FORMAT: VertexFormat = VertexFormat::U32Vec4;
}
impl HasVertexAttribute for i32 {
    const FORMAT: VertexFormat = VertexFormat::I32;
}
impl HasVertexAttribute for [i32; 2] {
    const FORMAT: VertexFormat = VertexFormat::I32Vec2;
}
impl HasVertexAttribute for [i32; 3] {
    const FORMAT: VertexFormat = VertexFormat::I32Vec3;
}
impl HasVertexAttribute for [i32; 4] {
    const FORMAT: VertexFormat = VertexFormat::I32Vec4;
}

impl HasVertexAttribute for mint::Vector2<f32> {
    const FORMAT: VertexFormat = VertexFormat::F32Vec2;
}
impl HasVertexAttribute for mint::Vector3<f32> {
    const FORMAT: VertexFormat = VertexFormat::F32Vec3;
}
impl HasVertexAttribute for mint::Vector4<f32> {
    const FORMAT: VertexFormat = VertexFormat::F32Vec4;
}
impl HasVertexAttribute for mint::Vector2<u32> {
    const FORMAT: VertexFormat = VertexFormat::U32Vec2;
}
impl HasVertexAttribute for mint::Vector3<u32> {
    const FORMAT: VertexFormat = VertexFormat::U32Vec3;
}
impl HasVertexAttribute for mint::Vector4<u32> {
    const FORMAT: VertexFormat = VertexFormat::U32Vec4;
}
impl HasVertexAttribute for mint::Vector2<i32> {
    const FORMAT: VertexFormat = VertexFormat::I32Vec2;
}
impl HasVertexAttribute for mint::Vector3<i32> {
    const FORMAT: VertexFormat = VertexFormat::I32Vec3;
}
impl HasVertexAttribute for mint::Vector4<i32> {
    const FORMAT: VertexFormat = VertexFormat::I32Vec4;
}

/// A vector a shader declares with synaga, as a vertex attribute.
#[cfg(feature = "synaga")]
mod synaga_vertex {
    use super::HasVertexAttribute;
    use crate::VertexFormat;
    use synaga_shader::{Vec2, Vec3, Vec4};

    macro_rules! format {
        ($($ty:ty => $format:ident),* $(,)?) => {
            $(impl HasVertexAttribute for $ty {
                const FORMAT: VertexFormat = VertexFormat::$format;
            })*
        };
    }

    format! {
        Vec2<f32> => F32Vec2,
        Vec3<f32> => F32Vec3,
        Vec4<f32> => F32Vec4,
        Vec2<u32> => U32Vec2,
        Vec3<u32> => U32Vec3,
        Vec4<u32> => U32Vec4,
        Vec2<i32> => I32Vec2,
        Vec3<i32> => I32Vec3,
        Vec4<i32> => I32Vec4,
    }
}
