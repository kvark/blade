use std::{marker::PhantomData, mem};

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

/// How the `Vertex` derive finds a field's format. A type has one if it is
/// `HasVertexAttribute`, or else if the `mint` type it converts into is, as
/// the vectors of a math library that supports `mint` do.
///
/// The derive calls `(&VertexAttributeOf::<T>(PhantomData)).vertex_format()`.
/// Method lookup tries `DirectVertexAttribute`, which takes the receiver as
/// it is, before `MintVertexAttribute`, which takes a reference to it. The
/// two can't be one blanket impl: `mint` might implement `IntoMint` for `f32`
/// some day, and the impls would then overlap.
#[doc(hidden)]
pub struct VertexAttributeOf<T>(pub PhantomData<T>);

#[doc(hidden)]
pub trait DirectVertexAttribute {
    fn vertex_format(&self) -> VertexFormat;
}
impl<T: HasVertexAttribute> DirectVertexAttribute for VertexAttributeOf<T> {
    fn vertex_format(&self) -> VertexFormat {
        T::FORMAT
    }
}

#[doc(hidden)]
pub trait MintVertexAttribute {
    fn vertex_format(&self) -> VertexFormat;
}
impl<T> MintVertexAttribute for &VertexAttributeOf<T>
where
    T: mint::IntoMint,
    T::MintType: HasVertexAttribute,
{
    fn vertex_format(&self) -> VertexFormat {
        <T::MintType as HasVertexAttribute>::FORMAT
    }
}
