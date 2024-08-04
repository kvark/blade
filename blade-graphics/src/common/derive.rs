use crate::derive::HasShaderBinding;

impl HasShaderBinding for super::TextureView {
    const TYPE: crate::ShaderBinding = crate::ShaderBinding::Texture;
}
impl HasShaderBinding for super::Sampler {
    const TYPE: crate::ShaderBinding = crate::ShaderBinding::Sampler;
}
impl HasShaderBinding for super::BufferPiece {
    const TYPE: crate::ShaderBinding = crate::ShaderBinding::Buffer;
}
impl<'a, const N: crate::ResourceIndex> HasShaderBinding for &'a super::BufferArray<N> {
    const TYPE: crate::ShaderBinding = crate::ShaderBinding::BufferArray { count: N };
}
impl<'a, const N: crate::ResourceIndex> HasShaderBinding for &'a super::TextureArray<N> {
    const TYPE: crate::ShaderBinding = crate::ShaderBinding::TextureArray { count: N };
}
impl HasShaderBinding for super::AccelerationStructure {
    const TYPE: crate::ShaderBinding = crate::ShaderBinding::AccelerationStructure;
}
