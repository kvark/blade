use std::{fmt::Debug, hash::Hash};
pub trait ResourceDevice {
    type Buffer: Clone + Copy + Debug + Hash + PartialEq;
    type Texture: Clone + Copy + Debug + Hash + PartialEq;
    type TextureView: Clone + Copy + Debug + Hash + PartialEq;
    type Sampler: Clone + Copy + Debug + Hash + PartialEq;

    fn create_buffer(&self, desc: super::BufferDesc) -> Self::Buffer;
    fn destroy_buffer(&self, buffer: Self::Buffer);
    fn create_texture(&self, desc: crate::TextureDesc) -> Self::Texture;
    fn destroy_texture(&self, texture: Self::Texture);
    fn create_texture_view(&self, desc: super::TextureViewDesc) -> Self::TextureView;
    fn destroy_texture_view(&self, view: Self::TextureView);
    fn create_sampler(&self, desc: super::SamplerDesc) -> Self::Sampler;
    fn destroy_sampler(&self, sampler: Self::Sampler);
}
