use std::{fmt::Debug, hash::Hash};
pub trait ResourceDevice {
    type Buffer: Clone + Copy + Debug + Hash + PartialEq;
    type Texture: Clone + Copy + Debug + Hash + PartialEq;
    type TextureView: Clone + Copy + Debug + Hash + PartialEq;
    type Sampler: Clone + Copy + Debug + Hash + PartialEq;

    fn create_buffer(&self, desc: super::BufferDesc) -> Self::Buffer;
    fn sync_buffer(&self, buffer: Self::Buffer);
    fn destroy_buffer(&self, buffer: Self::Buffer);
    fn create_texture(&self, desc: super::TextureDesc) -> Self::Texture;
    fn destroy_texture(&self, texture: Self::Texture);
    fn create_texture_view(&self, desc: super::TextureViewDesc) -> Self::TextureView;
    fn destroy_texture_view(&self, view: Self::TextureView);
    fn create_sampler(&self, desc: super::SamplerDesc) -> Self::Sampler;
    fn destroy_sampler(&self, sampler: Self::Sampler);
}

pub trait CommandDevice {
    type CommandEncoder;
    type SyncPoint: Clone + Debug;

    fn create_command_encoder(&self, desc: super::CommandEncoderDesc) -> Self::CommandEncoder;
    fn destroy_command_encoder(&self, encoder: Self::CommandEncoder);
    fn submit(&self, encoder: &mut Self::CommandEncoder) -> Self::SyncPoint;
    fn wait_for(&self, sp: &Self::SyncPoint, timeout_ms: u32) -> bool;
}

pub trait TransferEncoder {
    fn fill_buffer(&mut self, dst: super::BufferPiece, size: u64, value: u8);
    fn copy_buffer_to_buffer(
        &mut self,
        src: super::BufferPiece,
        dst: super::BufferPiece,
        size: u64,
    );
    fn copy_texture_to_texture(
        &mut self,
        src: super::TexturePiece,
        dst: super::TexturePiece,
        size: super::Extent,
    );

    fn copy_buffer_to_texture(
        &mut self,
        src: super::BufferPiece,
        bytes_per_row: u32,
        dst: super::TexturePiece,
        size: super::Extent,
    );

    fn copy_texture_to_buffer(
        &mut self,
        src: super::TexturePiece,
        dst: super::BufferPiece,
        bytes_per_row: u32,
        size: super::Extent,
    );
}

pub trait PipelineEncoder {
    fn bind<D: super::ShaderData>(&mut self, group: u32, data: &D);
}

pub trait ComputePipelineEncoder: PipelineEncoder {
    fn dispatch(&mut self, groups: [u32; 3]);
}

pub trait RenderPipelineEncoder: PipelineEncoder {
    //TODO: reconsider exposing this here
    fn set_scissor_rect(&mut self, rect: &super::ScissorRect);
    fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    );
    fn draw_indexed(
        &mut self,
        index_buf: super::BufferPiece,
        index_type: super::IndexType,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    );
}
