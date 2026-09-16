use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

use blade_graphics as gpu;
use wgpu::custom::{
    CommandEncoderInterface, ComputePassInterface, DispatchBindGroup, DispatchBlas,
    DispatchBuffer, DispatchCommandBuffer, DispatchComputePass, DispatchComputePipeline,
    DispatchQuerySet, DispatchRenderBundle, DispatchRenderPass, DispatchTexture,
    DispatchTextureView, RenderPassInterface,
};

use crate::{
    conv, todo, BladeBindGroup, BladeBuffer, BladeComputePipeline, BladeRenderPipeline,
    BladeTexture, BladeTextureView,
};

/// Blade's command encoder contains mapped pointers and is not `Send`. wgpu's
/// custom backend traits require `Send + Sync`; the encoder is only used from
/// the recording thread.
pub(crate) struct SendEnc(pub gpu::CommandEncoder);
unsafe impl Send for SendEnc {}
unsafe impl Sync for SendEnc {}
impl std::fmt::Debug for SendEnc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SendEnc").finish()
    }
}

#[derive(Debug)]
pub struct BladeCommandEncoder {
    inner: Arc<Mutex<Option<SendEnc>>>,
}

impl BladeCommandEncoder {
    pub(crate) fn new(encoder: gpu::CommandEncoder) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(SendEnc(encoder)))),
        }
    }

    fn with_encoder<R>(&self, f: impl FnOnce(&mut gpu::CommandEncoder) -> R) -> R {
        let mut guard = self.inner.lock().unwrap();
        let encoder = guard.as_mut().expect("encoder already finished");
        f(&mut encoder.0)
    }
}

#[derive(Debug)]
pub struct BladeCommandBuffer {
    encoder: Mutex<Option<SendEnc>>,
}

impl BladeCommandBuffer {
    pub(crate) fn take_encoder(&self) -> Option<gpu::CommandEncoder> {
        self.encoder.lock().unwrap().take().map(|e| e.0)
    }
}

fn buffer<'a>(dispatch: &'a DispatchBuffer) -> &'a BladeBuffer {
    dispatch
        .as_custom::<BladeBuffer>()
        .expect("buffer is not blade-wgpu")
}

impl CommandEncoderInterface for BladeCommandEncoder {
    fn copy_buffer_to_buffer(
        &self,
        source: &DispatchBuffer,
        source_offset: wgpu::BufferAddress,
        destination: &DispatchBuffer,
        destination_offset: wgpu::BufferAddress,
        copy_size: Option<wgpu::BufferAddress>,
    ) {
        profiling::scope!("blade-wgpu::copy_buffer_to_buffer");
        let src = buffer(source);
        let dst = buffer(destination);
        let size = copy_size.unwrap_or_else(|| src.size.saturating_sub(source_offset));
        self.with_encoder(|encoder| {
            let mut pass = encoder.transfer("copy-b2b");
            pass.copy_buffer_to_buffer(
                src.raw.at(source_offset),
                dst.raw.at(destination_offset),
                size,
            );
        });
    }

    fn copy_buffer_to_texture(
        &self,
        source: wgpu::TexelCopyBufferInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        profiling::scope!("blade-wgpu::copy_buffer_to_texture");
        let buf = source
            .buffer
            .as_custom::<BladeBuffer>()
            .expect("copy_buffer_to_texture buffer");
        let tex = destination
            .texture
            .as_custom::<BladeTexture>()
            .expect("copy_buffer_to_texture texture");
        let bytes_per_row = source.layout.bytes_per_row.unwrap_or(copy_size.width * 4);
        self.with_encoder(|encoder| {
            let mut pass = encoder.transfer("copy-b2t");
            pass.copy_buffer_to_texture(
                buf.raw.at(source.layout.offset),
                bytes_per_row,
                gpu::TexturePiece {
                    texture: tex.raw,
                    mip_level: destination.mip_level,
                    array_layer: destination.origin.z,
                    origin: [destination.origin.x, destination.origin.y, 0],
                },
                conv::extent3d(copy_size),
            );
        });
    }
    fn copy_texture_to_buffer(
        &self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyBufferInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        profiling::scope!("blade-wgpu::copy_texture_to_buffer");
        let tex = source
            .texture
            .as_custom::<BladeTexture>()
            .expect("copy_texture_to_buffer texture");
        let buf = destination
            .buffer
            .as_custom::<BladeBuffer>()
            .expect("copy_texture_to_buffer buffer");
        let bytes_per_row = destination
            .layout
            .bytes_per_row
            .unwrap_or(copy_size.width * 4);
        self.with_encoder(|encoder| {
            let mut pass = encoder.transfer("copy-t2b");
            pass.copy_texture_to_buffer(
                gpu::TexturePiece {
                    texture: tex.raw,
                    mip_level: source.mip_level,
                    array_layer: source.origin.z,
                    origin: [source.origin.x, source.origin.y, 0],
                },
                buf.raw.at(destination.layout.offset),
                bytes_per_row,
                conv::extent3d(copy_size),
            );
        });
    }
    fn copy_texture_to_texture(
        &self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        profiling::scope!("blade-wgpu::copy_texture_to_texture");
        let src = source
            .texture
            .as_custom::<BladeTexture>()
            .expect("copy_texture_to_texture src");
        let dst = destination
            .texture
            .as_custom::<BladeTexture>()
            .expect("copy_texture_to_texture dst");
        self.with_encoder(|encoder| {
            let mut pass = encoder.transfer("copy-t2t");
            pass.copy_texture_to_texture(
                gpu::TexturePiece {
                    texture: src.raw,
                    mip_level: source.mip_level,
                    array_layer: source.origin.z,
                    origin: [source.origin.x, source.origin.y, 0],
                },
                gpu::TexturePiece {
                    texture: dst.raw,
                    mip_level: destination.mip_level,
                    array_layer: destination.origin.z,
                    origin: [destination.origin.x, destination.origin.y, 0],
                },
                conv::extent3d(copy_size),
            );
        });
    }

    fn begin_compute_pass(&self, desc: &wgpu::ComputePassDescriptor<'_>) -> DispatchComputePass {
        DispatchComputePass::custom(BladeComputePass {
            encoder: self.inner.clone(),
            label: desc.label.unwrap_or("compute").to_string(),
            ops: Vec::new(),
        })
    }

    fn begin_render_pass(&self, desc: &wgpu::RenderPassDescriptor<'_>) -> DispatchRenderPass {
        let mut colors = Vec::new();
        for att in desc.color_attachments {
            let Some(att) = att else { continue };
            let view = att
                .view
                .as_custom::<BladeTextureView>()
                .expect("color attachment view")
                .raw;
            let resolve = att.resolve_target.map(|v| {
                v.as_custom::<BladeTextureView>()
                    .expect("resolve view")
                    .raw
            });
            let init = match att.ops.load {
                wgpu::LoadOp::Clear(c) => gpu::InitOp::Clear(conv::color(c)),
                wgpu::LoadOp::Load => gpu::InitOp::Load,
                wgpu::LoadOp::DontCare(_) => gpu::InitOp::DontCare,
            };
            let finish = if let Some(resolve) = resolve {
                gpu::FinishOp::ResolveTo(resolve)
            } else {
                match att.ops.store {
                    wgpu::StoreOp::Store => gpu::FinishOp::Store,
                    wgpu::StoreOp::Discard => gpu::FinishOp::Discard,
                }
            };
            colors.push(ColorAtt { view, init, finish });
        }
        let depth = desc.depth_stencil_attachment.as_ref().map(|att| {
            let view = att
                .view
                .as_custom::<BladeTextureView>()
                .expect("depth view")
                .raw;
            let init = match att.depth_ops.as_ref().map(|o| o.load) {
                Some(wgpu::LoadOp::Clear(d)) => {
                    let stencil = att
                        .stencil_ops
                        .as_ref()
                        .and_then(|o| match o.load {
                            wgpu::LoadOp::Clear(s) => Some(s),
                            _ => None,
                        })
                        .unwrap_or(0);
                    gpu::InitOp::Clear(gpu::TextureColor::Rgba([d, 0.0, 0.0, stencil as f32]))
                }
                Some(wgpu::LoadOp::Load) | None => gpu::InitOp::Load,
                Some(wgpu::LoadOp::DontCare(_)) => gpu::InitOp::DontCare,
            };
            let finish = match att.depth_ops.as_ref().map(|o| o.store) {
                Some(wgpu::StoreOp::Discard) => gpu::FinishOp::Discard,
                _ => gpu::FinishOp::Store,
            };
            ColorAtt { view, init, finish }
        });
        DispatchRenderPass::custom(BladeRenderPass {
            encoder: self.inner.clone(),
            label: desc.label.unwrap_or("render").to_string(),
            colors,
            depth,
            ops: Vec::new(),
            pipeline: None,
            groups: Vec::new(),
            vertex_buffers: Vec::new(),
            index: None,
            viewport: None,
            scissor: None,
            blend_constant: None,
            stencil_reference: None,
        })
    }

    fn finish(&mut self) -> DispatchCommandBuffer {
        profiling::scope!("blade-wgpu::CommandEncoder::finish");
        let encoder = self
            .inner
            .lock()
            .unwrap()
            .take()
            .expect("encoder already finished");
        DispatchCommandBuffer::custom(BladeCommandBuffer {
            encoder: Mutex::new(Some(encoder)),
        })
    }

    fn clear_texture(
        &self,
        _texture: &DispatchTexture,
        _subresource_range: &wgpu::ImageSubresourceRange,
    ) {
        todo("clear_texture")
    }
    fn clear_buffer(
        &self,
        _buffer: &DispatchBuffer,
        _offset: wgpu::BufferAddress,
        _size: Option<wgpu::BufferAddress>,
    ) {
        todo("clear_buffer")
    }
    fn insert_debug_marker(&self, _label: &str) {}
    fn push_debug_group(&self, _label: &str) {}
    fn pop_debug_group(&self) {}
    fn write_timestamp(&self, _query_set: &DispatchQuerySet, _query_index: u32) {}
    fn resolve_query_set(
        &self,
        _query_set: &DispatchQuerySet,
        _first_query: u32,
        _query_count: u32,
        _destination: &DispatchBuffer,
        _destination_offset: wgpu::BufferAddress,
    ) {
    }
    fn mark_acceleration_structures_built<'a>(
        &self,
        _blas: &mut dyn Iterator<Item = &'a wgpu::Blas>,
        _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>,
    ) {
    }
    fn build_acceleration_structures<'a>(
        &self,
        _blas: &mut dyn Iterator<Item = &'a wgpu::BlasBuildEntry<'a>>,
        _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>,
    ) {
        todo("build_acceleration_structures")
    }
    fn transition_resources<'a>(
        &mut self,
        _buffer_transitions: &mut dyn Iterator<Item = wgpu::wgt::BufferTransition<&'a DispatchBuffer>>,
        _texture_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::TextureTransition<&'a DispatchTexture>,
        >,
    ) {
    }
}

/// Commands are recorded, then flushed into a Blade compute pass on drop.
#[derive(Debug)]
pub struct BladeComputePass {
    encoder: Arc<Mutex<Option<SendEnc>>>,
    label: String,
    ops: Vec<ComputeOp>,
}

#[derive(Debug)]
enum ComputeOp {
    SetPipeline(Arc<gpu::ComputePipeline>),
    SetBindGroup {
        index: u32,
        group: Option<Arc<BladeBindGroup>>,
    },
    Dispatch([u32; 3]),
}

impl BladeComputePass {
    fn flush(&mut self) {
        profiling::scope!("blade-wgpu::compute_pass");
        if self.ops.is_empty() {
            return;
        }
        let mut guard = self.encoder.lock().unwrap();
        let encoder = guard.as_mut().expect("encoder missing during compute pass");
        let mut pass = encoder.0.compute(&self.label);
        let mut pipeline: Option<Arc<gpu::ComputePipeline>> = None;
        let mut groups: Vec<Option<Arc<BladeBindGroup>>> = Vec::new();
        for op in self.ops.drain(..) {
            match op {
                ComputeOp::SetPipeline(p) => pipeline = Some(p),
                ComputeOp::SetBindGroup { index, group } => {
                    let index = index as usize;
                    if groups.len() <= index {
                        groups.resize_with(index + 1, || None);
                    }
                    groups[index] = group;
                }
                ComputeOp::Dispatch(dims) => {
                    let p = pipeline.as_ref().expect("dispatch without pipeline");
                    let mut pe = pass.with(p);
                    for (i, group) in groups.iter().enumerate() {
                        if let Some(group) = group {
                            pe.bind(i as u32, group.as_ref());
                        }
                    }
                    pe.dispatch(dims);
                }
            }
        }
    }
}

impl Drop for BladeComputePass {
    fn drop(&mut self) {
        self.flush();
    }
}

impl ComputePassInterface for BladeComputePass {
    fn set_pipeline(&mut self, pipeline: &DispatchComputePipeline) {
        let p = pipeline
            .as_custom::<BladeComputePipeline>()
            .expect("compute pipeline is not blade-wgpu");
        self.ops.push(ComputeOp::SetPipeline(p.pipeline.clone()));
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&DispatchBindGroup>,
        offsets: &[wgpu::DynamicOffset],
    ) {
        let group = bind_group.map(|g| {
            Arc::new(
                g.as_custom::<BladeBindGroup>()
                    .expect("bind group is not blade-wgpu")
                    .with_dynamic_offsets(offsets),
            )
        });
        self.ops.push(ComputeOp::SetBindGroup { index, group });
    }

    fn set_immediates(&mut self, _offset: u32, _data: &[u8]) {
        todo("set_immediates")
    }
    fn insert_debug_marker(&mut self, _label: &str) {}
    fn push_debug_group(&mut self, _group_label: &str) {}
    fn pop_debug_group(&mut self) {}
    fn write_timestamp(&mut self, _query_set: &DispatchQuerySet, _query_index: u32) {
        todo("compute write_timestamp")
    }
    fn begin_pipeline_statistics_query(&mut self, _query_set: &DispatchQuerySet, _query_index: u32) {
        todo("begin_pipeline_statistics_query")
    }
    fn end_pipeline_statistics_query(&mut self) {}
    fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.ops.push(ComputeOp::Dispatch([x, y, z]));
    }
    fn dispatch_workgroups_indirect(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::BufferAddress,
    ) {
        todo("dispatch_workgroups_indirect")
    }
    fn transition_resources<'a>(
        &mut self,
        _buffer_transitions: &mut dyn Iterator<Item = wgpu::wgt::BufferTransition<&'a DispatchBuffer>>,
        _texture_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::TextureTransition<&'a DispatchTextureView>,
        >,
    ) {
    }
}

impl crate::BladeBindGroup {
    fn clone_slots(&self) -> Self {
        Self {
            slots: self.slots.clone(),
        }
    }

    fn with_dynamic_offsets(&self, offsets: &[wgpu::DynamicOffset]) -> Self {
        if offsets.is_empty() {
            return self.clone_slots();
        }
        let mut slots = self.slots.clone();
        let mut indexed: Vec<(u32, usize)> = slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                crate::Slot::Buffer {
                    binding,
                    dynamic: true,
                    ..
                } => Some((*binding, i)),
                _ => None,
            })
            .collect();
        indexed.sort_by_key(|(binding, _)| *binding);
        assert_eq!(
            indexed.len(),
            offsets.len(),
            "blade-wgpu: dynamic offset count {} != dynamic buffer count {}",
            offsets.len(),
            indexed.len()
        );
        for (offset, (_, i)) in offsets.iter().zip(indexed) {
            if let crate::Slot::Buffer { piece, .. } = &mut slots[i] {
                piece.offset += u64::from(*offset);
            }
        }
        Self { slots }
    }
}

#[derive(Clone, Copy, Debug)]
struct ColorAtt {
    view: gpu::TextureView,
    init: gpu::InitOp,
    finish: gpu::FinishOp,
}

#[derive(Debug)]
enum RenderOp {
    SetPipeline(Arc<gpu::RenderPipeline>),
    SetBindGroup {
        index: u32,
        group: Option<Arc<BladeBindGroup>>,
    },
    SetVertexBuffer {
        slot: u32,
        buffer: gpu::Buffer,
        offset: u64,
    },
    SetIndexBuffer {
        buffer: gpu::Buffer,
        offset: u64,
        format: gpu::IndexType,
    },
    Draw {
        vertices: Range<u32>,
        instances: Range<u32>,
    },
    DrawIndexed {
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    },
    DrawIndirect {
        buffer: gpu::Buffer,
        offset: u64,
    },
    DrawIndexedIndirect {
        buffer: gpu::Buffer,
        offset: u64,
    },
}

#[derive(Debug)]
pub struct BladeRenderPass {
    encoder: Arc<Mutex<Option<SendEnc>>>,
    label: String,
    colors: Vec<ColorAtt>,
    depth: Option<ColorAtt>,
    ops: Vec<RenderOp>,
    pipeline: Option<Arc<gpu::RenderPipeline>>,
    groups: Vec<Option<Arc<BladeBindGroup>>>,
    vertex_buffers: Vec<Option<(gpu::Buffer, u64)>>,
    index: Option<(gpu::Buffer, u64, gpu::IndexType)>,
    viewport: Option<gpu::Viewport>,
    scissor: Option<gpu::ScissorRect>,
    blend_constant: Option<gpu::TextureColor>,
    stencil_reference: Option<u32>,
}

impl BladeRenderPass {
    fn flush(&mut self) {
        profiling::scope!("blade-wgpu::render_pass");
        if self.ops.is_empty() && self.colors.is_empty() {
            return;
        }
        let mut guard = self.encoder.lock().unwrap();
        let encoder = guard.as_mut().expect("encoder missing during render pass");
        let color_targets: Vec<gpu::RenderTarget> = self
            .colors
            .iter()
            .map(|c| gpu::RenderTarget {
                view: c.view,
                init_op: c.init,
                finish_op: c.finish,
            })
            .collect();
        let depth_stencil = self.depth.map(|c| gpu::RenderTarget {
            view: c.view,
            init_op: c.init,
            finish_op: c.finish,
        });
        let mut pass = encoder.0.render(
            &self.label,
            gpu::RenderTargetSet {
                colors: &color_targets,
                depth_stencil,
            },
        );
        if let Some(vp) = &self.viewport {
            pass.set_viewport(vp);
        }
        if let Some(sc) = &self.scissor {
            pass.set_scissor_rect(sc);
        }
        if let Some(r) = self.stencil_reference {
            pass.set_stencil_reference(r);
        }
        let mut pipeline: Option<Arc<gpu::RenderPipeline>> = None;
        let mut groups: Vec<Option<Arc<BladeBindGroup>>> = Vec::new();
        let mut vbufs: Vec<Option<(gpu::Buffer, u64)>> = Vec::new();
        let mut index = None;
        for op in self.ops.drain(..) {
            match op {
                RenderOp::SetPipeline(p) => pipeline = Some(p),
                RenderOp::SetBindGroup { index, group } => {
                    let i = index as usize;
                    if groups.len() <= i {
                        groups.resize_with(i + 1, || None);
                    }
                    groups[i] = group;
                }
                RenderOp::SetVertexBuffer {
                    slot,
                    buffer,
                    offset,
                } => {
                    let i = slot as usize;
                    if vbufs.len() <= i {
                        vbufs.resize(i + 1, None);
                    }
                    vbufs[i] = Some((buffer, offset));
                }
                RenderOp::SetIndexBuffer {
                    buffer,
                    offset,
                    format,
                } => index = Some((buffer, offset, format)),
                RenderOp::Draw {
                    vertices,
                    instances,
                } => {
                    let p = pipeline.as_ref().expect("draw without pipeline");
                    let mut pe = pass.with(p);
                    for (i, group) in groups.iter().enumerate() {
                        if let Some(group) = group {
                            pe.bind(i as u32, group.as_ref());
                        }
                    }
                    for (i, vb) in vbufs.iter().enumerate() {
                        if let Some((buf, off)) = vb {
                            pe.bind_vertex(i as u32, buf.at(*off));
                        }
                    }
                    pe.draw(
                        vertices.start,
                        vertices.end.saturating_sub(vertices.start),
                        instances.start,
                        instances.end.saturating_sub(instances.start),
                    );
                }
                RenderOp::DrawIndexed {
                    indices,
                    base_vertex,
                    instances,
                } => {
                    let p = pipeline.as_ref().expect("draw_indexed without pipeline");
                    let (ib, ioff, ifmt) = index.expect("draw_indexed without index buffer");
                    let mut pe = pass.with(p);
                    for (i, group) in groups.iter().enumerate() {
                        if let Some(group) = group {
                            pe.bind(i as u32, group.as_ref());
                        }
                    }
                    for (i, vb) in vbufs.iter().enumerate() {
                        if let Some((buf, off)) = vb {
                            pe.bind_vertex(i as u32, buf.at(*off));
                        }
                    }
                    let index_stride = match ifmt {
                        gpu::IndexType::U16 => 2u64,
                        gpu::IndexType::U32 => 4u64,
                    };
                    pe.draw_indexed(
                        ib.at(ioff + u64::from(indices.start) * index_stride),
                        ifmt,
                        indices.end.saturating_sub(indices.start),
                        base_vertex,
                        instances.start,
                        instances.end.saturating_sub(instances.start),
                    );
                }
                RenderOp::DrawIndirect { buffer, offset } => {
                    let p = pipeline.as_ref().expect("draw_indirect without pipeline");
                    let mut pe = pass.with(p);
                    for (i, group) in groups.iter().enumerate() {
                        if let Some(group) = group {
                            pe.bind(i as u32, group.as_ref());
                        }
                    }
                    for (i, vb) in vbufs.iter().enumerate() {
                        if let Some((buf, off)) = vb {
                            pe.bind_vertex(i as u32, buf.at(*off));
                        }
                    }
                    pe.draw_indirect(buffer.at(offset));
                }
                RenderOp::DrawIndexedIndirect { buffer, offset } => {
                    let p = pipeline.as_ref().expect("draw_indexed_indirect without pipeline");
                    let (ib, ioff, ifmt) = index.expect("draw_indexed_indirect without index buffer");
                    let mut pe = pass.with(p);
                    for (i, group) in groups.iter().enumerate() {
                        if let Some(group) = group {
                            pe.bind(i as u32, group.as_ref());
                        }
                    }
                    for (i, vb) in vbufs.iter().enumerate() {
                        if let Some((buf, off)) = vb {
                            pe.bind_vertex(i as u32, buf.at(*off));
                        }
                    }
                    pe.draw_indexed_indirect(ib.at(ioff), ifmt, buffer.at(offset));
                }
            }
        }
        let _ = (pipeline, groups, vbufs, index);
    }
}

impl Drop for BladeRenderPass {
    fn drop(&mut self) {
        self.flush();
    }
}

impl RenderPassInterface for BladeRenderPass {
    fn set_pipeline(&mut self, pipeline: &wgpu::custom::DispatchRenderPipeline) {
        let p = pipeline
            .as_custom::<BladeRenderPipeline>()
            .expect("render pipeline is not blade-wgpu");
        self.ops.push(RenderOp::SetPipeline(p.pipeline.clone()));
        self.pipeline = Some(p.pipeline.clone());
    }
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&DispatchBindGroup>,
        offsets: &[wgpu::DynamicOffset],
    ) {
        let group = bind_group.map(|g| {
            Arc::new(
                g.as_custom::<BladeBindGroup>()
                    .expect("bind group is not blade-wgpu")
                    .with_dynamic_offsets(offsets),
            )
        });
        self.ops.push(RenderOp::SetBindGroup { index, group });
    }
    fn set_index_buffer(
        &mut self,
        buffer: &DispatchBuffer,
        index_format: wgpu::IndexFormat,
        offset: wgpu::BufferAddress,
        _size: Option<wgpu::BufferAddress>,
    ) {
        let buf = buffer
            .as_custom::<BladeBuffer>()
            .expect("index buffer is not blade-wgpu");
        self.index = Some((buf.raw, offset, conv::index_type(index_format)));
        self.ops.push(RenderOp::SetIndexBuffer {
            buffer: buf.raw,
            offset,
            format: conv::index_type(index_format),
        });
    }
    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: Option<&DispatchBuffer>,
        offset: wgpu::BufferAddress,
        _size: Option<wgpu::BufferAddress>,
    ) {
        let Some(buffer) = buffer else { return };
        let buf = buffer
            .as_custom::<BladeBuffer>()
            .expect("vertex buffer is not blade-wgpu");
        let i = slot as usize;
        if self.vertex_buffers.len() <= i {
            self.vertex_buffers.resize(i + 1, None);
        }
        self.vertex_buffers[i] = Some((buf.raw, offset));
        self.ops.push(RenderOp::SetVertexBuffer {
            slot,
            buffer: buf.raw,
            offset,
        });
    }
    fn set_immediates(&mut self, _offset: u32, _data: &[u8]) {
        todo("render set_immediates")
    }
    fn set_blend_constant(&mut self, color: wgpu::Color) {
        self.blend_constant = Some(conv::color(color));
    }
    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.scissor = Some(gpu::ScissorRect {
            x: x as i32,
            y: y as i32,
            w: width,
            h: height,
        });
    }
    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        self.viewport = Some(gpu::Viewport {
            x,
            y,
            w,
            h,
            depth: min_depth..max_depth,
        });
    }
    fn set_stencil_reference(&mut self, reference: u32) {
        self.stencil_reference = Some(reference);
    }
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.ops.push(RenderOp::Draw {
            vertices,
            instances,
        });
    }
    fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        self.ops.push(RenderOp::DrawIndexed {
            indices,
            base_vertex,
            instances,
        });
    }
    fn draw_mesh_tasks(&mut self, _x: u32, _y: u32, _z: u32) {
        todo("draw_mesh_tasks")
    }
    fn draw_indirect(&mut self, indirect_buffer: &DispatchBuffer, indirect_offset: wgpu::BufferAddress) {
        let buf = indirect_buffer
            .as_custom::<BladeBuffer>()
            .expect("indirect buffer is not blade-wgpu");
        self.ops.push(RenderOp::DrawIndirect {
            buffer: buf.raw,
            offset: indirect_offset,
        });
    }
    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
    ) {
        let buf = indirect_buffer
            .as_custom::<BladeBuffer>()
            .expect("indexed indirect buffer is not blade-wgpu");
        self.ops.push(RenderOp::DrawIndexedIndirect {
            buffer: buf.raw,
            offset: indirect_offset,
        });
    }
    fn draw_mesh_tasks_indirect(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::BufferAddress,
    ) {
        todo("draw_mesh_tasks_indirect")
    }
    fn multi_draw_indirect(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::BufferAddress,
        _count: u32,
    ) {
        todo("multi_draw_indirect")
    }
    fn multi_draw_indexed_indirect(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::BufferAddress,
        _count: u32,
    ) {
        todo("multi_draw_indexed_indirect")
    }
    fn multi_draw_indirect_count(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::BufferAddress,
        _count_buffer: &DispatchBuffer,
        _count_buffer_offset: wgpu::BufferAddress,
        _max_count: u32,
    ) {
        todo("multi_draw_indirect_count")
    }
    fn multi_draw_mesh_tasks_indirect(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::BufferAddress,
        _count: u32,
    ) {
        todo("multi_draw_mesh_tasks_indirect")
    }
    fn multi_draw_indexed_indirect_count(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::BufferAddress,
        _count_buffer: &DispatchBuffer,
        _count_buffer_offset: wgpu::BufferAddress,
        _max_count: u32,
    ) {
        todo("multi_draw_indexed_indirect_count")
    }
    fn multi_draw_mesh_tasks_indirect_count(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::BufferAddress,
        _count_buffer: &DispatchBuffer,
        _count_buffer_offset: wgpu::BufferAddress,
        _max_count: u32,
    ) {
        todo("multi_draw_mesh_tasks_indirect_count")
    }
    fn insert_debug_marker(&mut self, _label: &str) {}
    fn push_debug_group(&mut self, _group_label: &str) {}
    fn pop_debug_group(&mut self) {}
    fn write_timestamp(&mut self, _query_set: &DispatchQuerySet, _query_index: u32) {
        todo("render write_timestamp")
    }
    fn begin_occlusion_query(&mut self, _query_index: u32) {}
    fn end_occlusion_query(&mut self) {}
    fn begin_pipeline_statistics_query(&mut self, _query_set: &DispatchQuerySet, _query_index: u32) {}
    fn end_pipeline_statistics_query(&mut self) {}
    fn execute_bundles(&mut self, _render_bundles: &mut dyn Iterator<Item = &DispatchRenderBundle>) {
        todo("execute_bundles")
    }
}
