use std::{marker::PhantomData, mem, time::Duration};

impl<T: bytemuck::Pod> crate::ShaderBindable for T {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = ctx.targets[index as usize] as _;
        let ptr: *const T = self;
        let size = mem::size_of::<T>() as u64;
        if let Some(encoder) = ctx.vs_encoder {
            encoder.set_vertex_bytes(slot, size, ptr as *const _);
        }
        if let Some(encoder) = ctx.fs_encoder {
            encoder.set_fragment_bytes(slot, size, ptr as *const _);
        }
        if let Some(encoder) = ctx.cs_encoder {
            encoder.set_bytes(slot, size, ptr as *const _);
        }
    }
}
impl crate::ShaderBindable for super::TextureView {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = ctx.targets[index as usize] as _;
        let value = Some(self.as_ref());
        if let Some(encoder) = ctx.vs_encoder {
            encoder.set_vertex_texture(slot, value);
        }
        if let Some(encoder) = ctx.fs_encoder {
            encoder.set_fragment_texture(slot, value);
        }
        if let Some(encoder) = ctx.cs_encoder {
            encoder.set_texture(slot, value);
        }
    }
}
impl<'a, const N: crate::ResourceIndex> crate::ShaderBindable for &'a crate::TextureArray<N> {
    fn bind_to(&self, _ctx: &mut super::PipelineContext, _index: u32) {
        unimplemented!()
    }
}
impl crate::ShaderBindable for super::Sampler {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        //self.raw.set_sampler_state(index as _, sampler.as_ref());
        let slot = ctx.targets[index as usize] as _;
        let value = Some(self.as_ref());
        if let Some(encoder) = ctx.vs_encoder {
            encoder.set_vertex_sampler_state(slot, value);
        }
        if let Some(encoder) = ctx.fs_encoder {
            encoder.set_fragment_sampler_state(slot, value);
        }
        if let Some(encoder) = ctx.cs_encoder {
            encoder.set_sampler_state(slot, value);
        }
    }
}
impl crate::ShaderBindable for crate::BufferPiece {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = ctx.targets[index as usize] as _;
        let value = Some(self.buffer.as_ref());
        if let Some(encoder) = ctx.vs_encoder {
            encoder.set_vertex_buffer(slot, value, self.offset);
        }
        if let Some(encoder) = ctx.fs_encoder {
            encoder.set_fragment_buffer(slot, value, self.offset);
        }
        if let Some(encoder) = ctx.cs_encoder {
            encoder.set_buffer(slot, value, self.offset);
        }
    }
}
impl<'a, const N: crate::ResourceIndex> crate::ShaderBindable for &'a crate::BufferArray<N> {
    fn bind_to(&self, _ctx: &mut super::PipelineContext, _index: u32) {
        unimplemented!()
    }
}
impl crate::ShaderBindable for crate::AccelerationStructure {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = ctx.targets[index as usize] as _;
        let value = Some(self.as_ref());
        if let Some(encoder) = ctx.vs_encoder {
            encoder.set_vertex_acceleration_structure(slot, value);
        }
        if let Some(encoder) = ctx.fs_encoder {
            encoder.set_fragment_acceleration_structure(slot, value);
        }
        if let Some(encoder) = ctx.cs_encoder {
            encoder.set_acceleration_structure(slot, value);
        }
    }
}

impl super::TimingData {
    fn add(&mut self, label: &str) -> u64 {
        let counter_index = self.pass_names.len() as u64 * 2;
        self.pass_names.push(label.to_string());
        counter_index
    }
}

impl super::CommandEncoder {
    fn begin_pass(&mut self, label: &str) {
        if self.enable_debug_groups {
            //HACK: close the previous group
            if self.has_open_debug_group {
                self.raw.as_mut().unwrap().pop_debug_group();
            } else {
                self.has_open_debug_group = true;
            }
            self.raw.as_mut().unwrap().push_debug_group(label);
        }
    }

    pub(super) fn finish(&mut self) -> metal::CommandBuffer {
        if self.has_open_debug_group {
            self.raw.as_mut().unwrap().pop_debug_group();
        }
        self.raw.take().unwrap()
    }

    pub fn transfer(&mut self, label: &str) -> super::TransferCommandEncoder {
        self.begin_pass(label);
        let raw = objc::rc::autoreleasepool(|| {
            let descriptor = metal::BlitPassDescriptor::new();

            if let Some(ref mut td_array) = self.timing_datas {
                let td = td_array.first_mut().unwrap();
                let counter_index = td.add(label);
                let sba = descriptor.sample_buffer_attachments().object_at(0).unwrap();
                sba.set_sample_buffer(&td.sample_buffer);
                sba.set_start_of_encoder_sample_index(counter_index);
                sba.set_end_of_encoder_sample_index(counter_index + 1);
            }

            self.raw
                .as_mut()
                .unwrap()
                .blit_command_encoder_with_descriptor(&descriptor)
                .to_owned()
        });
        super::TransferCommandEncoder {
            raw,
            phantom: PhantomData,
        }
    }

    pub fn acceleration_structure(
        &mut self,
        label: &str,
    ) -> super::AccelerationStructureCommandEncoder {
        let raw = objc::rc::autoreleasepool(|| {
            let descriptor = metal::AccelerationStructurePassDescriptor::new();

            if let Some(ref mut td_array) = self.timing_datas {
                let td = td_array.first_mut().unwrap();
                let counter_index = td.add(label);
                let sba = descriptor.sample_buffer_attachments().object_at(0).unwrap();
                sba.set_sample_buffer(&td.sample_buffer);
                sba.set_start_of_encoder_sample_index(counter_index);
                sba.set_end_of_encoder_sample_index(counter_index + 1);
            }

            self.raw
                .as_mut()
                .unwrap()
                .new_acceleration_structure_command_encoder()
                .to_owned()
        });
        super::AccelerationStructureCommandEncoder {
            raw,
            phantom: PhantomData,
        }
    }

    pub fn compute(&mut self, label: &str) -> super::ComputeCommandEncoder {
        let raw = objc::rc::autoreleasepool(|| {
            let descriptor = metal::ComputePassDescriptor::new();
            if self.enable_dispatch_type {
                descriptor.set_dispatch_type(metal::MTLDispatchType::Concurrent);
            }

            if let Some(ref mut td_array) = self.timing_datas {
                let td = td_array.first_mut().unwrap();
                let counter_index = td.add(label);
                let sba = descriptor.sample_buffer_attachments().object_at(0).unwrap();
                sba.set_sample_buffer(&td.sample_buffer);
                sba.set_start_of_encoder_sample_index(counter_index);
                sba.set_end_of_encoder_sample_index(counter_index + 1);
            }

            self.raw
                .as_mut()
                .unwrap()
                .compute_command_encoder_with_descriptor(&descriptor)
                .to_owned()
        });
        super::ComputeCommandEncoder {
            raw,
            phantom: PhantomData,
        }
    }

    pub fn render(
        &mut self,
        label: &str,
        targets: crate::RenderTargetSet,
    ) -> super::RenderCommandEncoder {
        let raw = objc::rc::autoreleasepool(|| {
            let descriptor = metal::RenderPassDescriptor::new();

            for (i, rt) in targets.colors.iter().enumerate() {
                let at_descriptor = descriptor.color_attachments().object_at(i as u64).unwrap();
                at_descriptor.set_texture(Some(rt.view.as_ref()));

                let load_action = match rt.init_op {
                    crate::InitOp::Load => metal::MTLLoadAction::Load,
                    crate::InitOp::Clear(color) => {
                        let clear_color = map_clear_color(color);
                        at_descriptor.set_clear_color(clear_color);
                        metal::MTLLoadAction::Clear
                    }
                    crate::InitOp::DontCare => metal::MTLLoadAction::DontCare,
                };
                at_descriptor.set_load_action(load_action);

                let store_action = match rt.finish_op {
                    crate::FinishOp::Store | crate::FinishOp::Ignore => {
                        metal::MTLStoreAction::Store
                    }
                    crate::FinishOp::Discard => metal::MTLStoreAction::DontCare,
                    crate::FinishOp::ResolveTo(ref view) => {
                        at_descriptor.set_resolve_texture(Some(view.as_ref()));
                        metal::MTLStoreAction::MultisampleResolve
                    }
                };
                at_descriptor.set_store_action(store_action);
            }

            if let Some(ref rt) = targets.depth_stencil {
                let at_descriptor = descriptor.depth_attachment().unwrap();
                at_descriptor.set_texture(Some(rt.view.as_ref()));
                let load_action = match rt.init_op {
                    crate::InitOp::Load => metal::MTLLoadAction::Load,
                    crate::InitOp::Clear(color) => {
                        let clear_depth = match color {
                            crate::TextureColor::TransparentBlack
                            | crate::TextureColor::OpaqueBlack => 0.0,
                            crate::TextureColor::White => 1.0,
                        };
                        at_descriptor.set_clear_depth(clear_depth);
                        metal::MTLLoadAction::Clear
                    }
                    crate::InitOp::DontCare => metal::MTLLoadAction::DontCare,
                };
                let store_action = match rt.finish_op {
                    crate::FinishOp::Store | crate::FinishOp::Ignore => {
                        metal::MTLStoreAction::Store
                    }
                    crate::FinishOp::Discard => metal::MTLStoreAction::DontCare,
                    crate::FinishOp::ResolveTo(_) => panic!("Can't resolve depth texture"),
                };
                at_descriptor.set_load_action(load_action);
                at_descriptor.set_store_action(store_action);
            }

            if let Some(ref mut td_array) = self.timing_datas {
                let td = td_array.first_mut().unwrap();
                let counter_index = td.add(label);
                let sba = descriptor.sample_buffer_attachments().object_at(0).unwrap();
                sba.set_sample_buffer(&td.sample_buffer);
                sba.set_start_of_vertex_sample_index(counter_index);
                sba.set_end_of_fragment_sample_index(counter_index + 1);
            }

            self.raw
                .as_mut()
                .unwrap()
                .new_render_command_encoder(descriptor)
                .to_owned()
        });

        super::RenderCommandEncoder {
            raw,
            phantom: PhantomData,
        }
    }
}

#[hidden_trait::expose]
impl crate::traits::CommandEncoder for super::CommandEncoder {
    type Texture = super::Texture;
    type Frame = super::Frame;

    fn start(&mut self) {
        if let Some(ref mut td_array) = self.timing_datas {
            self.timings.clear();
            td_array.rotate_left(1);
            let td = td_array.first_mut().unwrap();
            if !td.pass_names.is_empty() {
                let counters = td
                    .sample_buffer
                    .resolve_counter_range(metal::NSRange::new(0, td.pass_names.len() as u64 * 2));
                for (name, chunk) in td.pass_names.drain(..).zip(counters.chunks(2)) {
                    let duration = Duration::from_nanos(chunk[1] - chunk[0]);
                    *self.timings.entry(name).or_default() += duration;
                }
            }
        }

        let queue = self.queue.lock().unwrap();
        self.raw = Some(objc::rc::autoreleasepool(|| {
            let cmd_buf = queue.new_command_buffer_with_unretained_references();
            if !self.name.is_empty() {
                cmd_buf.set_label(&self.name);
            }
            cmd_buf.to_owned()
        }));
        self.has_open_debug_group = false;
    }

    fn init_texture(&mut self, _texture: super::Texture) {}

    fn present(&mut self, frame: super::Frame) {
        self.raw.as_mut().unwrap().present_drawable(&frame.drawable);
    }

    fn timings(&self) -> &crate::Timings {
        &self.timings
    }
}

#[hidden_trait::expose]
impl crate::traits::TransferEncoder for super::TransferCommandEncoder<'_> {
    type BufferPiece = crate::BufferPiece;
    type TexturePiece = crate::TexturePiece;

    fn fill_buffer(&mut self, dst: crate::BufferPiece, size: u64, value: u8) {
        let range = metal::NSRange {
            location: dst.offset,
            length: size,
        };
        self.raw.fill_buffer(dst.buffer.as_ref(), range, value);
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: crate::BufferPiece,
        dst: crate::BufferPiece,
        size: u64,
    ) {
        self.raw.copy_from_buffer(
            src.buffer.as_ref(),
            src.offset,
            dst.buffer.as_ref(),
            dst.offset,
            size,
        );
    }
    fn copy_texture_to_texture(
        &mut self,
        src: crate::TexturePiece,
        dst: crate::TexturePiece,
        size: crate::Extent,
    ) {
        self.raw.copy_from_texture(
            src.texture.as_ref(),
            src.array_layer as u64,
            src.mip_level as u64,
            map_origin(&src.origin),
            map_extent(&size),
            dst.texture.as_ref(),
            dst.array_layer as u64,
            dst.mip_level as u64,
            map_origin(&dst.origin),
        );
    }

    fn copy_buffer_to_texture(
        &mut self,
        src: crate::BufferPiece,
        bytes_per_row: u32,
        dst: crate::TexturePiece,
        size: crate::Extent,
    ) {
        self.raw.copy_from_buffer_to_texture(
            src.buffer.as_ref(),
            src.offset,
            bytes_per_row as u64,
            0,
            map_extent(&size),
            dst.texture.as_ref(),
            dst.array_layer as u64,
            dst.mip_level as u64,
            map_origin(&dst.origin),
            metal::MTLBlitOption::empty(),
        );
    }

    fn copy_texture_to_buffer(
        &mut self,
        src: crate::TexturePiece,
        dst: crate::BufferPiece,
        bytes_per_row: u32,
        size: crate::Extent,
    ) {
        self.raw.copy_from_texture_to_buffer(
            src.texture.as_ref(),
            src.array_layer as u64,
            src.mip_level as u64,
            map_origin(&src.origin),
            map_extent(&size),
            dst.buffer.as_ref(),
            dst.offset,
            bytes_per_row as u64,
            0,
            metal::MTLBlitOption::empty(),
        );
    }
}

impl Drop for super::TransferCommandEncoder<'_> {
    fn drop(&mut self) {
        self.raw.end_encoding();
    }
}

#[hidden_trait::expose]
impl crate::traits::AccelerationStructureEncoder
    for super::AccelerationStructureCommandEncoder<'_>
{
    type AccelerationStructure = crate::AccelerationStructure;
    type AccelerationStructureMesh = crate::AccelerationStructureMesh;
    type BufferPiece = crate::BufferPiece;

    fn build_bottom_level(
        &mut self,
        acceleration_structure: super::AccelerationStructure,
        meshes: &[crate::AccelerationStructureMesh],
        scratch_data: crate::BufferPiece,
    ) {
        let descriptor = super::make_bottom_level_acceleration_structure_desc(meshes);
        self.raw.build_acceleration_structure(
            acceleration_structure.as_ref(),
            &descriptor,
            scratch_data.buffer.as_ref(),
            scratch_data.offset,
        );
    }

    fn build_top_level(
        &mut self,
        acceleration_structure: super::AccelerationStructure,
        bottom_level: &[super::AccelerationStructure],
        instance_count: u32,
        instance_data: crate::BufferPiece,
        scratch_data: crate::BufferPiece,
    ) {
        let mut primitive_acceleration_structures = Vec::with_capacity(bottom_level.len());
        for blas in bottom_level {
            primitive_acceleration_structures.push(blas.as_ref());
        }
        let descriptor = metal::InstanceAccelerationStructureDescriptor::descriptor();
        descriptor.set_instanced_acceleration_structures(&metal::Array::from_slice(
            &primitive_acceleration_structures,
        ));
        descriptor.set_instance_count(instance_count as _);
        descriptor.set_instance_descriptor_type(
            metal::MTLAccelerationStructureInstanceDescriptorType::UserID,
        );
        descriptor.set_instance_descriptor_buffer(instance_data.buffer.as_ref());
        descriptor.set_instance_descriptor_buffer_offset(instance_data.offset);

        self.raw.build_acceleration_structure(
            acceleration_structure.as_ref(),
            &descriptor,
            scratch_data.buffer.as_ref(),
            scratch_data.offset,
        );
    }
}

impl Drop for super::AccelerationStructureCommandEncoder<'_> {
    fn drop(&mut self) {
        self.raw.end_encoding();
    }
}

impl super::ComputeCommandEncoder<'_> {
    pub fn with<'p>(
        &'p mut self,
        pipeline: &'p super::ComputePipeline,
    ) -> super::ComputePipelineContext<'p> {
        self.raw.push_debug_group(&pipeline.name);
        self.raw.set_compute_pipeline_state(&pipeline.raw);
        if let Some(index) = pipeline.layout.sizes_buffer_slot {
            //TODO: get real sizes? shouldn't matter without bounds checks
            let runtime_sizes = [0u8; 8];
            self.raw.set_bytes(
                index as _,
                runtime_sizes.len() as _,
                runtime_sizes.as_ptr() as *const _,
            );
        }
        for (index, &size) in pipeline.wg_memory_sizes.iter().enumerate() {
            self.raw
                .set_threadgroup_memory_length(index as _, size as _);
        }

        super::ComputePipelineContext {
            encoder: &mut self.raw,
            wg_size: pipeline.wg_size,
            group_mappings: &pipeline.layout.group_mappings,
        }
    }
}

impl Drop for super::ComputeCommandEncoder<'_> {
    fn drop(&mut self) {
        self.raw.end_encoding();
    }
}

impl super::RenderCommandEncoder<'_> {
    pub fn set_scissor_rect(&mut self, rect: &crate::ScissorRect) {
        let scissor = metal::MTLScissorRect {
            x: rect.x as _,
            y: rect.y as _,
            width: rect.w as _,
            height: rect.h as _,
        };
        self.raw.set_scissor_rect(scissor);
    }

    pub fn with<'p>(
        &'p mut self,
        pipeline: &'p super::RenderPipeline,
    ) -> super::RenderPipelineContext<'p> {
        self.raw.push_debug_group(&pipeline.name);
        self.raw.set_render_pipeline_state(&pipeline.raw);
        if let Some(index) = pipeline.layout.sizes_buffer_slot {
            //TODO: get real sizes
            let runtime_sizes = [0u8; 8];
            self.raw.set_vertex_bytes(
                index as _,
                runtime_sizes.len() as _,
                runtime_sizes.as_ptr() as *const _,
            );
            self.raw.set_fragment_bytes(
                index as _,
                runtime_sizes.len() as _,
                runtime_sizes.as_ptr() as *const _,
            );
        }

        self.raw.set_front_facing_winding(pipeline.front_winding);
        self.raw.set_cull_mode(pipeline.cull_mode);
        self.raw.set_triangle_fill_mode(pipeline.triangle_fill_mode);
        self.raw.set_depth_clip_mode(pipeline.depth_clip_mode);
        if let Some((ref state, bias)) = pipeline.depth_stencil {
            self.raw.set_depth_stencil_state(state);
            self.raw
                .set_depth_bias(bias.constant as f32, bias.slope_scale, bias.clamp);
        }

        super::RenderPipelineContext {
            encoder: &mut self.raw,
            primitive_type: pipeline.primitive_type,
            group_mappings: &pipeline.layout.group_mappings,
        }
    }
}

impl Drop for super::RenderCommandEncoder<'_> {
    fn drop(&mut self) {
        self.raw.end_encoding();
    }
}

#[hidden_trait::expose]
impl crate::traits::PipelineEncoder for super::ComputePipelineContext<'_> {
    fn bind<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
        let info = &self.group_mappings[group as usize];

        data.fill(super::PipelineContext {
            cs_encoder: if info.visibility.contains(crate::ShaderVisibility::COMPUTE) {
                Some(self.encoder.as_ref())
            } else {
                None
            },
            vs_encoder: None,
            fs_encoder: None,
            targets: &info.targets,
        });
    }
}

#[hidden_trait::expose]
impl crate::traits::ComputePipelineEncoder for super::ComputePipelineContext<'_> {
    fn dispatch(&mut self, groups: [u32; 3]) {
        let raw_count = metal::MTLSize {
            width: groups[0] as u64,
            height: groups[1] as u64,
            depth: groups[2] as u64,
        };
        self.encoder.dispatch_thread_groups(raw_count, self.wg_size);
    }
}

impl Drop for super::ComputePipelineContext<'_> {
    fn drop(&mut self) {
        self.encoder.pop_debug_group();
    }
}

#[hidden_trait::expose]
impl crate::traits::PipelineEncoder for super::RenderPipelineContext<'_> {
    fn bind<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
        let info = &self.group_mappings[group as usize];

        data.fill(super::PipelineContext {
            cs_encoder: None,
            vs_encoder: if info.visibility.contains(crate::ShaderVisibility::VERTEX) {
                Some(self.encoder.as_ref())
            } else {
                None
            },
            fs_encoder: if info.visibility.contains(crate::ShaderVisibility::FRAGMENT) {
                Some(self.encoder.as_ref())
            } else {
                None
            },
            targets: &info.targets,
        });
    }
}

#[hidden_trait::expose]
impl crate::traits::RenderPipelineEncoder for super::RenderPipelineContext<'_> {
    type BufferPiece = crate::BufferPiece;

    fn set_scissor_rect(&mut self, rect: &crate::ScissorRect) {
        let scissor = metal::MTLScissorRect {
            x: rect.x as _,
            y: rect.y as _,
            width: rect.w as _,
            height: rect.h as _,
        };
        self.encoder.set_scissor_rect(scissor);
    }

    fn bind_vertex(&mut self, index: u32, vertex_buf: crate::BufferPiece) {
        self.encoder.set_vertex_buffer(
            index as u64,
            Some(vertex_buf.buffer.as_ref()),
            vertex_buf.offset,
        );
    }

    fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) {
        if first_instance != 0 {
            self.encoder.draw_primitives_instanced_base_instance(
                self.primitive_type,
                first_vertex as _,
                vertex_count as _,
                instance_count as _,
                first_instance as _,
            );
        } else if instance_count != 1 {
            self.encoder.draw_primitives_instanced(
                self.primitive_type,
                first_vertex as _,
                vertex_count as _,
                instance_count as _,
            );
        } else {
            self.encoder
                .draw_primitives(self.primitive_type, first_vertex as _, vertex_count as _);
        }
    }

    fn draw_indexed(
        &mut self,
        index_buf: crate::BufferPiece,
        index_type: crate::IndexType,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    ) {
        let raw_index_type = super::map_index_type(index_type);
        if base_vertex != 0 || start_instance != 0 {
            self.encoder
                .draw_indexed_primitives_instanced_base_instance(
                    self.primitive_type,
                    index_count as _,
                    raw_index_type,
                    index_buf.buffer.as_ref(),
                    index_buf.offset,
                    instance_count as _,
                    base_vertex as _,
                    start_instance as _,
                );
        } else if instance_count != 1 {
            self.encoder.draw_indexed_primitives_instanced(
                self.primitive_type,
                index_count as _,
                raw_index_type,
                index_buf.buffer.as_ref(),
                index_buf.offset,
                instance_count as _,
            );
        } else {
            self.encoder.draw_indexed_primitives(
                self.primitive_type,
                index_count as _,
                raw_index_type,
                index_buf.buffer.as_ref(),
                index_buf.offset,
            );
        }
    }

    fn draw_indirect(&mut self, indirect_buf: crate::BufferPiece) {
        self.encoder.draw_primitives_indirect(
            self.primitive_type,
            indirect_buf.buffer.as_ref(),
            indirect_buf.offset,
        );
    }

    fn draw_indexed_indirect(
        &mut self,
        index_buf: crate::BufferPiece,
        index_type: crate::IndexType,
        indirect_buf: crate::BufferPiece,
    ) {
        let raw_index_type = super::map_index_type(index_type);
        self.encoder.draw_indexed_primitives_indirect(
            self.primitive_type,
            raw_index_type,
            index_buf.buffer.as_ref(),
            index_buf.offset,
            indirect_buf.buffer.as_ref(),
            indirect_buf.offset,
        );
    }
}

impl Drop for super::RenderPipelineContext<'_> {
    fn drop(&mut self) {
        self.encoder.pop_debug_group();
    }
}

fn map_origin(origin: &[u32; 3]) -> metal::MTLOrigin {
    metal::MTLOrigin {
        x: origin[0] as u64,
        y: origin[1] as u64,
        z: origin[2] as u64,
    }
}

fn map_extent(extent: &crate::Extent) -> metal::MTLSize {
    metal::MTLSize {
        width: extent.width as u64,
        height: extent.height as u64,
        depth: extent.depth as u64,
    }
}

fn map_clear_color(color: crate::TextureColor) -> metal::MTLClearColor {
    match color {
        crate::TextureColor::TransparentBlack => metal::MTLClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            alpha: 0.0,
        },
        crate::TextureColor::OpaqueBlack => metal::MTLClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            alpha: 1.0,
        },
        crate::TextureColor::White => metal::MTLClearColor {
            red: 1.0,
            green: 1.0,
            blue: 1.0,
            alpha: 1.0,
        },
    }
}
