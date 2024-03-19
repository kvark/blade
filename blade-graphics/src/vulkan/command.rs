use ash::vk;
use std::str;

impl super::CrashHandler {
    fn add_marker(&mut self, marker: &str) -> u32 {
        if self.next_offset < self.raw_string.len() {
            self.raw_string[self.next_offset] = b'|';
            self.next_offset += 1;
        }
        let len = marker.as_bytes().len().min(self.raw_string.len());
        if self.next_offset + len > self.raw_string.len() {
            self.next_offset = 0;
        }
        let start = self.next_offset;
        self.next_offset += len;
        let end = self.next_offset;
        self.raw_string[start..end].copy_from_slice(&marker.as_bytes()[..len]);
        start as u32 | (end << 16) as u32
    }

    pub(super) fn extract(&self, id: u32) -> (&str, &str) {
        let start = id as usize & 0xFFFF;
        let end = (id >> 16) as usize;
        let history = str::from_utf8(&self.raw_string[..start]).unwrap_or_default();
        let marker = str::from_utf8(&self.raw_string[start..end]).unwrap();
        (history, marker)
    }
}

impl super::PipelineContext<'_> {
    #[inline]
    fn write<T>(&mut self, index: u32, value: T) {
        let offset = self.template_offsets[index as usize];
        unsafe {
            std::ptr::write(
                self.update_data.as_mut_ptr().offset(offset as isize) as *mut T,
                value,
            )
        };
    }

    #[inline]
    fn write_array<I: Iterator>(&mut self, index: u32, iter: I) {
        let base_offset = self.template_offsets[index as usize];
        let base_ptr =
            unsafe { self.update_data.as_mut_ptr().offset(base_offset as isize) as *mut I::Item };
        for (i, value) in iter.enumerate() {
            unsafe { std::ptr::write(base_ptr.add(i), value) };
        }
    }
}

impl<T: bytemuck::Pod> crate::ShaderBindable for T {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        ctx.write(index, *self);
    }
}
impl crate::ShaderBindable for super::TextureView {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        ctx.write(
            index,
            vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: self.raw,
                image_layout: vk::ImageLayout::GENERAL,
            },
        );
    }
}
impl<'a, const N: crate::ResourceIndex> crate::ShaderBindable for &'a crate::TextureArray<N> {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        ctx.write_array(
            index,
            self.data.iter().map(|view| vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: view.raw,
                image_layout: vk::ImageLayout::GENERAL,
            }),
        );
    }
}
impl crate::ShaderBindable for super::Sampler {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        ctx.write(
            index,
            vk::DescriptorImageInfo {
                sampler: self.raw,
                image_view: vk::ImageView::null(),
                image_layout: vk::ImageLayout::UNDEFINED,
            },
        );
    }
}
impl crate::ShaderBindable for crate::BufferPiece {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        ctx.write(
            index,
            vk::DescriptorBufferInfo {
                buffer: self.buffer.raw,
                offset: self.offset,
                range: vk::WHOLE_SIZE,
            },
        );
    }
}
impl<'a, const N: crate::ResourceIndex> crate::ShaderBindable for &'a crate::BufferArray<N> {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        ctx.write_array(
            index,
            self.data.iter().map(|piece| vk::DescriptorBufferInfo {
                buffer: piece.buffer.raw,
                offset: piece.offset,
                range: vk::WHOLE_SIZE,
            }),
        );
    }
}
impl crate::ShaderBindable for super::AccelerationStructure {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        ctx.write(index, self.raw);
    }
}

impl crate::TexturePiece {
    fn subresource_layers(&self, aspects: crate::TexelAspects) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: super::map_aspects(aspects),
            mip_level: self.mip_level,
            base_array_layer: self.array_layer,
            layer_count: 1,
        }
    }
}

fn map_origin(origin: &[u32; 3]) -> vk::Offset3D {
    vk::Offset3D {
        x: origin[0] as i32,
        y: origin[1] as i32,
        z: origin[2] as i32,
    }
}

fn make_buffer_image_copy(
    buffer: &crate::BufferPiece,
    bytes_per_row: u32,
    texture: &crate::TexturePiece,
    size: &crate::Extent,
) -> vk::BufferImageCopy {
    let block_info = texture.texture.format.block_info();
    vk::BufferImageCopy {
        buffer_offset: buffer.offset,
        buffer_row_length: block_info.dimensions.0 as u32
            * (bytes_per_row / block_info.size as u32),
        buffer_image_height: 0,
        image_subresource: texture.subresource_layers(texture.texture.format.aspects()),
        image_offset: map_origin(&texture.origin),
        image_extent: super::map_extent_3d(size),
    }
}

fn map_render_target(rt: &crate::RenderTarget) -> vk::RenderingAttachmentInfo {
    let mut builder = vk::RenderingAttachmentInfo::builder()
        .image_view(rt.view.raw)
        .image_layout(vk::ImageLayout::GENERAL)
        .load_op(vk::AttachmentLoadOp::LOAD);

    if let crate::InitOp::Clear(color) = rt.init_op {
        let cv = if rt.view.aspects.contains(crate::TexelAspects::COLOR) {
            vk::ClearValue {
                color: match color {
                    crate::TextureColor::TransparentBlack => vk::ClearColorValue::default(),
                    crate::TextureColor::OpaqueBlack => vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                    crate::TextureColor::White => vk::ClearColorValue { float32: [1.0; 4] },
                },
            }
        } else {
            vk::ClearValue {
                depth_stencil: match color {
                    crate::TextureColor::TransparentBlack => vk::ClearDepthStencilValue::default(),
                    crate::TextureColor::OpaqueBlack => vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                    crate::TextureColor::White => vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: !0,
                    },
                },
            }
        };
        builder = builder.load_op(vk::AttachmentLoadOp::CLEAR).clear_value(cv);
    }

    builder.build()
}

impl super::CommandEncoder {
    pub fn mark(&mut self, marker: &str) {
        if let Some(ref mut ch) = self.crash_handler {
            let id = ch.add_marker(marker);
            unsafe {
                (self
                    .device
                    .buffer_marker
                    .as_ref()
                    .unwrap()
                    .cmd_write_buffer_marker_amd)(
                    self.buffers[0].raw,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    ch.marker_buf.raw,
                    0,
                    id,
                );
            }
        }
    }

    pub fn start(&mut self) {
        self.buffers.rotate_left(1);

        let vk_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        unsafe {
            self.device
                .core
                .reset_descriptor_pool(
                    self.buffers[0].descriptor_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .unwrap();
            self.device
                .core
                .begin_command_buffer(self.buffers[0].raw, &vk_info)
                .unwrap();
        }
    }

    pub(super) fn finish(&mut self) -> vk::CommandBuffer {
        self.barrier();
        self.mark("finish");
        let raw = self.buffers[0].raw;
        unsafe { self.device.core.end_command_buffer(raw).unwrap() }
        raw
    }

    fn barrier(&mut self) {
        let wa = &self.device.workarounds;
        let barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | wa.extra_sync_src_access)
            .dst_access_mask(
                vk::AccessFlags::MEMORY_READ
                    | vk::AccessFlags::MEMORY_WRITE
                    | wa.extra_sync_dst_access,
            )
            .build();
        unsafe {
            self.device.core.cmd_pipeline_barrier(
                self.buffers[0].raw,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    pub fn init_texture(&mut self, texture: super::Texture) {
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(texture.raw)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: super::map_aspects(texture.format.aspects()),
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            })
            .build();
        unsafe {
            self.device.core.cmd_pipeline_barrier(
                self.buffers[0].raw,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    pub fn present(&mut self, frame: super::Frame) {
        assert_eq!(self.present, None);
        let wa = &self.device.workarounds;
        self.present = Some(super::Presentation {
            image_index: frame.image_index,
            acquire_semaphore: frame.acquire_semaphore,
        });

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(frame.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | wa.extra_sync_src_access)
            .build();
        unsafe {
            self.device.core.cmd_pipeline_barrier(
                self.buffers[0].raw,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    pub fn transfer(&mut self) -> super::TransferCommandEncoder {
        self.barrier();
        self.mark("pass/transfer");
        super::TransferCommandEncoder {
            raw: self.buffers[0].raw,
            device: &self.device,
        }
    }

    pub fn acceleration_structure(&mut self) -> super::AccelerationStructureCommandEncoder {
        self.barrier();
        self.mark("pass/acc-struct");
        super::AccelerationStructureCommandEncoder {
            raw: self.buffers[0].raw,
            device: &self.device,
        }
    }

    pub fn compute(&mut self) -> super::ComputeCommandEncoder {
        self.barrier();
        self.mark("pass/compute");
        super::ComputeCommandEncoder {
            cmd_buf: self.buffers[0],
            device: &self.device,
            update_data: &mut self.update_data,
        }
    }

    pub fn render(&mut self, targets: crate::RenderTargetSet) -> super::RenderCommandEncoder {
        self.barrier();
        self.mark("pass/render");

        let mut target_size = [0u16; 2];
        let mut color_attachments = Vec::with_capacity(targets.colors.len());
        let depth_stencil_attachment;
        for rt in targets.colors {
            target_size = rt.view.target_size;
            color_attachments.push(map_render_target(rt));
        }

        let mut rendering_info = vk::RenderingInfoKHR::builder()
            .layer_count(1)
            .color_attachments(&color_attachments);
        if let Some(rt) = targets.depth_stencil {
            target_size = rt.view.target_size;
            depth_stencil_attachment = map_render_target(&rt);
            if rt.view.aspects.contains(crate::TexelAspects::DEPTH) {
                rendering_info = rendering_info.depth_attachment(&depth_stencil_attachment);
            }
            if rt.view.aspects.contains(crate::TexelAspects::STENCIL) {
                rendering_info = rendering_info.stencil_attachment(&depth_stencil_attachment);
            }
        }

        let render_area = vk::Rect2D {
            offset: Default::default(),
            extent: vk::Extent2D {
                width: target_size[0] as u32,
                height: target_size[1] as u32,
            },
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: target_size[1] as f32,
            width: target_size[0] as f32,
            height: -(target_size[1] as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        rendering_info = rendering_info.render_area(render_area);

        let cmd_buf = self.buffers[0];
        unsafe {
            self.device
                .core
                .cmd_set_viewport(cmd_buf.raw, 0, &[viewport]);
            self.device
                .core
                .cmd_set_scissor(cmd_buf.raw, 0, &[render_area]);
            self.device
                .dynamic_rendering
                .cmd_begin_rendering(cmd_buf.raw, &rendering_info);
        };

        super::RenderCommandEncoder {
            cmd_buf,
            device: &self.device,
            update_data: &mut self.update_data,
        }
    }

    pub(super) fn check_gpu_crash<T>(&self, ret: Result<T, vk::Result>) -> Option<T> {
        match ret {
            Ok(value) => Some(value),
            Err(vk::Result::ERROR_DEVICE_LOST) => match self.crash_handler {
                Some(ref ch) => {
                    let last_id = unsafe { *(ch.marker_buf.data() as *mut u32) };
                    if last_id != 0 {
                        let (history, last_marker) = ch.extract(last_id);
                        log::error!("Last GPU executed marker is '{last_marker}'");
                        log::info!("Marker history: {}", history);
                    }
                    panic!("GPU has crashed in {}", ch.name);
                }
                None => {
                    panic!("GPU has crashed, and no debug information is available.");
                }
            },
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                log::warn!("GPU frame is out of date");
                None
            }
            Err(other) => panic!("GPU error {}", other),
        }
    }
}

#[hidden_trait::expose]
impl crate::traits::TransferEncoder for super::TransferCommandEncoder<'_> {
    fn fill_buffer(&mut self, dst: crate::BufferPiece, size: u64, value: u8) {
        let value_u32 = (value as u32) * 0x1010101;
        unsafe {
            self.device
                .core
                .cmd_fill_buffer(self.raw, dst.buffer.raw, dst.offset, size, value_u32)
        };
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: crate::BufferPiece,
        dst: crate::BufferPiece,
        size: u64,
    ) {
        let copy = vk::BufferCopy {
            src_offset: src.offset,
            dst_offset: dst.offset,
            size,
        };
        unsafe {
            self.device
                .core
                .cmd_copy_buffer(self.raw, src.buffer.raw, dst.buffer.raw, &[copy])
        };
    }

    fn copy_texture_to_texture(
        &mut self,
        src: crate::TexturePiece,
        dst: crate::TexturePiece,
        size: crate::Extent,
    ) {
        let copy = vk::ImageCopy {
            src_subresource: src.subresource_layers(crate::TexelAspects::all()),
            src_offset: map_origin(&src.origin),
            dst_subresource: dst.subresource_layers(crate::TexelAspects::all()),
            dst_offset: map_origin(&dst.origin),
            extent: super::map_extent_3d(&size),
        };
        unsafe {
            self.device.core.cmd_copy_image(
                self.raw,
                src.texture.raw,
                vk::ImageLayout::GENERAL,
                dst.texture.raw,
                vk::ImageLayout::GENERAL,
                &[copy],
            )
        };
    }

    fn copy_buffer_to_texture(
        &mut self,
        src: crate::BufferPiece,
        bytes_per_row: u32,
        dst: crate::TexturePiece,
        size: crate::Extent,
    ) {
        let copy = make_buffer_image_copy(&src, bytes_per_row, &dst, &size);
        unsafe {
            self.device.core.cmd_copy_buffer_to_image(
                self.raw,
                src.buffer.raw,
                dst.texture.raw,
                vk::ImageLayout::GENERAL,
                &[copy],
            )
        };
    }

    fn copy_texture_to_buffer(
        &mut self,
        src: crate::TexturePiece,
        dst: crate::BufferPiece,
        bytes_per_row: u32,
        size: crate::Extent,
    ) {
        let copy = make_buffer_image_copy(&dst, bytes_per_row, &src, &size);
        unsafe {
            self.device.core.cmd_copy_image_to_buffer(
                self.raw,
                src.texture.raw,
                vk::ImageLayout::GENERAL,
                dst.buffer.raw,
                &[copy],
            )
        };
    }
}

#[hidden_trait::expose]
impl crate::traits::AccelerationStructureEncoder
    for super::AccelerationStructureCommandEncoder<'_>
{
    fn build_bottom_level(
        &mut self,
        acceleration_structure: super::AccelerationStructure,
        meshes: &[crate::AccelerationStructureMesh],
        scratch_data: crate::BufferPiece,
    ) {
        let mut blas_input = self.device.map_acceleration_structure_meshes(meshes);
        blas_input.build_info.dst_acceleration_structure = acceleration_structure.raw;
        let scratch_address = self.device.get_device_address(&scratch_data);
        assert!(
            scratch_address & 0xFF == 0,
            "BLAS scratch address {scratch_address} is not aligned"
        );
        blas_input.build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_address,
        };

        let rt = self.device.ray_tracing.as_ref().unwrap();
        unsafe {
            rt.acceleration_structure.cmd_build_acceleration_structures(
                self.raw,
                &[blas_input.build_info],
                &[&blas_input.build_range_infos],
            );
        }
    }

    fn build_top_level(
        &mut self,
        acceleration_structure: super::AccelerationStructure,
        _bottom_level: &[super::AccelerationStructure],
        instance_count: u32,
        instance_data: crate::BufferPiece,
        scratch_data: crate::BufferPiece,
    ) {
        let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: instance_count,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        };
        let geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: self.device.get_device_address(&instance_data),
                    })
                    .build(),
            })
            .build();
        let geometries = [geometry];
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: self.device.get_device_address(&scratch_data),
            })
            .dst_acceleration_structure(acceleration_structure.raw)
            .build();

        let rt = self.device.ray_tracing.as_ref().unwrap();
        unsafe {
            rt.acceleration_structure.cmd_build_acceleration_structures(
                self.raw,
                &[build_info],
                &[&[build_range_info]],
            );
        }
    }
}

impl<'a> super::ComputeCommandEncoder<'a> {
    pub fn with<'b, 'p>(
        &'b mut self,
        pipeline: &'p super::ComputePipeline,
    ) -> super::PipelineEncoder<'b, 'p> {
        super::PipelineEncoder {
            cmd_buf: self.cmd_buf,
            layout: &pipeline.layout,
            bind_point: vk::PipelineBindPoint::COMPUTE,
            device: self.device,
            update_data: self.update_data,
        }
        .init(pipeline.raw)
    }
}

impl<'a> super::RenderCommandEncoder<'a> {
    pub fn set_scissor_rect(&mut self, rect: &crate::ScissorRect) {
        let vk_scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: rect.x as i32,
                y: rect.y as i32,
            },
            extent: vk::Extent2D {
                width: rect.w,
                height: rect.h,
            },
        };
        unsafe {
            self.device
                .core
                .cmd_set_scissor(self.cmd_buf.raw, 0, &[vk_scissor])
        };
    }

    pub fn with<'b, 'p>(
        &'b mut self,
        pipeline: &'p super::RenderPipeline,
    ) -> super::PipelineEncoder<'b, 'p> {
        super::PipelineEncoder {
            cmd_buf: self.cmd_buf,
            layout: &pipeline.layout,
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            device: self.device,
            update_data: self.update_data,
        }
        .init(pipeline.raw)
    }
}

impl Drop for super::RenderCommandEncoder<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .dynamic_rendering
                .cmd_end_rendering(self.cmd_buf.raw)
        };
    }
}

impl super::PipelineEncoder<'_, '_> {
    fn init(self, raw_pipeline: vk::Pipeline) -> Self {
        unsafe {
            self.device
                .core
                .cmd_bind_pipeline(self.cmd_buf.raw, self.bind_point, raw_pipeline)
        };
        self
    }
}

#[hidden_trait::expose]
impl crate::traits::PipelineEncoder for super::PipelineEncoder<'_, '_> {
    fn bind<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
        let dsl = &self.layout.descriptor_set_layouts[group as usize];
        self.update_data.clear();
        self.update_data.resize(dsl.template_size as usize, 0);
        data.fill(super::PipelineContext {
            update_data: self.update_data.as_mut_slice(),
            template_offsets: &dsl.template_offsets,
        });

        let descriptor_set_layouts = [dsl.raw];
        let descriptor_set_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.cmd_buf.descriptor_pool)
            .set_layouts(&descriptor_set_layouts);
        unsafe {
            let sets = self
                .device
                .core
                .allocate_descriptor_sets(&descriptor_set_info)
                .unwrap();
            self.device.core.update_descriptor_set_with_template(
                sets[0],
                dsl.update_template,
                self.update_data.as_ptr() as *const _,
            );
            self.device.core.cmd_bind_descriptor_sets(
                self.cmd_buf.raw,
                self.bind_point,
                self.layout.raw,
                group,
                &sets,
                &[],
            );
        }
    }
}

#[hidden_trait::expose]
impl crate::traits::ComputePipelineEncoder for super::PipelineEncoder<'_, '_> {
    fn dispatch(&mut self, groups: [u32; 3]) {
        unsafe {
            self.device
                .core
                .cmd_dispatch(self.cmd_buf.raw, groups[0], groups[1], groups[2])
        };
    }
}

#[hidden_trait::expose]
impl crate::traits::RenderPipelineEncoder for super::PipelineEncoder<'_, '_> {
    fn set_scissor_rect(&mut self, rect: &crate::ScissorRect) {
        let vk_scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: rect.x as i32,
                y: rect.y as i32,
            },
            extent: vk::Extent2D {
                width: rect.w,
                height: rect.h,
            },
        };
        unsafe {
            self.device
                .core
                .cmd_set_scissor(self.cmd_buf.raw, 0, &[vk_scissor])
        };
    }

    fn draw(
        &mut self,
        start_vertex: u32,
        vertex_count: u32,
        start_instance: u32,
        instance_count: u32,
    ) {
        unsafe {
            self.device.core.cmd_draw(
                self.cmd_buf.raw,
                vertex_count,
                instance_count,
                start_vertex,
                start_instance,
            );
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
        unsafe {
            self.device.core.cmd_bind_index_buffer(
                self.cmd_buf.raw,
                index_buf.buffer.raw,
                index_buf.offset,
                raw_index_type,
            );
            self.device.core.cmd_draw_indexed(
                self.cmd_buf.raw,
                index_count,
                instance_count,
                0,
                base_vertex,
                start_instance,
            );
        }
    }

    fn draw_indirect(&mut self, indirect_buf: crate::BufferPiece) {
        unsafe {
            self.device.core.cmd_draw_indirect(
                self.cmd_buf.raw,
                indirect_buf.buffer.raw,
                indirect_buf.offset,
                1,
                0,
            );
        }
    }

    fn draw_indexed_indirect(
        &mut self,
        index_buf: crate::BufferPiece,
        index_type: crate::IndexType,
        indirect_buf: crate::BufferPiece,
    ) {
        let raw_index_type = super::map_index_type(index_type);
        unsafe {
            self.device.core.cmd_bind_index_buffer(
                self.cmd_buf.raw,
                index_buf.buffer.raw,
                index_buf.offset,
                raw_index_type,
            );
            self.device.core.cmd_draw_indexed_indirect(
                self.cmd_buf.raw,
                indirect_buf.buffer.raw,
                indirect_buf.offset,
                1,
                0,
            );
        }
    }
}
