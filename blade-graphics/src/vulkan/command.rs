use ash::vk;

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

impl crate::TexturePiece {
    fn subresource_layers(&self, aspects: super::FormatAspects) -> vk::ImageSubresourceLayers {
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
    let format_info = super::describe_format(texture.texture.format);
    vk::BufferImageCopy {
        buffer_offset: buffer.offset,
        buffer_row_length: format_info.block.width as u32
            * (bytes_per_row / format_info.block.bytes as u32),
        buffer_image_height: 0,
        image_subresource: texture.subresource_layers(format_info.aspects),
        image_offset: map_origin(&texture.origin),
        image_extent: super::map_extent_3d(&size),
    }
}

fn map_render_target(rt: &crate::RenderTarget) -> vk::RenderingAttachmentInfo {
    let mut builder = vk::RenderingAttachmentInfo::builder()
        .image_view(rt.view.raw)
        .image_layout(vk::ImageLayout::GENERAL)
        .load_op(vk::AttachmentLoadOp::LOAD);

    if let crate::InitOp::Clear(color) = rt.init_op {
        let cv = if rt.view.aspects.contains(super::FormatAspects::COLOR) {
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

fn map_index_type(index_type: crate::IndexType) -> vk::IndexType {
    match index_type {
        crate::IndexType::U16 => vk::IndexType::UINT16,
        crate::IndexType::U32 => vk::IndexType::UINT32,
    }
}

impl super::CommandEncoder {
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
        let raw = self.buffers[0].raw;
        unsafe { self.device.core.end_command_buffer(raw).unwrap() }
        raw
    }

    fn barrier(&mut self) {
        //TODO: figure out why TRANSFER_WRITE is not covered by MEMORY_WRITE
        let barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::MEMORY_READ
                    | vk::AccessFlags::MEMORY_WRITE
                    | vk::AccessFlags::TRANSFER_READ,
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
        let format_info = super::describe_format(texture.format);
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(texture.raw)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: super::map_aspects(format_info.aspects),
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
        self.present = Some(super::Presentation {
            image_index: frame.image_index,
            acquire_semaphore: frame.acquire_semaphore,
        });

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(frame.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
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

    pub fn transfer(&mut self) -> super::TransferCommandEncoder {
        self.barrier();
        super::TransferCommandEncoder {
            raw: self.buffers[0].raw,
            device: &self.device,
        }
    }

    pub fn compute(&mut self) -> super::ComputeCommandEncoder {
        self.barrier();
        super::ComputeCommandEncoder {
            cmd_buf: self.buffers[0],
            device: &self.device,
            update_data: &mut self.update_data,
        }
    }

    pub fn render(&mut self, targets: crate::RenderTargetSet) -> super::RenderCommandEncoder {
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
            if rt.view.aspects.contains(super::FormatAspects::DEPTH) {
                rendering_info = rendering_info.depth_attachment(&depth_stencil_attachment);
            }
            if rt.view.aspects.contains(super::FormatAspects::STENCIL) {
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
}

impl super::TransferCommandEncoder<'_> {
    pub fn copy_buffer_to_buffer(
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

    pub fn copy_texture_to_texture(
        &mut self,
        src: crate::TexturePiece,
        dst: crate::TexturePiece,
        size: crate::Extent,
    ) {
        let copy = vk::ImageCopy {
            src_subresource: src.subresource_layers(super::FormatAspects::all()),
            src_offset: map_origin(&src.origin),
            dst_subresource: dst.subresource_layers(super::FormatAspects::all()),
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

    pub fn copy_buffer_to_texture(
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

    pub fn copy_texture_to_buffer(
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

    pub fn bind<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
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

    pub fn dispatch(&mut self, groups: [u32; 3]) {
        unsafe {
            self.device
                .core
                .cmd_dispatch(self.cmd_buf.raw, groups[0], groups[1], groups[2])
        };
    }

    //TODO: reconsider exposing this
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

    pub fn draw(
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

    pub fn draw_indexed(
        &mut self,
        index_buf: crate::BufferPiece,
        index_type: crate::IndexType,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    ) {
        let raw_index_type = map_index_type(index_type);
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
}
