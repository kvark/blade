use ash::vk;

struct ShaderDataEncoder<'a> {
    update_data: &'a mut [u8],
    template_offsets: &'a [u32],
}
impl ShaderDataEncoder<'_> {
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
impl crate::ShaderDataEncoder for ShaderDataEncoder<'_> {
    fn set_buffer(&mut self, index: u32, piece: crate::BufferPiece) {
        self.write(
            index,
            vk::DescriptorBufferInfo {
                buffer: piece.buffer.raw,
                offset: piece.offset,
                range: vk::WHOLE_SIZE,
            },
        );
    }
    fn set_texture(&mut self, index: u32, view: super::TextureView) {
        self.write(
            index,
            vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: view.raw,
                image_layout: vk::ImageLayout::GENERAL,
            },
        );
    }
    fn set_sampler(&mut self, index: u32, sampler: super::Sampler) {
        self.write(
            index,
            vk::DescriptorImageInfo {
                sampler: sampler.raw,
                image_view: vk::ImageView::null(),
                image_layout: vk::ImageLayout::UNDEFINED,
            },
        );
    }
    fn set_plain<P: bytemuck::Pod>(&mut self, index: u32, data: P) {
        self.write(index, data);
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

impl super::CommandEncoder {
    pub fn start(&mut self) {
        self.buffers.rotate_left(1);
        let vk_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        unsafe {
            self.device
                .reset_descriptor_pool(
                    self.buffers[0].descriptor_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .unwrap();
            self.device
                .begin_command_buffer(self.buffers[0].raw, &vk_info)
                .unwrap();
        }
    }

    pub fn transfer(&mut self) -> super::TransferCommandEncoder {
        super::TransferCommandEncoder {
            raw: self.buffers[0].raw,
            device: &self.device,
        }
    }

    pub fn compute(&mut self) -> super::ComputeCommandEncoder {
        super::ComputeCommandEncoder {
            cmd_buf: self.buffers[0],
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
            self.device.cmd_copy_image(
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
            self.device.cmd_copy_buffer_to_image(
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
            self.device.cmd_copy_image_to_buffer(
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
        unsafe {
            self.device.cmd_bind_pipeline(
                self.cmd_buf.raw,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.raw,
            )
        };
        super::PipelineEncoder {
            cmd_buf: self.cmd_buf,
            layout: &pipeline.layout,
            device: self.device,
            update_data: self.update_data,
        }
    }
}

impl super::PipelineEncoder<'_, '_> {
    pub fn bind<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
        let dsl = &self.layout.descriptor_set_layouts[group as usize];
        self.update_data.clear();
        self.update_data.resize(dsl.template_size as usize, 0);
        data.fill(ShaderDataEncoder {
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
                .allocate_descriptor_sets(&descriptor_set_info)
                .unwrap();
            self.device.update_descriptor_set_with_template(
                sets[0],
                dsl.update_template,
                self.update_data.as_ptr() as *const _,
            );
            self.device.cmd_bind_descriptor_sets(
                self.cmd_buf.raw,
                vk::PipelineBindPoint::COMPUTE,
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
                .cmd_dispatch(self.cmd_buf.raw, groups[0], groups[1], groups[2])
        };
    }
}
