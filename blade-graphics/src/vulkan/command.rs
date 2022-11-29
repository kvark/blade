use ash::vk;

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
                .begin_command_buffer(self.buffers[0].raw, &vk_info)
                .unwrap()
        };
    }

    pub fn transfer(&mut self) -> super::TransferCommandEncoder {
        super::TransferCommandEncoder {
            raw: self.buffers[0].raw,
            device: &self.device,
        }
    }

    pub fn compute(&mut self) -> super::ComputeCommandEncoder {
        super::ComputeCommandEncoder {
            raw: self.buffers[0].raw,
            device: &self.device,
            plain_data: &mut self.plain_data,
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
    pub fn with<'b>(&'b mut self, pipeline: &super::ComputePipeline) -> super::PipelineEncoder<'b> {
        unsafe {
            self.device
                .cmd_bind_pipeline(self.raw, vk::PipelineBindPoint::COMPUTE, pipeline.raw)
        };
        super::PipelineEncoder {
            raw: self.raw,
            device: self.device,
            plain_data: self.plain_data,
        }
    }
}

impl super::PipelineEncoder<'_> {
    pub fn bind<D: crate::ShaderData>(&mut self, _group: u32, _data: &D) {
        unimplemented!()
    }

    pub fn dispatch(&mut self, _groups: [u32; 3]) {
        unimplemented!()
    }
}
