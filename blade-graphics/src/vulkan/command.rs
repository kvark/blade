use ash::vk;

impl super::CommandEncoder {
    pub fn start(&mut self) {}

    pub fn transfer(&mut self) -> super::TransferCommandEncoder {
        super::TransferCommandEncoder {
            raw: self.raw,
            device: &self.device,
        }
    }

    pub fn compute(&mut self) -> super::ComputeCommandEncoder {
        super::ComputeCommandEncoder {
            raw: self.raw,
            device: &self.device,
            plain_data: &mut self.plain_data,
        }
    }
}

impl super::TransferCommandEncoder<'_> {
    pub fn copy_buffer_to_buffer(
        &mut self,
        _src: crate::BufferPiece,
        _dst: crate::BufferPiece,
        _size: u64,
    ) {
        unimplemented!()
    }
    pub fn copy_texture_to_texture(
        &mut self,
        _src: crate::TexturePiece,
        _dst: crate::TexturePiece,
        _size: crate::Extent,
    ) {
        unimplemented!()
    }
    pub fn copy_buffer_to_texture(
        &mut self,
        _src: crate::BufferPiece,
        _bytes_per_row: u32,
        _dst: crate::TexturePiece,
        _size: crate::Extent,
    ) {
        unimplemented!()
    }
    pub fn copy_texture_to_buffer(
        &mut self,
        _src: crate::TexturePiece,
        _dst: crate::BufferPiece,
        _bytes_per_row: u32,
        _size: crate::Extent,
    ) {
        unimplemented!()
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
