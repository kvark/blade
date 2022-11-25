use std::marker::PhantomData;

impl super::CommandEncoder {
    pub fn start(&mut self) {}

    pub fn with_transfers(&mut self) -> super::TransferCommandEncoder {
        super::TransferCommandEncoder {
            phantom: PhantomData,
        }
    }

    pub fn with_pipeline(
        &mut self,
        _pipeline: &super::ComputePipeline,
    ) -> super::ComputePipelineContext {
        super::ComputePipelineContext {
            phantom: PhantomData,
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

impl super::ComputePipelineContext<'_> {
    pub fn bind_data<D: crate::ShaderData>(&mut self, _group: u32, _data: &D) {
        unimplemented!()
    }

    pub fn dispatch(&mut self, _groups: [u32; 3]) {
        unimplemented!()
    }
}
