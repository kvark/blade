use hal::CommandEncoder as _;

impl super::CommandEncoder {
    pub fn begin(&mut self) {
        unsafe {
            self.raw.begin_encoding(None).unwrap()
        };
    }

    pub fn submit(&mut self) {
        //TODO
    }

    pub fn with_pipeline(&mut self, pipeline: &super::RenderPipeline) -> super::RenderPipelineCommandEncoder {
        super::RenderPipelineCommandEncoder {
            raw: &mut self.raw,
        }
    }
}

impl<'enc> super::RenderPipelineCommandEncoder<'enc> {
    pub fn bind_data<'data, D: super::ShaderData<'data>>(&mut self, group: u32, data: &D) {
        unimplemented!()
    }

    pub fn draw(&mut self, start_vertex: u32, vertex_count: u32, start_instance: u32, instance_count: u32) {
        unsafe {
            self.raw.draw(start_vertex, vertex_count, start_instance, instance_count);
        }
    }
}
