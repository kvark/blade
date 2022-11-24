impl super::Context {
    pub fn create_buffer(&self, _desc: crate::BufferDesc) -> super::Buffer {
        unimplemented!()
    }

    pub fn create_texture(&self, _desc: crate::TextureDesc) -> super::Texture {
        unimplemented!()
    }

    pub fn create_texture_view(&self, _desc: crate::TextureViewDesc) -> super::TextureView {
        unimplemented!()
    }

    pub fn create_sampler(&self, _desc: crate::SamplerDesc) -> super::Sampler {
        unimplemented!()
    }
}
