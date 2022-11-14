impl super::Context {
    pub fn create_shader(&self, desc: super::ShaderDesc) -> super::Shader {
        let module = naga::front::wgsl::parse_str(desc.source).unwrap();

        let caps = naga::valid::Capabilities::empty();
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
            .validate(&module)
            .unwrap();

        super::Shader {
            module,
            info,
        }
    }
}
