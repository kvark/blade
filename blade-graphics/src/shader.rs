impl From<naga::ShaderStage> for super::ShaderVisibility {
    fn from(stage: naga::ShaderStage) -> Self {
        match stage {
            naga::ShaderStage::Compute => Self::COMPUTE,
            naga::ShaderStage::Vertex => Self::VERTEX,
            naga::ShaderStage::Fragment => Self::FRAGMENT,
        }
    }
}

impl super::Context {
    pub fn create_shader(&self, desc: super::ShaderDesc) -> super::Shader {
        let module = match naga::front::wgsl::parse_str(desc.source) {
            Ok(module) => module,
            Err(ref e) => {
                e.emit_to_stderr_with_path(desc.source, "");
                panic!("Shader compilation failed");
            }
        };

        // Bindings are set up at pipeline creation, ignore here
        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        let caps = naga::valid::Capabilities::empty();
        let info = naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .unwrap_or_else(|e| {
                crate::util::emit_annotated_error(&e, "", desc.source);
                crate::util::print_err(&e);
                panic!("Shader validation failed");
            });

        super::Shader { module, info }
    }
}

impl super::Shader {
    pub fn at<'a>(&'a self, entry_point: &'a str) -> super::ShaderFunction<'a> {
        super::ShaderFunction {
            shader: self,
            entry_point,
        }
    }

    pub fn get_struct_size(&self, struct_name: &str) -> u32 {
        let (_, ty) = self
            .module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_ref().map(|s| s.as_str()) == Some(struct_name))
            .expect("Struct type not found");
        match ty.inner {
            naga::TypeInner::Struct { members: _, span } => span,
            _ => panic!("Type is not a struct"),
        }
    }
}
