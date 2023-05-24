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
    pub fn try_create_shader(
        &self,
        desc: super::ShaderDesc,
    ) -> Result<super::Shader, &'static str> {
        let module = naga::front::wgsl::parse_str(desc.source).map_err(|e| {
            e.emit_to_stderr_with_path(desc.source, "");
            "compilation failed"
        })?;

        let device_caps = self.capabilities();

        // Bindings are set up at pipeline creation, ignore here
        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        let mut caps = naga::valid::Capabilities::empty();
        caps.set(
            naga::valid::Capabilities::RAY_QUERY | naga::valid::Capabilities::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            !device_caps.ray_query.is_empty(),
        );
        let info = naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .map_err(|e| {
                crate::util::emit_annotated_error(&e, "", desc.source);
                crate::util::print_err(&e);
                "validation failed"
            })?;

        Ok(super::Shader { module, info })
    }

    pub fn create_shader(&self, desc: super::ShaderDesc) -> super::Shader {
        self.try_create_shader(desc).unwrap()
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
            .find(|&(_, ty)| ty.name.as_deref() == Some(struct_name))
            .expect("Struct type not found");
        match ty.inner {
            naga::TypeInner::Struct { members: _, span } => span,
            _ => panic!("Type is not a struct"),
        }
    }
}
