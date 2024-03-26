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
        match self
            .module
            .types
            .iter()
            .find(|&(_, ty)| ty.name.as_deref() == Some(struct_name))
        {
            Some((_, ty)) => match ty.inner {
                naga::TypeInner::Struct { members: _, span } => span,
                _ => panic!("Type '{struct_name}' is not a struct in the shader"),
            },
            None => panic!("Struct '{struct_name}' is not found in the shader"),
        }
    }

    pub fn check_struct_size<T>(&self) {
        use std::{any::type_name, mem::size_of};
        let name = type_name::<T>().rsplit("::").next().unwrap();
        assert_eq!(
            size_of::<T>(),
            self.get_struct_size(name) as usize,
            "Host struct '{name}' size doesn't match the shader"
        );
    }

    pub(crate) fn fill_vertex_locations(
        module: &mut naga::Module,
        ep: &naga::EntryPoint,
        fetch_states: &[crate::VertexFetchState],
    ) -> Vec<crate::VertexAttributeMapping> {
        if ep.stage != naga::ShaderStage::Vertex {
            assert!(fetch_states.is_empty());
            return Vec::new();
        }
        let mut attribute_mappings = Vec::new();
        for argument in ep.function.arguments.iter() {
            if argument.binding.is_some() {
                continue;
            }

            let arg_name = match argument.name {
                Some(ref name) => name.as_str(),
                None => "?",
            };
            log::debug!("Processing vertex argument: {}", arg_name);
            let mut ty = module.types[argument.ty].clone();
            let members = match ty.inner {
                naga::TypeInner::Struct {
                    ref mut members, ..
                } => members,
                ref other => {
                    log::error!("Unexpected type for {}: {:?}", arg_name, other);
                    continue;
                }
            };

            'member: for member in members.iter_mut() {
                let member_name = match member.name {
                    Some(ref name) => name.as_str(),
                    None => "?",
                };
                if let Some(ref binding) = member.binding {
                    log::warn!("Member '{}' alread has binding: {:?}", member_name, binding);
                    continue;
                }
                let binding = naga::Binding::Location {
                    location: attribute_mappings.len() as u32,
                    second_blend_source: false,
                    interpolation: None,
                    sampling: None,
                };
                for (buffer_index, vertex_fetch) in fetch_states.iter().enumerate() {
                    for (attribute_index, &(at_name, _)) in
                        vertex_fetch.layout.attributes.iter().enumerate()
                    {
                        if at_name == member_name {
                            member.binding = Some(binding);
                            attribute_mappings.push(crate::VertexAttributeMapping {
                                buffer_index,
                                attribute_index,
                            });
                            continue 'member;
                        }
                    }
                }
                assert_ne!(
                    member.binding, None,
                    "Field {} is not covered by the vertex fetch layouts!",
                    member_name
                );
            }
        }
        attribute_mappings
    }
}
