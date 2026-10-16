use once_cell::sync::Lazy;
use std::borrow::Cow;

impl From<naga::ShaderStage> for super::ShaderVisibility {
    fn from(stage: naga::ShaderStage) -> Self {
        match stage {
            naga::ShaderStage::Compute => Self::COMPUTE,
            naga::ShaderStage::Vertex => Self::VERTEX,
            naga::ShaderStage::Fragment => Self::FRAGMENT,
            _ => Self::empty(),
        }
    }
}

fn validation_capabilities(device_caps: &super::Capabilities) -> naga::valid::Capabilities {
    // Naga's default is the set it considers universally available:
    // `MULTISAMPLED_SHADING` and `CUBE_ARRAY_TEXTURES`. We support both
    // (`MultisampleState`, `ViewDimension::CubeArray`), so start there
    // rather than from nothing, and pick up future additions to it.
    let mut caps = naga::valid::Capabilities::default();
    caps.set(
        naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY
            | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
            | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING
            | naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING,
        device_caps.binding_array,
    );
    caps.set(
        naga::valid::Capabilities::RAY_QUERY
            | naga::valid::Capabilities::ACCELERATION_STRUCTURE_BINDING_ARRAY,
        !device_caps.ray_query.is_empty(),
    );
    caps.set(
        naga::valid::Capabilities::DUAL_SOURCE_BLENDING,
        device_caps.dual_source_blending,
    );
    caps.set(
        naga::valid::Capabilities::SHADER_FLOAT16,
        device_caps.shader_float16,
    );
    caps.set(
        naga::valid::Capabilities::SHADER_BFLOAT16,
        device_caps.shader_bfloat16,
    );
    caps.set(
        naga::valid::Capabilities::COOPERATIVE_MATRIX,
        device_caps.cooperative_matrix.is_supported(),
    );
    caps.set(
        naga::valid::Capabilities::COOPERATIVE_MATRIX_BFLOAT16,
        !device_caps.cooperative_matrix.bf16_f32_shapes.is_empty(),
    );
    // Subgroup operations (subgroupAdd, subgroupBroadcast, etc.) are
    // broadly supported on modern GPUs via VK_EXT_subgroup_size_control.
    // Enable unconditionally so naga validates subgroup ops and emits
    // the correct SPIR-V capabilities (GroupNonUniform, etc.).
    // `SUBGROUP_BARRIER` is separate, and every backend we target has it.
    // (It is split out for HLSL, which we don't generate.)
    caps.set(
        naga::valid::Capabilities::SUBGROUP | naga::valid::Capabilities::SUBGROUP_BARRIER,
        true,
    );
    // `quantizeToF16`, `pack2x16float` and `unpack2x16float` only store
    // f16-precision values inside f32, so they don't need the `f16`
    // extension. Every backend we target supports them.
    caps.set(naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32, true);
    caps
}

impl super::Context {
    fn validate_module(
        &self,
        module: &naga::Module,
        source: &str,
    ) -> Result<naga::valid::ModuleInfo, &'static str> {
        // Bindings are set up at pipeline creation, ignore here
        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        let caps = validation_capabilities(&self.capabilities());

        naga::valid::Validator::new(flags, caps)
            .validate(module)
            .map_err(|e| {
                crate::util::emit_annotated_error(&e, "", source);
                crate::util::print_err(&e);
                "validation failed"
            })
    }

    pub fn try_create_shader(
        &self,
        desc: super::ShaderDesc,
    ) -> Result<super::Shader, &'static str> {
        let module = match desc.naga_module {
            Some(module) => module,
            None => naga::front::wgsl::parse_str(desc.source).map_err(|e| {
                eprintln!("{}", e.emit_to_string_with_path(desc.source, ""));
                "compilation failed"
            })?,
        };
        let info = self.validate_module(&module, desc.source)?;
        Ok(super::Shader {
            module,
            info,
            source: desc.source.to_owned(),
        })
    }

    pub fn create_shader(&self, desc: super::ShaderDesc) -> super::Shader {
        self.try_create_shader(desc).unwrap()
    }
}

pub static EMPTY_CONSTANTS: Lazy<super::PipelineConstants> = Lazy::new(Default::default);

#[cfg(all(
    not(any(gles, target_arch = "wasm32")),
    any(
        vulkan,
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    )
))]
impl<'a> super::ShaderFunction<'a> {
    /// Resolve pipeline constants and keep only what this entry point uses.
    ///
    /// The returned module has this entry point alone, at index 0.
    pub(crate) fn resolve_entry_point(&self) -> (naga::Module, Cow<'a, naga::valid::ModuleInfo>) {
        let entry = &self.shader.module.entry_points[self.entry_point_index()];
        if self.shader.module.entry_points.len() == 1
            && self.shader.module.overrides.is_empty()
            && self.constants.is_empty()
        {
            return (self.shader.module.clone(), Cow::Borrowed(&self.shader.info));
        }
        // Blade assigns resource bindings from the pipeline layout, and vertex
        // input locations from the vertex fetch states, later. Naga revalidates
        // bindings when compaction or overrides change a module. Give the
        // unbound ones temporary, collision-free bindings for that validation,
        // then remove only those bindings afterwards.
        let mut input = self.shader.module.clone();
        let temporary_group = (0..)
            .find(|group| {
                input.global_variables.iter().all(|(_, var)| {
                    var.binding
                        .as_ref()
                        .is_none_or(|binding| binding.group != *group)
                })
            })
            .unwrap();
        for (index, (_, var)) in input.global_variables.iter_mut().enumerate() {
            if var.binding.is_none()
                && matches!(
                    var.space,
                    naga::AddressSpace::Uniform
                        | naga::AddressSpace::Storage { .. }
                        | naga::AddressSpace::Handle
                )
            {
                var.binding = Some(naga::ResourceBinding {
                    group: temporary_group,
                    binding: index as u32,
                });
            }
        }
        let mut temporary_members = Vec::new();
        if entry.stage == naga::ShaderStage::Vertex {
            let mut next_location = 0;
            for argument in entry.function.arguments.iter() {
                let mut bindings = vec![&argument.binding];
                if let naga::TypeInner::Struct { ref members, .. } = input.types[argument.ty].inner
                {
                    bindings.extend(members.iter().map(|member| &member.binding));
                }
                for binding in bindings {
                    if let Some(naga::Binding::Location { location, .. }) = *binding {
                        next_location = next_location.max(location + 1);
                    }
                }
            }
            for (argument_index, argument) in entry.function.arguments.iter().enumerate() {
                if argument.binding.is_some() {
                    continue;
                }
                let mut ty = input.types[argument.ty].clone();
                let naga::TypeInner::Struct {
                    ref mut members, ..
                } = ty.inner
                else {
                    continue;
                };
                let mut unbound = Vec::new();
                for (member_index, member) in members.iter_mut().enumerate() {
                    if member.binding.is_none() {
                        member.binding = Some(naga::Binding::Location {
                            location: next_location,
                            interpolation: None,
                            sampling: None,
                            blend_src: None,
                            per_primitive: false,
                        });
                        next_location += 1;
                        unbound.push(member_index);
                    }
                }
                if !unbound.is_empty() {
                    input.types.replace(argument.ty, ty);
                    temporary_members.push((argument_index, unbound));
                }
            }
        }
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &input,
            &self.shader.info,
            Some((entry.stage, self.entry_point)),
            self.constants,
        )
        .unwrap();
        let mut module = module.into_owned();
        for (_, var) in module.global_variables.iter_mut() {
            if var
                .binding
                .as_ref()
                .is_some_and(|binding| binding.group == temporary_group)
            {
                var.binding = None;
            }
        }
        for (argument_index, unbound) in temporary_members {
            let handle = module.entry_points[0].function.arguments[argument_index].ty;
            let mut ty = module.types[handle].clone();
            if let naga::TypeInner::Struct {
                ref mut members, ..
            } = ty.inner
            {
                for member_index in unbound {
                    members[member_index].binding = None;
                }
            }
            module.types.replace(handle, ty);
        }
        (module, Cow::Owned(info.into_owned()))
    }
}

impl super::Shader {
    pub fn at<'a>(&'a self, entry_point: &'a str) -> super::ShaderFunction<'a> {
        super::ShaderFunction {
            shader: self,
            entry_point,
            constants: Lazy::force(&EMPTY_CONSTANTS),
        }
    }

    pub fn with_constants<'a>(
        &'a self,
        entry_point: &'a str,
        constants: &'a super::PipelineConstants,
    ) -> super::ShaderFunction<'a> {
        super::ShaderFunction {
            shader: self,
            entry_point,
            constants,
        }
    }

    pub fn resolve_constants<'a>(
        &'a self,
        constants: &super::PipelineConstants,
    ) -> (naga::Module, Cow<'a, naga::valid::ModuleInfo>) {
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &self.module,
            &self.info,
            None,
            constants,
        )
        .unwrap();
        (module.into_owned(), info)
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

    pub(crate) fn fill_resource_bindings(
        module: &mut naga::Module,
        sd_infos: &mut [crate::ShaderDataInfo],
        naga_stage: naga::ShaderStage,
        ep_info: &naga::valid::FunctionInfo,
        group_layouts: &[&crate::ShaderDataLayout],
    ) {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();

        for (handle, var) in module.global_variables.iter_mut() {
            if ep_info[handle].is_empty() {
                continue;
            }
            let var_access = match var.space {
                naga::AddressSpace::Storage { access } => access,
                naga::AddressSpace::Uniform | naga::AddressSpace::Handle => {
                    naga::StorageAccess::empty()
                }
                _ => continue,
            };

            assert_eq!(var.binding, None);
            let var_name = var.name.as_ref().unwrap();
            for (group_index, (&layout, info)) in
                group_layouts.iter().zip(sd_infos.iter_mut()).enumerate()
            {
                if let Some((binding_index, &(_, proto_binding))) = layout
                    .bindings
                    .iter()
                    .enumerate()
                    .find(|&(_, &(name, _))| name == var_name)
                {
                    let (expected_proto, access) = match module.types[var.ty].inner {
                        naga::TypeInner::Image {
                            class: naga::ImageClass::Storage { access, format: _ },
                            ..
                        } => (crate::ShaderBinding::Texture, access),
                        naga::TypeInner::Image { .. } => {
                            (crate::ShaderBinding::Texture, naga::StorageAccess::empty())
                        }
                        naga::TypeInner::Sampler { .. } => {
                            (crate::ShaderBinding::Sampler, naga::StorageAccess::empty())
                        }
                        naga::TypeInner::AccelerationStructure { vertex_return: _ } => (
                            crate::ShaderBinding::AccelerationStructure,
                            naga::StorageAccess::empty(),
                        ),
                        naga::TypeInner::BindingArray { base, size: _ } => {
                            //Note: we could extract the count from `size` for more rigor
                            let count = match proto_binding {
                                crate::ShaderBinding::TextureArray { count } => count,
                                crate::ShaderBinding::BufferArray { count } => count,
                                crate::ShaderBinding::AccelerationStructureArray { count } => count,
                                _ => 0,
                            };
                            let proto = match module.types[base].inner {
                                naga::TypeInner::Image { .. } => {
                                    crate::ShaderBinding::TextureArray { count }
                                }
                                naga::TypeInner::Struct { .. } => {
                                    crate::ShaderBinding::BufferArray { count }
                                }
                                naga::TypeInner::AccelerationStructure { .. } => {
                                    crate::ShaderBinding::AccelerationStructureArray { count }
                                }
                                ref other => panic!("Unsupported binding array for {:?}", other),
                            };
                            (proto, var_access)
                        }
                        _ => {
                            let type_layout = &layouter[var.ty];
                            let proto = if var_access.is_empty()
                                && proto_binding != crate::ShaderBinding::Buffer
                            {
                                crate::ShaderBinding::Plain {
                                    size: type_layout.size,
                                }
                            } else {
                                crate::ShaderBinding::Buffer
                            };
                            (proto, var_access)
                        }
                    };
                    assert_eq!(
                        proto_binding, expected_proto,
                        "Mismatched type for binding '{}'",
                        var_name
                    );
                    assert_eq!(var.binding, None);
                    var.binding = Some(naga::ResourceBinding {
                        group: group_index as u32,
                        binding: binding_index as u32,
                    });
                    info.visibility |= naga_stage.into();
                    info.binding_access[binding_index] |= access;
                    break;
                }
            }

            assert!(
                var.binding.is_some(),
                "Unable to resolve binding for '{}' in stage '{:?}'",
                var_name,
                naga_stage,
            );
        }
    }

    pub(crate) fn fill_vertex_locations(
        module: &mut naga::Module,
        selected_ep_index: usize,
        fetch_states: &[crate::VertexFetchState],
    ) -> Vec<crate::VertexAttributeMapping> {
        let mut attribute_mappings = Vec::new();
        for (ep_index, ep) in module.entry_points.iter().enumerate() {
            if ep.stage != naga::ShaderStage::Vertex {
                continue;
            }
            if ep_index != selected_ep_index {
                continue;
            }

            for argument in ep.function.arguments.iter() {
                if argument.binding.is_some() {
                    continue;
                }

                let arg_name = match argument.name {
                    Some(ref name) => name.as_str(),
                    None => "?",
                };
                let mut ty = module.types[argument.ty].clone();
                let members = match ty.inner {
                    naga::TypeInner::Struct {
                        ref mut members, ..
                    } => members,
                    ref other => {
                        log::error!("Unexpected type for '{}': {:?}", arg_name, other);
                        continue;
                    }
                };

                log::debug!("Processing vertex argument: {}", arg_name);
                'member: for member in members.iter_mut() {
                    let member_name = match member.name {
                        Some(ref name) => name.as_str(),
                        None => "?",
                    };
                    if let Some(ref binding) = member.binding {
                        log::warn!(
                            "Member '{}' already has binding: {:?}",
                            member_name,
                            binding
                        );
                        continue;
                    }
                    let binding = naga::Binding::Location {
                        location: attribute_mappings.len() as u32,
                        interpolation: None,
                        sampling: None,
                        blend_src: None,
                        per_primitive: false,
                    };
                    for (buffer_index, vertex_fetch) in fetch_states.iter().enumerate() {
                        for (attribute_index, &(at_name, _)) in
                            vertex_fetch.layout.attributes.iter().enumerate()
                        {
                            if at_name == member_name {
                                log::debug!(
                                    "Assigning location({}) for member '{}' to be using input {}:{}",
                                    attribute_mappings.len(),
                                    member_name,
                                    buffer_index,
                                    attribute_index
                                );
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
                module.types.replace(argument.ty, ty);
            }
        }
        attribute_mappings
    }
}

#[cfg(test)]
mod tests {
    use super::validation_capabilities;

    const BF16_SOURCE: &str = r#"
enable wgpu_bfloat16;

@compute @workgroup_size(1)
fn main() {
    let value = bf16(1.0);
    _ = f32(value);
}
"#;

    #[test]
    fn bfloat16_validation_follows_device_capabilities() {
        let module = naga::front::wgsl::parse_str(BF16_SOURCE).unwrap();
        let flags = naga::valid::ValidationFlags::all();
        let mut device_caps = crate::Capabilities::default();

        let low_end_caps = validation_capabilities(&device_caps);
        assert!(
            naga::valid::Validator::new(flags, low_end_caps)
                .validate(&module)
                .is_err()
        );

        device_caps.shader_bfloat16 = true;
        let bfloat16_caps = validation_capabilities(&device_caps);
        assert!(
            naga::valid::Validator::new(flags, bfloat16_caps)
                .validate(&module)
                .is_ok()
        );
        assert!(!bfloat16_caps.contains(naga::valid::Capabilities::COOPERATIVE_MATRIX_BFLOAT16));

        device_caps
            .cooperative_matrix
            .bf16_f32_shapes
            .push([16, 16, 16]);
        let cooperative_caps = validation_capabilities(&device_caps);
        assert!(cooperative_caps.contains(naga::valid::Capabilities::COOPERATIVE_MATRIX));
        assert!(cooperative_caps.contains(naga::valid::Capabilities::COOPERATIVE_MATRIX_BFLOAT16));
    }
}

#[cfg(all(
    test,
    not(any(gles, target_arch = "wasm32")),
    any(
        vulkan,
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    )
))]
mod entry_point_tests {
    fn shader(source: &str) -> crate::Shader {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        // Bindings are set up at pipeline creation, as in `validate_module`.
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        crate::Shader {
            module,
            info,
            source: source.to_owned(),
        }
    }

    #[test]
    fn entry_point_resolution_preserves_unbound_resources_and_real_bindings() {
        let source = r#"
override factor: f32 = 2.0;
var<storage> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1) fn first() { output[0] = input[0] * factor; }
@compute @workgroup_size(1) fn second() { output[0] = input[1]; }
"#;
        for overrides in [false, true] {
            let source = if overrides {
                source.to_owned()
            } else {
                source.replace("override factor", "const factor")
            };
            let shader = shader(&source);
            for name in ["first", "second"] {
                let entry = shader.at(name);
                let (module, _) = entry.resolve_entry_point();
                assert_eq!(module.entry_points.len(), 1);
                assert!(
                    module
                        .global_expressions
                        .iter()
                        .all(|(_, expression)| !matches!(
                            expression,
                            naga::Expression::Override(_)
                        ))
                );
                for (_, var) in module.global_variables.iter() {
                    match var.name.as_deref() {
                        Some("input") => assert_eq!(var.binding, None),
                        Some("output") => assert_eq!(
                            var.binding,
                            Some(naga::ResourceBinding {
                                group: 0,
                                binding: 3
                            })
                        ),
                        other => panic!("unexpected resource {other:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn workgroup_size_comes_from_resolved_overrides() {
        let shader = shader(
            r#"
override threads: u32 = 64u;
var<storage, read_write> output: array<f32>;
@compute @workgroup_size(threads) fn main() { output[0] = 1.0; }
"#,
        );
        for (threads, constants) in [
            (64, crate::PipelineConstants::default()),
            (
                128,
                crate::PipelineConstants::from_iter([("threads".to_owned(), 128.0)]),
            ),
        ] {
            let (module, _) = shader
                .with_constants("main", &constants)
                .resolve_entry_point();
            assert_eq!(module.entry_points[0].workgroup_size, [threads, 1, 1]);
        }
    }

    #[test]
    fn vertex_inputs_stay_unlocated_until_vertex_fetch() {
        let source = r#"
struct Vertex {
    pos: vec2<f32>,
    uv: vec2<f32>,
}
struct Varyings {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
var<uniform> scale: f32;
@vertex fn vs_main(vertex: Vertex) -> Varyings {
    return Varyings(vec4<f32>(vertex.pos * scale, 0.0, 1.0), vertex.uv);
}
@fragment fn fs_main(input: Varyings) -> @location(0) vec4<f32> {
    return vec4<f32>(input.uv, 0.0, 1.0);
}
"#;
        let locations = |module: &naga::Module| {
            let ty = module.entry_points[0].function.arguments[0].ty;
            let naga::TypeInner::Struct { ref members, .. } = module.types[ty].inner else {
                unreachable!();
            };
            members
                .iter()
                .map(|member| match member.binding {
                    Some(naga::Binding::Location { location, .. }) => Some(location),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let attribute = |offset| crate::VertexAttribute {
            offset,
            format: crate::VertexFormat::F32Vec2,
        };
        let layout = crate::VertexLayout {
            attributes: vec![("pos", attribute(0)), ("uv", attribute(8))],
            stride: 16,
        };
        let fetch = [crate::VertexFetchState {
            layout: &layout,
            instanced: false,
        }];
        for source in [
            source.to_owned(),
            source.replace("var<uniform> scale: f32;", "override scale: f32 = 1.0;"),
        ] {
            let shader = shader(&source);
            let (mut module, info) = shader.at("vs_main").resolve_entry_point();
            assert_eq!(locations(&module), [None, None]);
            let mappings = crate::Shader::fill_vertex_locations(&mut module, 0, &fetch);
            assert_eq!(mappings.len(), 2);
            assert_eq!(locations(&module), [Some(0), Some(1)]);
            naga::back::spv::write_vec(&module, &info, &Default::default(), None).unwrap();
        }
        // Temporary locations must not collide with explicit ones.
        let shader = shader(&source.replace(
            "    uv: vec2<f32>,\n}",
            "    @location(0) uv: vec2<f32>,\n}",
        ));
        let (module, _) = shader.at("vs_main").resolve_entry_point();
        assert_eq!(locations(&module), [None, Some(0)]);
    }

    #[test]
    fn other_entry_points_do_not_leak_into_spirv() {
        let shader = shader(
            r#"
enable f16;
var<storage, read_write> halves: array<f16>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { halves[id.x] = f16(1.0); }
@vertex fn vertex() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
"#,
        );
        for (name, float16) in [("vertex", false), ("main", true)] {
            let (module, info) = shader.at(name).resolve_entry_point();
            assert_eq!(module.entry_points.len(), 1);
            assert_eq!(module.entry_points[0].name, name);
            assert_eq!(module.global_variables.len(), usize::from(float16));
            let options = naga::back::spv::Options {
                lang_version: (1, 3),
                ..Default::default()
            };
            let mut writer = naga::back::spv::Writer::new(&options).unwrap();
            let mut words = Vec::new();
            writer
                .write(&module, &info, None, &None, &mut words)
                .unwrap();
            assert_eq!(
                writer
                    .get_capabilities_used()
                    .contains(&naga::back::spv::Capability::Float16),
                float16
            );
        }
    }
}
