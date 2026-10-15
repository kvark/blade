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
        device_caps.cooperative_matrix.supports_bfloat16(),
    );
    caps.set(
        naga::valid::Capabilities::COOPERATIVE_MATRIX_WORKGROUP,
        !device_caps
            .cooperative_matrix
            .workgroup
            .granularities
            .is_empty(),
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
            && !naga::back::uses_workgroup_cooperative_matrix(&self.shader.module)
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
        let (mut module, info) = if matches!(module, Cow::Borrowed(_))
            && naga::back::uses_workgroup_cooperative_matrix(&module)
        {
            // process_overrides skips compaction for a single entry point without overrides.
            let mut module = module.into_owned();
            naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
            let flags =
                naga::valid::ValidationFlags::all() - naga::valid::ValidationFlags::BINDINGS;
            let info = naga::valid::Validator::new(flags, naga::valid::Capabilities::all())
                .validate_resolved_overrides(&module)
                .unwrap();
            (module, info)
        } else {
            (module.into_owned(), info.into_owned())
        };
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
        (module, Cow::Owned(info))
    }
}

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
pub(crate) fn validate_workgroup_matrices(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    caps: &super::Capabilities,
) -> Result<(), &'static str> {
    if !naga::back::uses_workgroup_cooperative_matrix(module) {
        return Ok(());
    }
    assert_eq!(module.entry_points.len(), 1);
    let entry = &module.entry_points[0];
    if entry.stage != naga::ShaderStage::Compute {
        return Err("workgroup matrices require a compute entry point");
    }
    let matrix = &caps.cooperative_matrix.workgroup;
    let Some(invocations) = entry
        .workgroup_size
        .into_iter()
        .try_fold(1u32, u32::checked_mul)
    else {
        return Err("workgroup invocation count overflows");
    };
    if invocations == 0 || invocations > matrix.max_workgroup_size {
        return Err("workgroup matrix invocation count exceeds device limits");
    }
    let operand = |scalar| match scalar {
        naga::Scalar::F16 => Some(super::CooperativeMatrixOperand::F16),
        naga::Scalar::BF16 => Some(super::CooperativeMatrixOperand::Bf16),
        _ => None,
    };
    let validate_type = |ty: &naga::TypeInner| {
        let naga::TypeInner::CooperativeMatrix {
            columns,
            rows,
            scalar,
            role,
            scope: naga::CooperativeScope::Workgroup,
        } = *ty
        else {
            return Ok(());
        };
        if columns == 0
            || rows == 0
            || columns > matrix.max_dimension
            || rows > matrix.max_dimension
        {
            return Err("workgroup matrix dimension exceeds device limits");
        }
        let supported = matrix.granularities.iter().any(|g| {
            let (row_granularity, column_granularity, scalar_matches) = match role {
                naga::CooperativeRole::A => {
                    (g.size[0], g.size[2], operand(scalar) == Some(g.operand))
                }
                naga::CooperativeRole::B => {
                    (g.size[2], g.size[1], operand(scalar) == Some(g.operand))
                }
                naga::CooperativeRole::C => (g.size[0], g.size[1], scalar == naga::Scalar::F32),
            };
            g.invocations == invocations
                && scalar_matches
                && row_granularity != 0
                && column_granularity != 0
                && rows % row_granularity == 0
                && columns % column_granularity == 0
        });
        if supported {
            Ok(())
        } else {
            Err("unsupported workgroup matrix type or invocation count")
        }
    };
    for (_, ty) in module.types.iter() {
        validate_type(&ty.inner)?;
    }
    let functions = module
        .functions
        .iter()
        .map(|(handle, f)| (f, &info[handle]))
        .chain(std::iter::once((&entry.function, info.get_entry_point(0))));
    for (function, function_info) in functions {
        for (handle, expression) in function.expressions.iter() {
            validate_type(function_info[handle].ty.inner_with(&module.types))?;
            if let naga::Expression::CooperativeMultiplyAdd { a, b, c } = *expression {
                let naga::TypeInner::CooperativeMatrix {
                    columns: k,
                    rows: m,
                    scalar,
                    scope: naga::CooperativeScope::Workgroup,
                    ..
                } = *function_info[a].ty.inner_with(&module.types)
                else {
                    continue;
                };
                let naga::TypeInner::CooperativeMatrix { columns: n, .. } =
                    *function_info[b].ty.inner_with(&module.types)
                else {
                    unreachable!();
                };
                let naga::TypeInner::CooperativeMatrix {
                    scalar: accumulator,
                    ..
                } = *function_info[c].ty.inner_with(&module.types)
                else {
                    unreachable!();
                };
                if accumulator != naga::Scalar::F32
                    || !operand(scalar)
                        .is_some_and(|operand| matrix.supports(operand, [m, n, k], invocations))
                {
                    return Err("workgroup matrix product does not match one device configuration");
                }
            }
        }
    }
    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|_| "invalid workgroup memory layout")?;
    let mut shared = u64::from(matrix.reserved_shared_memory);
    for (handle, variable) in module.global_variables.iter() {
        if variable.space == naga::AddressSpace::WorkGroup
            && !info.get_entry_point(0)[handle].is_empty()
        {
            shared += u64::from(layouter[variable.ty].to_stride());
        }
    }
    if shared > u64::from(caps.max_compute_shared_memory_size) {
        return Err("workgroup storage plus matrix reserved memory exceeds device limits");
    }
    Ok(())
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
    #[test]
    fn workgroup_matrix_capabilities_are_independent_of_subgroup_shapes() {
        let mut caps = crate::Capabilities {
            shader_float16: true,
            shader_bfloat16: true,
            ..Default::default()
        };
        let source = "enable wgpu_cooperative_matrix; enable wgpu_bfloat16; @compute @workgroup_size(64) fn main() { var a: coop_mat64x64<bf16,A,workgroup>; }";
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let validate = |caps: &crate::Capabilities| {
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                validation_capabilities(caps),
            )
            .validate(&module)
            .is_ok()
        };
        assert!(!validate(&caps));
        caps.cooperative_matrix.workgroup = crate::WorkgroupCooperativeMatrix {
            granularities: vec![crate::CooperativeMatrixGranularity {
                operand: crate::CooperativeMatrixOperand::Bf16,
                size: [32, 16, 16],
                invocations: 64,
            }],
            max_workgroup_size: 256,
            max_dimension: 1024,
            reserved_shared_memory: 8192,
        };
        assert!(validate(&caps));
        assert!(caps.cooperative_matrix.supports_bfloat16());
        let workgroup = &caps.cooperative_matrix.workgroup;
        assert!(workgroup.supports(crate::CooperativeMatrixOperand::Bf16, [64, 128, 32], 64));
        for (size, threads) in [
            ([16, 128, 32], 64),
            ([64, 128, 32], 128),
            ([2048, 128, 32], 64),
            ([0, 128, 32], 64),
        ] {
            assert!(!workgroup.supports(crate::CooperativeMatrixOperand::Bf16, size, threads));
        }
        assert!(!workgroup.supports(crate::CooperativeMatrixOperand::F16, [64, 128, 32], 64));
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
mod workgroup_tests {
    use super::validate_workgroup_matrices;

    const SOURCE: &str = r#"
enable f16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage> a: array<f16>;
@group(0) @binding(1) var<storage> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@compute @workgroup_size(64)
fn main() {
    let left = coopLoadT<coop_mat32x64<f16,A,workgroup>>(&a[0], 32u);
    let right = coopLoadT<coop_mat64x32<f16,B,workgroup>>(&b[0], 64u);
    let result = coopMultiplyAdd(left, right, coop_mat64x64<f32,C,workgroup>());
    coopStoreT(result, &c[0], 64u);
}
"#;

    fn shader(source: &str) -> crate::Shader {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
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

    fn caps() -> crate::Capabilities {
        let mut caps = crate::Capabilities {
            max_compute_shared_memory_size: 65536,
            ..Default::default()
        };
        caps.cooperative_matrix.workgroup = crate::WorkgroupCooperativeMatrix {
            granularities: vec![crate::CooperativeMatrixGranularity {
                operand: crate::CooperativeMatrixOperand::F16,
                size: [16, 16, 16],
                invocations: 64,
            }],
            max_workgroup_size: 256,
            max_dimension: 1024,
            reserved_shared_memory: 8192,
        };
        caps
    }

    fn check(source: &str, caps: &crate::Capabilities) -> Result<(), &'static str> {
        let shader = shader(source);
        let entry = shader.at("main");
        let (module, info) = entry.resolve_entry_point();
        validate_workgroup_matrices(&module, &info, caps)
    }

    #[test]
    fn device_shapes_invocations_and_shared_memory() {
        let mut caps = caps();
        check(SOURCE, &caps).unwrap();
        assert_eq!(
            check(
                &SOURCE.replace("workgroup_size(64)", "workgroup_size(128)"),
                &caps
            ),
            Err("unsupported workgroup matrix type or invocation count")
        );
        let ragged = SOURCE.replace("32x64", "24x64").replace("64x32", "64x24");
        assert_eq!(
            check(&ragged, &caps),
            Err("unsupported workgroup matrix type or invocation count")
        );
        let shared = SOURCE
            .replace(
                "@compute",
                "var<workgroup> scratch: array<f32, 16384>;\n@compute",
            )
            .replace("    let left", "    scratch[0] = 1.0;\n    let left");
        assert_eq!(
            check(&shared, &caps),
            Err("workgroup storage plus matrix reserved memory exceeds device limits")
        );
        check(&shared.replace("16384", "14336"), &caps).unwrap();
        caps.cooperative_matrix.workgroup.max_dimension = 32;
        assert_eq!(
            check(SOURCE, &caps),
            Err("workgroup matrix dimension exceeds device limits")
        );
    }

    #[test]
    fn product_must_match_one_configuration() {
        let mut caps = caps();
        caps.cooperative_matrix.workgroup.granularities =
            [[64, 128, 32], [128, 64, 32], [64, 64, 64]]
                .map(|size| crate::CooperativeMatrixGranularity {
                    operand: crate::CooperativeMatrixOperand::F16,
                    size,
                    invocations: 64,
                })
                .to_vec();
        assert_eq!(
            check(SOURCE, &caps),
            Err("workgroup matrix product does not match one device configuration")
        );
    }

    #[test]
    fn workgroup_size_checked_after_overrides() {
        let shader = shader(
            &SOURCE
                .replace("@compute", "override threads: u32 = 64u;\n@compute")
                .replace("workgroup_size(64)", "workgroup_size(threads)"),
        );
        for threads in [64, 128] {
            let constants =
                crate::PipelineConstants::from_iter([("threads".to_owned(), f64::from(threads))]);
            let entry = shader.with_constants("main", &constants);
            let (module, info) = entry.resolve_entry_point();
            assert_eq!(module.entry_points[0].workgroup_size, [threads, 1, 1]);
            assert_eq!(
                validate_workgroup_matrices(&module, &info, &caps()).is_ok(),
                threads == 64
            );
        }
    }

    #[test]
    fn single_entry_point_drops_unreachable_workgroup_functions() {
        let source = SOURCE.replace("@compute @workgroup_size(64)", "");
        let shader = shader(&format!(
            "{source}\n@vertex fn vertex() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0); }}"
        ));
        let entry = shader.at("vertex");
        let (module, info) = entry.resolve_entry_point();
        assert!(!naga::back::uses_workgroup_cooperative_matrix(&module));
        assert!(module.functions.is_empty());
        validate_workgroup_matrices(&module, &info, &caps()).unwrap();
    }

    #[test]
    fn mixed_entry_points_drop_unrelated_types_and_bindings() {
        let shader = shader(&format!(
            "{SOURCE}\n@vertex fn vertex() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0); }}"
        ));
        for (name, version, workgroup) in [("vertex", (1, 3), false), ("main", (1, 6), true)] {
            let entry = shader.at(name);
            let (module, info) = entry.resolve_entry_point();
            assert_eq!(module.entry_points.len(), 1);
            assert_eq!(module.entry_points[0].name, name);
            assert_eq!(
                naga::back::uses_workgroup_cooperative_matrix(&module),
                workgroup
            );
            validate_workgroup_matrices(&module, &info, &caps()).unwrap();
            if !workgroup {
                assert!(module.global_variables.is_empty());
            }
            naga::back::spv::write_vec(
                &module,
                &info,
                &naga::back::spv::Options {
                    lang_version: version,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        }
    }
}
