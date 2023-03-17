use ash::vk;
use naga::back::spv;
use std::{ffi, mem, ptr};

const DUMP_PREFIX: Option<&str> = None;

struct CompiledShader {
    vk_module: vk::ShaderModule,
    _entry_point: ffi::CString,
    create_info: vk::PipelineShaderStageCreateInfo,
    wg_size: [u32; 3],
}

struct BindGroupInfo {
    visibility: crate::ShaderVisibility,
    binding_access: Vec<naga::StorageAccess>,
}

impl BindGroupInfo {
    fn new(layout: &crate::ShaderDataLayout) -> Self {
        Self {
            visibility: crate::ShaderVisibility::empty(),
            binding_access: vec![naga::StorageAccess::empty(); layout.bindings.len()],
        }
    }
}

impl super::Context {
    fn make_spv_options(&self) -> spv::Options {
        spv::Options {
            lang_version: match self.device.ray_tracing {
                // Required for ray queries
                Some(_) => (1, 4),
                None => (1, 3),
            },
            flags: self.naga_flags,
            binding_map: spv::BindingMap::default(),
            capabilities: None,
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
            zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::None,
        }
    }

    fn load_shader(
        &self,
        sf: crate::ShaderFunction,
        naga_options: &spv::Options,
        group_layouts: &[&crate::ShaderDataLayout],
        group_infos: &mut [BindGroupInfo],
    ) -> CompiledShader {
        let ep_index = sf.entry_point_index();
        let ep_info = sf.shader.info.get_entry_point(ep_index);
        let ep = &sf.shader.module.entry_points[ep_index];

        let mut module = sf.shader.module.clone();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(&module.types, &module.constants).unwrap();

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
                group_layouts.iter().zip(group_infos.iter_mut()).enumerate()
            {
                if let Some((binding_index, &(_, proto_binding))) = layout
                    .bindings
                    .iter()
                    .enumerate()
                    .find(|(_, &(name, _))| name == var_name)
                {
                    let res_binding = naga::ResourceBinding {
                        group: group_index as u32,
                        binding: binding_index as u32,
                    };
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
                        naga::TypeInner::AccelerationStructure => (
                            crate::ShaderBinding::AccelerationStructure,
                            naga::StorageAccess::empty(),
                        ),
                        naga::TypeInner::BindingArray { base, size: _ } => {
                            //Note: we could extract the count from `size` for more rigor
                            let count = match proto_binding {
                                crate::ShaderBinding::BufferArray { count } => count,
                                _ => 0,
                            };
                            let proto = match module.types[base].inner {
                                naga::TypeInner::Struct { .. } => {
                                    crate::ShaderBinding::BufferArray { count }
                                }
                                ref other => panic!("Unsupported binding array for {:?}", other),
                            };
                            (proto, var_access)
                        }
                        _ => {
                            let type_layout = &layouter[var.ty];
                            let proto = if var_access.is_empty() {
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
                    var.binding = Some(res_binding.clone());
                    info.visibility |= ep.stage.into();
                    info.binding_access[binding_index] |= access;
                    break;
                }
            }

            assert!(
                var.binding.is_some(),
                "Unable to resolve binding for '{}' in the entry point '{}'",
                var_name,
                sf.entry_point,
            );
        }

        let pipeline_options = spv::PipelineOptions {
            shader_stage: ep.stage,
            entry_point: sf.entry_point.to_string(),
        };

        let spv = spv::write_vec(
            &module,
            &sf.shader.info,
            naga_options,
            Some(&pipeline_options),
        )
        .unwrap();

        if let Some(dump_prefix) = DUMP_PREFIX {
            let mut file_name = String::new();
            for i in 1.. {
                file_name = format!("{}{}_{:?}{}.spv", dump_prefix, sf.entry_point, ep.stage, i);
                if !std::path::Path::new(&file_name).exists() {
                    break;
                }
            }
            let spv_bytes =
                unsafe { std::slice::from_raw_parts(spv.as_ptr() as *const u8, spv.len() * 4) };
            println!("Dumping {}", file_name);
            std::fs::write(file_name, spv_bytes).unwrap();
        }

        let vk_info = vk::ShaderModuleCreateInfo::builder().code(&spv);

        let vk_module = unsafe {
            self.device
                .core
                .create_shader_module(&vk_info, None)
                .unwrap()
        };

        let vk_stage = match ep.stage {
            naga::ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
            naga::ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            naga::ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        };

        let entry_point = ffi::CString::new(sf.entry_point).unwrap();
        let create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk_stage)
            .module(vk_module)
            .name(&entry_point)
            .build();

        CompiledShader {
            vk_module,
            _entry_point: entry_point,
            create_info,
            wg_size: ep.workgroup_size,
        }
    }

    fn create_descriptor_set_layout(
        &self,
        layout: &crate::ShaderDataLayout,
        info: &BindGroupInfo,
    ) -> super::DescriptorSetLayout {
        if info.visibility.is_empty() {
            return super::DescriptorSetLayout::default();
        }
        let stage_flags = map_shader_visibility(info.visibility);
        let mut vk_bindings = Vec::with_capacity(layout.bindings.len());
        let mut template_entries = Vec::with_capacity(layout.bindings.len());
        let mut template_offsets = Vec::with_capacity(layout.bindings.len());
        let mut binding_flags = Vec::with_capacity(layout.bindings.len());
        let mut update_offset = 0;
        for (binding_index, (&(_, binding), &access)) in layout
            .bindings
            .iter()
            .zip(info.binding_access.iter())
            .enumerate()
        {
            let (descriptor_type, descriptor_size, descriptor_count, flag) = match binding {
                crate::ShaderBinding::Texture => (
                    if access.is_empty() {
                        vk::DescriptorType::SAMPLED_IMAGE
                    } else {
                        vk::DescriptorType::STORAGE_IMAGE
                    },
                    mem::size_of::<vk::DescriptorImageInfo>(),
                    1u32,
                    vk::DescriptorBindingFlags::empty(),
                ),
                crate::ShaderBinding::Sampler => (
                    vk::DescriptorType::SAMPLER,
                    mem::size_of::<vk::DescriptorImageInfo>(),
                    1u32,
                    vk::DescriptorBindingFlags::empty(),
                ),
                crate::ShaderBinding::Buffer => (
                    vk::DescriptorType::STORAGE_BUFFER,
                    mem::size_of::<vk::DescriptorBufferInfo>(),
                    1u32,
                    vk::DescriptorBindingFlags::empty(),
                ),
                crate::ShaderBinding::BufferArray { count } => (
                    vk::DescriptorType::STORAGE_BUFFER,
                    mem::size_of::<vk::DescriptorBufferInfo>(),
                    count,
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND,
                ),
                crate::ShaderBinding::AccelerationStructure => (
                    vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    mem::size_of::<vk::AccelerationStructureKHR>(),
                    1u32,
                    vk::DescriptorBindingFlags::empty(),
                ),
                crate::ShaderBinding::Plain { size } => (
                    vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                    1,
                    size,
                    vk::DescriptorBindingFlags::empty(),
                ),
            };
            vk_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: binding_index as u32,
                descriptor_type,
                descriptor_count,
                stage_flags,
                p_immutable_samplers: ptr::null(),
            });
            template_entries.push(vk::DescriptorUpdateTemplateEntryKHR {
                dst_binding: binding_index as u32,
                dst_array_element: 0,
                descriptor_count,
                descriptor_type,
                offset: update_offset,
                stride: descriptor_size,
            });
            template_offsets.push(update_offset as u32);
            binding_flags.push(flag);
            update_offset += descriptor_size * descriptor_count as usize;
        }

        let mut binding_flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&binding_flags);
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&vk_bindings)
            .push_next(&mut binding_flags_info);
        let raw = unsafe {
            self.device
                .core
                .create_descriptor_set_layout(&set_layout_info, None)
                .unwrap()
        };

        let template_create_info = vk::DescriptorUpdateTemplateCreateInfo::builder()
            .descriptor_update_entries(&template_entries)
            .template_type(vk::DescriptorUpdateTemplateTypeKHR::DESCRIPTOR_SET)
            .descriptor_set_layout(raw);
        let update_template = unsafe {
            self.device
                .core
                .create_descriptor_update_template(&template_create_info, None)
                .unwrap()
        };

        super::DescriptorSetLayout {
            raw,
            update_template,
            template_size: update_offset as u32,
            template_offsets: template_offsets.into_boxed_slice(),
        }
    }

    fn create_pipeline_layout(
        &self,
        group_layouts: &[&crate::ShaderDataLayout],
        group_infos: &[BindGroupInfo],
    ) -> super::PipelineLayout {
        let mut descriptor_set_layouts = Vec::with_capacity(group_layouts.len());
        let mut vk_set_layouts = Vec::with_capacity(group_layouts.len());
        for (&layout, info) in group_layouts.iter().zip(group_infos) {
            let dsl = self.create_descriptor_set_layout(layout, info);
            vk_set_layouts.push(dsl.raw);
            descriptor_set_layouts.push(dsl);
        }

        let vk_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&vk_set_layouts);
        let raw = unsafe {
            self.device
                .core
                .create_pipeline_layout(&vk_info, None)
                .unwrap()
        };

        super::PipelineLayout {
            raw,
            descriptor_set_layouts,
        }
    }

    pub fn create_compute_pipeline(
        &self,
        desc: crate::ComputePipelineDesc,
    ) -> super::ComputePipeline {
        let mut group_infos = desc
            .data_layouts
            .iter()
            .cloned()
            .map(BindGroupInfo::new)
            .collect::<Vec<_>>();

        let options = self.make_spv_options();
        let cs = self.load_shader(desc.compute, &options, desc.data_layouts, &mut group_infos);

        let layout = self.create_pipeline_layout(desc.data_layouts, &group_infos);

        let create_info = vk::ComputePipelineCreateInfo::builder()
            .layout(layout.raw)
            .stage(cs.create_info)
            .build();

        let mut raw_vec = unsafe {
            self.device
                .core
                .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .unwrap()
        };
        let raw = raw_vec.pop().unwrap();

        unsafe { self.device.core.destroy_shader_module(cs.vk_module, None) };

        if !desc.name.is_empty() {
            self.set_object_name(vk::ObjectType::PIPELINE, raw, desc.name);
        }
        super::ComputePipeline {
            raw,
            layout,
            wg_size: cs.wg_size,
        }
    }

    pub fn create_render_pipeline(&self, desc: crate::RenderPipelineDesc) -> super::RenderPipeline {
        let mut group_infos = desc
            .data_layouts
            .iter()
            .cloned()
            .map(BindGroupInfo::new)
            .collect::<Vec<_>>();

        let options = self.make_spv_options();
        let vs = self.load_shader(desc.vertex, &options, desc.data_layouts, &mut group_infos);
        let fs = self.load_shader(desc.fragment, &options, desc.data_layouts, &mut group_infos);

        let layout = self.create_pipeline_layout(desc.data_layouts, &group_infos);

        let stages = [vs.create_info, fs.create_info];
        let vk_vertex_input = vk::PipelineVertexInputStateCreateInfo::builder().build();
        let (raw_topology, supports_restart) = map_primitive_topology(desc.primitive.topology);
        let vk_input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(raw_topology)
            .primitive_restart_enable(supports_restart)
            .build();
        let mut vk_rasterization = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(if desc.primitive.wireframe {
                vk::PolygonMode::LINE
            } else {
                vk::PolygonMode::FILL
            })
            .front_face(map_front_face(desc.primitive.front_face))
            .line_width(1.0);
        let mut vk_depth_clip_state =
            vk::PipelineRasterizationDepthClipStateCreateInfoEXT::builder()
                .depth_clip_enable(false)
                .build();
        if desc.primitive.unclipped_depth {
            vk_rasterization = vk_rasterization.push_next(&mut vk_depth_clip_state);
        }

        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_REFERENCE,
        ];
        let vk_dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states)
            .build();

        let vk_viewport = vk::PipelineViewportStateCreateInfo::builder()
            .flags(vk::PipelineViewportStateCreateFlags::empty())
            .scissor_count(1)
            .viewport_count(1)
            .build();

        let vk_sample_mask = [1u32, 0];
        let vk_multisample = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_mask(&vk_sample_mask)
            .build();

        let mut ds_format = vk::Format::UNDEFINED;
        let mut vk_depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder();
        if let Some(ref ds) = desc.depth_stencil {
            ds_format = super::map_texture_format(ds.format);

            if ds.depth_write_enabled || ds.depth_compare != crate::CompareFunction::Always {
                vk_depth_stencil = vk_depth_stencil
                    .depth_test_enable(true)
                    .depth_write_enable(ds.depth_write_enabled)
                    .depth_compare_op(map_comparison(ds.depth_compare));
            }
            if ds.stencil != crate::StencilState::default() {
                let s = &ds.stencil;
                let front = map_stencil_face(&s.front, s.read_mask, s.write_mask);
                let back = map_stencil_face(&s.back, s.read_mask, s.write_mask);
                vk_depth_stencil = vk_depth_stencil
                    .stencil_test_enable(true)
                    .front(front)
                    .back(back);
            }

            if ds.bias != crate::DepthBiasState::default() {
                vk_rasterization = vk_rasterization
                    .depth_bias_enable(true)
                    .depth_bias_constant_factor(ds.bias.constant as f32)
                    .depth_bias_clamp(ds.bias.clamp)
                    .depth_bias_slope_factor(ds.bias.slope_scale);
            }
        }

        let mut color_formats = Vec::with_capacity(desc.color_targets.len());
        let mut vk_attachments = Vec::with_capacity(desc.color_targets.len());
        for ct in desc.color_targets {
            let mut vk_attachment = vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::from_raw(ct.write_mask.bits()));
            if let Some(ref blend) = ct.blend {
                let (color_op, color_src, color_dst) = map_blend_component(&blend.color);
                let (alpha_op, alpha_src, alpha_dst) = map_blend_component(&blend.alpha);
                vk_attachment = vk_attachment
                    .blend_enable(true)
                    .color_blend_op(color_op)
                    .src_color_blend_factor(color_src)
                    .dst_color_blend_factor(color_dst)
                    .alpha_blend_op(alpha_op)
                    .src_alpha_blend_factor(alpha_src)
                    .dst_alpha_blend_factor(alpha_dst);
            }

            color_formats.push(super::map_texture_format(ct.format));
            vk_attachments.push(vk_attachment.build());
        }
        let vk_color_blend = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&vk_attachments)
            .build();

        let mut rendering_info = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(ds_format)
            .stencil_attachment_format(ds_format)
            .build();

        let create_info = vk::GraphicsPipelineCreateInfo::builder()
            .layout(layout.raw)
            .stages(&stages)
            .vertex_input_state(&vk_vertex_input)
            .input_assembly_state(&vk_input_assembly)
            .rasterization_state(&vk_rasterization)
            .viewport_state(&vk_viewport)
            .multisample_state(&vk_multisample)
            .depth_stencil_state(&vk_depth_stencil)
            .color_blend_state(&vk_color_blend)
            .dynamic_state(&vk_dynamic_state)
            .push_next(&mut rendering_info)
            .build();

        let mut raw_vec = unsafe {
            self.device
                .core
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .unwrap()
        };
        let raw = raw_vec.pop().unwrap();

        unsafe { self.device.core.destroy_shader_module(vs.vk_module, None) };
        unsafe { self.device.core.destroy_shader_module(fs.vk_module, None) };

        if !desc.name.is_empty() {
            self.set_object_name(vk::ObjectType::PIPELINE, raw, desc.name);
        }
        super::RenderPipeline { raw, layout }
    }
}

fn map_shader_visibility(visibility: crate::ShaderVisibility) -> vk::ShaderStageFlags {
    use crate::ShaderVisibility as Sv;
    use vk::ShaderStageFlags as Flags;

    let mut flags = Flags::empty();
    if visibility.contains(Sv::COMPUTE) {
        flags |= Flags::COMPUTE;
    }
    if visibility.contains(Sv::VERTEX) {
        flags |= Flags::VERTEX;
    }
    if visibility.contains(Sv::FRAGMENT) {
        flags |= Flags::FRAGMENT;
    }

    flags
}

fn map_primitive_topology(topology: crate::PrimitiveTopology) -> (vk::PrimitiveTopology, bool) {
    use crate::PrimitiveTopology as Pt;
    match topology {
        Pt::PointList => (vk::PrimitiveTopology::POINT_LIST, false),
        Pt::LineList => (vk::PrimitiveTopology::LINE_LIST, false),
        Pt::LineStrip => (vk::PrimitiveTopology::LINE_STRIP, true),
        Pt::TriangleList => (vk::PrimitiveTopology::TRIANGLE_LIST, false),
        Pt::TriangleStrip => (vk::PrimitiveTopology::TRIANGLE_STRIP, true),
    }
}

fn map_front_face(front_face: crate::FrontFace) -> vk::FrontFace {
    match front_face {
        crate::FrontFace::Cw => vk::FrontFace::CLOCKWISE,
        crate::FrontFace::Ccw => vk::FrontFace::COUNTER_CLOCKWISE,
    }
}

fn map_comparison(fun: crate::CompareFunction) -> vk::CompareOp {
    use crate::CompareFunction as Cf;
    match fun {
        Cf::Never => vk::CompareOp::NEVER,
        Cf::Less => vk::CompareOp::LESS,
        Cf::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        Cf::Equal => vk::CompareOp::EQUAL,
        Cf::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        Cf::Greater => vk::CompareOp::GREATER,
        Cf::NotEqual => vk::CompareOp::NOT_EQUAL,
        Cf::Always => vk::CompareOp::ALWAYS,
    }
}

fn map_stencil_op(op: crate::StencilOperation) -> vk::StencilOp {
    use crate::StencilOperation as So;
    match op {
        So::Keep => vk::StencilOp::KEEP,
        So::Zero => vk::StencilOp::ZERO,
        So::Replace => vk::StencilOp::REPLACE,
        So::Invert => vk::StencilOp::INVERT,
        So::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        So::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        So::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        So::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

fn map_stencil_face(
    face: &crate::StencilFaceState,
    compare_mask: u32,
    write_mask: u32,
) -> vk::StencilOpState {
    vk::StencilOpState {
        fail_op: map_stencil_op(face.fail_op),
        pass_op: map_stencil_op(face.pass_op),
        depth_fail_op: map_stencil_op(face.depth_fail_op),
        compare_op: map_comparison(face.compare),
        compare_mask,
        write_mask,
        reference: 0,
    }
}

fn map_blend_factor(factor: crate::BlendFactor) -> vk::BlendFactor {
    use crate::BlendFactor as Bf;
    match factor {
        Bf::Zero => vk::BlendFactor::ZERO,
        Bf::One => vk::BlendFactor::ONE,
        Bf::Src => vk::BlendFactor::SRC_COLOR,
        Bf::OneMinusSrc => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        Bf::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        Bf::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        Bf::Dst => vk::BlendFactor::DST_COLOR,
        Bf::OneMinusDst => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        Bf::DstAlpha => vk::BlendFactor::DST_ALPHA,
        Bf::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        Bf::SrcAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
        Bf::Constant => vk::BlendFactor::CONSTANT_COLOR,
        Bf::OneMinusConstant => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
    }
}

fn map_blend_op(operation: crate::BlendOperation) -> vk::BlendOp {
    use crate::BlendOperation as Bo;
    match operation {
        Bo::Add => vk::BlendOp::ADD,
        Bo::Subtract => vk::BlendOp::SUBTRACT,
        Bo::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        Bo::Min => vk::BlendOp::MIN,
        Bo::Max => vk::BlendOp::MAX,
    }
}

fn map_blend_component(
    component: &crate::BlendComponent,
) -> (vk::BlendOp, vk::BlendFactor, vk::BlendFactor) {
    let op = map_blend_op(component.operation);
    let src = map_blend_factor(component.src_factor);
    let dst = map_blend_factor(component.dst_factor);
    (op, src, dst)
}
