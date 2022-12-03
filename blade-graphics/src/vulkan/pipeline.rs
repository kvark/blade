use ash::vk;
use naga::back::spv;
use std::{ffi, mem, ptr};

struct CompiledShader {
    vk_module: vk::ShaderModule,
    _entry_point: ffi::CString,
    create_info: vk::PipelineShaderStageCreateInfo,
    wg_size: [u32; 3],
}

impl super::Context {
    fn load_shader(
        &self,
        sf: crate::ShaderFunction,
        naga_stage: naga::ShaderStage,
        naga_options: &spv::Options,
    ) -> CompiledShader {
        let pipeline_options = spv::PipelineOptions {
            shader_stage: naga_stage,
            entry_point: sf.entry_point.to_string(),
        };

        let spv = spv::write_vec(
            &sf.shader.module,
            &sf.shader.info,
            naga_options,
            Some(&pipeline_options),
        )
        .unwrap();

        let ep = sf
            .shader
            .module
            .entry_points
            .iter()
            .find(|ep| ep.stage == naga_stage && ep.name == sf.entry_point)
            .expect("Entry point not found in the shader");

        let vk_info = vk::ShaderModuleCreateInfo::builder().code(&spv);

        let vk_module = unsafe {
            self.device
                .core
                .create_shader_module(&vk_info, None)
                .unwrap()
        };

        let vk_stage = match naga_stage {
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
        visibility: crate::ShaderVisibility,
    ) -> super::DescriptorSetLayout {
        if visibility.is_empty() {
            return super::DescriptorSetLayout::default();
        }
        let stage_flags = map_shader_visibility(visibility);
        let mut vk_bindings = Vec::new();
        let mut binding_index = 1;
        let mut plain_data_size = 0u32;
        let mut template_entries = Vec::new();
        let mut template_offsets = Vec::with_capacity(layout.bindings.len());
        let mut update_offset = 0;
        for &(_, binding) in layout.bindings.iter() {
            let (descriptor_type, descriptor_size) = match binding {
                crate::ShaderBinding::Texture { .. } => (
                    vk::DescriptorType::SAMPLED_IMAGE,
                    mem::size_of::<vk::DescriptorImageInfo>(),
                ),
                crate::ShaderBinding::TextureStorage { .. } => (
                    vk::DescriptorType::STORAGE_IMAGE,
                    mem::size_of::<vk::DescriptorImageInfo>(),
                ),
                crate::ShaderBinding::Sampler { .. } => (
                    vk::DescriptorType::SAMPLER,
                    mem::size_of::<vk::DescriptorImageInfo>(),
                ),
                crate::ShaderBinding::Buffer { .. } => (
                    vk::DescriptorType::STORAGE_BUFFER,
                    mem::size_of::<vk::DescriptorBufferInfo>(),
                ),
                crate::ShaderBinding::Plain { ty, container } => {
                    let elem_size = match ty {
                        crate::PlainType::F32 | crate::PlainType::I32 | crate::PlainType::U32 => 4,
                    };
                    //TODO: alignment
                    let count = match container {
                        crate::PlainContainer::Scalar => 1,
                        crate::PlainContainer::Vector(size) => size as u32,
                        crate::PlainContainer::Matrix(rows, cols) => rows as u32 * cols as u32,
                    };
                    template_offsets.push(plain_data_size);
                    plain_data_size += elem_size * count;
                    continue;
                }
            };
            vk_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: binding_index,
                descriptor_type,
                descriptor_count: 1,
                stage_flags,
                p_immutable_samplers: ptr::null(),
            });
            template_entries.push(vk::DescriptorUpdateTemplateEntryKHR {
                dst_binding: binding_index,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type,
                offset: update_offset,
                stride: 0,
            });
            binding_index += 1;
            template_offsets.push(update_offset as u32);
            update_offset += descriptor_size;
        }

        let handle_section_size = update_offset as u32;
        if plain_data_size != 0 {
            vk_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                descriptor_count: plain_data_size,
                stage_flags,
                p_immutable_samplers: ptr::null(),
            });
            template_entries.push(vk::DescriptorUpdateTemplateEntryKHR {
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: plain_data_size,
                descriptor_type: vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                offset: update_offset,
                stride: 0,
            });
            for (template_offset, &(_, binding)) in
                template_offsets.iter_mut().zip(layout.bindings.iter())
            {
                if let crate::ShaderBinding::Plain { .. } = binding {
                    *template_offset += handle_section_size;
                }
            }
        }

        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&vk_bindings);
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
            template_size: handle_section_size + plain_data_size,
            template_offsets: template_offsets.into_boxed_slice(),
        }
    }

    fn create_pipeline_layout(
        &self,
        combined: &[(&crate::ShaderDataLayout, crate::ShaderVisibility)],
    ) -> super::PipelineLayout {
        let mut descriptor_set_layouts = Vec::with_capacity(combined.len());
        let mut vk_set_layouts = Vec::with_capacity(combined.len());
        for &(layout, visibility) in combined {
            let dsl = self.create_descriptor_set_layout(layout, visibility);
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
        let combined =
            crate::merge_shader_layouts(&[(desc.compute.shader, crate::ShaderVisibility::COMPUTE)]);
        let layout = self.create_pipeline_layout(&combined);

        let options = spv::Options {
            lang_version: (1, 3),
            flags: self.naga_flags,
            binding_map: spv::BindingMap::default(),
            capabilities: None,
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        };

        let cs = self.load_shader(desc.compute, naga::ShaderStage::Compute, &options);

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
        let combined = crate::merge_shader_layouts(&[
            (desc.vertex.shader, crate::ShaderVisibility::VERTEX),
            (desc.fragment.shader, crate::ShaderVisibility::FRAGMENT),
        ]);
        let layout = self.create_pipeline_layout(&combined);

        let options = spv::Options {
            lang_version: (1, 3),
            flags: self.naga_flags,
            binding_map: spv::BindingMap::default(),
            capabilities: None,
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        };
        let vs = self.load_shader(desc.vertex, naga::ShaderStage::Vertex, &options);
        let fs = self.load_shader(desc.fragment, naga::ShaderStage::Fragment, &options);
        let stages = [vs.create_info, fs.create_info];

        let vk_vertex_input = vk::PipelineVertexInputStateCreateInfo::builder().build();
        let vk_input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(map_primitive_topology(desc.primitive.topology))
            .primitive_restart_enable(true)
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
            ds_format = super::describe_format(ds.format).raw;

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

            color_formats.push(super::describe_format(ct.format).raw);
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

fn map_primitive_topology(topology: crate::PrimitiveTopology) -> vk::PrimitiveTopology {
    use crate::PrimitiveTopology as Pt;
    match topology {
        Pt::PointList => vk::PrimitiveTopology::POINT_LIST,
        Pt::LineList => vk::PrimitiveTopology::LINE_LIST,
        Pt::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        Pt::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        Pt::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
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
