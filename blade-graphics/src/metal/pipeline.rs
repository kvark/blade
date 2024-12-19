use naga::back::msl;

fn map_blend_factor(factor: crate::BlendFactor) -> metal::MTLBlendFactor {
    use crate::BlendFactor as Bf;
    use metal::MTLBlendFactor::*;

    match factor {
        Bf::Zero => Zero,
        Bf::One => One,
        Bf::Src => SourceColor,
        Bf::OneMinusSrc => OneMinusSourceColor,
        Bf::Dst => DestinationColor,
        Bf::OneMinusDst => OneMinusDestinationColor,
        Bf::SrcAlpha => SourceAlpha,
        Bf::OneMinusSrcAlpha => OneMinusSourceAlpha,
        Bf::DstAlpha => DestinationAlpha,
        Bf::OneMinusDstAlpha => OneMinusDestinationAlpha,
        Bf::Constant => BlendColor,
        Bf::OneMinusConstant => OneMinusBlendColor,
        Bf::SrcAlphaSaturated => SourceAlphaSaturated,
    }
}

fn map_blend_op(operation: crate::BlendOperation) -> metal::MTLBlendOperation {
    use crate::BlendOperation as Bo;
    use metal::MTLBlendOperation::*;

    match operation {
        Bo::Add => Add,
        Bo::Subtract => Subtract,
        Bo::ReverseSubtract => ReverseSubtract,
        Bo::Min => Min,
        Bo::Max => Max,
    }
}

fn map_blend_component(
    component: &crate::BlendComponent,
) -> (
    metal::MTLBlendOperation,
    metal::MTLBlendFactor,
    metal::MTLBlendFactor,
) {
    (
        map_blend_op(component.operation),
        map_blend_factor(component.src_factor),
        map_blend_factor(component.dst_factor),
    )
}

fn map_stencil_op(op: crate::StencilOperation) -> metal::MTLStencilOperation {
    use crate::StencilOperation as So;
    use metal::MTLStencilOperation::*;

    match op {
        So::Keep => Keep,
        So::Zero => Zero,
        So::Replace => Replace,
        So::IncrementClamp => IncrementClamp,
        So::IncrementWrap => IncrementWrap,
        So::DecrementClamp => DecrementClamp,
        So::DecrementWrap => DecrementWrap,
        So::Invert => Invert,
    }
}

fn create_stencil_desc(
    face: &crate::StencilFaceState,
    read_mask: u32,
    write_mask: u32,
) -> metal::StencilDescriptor {
    let desc = metal::StencilDescriptor::new();
    desc.set_stencil_compare_function(super::map_compare_function(face.compare));
    desc.set_read_mask(read_mask);
    desc.set_write_mask(write_mask);
    desc.set_stencil_failure_operation(map_stencil_op(face.fail_op));
    desc.set_depth_failure_operation(map_stencil_op(face.depth_fail_op));
    desc.set_depth_stencil_pass_operation(map_stencil_op(face.pass_op));
    desc
}

fn create_depth_stencil_desc(state: &crate::DepthStencilState) -> metal::DepthStencilDescriptor {
    let desc = metal::DepthStencilDescriptor::new();
    desc.set_depth_compare_function(super::map_compare_function(state.depth_compare));
    desc.set_depth_write_enabled(state.depth_write_enabled);

    let s = &state.stencil;
    if s.front != crate::StencilFaceState::IGNORE {
        let face_desc = create_stencil_desc(&s.front, s.read_mask, s.write_mask);
        desc.set_front_face_stencil(Some(&face_desc));
    }
    if s.back != crate::StencilFaceState::IGNORE {
        let face_desc = create_stencil_desc(&s.back, s.read_mask, s.write_mask);
        desc.set_back_face_stencil(Some(&face_desc));
    }

    desc
}

struct CompiledShader {
    library: metal::Library,
    function: metal::Function,
    attribute_mappings: Vec<crate::VertexAttributeMapping>,
    wg_size: metal::MTLSize,
    wg_memory_sizes: Vec<u32>,
}

bitflags::bitflags! {
    #[repr(transparent)]
    struct ShaderFlags: u32 {
        const ALLOW_POINT_SIZE = 1<<0;
    }
}

fn _align_to(offset: u32, alignment: u32) -> u32 {
    let remain = offset % alignment;
    if remain != 0 {
        offset + alignment - remain
    } else {
        offset
    }
}

fn make_pipeline_layout(
    bind_group_layouts: &[&crate::ShaderDataLayout],
    reserved_vertex_buffers: u32,
) -> super::PipelineLayout {
    let mut group_mappings = Vec::with_capacity(bind_group_layouts.len());
    let mut unsized_buffer_count = 0;
    let mut num_textures = 0u32;
    let mut num_samplers = 0u32;
    let mut num_buffers = reserved_vertex_buffers;
    for layout in bind_group_layouts.iter() {
        let mut targets = Vec::with_capacity(layout.bindings.len());
        for &(_, ref binding) in layout.bindings.iter() {
            targets.push(match *binding {
                crate::ShaderBinding::Texture => {
                    num_textures += 1;
                    num_textures - 1
                }
                crate::ShaderBinding::Sampler => {
                    num_samplers += 1;
                    num_samplers - 1
                }
                crate::ShaderBinding::Buffer => {
                    unsized_buffer_count += 1;
                    num_buffers += 1;
                    num_buffers - 1
                }
                crate::ShaderBinding::TextureArray { .. }
                | crate::ShaderBinding::BufferArray { .. } => unimplemented!(),
                crate::ShaderBinding::AccelerationStructure => {
                    num_buffers += 1;
                    num_buffers - 1
                }
                crate::ShaderBinding::Plain { .. } => {
                    num_buffers += 1;
                    num_buffers - 1
                }
            });
        }

        group_mappings.push(super::ShaderDataMapping {
            visibility: crate::ShaderVisibility::empty(),
            targets: targets.into_boxed_slice(),
        });
    }

    super::PipelineLayout {
        group_mappings: group_mappings.into_boxed_slice(),
        group_infos: bind_group_layouts
            .iter()
            .map(|layout| layout.to_info())
            .collect(),
        sizes_buffer_slot: if unsized_buffer_count != 0 {
            Some(num_buffers)
        } else {
            None
        },
    }
}

impl super::Context {
    fn load_shader(
        &self,
        sf: crate::ShaderFunction,
        bind_group_layouts: &[&crate::ShaderDataLayout],
        vertex_fetch_states: &[crate::VertexFetchState],
        pipeline_layout: &mut super::PipelineLayout,
        flags: ShaderFlags,
    ) -> CompiledShader {
        let ep_index = sf.entry_point_index();
        let ep = &sf.shader.module.entry_points[ep_index];
        let ep_info = sf.shader.info.get_entry_point(ep_index);
        let _ = sf.shader.source;

        let mut module = sf.shader.module.clone();
        crate::Shader::fill_resource_bindings(
            &mut module,
            &mut pipeline_layout.group_infos,
            ep.stage,
            ep_info,
            bind_group_layouts,
        );
        let attribute_mappings =
            crate::Shader::fill_vertex_locations(&mut module, ep_index, vertex_fetch_states);

        // figure out how much workgroup memory is needed for each binding
        let mut wg_memory_sizes = Vec::new();
        for (var_handle, var) in module.global_variables.iter() {
            if var.space == naga::AddressSpace::WorkGroup && !ep_info[var_handle].is_empty() {
                let size = module.types[var.ty].inner.size(module.to_ctx());
                //TODO: use `u32::next_multiple_of`
                wg_memory_sizes.push(((size - 1) | 0xF) + 1); // multiple of 16
            }
        }

        // copy the visibility for convenience
        for (group_mapping, group_info) in pipeline_layout
            .group_mappings
            .iter_mut()
            .zip(pipeline_layout.group_infos.iter())
        {
            group_mapping.visibility = group_info.visibility;
        }

        let mut naga_resources = msl::EntryPointResources::default();
        if let Some(slot) = pipeline_layout.sizes_buffer_slot {
            naga_resources.sizes_buffer = Some(slot as _);
        }
        for (group_index, (group_layout, group_mapping)) in bind_group_layouts
            .iter()
            .zip(pipeline_layout.group_mappings.iter_mut())
            .enumerate()
        {
            for (binding_index, (&(_, proto), &slot)) in group_layout
                .bindings
                .iter()
                .zip(group_mapping.targets.iter())
                .enumerate()
            {
                let res_binding = naga::ResourceBinding {
                    group: group_index as u32,
                    binding: binding_index as u32,
                };
                let bind_target = match proto {
                    crate::ShaderBinding::Texture => msl::BindTarget {
                        texture: Some(slot as _),
                        ..Default::default()
                    },
                    crate::ShaderBinding::Sampler => msl::BindTarget {
                        sampler: Some(msl::BindSamplerTarget::Resource(slot as _)),
                        ..Default::default()
                    },
                    crate::ShaderBinding::Buffer
                    | crate::ShaderBinding::Plain { .. }
                    | crate::ShaderBinding::AccelerationStructure => msl::BindTarget {
                        buffer: Some(slot as _),
                        ..Default::default()
                    },
                    crate::ShaderBinding::TextureArray { .. }
                    | crate::ShaderBinding::BufferArray { .. } => todo!(),
                };
                naga_resources.resources.insert(res_binding, bind_target);
            }
        }

        let naga_options = msl::Options {
            lang_version: (
                (self.info.language_version as u32 >> 16) as u8,
                self.info.language_version as u8,
            ),
            inline_samplers: Default::default(),
            spirv_cross_compatibility: false,
            fake_missing_bindings: false,
            per_entry_point_map: Some((sf.entry_point.to_string(), naga_resources))
                .into_iter()
                .collect(),
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
            zero_initialize_workgroup_memory: false,
        };

        let pipeline_options = msl::PipelineOptions {
            allow_and_force_point_size: flags.contains(ShaderFlags::ALLOW_POINT_SIZE),
            vertex_pulling_transform: false,
            vertex_buffer_mappings: Vec::new(),
        };
        let (source, info) =
            msl::write_string(&module, &sf.shader.info, &naga_options, &pipeline_options).unwrap();

        log::debug!(
            "Naga generated shader for entry point '{}' and stage {:?}\n{}",
            sf.entry_point,
            ep.stage,
            &source
        );

        let options = metal::CompileOptions::new();
        options.set_language_version(self.info.language_version);
        options.set_preserve_invariance(true);

        let library = self
            .device
            .lock()
            .unwrap()
            .new_library_with_source(source.as_ref(), &options)
            .unwrap_or_else(|err| {
                let string = err.replace("\\n", "\n");
                panic!("MSL compilation error:\n{}", string);
            });

        let ep = &module.entry_points[ep_index];
        let name = info.entry_point_names[ep_index].as_ref().unwrap();
        let wg_size = metal::MTLSize {
            width: ep.workgroup_size[0] as _,
            height: ep.workgroup_size[1] as _,
            depth: ep.workgroup_size[2] as _,
        };

        let function = library.get_function(name, None).unwrap();

        CompiledShader {
            library,
            function,
            attribute_mappings,
            wg_size,
            wg_memory_sizes,
        }
    }
}

#[hidden_trait::expose]
impl crate::traits::ShaderDevice for super::Context {
    type ComputePipeline = super::ComputePipeline;
    type RenderPipeline = super::RenderPipeline;

    fn create_compute_pipeline(&self, desc: crate::ComputePipelineDesc) -> super::ComputePipeline {
        let mut layout = make_pipeline_layout(desc.data_layouts, 0);

        objc::rc::autoreleasepool(|| {
            let descriptor = metal::ComputePipelineDescriptor::new();

            let cs = self.load_shader(
                desc.compute,
                desc.data_layouts,
                &[],
                &mut layout,
                ShaderFlags::empty(),
            );
            descriptor.set_compute_function(Some(&cs.function));

            if !desc.name.is_empty() {
                descriptor.set_label(desc.name);
            }

            let raw = self
                .device
                .lock()
                .unwrap()
                .new_compute_pipeline_state(&descriptor)
                .unwrap();

            super::ComputePipeline {
                raw,
                name: desc.name.to_string(),
                lib: cs.library,
                layout,
                wg_size: cs.wg_size,
                wg_memory_sizes: cs.wg_memory_sizes.into_boxed_slice(),
            }
        })
    }

    fn destroy_compute_pipeline(&self, _pipeline: &mut super::ComputePipeline) {
        //TODO: is there a way to release?
    }

    fn create_render_pipeline(&self, desc: crate::RenderPipelineDesc) -> super::RenderPipeline {
        let mut layout = make_pipeline_layout(desc.data_layouts, desc.vertex_fetches.len() as u32);

        let triangle_fill_mode = match desc.primitive.wireframe {
            false => metal::MTLTriangleFillMode::Fill,
            true => metal::MTLTriangleFillMode::Lines,
        };

        let (primitive_class, primitive_type) = match desc.primitive.topology {
            crate::PrimitiveTopology::PointList => (
                metal::MTLPrimitiveTopologyClass::Point,
                metal::MTLPrimitiveType::Point,
            ),
            crate::PrimitiveTopology::LineList => (
                metal::MTLPrimitiveTopologyClass::Line,
                metal::MTLPrimitiveType::Line,
            ),
            crate::PrimitiveTopology::LineStrip => (
                metal::MTLPrimitiveTopologyClass::Line,
                metal::MTLPrimitiveType::LineStrip,
            ),
            crate::PrimitiveTopology::TriangleList => (
                metal::MTLPrimitiveTopologyClass::Triangle,
                metal::MTLPrimitiveType::Triangle,
            ),
            crate::PrimitiveTopology::TriangleStrip => (
                metal::MTLPrimitiveTopologyClass::Triangle,
                metal::MTLPrimitiveType::TriangleStrip,
            ),
        };

        objc::rc::autoreleasepool(|| {
            let descriptor = metal::RenderPipelineDescriptor::new();

            let vs = self.load_shader(
                desc.vertex,
                desc.data_layouts,
                desc.vertex_fetches,
                &mut layout,
                match primitive_class {
                    metal::MTLPrimitiveTopologyClass::Point => ShaderFlags::ALLOW_POINT_SIZE,
                    _ => ShaderFlags::empty(),
                },
            );
            descriptor.set_vertex_function(Some(&vs.function));
            descriptor.set_raster_sample_count(desc.multisample_state.sample_count as _);
            descriptor.set_alpha_to_coverage_enabled(desc.multisample_state.alpha_to_coverage);

            // Fragment shader
            let fs = desc.fragment.map(|desc_fragment| {
                self.load_shader(
                    desc_fragment,
                    desc.data_layouts,
                    &[],
                    &mut layout,
                    ShaderFlags::empty(),
                )
            });
            descriptor.set_fragment_function(fs.as_ref().map(|fs| fs.function.as_ref()));

            let vertex_descriptor = metal::VertexDescriptor::new();
            for (i, vf) in desc.vertex_fetches.iter().enumerate() {
                let buffer_desc = vertex_descriptor.layouts().object_at(i as u64).unwrap();
                buffer_desc.set_stride(vf.layout.stride as u64);
                buffer_desc.set_step_function(if vf.instanced {
                    metal::MTLVertexStepFunction::PerInstance
                } else {
                    metal::MTLVertexStepFunction::PerVertex
                });
            }
            for (i, mapping) in vs.attribute_mappings.into_iter().enumerate() {
                let attribute_desc = vertex_descriptor.attributes().object_at(i as u64).unwrap();
                let vf = &desc.vertex_fetches[mapping.buffer_index];
                let (_, attrib) = vf.layout.attributes[mapping.attribute_index];
                let (vertex_format, _) = super::map_vertex_format(attrib.format);
                attribute_desc.set_format(vertex_format);
                attribute_desc.set_buffer_index(mapping.buffer_index as u64);
                attribute_desc.set_offset(attrib.offset as u64);
            }
            descriptor.set_vertex_descriptor(Some(vertex_descriptor));

            for (i, ct) in desc.color_targets.iter().enumerate() {
                let at_descriptor = descriptor.color_attachments().object_at(i as u64).unwrap();
                at_descriptor.set_pixel_format(super::map_texture_format(ct.format));

                let mut write_mask = metal::MTLColorWriteMask::empty();
                if ct.write_mask.contains(crate::ColorWrites::RED) {
                    write_mask |= metal::MTLColorWriteMask::Red;
                }
                if ct.write_mask.contains(crate::ColorWrites::GREEN) {
                    write_mask |= metal::MTLColorWriteMask::Green;
                }
                if ct.write_mask.contains(crate::ColorWrites::BLUE) {
                    write_mask |= metal::MTLColorWriteMask::Blue;
                }
                if ct.write_mask.contains(crate::ColorWrites::ALPHA) {
                    write_mask |= metal::MTLColorWriteMask::Alpha;
                }
                at_descriptor.set_write_mask(write_mask);

                if let Some(ref blend) = ct.blend {
                    at_descriptor.set_blending_enabled(true);
                    let (color_op, color_src, color_dst) = map_blend_component(&blend.color);
                    let (alpha_op, alpha_src, alpha_dst) = map_blend_component(&blend.alpha);

                    at_descriptor.set_rgb_blend_operation(color_op);
                    at_descriptor.set_source_rgb_blend_factor(color_src);
                    at_descriptor.set_destination_rgb_blend_factor(color_dst);

                    at_descriptor.set_alpha_blend_operation(alpha_op);
                    at_descriptor.set_source_alpha_blend_factor(alpha_src);
                    at_descriptor.set_destination_alpha_blend_factor(alpha_dst);
                }
            }

            let depth_stencil = match desc.depth_stencil {
                Some(ref ds) => {
                    let raw_format = super::map_texture_format(ds.format);
                    descriptor.set_depth_attachment_pixel_format(raw_format);
                    //TODO: descriptor.set_stencil_attachment_pixel_format(raw_format);

                    let ds_descriptor = create_depth_stencil_desc(ds);
                    let raw = self
                        .device
                        .lock()
                        .unwrap()
                        .new_depth_stencil_state(&ds_descriptor);
                    Some((raw, ds.bias))
                }
                None => None,
            };

            if !desc.name.is_empty() {
                descriptor.set_label(desc.name);
            }

            let raw = self
                .device
                .lock()
                .unwrap()
                .new_render_pipeline_state(&descriptor)
                .unwrap();

            super::RenderPipeline {
                raw,
                name: desc.name.to_string(),
                vs_lib: vs.library,
                fs_lib: fs.map(|fs| fs.library),
                layout,
                primitive_type,
                triangle_fill_mode,
                front_winding: match desc.primitive.front_face {
                    crate::FrontFace::Cw => metal::MTLWinding::Clockwise,
                    crate::FrontFace::Ccw => metal::MTLWinding::CounterClockwise,
                },
                cull_mode: match desc.primitive.cull_mode {
                    None => metal::MTLCullMode::None,
                    Some(crate::Face::Front) => metal::MTLCullMode::Front,
                    Some(crate::Face::Back) => metal::MTLCullMode::Back,
                },
                depth_clip_mode: if desc.primitive.unclipped_depth {
                    metal::MTLDepthClipMode::Clamp
                } else {
                    metal::MTLDepthClipMode::Clip
                },
                depth_stencil,
            }
        })
    }

    fn destroy_render_pipeline(&self, _pipeline: &mut super::RenderPipeline) {
        //TODO: is there a way to release?
    }
}
