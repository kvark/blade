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
    wg_size: metal::MTLSize,
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

impl super::PipelineLayout {
    fn new(bind_group_layouts: &[&crate::ShaderDataLayout]) -> Self {
        let mut bind_group_infos = Vec::with_capacity(bind_group_layouts.len());
        let mut unsized_buffer_count = 0;
        let mut num_textures = 0u32;
        let mut num_samplers = 0u32;
        let mut num_buffers = 0u32;
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
                    crate::ShaderBinding::Plain { .. } => {
                        num_buffers += 1;
                        num_buffers - 1
                    }
                });
            }

            bind_group_infos.push(super::BindGroupInfo {
                visibility: crate::ShaderVisibility::empty(),
                targets: targets.into_boxed_slice(),
            });
        }

        super::PipelineLayout {
            bind_groups: bind_group_infos.into_boxed_slice(),
            sizes_buffer_slot: if unsized_buffer_count != 0 {
                Some(num_buffers)
            } else {
                None
            },
        }
    }
}

impl super::Context {
    fn load_shader(
        &self,
        sf: crate::ShaderFunction,
        bind_group_layouts: &[&crate::ShaderDataLayout],
        layout: &mut super::PipelineLayout,
        flags: ShaderFlags,
    ) -> CompiledShader {
        let mut naga_resources = msl::PerStageResources::default();
        if let Some(slot) = layout.sizes_buffer_slot {
            naga_resources.sizes_buffer = Some(slot as _);
        }

        let ep_index = sf.entry_point_index();
        let ep_info = sf.shader.info.get_entry_point(ep_index);
        let naga_stage = sf.shader.module.entry_points[ep_index].stage;
        let mut module = sf.shader.module.clone();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(&module.types, &module.constants).unwrap();

        for (handle, var) in module.global_variables.iter_mut() {
            if ep_info[handle].is_empty() {
                continue;
            }
            let access = match var.space {
                naga::AddressSpace::Storage { access } => access,
                naga::AddressSpace::Uniform | naga::AddressSpace::Handle => {
                    naga::StorageAccess::empty()
                }
                _ => continue,
            };

            assert_eq!(var.binding, None);
            let var_name = var.name.as_ref().unwrap();
            for (group_index, (bgl, bgi)) in bind_group_layouts
                .iter()
                .zip(layout.bind_groups.iter_mut())
                .enumerate()
            {
                if let Some((binding_index, (&(_, proto_binding), &resource_index))) = bgl
                    .bindings
                    .iter()
                    .zip(bgi.targets.iter())
                    .enumerate()
                    .find(|(_, (&(name, _), _))| name == var_name)
                {
                    let res_binding = naga::ResourceBinding {
                        group: group_index as u32,
                        binding: binding_index as u32,
                    };
                    let (expected_proto, bind_target) = match module.types[var.ty].inner {
                        naga::TypeInner::Image { .. } => (
                            crate::ShaderBinding::Texture,
                            msl::BindTarget {
                                texture: Some(resource_index as _),
                                ..Default::default()
                            },
                        ),
                        naga::TypeInner::Sampler { .. } => (
                            crate::ShaderBinding::Sampler,
                            msl::BindTarget {
                                sampler: Some(msl::BindSamplerTarget::Resource(
                                    resource_index as _,
                                )),
                                ..Default::default()
                            },
                        ),
                        _ => {
                            let type_layout = &layouter[var.ty];
                            let expected_proto = if access.is_empty() {
                                crate::ShaderBinding::Plain {
                                    size: type_layout.size,
                                }
                            } else {
                                crate::ShaderBinding::Buffer
                            };
                            (
                                expected_proto,
                                msl::BindTarget {
                                    buffer: Some(resource_index as _),
                                    ..Default::default()
                                },
                            )
                        }
                    };
                    assert_eq!(
                        proto_binding, expected_proto,
                        "Mismatched type for binding '{}'",
                        var_name
                    );
                    assert_eq!(var.binding, None);
                    var.binding = Some(res_binding.clone());
                    naga_resources.resources.insert(res_binding, bind_target);
                    bgi.visibility |= naga_stage.into();
                    break;
                }
            }

            assert!(
                var.binding.is_some(),
                "Unable to resolve binding for '{}'",
                var_name
            );
        }

        let msl_version = metal::MTLLanguageVersion::V2_2;
        let naga_options = msl::Options {
            lang_version: ((msl_version as u32 >> 16) as u8, msl_version as u8),
            inline_samplers: Default::default(),
            spirv_cross_compatibility: false,
            fake_missing_bindings: false,
            per_stage_map: match naga_stage {
                naga::ShaderStage::Compute => msl::PerStageMap {
                    cs: naga_resources,
                    ..Default::default()
                },
                naga::ShaderStage::Vertex => msl::PerStageMap {
                    vs: naga_resources,
                    ..Default::default()
                },
                naga::ShaderStage::Fragment => msl::PerStageMap {
                    fs: naga_resources,
                    ..Default::default()
                },
            },
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        };

        let pipeline_options = msl::PipelineOptions {
            allow_point_size: flags.contains(ShaderFlags::ALLOW_POINT_SIZE),
        };
        let (source, info) =
            msl::write_string(&module, &sf.shader.info, &naga_options, &pipeline_options).unwrap();

        log::debug!(
            "Naga generated shader for entry point '{}' and stage {:?}\n{}",
            sf.entry_point,
            naga_stage,
            &source
        );

        let options = metal::CompileOptions::new();
        options.set_language_version(msl_version);
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
            wg_size,
        }
    }

    pub fn create_compute_pipeline(
        &self,
        desc: crate::ComputePipelineDesc,
    ) -> super::ComputePipeline {
        let mut layout = super::PipelineLayout::new(desc.data_layouts);

        objc::rc::autoreleasepool(|| {
            let descriptor = metal::ComputePipelineDescriptor::new();

            let cs = self.load_shader(
                desc.compute,
                desc.data_layouts,
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
                lib: cs.library,
                layout,
                wg_size: cs.wg_size,
            }
        })
    }

    pub fn create_render_pipeline(&self, desc: crate::RenderPipelineDesc) -> super::RenderPipeline {
        let mut layout = super::PipelineLayout::new(desc.data_layouts);

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
                &mut layout,
                match primitive_class {
                    metal::MTLPrimitiveTopologyClass::Point => ShaderFlags::ALLOW_POINT_SIZE,
                    _ => ShaderFlags::empty(),
                },
            );
            descriptor.set_vertex_function(Some(&vs.function));

            // Fragment shader
            let fs = self.load_shader(
                desc.fragment,
                desc.data_layouts,
                &mut layout,
                ShaderFlags::empty(),
            );
            descriptor.set_fragment_function(Some(&fs.function));

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
                vs_lib: vs.library,
                fs_lib: fs.library,
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
}
