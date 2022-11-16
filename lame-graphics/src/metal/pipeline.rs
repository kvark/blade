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
        //Bf::ConstantAlpha => BlendAlpha,
        //Bf::OneMinusConstantAlpha => OneMinusBlendAlpha,
        Bf::SrcAlphaSaturated => SourceAlphaSaturated,
        //Bf::Src1 => Source1Color,
        //Bf::OneMinusSrc1 => OneMinusSource1Color,
        //Bf::Src1Alpha => Source1Alpha,
        //Bf::OneMinusSrc1Alpha => OneMinusSource1Alpha,
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
    use metal::MTLStencilOperation::*;
    use crate::StencilOperation as So;

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

fn map_compare_function(fun: crate::CompareFunction) -> metal::MTLCompareFunction {
    use metal::MTLCompareFunction::*;
    use crate::CompareFunction as Cf;

    match fun {
        Cf::Never => Never,
        Cf::Less => Less,
        Cf::LessEqual => LessEqual,
        Cf::Equal => Equal,
        Cf::GreaterEqual => GreaterEqual,
        Cf::Greater => Greater,
        Cf::NotEqual => NotEqual,
        Cf::Always => Always,
    }
}

fn create_stencil_desc(
    face: &crate::StencilFaceState,
    read_mask: u32,
    write_mask: u32,
) -> metal::StencilDescriptor {
    let desc = metal::StencilDescriptor::new();
    desc.set_stencil_compare_function(map_compare_function(face.compare));
    desc.set_read_mask(read_mask);
    desc.set_write_mask(write_mask);
    desc.set_stencil_failure_operation(map_stencil_op(face.fail_op));
    desc.set_depth_failure_operation(map_stencil_op(face.depth_fail_op));
    desc.set_depth_stencil_pass_operation(map_stencil_op(face.pass_op));
    desc
}

fn create_depth_stencil_desc(state: &crate::DepthStencilState) -> metal::DepthStencilDescriptor {
    let desc = metal::DepthStencilDescriptor::new();
    desc.set_depth_compare_function(map_compare_function(state.depth_compare));
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

impl super::Context {
    fn load_shader(
        &self,
        sf: crate::ShaderFunction,
        flags: ShaderFlags,
        naga_stage: naga::ShaderStage,
    ) -> CompiledShader {
        let pipeline_options = naga::back::msl::PipelineOptions {
            allow_point_size: flags.contains(ShaderFlags::ALLOW_POINT_SIZE),
        };
        let msl_version = metal::MTLLanguageVersion::V2_2;

        let naga_options = naga::back::msl::Options {
            lang_version: (2, 2),
            inline_samplers: Default::default(),
            spirv_cross_compatibility: false,
            fake_missing_bindings: false,
            per_stage_map: naga::back::msl::PerStageMap {
                vs: naga::back::msl::PerStageResources::default(),
                fs: naga::back::msl::PerStageResources::default(),
                cs: naga::back::msl::PerStageResources::default(),
            },
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        };

        let module = &sf.shader.module;
        let (source, info) = naga::back::msl::write_string(
            module,
            &sf.shader.info,
            &naga_options,
            &pipeline_options,
        ).unwrap();

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
            .unwrap();

        let ep_index = module
            .entry_points
            .iter()
            .position(|ep| ep.stage == naga_stage && ep.name == sf.entry_point)
            .expect("Entry point not found in the shader");
        let ep = &module.entry_points[ep_index];
        let name = info.entry_point_names[ep_index]
            .as_ref()
            .unwrap();
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

    fn merge_layouts<'a>(multi_layouts: &[(&'a crate::Shader, crate::ShaderVisibility)]) -> Vec<(Option<&'a crate::ShaderDataLayout>, crate::ShaderVisibility)> {
        let count = multi_layouts.iter().map(|(shader, _)| shader.bind_groups.len()).max().unwrap_or_default();
        let mut merged = Vec::with_capacity(count);
        for i in 0 .. count {
            let mut layout_maybe = None;
            let mut visibility = crate::ShaderVisibility::empty();
            for &(shader, shader_visibility) in multi_layouts {
                if let Some(data_layout) = shader.bind_groups.get(i).map_or(None, |opt| opt.as_ref()) {
                    visibility |= shader_visibility;
                    if let Some(layout) = layout_maybe {
                        assert_eq!(data_layout, layout);
                    } else {
                        layout_maybe = Some(data_layout);
                    }
                }
            }
            merged.push((layout_maybe, visibility));
        }
        merged
    }

    pub fn create_render_pipeline(&self, desc: crate::RenderPipelineDesc) -> super::RenderPipeline {
        let mut bind_group_infos = Vec::new();
        let layouts = Self::merge_layouts(&[
            (desc.vertex.shader, crate::ShaderVisibility::VERTEX),
            (desc.fragment.shader, crate::ShaderVisibility::FRAGMENT),
        ]);
        let mut num_buffers = 0;
        let mut num_textures = 0;
        for (layout, visibility) in layouts {
            let bindings = layout.map_or(&[][..], |l| &l.bindings);
            let mut targets = Vec::with_capacity(bindings.len());
            let mut plain_data_size = 0;
            for &(_, ref binding) in bindings.iter() {
                let target = match *binding {
                    crate::ShaderBinding::Resource {
                        ty: crate::ResourceType::Texture,
                    } => {
                        num_textures += 1;
                        num_textures - 1
                    },
                    crate::ShaderBinding::Plain {
                        ty,
                        container,
                    } => {
                        let offset = plain_data_size;
                        let scalar_size = match ty {
                            crate::PlainType::F32 => 4u32,
                        };
                        let count = match container {
                            crate::PlainContainer::Scalar => 1u32,
                            crate::PlainContainer::Vector(size) => size as u32,
                        };
                        plain_data_size += scalar_size * count;
                        //TODO: take alignment into account
                        offset
                    }
                };
                targets.push(target);
            }

            let plain_buffer_slot = if plain_data_size != 0 {
                num_buffers += 1;
                Some(num_buffers - 1)
            } else {
                None
            };
            bind_group_infos.push(super::BindGroupInfo {
                visibility,
                targets: targets.into_boxed_slice(),
                plain_buffer_slot,
                plain_data_size,
            });
        }

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
                match primitive_class {
                    metal::MTLPrimitiveTopologyClass::Point => ShaderFlags::ALLOW_POINT_SIZE,
                    _ => ShaderFlags::empty(),
                },
                naga::ShaderStage::Vertex,
            );
            descriptor.set_vertex_function(Some(&vs.function));

            // Fragment shader
            let fs = self.load_shader(
                desc.fragment,
                ShaderFlags::empty(),
                naga::ShaderStage::Fragment,
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
                bind_groups: bind_group_infos.into_boxed_slice(),
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
