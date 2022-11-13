impl super::CommandEncoder {
    pub fn start(&mut self) {
        let queue = self.queue.lock().unwrap();
        self.raw = Some(objc::rc::autoreleasepool(|| {
            queue.new_command_buffer().to_owned()
        }));
    }

    pub fn with_render_targets(&mut self, targets: crate::RenderTargetSet) -> super::RenderCommandEncoder {
        let raw = objc::rc::autoreleasepool(|| {
            let descriptor = metal::RenderPassDescriptor::new();

            for (i, rt) in targets.colors.iter().enumerate() {
                let at_descriptor = descriptor.color_attachments().object_at(i as u64).unwrap();
                at_descriptor.set_texture(Some(rt.view.as_ref()));

                let load_action = match rt.init_op {
                    crate::InitOp::Load => metal::MTLLoadAction::Load,
                    crate::InitOp::Clear(color) => {
                        let clear_color = match color {
                            crate::TextureColor::TransparentBlack => metal::MTLClearColor {
                                red: 0.0,
                                green: 0.0,
                                blue: 0.0,
                                alpha: 0.0,
                            },
                            crate::TextureColor::OpaqueBlack => metal::MTLClearColor {
                                red: 0.0,
                                green: 0.0,
                                blue: 0.0,
                                alpha: 1.0,
                            },
                            crate::TextureColor::White => metal::MTLClearColor {
                                red: 1.0,
                                green: 1.0,
                                blue: 1.0,
                                alpha: 1.0,
                            },
                        };
                        at_descriptor.set_clear_color(clear_color);
                        metal::MTLLoadAction::Clear
                    },
                };
                at_descriptor.set_load_action(load_action);

                let store_action = match rt.finish_op {
                    crate::FinishOp::Store => metal::MTLStoreAction::Store,
                    crate::FinishOp::Discard => metal::MTLStoreAction::DontCare,
                    crate::FinishOp::ResolveTo(ref view) => {
                        at_descriptor.set_resolve_texture(Some(view.as_ref()));
                        metal::MTLStoreAction::MultisampleResolve
                    },
                };
                at_descriptor.set_store_action(store_action);
            }

            if let Some(ref rt) = targets.depth_stencil {
                let at_descriptor = descriptor.depth_attachment().unwrap();
                at_descriptor.set_texture(Some(rt.view.as_ref()));
                let load_action = match rt.init_op {
                    crate::InitOp::Load => metal::MTLLoadAction::Load,
                    crate::InitOp::Clear(color) => {
                        let clear_depth = match color {
                            crate::TextureColor::TransparentBlack |
                            crate::TextureColor::OpaqueBlack => 0.0,
                            crate::TextureColor::White => 1.0,
                        };
                        at_descriptor.set_clear_depth(clear_depth);
                        metal::MTLLoadAction::Clear
                    },
                };
                let store_action = match rt.finish_op {
                    crate::FinishOp::Store => metal::MTLStoreAction::Store,
                    crate::FinishOp::Discard => metal::MTLStoreAction::DontCare,
                    crate::FinishOp::ResolveTo(_) => panic!("Can't resolve depth texture"),
                };
                at_descriptor.set_load_action(load_action);
                at_descriptor.set_store_action(store_action);
            }

            self.raw.as_mut().unwrap().new_render_command_encoder(descriptor).to_owned()
        });

        super::RenderCommandEncoder {
            owner: self.raw.as_mut().unwrap(),
            raw,
        }
    }
}

impl super::RenderCommandEncoder<'_> {
    pub fn with_pipeline(&mut self, pipeline: &super::RenderPipeline) -> super::RenderPipelineContext {
        self.raw.set_render_pipeline_state(&pipeline.raw);
        self.raw.set_front_facing_winding(pipeline.front_winding);
        self.raw.set_cull_mode(pipeline.cull_mode);
        self.raw.set_triangle_fill_mode(pipeline.triangle_fill_mode);
        if let Some(depth_clip) = pipeline.depth_clip_mode {
            self.raw.set_depth_clip_mode(depth_clip);
        }
        if let Some((ref state, bias)) = pipeline.depth_stencil {
            self.raw.set_depth_stencil_state(state);
            self.raw.set_depth_bias(bias.constant as f32, bias.slope_scale, bias.clamp);
        }

        super::RenderPipelineContext {
            encoder: &mut self.raw,
            primitive_type: pipeline.primitive_type,
        }
    }
}

impl super::RenderPipelineContext<'_> {
    pub fn bind_data<D: crate::ShaderData>(&mut self, group: u32, data: &D) {

    }

    pub fn draw(&mut self, first_vertex: u32, vertex_count: u32, first_instance: u32, instance_count: u32) {
        if first_instance != 0 {
            self.encoder.draw_primitives_instanced_base_instance(
                self.primitive_type,
                first_vertex as _,
                vertex_count as _,
                instance_count as _,
                first_instance as _,
            );
        } else if instance_count != 1 {
            self.encoder.draw_primitives_instanced(
                self.primitive_type,
                first_vertex as _,
                vertex_count as _,
                instance_count as _,
            );
        } else {
            self.encoder.draw_primitives(
                self.primitive_type,
                first_vertex as _,
                vertex_count as _,
            );
        }
    }
}
