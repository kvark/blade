use std::marker::PhantomData;

fn map_origin(extent: &crate::Extent) -> metal::MTLOrigin {
    metal::MTLOrigin {
        x: extent.width as u64,
        y: extent.height as u64,
        z: extent.depth as u64,
    }
}

fn map_extent(extent: &crate::Extent) -> metal::MTLSize {
    metal::MTLSize {
        width: extent.width as u64,
        height: extent.height as u64,
        depth: extent.depth as u64,
    }
}

struct ShaderDataEncoder<'a> {
    //raw: metal::ArgumentEncoderRef,
    cs_encoder: Option<&'a metal::ComputeCommandEncoderRef>,
    vs_encoder: Option<&'a metal::RenderCommandEncoderRef>,
    fs_encoder: Option<&'a metal::RenderCommandEncoderRef>,
    targets: &'a [u32],
    plain_data: &'a mut [u8],
}

impl crate::ShaderDataEncoder for ShaderDataEncoder<'_> {
    fn set_buffer(&mut self, index: u32, piece: crate::BufferPiece) {
        let slot = self.targets[index as usize] as _;
        let value = Some(piece.buffer.as_ref());
        if let Some(encoder) = self.vs_encoder {
            encoder.set_vertex_buffer(slot, value, piece.offset);
        }
        if let Some(encoder) = self.fs_encoder {
            encoder.set_fragment_buffer(slot, value, piece.offset);
        }
        if let Some(encoder) = self.cs_encoder {
            encoder.set_buffer(slot, value, piece.offset);
        }
    }
    fn set_texture(&mut self, index: u32, view: super::TextureView) {
        //self.raw.set_texture(index as _, view.as_ref());
        let slot = self.targets[index as usize] as _;
        let value = Some(view.as_ref());
        if let Some(encoder) = self.vs_encoder {
            encoder.set_vertex_texture(slot, value);
        }
        if let Some(encoder) = self.fs_encoder {
            encoder.set_fragment_texture(slot, value);
        }
        if let Some(encoder) = self.cs_encoder {
            encoder.set_texture(slot, value);
        }
    }
    fn set_sampler(&mut self, index: u32, sampler: super::Sampler) {
        //self.raw.set_sampler_state(index as _, sampler.as_ref());
        let slot = self.targets[index as usize] as _;
        let value = Some(sampler.as_ref());
        if let Some(encoder) = self.vs_encoder {
            encoder.set_vertex_sampler_state(slot, value);
        }
        if let Some(encoder) = self.fs_encoder {
            encoder.set_fragment_sampler_state(slot, value);
        }
        if let Some(encoder) = self.cs_encoder {
            encoder.set_sampler_state(slot, value);
        }
    }
    fn set_plain<P: bytemuck::Pod>(&mut self, index: u32, data: P) {
        let offset = self.targets[index as usize] as usize;
        unsafe {
            std::ptr::write_unaligned(self.plain_data.as_mut_ptr().add(offset) as *mut P, data);
        }
    }
}

impl super::CommandEncoder {
    pub fn start(&mut self) {
        let queue = self.queue.lock().unwrap();
        self.raw = Some(objc::rc::autoreleasepool(|| {
            let cmd_buf = queue.new_command_buffer();
            if !self.name.is_empty() {
                cmd_buf.set_label(&self.name);
            }
            cmd_buf.to_owned()
        }));
    }

    pub fn with_transfers(&mut self) -> super::TransferCommandEncoder {
        let raw = objc::rc::autoreleasepool(|| {
            self.raw.as_mut().unwrap().new_blit_command_encoder().to_owned()
        });
        super::TransferCommandEncoder {
            raw,
            phantom: PhantomData,
        }
    }

    pub fn with_pipeline<'p>(&'p mut self, pipeline: &'p super::ComputePipeline) -> super::ComputePipelineContext<'p> {
        let max_data_size = pipeline.layout.bind_groups.iter().map(|bg| bg.plain_data_size as usize).max().unwrap_or_default();
        self.plain_data.resize(max_data_size, 0);

        let encoder = objc::rc::autoreleasepool(|| {
            self.raw.as_mut().unwrap().new_compute_command_encoder().to_owned()
        });
        encoder.set_compute_pipeline_state(&pipeline.raw);
        if let Some(index) = pipeline.layout.sizes_buffer_slot {
            //TODO: get real sizes
            let runtime_sizes = [0u8; 8];
            encoder.set_bytes(index as _, runtime_sizes.len() as _, runtime_sizes.as_ptr() as *const _);
        }

        super::ComputePipelineContext {
            encoder,
            bind_groups: &pipeline.layout.bind_groups,
            plain_data: self.plain_data.as_mut(),
            wg_size: pipeline.wg_size,
        }
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
            raw,
            plain_data: &mut self.plain_data,
        }
    }
}

impl super::TransferCommandEncoder<'_> {
    pub fn copy_buffer_to_buffer(&mut self, src: crate::BufferPiece, dst: crate::BufferPiece, size: u64) {
        self.raw.copy_from_buffer(
            src.buffer.as_ref(),
            src.offset,
            dst.buffer.as_ref(),
            dst.offset,
            size,
        );
    }
    pub fn copy_texture_to_texture(&mut self, src: crate::TexturePiece, dst: crate::TexturePiece, size: crate::Extent) {
        self.raw.copy_from_texture(
            src.texture.as_ref(),
            src.array_layer as u64,
            src.mip_level as u64,
            map_origin(&src.origin),
            map_extent(&size),
            dst.texture.as_ref(),
            dst.array_layer as u64,
            dst.mip_level as u64,
            map_origin(&dst.origin),
        );
    }
    pub fn copy_buffer_to_texture(&mut self, src: crate::BufferPiece, bytes_per_row: u32, dst: crate::TexturePiece, size: crate::Extent) {
        self.raw.copy_from_buffer_to_texture(
            src.buffer.as_ref(),
            src.offset,
            bytes_per_row as u64,
            0,
            map_extent(&size),
            dst.texture.as_ref(),
            dst.array_layer as u64,
            dst.mip_level as u64,
            map_origin(&dst.origin),
            metal::MTLBlitOption::empty(),
        );
    }
    pub fn copy_texture_to_buffer(&mut self, src: crate::TexturePiece, dst: crate::BufferPiece, bytes_per_row: u32, size: crate::Extent) {
        self.raw.copy_from_texture_to_buffer(
                src.texture.as_ref(),
                src.array_layer as u64,
                src.mip_level as u64,
                map_origin(&src.origin),
                map_extent(&size),
                dst.buffer.as_ref(),
                dst.offset,
                bytes_per_row as u64,
                0,
                metal::MTLBlitOption::empty(),
            );
    }
}

impl Drop for super::TransferCommandEncoder<'_> {
    fn drop(&mut self) {
        self.raw.end_encoding();
    }
}

impl super::RenderCommandEncoder<'_> {
    pub fn with_pipeline<'p>(&'p mut self, pipeline: &'p super::RenderPipeline) -> super::RenderPipelineContext<'p> {
        self.raw.set_render_pipeline_state(&pipeline.raw);
        if let Some(index) = pipeline.layout.sizes_buffer_slot {
            //TODO: get real sizes
            let runtime_sizes = [0u8; 8];
            self.raw.set_vertex_bytes(index as _, runtime_sizes.len() as _, runtime_sizes.as_ptr() as *const _);
            self.raw.set_fragment_bytes(index as _, runtime_sizes.len() as _, runtime_sizes.as_ptr() as *const _);
        }

        self.raw.set_front_facing_winding(pipeline.front_winding);
        self.raw.set_cull_mode(pipeline.cull_mode);
        self.raw.set_triangle_fill_mode(pipeline.triangle_fill_mode);
        self.raw.set_depth_clip_mode(pipeline.depth_clip_mode);
        if let Some((ref state, bias)) = pipeline.depth_stencil {
            self.raw.set_depth_stencil_state(state);
            self.raw.set_depth_bias(bias.constant as f32, bias.slope_scale, bias.clamp);
        }

        let max_data_size = pipeline.layout.bind_groups.iter().map(|bg| bg.plain_data_size as usize).max().unwrap_or_default();
        self.plain_data.resize(max_data_size, 0);

        super::RenderPipelineContext {
            encoder: &mut self.raw,
            primitive_type: pipeline.primitive_type,
            bind_groups: &pipeline.layout.bind_groups,
            plain_data: self.plain_data.as_mut(),
        }
    }
}

impl Drop for super::RenderCommandEncoder<'_> {
    fn drop(&mut self) {
        self.raw.end_encoding();
    }
}

impl super::ComputePipelineContext<'_> {
    pub fn bind_data<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
        let info = &self.bind_groups[group as usize];

        data.fill(ShaderDataEncoder {
            cs_encoder: if info.visibility.contains(crate::ShaderVisibility::COMPUTE) {
                Some(self.encoder.as_ref())
            } else {
                None
            },
            vs_encoder: None,
            fs_encoder: None,
            targets: &info.targets,
            plain_data: self.plain_data,
        });

        if let Some(slot) = info.plain_buffer_slot {
            let data = self.plain_data.as_ptr() as *const _;
            if info.visibility.contains(crate::ShaderVisibility::COMPUTE) {
                self.encoder.set_bytes(slot as _, info.plain_data_size as _, data);
            }
        }
    }

    pub fn dispatch(&mut self, groups: [u32; 3]) {
        let raw_count = metal::MTLSize {
            width: groups[0] as u64,
            height: groups[1] as u64,
            depth: groups[2] as u64,
        };
        self.encoder.dispatch_thread_groups(raw_count, self.wg_size);
    }
}

impl Drop for super::ComputePipelineContext<'_> {
    fn drop(&mut self) {
        self.encoder.end_encoding();
    }
}

impl super::RenderPipelineContext<'_> {
    pub fn bind_data<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
        let info = &self.bind_groups[group as usize];

        data.fill(ShaderDataEncoder {
            cs_encoder: None,
            vs_encoder: if info.visibility.contains(crate::ShaderVisibility::VERTEX) {
                Some(self.encoder.as_ref())
            } else {
                None
            },
            fs_encoder:  if info.visibility.contains(crate::ShaderVisibility::FRAGMENT) {
                Some(self.encoder.as_ref())
            } else {
                None
            },
            targets: &info.targets,
            plain_data: self.plain_data,
        });

        if let Some(slot) = info.plain_buffer_slot {
            let data = self.plain_data.as_ptr() as *const _;
            if info.visibility.contains(crate::ShaderVisibility::VERTEX) {
                self.encoder.set_vertex_bytes(slot as _, info.plain_data_size as _, data);
            }
            if info.visibility.contains(crate::ShaderVisibility::FRAGMENT) {
                self.encoder.set_fragment_bytes(slot as _, info.plain_data_size as _, data);
            }
        }
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
