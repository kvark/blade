impl<T: bytemuck::Pod> crate::ShaderBindable for T {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        //TODO
    }
}
impl crate::ShaderBindable for super::TextureView {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        //TODO
    }
}
impl crate::ShaderBindable for super::Sampler {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        //TODO
    }
}
impl crate::ShaderBindable for crate::BufferPiece {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        //TODO
    }
}

impl super::CommandEncoder {
    pub fn start(&mut self) {
        self.commands.clear();
    }

    pub fn init_texture(&mut self, _texture: super::Texture) {}

    pub fn present(&mut self, _frame: super::Frame) {
        unimplemented!()
    }

    pub fn transfer(&mut self) -> super::PassEncoder<()> {
        super::PassEncoder {
            commands: &mut self.commands,
            pipeline: Default::default(),
        }
    }

    pub fn compute(&mut self) -> super::PassEncoder<super::ComputePipeline> {
        super::PassEncoder {
            commands: &mut self.commands,
            pipeline: Default::default(),
        }
    }

    pub fn render(
        &mut self,
        targets: crate::RenderTargetSet,
    ) -> super::PassEncoder<super::RenderPipeline> {
        super::PassEncoder {
            commands: &mut self.commands,
            pipeline: Default::default(),
        }
    }
}

impl super::PassEncoder<'_, super::ComputePipeline> {
    pub fn with(&mut self, pipeline: &super::ComputePipeline) -> super::PipelineEncoder {
        super::PipelineEncoder {
            commands: self.commands,
        }
    }
}

#[hidden_trait::expose]
impl crate::traits::TransferEncoder for super::PassEncoder<'_, ()> {
    fn fill_buffer(&mut self, dst: crate::BufferPiece, size: u64, value: u8) {
        self.commands.push(super::Command::FillBuffer {
            dst: dst.into(),
            size,
            value,
        });
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: crate::BufferPiece,
        dst: crate::BufferPiece,
        size: u64,
    ) {
        self.commands.push(super::Command::CopyBufferToBuffer {
            src: src.into(),
            dst: dst.into(),
            size,
        });
    }
    fn copy_texture_to_texture(
        &mut self,
        src: crate::TexturePiece,
        dst: crate::TexturePiece,
        size: crate::Extent,
    ) {
        self.commands.push(super::Command::CopyTextureToTexture {
            src: src.into(),
            dst: dst.into(),
            size,
        });
    }

    fn copy_buffer_to_texture(
        &mut self,
        src: crate::BufferPiece,
        bytes_per_row: u32,
        dst: crate::TexturePiece,
        size: crate::Extent,
    ) {
        self.commands.push(super::Command::CopyBufferToTexture {
            src: src.into(),
            bytes_per_row,
            dst: dst.into(),
            size,
        });
    }

    fn copy_texture_to_buffer(
        &mut self,
        src: crate::TexturePiece,
        dst: crate::BufferPiece,
        bytes_per_row: u32,
        size: crate::Extent,
    ) {
        self.commands.push(super::Command::CopyTextureToBuffer {
            src: src.into(),
            dst: dst.into(),
            bytes_per_row,
            size,
        });
    }
}

#[hidden_trait::expose]
impl crate::traits::PipelineEncoder for super::PipelineEncoder<'_> {
    fn bind<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
        data.fill(super::PipelineContext {});
    }
}

#[hidden_trait::expose]
impl crate::traits::ComputePipelineEncoder for super::PipelineEncoder<'_> {
    fn dispatch(&mut self, groups: [u32; 3]) {
        self.commands.push(super::Command::Dispatch(groups));
    }
}

impl super::Command {
    pub(super) unsafe fn execute(&self, gl: &glow::Context) {
        use glow::HasContext as _;
        match *self {
            Self::Draw {
                topology,
                start_vertex,
                vertex_count,
                instance_count,
            } => {
                if instance_count == 1 {
                    gl.draw_arrays(topology, start_vertex as i32, vertex_count as i32);
                } else {
                    gl.draw_arrays_instanced(
                        topology,
                        start_vertex as i32,
                        vertex_count as i32,
                        instance_count as i32,
                    );
                }
            }
            Self::DrawIndexed {
                topology,
                ref index_buf,
                index_type,
                index_count,
                base_vertex,
                instance_count,
            } => {
                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(index_buf.raw));
                match (base_vertex, instance_count) {
                    (0, 1) => gl.draw_elements(
                        topology,
                        index_count as i32,
                        index_type,
                        index_buf.offset as i32,
                    ),
                    (0, _) => gl.draw_elements_instanced(
                        topology,
                        index_count as i32,
                        index_type,
                        index_buf.offset as i32,
                        instance_count as i32,
                    ),
                    (_, 1) => gl.draw_elements_base_vertex(
                        topology,
                        index_count as i32,
                        index_type,
                        index_buf.offset as i32,
                        base_vertex,
                    ),
                    (_, _) => gl.draw_elements_instanced_base_vertex(
                        topology,
                        index_count as _,
                        index_type,
                        index_buf.offset as i32,
                        instance_count as i32,
                        base_vertex,
                    ),
                };
            }
            Self::DrawIndirect {
                topology,
                ref indirect_buf,
            } => {
                gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(indirect_buf.raw));
                gl.draw_arrays_indirect_offset(topology, indirect_buf.offset as i32);
            }
            Self::DrawIndexedIndirect {
                topology,
                raw_index_buf,
                index_type,
                ref indirect_buf,
            } => {
                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(raw_index_buf));
                gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(indirect_buf.raw));
                gl.draw_elements_indirect_offset(topology, index_type, indirect_buf.offset as i32);
            }
            Self::Dispatch(groups) => {
                gl.dispatch_compute(groups[0], groups[1], groups[2]);
            }
            Self::DispatchIndirect { ref indirect_buf } => {
                gl.bind_buffer(glow::DISPATCH_INDIRECT_BUFFER, Some(indirect_buf.raw));
                gl.dispatch_compute_indirect(indirect_buf.offset as i32);
            }
            Self::FillBuffer {
                ref dst,
                size,
                value,
            } => unimplemented!(),
            Self::CopyBufferToBuffer {
                ref src,
                ref dst,
                size,
            } => unimplemented!(),
            Self::CopyTextureToTexture {
                ref src,
                ref dst,
                ref size,
            } => unimplemented!(),
            Self::CopyBufferToTexture {
                ref src,
                ref dst,
                bytes_per_row,
                ref size,
            } => unimplemented!(),
            Self::CopyTextureToBuffer {
                ref src,
                ref dst,
                bytes_per_row,
                ref size,
            } => unimplemented!(),
            Self::ResetFramebuffer { is_default } => unimplemented!(),
            Self::BindAttachment {
                attachment,
                ref view,
            } => unimplemented!(),
            Self::InvalidateAttachment(id) => unimplemented!(),
            Self::SetDrawColorBuffers(count) => unimplemented!(),
            Self::ClearColor {
                draw_buffer,
                color,
                ty,
            } => unimplemented!(),
            Self::ClearDepthStencil { depth, stencil } => unimplemented!(),
            Self::Barrier => unimplemented!(),
            Self::SetViewport {
                ref rect,
                ref depth,
            } => unimplemented!(),
            Self::SetScissor(ref rect) => unimplemented!(),
            Self::SetStencilFunc {
                face,
                function,
                reference,
                read_mask,
            } => unimplemented!(),
            Self::SetStencilOps {
                face,
                write_mask,
                //ops: crate::StencilOps,
            } => unimplemented!(),
            //SetDepth(DepthState),
            //SetDepthBias(wgt::DepthBiasState),
            //ConfigureDepthStencil(crate::FormatAspects),
            Self::SetProgram(raw_program) => unimplemented!(),
            //SetPrimitive(PrimitiveState),
            Self::SetBlendConstant(constant) => unimplemented!(),
            Self::SetColorTarget {
                draw_buffer_index,
                //desc: ColorTargetDesc,
            } => unimplemented!(),
            Self::BindBuffer {
                target,
                slot,
                ref buffer,
            } => unimplemented!(),
            Self::BindSampler(slot, maybe_sampler) => unimplemented!(),
            Self::BindTexture {
                slot,
                texture,
                target,
            } => unimplemented!(),
            Self::BindImage { slot, ref binding } => unimplemented!(),
        }
    }
}
