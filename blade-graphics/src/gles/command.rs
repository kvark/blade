const COLOR_ATTACHMENTS: &[u32] = &[
    glow::COLOR_ATTACHMENT0,
    glow::COLOR_ATTACHMENT1,
    glow::COLOR_ATTACHMENT2,
    glow::COLOR_ATTACHMENT3,
];

impl<T: bytemuck::Pod> super::ShaderBindable for T {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let self_slice = bytemuck::bytes_of(self);
        let alignment = ctx.limits.uniform_buffer_alignment as usize;
        let rem = ctx.plain_data.len() % alignment;
        if rem != 0 {
            ctx.plain_data
                .resize(ctx.plain_data.len() - rem + alignment, 0);
        }
        let offset = ctx.plain_data.len() as u32;
        let size = super::round_up_uniform_size(self_slice.len() as u32);
        ctx.plain_data.extend_from_slice(self_slice);
        ctx.plain_data
            .extend((self_slice.len() as u32..size).map(|_| 0));

        for &slot in ctx.targets[index as usize].iter() {
            ctx.commands
                .push(super::Command::BindUniform { slot, offset, size });
        }
    }
}
impl super::ShaderBindable for super::TextureView {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let (texture, target) = self.inner.as_native();
        for &slot in ctx.targets[index as usize].iter() {
            ctx.commands.push(super::Command::BindTexture {
                slot,
                texture,
                target,
            });
        }
    }
}
impl<'a, const N: crate::ResourceIndex> super::ShaderBindable for &'a super::TextureArray<N> {
    fn bind_to(&self, _ctx: &mut super::PipelineContext, _index: u32) {
        unimplemented!()
    }
}
impl super::ShaderBindable for super::Sampler {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        for &slot in ctx.targets[index as usize].iter() {
            ctx.commands.push(super::Command::BindSampler {
                slot,
                sampler: self.raw,
            });
        }
    }
}
impl super::ShaderBindable for super::BufferPiece {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        for &slot in ctx.targets[index as usize].iter() {
            ctx.commands.push(super::Command::BindBuffer {
                target: glow::SHADER_STORAGE_BUFFER,
                slot,
                buffer: (*self).into(),
                size: (self.buffer.size - self.offset) as u32,
            });
        }
    }
}
impl<'a, const N: crate::ResourceIndex> super::ShaderBindable for &'a super::BufferArray<N> {
    fn bind_to(&self, _ctx: &mut super::PipelineContext, _index: u32) {
        unimplemented!()
    }
}
impl super::ShaderBindable for super::AccelerationStructure {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        for _ in ctx.targets[index as usize].iter() {
            unimplemented!()
        }
    }
}

impl super::CommandEncoder {
    pub fn start(&mut self) {
        self.commands.clear();
        self.plain_data.clear();
        self.has_present = false;
    }

    pub fn init_texture(&mut self, _texture: super::Texture) {}

    pub fn present(&mut self, _frame: super::Frame) {
        self.has_present = true;
    }

    pub fn transfer(&mut self) -> super::PassEncoder<()> {
        super::PassEncoder {
            commands: &mut self.commands,
            plain_data: &mut self.plain_data,
            kind: super::PassKind::Transfer,
            invalidate_attachments: Vec::new(),
            pipeline: Default::default(),
            limits: &self.limits,
        }
    }

    pub fn acceleration_structure(&mut self) -> super::PassEncoder<()> {
        unimplemented!()
    }

    pub fn compute(&mut self) -> super::PassEncoder<super::ComputePipeline> {
        super::PassEncoder {
            commands: &mut self.commands,
            plain_data: &mut self.plain_data,
            kind: super::PassKind::Compute,
            invalidate_attachments: Vec::new(),
            pipeline: Default::default(),
            limits: &self.limits,
        }
    }

    pub fn render(
        &mut self,
        targets: super::RenderTargetSet,
    ) -> super::PassEncoder<super::RenderPipeline> {
        let mut target_size = [0u16; 2];
        let mut invalidate_attachments = Vec::new();
        for (i, rt) in targets.colors.iter().enumerate() {
            let attachment = glow::COLOR_ATTACHMENT0 + i as u32;
            target_size = rt.view.target_size;
            self.commands.push(super::Command::BindAttachment {
                attachment,
                view: rt.view,
            });
            if let super::FinishOp::Discard = rt.finish_op {
                invalidate_attachments.push(attachment);
            }
        }
        if let Some(ref rt) = targets.depth_stencil {
            let attachment = match rt.view.aspects {
                crate::TexelAspects::DEPTH => glow::DEPTH_ATTACHMENT,
                crate::TexelAspects::STENCIL => glow::STENCIL_ATTACHMENT,
                _ => glow::DEPTH_STENCIL_ATTACHMENT,
            };
            target_size = rt.view.target_size;
            self.commands.push(super::Command::BindAttachment {
                attachment,
                view: rt.view,
            });
            if let super::FinishOp::Discard = rt.finish_op {
                invalidate_attachments.push(attachment);
            }
        }

        self.commands.push(super::Command::SetDrawColorBuffers(
            targets.colors.len() as _
        ));
        self.commands.push(super::Command::SetViewport {
            size: target_size,
            depth: 0.0..1.0,
        });
        self.commands
            .push(super::Command::SetScissor(crate::ScissorRect {
                x: 0,
                y: 0,
                w: target_size[0] as u32,
                h: target_size[1] as u32,
            }));

        // issue the clears
        for (i, rt) in targets.colors.iter().enumerate() {
            if let crate::InitOp::Clear(color) = rt.init_op {
                self.commands.push(super::Command::ClearColor {
                    draw_buffer: i as u32,
                    color,
                    ty: super::ColorType::Float, //TODO: get from the format
                });
            }
        }
        if let Some(ref rt) = targets.depth_stencil {
            if let crate::InitOp::Clear(color) = rt.init_op {
                self.commands.push(super::Command::ClearDepthStencil {
                    depth: if rt.view.aspects.contains(crate::TexelAspects::DEPTH) {
                        Some(match color {
                            crate::TextureColor::White => 1.0,
                            _ => 0.0,
                        })
                    } else {
                        None
                    },
                    stencil: None, //TODO
                });
            }
        }

        super::PassEncoder {
            commands: &mut self.commands,
            plain_data: &mut self.plain_data,
            kind: super::PassKind::Render,
            invalidate_attachments,
            pipeline: Default::default(),
            limits: &self.limits,
        }
    }
}

impl super::PassEncoder<'_, super::ComputePipeline> {
    pub fn with<'b>(
        &'b mut self,
        pipeline: &'b super::ComputePipeline,
    ) -> super::PipelineEncoder<'b> {
        self.commands
            .push(super::Command::SetProgram(pipeline.inner.program));
        super::PipelineEncoder {
            commands: self.commands,
            plain_data: self.plain_data,
            group_mappings: &pipeline.inner.group_mappings,
            topology: 0,
            limits: self.limits,
            vertex_attributes: &[],
        }
    }
}

impl super::PassEncoder<'_, super::RenderPipeline> {
    pub fn with<'b>(
        &'b mut self,
        pipeline: &'b super::RenderPipeline,
    ) -> super::PipelineEncoder<'b> {
        self.commands
            .push(super::Command::SetProgram(pipeline.inner.program));

        match &pipeline.inner.color_targets[..] {
            &[(blend_state, write_masks)] => self
                .commands
                .push(super::Command::SetAllColorTargets(blend_state, write_masks)),
            separate => self.commands.extend(separate.iter().zip(0..).map(
                |(&(blend_state, write_masks), i)| {
                    super::Command::SetSingleColorTarget(i, blend_state, write_masks)
                },
            )),
        }
        super::PipelineEncoder {
            commands: self.commands,
            plain_data: self.plain_data,
            group_mappings: &pipeline.inner.group_mappings,
            topology: map_primitive_topology(pipeline.topology),
            limits: self.limits,
            vertex_attributes: &pipeline.inner.vertex_attribute_infos,
        }
    }
}

impl<T> Drop for super::PassEncoder<'_, T> {
    fn drop(&mut self) {
        self.commands.push(super::Command::UnsetProgram);
        for attachment in self.invalidate_attachments.drain(..) {
            self.commands
                .push(super::Command::InvalidateAttachment(attachment));
        }
        match self.kind {
            super::PassKind::Transfer | super::PassKind::AccelerationStructure => {}
            super::PassKind::Compute => {
                self.commands.push(super::Command::ResetAllSamplers);
            }
            super::PassKind::Render => {
                self.commands.push(super::Command::ResetAllSamplers);
                self.commands.push(super::Command::ResetFramebuffer);
            }
        }
    }
}

#[hidden_trait::expose]
impl crate::traits::TransferEncoder for super::PassEncoder<'_, ()> {
    type BufferPiece = super::BufferPiece;
    type TexturePiece = super::TexturePiece;

    fn fill_buffer(&mut self, dst: super::BufferPiece, size: u64, value: u8) {
        self.commands.push(super::Command::FillBuffer {
            dst: dst.into(),
            size,
            value,
        });
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: super::BufferPiece,
        dst: super::BufferPiece,
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
        src: super::TexturePiece,
        dst: super::TexturePiece,
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
        src: super::BufferPiece,
        bytes_per_row: u32,
        dst: super::TexturePiece,
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
        src: super::TexturePiece,
        dst: super::BufferPiece,
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
impl crate::traits::AccelerationStructureEncoder for super::PassEncoder<'_, ()> {
    type AccelerationStructure = super::AccelerationStructure;
    type AccelerationStructureMesh = super::AccelerationStructureMesh;
    type BufferPiece = super::BufferPiece;

    fn build_bottom_level(
        &mut self,
        _acceleration_structure: super::AccelerationStructure,
        _meshes: &[super::AccelerationStructureMesh],
        _scratch_data: super::BufferPiece,
    ) {
        unimplemented!()
    }

    fn build_top_level(
        &mut self,
        _acceleration_structure: super::AccelerationStructure,
        _bottom_level: &[super::AccelerationStructure],
        _instance_count: u32,
        _instance_data: super::BufferPiece,
        _scratch_data: super::BufferPiece,
    ) {
        unimplemented!()
    }
}

#[hidden_trait::expose]
impl super::traits::PipelineEncoder for super::PipelineEncoder<'_> {
    fn bind<D: super::ShaderData>(&mut self, group: u32, data: &D) {
        data.fill(super::PipelineContext {
            commands: self.commands,
            plain_data: self.plain_data,
            targets: &self.group_mappings[group as usize].targets,
            limits: self.limits,
        });
    }
}

#[hidden_trait::expose]
impl crate::traits::ComputePipelineEncoder for super::PipelineEncoder<'_> {
    fn dispatch(&mut self, groups: [u32; 3]) {
        self.commands.push(super::Command::Dispatch(groups));
    }
}

#[hidden_trait::expose]
impl crate::traits::RenderPipelineEncoder for super::PipelineEncoder<'_> {
    type BufferPiece = super::BufferPiece;

    fn set_scissor_rect(&mut self, rect: &crate::ScissorRect) {
        self.commands.push(super::Command::SetScissor(rect.clone()));
    }

    fn bind_vertex(&mut self, index: u32, vertex_buf: super::BufferPiece) {
        self.commands.push(super::Command::BindVertex {
            buffer: vertex_buf.buffer.raw,
        });
        for (i, info) in self.vertex_attributes.iter().enumerate() {
            self.commands.push(super::Command::SetVertexAttribute {
                index: i as u32,
                format: info.attrib.format,
                offset: (vertex_buf.offset + info.attrib.offset as u64)
                    .try_into()
                    .unwrap(),
                stride: info.stride,
                instanced: info.instanced,
            });
        }
    }

    fn draw(
        &mut self,
        start_vertex: u32,
        vertex_count: u32,
        start_instance: u32,
        instance_count: u32,
    ) {
        assert_eq!(start_instance, 0);
        self.commands.push(super::Command::Draw {
            topology: self.topology,
            start_vertex,
            vertex_count,
            instance_count,
        });
    }

    fn draw_indexed(
        &mut self,
        index_buf: super::BufferPiece,
        index_type: crate::IndexType,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    ) {
        assert_eq!(start_instance, 0);
        self.commands.push(super::Command::DrawIndexed {
            topology: self.topology,
            index_buf: index_buf.into(),
            index_type: map_index_type(index_type),
            index_count,
            base_vertex,
            instance_count,
        });
    }

    fn draw_indirect(&mut self, _indirect_buf: super::BufferPiece) {
        unimplemented!()
    }

    fn draw_indexed_indirect(
        &mut self,
        _index_buf: super::BufferPiece,
        _index_type: crate::IndexType,
        _indirect_buf: super::BufferPiece,
    ) {
        unimplemented!()
    }
}

impl crate::VertexFormat {
    fn describe(&self) -> (i32, u32) {
        match *self {
            Self::F32 => (1, glow::FLOAT),
            Self::F32Vec2 => (2, glow::FLOAT),
            Self::F32Vec3 => (3, glow::FLOAT),
            Self::F32Vec4 => (4, glow::FLOAT),
            Self::U32 => (1, glow::UNSIGNED_INT),
            Self::U32Vec2 => (2, glow::UNSIGNED_INT),
            Self::U32Vec3 => (3, glow::UNSIGNED_INT),
            Self::U32Vec4 => (4, glow::UNSIGNED_INT),
            Self::I32 => (1, glow::INT),
            Self::I32Vec2 => (2, glow::INT),
            Self::I32Vec3 => (3, glow::INT),
            Self::I32Vec4 => (4, glow::INT),
        }
    }
}

impl crate::BlendFactor {
    fn to_gles(self) -> u32 {
        match self {
            Self::Zero => glow::ZERO,
            Self::One => glow::ONE,
            Self::Src => glow::SRC_COLOR,
            Self::OneMinusSrc => glow::ONE_MINUS_SRC_COLOR,
            Self::SrcAlpha => glow::SRC_ALPHA,
            Self::OneMinusSrcAlpha => glow::ONE_MINUS_SRC_ALPHA,
            Self::Dst => glow::DST_COLOR,
            Self::OneMinusDst => glow::ONE_MINUS_DST_COLOR,
            Self::DstAlpha => glow::DST_ALPHA,
            Self::OneMinusDstAlpha => glow::ONE_MINUS_DST_ALPHA,
            Self::SrcAlphaSaturated => glow::SRC_ALPHA_SATURATE,
            Self::Constant => glow::CONSTANT_ALPHA,
            Self::OneMinusConstant => glow::ONE_MINUS_CONSTANT_ALPHA,
        }
    }
}

impl crate::BlendOperation {
    fn to_gles(self) -> u32 {
        match self {
            Self::Add => glow::FUNC_ADD,
            Self::Subtract => glow::FUNC_SUBTRACT,
            Self::ReverseSubtract => glow::FUNC_REVERSE_SUBTRACT,
            Self::Min => glow::MIN,
            Self::Max => glow::MAX,
        }
    }
}

const CUBEMAP_FACES: [u32; 6] = [
    glow::TEXTURE_CUBE_MAP_POSITIVE_X,
    glow::TEXTURE_CUBE_MAP_NEGATIVE_X,
    glow::TEXTURE_CUBE_MAP_POSITIVE_Y,
    glow::TEXTURE_CUBE_MAP_NEGATIVE_Y,
    glow::TEXTURE_CUBE_MAP_POSITIVE_Z,
    glow::TEXTURE_CUBE_MAP_NEGATIVE_Z,
];

impl super::Command {
    pub(super) unsafe fn execute(&self, gl: &glow::Context, ec: &super::ExecutionContext) {
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
            } => {
                gl.bind_buffer(glow::COPY_READ_BUFFER, Some(src.raw));
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(dst.raw));
                gl.copy_buffer_sub_data(
                    glow::COPY_READ_BUFFER,
                    glow::COPY_WRITE_BUFFER,
                    src.offset as _,
                    dst.offset as _,
                    size as _,
                );
            }
            Self::CopyTextureToTexture {
                ref src,
                ref dst,
                ref size,
            } => {
                gl.bind_texture(dst.target, Some(dst.raw));
                gl.framebuffer_texture_2d(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    src.target,
                    Some(src.raw),
                    src.mip_level as i32,
                );
                gl.bind_texture(dst.target, Some(dst.raw));
                gl.copy_tex_sub_image_2d(
                    dst.target,
                    dst.mip_level as i32,
                    dst.origin[0] as i32,
                    dst.origin[1] as i32,
                    src.origin[0] as i32,
                    src.origin[1] as i32,
                    size.width as i32,
                    size.height as i32,
                );
                gl.framebuffer_renderbuffer(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::RENDERBUFFER,
                    None,
                );
            }
            Self::CopyBufferToTexture {
                ref src,
                ref dst,
                bytes_per_row,
                ref size,
            } => {
                let format_desc = super::describe_texture_format(dst.format);
                let block_info = dst.format.block_info();
                let row_texels =
                    bytes_per_row / block_info.size as u32 * block_info.dimensions.0 as u32;
                gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
                gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(src.raw));
                gl.bind_texture(dst.target, Some(dst.raw));
                let unpack_data = glow::PixelUnpackData::BufferOffset(src.offset as u32);
                match dst.target {
                    glow::TEXTURE_3D => gl.tex_sub_image_3d(
                        dst.target,
                        dst.mip_level as i32,
                        dst.origin[0] as i32,
                        dst.origin[1] as i32,
                        dst.origin[2] as i32,
                        size.width as i32,
                        size.height as i32,
                        size.depth as i32,
                        format_desc.external,
                        format_desc.data_type,
                        unpack_data,
                    ),
                    glow::TEXTURE_2D_ARRAY => gl.tex_sub_image_3d(
                        dst.target,
                        dst.mip_level as i32,
                        dst.origin[0] as i32,
                        dst.origin[1] as i32,
                        dst.origin[2] as i32,
                        size.width as i32,
                        size.height as i32,
                        size.depth as i32,
                        format_desc.external,
                        format_desc.data_type,
                        unpack_data,
                    ),
                    glow::TEXTURE_2D => gl.tex_sub_image_2d(
                        dst.target,
                        dst.mip_level as i32,
                        dst.origin[0] as i32,
                        dst.origin[1] as i32,
                        size.width as i32,
                        size.height as i32,
                        format_desc.external,
                        format_desc.data_type,
                        unpack_data,
                    ),
                    glow::TEXTURE_CUBE_MAP => gl.tex_sub_image_2d(
                        CUBEMAP_FACES[dst.array_layer as usize],
                        dst.mip_level as i32,
                        dst.origin[0] as i32,
                        dst.origin[1] as i32,
                        size.width as i32,
                        size.height as i32,
                        format_desc.external,
                        format_desc.data_type,
                        unpack_data,
                    ),
                    //Note: not sure if this is correct!
                    glow::TEXTURE_CUBE_MAP_ARRAY => gl.tex_sub_image_3d(
                        dst.target,
                        dst.mip_level as i32,
                        dst.origin[0] as i32,
                        dst.origin[1] as i32,
                        dst.origin[2] as i32,
                        size.width as i32,
                        size.height as i32,
                        size.depth as i32,
                        format_desc.external,
                        format_desc.data_type,
                        unpack_data,
                    ),
                    _ => unreachable!(),
                }
                gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            }
            Self::CopyTextureToBuffer {
                ref src,
                ref dst,
                bytes_per_row,
                ref size,
            } => unimplemented!(),
            Self::ResetFramebuffer => {
                for &attachment in COLOR_ATTACHMENTS.iter() {
                    gl.framebuffer_renderbuffer(
                        glow::DRAW_FRAMEBUFFER,
                        attachment,
                        glow::RENDERBUFFER,
                        None,
                    );
                }
                gl.framebuffer_renderbuffer(
                    glow::DRAW_FRAMEBUFFER,
                    glow::DEPTH_STENCIL_ATTACHMENT,
                    glow::RENDERBUFFER,
                    None,
                );
            }
            Self::BindAttachment {
                attachment,
                ref view,
            } => match view.inner {
                super::TextureInner::Renderbuffer { raw } => {
                    gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(ec.framebuf));
                    gl.bind_renderbuffer(glow::RENDERBUFFER, Some(raw));
                    gl.framebuffer_renderbuffer(
                        glow::DRAW_FRAMEBUFFER,
                        attachment,
                        glow::RENDERBUFFER,
                        Some(raw),
                    );
                }
                super::TextureInner::Texture { raw, target } => {
                    let mip_level = 0; //TODO
                    gl.framebuffer_texture_2d(
                        glow::DRAW_FRAMEBUFFER,
                        attachment,
                        target,
                        Some(raw),
                        mip_level,
                    );
                }
            },
            Self::InvalidateAttachment(attachment) => {
                gl.invalidate_framebuffer(glow::DRAW_FRAMEBUFFER, &[attachment]);
            }
            Self::SetDrawColorBuffers(count) => {
                gl.draw_buffers(&COLOR_ATTACHMENTS[..count as usize]);
            }
            Self::SetAllColorTargets(blend, write_mask) => {
                if let Some(blend_state) = blend {
                    gl.enable(glow::BLEND);
                    gl.blend_func_separate(
                        blend_state.color.src_factor.to_gles(),
                        blend_state.color.dst_factor.to_gles(),
                        blend_state.alpha.src_factor.to_gles(),
                        blend_state.alpha.dst_factor.to_gles(),
                    );
                    gl.blend_equation(blend_state.color.operation.to_gles());
                } else {
                    gl.disable(glow::BLEND);
                }
                gl.color_mask(
                    write_mask.contains(crate::ColorWrites::RED),
                    write_mask.contains(crate::ColorWrites::GREEN),
                    write_mask.contains(crate::ColorWrites::BLUE),
                    write_mask.contains(crate::ColorWrites::ALPHA),
                );
            }
            Self::SetSingleColorTarget(i, blend, write_mask) => {
                if let Some(blend_state) = blend {
                    gl.enable_draw_buffer(glow::BLEND, i);
                    gl.blend_func_separate_draw_buffer(
                        i,
                        blend_state.color.src_factor.to_gles(),
                        blend_state.color.dst_factor.to_gles(),
                        blend_state.alpha.src_factor.to_gles(),
                        blend_state.alpha.dst_factor.to_gles(),
                    );
                    gl.blend_equation_draw_buffer(i, blend_state.color.operation.to_gles());
                } else {
                    gl.disable_draw_buffer(glow::BLEND, i);
                }
                gl.color_mask_draw_buffer(
                    i,
                    write_mask.contains(crate::ColorWrites::RED),
                    write_mask.contains(crate::ColorWrites::GREEN),
                    write_mask.contains(crate::ColorWrites::BLUE),
                    write_mask.contains(crate::ColorWrites::ALPHA),
                );
            }
            Self::ClearColor {
                draw_buffer,
                color,
                ty,
            } => match ty {
                super::ColorType::Float => {
                    gl.clear_buffer_f32_slice(
                        glow::COLOR,
                        draw_buffer,
                        &match color {
                            crate::TextureColor::TransparentBlack => [0.0; 4],
                            crate::TextureColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
                            crate::TextureColor::White => [1.0; 4],
                        },
                    );
                }
                super::ColorType::Uint => {
                    gl.clear_buffer_u32_slice(
                        glow::COLOR,
                        draw_buffer,
                        &match color {
                            crate::TextureColor::TransparentBlack => [0; 4],
                            crate::TextureColor::OpaqueBlack => [0, 0, 0, !0],
                            crate::TextureColor::White => [!0; 4],
                        },
                    );
                }
                super::ColorType::Sint => {
                    gl.clear_buffer_i32_slice(
                        glow::COLOR,
                        draw_buffer,
                        &match color {
                            crate::TextureColor::TransparentBlack => [0; 4],
                            crate::TextureColor::OpaqueBlack => [0, 0, 0, !0],
                            crate::TextureColor::White => [!0; 4],
                        },
                    );
                }
            },
            Self::ClearDepthStencil { depth, stencil } => match (depth, stencil) {
                (Some(d), Some(s)) => {
                    gl.clear_buffer_depth_stencil(glow::DEPTH_STENCIL, 0, d, s as i32)
                }
                (Some(d), None) => gl.clear_buffer_f32_slice(glow::DEPTH, 0, &[d]),
                (None, Some(s)) => gl.clear_buffer_i32_slice(glow::STENCIL, 0, &[s as i32]),
                (None, None) => (),
            },
            Self::Barrier => unimplemented!(),
            Self::SetViewport { size, ref depth } => {
                gl.viewport(0, 0, size[0] as i32, size[1] as i32);
                gl.depth_range_f32(depth.start, depth.end);
            }
            Self::SetScissor(ref rect) => {
                gl.scissor(rect.x, rect.y, rect.w as i32, rect.h as i32);
            }
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
            Self::SetProgram(raw_program) => {
                gl.use_program(Some(raw_program));
            }
            Self::UnsetProgram => {
                gl.use_program(None);
            }
            //SetPrimitive(PrimitiveState),
            Self::SetBlendConstant([r, g, b, a]) => gl.blend_color(r, g, b, a),
            Self::SetColorTarget {
                draw_buffer_index,
                //desc: ColorTargetDesc,
            } => unimplemented!(),
            Self::BindUniform { slot, offset, size } => {
                gl.bind_buffer_range(
                    glow::UNIFORM_BUFFER,
                    slot,
                    Some(ec.plain_buffer),
                    offset as i32,
                    size as i32,
                );
            }
            Self::BindVertex { buffer } => {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
            }
            Self::SetVertexAttribute {
                index,
                format,
                offset,
                stride,
                instanced,
            } => {
                let (data_size, data_type) = format.describe();
                match data_type {
                    glow::FLOAT => gl.vertex_attrib_pointer_f32(
                        index, data_size, data_type, false, stride, offset,
                    ),
                    glow::INT | glow::UNSIGNED_INT => {
                        gl.vertex_attrib_pointer_i32(index, data_size, data_type, stride, offset)
                    }
                    _ => unreachable!(),
                }
                gl.vertex_attrib_divisor(index, if instanced { 1 } else { 0 });
                gl.enable_vertex_attrib_array(index);
            }
            Self::DisableVertexAttributes { count } => {
                for index in 0..count {
                    gl.disable_vertex_attrib_array(index);
                }
            }
            Self::BindBuffer {
                target,
                slot,
                ref buffer,
                size,
            } => {
                gl.bind_buffer_range(
                    target,
                    slot,
                    Some(buffer.raw),
                    buffer.offset as i32,
                    size as i32,
                );
            }
            Self::BindSampler { slot, sampler } => {
                gl.bind_sampler(slot, Some(sampler));
            }
            Self::BindTexture {
                slot,
                texture,
                target,
            } => {
                gl.active_texture(glow::TEXTURE0 + slot);
                gl.bind_texture(target, Some(texture));
            }
            Self::BindImage { slot, ref binding } => unimplemented!(),
            Self::ResetAllSamplers => {
                gl.active_texture(glow::TEXTURE0);
                for slot in 0..4 {
                    gl.bind_sampler(slot, None);
                }
            }
        }
    }
}

fn map_index_type(ty: crate::IndexType) -> u32 {
    match ty {
        crate::IndexType::U16 => glow::UNSIGNED_SHORT,
        crate::IndexType::U32 => glow::UNSIGNED_INT,
    }
}

fn map_primitive_topology(topology: crate::PrimitiveTopology) -> u32 {
    use crate::PrimitiveTopology as Pt;
    match topology {
        Pt::PointList => glow::POINTS,
        Pt::LineList => glow::LINES,
        Pt::LineStrip => glow::LINE_STRIP,
        Pt::TriangleList => glow::TRIANGLES,
        Pt::TriangleStrip => glow::TRIANGLE_STRIP,
    }
}
