pub type BufferPiece = crate::GenericBufferPiece<Buffer>;

#[derive(Clone, Copy, Debug)]
pub struct TexturePiece {
    pub texture: Texture,
    pub mip_level: u32,
    pub array_layer: u32,
    pub origin: [u32; 3],
}

impl From<Texture> for TexturePiece {
    fn from(texture: Texture) -> Self {
        Self {
            texture,
            mip_level: 0,
            array_layer: 0,
            origin: [0; 3],
        }
    }
}

pub type BufferArray<const N: crate::ResourceIndex> = crate::ResourceArray<BufferPiece, N>;
pub type TextureArray<const N: crate::ResourceIndex> = crate::ResourceArray<TextureView, N>;

#[derive(Clone, Debug)]
pub struct AccelerationStructureMesh {
    pub vertex_data: BufferPiece,
    pub vertex_format: crate::VertexFormat,
    pub vertex_stride: u32,
    pub vertex_count: u32,
    pub index_data: BufferPiece,
    pub index_type: Option<crate::IndexType>,
    pub triangle_count: u32,
    pub transform_data: BufferPiece,
    pub is_opaque: bool,
}

pub trait ShaderBindable: Clone + Copy + crate::derive::HasShaderBinding {
    fn bind_to(&self, context: &mut PipelineContext, index: u32);
}

#[derive(Clone, Copy, Debug)]
pub enum FinishOp {
    Store,
    Discard,
    ResolveTo(TextureView),
    Ignore,
}

#[derive(Debug)]
pub struct RenderTarget {
    pub view: TextureView,
    pub init_op: crate::InitOp,
    pub finish_op: FinishOp,
}

#[derive(Debug)]
pub struct RenderTargetSet<'a> {
    pub colors: &'a [RenderTarget],
    pub depth_stencil: Option<RenderTarget>,
}

impl Context {
    pub fn try_create_shader(
        &self,
        desc: super::ShaderDesc,
    ) -> Result<super::Shader, &'static str> {
        let module = naga::front::wgsl::parse_str(desc.source).map_err(|e| {
            e.emit_to_stderr_with_path(desc.source, "");
            "compilation failed"
        })?;

        let device_caps = self.capabilities();

        // Bindings are set up at pipeline creation, ignore here
        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        let mut caps = naga::valid::Capabilities::empty();
        caps.set(
            naga::valid::Capabilities::RAY_QUERY | naga::valid::Capabilities::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            !device_caps.ray_query.is_empty(),
        );
        let info = naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .map_err(|e| {
                crate::util::emit_annotated_error(&e, "", desc.source);
                crate::util::print_err(&e);
                "validation failed"
            })?;

        Ok(super::Shader { module, info })
    }

    pub fn create_shader(&self, desc: super::ShaderDesc) -> super::Shader {
        self.try_create_shader(desc).unwrap()
    }
}

pub trait ShaderData {
    fn layout() -> crate::ShaderDataLayout;
    fn fill(&self, context: PipelineContext);
}

mod traits {
    pub trait PipelineEncoder {
        fn bind<D: super::ShaderData>(&mut self, group: u32, data: &D);
    }
}
