pub use naga::VectorSize;

#[cfg_attr(any(target_os = "ios", target_os = "macos"), path = "metal/mod.rs")]
#[cfg_attr(not(any(target_os = "ios", target_os = "macos")), path = "vulkan/mod.rs")]
mod hal;

pub use hal::*;

#[derive(Debug)]
pub struct ContextDesc {
    pub validation: bool,
}

#[derive(Debug)]
pub struct NotSupportedError;

#[derive(Debug)]
pub enum Memory {
    Device,
    Shared,
    Upload,
}

#[derive(Debug)]
pub struct BufferDesc<'a> {
    pub label: &'a str,
    pub size: u64,
    pub memory: Memory,
}

#[derive(Debug)]
pub enum TextureFormat {
    Rgba8Unorm,
}

#[derive(Debug)]
pub struct TextureDesc {
    pub format: TextureFormat,
}

#[derive(Debug)]
pub struct TextureViewDesc {
    pub texture: Texture,
}

pub struct Shader {
    module: naga::Module,
}

pub struct ShaderFunction<'a> {
    pub shader: &'a Shader,
    pub entry_point: &'a str,
}

impl Shader {
    pub fn at<'a>(&'a self, entry_point: &'a str) -> ShaderFunction<'a> {
        ShaderFunction {
            shader: self,
            entry_point,
        }
    }
}

#[derive(Debug)]
pub enum PlainType {
    F32,
}

#[derive(Debug)]
pub enum PlainContainer {
    Scalar,
    Vector(VectorSize),
}

#[derive(Debug)]
pub enum BindingType {
    //TODO
}

#[derive(Debug)]
pub enum ShaderBinding {
    Resource {
        ty: BindingType,
    },
    Plain {
        ty: PlainType,
        container: PlainContainer,
        offset: u32,
    },
}

pub struct ShaderDataLayout {
    pub plain_size: u32,
    pub bindings: Vec<(String, ShaderBinding)>,
}

pub struct ShaderDesc<'a> {
    pub source: &'a str,
    pub data_layouts: &'a[&'a ShaderDataLayout],
}

pub struct RenderPipelineDesc<'a> {
    pub layouts: &'a [&'a ShaderDataLayout],
    pub vertex: ShaderFunction<'a>,
    pub fragment: ShaderFunction<'a>,
}

impl Context {
    pub fn create_shader(&self, desc: &ShaderDesc) -> Shader {
        unimplemented!()
    }
}

#[doc(hidden)]
pub struct ShaderDataCollector<'a> {
    pub plain_data: &'a mut [u8],
    //pub buffers: Vec<hal::BufferBinding<'a, Api>>,
    //pub samplers: Vec<&'a <Api as hal::Api>::Sampler>,
    //pub textures: Vec<hal::TextureBinding<'a, Api>>,
}

pub trait ShaderData<'a> {
    fn layout() -> ShaderDataLayout;
    fn fill(&self, collector: &mut ShaderDataCollector<'a>);
}
