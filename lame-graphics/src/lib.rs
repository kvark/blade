pub use naga::VectorSize;
pub use wgt::{BindingType, BufferBindingType, TextureSampleType, StorageTextureAccess, TextureFormat, TextureUsages, TextureViewDimension};
#[doc(hidden)]
pub use hal::{TextureBinding, TextureUses};
use hal::{Adapter as _, Instance as _};

#[cfg(any(target_os = "ios", target_os = "macos"))]
use hal::api::Metal as Api;
#[cfg(not(any(target_os = "ios", target_os = "macos")))]
use hal::api::Vulkan as Api;

mod command;

pub struct ContextDesc {
    pub validation: bool,
}

pub struct Context {
    device: <Api as hal::Api>::Device,
    queue: <Api as hal::Api>::Queue,
    adapter: <Api as hal::Api>::Adapter,
    instance: <Api as hal::Api>::Instance,
}

#[derive(Debug)]
pub struct NotSupportedError;

#[derive(Debug)]
pub struct Texture {
    pub raw: <Api as hal::Api>::Texture,
}

#[derive(Debug)]
pub struct TextureView {
    #[doc(hidden)]
    pub raw: <Api as hal::Api>::TextureView,
}

#[derive(Debug)]
pub struct Buffer {
    raw: <Api as hal::Api>::Buffer,
}

pub struct BufferDesc {
    pub size: wgt::BufferSize,
}

pub struct TextureDesc {
    pub format: TextureFormat,
}

pub struct TextureViewDesc<'a> {
    pub texture: &'a Texture,
}

pub struct Shader {
    module: naga::Module,
}

pub struct ShaderStage<'a> {
    pub shader: &'a Shader,
    pub entry_point: &'a str,
}

impl Shader {
    pub fn at<'a>(&'a self, entry_point: &'a str) -> ShaderStage<'a> {
        ShaderStage {
            shader: self,
            entry_point,
        }
    }
}

pub struct CommandEncoder {
    raw: <Api as hal::Api>::CommandEncoder,
}

pub struct RenderPipelineCommandEncoder<'a> {
    raw: &'a mut <Api as hal::Api>::CommandEncoder,
}

pub enum PlainType {
    F32,
}
pub enum PlainContainer {
    Scalar,
    Vector(VectorSize),
}

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

pub struct ShaderFunction<'a> {
    pub shader: &'a Shader,
    pub entry_point: &'a str,
}

pub struct RenderPipelineDesc<'a> {
    pub layouts: &'a [&'a ShaderDataLayout],
    pub vertex: ShaderStage<'a>,
    pub fragment: ShaderStage<'a>,
}

pub struct RenderPipeline {
    raw: <Api as hal::Api>::RenderPipeline,
}

impl Context {
    pub fn init(desc: &ContextDesc) -> Result<Self, NotSupportedError> {
        let instance_desc = hal::InstanceDescriptor {
            name: "lame",
            flags: if desc.validation {
                hal::InstanceFlags::all()
            } else {
                hal::InstanceFlags::empty()
            },
        };
        let instance = unsafe { <<Api as hal::Api>::Instance as hal::Instance<Api>>::init(&instance_desc) }
            .map_err(|_| NotSupportedError)?;

        let (adapter, capabilities) = unsafe {
            let mut adapters = instance.enumerate_adapters();
            if adapters.is_empty() {
                return Err(NotSupportedError);
            }
            let exposed = adapters.swap_remove(0);
            (exposed.adapter, exposed.capabilities)
        };

        let features = wgt::Features::DIRECT_RESOURCE_BINDING | wgt::Features::INLINE_UNIFORM_DATA;
        let hal::OpenDevice { device, mut queue } = unsafe {
            adapter
                .open(wgt::Features::empty(), &wgt::Limits::default())
                .unwrap()
        };
        Ok(Context {
            device,
            queue,
            adapter,
            instance,
        })
    }

    pub fn create_buffer(&self, desc: &BufferDesc) -> Buffer {
        unimplemented!()
    }

    pub fn create_texture(&self, desc: &TextureDesc) -> Texture {
        unimplemented!()
    }

    pub fn create_texture_view(&self, desc: &TextureViewDesc) -> TextureView {
        unimplemented!()
    }

    pub fn create_shader(&self, desc: &ShaderDesc) -> Shader {
        unimplemented!()
    }

    pub fn create_command_encoder(&self) -> CommandEncoder {
        unimplemented!()
    }

    pub fn create_render_pipeline(&self, desc: &RenderPipelineDesc) -> RenderPipeline {
        unimplemented!()
    }
}

#[doc(hidden)]
pub struct ShaderDataCollector<'a> {
    pub plain_data: &'a mut [u8],
    pub buffers: Vec<hal::BufferBinding<'a, Api>>,
    pub samplers: Vec<&'a <Api as hal::Api>::Sampler>,
    pub textures: Vec<hal::TextureBinding<'a, Api>>,
}

pub trait ShaderData<'a> {
    fn layout() -> ShaderDataLayout;
    fn fill(&self, collector: &mut ShaderDataCollector<'a>);
}
