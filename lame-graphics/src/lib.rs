#![allow(
    // We don't use syntax sugar where it's not necessary.
    clippy::match_like_matches_macro,
    // Redundant matching is more explicit.
    clippy::redundant_pattern_matching,
    // Explicit lifetimes are often easier to reason about.
    clippy::needless_lifetimes,
    // No need for defaults in the internal types.
    clippy::new_without_default,
    // Matches are good and extendable, no need to make an exception here.
    clippy::single_match,
    // Push commands are more regular than macros.
    clippy::vec_init_then_push,
    // "if panic" is a good uniform construct.
    clippy::if_then_panic,
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

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
    pub name: &'a str,
    pub size: u64,
    pub memory: Memory,
}

#[derive(Debug)]
pub enum TextureFormat {
    Rgba8Unorm,
}

#[derive(Debug)]
pub struct TextureDesc<'a> {
    pub name: &'a str,
    pub format: TextureFormat,
}

#[derive(Debug)]
pub struct TextureViewDesc<'a> {
    pub name: &'a str,
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
    Texture,
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

pub struct CommandEncoderDesc<'a> {
    pub name: &'a str,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DepthBiasState {
    /// Constant depth biasing factor, in basic units of the depth format.
    pub constant: i32,
    /// Slope depth biasing factor.
    pub slope_scale: f32,
    /// Depth bias clamp value (absolute).
    pub clamp: f32,
}

pub struct RenderPipelineDesc<'a> {
    pub name: &'a str,
    pub layouts: &'a [&'a ShaderDataLayout],
    pub vertex: ShaderFunction<'a>,
    pub fragment: ShaderFunction<'a>,
}

#[derive(Clone, Copy, Debug)]
pub enum TextureColor {
    TransparentBlack,
    OpaqueBlack,
    White,
}

#[derive(Clone, Copy, Debug)]
pub enum InitOp {
    Load,
    Clear(TextureColor),
}

#[derive(Clone, Copy, Debug)]
pub enum FinishOp {
    Store,
    Discard,
    ResolveTo(TextureView),
}

#[derive(Debug)]
pub struct RenderTarget {
    pub view: TextureView,
    pub init_op: InitOp,
    pub finish_op: FinishOp,
}

#[derive(Debug)]
pub struct RenderTargetSet<'a> {
    pub colors: &'a [RenderTarget],
    pub depth_stencil: Option<RenderTarget>,
}

impl Context {
    pub fn create_shader(&self, desc: ShaderDesc) -> Shader {
        unimplemented!()
    }
}

#[doc(hidden)]
pub struct ShaderDataCollector {
    //pub plain_data: &'a mut [u8],
    //pub buffers: Vec<hal::BufferBinding<'a, Api>>,
    //pub samplers: Vec<&'a <Api as hal::Api>::Sampler>,
    //pub textures: Vec<hal::TextureBinding<'a, Api>>,
}

pub trait ShaderData {
    fn layout() -> ShaderDataLayout;
    fn fill(&self, collector: &mut ShaderDataCollector);
}
