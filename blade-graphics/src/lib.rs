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

pub use naga::{StorageAccess, VectorSize};

#[cfg_attr(
    all(not(portability), any(target_os = "ios", target_os = "macos")),
    path = "metal/mod.rs"
)]
#[cfg_attr(
    any(portability, not(any(target_os = "ios", target_os = "macos"))),
    path = "vulkan/mod.rs"
)]
mod hal;
mod shader;
mod traits;
mod util;
pub mod limits {
    pub const PLAIN_DATA_SIZE: u32 = 256;
    pub const RESOURCES_IN_GROUP: u32 = 8;
}

pub use hal::*;

use std::num::NonZeroU32;

#[derive(Debug)]
pub struct ContextDesc {
    pub validation: bool,
    pub capture: bool,
}

#[derive(Debug)]
pub struct NotSupportedError;

#[derive(Clone, Copy, Debug, PartialEq)]
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

#[derive(Clone, Copy, Debug)]
pub struct BufferPiece {
    pub buffer: Buffer,
    pub offset: u64,
}

impl From<Buffer> for BufferPiece {
    fn from(buffer: Buffer) -> Self {
        Self { buffer, offset: 0 }
    }
}

impl BufferPiece {
    pub fn data(&self) -> *mut u8 {
        let base = self.buffer.data();
        assert!(!base.is_null());
        unsafe { base.offset(self.offset as isize) }
    }
}

impl Buffer {
    pub fn at(self, offset: u64) -> BufferPiece {
        BufferPiece {
            buffer: self,
            offset,
        }
    }
}

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

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub enum TextureFormat {
    Rgba8Unorm,
    Bgra8UnormSrgb,
    Depth32Float,
}

/// Dimensionality of a texture.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum TextureDimension {
    /// 1D texture
    D1,
    /// 2D texture
    D2,
    /// 3D texture
    D3,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum ViewDimension {
    D1,
    D1Array,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct Extent {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Extent {
    pub fn max_mip_levels(&self) -> u32 {
        self.width
            .max(self.height)
            .max(self.depth)
            .next_power_of_two()
            .trailing_zeros()
    }
    pub fn at_mip_level(&self, level: u32) -> Self {
        Self {
            width: (self.width >> level).max(1),
            height: (self.height >> level).max(1),
            depth: (self.depth >> level).max(1),
        }
    }
}

bitflags::bitflags! {
    pub struct TextureUsage: u32 {
        const COPY = 1 << 0;
        const TARGET = 1 << 1;
        const RESOURCE = 1 << 2;
        const STORAGE = 1 << 3;
    }
}

#[derive(Debug)]
pub struct TextureDesc<'a> {
    pub name: &'a str,
    pub format: TextureFormat,
    pub size: Extent,
    pub array_layer_count: u32,
    pub mip_level_count: u32,
    pub dimension: TextureDimension,
    pub usage: TextureUsage,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TextureSubresources {
    pub base_mip_level: u32,
    pub mip_level_count: Option<NonZeroU32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<NonZeroU32>,
}

#[derive(Debug)]
pub struct TextureViewDesc<'a> {
    pub name: &'a str,
    pub texture: Texture,
    pub format: TextureFormat,
    pub dimension: ViewDimension,
    pub subresources: &'a TextureSubresources,
}

bitflags::bitflags! {
    #[derive(Default)]
    struct ShaderVisibility: u32 {
        const COMPUTE = 1 << 0;
        const VERTEX = 1 << 1;
        const FRAGMENT = 1 << 2;
    }
}

/// How edges should be handled in texture addressing.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum AddressMode {
    /// Clamp the value to the edge of the texture.
    #[default]
    ClampToEdge,
    /// Repeat the texture in a tiling fashion.
    Repeat,
    /// Repeat the texture, mirroring it every repeat.
    MirrorRepeat,
    /// Clamp the value to the border of the texture.
    ClampToBorder,
}

/// Texel mixing mode when sampling between texels.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum FilterMode {
    /// Nearest neighbor sampling.
    #[default]
    Nearest,
    /// Linear Interpolation
    Linear,
}

/// Comparison function used for depth and stencil operations.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum CompareFunction {
    /// Function never passes
    Never,
    /// Function passes if new value less than existing value
    Less,
    /// Function passes if new value is equal to existing value. When using
    /// this compare function, make sure to mark your Vertex Shader's `@builtin(position)`
    /// output as `@invariant` to prevent artifacting.
    Equal,
    /// Function passes if new value is less than or equal to existing value
    LessEqual,
    /// Function passes if new value is greater than existing value
    Greater,
    /// Function passes if new value is not equal to existing value. When using
    /// this compare function, make sure to mark your Vertex Shader's `@builtin(position)`
    /// output as `@invariant` to prevent artifacting.
    NotEqual,
    /// Function passes if new value is greater than or equal to existing value
    GreaterEqual,
    /// Function always passes
    #[default]
    Always,
}

#[derive(Clone, Copy, Debug)]
pub enum TextureColor {
    TransparentBlack,
    OpaqueBlack,
    White,
}

#[derive(Debug, Default)]
pub struct SamplerDesc<'a> {
    pub name: &'a str,
    pub address_modes: [AddressMode; 3],
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: FilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: Option<f32>,
    pub compare: Option<CompareFunction>,
    pub anisotropy_clamp: u32,
    pub border_color: Option<TextureColor>,
}

pub struct Shader {
    module: naga::Module,
    info: naga::valid::ModuleInfo,
}

#[derive(Clone, Copy)]
pub struct ShaderFunction<'a> {
    pub shader: &'a Shader,
    pub entry_point: &'a str,
}

impl ShaderFunction<'_> {
    fn entry_point_index(&self) -> usize {
        self.shader
            .module
            .entry_points
            .iter()
            .position(|ep| ep.name == self.entry_point)
            .expect("Entry point not found in the shader")
    }
}

impl Shader {
    pub fn at<'a>(&'a self, entry_point: &'a str) -> ShaderFunction<'a> {
        ShaderFunction {
            shader: self,
            entry_point,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShaderBinding {
    Texture,
    Sampler,
    Buffer,
    Plain { size: u32 },
}

pub trait ShaderBindable: Clone + Copy {
    fn bind_to(&self, context: &mut PipelineContext, index: u32);
}

pub trait HasShaderBinding: ShaderBindable {
    const TYPE: ShaderBinding;
}
impl<T: bytemuck::Pod> HasShaderBinding for T {
    const TYPE: ShaderBinding = ShaderBinding::Plain {
        size: std::mem::size_of::<T>() as u32,
    };
}
impl HasShaderBinding for TextureView {
    const TYPE: ShaderBinding = ShaderBinding::Texture;
}
impl HasShaderBinding for Sampler {
    const TYPE: ShaderBinding = ShaderBinding::Sampler;
}
impl HasShaderBinding for BufferPiece {
    const TYPE: ShaderBinding = ShaderBinding::Buffer;
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShaderDataLayout {
    pub bindings: Vec<(&'static str, ShaderBinding)>,
}

impl ShaderDataLayout {
    pub const EMPTY: &Self = &Self {
        bindings: Vec::new(),
    };
}

pub trait ShaderData {
    fn layout() -> ShaderDataLayout;
    fn fill(&self, context: PipelineContext);
}

pub struct ShaderDesc<'a> {
    pub source: &'a str,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum CommandType {
    Transfer,
    Compute,
    #[default]
    General,
}

pub struct CommandEncoderDesc<'a> {
    pub name: &'a str,
    /// Number of buffers that this encoder needs to keep alive.
    /// For example, one buffer is being run on GPU while the
    /// other is being actively encoded, which makes 2.
    pub buffer_count: u32,
}

pub struct ComputePipelineDesc<'a> {
    pub name: &'a str,
    pub data_layouts: &'a [&'a ShaderDataLayout],
    pub compute: ShaderFunction<'a>,
}

/// Primitive type the input mesh is composed of.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum PrimitiveTopology {
    /// Vertex data is a list of points. Each vertex is a new point.
    PointList,
    /// Vertex data is a list of lines. Each pair of vertices composes a new line.
    ///
    /// Vertices `0 1 2 3` create two lines `0 1` and `2 3`
    LineList,
    /// Vertex data is a strip of lines. Each set of two adjacent vertices form a line.
    ///
    /// Vertices `0 1 2 3` create three lines `0 1`, `1 2`, and `2 3`.
    LineStrip,
    /// Vertex data is a list of triangles. Each set of 3 vertices composes a new triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create two triangles `0 1 2` and `3 4 5`
    #[default]
    TriangleList,
    /// Vertex data is a triangle strip. Each set of three adjacent vertices form a triangle.
    ///
    /// Vertices `0 1 2 3 4 5` creates four triangles `0 1 2`, `2 1 3`, `2 3 4`, and `4 3 5`
    TriangleStrip,
}

/// Vertex winding order which classifies the "front" face of a triangle.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum FrontFace {
    /// Triangles with vertices in counter clockwise order are considered the front face.
    ///
    /// This is the default with right handed coordinate spaces.
    #[default]
    Ccw,
    /// Triangles with vertices in clockwise order are considered the front face.
    ///
    /// This is the default with left handed coordinate spaces.
    Cw,
}

/// Face of a vertex.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Face {
    /// Front face
    Front,
    /// Back face
    Back,
}

#[derive(Clone, Debug, Default)]
pub struct PrimitiveState {
    /// The primitive topology used to interpret vertices.
    pub topology: PrimitiveTopology,
    /// The face to consider the front for the purpose of culling and stencil operations.
    pub front_face: FrontFace,
    /// The face culling mode.
    pub cull_mode: Option<Face>,
    /// If set to true, the polygon depth is not clipped to 0-1 before rasterization.
    pub unclipped_depth: bool,
    /// If true, only the primitive edges are rasterized..
    pub wireframe: bool,
}

/// Operation to perform on the stencil value.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum StencilOperation {
    /// Keep stencil value unchanged.
    #[default]
    Keep,
    /// Set stencil value to zero.
    Zero,
    /// Replace stencil value with value provided.
    Replace,
    /// Bitwise inverts stencil value.
    Invert,
    /// Increments stencil value by one, clamping on overflow.
    IncrementClamp,
    /// Decrements stencil value by one, clamping on underflow.
    DecrementClamp,
    /// Increments stencil value by one, wrapping on overflow.
    IncrementWrap,
    /// Decrements stencil value by one, wrapping on underflow.
    DecrementWrap,
}

/// Describes stencil state in a render pipeline.
///
/// If you are not using stencil state, set this to [`StencilFaceState::IGNORE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilFaceState {
    /// Comparison function that determines if the fail_op or pass_op is used on the stencil buffer.
    pub compare: CompareFunction,
    /// Operation that is preformed when stencil test fails.
    pub fail_op: StencilOperation,
    /// Operation that is performed when depth test fails but stencil test succeeds.
    pub depth_fail_op: StencilOperation,
    /// Operation that is performed when stencil test success.
    pub pass_op: StencilOperation,
}

impl StencilFaceState {
    /// Ignore the stencil state for the face.
    pub const IGNORE: Self = StencilFaceState {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self::IGNORE
    }
}

/// State of the stencil operation (fixed-pipeline stage).
///
/// For use in [`DepthStencilState`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct StencilState {
    /// Front face mode.
    pub front: StencilFaceState,
    /// Back face mode.
    pub back: StencilFaceState,
    /// Stencil values are AND'd with this mask when reading and writing from the stencil buffer. Only low 8 bits are used.
    pub read_mask: u32,
    /// Stencil values are AND'd with this mask when writing to the stencil buffer. Only low 8 bits are used.
    pub write_mask: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DepthBiasState {
    /// Constant depth biasing factor, in basic units of the depth format.
    pub constant: i32,
    /// Slope depth biasing factor.
    pub slope_scale: f32,
    /// Depth bias clamp value (absolute).
    pub clamp: f32,
}

/// Describes the depth/stencil state in a render pipeline.
#[derive(Clone, Debug)]
pub struct DepthStencilState {
    /// Format of the depth/stencil texture view.
    pub format: TextureFormat,
    /// If disabled, depth will not be written to.
    pub depth_write_enabled: bool,
    /// Comparison function used to compare depth values in the depth test.
    pub depth_compare: CompareFunction,
    /// Stencil state.
    pub stencil: StencilState,
    /// Depth bias state.
    pub bias: DepthBiasState,
}

/// Alpha blend factor.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum BlendFactor {
    /// 0.0
    Zero,
    /// 1.0
    One,
    /// S.component
    Src,
    /// 1.0 - S.component
    OneMinusSrc,
    /// S.alpha
    SrcAlpha,
    /// 1.0 - S.alpha
    OneMinusSrcAlpha,
    /// D.component
    Dst,
    /// 1.0 - D.component
    OneMinusDst,
    /// D.alpha
    DstAlpha,
    /// 1.0 - D.alpha
    OneMinusDstAlpha,
    /// min(S.alpha, 1.0 - D.alpha)
    SrcAlphaSaturated,
    /// Constant
    Constant,
    /// 1.0 - Constant
    OneMinusConstant,
}

/// Alpha blend operation.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum BlendOperation {
    /// Src + Dst
    #[default]
    Add,
    /// Src - Dst
    Subtract,
    /// Dst - Src
    ReverseSubtract,
    /// min(Src, Dst)
    Min,
    /// max(Src, Dst)
    Max,
}

/// Describes a blend component of a [`BlendState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    /// Multiplier for the source, which is produced by the fragment shader.
    pub src_factor: BlendFactor,
    /// Multiplier for the destination, which is stored in the target.
    pub dst_factor: BlendFactor,
    /// The binary operation applied to the source and destination,
    /// multiplied by their respective factors.
    pub operation: BlendOperation,
}

impl BlendComponent {
    /// Default blending state that replaces destination with the source.
    pub const REPLACE: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    };

    /// Blend state of (1 * src) + ((1 - src_alpha) * dst)
    pub const OVER: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    };
}

impl Default for BlendComponent {
    fn default() -> Self {
        Self::REPLACE
    }
}

/// Describe the blend state of a render pipeline,
/// within [`ColorTargetState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendState {
    /// Color equation.
    pub color: BlendComponent,
    /// Alpha equation.
    pub alpha: BlendComponent,
}

impl BlendState {
    /// Blend mode that does no color blending, just overwrites the output with the contents of the shader.
    pub const REPLACE: Self = Self {
        color: BlendComponent::REPLACE,
        alpha: BlendComponent::REPLACE,
    };

    /// Blend mode that does standard alpha blending with non-premultiplied alpha.
    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent::OVER,
    };

    /// Blend mode that does standard alpha blending with premultiplied alpha.
    pub const PREMULTIPLIED_ALPHA_BLENDING: Self = Self {
        color: BlendComponent::OVER,
        alpha: BlendComponent::OVER,
    };
}

bitflags::bitflags! {
    /// Color write mask. Disabled color channels will not be written to.
    #[repr(transparent)]
    pub struct ColorWrites: u32 {
        /// Enable red channel writes
        const RED = 1 << 0;
        /// Enable green channel writes
        const GREEN = 1 << 1;
        /// Enable blue channel writes
        const BLUE = 1 << 2;
        /// Enable alpha channel writes
        const ALPHA = 1 << 3;
        /// Enable red, green, and blue channel writes
        const COLOR = Self::RED.bits | Self::GREEN.bits | Self::BLUE.bits;
        /// Enable writes to all channels.
        const ALL = Self::RED.bits | Self::GREEN.bits | Self::BLUE.bits | Self::ALPHA.bits;
    }
}

impl Default for ColorWrites {
    fn default() -> Self {
        Self::ALL
    }
}

/// Describes the color state of a render pipeline.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColorTargetState {
    /// The [`TextureFormat`] of the image that this pipeline will render to.
    pub format: TextureFormat,
    /// The blending that is used for this pipeline.
    pub blend: Option<BlendState>,
    /// Mask which enables/disables writes to different color/alpha channel.
    pub write_mask: ColorWrites,
}

impl From<TextureFormat> for ColorTargetState {
    fn from(format: TextureFormat) -> Self {
        Self {
            format,
            blend: None,
            write_mask: ColorWrites::ALL,
        }
    }
}

pub struct RenderPipelineDesc<'a> {
    pub name: &'a str,
    pub data_layouts: &'a [&'a ShaderDataLayout],
    pub vertex: ShaderFunction<'a>,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub fragment: ShaderFunction<'a>,
    pub color_targets: &'a [ColorTargetState],
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
    Ignore,
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

#[derive(Debug)]
pub struct SurfaceConfig {
    pub size: Extent,
    pub usage: TextureUsage,
    pub frame_count: u32,
}
