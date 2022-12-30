mod command;
mod pipeline;
#[cfg_attr(not(target_arch = "wasm32"), path = "egl.rs")]
#[cfg_attr(target_arch = "wasm32", path = "web.rs")]
mod platform;
mod resource;

type BindTarget = u32;

pub use platform::Context;
use std::{marker::PhantomData, ops::Range};

const DEBUG_ID: u32 = 0;

bitflags::bitflags! {
    struct Capabilities: u32 {
        const BUFFER_STORAGE = 1 << 0;
    }
}

#[derive(Clone, Debug)]
struct Limits {
    uniform_buffer_alignment: u32,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Buffer {
    raw: glow::Buffer,
    size: u64,
    data: *mut u8,
}

impl Buffer {
    pub fn data(&self) -> *mut u8 {
        self.data
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
enum TextureInner {
    Renderbuffer {
        raw: glow::Renderbuffer,
    },
    Texture {
        raw: glow::Texture,
        target: BindTarget,
    },
}

impl TextureInner {
    fn as_native(&self) -> (glow::Texture, BindTarget) {
        match *self {
            Self::Renderbuffer { .. } => {
                panic!("Unexpected renderbuffer");
            }
            Self::Texture { raw, target } => (raw, target),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Texture {
    inner: TextureInner,
    target_size: [u16; 2],
    format: crate::TextureFormat,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct TextureView {
    inner: TextureInner,
    target_size: [u16; 2],
    aspects: crate::TexelAspects,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Sampler {
    raw: glow::Sampler,
}

type SlotList = Vec<u32>;

struct BindGroupInfo {
    targets: Box<[SlotList]>,
}

struct PipelineInner {
    program: glow::Program,
    bind_group_infos: Box<[BindGroupInfo]>,
}

pub struct ComputePipeline {
    inner: PipelineInner,
    wg_size: [u32; 3],
}

impl ComputePipeline {
    pub fn get_workgroup_size(&self) -> [u32; 3] {
        self.wg_size
    }
}

pub struct RenderPipeline {
    inner: PipelineInner,
    topology: crate::PrimitiveTopology,
}

pub struct Frame {
    texture: Texture,
}

impl Frame {
    pub fn texture(&self) -> Texture {
        self.texture
    }

    pub fn texture_view(&self) -> TextureView {
        TextureView {
            inner: self.texture.inner,
            target_size: self.texture.target_size,
            aspects: crate::TexelAspects::COLOR,
        }
    }
}

#[derive(Clone, Debug)]
struct BufferPart {
    raw: glow::Buffer,
    offset: u64,
}
impl From<crate::BufferPiece> for BufferPart {
    fn from(piece: crate::BufferPiece) -> Self {
        Self {
            raw: piece.buffer.raw,
            offset: piece.offset,
        }
    }
}

#[derive(Clone, Debug)]
struct TexturePart {
    raw: glow::Texture,
    target: BindTarget,
    format: crate::TextureFormat,
    mip_level: u32,
    array_layer: u32,
    origin: [u32; 3],
}
impl From<crate::TexturePiece> for TexturePart {
    fn from(piece: crate::TexturePiece) -> Self {
        let (raw, target) = piece.texture.inner.as_native();
        Self {
            raw,
            target,
            format: piece.texture.format,
            mip_level: piece.mip_level,
            array_layer: piece.array_layer,
            origin: piece.origin,
        }
    }
}

#[derive(Clone, Debug)]
struct ImageBinding {
    raw: glow::Texture,
    mip_level: u32,
    array_layer: Option<u32>,
    access: u32,
    format: u32,
}

#[derive(Clone, Copy, Debug)]
enum ColorType {
    Float,
    Uint,
    Sint,
}

#[derive(Debug)]
enum Command {
    Draw {
        topology: u32,
        start_vertex: u32,
        vertex_count: u32,
        instance_count: u32,
    },
    DrawIndexed {
        topology: u32,
        index_buf: BufferPart,
        index_type: u32,
        index_count: u32,
        base_vertex: i32,
        instance_count: u32,
    },
    DrawIndirect {
        topology: u32,
        indirect_buf: BufferPart,
    },
    DrawIndexedIndirect {
        topology: u32,
        raw_index_buf: glow::Buffer,
        index_type: u32,
        indirect_buf: BufferPart,
    },
    Dispatch([u32; 3]),
    DispatchIndirect {
        indirect_buf: BufferPart,
    },
    FillBuffer {
        dst: BufferPart,
        size: u64,
        value: u8,
    },
    CopyBufferToBuffer {
        src: BufferPart,
        dst: BufferPart,
        size: u64,
    },
    CopyTextureToTexture {
        src: TexturePart,
        dst: TexturePart,
        size: crate::Extent,
    },
    CopyBufferToTexture {
        src: BufferPart,
        dst: TexturePart,
        bytes_per_row: u32,
        size: crate::Extent,
    },
    CopyTextureToBuffer {
        src: TexturePart,
        dst: BufferPart,
        bytes_per_row: u32,
        size: crate::Extent,
    },
    ResetFramebuffer,
    BindAttachment {
        attachment: u32,
        view: TextureView,
    },
    InvalidateAttachment(u32),
    SetDrawColorBuffers(u8),
    ClearColor {
        draw_buffer: u32,
        color: crate::TextureColor,
        ty: ColorType,
    },
    ClearDepthStencil {
        depth: Option<f32>,
        stencil: Option<u32>,
    },
    Barrier,
    SetViewport {
        size: [u16; 2],
        depth: Range<f32>,
    },
    SetScissor(crate::ScissorRect),
    SetStencilFunc {
        face: u32,
        function: u32,
        reference: u32,
        read_mask: u32,
    },
    SetStencilOps {
        face: u32,
        write_mask: u32,
        //ops: crate::StencilOps,
    },
    //SetDepth(DepthState),
    //SetDepthBias(wgt::DepthBiasState),
    //ConfigureDepthStencil(crate::FormatAspects),
    SetProgram(glow::Program),
    UnsetProgram,
    //SetPrimitive(PrimitiveState),
    SetBlendConstant([f32; 4]),
    SetColorTarget {
        draw_buffer_index: Option<u32>,
        //desc: ColorTargetDesc,
    },
    BindUniform {
        slot: u32,
        offset: u32,
        size: u32,
    },
    BindBuffer {
        target: BindTarget,
        slot: u32,
        buffer: BufferPart,
    },
    BindSampler {
        slot: u32,
        sampler: glow::Sampler,
    },
    BindTexture {
        slot: u32,
        texture: glow::Texture,
        target: BindTarget,
    },
    BindImage {
        slot: u32,
        binding: ImageBinding,
    },
    ResetAllSamplers,
}

pub struct CommandEncoder {
    name: String,
    commands: Vec<Command>,
    plain_data: Vec<u8>,
    has_present: bool,
    limits: Limits,
}

enum PassKind {
    Transfer,
    Compute,
    Render,
}

pub struct PassEncoder<'a, P> {
    commands: &'a mut Vec<Command>,
    plain_data: &'a mut Vec<u8>,
    kind: PassKind,
    invalidate_attachments: Vec<u32>,
    pipeline: PhantomData<P>,
    limits: &'a Limits,
}

pub type ComputeCommandEncoder<'a> = PassEncoder<'a, ComputePipeline>;
pub type RenderCommandEncoder<'a> = PassEncoder<'a, RenderPipeline>;

pub struct PipelineEncoder<'a> {
    commands: &'a mut Vec<Command>,
    plain_data: &'a mut Vec<u8>,
    bind_group_infos: &'a [BindGroupInfo],
    topology: u32,
    limits: &'a Limits,
}

pub struct PipelineContext<'a> {
    commands: &'a mut Vec<Command>,
    plain_data: &'a mut Vec<u8>,
    targets: &'a [SlotList],
    limits: &'a Limits,
}

#[derive(Clone, Debug)]
pub struct SyncPoint {}

struct ExecutionContext {
    framebuf: glow::Framebuffer,
    plain_buffer: glow::Buffer,
}

#[hidden_trait::expose]
impl crate::traits::CommandDevice for Context {
    type CommandEncoder = CommandEncoder;
    type SyncPoint = SyncPoint;

    fn create_command_encoder(&self, desc: super::CommandEncoderDesc) -> CommandEncoder {
        CommandEncoder {
            name: desc.name.to_string(),
            commands: Vec::new(),
            plain_data: Vec::new(),
            has_present: false,
            limits: self.limits.clone(),
        }
    }

    fn destroy_command_encoder(&self, _command_encoder: CommandEncoder) {}

    fn submit(&self, encoder: &mut CommandEncoder) -> SyncPoint {
        {
            use glow::HasContext as _;

            let gl = self.lock();
            let push_group = !encoder.name.is_empty() && gl.supports_debug();
            let ec = unsafe {
                if push_group {
                    gl.push_debug_group(glow::DEBUG_SOURCE_APPLICATION, DEBUG_ID, &encoder.name);
                }
                let framebuf = gl.create_framebuffer().unwrap();
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuf));
                let plain_buffer = gl.create_buffer().unwrap();
                if !encoder.plain_data.is_empty() {
                    log::trace!("Allocating plain data of size {}", encoder.plain_data.len());
                    gl.bind_buffer(glow::UNIFORM_BUFFER, Some(plain_buffer));
                    gl.buffer_data_u8_slice(
                        glow::UNIFORM_BUFFER,
                        &encoder.plain_data,
                        glow::STATIC_DRAW,
                    );
                }
                ExecutionContext {
                    framebuf,
                    plain_buffer,
                }
            };
            for command in encoder.commands.iter() {
                log::trace!("{:?}", command);
                unsafe { command.execute(&*gl, &ec) };
            }
            unsafe {
                gl.delete_framebuffer(ec.framebuf);
                gl.delete_buffer(ec.plain_buffer);
                if push_group {
                    gl.pop_debug_group();
                }
            }
        }
        if encoder.has_present {
            self.present();
        }
        SyncPoint {}
    }

    fn wait_for(&self, _sp: &SyncPoint, _timeout_ms: u32) -> bool {
        false //TODO
    }
}

// Align the size up to 16 bytes, as expected by GL.
fn round_up_uniform_size(size: u32) -> u32 {
    if size & 0xF != 0 {
        (size | 0xF) + 1
    } else {
        size
    }
}

struct FormatInfo {
    internal: u32,
    external: u32,
    data_type: u32,
}

fn describe_texture_format(format: crate::TextureFormat) -> FormatInfo {
    use crate::TextureFormat as Tf;
    let (internal, external, data_type) = match format {
        Tf::Rgba8Unorm => (glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE),
        Tf::Rgba8UnormSrgb => (glow::SRGB8_ALPHA8, glow::RGBA, glow::UNSIGNED_BYTE),
        Tf::Bgra8UnormSrgb => (glow::SRGB8_ALPHA8, glow::BGRA, glow::UNSIGNED_BYTE),
        Tf::Depth32Float => (glow::DEPTH_COMPONENT32F, glow::DEPTH_COMPONENT, glow::FLOAT),
    };
    FormatInfo {
        internal,
        external,
        data_type,
    }
}

fn map_compare_func(fun: crate::CompareFunction) -> u32 {
    use crate::CompareFunction as Cf;
    match fun {
        Cf::Never => glow::NEVER,
        Cf::Less => glow::LESS,
        Cf::LessEqual => glow::LEQUAL,
        Cf::Equal => glow::EQUAL,
        Cf::GreaterEqual => glow::GEQUAL,
        Cf::Greater => glow::GREATER,
        Cf::NotEqual => glow::NOTEQUAL,
        Cf::Always => glow::ALWAYS,
    }
}
