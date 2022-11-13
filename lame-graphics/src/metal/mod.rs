use std::{
    ptr,
    sync::{Arc, Mutex},
};

use foreign_types::{ForeignTypeRef as _};
use objc::{msg_send, sel, sel_impl};

mod command;

pub struct Context {
    device: Mutex<metal::Device>,
    queue: Arc<Mutex<metal::CommandQueue>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Buffer {
    raw: *mut metal::MTLBuffer,
}

impl Buffer {
    pub fn is_valid(&self) -> bool {
        !self.raw.is_null()
    }

    fn as_ref(&self) -> &metal::BufferRef {
        unsafe { metal::BufferRef::from_ptr(self.raw) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Texture {
    raw: ptr::NonNull<metal::MTLTexture>,
}

impl Texture {
    fn as_ref(&self) -> &metal::TextureRef {
        unsafe { metal::TextureRef::from_ptr(self.raw.as_ptr()) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureView {
    raw: ptr::NonNull<metal::MTLTexture>,
}

impl TextureView {
    fn as_ref(&self) -> &metal::TextureRef {
        unsafe { metal::TextureRef::from_ptr(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct CommandEncoder {
    raw: Option<metal::CommandBuffer>,
    queue: Arc<Mutex<metal::CommandQueue>>,
}

#[derive(Debug)]
pub struct RenderPipeline {
    raw: metal::RenderPipelineState,
    primitive_type: metal::MTLPrimitiveType,
    triangle_fill_mode: metal::MTLTriangleFillMode,
    front_winding: metal::MTLWinding,
    cull_mode: metal::MTLCullMode,
    depth_clip_mode: Option<metal::MTLDepthClipMode>,
    depth_stencil: Option<(metal::DepthStencilState, super::DepthBiasState)>,
}

#[derive(Debug)]
pub struct RenderCommandEncoder<'a> {
    owner: &'a mut metal::CommandBuffer,
    raw: metal::RenderCommandEncoder,
}

#[derive(Debug)]
pub struct RenderPipelineContext<'a> {
    encoder: &'a mut metal::RenderCommandEncoder,
    primitive_type: metal::MTLPrimitiveType,
}

impl Context {
    pub unsafe fn init(desc: super::ContextDesc) -> Result<Self, super::NotSupportedError> {
        if desc.validation {
            std::env::set_var("METAL_DEVICE_WRAPPER_TYPE", "1");
        }
        let device = metal::Device::system_default()
            .ok_or(super::NotSupportedError)?;
        let queue = device.new_command_queue();
        Ok(Context {
            device: Mutex::new(device),
            queue: Arc::new(Mutex::new(queue)),
        })
    }

    pub fn create_buffer(&self, desc: super::BufferDesc) -> Buffer {
        let options = match desc.memory {
            super::Memory::Device =>
                metal::MTLResourceOptions::StorageModePrivate,
            super::Memory::Shared =>
                metal::MTLResourceOptions::StorageModeShared,
            super::Memory::Upload => metal::MTLResourceOptions::StorageModeShared | metal::MTLResourceOptions::CPUCacheModeWriteCombined,
        };
        let raw = objc::rc::autoreleasepool(|| unsafe {
            let raw = self.device.lock().unwrap().new_buffer(desc.size, options);
            if !desc.name.is_empty() {
                raw.set_label(&desc.name);
            }
            msg_send![raw.as_ref(), retain]
        });
        Buffer {
            raw,
        }
    }

    pub fn destroy_buffer(&self, buffer: Buffer) {
        unsafe {
            let () = msg_send![buffer.raw, release];
        }
    }

    pub fn create_texture(&self, desc: super::TextureDesc) -> Texture {
        unimplemented!()
    }

    pub fn create_texture_view(&self, desc: super::TextureViewDesc) -> TextureView {
        unimplemented!()
    }

    pub fn create_command_encoder(&self, desc: super::CommandEncoderDesc) -> CommandEncoder {
        CommandEncoder {
            raw: None,
            queue: Arc::clone(&self.queue),
        }
    }

    pub fn create_render_pipeline(&self, desc: super::RenderPipelineDesc) -> RenderPipeline {
        unimplemented!()
    }

    pub fn submit(&self, encoder: &mut CommandEncoder) {
        encoder.raw.take().unwrap().commit();
    }
}
