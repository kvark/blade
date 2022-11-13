use std::{
    ptr::NonNull,
    sync::Mutex,
};

use foreign_types::ForeignType as _;
use objc::{msg_send, sel, sel_impl};

pub struct Context {
    device: Mutex<metal::Device>,
    queue: Mutex<metal::CommandQueue>,
}

#[derive(Debug)]
pub struct Buffer {
    raw: NonNull<metal::MTLBuffer>,
}

#[derive(Debug)]
pub struct Texture {
    raw: NonNull<metal::MTLTexture>,
}

#[derive(Debug)]
pub struct TextureView {
    raw: NonNull<metal::MTLTexture>,
}

#[derive(Debug)]
pub struct CommandEncoder {
    raw: metal::CommandEncoder,
}

#[derive(Debug)]
pub struct RenderPipeline {
    raw: metal::RenderPipelineState,
}

impl Context {
    pub unsafe fn init(desc: &super::ContextDesc) -> Result<Self, super::NotSupportedError> {
        if desc.validation {
            std::env::set_var("METAL_DEVICE_WRAPPER_TYPE", "1");
        }
        let device = metal::Device::system_default()
            .ok_or(super::NotSupportedError)?;
        let queue = device.new_command_queue();
        Ok(Context {
            device: Mutex::new(device),
            queue: Mutex::new(queue),
        })
    }

    pub fn create_buffer(&self, desc: &super::BufferDesc) -> Buffer {
        let options = match desc.memory {
            super::Memory::Device =>
                metal::MTLResourceOptions::StorageModePrivate,
            super::Memory::Shared =>
                metal::MTLResourceOptions::StorageModeShared,
            super::Memory::Upload => metal::MTLResourceOptions::StorageModeShared | metal::MTLResourceOptions::CPUCacheModeWriteCombined,
        };
        let raw = objc::rc::autoreleasepool(|| unsafe {
            let raw = self.device.lock().unwrap().new_buffer(desc.size, options);
            if !desc.label.is_empty() {
                raw.set_label(desc.label);
            }
            let _: *mut () = msg_send![raw.as_ref(), retain];
            NonNull::new_unchecked(raw.as_ptr())
        });
        Buffer {
            raw,
        }
    }

    pub fn destroy_buffer(&self, buffer: Buffer) {
        unsafe {
            let () = msg_send![buffer.raw.as_ptr(), release];
        }
    }

    pub fn create_texture(&self, desc: &super::TextureDesc) -> Texture {
        unimplemented!()
    }

    pub fn create_texture_view(&self, desc: &super::TextureViewDesc) -> TextureView {
        unimplemented!()
    }

    pub fn create_command_encoder(&self) -> CommandEncoder {
        unimplemented!()
    }

    pub fn create_render_pipeline(&self, desc: &super::RenderPipelineDesc) -> RenderPipeline {
        unimplemented!()
    }
}
