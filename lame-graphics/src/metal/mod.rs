use std::{
    ptr,
    sync::{Arc, Mutex},
};

use foreign_types::{ForeignTypeRef as _};
use objc::{msg_send, sel, sel_impl};

mod command;
mod pipeline;

pub struct Context {
    device: Mutex<metal::Device>,
    queue: Arc<Mutex<metal::CommandQueue>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Buffer {
    raw: *mut metal::MTLBuffer,
}

impl Default for Buffer {
    fn default() -> Self {
        Self {
            raw: ptr::null_mut(),
        }
    }
}

impl Buffer {
    fn as_ref(&self) -> &metal::BufferRef {
        unsafe { metal::BufferRef::from_ptr(self.raw) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Texture {
    raw: *mut metal::MTLTexture,
}

impl Default for Texture {
    fn default() -> Self {
        Self {
            raw: ptr::null_mut(),
        }
    }
}

impl Texture {
    fn as_ref(&self) -> &metal::TextureRef {
        unsafe { metal::TextureRef::from_ptr(self.raw) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureView {
    raw: *mut metal::MTLTexture,
}

impl Default for TextureView {
    fn default() -> Self {
        Self {
            raw: ptr::null_mut(),
        }
    }
}

impl TextureView {
    fn as_ref(&self) -> &metal::TextureRef {
        unsafe { metal::TextureRef::from_ptr(self.raw) }
    }
}

#[derive(Debug)]
pub struct CommandEncoder {
    raw: Option<metal::CommandBuffer>,
    queue: Arc<Mutex<metal::CommandQueue>>,
    plain_data: Vec<u8>,
}

#[derive(Debug)]
struct BindGroupInfo {
    visibility: crate::ShaderVisibility,
    targets: Box<[u32]>,
    plain_buffer_slot: Option<u32>,
    plain_data_size: u32,
}

#[derive(Debug)]
pub struct RenderPipeline {
    raw: metal::RenderPipelineState,
    #[allow(dead_code)]
    vs_lib: metal::Library,
    #[allow(dead_code)]
    fs_lib: metal::Library,
    bind_groups: Box<[BindGroupInfo]>,
    primitive_type: metal::MTLPrimitiveType,
    triangle_fill_mode: metal::MTLTriangleFillMode,
    front_winding: metal::MTLWinding,
    cull_mode: metal::MTLCullMode,
    depth_clip_mode: metal::MTLDepthClipMode,
    depth_stencil: Option<(metal::DepthStencilState, super::DepthBiasState)>,
}

#[derive(Debug)]
pub struct RenderCommandEncoder<'a> {
    raw: metal::RenderCommandEncoder,
    plain_data: &'a mut Vec<u8>,
}

#[derive(Debug)]
pub struct RenderPipelineContext<'a> {
    encoder: &'a mut metal::RenderCommandEncoder,
    primitive_type: metal::MTLPrimitiveType,
    bind_groups: &'a [BindGroupInfo],
    plain_data: &'a mut [u8],
}

struct PerStageCounter {
    vs: u32,
    fs: u32,
    cs: u32,
}

fn map_texture_format(format: super::TextureFormat) -> metal::MTLPixelFormat {
    use super::TextureFormat as Tf;
    use metal::MTLPixelFormat::*;
    match format {
        Tf::Rgba8Unorm => RGBA8Unorm,
    }
}

fn map_texture_usage(usage: crate::TextureUsage) -> metal::MTLTextureUsage {
    use crate::TextureUsage as Tu;

    let mut mtl_usage = metal::MTLTextureUsage::Unknown;

    mtl_usage.set(
        metal::MTLTextureUsage::RenderTarget,
        usage.intersects(Tu::TARGET),
    );
    mtl_usage.set(
        metal::MTLTextureUsage::ShaderRead,
        usage.intersects(
            Tu::RESOURCE,
        ),
    );
    mtl_usage.set(metal::MTLTextureUsage::ShaderWrite, usage.intersects(Tu::STORAGE));

    mtl_usage
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
        let raw = objc::rc::autoreleasepool(|| {
            let raw = self.device.lock().unwrap().new_buffer(desc.size, options);
            if !desc.name.is_empty() {
                raw.set_label(&desc.name);
            }
            unsafe {
                msg_send![raw.as_ref(), retain]
            }
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
        let mtl_format = map_texture_format(desc.format);

        let mtl_type = match desc.dimension {
            crate::TextureDimension::D1 => {
                if desc.array_layers > 1 {
                    metal::MTLTextureType::D1Array
                } else {
                    metal::MTLTextureType::D1
                }
            }
            crate::TextureDimension::D2 => {
                if desc.array_layers > 1 {
                    metal::MTLTextureType::D2Array
                } else {
                    metal::MTLTextureType::D2
                }
            }
            crate::TextureDimension::D3 => {
                metal::MTLTextureType::D3
            }
        };
        let mtl_usage = map_texture_usage(desc.usage);

        let raw = objc::rc::autoreleasepool(|| {
            let descriptor = metal::TextureDescriptor::new();

            descriptor.set_texture_type(mtl_type);
            descriptor.set_width(desc.size.width as u64);
            descriptor.set_height(desc.size.height as u64);
            descriptor.set_depth(desc.size.depth as u64);
            descriptor.set_array_length(desc.array_layers as u64);
            descriptor.set_mipmap_level_count(desc.mip_level_count as u64);
            descriptor.set_pixel_format(mtl_format);
            descriptor.set_usage(mtl_usage);
            descriptor.set_storage_mode(metal::MTLStorageMode::Private);

            let raw = self.device.lock().unwrap().new_texture(&descriptor);
            if !desc.name.is_empty() {
                raw.set_label(desc.name);
            }

            unsafe {
                msg_send![raw.as_ref(), retain]
            }
        });

        Texture {
            raw,
        }
    }

    pub fn create_texture_view(&self, desc: super::TextureViewDesc) -> TextureView {
        //TODO: proper subresource selection
        let raw = unsafe {
            msg_send![desc.texture.raw, retain]
        };
        TextureView {
            raw
        }
    }

    pub fn create_command_encoder(&self, desc: super::CommandEncoderDesc) -> CommandEncoder {
        CommandEncoder {
            raw: None,
            queue: Arc::clone(&self.queue),
            plain_data: Vec::new(),
        }
    }

    pub fn submit(&self, encoder: &mut CommandEncoder) {
        encoder.raw.take().unwrap().commit();
    }
}
