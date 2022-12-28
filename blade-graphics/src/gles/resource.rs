use glow::HasContext as _;
use std::{mem, ptr};

#[hidden_trait::expose]
impl crate::traits::ResourceDevice for super::Context {
    type Buffer = super::Buffer;
    type Texture = super::Texture;
    type TextureView = super::TextureView;
    type Sampler = super::Sampler;

    fn create_buffer(&self, desc: crate::BufferDesc) -> super::Buffer {
        let gl = self.lock();

        let mut storage_flags = 0;
        let mut map_flags = 0;
        match desc.memory {
            crate::Memory::Device => {}
            crate::Memory::Shared => {
                map_flags = glow::MAP_READ_BIT | glow::MAP_WRITE_BIT | glow::MAP_UNSYNCHRONIZED_BIT;
                storage_flags |= glow::MAP_PERSISTENT_BIT
                    | glow::MAP_COHERENT_BIT
                    | glow::MAP_READ_BIT
                    | glow::MAP_WRITE_BIT;
            }
            crate::Memory::Upload => {
                map_flags = glow::MAP_WRITE_BIT | glow::MAP_UNSYNCHRONIZED_BIT;
                storage_flags |=
                    glow::MAP_PERSISTENT_BIT | glow::MAP_COHERENT_BIT | glow::MAP_WRITE_BIT;
            }
        }

        let raw = unsafe { gl.create_buffer() }.unwrap();
        let mut data = ptr::null_mut();
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(raw));
            gl.buffer_storage(glow::ARRAY_BUFFER, desc.size as _, None, storage_flags);
            if map_flags != 0 {
                data = gl.map_buffer_range(glow::ARRAY_BUFFER, 0, desc.size as _, map_flags);
            }
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            if !desc.name.is_empty() && gl.supports_debug() {
                gl.object_label(glow::BUFFER, mem::transmute(raw), Some(desc.name));
            }
        }
        super::Buffer { raw, data }
    }

    fn destroy_buffer(&self, buffer: super::Buffer) {
        let gl = self.lock();
        unsafe { gl.delete_buffer(buffer.raw) };
    }

    fn create_texture(&self, desc: crate::TextureDesc) -> super::Texture {
        let gl = self.lock();
        let format_desc = super::describe_texture_format(desc.format);

        let inner = if crate::TextureUsage::TARGET.contains(desc.usage)
            && desc.dimension == crate::TextureDimension::D2
            && desc.array_layer_count == 1
        {
            let raw = unsafe { gl.create_renderbuffer().unwrap() };
            unsafe {
                gl.bind_renderbuffer(glow::RENDERBUFFER, Some(raw));
                gl.renderbuffer_storage(
                    glow::RENDERBUFFER,
                    format_desc.internal,
                    desc.size.width as i32,
                    desc.size.height as i32,
                );
                gl.bind_renderbuffer(glow::RENDERBUFFER, None);
                if !desc.name.is_empty() && gl.supports_debug() {
                    gl.object_label(glow::RENDERBUFFER, mem::transmute(raw), Some(desc.name));
                }
            }
            super::TextureInner::Renderbuffer { raw }
        } else {
            let raw = unsafe { gl.create_texture().unwrap() };
            let target = match desc.dimension {
                crate::TextureDimension::D1 => {
                    if desc.array_layer_count > 1 {
                        glow::TEXTURE_1D_ARRAY
                    } else {
                        glow::TEXTURE_1D
                    }
                }
                crate::TextureDimension::D2 => {
                    if desc.array_layer_count > 1 {
                        glow::TEXTURE_2D_ARRAY
                    } else {
                        glow::TEXTURE_2D
                    }
                }
                crate::TextureDimension::D3 => glow::TEXTURE_3D,
            };

            unsafe {
                gl.bind_texture(target, Some(raw));
                // Reset default filtering mode. This has to be done before
                // assigning the storage.
                gl.tex_parameter_i32(target, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(target, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);

                match desc.dimension {
                    crate::TextureDimension::D3 => {
                        gl.tex_storage_3d(
                            target,
                            desc.mip_level_count as i32,
                            format_desc.internal,
                            desc.size.width as i32,
                            desc.size.height as i32,
                            desc.size.depth as i32,
                        );
                    }
                    crate::TextureDimension::D2 => {
                        gl.tex_storage_2d(
                            target,
                            desc.mip_level_count as i32,
                            format_desc.internal,
                            desc.size.width as i32,
                            desc.size.height as i32,
                        );
                    }
                    crate::TextureDimension::D1 => {
                        gl.tex_storage_1d(
                            target,
                            desc.mip_level_count as i32,
                            format_desc.internal,
                            desc.size.width as i32,
                        );
                    }
                }

                gl.bind_texture(target, None);
                if !desc.name.is_empty() && gl.supports_debug() {
                    gl.object_label(glow::TEXTURE, mem::transmute(raw), Some(desc.name));
                }
            }

            super::TextureInner::Texture { raw, target }
        };

        super::Texture {
            inner,
            format: desc.format,
        }
    }

    fn destroy_texture(&self, texture: super::Texture) {
        let gl = self.lock();
        match texture.inner {
            super::TextureInner::Renderbuffer { raw, .. } => unsafe {
                gl.delete_renderbuffer(raw);
            },
            super::TextureInner::DefaultRenderbuffer => {}
            super::TextureInner::Texture { raw, .. } => unsafe {
                gl.delete_texture(raw);
            },
        }
    }

    fn create_texture_view(&self, desc: crate::TextureViewDesc) -> super::TextureView {
        super::TextureView {
            inner: desc.texture.inner,
        }
    }

    fn destroy_texture_view(&self, _view: super::TextureView) {}

    fn create_sampler(&self, desc: crate::SamplerDesc) -> super::Sampler {
        let gl = self.lock();

        let wrap_enums = [
            glow::TEXTURE_WRAP_S,
            glow::TEXTURE_WRAP_T,
            glow::TEXTURE_WRAP_R,
        ];
        let (min, mag) = map_filter_modes(desc.min_filter, desc.mag_filter, desc.mipmap_filter);
        let border = match desc.border_color {
            None | Some(crate::TextureColor::TransparentBlack) => [0.0; 4],
            Some(crate::TextureColor::OpaqueBlack) => [0.0, 0.0, 0.0, 1.0],
            Some(crate::TextureColor::White) => [1.0; 4],
        };

        let raw = unsafe { gl.create_sampler().unwrap() };
        unsafe {
            gl.sampler_parameter_i32(raw, glow::TEXTURE_MIN_FILTER, min as i32);
            gl.sampler_parameter_i32(raw, glow::TEXTURE_MAG_FILTER, mag as i32);

            for (&address_mode, wrap_enum) in desc.address_modes.iter().zip(wrap_enums) {
                gl.sampler_parameter_i32(raw, wrap_enum, map_address_mode(address_mode) as i32)
            }
            if desc.border_color.is_some() {
                gl.sampler_parameter_f32_slice(raw, glow::TEXTURE_BORDER_COLOR, &border)
            }
            gl.sampler_parameter_f32(raw, glow::TEXTURE_MIN_LOD, desc.lod_min_clamp);
            if let Some(clamp) = desc.lod_max_clamp {
                gl.sampler_parameter_f32(raw, glow::TEXTURE_MAX_LOD, clamp);
            }
            if desc.anisotropy_clamp > 1 {
                gl.sampler_parameter_i32(
                    raw,
                    glow::TEXTURE_MAX_ANISOTROPY,
                    desc.anisotropy_clamp as i32,
                );
            }

            if let Some(compare) = desc.compare {
                gl.sampler_parameter_i32(
                    raw,
                    glow::TEXTURE_COMPARE_MODE,
                    glow::COMPARE_REF_TO_TEXTURE as i32,
                );
                gl.sampler_parameter_i32(
                    raw,
                    glow::TEXTURE_COMPARE_FUNC,
                    super::map_compare_func(compare) as i32,
                );
            }

            if !desc.name.is_empty() && gl.supports_debug() {
                gl.object_label(glow::SAMPLER, mem::transmute(raw), Some(desc.name));
            }
        }
        super::Sampler { raw }
    }

    fn destroy_sampler(&self, sampler: super::Sampler) {
        let gl = self.lock();
        unsafe { gl.delete_sampler(sampler.raw) };
    }
}

fn map_filter_modes(
    min: crate::FilterMode,
    mag: crate::FilterMode,
    mip: crate::FilterMode,
) -> (u32, u32) {
    use crate::FilterMode as Fm;

    let mag_filter = match mag {
        Fm::Nearest => glow::NEAREST,
        Fm::Linear => glow::LINEAR,
    };

    let min_filter = match (min, mip) {
        (Fm::Nearest, Fm::Nearest) => glow::NEAREST_MIPMAP_NEAREST,
        (Fm::Nearest, Fm::Linear) => glow::NEAREST_MIPMAP_LINEAR,
        (Fm::Linear, Fm::Nearest) => glow::LINEAR_MIPMAP_NEAREST,
        (Fm::Linear, Fm::Linear) => glow::LINEAR_MIPMAP_LINEAR,
    };

    (min_filter, mag_filter)
}

fn map_address_mode(mode: crate::AddressMode) -> u32 {
    match mode {
        crate::AddressMode::Repeat => glow::REPEAT,
        crate::AddressMode::MirrorRepeat => glow::MIRRORED_REPEAT,
        crate::AddressMode::ClampToEdge => glow::CLAMP_TO_EDGE,
        crate::AddressMode::ClampToBorder => glow::CLAMP_TO_BORDER,
    }
}
