use objc::{msg_send, sel, sel_impl};

fn map_texture_usage(usage: crate::TextureUsage) -> metal::MTLTextureUsage {
    use crate::TextureUsage as Tu;

    let mut mtl_usage = metal::MTLTextureUsage::Unknown;

    mtl_usage.set(
        metal::MTLTextureUsage::RenderTarget,
        usage.intersects(Tu::TARGET),
    );
    mtl_usage.set(
        metal::MTLTextureUsage::ShaderRead,
        usage.intersects(Tu::RESOURCE),
    );
    mtl_usage.set(
        metal::MTLTextureUsage::ShaderWrite,
        usage.intersects(Tu::STORAGE),
    );

    mtl_usage
}

fn map_texture_view_dimension(dimension: crate::TextureViewDimension) -> metal::MTLTextureType {
    use crate::TextureViewDimension as Tvd;
    use metal::MTLTextureType::*;
    match dimension {
        Tvd::D1 => D1,
        Tvd::D2 => D2,
        Tvd::D2Array => D2Array,
        Tvd::D3 => D3,
        Tvd::Cube => Cube,
        Tvd::CubeArray => CubeArray,
    }
}

fn map_filter_mode(filter: crate::FilterMode) -> metal::MTLSamplerMinMagFilter {
    use metal::MTLSamplerMinMagFilter::*;
    match filter {
        crate::FilterMode::Nearest => Nearest,
        crate::FilterMode::Linear => Linear,
    }
}

fn map_address_mode(address: crate::AddressMode) -> metal::MTLSamplerAddressMode {
    use crate::AddressMode as Am;
    use metal::MTLSamplerAddressMode::*;
    match address {
        Am::Repeat => Repeat,
        Am::MirrorRepeat => MirrorRepeat,
        Am::ClampToEdge => ClampToEdge,
        Am::ClampToBorder => ClampToBorderColor,
    }
}

fn map_border_color(color: crate::TextureColor) -> metal::MTLSamplerBorderColor {
    use crate::TextureColor as Tc;
    use metal::MTLSamplerBorderColor::*;
    match color {
        Tc::TransparentBlack => TransparentBlack,
        Tc::OpaqueBlack => OpaqueBlack,
        Tc::White => OpaqueWhite,
    }
}

impl super::Context {
    pub fn create_buffer(&self, desc: crate::BufferDesc) -> super::Buffer {
        let options = match desc.memory {
            crate::Memory::Device => metal::MTLResourceOptions::StorageModePrivate,
            crate::Memory::Shared => metal::MTLResourceOptions::StorageModeShared,
            crate::Memory::Upload => {
                metal::MTLResourceOptions::StorageModeShared
                    | metal::MTLResourceOptions::CPUCacheModeWriteCombined
            }
        };
        let raw = objc::rc::autoreleasepool(|| {
            let raw = self.device.lock().unwrap().new_buffer(desc.size, options);
            if !desc.name.is_empty() {
                raw.set_label(&desc.name);
            }
            unsafe { msg_send![raw.as_ref(), retain] }
        });
        super::Buffer { raw }
    }

    pub fn destroy_buffer(&self, buffer: super::Buffer) {
        unsafe {
            let () = msg_send![buffer.raw, release];
        }
    }

    pub fn create_texture(&self, desc: crate::TextureDesc) -> super::Texture {
        let mtl_format = super::map_texture_format(desc.format);

        let mtl_type = match desc.dimension {
            crate::TextureDimension::D1 => {
                if desc.array_layer_count > 1 {
                    metal::MTLTextureType::D1Array
                } else {
                    metal::MTLTextureType::D1
                }
            }
            crate::TextureDimension::D2 => {
                if desc.array_layer_count > 1 {
                    metal::MTLTextureType::D2Array
                } else {
                    metal::MTLTextureType::D2
                }
            }
            crate::TextureDimension::D3 => metal::MTLTextureType::D3,
        };
        let mtl_usage = map_texture_usage(desc.usage);

        let raw = objc::rc::autoreleasepool(|| {
            let descriptor = metal::TextureDescriptor::new();

            descriptor.set_texture_type(mtl_type);
            descriptor.set_width(desc.size.width as u64);
            descriptor.set_height(desc.size.height as u64);
            descriptor.set_depth(desc.size.depth as u64);
            descriptor.set_array_length(desc.array_layer_count as u64);
            descriptor.set_mipmap_level_count(desc.mip_level_count as u64);
            descriptor.set_pixel_format(mtl_format);
            descriptor.set_usage(mtl_usage);
            descriptor.set_storage_mode(metal::MTLStorageMode::Private);

            let raw = self.device.lock().unwrap().new_texture(&descriptor);
            if !desc.name.is_empty() {
                raw.set_label(desc.name);
            }

            unsafe { msg_send![raw.as_ref(), retain] }
        });

        super::Texture { raw }
    }

    pub fn create_texture_view(&self, desc: crate::TextureViewDesc) -> super::TextureView {
        let texture = desc.texture.as_ref();
        let mtl_format = super::map_texture_format(desc.format);
        let mtl_type = map_texture_view_dimension(desc.dimension);
        let mip_level_count = match desc.subresources.mip_level_count {
            Some(count) => count.get() as u64,
            None => texture.mipmap_level_count() - desc.subresources.base_mip_level as u64,
        };
        let array_layer_count = match desc.subresources.array_layer_count {
            Some(count) => count.get() as u64,
            None => texture.array_length() - desc.subresources.base_array_layer as u64,
        };

        let raw = objc::rc::autoreleasepool(|| {
            let raw = texture.new_texture_view_from_slice(
                mtl_format,
                mtl_type,
                metal::NSRange {
                    location: desc.subresources.base_mip_level as _,
                    length: mip_level_count,
                },
                metal::NSRange {
                    location: desc.subresources.base_array_layer as _,
                    length: array_layer_count,
                },
            );
            if !desc.name.is_empty() {
                raw.set_label(desc.name);
            }
            unsafe { msg_send![raw.as_ref(), retain] }
        });
        super::TextureView { raw }
    }

    pub fn destroy_texture_view(&self, view: super::TextureView) {
        unsafe {
            let () = msg_send![view.raw, release];
        }
    }

    pub fn create_sampler(&self, desc: crate::SamplerDesc) -> super::Sampler {
        let raw = objc::rc::autoreleasepool(|| {
            let descriptor = metal::SamplerDescriptor::new();

            descriptor.set_min_filter(map_filter_mode(desc.min_filter));
            descriptor.set_mag_filter(map_filter_mode(desc.mag_filter));
            descriptor.set_mip_filter(match desc.mipmap_filter {
                crate::FilterMode::Nearest => metal::MTLSamplerMipFilter::Nearest,
                crate::FilterMode::Linear => metal::MTLSamplerMipFilter::Linear,
            });

            descriptor.set_address_mode_s(map_address_mode(desc.address_modes[0]));
            descriptor.set_address_mode_t(map_address_mode(desc.address_modes[1]));
            descriptor.set_address_mode_r(map_address_mode(desc.address_modes[2]));

            if desc.anisotropy_clamp > 1 {
                descriptor.set_max_anisotropy(desc.anisotropy_clamp as u64);
            }

            descriptor.set_lod_min_clamp(desc.lod_min_clamp);
            if let Some(lod) = desc.lod_max_clamp {
                descriptor.set_lod_max_clamp(lod);
            }

            // optimization
            descriptor.set_lod_average(true);

            if let Some(fun) = desc.compare {
                descriptor.set_compare_function(super::map_compare_function(fun));
            }

            if let Some(border_color) = desc.border_color {
                descriptor.set_border_color(map_border_color(border_color));
            }

            if !desc.name.is_empty() {
                descriptor.set_label(desc.name);
            }
            let raw = self.device.lock().unwrap().new_sampler(&descriptor);
            unsafe { msg_send![raw.as_ref(), retain] }
        });

        super::Sampler { raw }
    }

    pub fn destroy_sampler(&self, sampler: super::Sampler) {
        unsafe {
            let () = msg_send![sampler.raw, release];
        }
    }
}
