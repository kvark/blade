use std::ptr;

pub struct DummyResources {
    pub size: blade_graphics::Extent,
    pub white_texture: blade_graphics::Texture,
    pub white_view: blade_graphics::TextureView,
    pub black_texture: blade_graphics::Texture,
    pub black_view: blade_graphics::TextureView,
    pub red_texture: blade_graphics::Texture,
    pub red_view: blade_graphics::TextureView,
    staging_buf: blade_graphics::Buffer,
}

impl DummyResources {
    pub fn new(
        command_encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
    ) -> Self {
        let size = blade_graphics::Extent {
            width: 1,
            height: 1,
            depth: 1,
        };
        let white_texture = gpu.create_texture(blade_graphics::TextureDesc {
            name: "dummy/white",
            format: blade_graphics::TextureFormat::Rgba8Unorm,
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: blade_graphics::TextureDimension::D2,
            usage: blade_graphics::TextureUsage::COPY | blade_graphics::TextureUsage::RESOURCE,
        });
        let white_view = gpu.create_texture_view(blade_graphics::TextureViewDesc {
            name: "dummy/white",
            texture: white_texture,
            format: blade_graphics::TextureFormat::Rgba8Unorm,
            dimension: blade_graphics::ViewDimension::D2,
            subresources: &blade_graphics::TextureSubresources::default(),
        });
        let black_texture = gpu.create_texture(blade_graphics::TextureDesc {
            name: "dummy/black",
            format: blade_graphics::TextureFormat::Rgba8Unorm,
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: blade_graphics::TextureDimension::D2,
            usage: blade_graphics::TextureUsage::COPY | blade_graphics::TextureUsage::RESOURCE,
        });
        let black_view = gpu.create_texture_view(blade_graphics::TextureViewDesc {
            name: "dummy/black",
            texture: black_texture,
            format: blade_graphics::TextureFormat::Rgba8Unorm,
            dimension: blade_graphics::ViewDimension::D2,
            subresources: &blade_graphics::TextureSubresources::default(),
        });
        let red_texture = gpu.create_texture(blade_graphics::TextureDesc {
            name: "dummy/red",
            format: blade_graphics::TextureFormat::Rgba8Unorm,
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: blade_graphics::TextureDimension::D2,
            usage: blade_graphics::TextureUsage::COPY | blade_graphics::TextureUsage::RESOURCE,
        });
        let red_view = gpu.create_texture_view(blade_graphics::TextureViewDesc {
            name: "dummy/red",
            texture: red_texture,
            format: blade_graphics::TextureFormat::Rgba8Unorm,
            dimension: blade_graphics::ViewDimension::D2,
            subresources: &blade_graphics::TextureSubresources::default(),
        });

        command_encoder.init_texture(white_texture);
        command_encoder.init_texture(black_texture);
        command_encoder.init_texture(red_texture);
        let mut transfers = command_encoder.transfer();
        let staging_buf = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "dummy/staging",
            size: 4 * 3,
            memory: blade_graphics::Memory::Upload,
        });
        unsafe {
            ptr::write(
                staging_buf.data() as *mut _,
                [!0u8, !0, !0, !0, 0, 0, 0, 0, !0, 0, 0, 0],
            );
        }
        transfers.copy_buffer_to_texture(staging_buf.at(0), 4, white_texture.into(), size);
        transfers.copy_buffer_to_texture(staging_buf.at(4), 4, black_texture.into(), size);
        transfers.copy_buffer_to_texture(staging_buf.at(8), 4, red_texture.into(), size);

        Self {
            size,
            white_texture,
            white_view,
            black_texture,
            black_view,
            red_texture,
            red_view,
            staging_buf,
        }
    }

    pub fn destroy(&mut self, gpu: &blade_graphics::Context) {
        gpu.destroy_texture_view(self.white_view);
        gpu.destroy_texture(self.white_texture);
        gpu.destroy_texture_view(self.black_view);
        gpu.destroy_texture(self.black_texture);
        gpu.destroy_texture_view(self.red_view);
        gpu.destroy_texture(self.red_texture);
        gpu.destroy_buffer(self.staging_buf);
    }
}
