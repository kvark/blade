use std::{fs, num::NonZeroU32};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct EnvPreprocParams {
    target_level: u32,
}

#[derive(blade_macros::ShaderData)]
struct EnvPreprocData {
    source: blade_graphics::TextureView,
    destination: blade_graphics::TextureView,
    params: EnvPreprocParams,
}

pub struct EnvironmentMap {
    pub main_view: blade_graphics::TextureView,
    pub size: blade_graphics::Extent,
    pub weight_texture: blade_graphics::Texture,
    pub weight_view: blade_graphics::TextureView,
    pub weight_mips: Vec<blade_graphics::TextureView>,
    pub preproc_pipeline: blade_graphics::ComputePipeline,
}

impl EnvironmentMap {
    pub fn init_pipeline(
        gpu: &blade_graphics::Context,
    ) -> Result<blade_graphics::ComputePipeline, &'static str> {
        let source = fs::read_to_string("blade-render/code/env-preproc.wgsl").unwrap();
        let shader = gpu.try_create_shader(blade_graphics::ShaderDesc { source: &source })?;
        let layout = <EnvPreprocData as blade_graphics::ShaderData>::layout();
        shader.check_struct_size::<EnvPreprocParams>();

        Ok(
            gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
                name: "env-preproc",
                data_layouts: &[&layout],
                compute: shader.at("downsample"),
            }),
        )
    }

    pub fn weight_size(&self) -> blade_graphics::Extent {
        // The weight texture has to include all of the edge pixels, starting at mip 1
        blade_graphics::Extent {
            width: self.size.width.next_power_of_two() / 2,
            height: self.size.height.next_power_of_two() / 2,
            depth: 1,
        }
    }

    pub fn destroy(&mut self, gpu: &blade_graphics::Context) {
        if self.weight_texture != blade_graphics::Texture::default() {
            gpu.destroy_texture(self.weight_texture);
            gpu.destroy_texture_view(self.weight_view);
        }
        for view in self.weight_mips.drain(..) {
            gpu.destroy_texture_view(view);
        }
    }

    pub fn assign(
        &mut self,
        view: blade_graphics::TextureView,
        extent: blade_graphics::Extent,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
    ) {
        if self.main_view == view {
            return;
        }
        self.main_view = view;
        self.size = extent;
        self.destroy(gpu);

        let mip_level_count = extent
            .width
            .max(extent.height)
            .next_power_of_two()
            .trailing_zeros();
        let weight_extent = self.weight_size();
        let format = blade_graphics::TextureFormat::Rgba16Float;
        self.weight_texture = gpu.create_texture(blade_graphics::TextureDesc {
            name: "env-weight",
            format,
            size: weight_extent,
            dimension: blade_graphics::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count,
            usage: blade_graphics::TextureUsage::RESOURCE | blade_graphics::TextureUsage::STORAGE,
        });
        self.weight_view = gpu.create_texture_view(blade_graphics::TextureViewDesc {
            name: "env-weight",
            texture: self.weight_texture,
            format,
            dimension: blade_graphics::ViewDimension::D2,
            subresources: &Default::default(),
        });
        for base_mip_level in 0..mip_level_count {
            let view = gpu.create_texture_view(blade_graphics::TextureViewDesc {
                name: &format!("env-weight-mip{}", base_mip_level),
                texture: self.weight_texture,
                format,
                dimension: blade_graphics::ViewDimension::D2,
                subresources: &blade_graphics::TextureSubresources {
                    base_mip_level,
                    mip_level_count: NonZeroU32::new(1),
                    ..Default::default()
                },
            });
            self.weight_mips.push(view);
        }

        encoder.init_texture(self.weight_texture);
        for target_level in 0..mip_level_count {
            let groups = self
                .preproc_pipeline
                .get_dispatch_for(weight_extent.at_mip_level(target_level));
            let mut compute = encoder.compute();
            let mut pass = compute.with(&self.preproc_pipeline);
            pass.bind(
                0,
                &EnvPreprocData {
                    source: if target_level == 0 {
                        view
                    } else {
                        self.weight_mips[target_level as usize - 1]
                    },
                    destination: self.weight_mips[target_level as usize],
                    params: EnvPreprocParams { target_level },
                },
            );
            pass.dispatch(groups);
        }
    }
}
