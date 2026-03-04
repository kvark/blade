use crate::{AssetHub, CameraParams, DummyResources, Object, Shaders, Vertex};
use blade_graphics as gpu;
use std::mem;

#[derive(Clone, Copy, Debug)]
pub struct RasterConfig {
    pub clear_color: gpu::TextureColor,
    pub light_dir: mint::Vector3<f32>,
    pub light_color: mint::Vector3<f32>,
    pub ambient_color: mint::Vector3<f32>,
    pub roughness: f32,
    pub metallic: f32,
}

impl Default for RasterConfig {
    fn default() -> Self {
        Self {
            clear_color: gpu::TextureColor::OpaqueBlack,
            light_dir: mint::Vector3 {
                x: -0.3,
                y: -1.0,
                z: -0.2,
            },
            light_color: mint::Vector3 {
                x: 3.0,
                y: 3.0,
                z: 3.0,
            },
            ambient_color: mint::Vector3 {
                x: 0.05,
                y: 0.05,
                z: 0.05,
            },
            roughness: 0.4,
            metallic: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct RasterFrameParams {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    ambient_color: [f32; 4],
    material: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct RasterDrawParams {
    model: [f32; 16],
    normal: [f32; 16],
    base_color_factor: [f32; 4],
    material: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct RasterFrameData {
    frame: RasterFrameParams,
    samp: gpu::Sampler,
}

#[derive(blade_macros::ShaderData)]
struct RasterDrawData {
    draw: RasterDrawParams,
    base_color_tex: gpu::TextureView,
    normal_tex: gpu::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct RasterSkyData {
    frame: RasterFrameParams,
    samp: gpu::Sampler,
    env_map: gpu::TextureView,
}

struct RasterPipelines {
    main: gpu::RenderPipeline,
    sky: gpu::RenderPipeline,
}

impl RasterPipelines {
    fn create_main(
        shader: &gpu::Shader,
        info: gpu::SurfaceInfo,
        gpu: &gpu::Context,
    ) -> gpu::RenderPipeline {
        shader.check_struct_size::<RasterFrameParams>();
        shader.check_struct_size::<RasterDrawParams>();
        let frame_layout = <RasterFrameData as gpu::ShaderData>::layout();
        let draw_layout = <RasterDrawData as gpu::ShaderData>::layout();
        let vertex_layout = <Vertex as gpu::Vertex>::layout();
        gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "raster",
            data_layouts: &[&frame_layout, &draw_layout],
            vertex: shader.at("raster_vs"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &vertex_layout,
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: gpu::CompareFunction::Less,
                stencil: gpu::StencilState::default(),
                bias: gpu::DepthBiasState::default(),
            }),
            fragment: Some(shader.at("raster_fs")),
            color_targets: &[info.format.into()],
            multisample_state: gpu::MultisampleState::default(),
        })
    }

    fn create_sky(
        shader: &gpu::Shader,
        info: gpu::SurfaceInfo,
        gpu: &gpu::Context,
    ) -> gpu::RenderPipeline {
        shader.check_struct_size::<RasterFrameParams>();
        let sky_layout = <RasterSkyData as gpu::ShaderData>::layout();
        gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "raster-sky",
            data_layouts: &[&sky_layout],
            vertex: shader.at("raster_sky_vs"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: gpu::CompareFunction::LessEqual,
                stencil: gpu::StencilState::default(),
                bias: gpu::DepthBiasState::default(),
            }),
            fragment: Some(shader.at("raster_sky_fs")),
            color_targets: &[info.format.into()],
            multisample_state: gpu::MultisampleState::default(),
        })
    }

    fn init(
        shaders: &Shaders,
        config: &crate::render::RenderConfig,
        gpu: &gpu::Context,
        shader_man: &blade_asset::AssetManager<crate::shader::Baker>,
    ) -> Result<Self, &'static str> {
        let shader = shader_man[shaders.raster].raw.as_ref().unwrap();
        Ok(Self {
            main: Self::create_main(shader, config.surface_info, gpu),
            sky: Self::create_sky(shader, config.surface_info, gpu),
        })
    }
}

pub struct Rasterizer {
    shaders: Shaders,
    pipelines: RasterPipelines,
    sampler_linear: gpu::Sampler,
    debug: crate::render::DebugRender,
    dummy: DummyResources,
    depth_texture: gpu::Texture,
    depth_view: gpu::TextureView,
    surface_size: gpu::Extent,
    surface_info: gpu::SurfaceInfo,
}

impl Rasterizer {
    #[profiling::function]
    pub fn new(
        encoder: &mut gpu::CommandEncoder,
        gpu: &gpu::Context,
        shaders: Shaders,
        shader_man: &blade_asset::AssetManager<crate::shader::Baker>,
        config: &crate::render::RenderConfig,
    ) -> Self {
        let pipelines = RasterPipelines::init(&shaders, config, gpu, shader_man).unwrap();
        let debug = {
            let sh_draw = shader_man[shaders.debug_draw].raw.as_ref().unwrap();
            let sh_blit = shader_man[shaders.debug_blit].raw.as_ref().unwrap();
            crate::render::DebugRender::init(
                encoder,
                gpu,
                sh_draw,
                sh_blit,
                config.max_debug_lines,
                config.surface_info,
            )
        };
        let dummy = DummyResources::new(encoder, gpu);
        let sampler_linear = gpu.create_sampler(gpu::SamplerDesc {
            name: "raster-linear",
            address_modes: [gpu::AddressMode::Repeat; 3],
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            mipmap_filter: gpu::FilterMode::Linear,
            ..Default::default()
        });
        let (depth_texture, depth_view) = Self::create_depth_target(config.surface_size, gpu);

        Self {
            shaders,
            pipelines,
            sampler_linear,
            debug,
            dummy,
            depth_texture,
            depth_view,
            surface_size: config.surface_size,
            surface_info: config.surface_info,
        }
    }

    pub fn destroy(&mut self, gpu: &gpu::Context) {
        self.debug.destroy(gpu);
        self.dummy.destroy(gpu);
        gpu.destroy_texture_view(self.depth_view);
        gpu.destroy_texture(self.depth_texture);
        gpu.destroy_sampler(self.sampler_linear);
        gpu.destroy_render_pipeline(&mut self.pipelines.main);
        gpu.destroy_render_pipeline(&mut self.pipelines.sky);
    }

    #[profiling::function]
    pub fn hot_reload(
        &mut self,
        asset_hub: &AssetHub,
        gpu: &gpu::Context,
        sync_point: &gpu::SyncPoint,
    ) -> bool {
        let mut tasks = Vec::new();
        let old = self.shaders.clone();

        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.raster));

        if tasks.is_empty() {
            return false;
        }

        log::info!("Hot reloading raster shaders");
        gpu.wait_for(sync_point, !0);
        for task in tasks {
            let _ = task.join();
        }

        if self.shaders.raster != old.raster {
            if let Ok(ref shader) = asset_hub.shaders[self.shaders.raster].raw {
                self.pipelines.main = RasterPipelines::create_main(shader, self.surface_info, gpu);
                self.pipelines.sky = RasterPipelines::create_sky(shader, self.surface_info, gpu);
            }
        }

        true
    }

    pub fn get_surface_size(&self) -> gpu::Extent {
        self.surface_size
    }

    pub fn depth_view(&self) -> gpu::TextureView {
        self.depth_view
    }

    pub fn depth_texture(&self) -> gpu::Texture {
        self.depth_texture
    }

    pub fn resize_screen(
        &mut self,
        size: gpu::Extent,
        _encoder: &mut gpu::CommandEncoder,
        gpu: &gpu::Context,
    ) {
        if size == self.surface_size {
            return;
        }
        gpu.destroy_texture_view(self.depth_view);
        gpu.destroy_texture(self.depth_texture);
        let (depth_texture, depth_view) = Self::create_depth_target(size, gpu);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.surface_size = size;
    }

    #[profiling::function]
    pub fn render(
        &mut self,
        pass: &mut gpu::RenderCommandEncoder,
        camera: &crate::Camera,
        objects: &[Object],
        asset_hub: &AssetHub,
        environment_map: Option<blade_asset::Handle<crate::Texture>>,
        config: RasterConfig,
    ) {
        let env_map_enabled = environment_map.is_some();
        let frame_params = self.make_frame_params(camera, config, env_map_enabled);
        if let mut pc = pass.with(&self.pipelines.main) {
            pc.bind(
                0,
                &RasterFrameData {
                    frame: frame_params,
                    samp: self.sampler_linear,
                },
            );

            for object in objects.iter() {
                let model = &asset_hub.models[object.model];
                let object_transform = mat4_transform(&object.transform);
                let object_normal = object_transform.inverse().transpose();

                pc.bind_vertex(0, model.vertex_buffer.at(0));

                for geometry in model.geometries.iter() {
                    let geometry_transform = mat4_transform(&geometry.transform);
                    let world_transform = object_transform * geometry_transform;
                    let normal_transform = object_normal * geometry_transform.inverse().transpose();
                    let material = &model.materials[geometry.material_index];

                    let (normal_tex, normal_scale) = match material.normal_texture {
                        Some(handle) => {
                            let texture = &asset_hub.textures[handle];
                            (texture.view, material.normal_scale)
                        }
                        None => (self.dummy.white_view, 0.0),
                    };
                    let base_color_tex = match material.base_color_texture {
                        Some(handle) => asset_hub.textures[handle].view,
                        None => self.dummy.white_view,
                    };

                    pc.bind(
                        1,
                        &RasterDrawData {
                            draw: RasterDrawParams {
                                model: world_transform.to_cols_array(),
                                normal: normal_transform.to_cols_array(),
                                base_color_factor: material.base_color_factor,
                                material: [normal_scale, 0.0, 0.0, 0.0],
                            },
                            base_color_tex,
                            normal_tex,
                        },
                    );

                    let vertex_count = geometry.vertex_range.end - geometry.vertex_range.start;
                    let index_count = geometry.triangle_count * 3;
                    match geometry.index_type {
                        Some(index_type) => {
                            pc.draw_indexed(
                                model.index_buffer.at(geometry.index_offset),
                                index_type,
                                index_count,
                                geometry.vertex_range.start as i32,
                                0,
                                1,
                            );
                        }
                        None => {
                            pc.draw(geometry.vertex_range.start, vertex_count, 0, 1);
                        }
                    }
                }
            }
        }

        let env_map = environment_map
            .map(|handle| asset_hub.textures[handle].view)
            .unwrap_or(self.dummy.black_view);
        self.render_sky(pass, frame_params, env_map);
    }

    pub fn render_debug_lines(
        &self,
        pass: &mut gpu::RenderCommandEncoder,
        camera: &crate::Camera,
        debug_lines: &[crate::DebugLine],
    ) {
        if debug_lines.is_empty() {
            return;
        }
        let camera_params = self.make_camera_params(camera);
        self.debug
            .render_lines(debug_lines, camera_params, self.depth_view, pass);
    }

    fn render_sky(
        &self,
        pass: &mut gpu::RenderCommandEncoder,
        frame_params: RasterFrameParams,
        env_map: gpu::TextureView,
    ) {
        let mut pc = pass.with(&self.pipelines.sky);
        pc.bind(
            0,
            &RasterSkyData {
                frame: frame_params,
                samp: self.sampler_linear,
                env_map,
            },
        );
        pc.draw(0, 3, 0, 1);
    }

    fn create_depth_target(
        size: gpu::Extent,
        gpu: &gpu::Context,
    ) -> (gpu::Texture, gpu::TextureView) {
        let texture = gpu.create_texture(gpu::TextureDesc {
            name: "raster depth",
            size,
            format: gpu::TextureFormat::Depth32Float,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            external: None,
        });
        let view = gpu.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "raster depth",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );
        (texture, view)
    }

    fn make_camera_params(&self, camera: &crate::Camera) -> CameraParams {
        let fov_x = 2.0
            * ((camera.fov_y * 0.5).tan() * self.surface_size.width as f32
                / self.surface_size.height as f32)
                .atan();
        CameraParams {
            position: camera.pos.into(),
            depth: camera.depth,
            orientation: camera.rot.into(),
            fov: [fov_x, camera.fov_y],
            target_size: [self.surface_size.width, self.surface_size.height],
        }
    }

    fn make_frame_params(
        &self,
        camera: &crate::Camera,
        config: RasterConfig,
        env_map_enabled: bool,
    ) -> RasterFrameParams {
        let pos = glam::Vec3::from(camera.pos);
        let rot = glam::Quat::from(camera.rot);
        let view = glam::Mat4::from_rotation_translation(rot, pos).inverse();
        let aspect = self.surface_size.width as f32 / self.surface_size.height.max(1) as f32;
        let proj = glam::Mat4::perspective_rh(camera.fov_y, aspect, 0.01, camera.depth);
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();
        let light_dir = glam::Vec3::from(config.light_dir).normalize_or_zero();
        RasterFrameParams {
            view_proj: view_proj.to_cols_array(),
            inv_view_proj: inv_view_proj.to_cols_array(),
            camera_pos: [pos.x, pos.y, pos.z, 1.0],
            light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
            light_color: {
                let c = config.light_color;
                [c.x, c.y, c.z, 0.0]
            },
            ambient_color: {
                let c = config.ambient_color;
                [c.x, c.y, c.z, 0.0]
            },
            material: [
                config.roughness,
                config.metallic,
                env_map_enabled as u32 as f32,
                0.0,
            ],
        }
    }
}

impl gpu::Vertex for Vertex {
    fn layout() -> gpu::VertexLayout {
        gpu::VertexLayout {
            attributes: vec![
                (
                    "position",
                    gpu::VertexAttribute {
                        offset: 0,
                        format: gpu::VertexFormat::F32Vec3,
                    },
                ),
                (
                    "bitangent_sign",
                    gpu::VertexAttribute {
                        offset: 12,
                        format: gpu::VertexFormat::F32,
                    },
                ),
                (
                    "tex_coords",
                    gpu::VertexAttribute {
                        offset: 16,
                        format: gpu::VertexFormat::F32Vec2,
                    },
                ),
                (
                    "normal",
                    gpu::VertexAttribute {
                        offset: 24,
                        format: gpu::VertexFormat::U32,
                    },
                ),
                (
                    "tangent",
                    gpu::VertexAttribute {
                        offset: 28,
                        format: gpu::VertexFormat::U32,
                    },
                ),
            ],
            stride: mem::size_of::<Vertex>() as u32,
        }
    }
}

fn mat4_transform(t: &gpu::Transform) -> glam::Mat4 {
    glam::Mat4 {
        x_axis: t.x.into(),
        y_axis: t.y.into(),
        z_axis: t.z.into(),
        w_axis: glam::Vec4::W,
    }
    .transpose()
}
