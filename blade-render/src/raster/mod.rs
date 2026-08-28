use crate::{
    AssetHub, CameraParams, DummyResources, FrameResources, Object, RenderConfig, Shaders, Vertex,
    skin::{self, SkinPass, SkinningParams},
};
use blade_graphics as gpu;
use std::mem;

fn geometry_matrix(
    model: &crate::Model,
    geometry: &crate::model::Geometry,
    pose: Option<&crate::Pose>,
) -> glam::Mat4 {
    mat4_transform(&model.geometry_transform(geometry, pose))
}

/// Maximum local lights the forward pass considers per fragment.
/// Extra lights in `RasterConfig::point_lights` are ignored.
pub const MAX_POINT_LIGHTS: usize = 8;

/// A local omni light. Radius is a hard cutoff in world units.
#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: mint::Vector3<f32>,
    pub color: mint::Vector3<f32>,
    pub radius: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: mint::Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            color: mint::Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            radius: 1.0,
        }
    }
}

/// Configuration of the rasterized frame.
///
/// Note: the surface appearance is described by the materials of the
/// models, this is only about the scene-wide lighting.
#[derive(Clone, Debug)]
pub struct RasterConfig {
    pub clear_color: gpu::TextureColor,
    /// Direction *towards* the single directional light.
    pub light_dir: mint::Vector3<f32>,
    pub light_color: mint::Vector3<f32>,
    pub ambient_color: mint::Vector3<f32>,
    /// When true, the sky fallback renders stars instead of a blue gradient.
    pub space_sky: bool,
    /// Optional real-time directional shadow-map effect.
    pub directional_shadows: Option<DirectionalShadowConfig>,
    /// Local omni lights. The rasterizer uploads at most `MAX_POINT_LIGHTS`.
    pub point_lights: Vec<PointLight>,
}

/// Controls the rasterizer's camera-relative directional shadow map.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalShadowConfig {
    /// Width and height of the square shadow map.
    pub resolution: u32,
    /// Half-width of the shadowed world-space region around the camera.
    pub distance: f32,
    /// Total depth captured along the light direction.
    pub depth: f32,
    /// Fraction of direct lighting removed in full shadow.
    pub strength: f32,
    /// World-space receiver offset along its shading normal.
    pub normal_bias: f32,
}

impl Default for DirectionalShadowConfig {
    fn default() -> Self {
        Self {
            resolution: 512,
            distance: 70.0,
            depth: 240.0,
            strength: 0.88,
            normal_bias: 0.06,
        }
    }
}

impl Default for RasterConfig {
    fn default() -> Self {
        Self {
            clear_color: gpu::TextureColor::OpaqueBlack,
            light_dir: mint::Vector3 {
                x: 0.3,
                y: 1.0,
                z: 0.2,
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
            space_sky: false,
            directional_shadows: None,
            point_lights: Vec::new(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct RasterFrameParams {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    light_view_proj: [f32; 16],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    ambient_color: [f32; 4],
    settings: [f32; 4],
    shadow_params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PointLightGpu {
    pos_radius: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PointLightParams {
    count_seed: [f32; 4],
    lights: [PointLightGpu; MAX_POINT_LIGHTS],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct RasterDrawParams {
    model: [f32; 16],
    /// Rotation of the object/geometry transform. Skinning assumes uniform
    /// scale, so a quaternion is sufficient for normals: models with
    /// non-uniform scale log a warning at load time and their normals may
    /// end up slightly skewed.
    normal_quat: [f32; 4],
    base_color_factor: [f32; 4],
    emissive_factor: [f32; 4],
    material: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct ShadowFrameParams {
    light_view_proj: [f32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct ShadowDrawParams {
    model: [f32; 16],
}

#[derive(blade_macros::ShaderData)]
struct RasterMainData {
    frame_params: RasterFrameParams,
    light_params: PointLightParams,
    draw_params: RasterDrawParams,
    samp: gpu::Sampler,
    base_color_tex: gpu::TextureView,
    normal_tex: gpu::TextureView,
    metallic_roughness_tex: gpu::TextureView,
    emissive_tex: gpu::TextureView,
    shadow_samp: gpu::Sampler,
    shadow_tex: gpu::TextureView,
}

// The skinned extension is bound to slot 1, so the rigid pipelines keep
// the same data layout and don't pay for the 3KB joint palette upload.
#[derive(blade_macros::ShaderData)]
struct RasterSkinData {
    skinning_params: SkinningParams,
}

#[derive(blade_macros::ShaderData)]
struct RasterShadowData {
    shadow_frame_params: ShadowFrameParams,
    shadow_draw_params: ShadowDrawParams,
}

#[derive(blade_macros::ShaderData)]
struct RasterSkyData {
    sky_params: RasterFrameParams,
    samp: gpu::Sampler,
    env_map: gpu::TextureView,
}

#[derive(Clone, Copy)]
enum Variant {
    Rigid,
    Skinned,
}

struct RasterPipelines {
    main: gpu::RenderPipeline,
    main_skinned: gpu::RenderPipeline,
    sky: gpu::RenderPipeline,
    shadow: gpu::RenderPipeline,
    shadow_skinned: gpu::RenderPipeline,
}

impl RasterPipelines {
    fn create_main(
        shader: &gpu::Shader,
        info: gpu::SurfaceInfo,
        gpu: &gpu::Context,
        variant: Variant,
    ) -> gpu::RenderPipeline {
        shader.check_struct_size::<RasterFrameParams>();
        shader.check_struct_size::<PointLightParams>();
        shader.check_struct_size::<RasterDrawParams>();
        shader.check_struct_size::<SkinningParams>();
        let main_layout = <RasterMainData as gpu::ShaderData>::layout();
        let skin_layout = <RasterSkinData as gpu::ShaderData>::layout();
        let vertex_layout = <Vertex as gpu::Vertex>::layout();
        let skin_vertex_layout = <crate::SkinVertex as gpu::Vertex>::layout();
        let (name, vertex_entry) = match variant {
            Variant::Rigid => ("raster", "raster_vs"),
            Variant::Skinned => ("raster-skinned", "raster_skinned_vs"),
        };
        let (data_layouts, vertex_fetches) = match variant {
            Variant::Rigid => (
                vec![&main_layout],
                vec![gpu::VertexFetchState {
                    layout: &vertex_layout,
                    instanced: false,
                }],
            ),
            Variant::Skinned => (
                vec![&main_layout, &skin_layout],
                vec![
                    gpu::VertexFetchState {
                        layout: &vertex_layout,
                        instanced: false,
                    },
                    gpu::VertexFetchState {
                        layout: &skin_vertex_layout,
                        instanced: false,
                    },
                ],
            ),
        };
        gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name,
            data_layouts: &data_layouts,
            vertex: shader.at(vertex_entry),
            vertex_fetches: &vertex_fetches,
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

    fn create_shadow(
        shader: &gpu::Shader,
        gpu: &gpu::Context,
        variant: Variant,
    ) -> gpu::RenderPipeline {
        shader.check_struct_size::<ShadowFrameParams>();
        shader.check_struct_size::<ShadowDrawParams>();
        shader.check_struct_size::<SkinningParams>();
        let shadow_layout = <RasterShadowData as gpu::ShaderData>::layout();
        let skin_layout = <RasterSkinData as gpu::ShaderData>::layout();
        let vertex_layout = <Vertex as gpu::Vertex>::layout();
        let skin_vertex_layout = <crate::SkinVertex as gpu::Vertex>::layout();
        let (name, vertex_entry) = match variant {
            Variant::Rigid => ("raster-shadow", "raster_shadow_vs"),
            Variant::Skinned => ("raster-shadow-skinned", "raster_shadow_skinned_vs"),
        };
        let (data_layouts, vertex_fetches) = match variant {
            Variant::Rigid => (
                vec![&shadow_layout],
                vec![gpu::VertexFetchState {
                    layout: &vertex_layout,
                    instanced: false,
                }],
            ),
            Variant::Skinned => (
                vec![&shadow_layout, &skin_layout],
                vec![
                    gpu::VertexFetchState {
                        layout: &vertex_layout,
                        instanced: false,
                    },
                    gpu::VertexFetchState {
                        layout: &skin_vertex_layout,
                        instanced: false,
                    },
                ],
            ),
        };
        gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name,
            data_layouts: &data_layouts,
            vertex: shader.at(vertex_entry),
            vertex_fetches: &vertex_fetches,
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: gpu::CompareFunction::Less,
                stencil: gpu::StencilState::default(),
                bias: gpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            // WebGL2/GLES cannot link a vertex-only program. An empty
            // fragment stage is valid on Vulkan/Metal too.
            fragment: Some(shader.at("raster_shadow_fs")),
            color_targets: &[],
            multisample_state: gpu::MultisampleState::default(),
        })
    }

    fn init(
        shaders: &Shaders,
        config: &RenderConfig,
        gpu: &gpu::Context,
        shader_man: &blade_asset::AssetManager<crate::shader::Baker>,
    ) -> Result<Self, &'static str> {
        let shader = shader_man[shaders.raster].raw.as_ref().unwrap();
        Ok(Self {
            main: Self::create_main(shader, config.surface_info, gpu, Variant::Rigid),
            main_skinned: Self::create_main(shader, config.surface_info, gpu, Variant::Skinned),
            sky: Self::create_sky(shader, config.surface_info, gpu),
            shadow: Self::create_shadow(shader, gpu, Variant::Rigid),
            shadow_skinned: Self::create_shadow(shader, gpu, Variant::Skinned),
        })
    }
}

struct SkinnedVertexSlot {
    vertex_buffer: gpu::Buffer,
    vertex_count: usize,
}

impl SkinnedVertexSlot {
    fn destroy(self, gpu: &gpu::Context) {
        gpu.destroy_buffer(self.vertex_buffer);
    }

    fn retire(self, temp: &mut crate::FrameResources) {
        temp.buffers.push(self.vertex_buffer);
    }
}

/// Per-object GPU storage for compute-skinned raster vertices.
///
/// Persist this with the [`crate::Object`] across frames.
///
/// Like the ray-tracing path, this relies on the caller running at most two
/// frames in flight. Three generations are kept so that a single redundant
/// [`Rasterizer::prepare`] call within a frame still writes into a retired
/// buffer instead of one the GPU may be reading.
#[derive(Default)]
pub struct ObjectSkin {
    generations: crate::util::RollingBuffer<SkinnedVertexSlot, 3>,
}

impl ObjectSkin {
    /// Release the vertex buffers after prior GPU work has completed.
    pub fn destroy(&mut self, gpu: &gpu::Context) {
        for slot in self.generations.drain() {
            slot.destroy(gpu);
        }
    }

    pub(crate) fn retire(&mut self, temp: &mut crate::FrameResources) {
        for slot in self.generations.drain() {
            slot.retire(temp);
        }
    }

    fn prepared(&self) -> Option<gpu::Buffer> {
        self.generations.last().map(|slot| slot.vertex_buffer)
    }
}

pub struct Rasterizer {
    shaders: Shaders,
    pipelines: RasterPipelines,
    /// Native compute skin; `None` on GLES, which keeps the vertex-stage path.
    skin: Option<SkinPass>,
    sampler_linear: gpu::Sampler,
    sampler_shadow: gpu::Sampler,
    debug: Option<crate::render::DebugRender>,
    dummy: DummyResources,
    depth_texture: gpu::Texture,
    depth_view: gpu::TextureView,
    shadow_texture: gpu::Texture,
    shadow_view: gpu::TextureView,
    shadow_size: u32,
    surface_size: gpu::Extent,
    surface_info: gpu::SurfaceInfo,
    color_space: gpu::ColorSpace,
}

impl Rasterizer {
    #[profiling::function]
    pub fn new(
        encoder: &mut gpu::CommandEncoder,
        gpu: &gpu::Context,
        shaders: Shaders,
        shader_man: &blade_asset::AssetManager<crate::shader::Baker>,
        config: &RenderConfig,
    ) -> Self {
        let pipelines = RasterPipelines::init(&shaders, config, gpu, shader_man).unwrap();
        let skin = if cfg!(any(gles, target_arch = "wasm32")) {
            None
        } else {
            shader_man[shaders.skin]
                .raw
                .as_ref()
                .ok()
                .and_then(|shader| SkinPass::new(shader, gpu))
        };
        let capabilities = gpu.capabilities();
        let debug = if cfg!(target_os = "android")
            || !capabilities.compute
            || !capabilities.indirect_draw
        {
            None
        } else {
            let sh_draw = shader_man[shaders.debug_draw].raw.as_ref().unwrap();
            let sh_blit = shader_man[shaders.debug_blit].raw.as_ref().unwrap();
            Some(crate::render::DebugRender::init(
                encoder,
                gpu,
                sh_draw,
                sh_blit,
                config.max_debug_lines,
                config.surface_info,
            ))
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
        let sampler_shadow = gpu.create_sampler(gpu::SamplerDesc {
            name: "raster-shadow",
            address_modes: [gpu::AddressMode::ClampToEdge; 3],
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            mipmap_filter: gpu::FilterMode::Nearest,
            compare: Some(gpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let (depth_texture, depth_view) = Self::create_depth_target(config.surface_size, gpu);
        // The engine can safely replace this default when raster configuration changes.
        let shadow_size = DirectionalShadowConfig::default().resolution;
        let (shadow_texture, shadow_view) = Self::create_shadow_target(shadow_size, gpu);
        Self {
            shaders,
            pipelines,
            skin,
            sampler_linear,
            sampler_shadow,
            debug,
            dummy,
            depth_texture,
            depth_view,
            shadow_texture,
            shadow_view,
            shadow_size,
            surface_size: config.surface_size,
            surface_info: config.surface_info,
            color_space: config.color_space,
        }
    }

    pub fn destroy(&mut self, gpu: &gpu::Context) {
        if let Some(debug) = self.debug.as_mut() {
            debug.destroy(gpu);
        }
        self.dummy.destroy(gpu);
        gpu.destroy_texture_view(self.depth_view);
        gpu.destroy_texture(self.depth_texture);
        gpu.destroy_texture_view(self.shadow_view);
        gpu.destroy_texture(self.shadow_texture);
        gpu.destroy_sampler(self.sampler_linear);
        gpu.destroy_sampler(self.sampler_shadow);
        gpu.destroy_render_pipeline(&mut self.pipelines.main);
        gpu.destroy_render_pipeline(&mut self.pipelines.main_skinned);
        gpu.destroy_render_pipeline(&mut self.pipelines.sky);
        gpu.destroy_render_pipeline(&mut self.pipelines.shadow);
        gpu.destroy_render_pipeline(&mut self.pipelines.shadow_skinned);
        if let Some(skin) = self.skin.as_mut() {
            skin.destroy(gpu);
        }
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
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.skin));

        if tasks.is_empty() {
            return false;
        }

        log::info!("Hot reloading raster shaders");
        let _ = gpu.wait_for(sync_point, !0);
        for task in tasks {
            let _ = task.join();
        }

        if self.shaders.raster != old.raster
            && let Ok(ref shader) = asset_hub.shaders[self.shaders.raster].raw
        {
            self.pipelines.main =
                RasterPipelines::create_main(shader, self.surface_info, gpu, Variant::Rigid);
            self.pipelines.main_skinned =
                RasterPipelines::create_main(shader, self.surface_info, gpu, Variant::Skinned);
            self.pipelines.sky = RasterPipelines::create_sky(shader, self.surface_info, gpu);
            self.pipelines.shadow = RasterPipelines::create_shadow(shader, gpu, Variant::Rigid);
            self.pipelines.shadow_skinned =
                RasterPipelines::create_shadow(shader, gpu, Variant::Skinned);
        }
        if self.shaders.skin != old.skin
            && let (Some(skin), Ok(shader)) = (
                self.skin.as_mut(),
                asset_hub.shaders[self.shaders.skin].raw.as_ref(),
            )
        {
            skin.recreate(shader, gpu);
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

    pub fn directional_shadow_resolution(&self) -> u32 {
        self.shadow_size
    }

    /// Resize the shadow map. The caller must ensure earlier GPU work using the
    /// map has completed; `blade-engine::Engine::set_raster_config` does this.
    pub fn set_directional_shadow_resolution(&mut self, size: u32, gpu: &gpu::Context) {
        let size = size.clamp(256, 4096);
        if size == self.shadow_size {
            return;
        }
        gpu.destroy_texture_view(self.shadow_view);
        gpu.destroy_texture(self.shadow_texture);
        (self.shadow_texture, self.shadow_view) = Self::create_shadow_target(size, gpu);
        self.shadow_size = size;
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

    /// Skin meshes into vertex buffers reused by shadows and the main pass.
    ///
    /// No-op on GLES, which keeps vertex-stage skinning. Call this exactly once
    /// per frame on the command encoder before [`Self::render_directional_shadows`]
    /// and [`Self::render`]. Replaced buffers are retired through `temp` and
    /// released when the frame's fence signals.
    #[profiling::function]
    pub fn prepare(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        objects: &mut [Object],
        asset_hub: &AssetHub,
        gpu: &gpu::Context,
        temp: &mut FrameResources,
    ) {
        if self.skin.is_none() {
            return;
        }

        let mut skin_jobs = Vec::new();
        let mut copies = Vec::new();
        for object in objects.iter_mut() {
            let model = &asset_hub.models[object.model];
            if !skin::model_needs_vertex_skin(model) {
                // The model no longer needs skinning: drop any buffers prepared
                // for it so draws fall back to the model's own vertex data.
                object.skin.retire(temp);
                continue;
            }
            let vertex_count = model.vertex_count();
            let vertex_size = (vertex_count * mem::size_of::<Vertex>()) as u64;
            let slot = object.skin.generations.next();
            let reuse = slot
                .as_ref()
                .is_some_and(|slot| slot.vertex_count == vertex_count);
            if !reuse {
                if let Some(old) = slot.take() {
                    old.retire(temp);
                }
                *slot = Some(SkinnedVertexSlot {
                    vertex_buffer: gpu.create_buffer(gpu::BufferDesc {
                        name: "skinned vertices",
                        size: vertex_size.max(1),
                        memory: gpu::Memory::Device,
                        transient: false,
                    }),
                    vertex_count,
                });
            }
            let vertex_buffer = slot.as_ref().unwrap().vertex_buffer;
            skin::queue_model(
                model,
                object.pose.as_ref(),
                None,
                vertex_buffer,
                0,
                &mut skin_jobs,
                &mut copies,
            );
        }
        skin::encode(encoder, self.skin.as_ref(), &copies, &skin_jobs);
    }

    fn object_vertices(object: &Object, model: &crate::Model) -> (gpu::Buffer, bool) {
        match object.skin.prepared() {
            Some(buffer) => (buffer, true),
            None => (model.vertex_buffer, false),
        }
    }

    /// Render the directional-light depth prepass used by the main raster pass.
    ///
    /// Keeping this as an explicit encoder-level operation lets `blade-engine`
    /// schedule additional effects before it opens the final color render pass.
    #[profiling::function]
    pub fn render_directional_shadows(
        &self,
        encoder: &mut gpu::CommandEncoder,
        camera: &crate::Camera,
        objects: &[Object],
        asset_hub: &AssetHub,
        config: &RasterConfig,
    ) {
        let Some(shadow_config) = config.directional_shadows else {
            return;
        };
        let light_view_proj = make_light_view_proj(camera, config.light_dir, shadow_config);
        encoder.init_texture(self.shadow_texture);
        if let mut pass = encoder.render(
            "directional shadows",
            gpu::RenderTargetSet {
                colors: &[],
                depth_stencil: Some(gpu::RenderTarget {
                    view: self.shadow_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                    finish_op: gpu::FinishOp::Store,
                }),
            },
        ) {
            for object in objects {
                let model = &asset_hub.models[object.model];
                let object_transform = mat4_transform(&object.transform);
                let pose = object.pose.as_ref();
                let (vertex_buffer, compute_skinned) = Self::object_vertices(object, model);
                for geometry in &model.geometries {
                    let skinned = geometry.skin_index.is_some() && !compute_skinned;
                    let pipeline = if skinned {
                        &self.pipelines.shadow_skinned
                    } else {
                        &self.pipelines.shadow
                    };
                    let mut pc = pass.with(pipeline);
                    let geometry_transform = geometry_matrix(model, geometry, pose);
                    let world_transform = object_transform * geometry_transform;
                    pc.bind(
                        0,
                        &RasterShadowData {
                            shadow_frame_params: ShadowFrameParams {
                                light_view_proj: light_view_proj.to_cols_array(),
                            },
                            shadow_draw_params: ShadowDrawParams {
                                model: world_transform.to_cols_array(),
                            },
                        },
                    );
                    if skinned {
                        pc.bind(
                            1,
                            &RasterSkinData {
                                skinning_params: skin::make_skinning_params(model, geometry, pose),
                            },
                        );
                    }
                    let vertex_count = geometry.vertex_range.end - geometry.vertex_range.start;
                    let index_count = geometry.triangle_count * 3;
                    let vertex_offset =
                        geometry.vertex_range.start as u64 * mem::size_of::<Vertex>() as u64;
                    pc.bind_vertex(0, vertex_buffer.at(vertex_offset));
                    if skinned {
                        pc.bind_vertex(
                            1,
                            model
                                .skin_vertex_buffer
                                .at(geometry.vertex_range.start as u64
                                    * mem::size_of::<crate::SkinVertex>() as u64),
                        );
                    }
                    match geometry.index_type {
                        Some(index_type) => pc.draw_indexed(
                            model.index_buffer.at(geometry.index_offset),
                            index_type,
                            index_count,
                            0,
                            0,
                            1,
                        ),
                        None => pc.draw(0, vertex_count, 0, 1),
                    }
                }
            }
        }
    }

    #[profiling::function]
    pub fn render(
        &mut self,
        pass: &mut gpu::RenderCommandEncoder,
        camera: &crate::Camera,
        objects: &[Object],
        asset_hub: &AssetHub,
        environment_map: Option<blade_asset::Handle<crate::Texture>>,
        config: &RasterConfig,
    ) {
        let env_map_enabled = environment_map.is_some();
        let frame_params = self.make_frame_params(camera, config, env_map_enabled);
        let light_params = pack_point_lights(config, camera);
        {
            for object in objects {
                let model = &asset_hub.models[object.model];
                let object_transform = mat4_transform(&object.transform);
                let object_normal = object_transform.inverse().transpose();
                let pose = object.pose.as_ref();
                let (vertex_buffer, compute_skinned) = Self::object_vertices(object, model);

                for geometry in model.geometries.iter() {
                    let skinned = geometry.skin_index.is_some() && !compute_skinned;
                    let pipeline = if skinned {
                        &self.pipelines.main_skinned
                    } else {
                        &self.pipelines.main
                    };
                    let mut pc = pass.with(pipeline);
                    let geometry_transform = geometry_matrix(model, geometry, pose);
                    let world_transform = object_transform * geometry_transform;
                    let normal_transform = object_normal * geometry_transform.inverse().transpose();
                    // Treat the linear part as a similarity: the rotation is
                    // recovered by normalizing the columns of the inverse
                    // transpose. Non-uniform scale is assumed away (warned
                    // about at asset load) since it's a rare edge case.
                    let normal_basis = glam::Mat3::from_cols(
                        normal_transform.x_axis.truncate().normalize_or_zero(),
                        normal_transform.y_axis.truncate().normalize_or_zero(),
                        normal_transform.z_axis.truncate().normalize_or_zero(),
                    );
                    let normal_quat = glam::Quat::from_mat3(&normal_basis).normalize();
                    let material = &model.materials[geometry.material_index];

                    let (normal_tex, normal_scale) = match material.normal_texture {
                        Some(handle) => {
                            let texture = &asset_hub.textures[handle];
                            (texture.view, material.normal_scale)
                        }
                        None => (self.dummy.white_view, 0.0),
                    };
                    //Note: the dummies are white, so that the factors are unaffected
                    let texture_or_white =
                        |handle: Option<blade_asset::Handle<crate::Texture>>| match handle {
                            Some(handle) => asset_hub.textures[handle].view,
                            None => self.dummy.white_view,
                        };
                    let draw_params = RasterDrawParams {
                        model: world_transform.to_cols_array(),
                        normal_quat: normal_quat.to_array(),
                        base_color_factor: [
                            material.base_color_factor[0] * object.color_tint[0],
                            material.base_color_factor[1] * object.color_tint[1],
                            material.base_color_factor[2] * object.color_tint[2],
                            material.base_color_factor[3] * object.color_tint[3],
                        ],
                        emissive_factor: [
                            material.emissive_factor[0],
                            material.emissive_factor[1],
                            material.emissive_factor[2],
                            0.0,
                        ],
                        material: [normal_scale, material.metalness, material.roughness, 0.0],
                    };
                    pc.bind(
                        0,
                        &RasterMainData {
                            frame_params,
                            light_params,
                            draw_params,
                            samp: self.sampler_linear,
                            base_color_tex: texture_or_white(material.base_color_texture),
                            normal_tex,
                            metallic_roughness_tex: texture_or_white(
                                material.metallic_roughness_texture,
                            ),
                            emissive_tex: texture_or_white(material.emissive_texture),
                            shadow_samp: self.sampler_shadow,
                            shadow_tex: self.shadow_view,
                        },
                    );
                    if skinned {
                        pc.bind(
                            1,
                            &RasterSkinData {
                                skinning_params: skin::make_skinning_params(model, geometry, pose),
                            },
                        );
                    }
                    let vertex_count = geometry.vertex_range.end - geometry.vertex_range.start;
                    let index_count = geometry.triangle_count * 3;
                    let vertex_offset =
                        geometry.vertex_range.start as u64 * mem::size_of::<Vertex>() as u64;
                    pc.bind_vertex(0, vertex_buffer.at(vertex_offset));
                    if skinned {
                        pc.bind_vertex(
                            1,
                            model
                                .skin_vertex_buffer
                                .at(geometry.vertex_range.start as u64
                                    * mem::size_of::<crate::SkinVertex>() as u64),
                        );
                    }
                    match geometry.index_type {
                        Some(index_type) => pc.draw_indexed(
                            model.index_buffer.at(geometry.index_offset),
                            index_type,
                            index_count,
                            0,
                            0,
                            1,
                        ),
                        None => pc.draw(0, vertex_count, 0, 1),
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
        let Some(debug) = self.debug.as_ref() else {
            return;
        };
        if debug_lines.is_empty() {
            return;
        }
        let camera_params = self.make_camera_params(camera);
        debug.render_lines(debug_lines, camera_params, self.depth_view, pass);
    }

    /// Render just the sky background (env map or procedural fallback).
    pub fn render_sky_only(
        &self,
        pass: &mut gpu::RenderCommandEncoder,
        camera: &crate::Camera,
        environment_map: Option<blade_asset::Handle<crate::Texture>>,
        asset_hub: &AssetHub,
        config: &RasterConfig,
    ) {
        let env_map_enabled = environment_map.is_some();
        let frame_params = self.make_frame_params(camera, config, env_map_enabled);
        let env_map = environment_map
            .map(|handle| asset_hub.textures[handle].view)
            .unwrap_or(self.dummy.black_view);
        self.render_sky(pass, frame_params, env_map);
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
                sky_params: frame_params,
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

    fn create_shadow_target(size: u32, gpu: &gpu::Context) -> (gpu::Texture, gpu::TextureView) {
        let texture = gpu.create_texture(gpu::TextureDesc {
            name: "directional shadow map",
            size: gpu::Extent {
                width: size,
                height: size,
                depth: 1,
            },
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
                name: "directional shadow map",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );
        (texture, view)
    }

    fn make_camera_params(&self, camera: &crate::Camera) -> CameraParams {
        CameraParams::new(camera, [self.surface_size.width, self.surface_size.height])
    }

    fn make_frame_params(
        &self,
        camera: &crate::Camera,
        config: &RasterConfig,
        env_map_enabled: bool,
    ) -> RasterFrameParams {
        let pos = glam::Vec3::from(camera.pos);
        let rot = glam::Quat::from(camera.rot);
        let view = glam::Mat4::from_rotation_translation(rot, pos).inverse();
        let near = 0.01;
        let far = camera.depth;
        let proj = if let Some(fov) = camera.fov {
            // Asymmetric off-center projection for XR
            let left = -fov.left.tan() * near;
            let right = fov.right.tan() * near;
            let bottom = -fov.down.tan() * near;
            let top = fov.up.tan() * near;
            let w = right - left;
            let h = top - bottom;
            glam::Mat4::from_cols(
                glam::Vec4::new(2.0 * near / w, 0.0, 0.0, 0.0),
                glam::Vec4::new(0.0, 2.0 * near / h, 0.0, 0.0),
                glam::Vec4::new(
                    (right + left) / w,
                    (top + bottom) / h,
                    far / (near - far),
                    -1.0,
                ),
                glam::Vec4::new(0.0, 0.0, far * near / (near - far), 0.0),
            )
        } else {
            let aspect = self.surface_size.width as f32 / self.surface_size.height.max(1) as f32;
            glam::Mat4::perspective_rh(camera.fov_y, aspect, near, far)
        };
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();
        let light_dir = glam::Vec3::from(config.light_dir).normalize_or_zero();
        let light_view_proj = config
            .directional_shadows
            .map(|shadow| make_light_view_proj(camera, config.light_dir, shadow))
            .unwrap_or(glam::Mat4::IDENTITY);
        let shadow = config.directional_shadows.unwrap_or_default();
        RasterFrameParams {
            view_proj: view_proj.to_cols_array(),
            inv_view_proj: inv_view_proj.to_cols_array(),
            light_view_proj: light_view_proj.to_cols_array(),
            camera_pos: [pos.x, pos.y, pos.z, 1.0],
            light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
            light_color: {
                let c = config.light_color;
                [c.x, c.y, c.z, 0.0]
            },
            ambient_color: {
                let c = config.ambient_color;
                [c.x, c.y, c.z, config.space_sky as u32 as f32]
            },
            settings: [
                env_map_enabled as u32 as f32,
                // the surface may expect us to encode the values ourselves
                (self.color_space == gpu::ColorSpace::Srgb) as u32 as f32,
                0.0,
                0.0,
            ],
            shadow_params: [
                config.directional_shadows.is_some() as u32 as f32,
                shadow.strength.clamp(0.0, 1.0),
                shadow.normal_bias.max(0.0),
                1.0 / self.shadow_size as f32,
            ],
        }
    }
}

fn stochastic_light_seed(camera_pos: glam::Vec3) -> f32 {
    camera_pos.dot(glam::Vec3::new(1.0, 1.37, 9.17))
}

fn pack_point_lights(config: &RasterConfig, camera: &crate::Camera) -> PointLightParams {
    let mut lights = [PointLightGpu {
        pos_radius: [0.0; 4],
        color: [0.0; 4],
    }; MAX_POINT_LIGHTS];
    let count = config.point_lights.len().min(MAX_POINT_LIGHTS);
    for (slot, src) in lights
        .iter_mut()
        .zip(config.point_lights.iter())
        .take(count)
    {
        *slot = PointLightGpu {
            pos_radius: [
                src.position.x,
                src.position.y,
                src.position.z,
                src.radius.max(0.01),
            ],
            color: [src.color.x, src.color.y, src.color.z, 0.0],
        };
    }
    PointLightParams {
        count_seed: [
            count as f32,
            stochastic_light_seed(glam::Vec3::from(camera.pos)),
            0.0,
            0.0,
        ],
        lights,
    }
}

fn make_light_view_proj(
    camera: &crate::Camera,
    light_dir: mint::Vector3<f32>,
    config: DirectionalShadowConfig,
) -> glam::Mat4 {
    let center = glam::Vec3::from(camera.pos);
    let light_dir = glam::Vec3::from(light_dir).normalize_or_zero();
    let light_dir = if light_dir.length_squared() > 1e-5 {
        light_dir
    } else {
        glam::Vec3::Y
    };
    let depth = config.depth.max(2.0);
    let extent = config.distance.max(1.0);
    let eye = center + light_dir * (depth * 0.5);
    let up = if light_dir.dot(glam::Vec3::Y).abs() < 0.95 {
        glam::Vec3::Y
    } else {
        glam::Vec3::Z
    };
    let view = glam::Mat4::look_at_rh(eye, center, up);
    let projection = glam::Mat4::orthographic_rh(-extent, extent, -extent, extent, 0.1, depth);
    projection * view
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

impl gpu::Vertex for crate::SkinVertex {
    fn layout() -> gpu::VertexLayout {
        gpu::VertexLayout {
            attributes: vec![
                (
                    "joints",
                    gpu::VertexAttribute {
                        offset: 0,
                        format: gpu::VertexFormat::U32,
                    },
                ),
                (
                    "weights",
                    gpu::VertexAttribute {
                        offset: 4,
                        format: gpu::VertexFormat::U32,
                    },
                ),
            ],
            stride: mem::size_of::<crate::SkinVertex>() as u32,
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
