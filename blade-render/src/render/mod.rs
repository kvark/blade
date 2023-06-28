mod dummy;
mod env_map;
mod scene;

pub use dummy::DummyResources;
pub use env_map::EnvironmentMap;

use std::{collections::HashMap, mem, path::Path, ptr};

const MAX_RESOURCES: u32 = 1000;

#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    pub screen_size: blade_graphics::Extent,
    pub surface_format: blade_graphics::TextureFormat,
    pub max_debug_lines: u32,
}

struct Samplers {
    nearest: blade_graphics::Sampler,
    linear: blade_graphics::Sampler,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(u32)]
pub enum DebugMode {
    None = 0,
    Depth = 1,
    Normal = 2,
}

impl Default for DebugMode {
    fn default() -> Self {
        Self::None
    }
}

bitflags::bitflags! {
    #[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq, PartialOrd)]
    pub struct DebugFlags: u32 {
        const GEOMETRY = 1;
        const RESTIR = 2;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DebugBlitInput {
    Dummy,
    Environment,
    EnvironmentWeight,
}
impl Default for DebugBlitInput {
    fn default() -> Self {
        Self::Dummy
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DebugBlit {
    pub input: DebugBlitInput,
    pub offset: [i32; 2],
    pub scale_power: i32,
    pub mip_level: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DebugConfig {
    pub view_mode: DebugMode,
    pub flags: DebugFlags,
    pub mouse_pos: Option<[i32; 2]>,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct RayConfig {
    pub num_environment_samples: u32,
    pub environment_importance_sampling: bool,
    pub temporal_history: u32,
    pub spatial_taps: u32,
    pub spatial_tap_history: u32,
    pub spatial_radius: u32,
}

// Has to match the shader!
#[repr(C)]
#[derive(Debug)]
struct DebugVariance {
    color_sum: [f32; 3],
    pad: u32,
    color2_sum: [f32; 3],
    count: u32,
}

struct DebugRender {
    capacity: u32,
    buffer: blade_graphics::Buffer,
    variance_buffer: blade_graphics::Buffer,
    draw_pipeline: blade_graphics::RenderPipeline,
    blit_pipeline: blade_graphics::RenderPipeline,
    line_size: u32,
    buffer_size: u32,
}

struct FrameData {
    reservoir_buf: blade_graphics::Buffer,
    depth: blade_graphics::Texture,
    depth_view: blade_graphics::TextureView,
    basis: blade_graphics::Texture,
    basis_view: blade_graphics::TextureView,
    albedo: blade_graphics::Texture,
    albedo_view: blade_graphics::TextureView,
    camera_params: CameraParams,
}

impl FrameData {
    fn create_target(
        name: &str,
        format: blade_graphics::TextureFormat,
        size: blade_graphics::Extent,
        gpu: &blade_graphics::Context,
    ) -> (blade_graphics::Texture, blade_graphics::TextureView) {
        let texture = gpu.create_texture(blade_graphics::TextureDesc {
            name,
            format,
            size,
            dimension: blade_graphics::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: blade_graphics::TextureUsage::RESOURCE | blade_graphics::TextureUsage::STORAGE,
        });
        let view = gpu.create_texture_view(blade_graphics::TextureViewDesc {
            name,
            texture,
            format,
            dimension: blade_graphics::ViewDimension::D2,
            subresources: &blade_graphics::TextureSubresources::default(),
        });
        (texture, view)
    }

    fn new(
        size: blade_graphics::Extent,
        reservoir_size: u32,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
    ) -> Self {
        let total_reservoirs = size.width as usize * size.height as usize;
        let reservoir_buf = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "reservoirs",
            size: reservoir_size as u64 * total_reservoirs as u64,
            memory: blade_graphics::Memory::Device,
        });

        let (depth, depth_view) =
            Self::create_target("depth", blade_graphics::TextureFormat::R32Float, size, gpu);
        encoder.init_texture(depth);
        let (basis, basis_view) = Self::create_target(
            "basis",
            blade_graphics::TextureFormat::Rgba8Snorm,
            size,
            gpu,
        );
        encoder.init_texture(basis);
        let (albedo, albedo_view) = Self::create_target(
            "basis",
            blade_graphics::TextureFormat::Rgba8Unorm,
            size,
            gpu,
        );
        encoder.init_texture(albedo);
        Self {
            reservoir_buf,
            depth,
            depth_view,
            basis,
            basis_view,
            albedo,
            albedo_view,
            camera_params: CameraParams::default(),
        }
    }

    fn destroy(&self, gpu: &blade_graphics::Context) {
        gpu.destroy_buffer(self.reservoir_buf);
        gpu.destroy_texture_view(self.depth_view);
        gpu.destroy_texture(self.depth);
        gpu.destroy_texture_view(self.basis_view);
        gpu.destroy_texture(self.basis);
        gpu.destroy_texture_view(self.albedo_view);
        gpu.destroy_texture(self.albedo);
    }
}

/// Blade Renderer is a comprehensive rendering solution for
/// end user applications.
///
/// It takes care of the shaders, geometry buffers, acceleration structures,
/// dummy resources, and debug drawing.
///
/// It doesn't:
///   - manage or submit any command encoders
///   - know about the window to display on
pub struct Renderer {
    config: RenderConfig,
    shaders: Shaders,
    frame_data: [FrameData; 2],
    main_texture: blade_graphics::Texture,
    main_view: blade_graphics::TextureView,
    fill_pipeline: blade_graphics::ComputePipeline,
    main_pipeline: blade_graphics::ComputePipeline,
    blit_pipeline: blade_graphics::RenderPipeline,
    scene: super::Scene,
    acceleration_structure: blade_graphics::AccelerationStructure,
    env_map: EnvironmentMap,
    dummy: DummyResources,
    hit_buffer: blade_graphics::Buffer,
    vertex_buffers: blade_graphics::BufferArray<MAX_RESOURCES>,
    index_buffers: blade_graphics::BufferArray<MAX_RESOURCES>,
    textures: blade_graphics::TextureArray<MAX_RESOURCES>,
    samplers: Samplers,
    reservoir_size: u32,
    debug: DebugRender,
    is_tlas_dirty: bool,
    screen_size: blade_graphics::Extent,
    frame_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct CameraParams {
    position: [f32; 3],
    depth: f32,
    orientation: [f32; 4],
    fov: [f32; 2],
    target_size: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct DebugParams {
    view_mode: u32,
    flags: u32,
    mouse_pos: [i32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct MainParams {
    frame_index: u32,
    num_environment_samples: u32,
    environment_importance_sampling: u32,
    temporal_history: u32,
    spatial_taps: u32,
    spatial_tap_history: u32,
    spatial_radius: u32,
}

#[derive(blade_macros::ShaderData)]
struct FillData<'a> {
    camera: CameraParams,
    debug: DebugParams,
    acc_struct: blade_graphics::AccelerationStructure,
    hit_entries: blade_graphics::BufferPiece,
    index_buffers: &'a blade_graphics::BufferArray<MAX_RESOURCES>,
    vertex_buffers: &'a blade_graphics::BufferArray<MAX_RESOURCES>,
    textures: &'a blade_graphics::TextureArray<MAX_RESOURCES>,
    sampler_linear: blade_graphics::Sampler,
    debug_buf: blade_graphics::BufferPiece,
    out_depth: blade_graphics::TextureView,
    out_basis: blade_graphics::TextureView,
    out_albedo: blade_graphics::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct MainData {
    camera: CameraParams,
    prev_camera: CameraParams,
    debug: DebugParams,
    parameters: MainParams,
    acc_struct: blade_graphics::AccelerationStructure,
    sampler_linear: blade_graphics::Sampler,
    sampler_nearest: blade_graphics::Sampler,
    env_map: blade_graphics::TextureView,
    env_weights: blade_graphics::TextureView,
    t_depth: blade_graphics::TextureView,
    t_prev_depth: blade_graphics::TextureView,
    t_basis: blade_graphics::TextureView,
    t_prev_basis: blade_graphics::TextureView,
    t_albedo: blade_graphics::TextureView,
    t_prev_albedo: blade_graphics::TextureView,
    debug_buf: blade_graphics::BufferPiece,
    reservoirs: blade_graphics::BufferPiece,
    prev_reservoirs: blade_graphics::BufferPiece,
    output: blade_graphics::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct ToneMapParams {
    enabled: u32,
    average_lum: f32,
    key_value: f32,
    white_level: f32,
}

#[derive(blade_macros::ShaderData)]
struct BlitData {
    input: blade_graphics::TextureView,
    tone_map_params: ToneMapParams,
}

#[derive(blade_macros::ShaderData)]
struct DebugDrawData {
    camera: CameraParams,
    debug_buf: blade_graphics::BufferPiece,
    depth: blade_graphics::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct DebugBlitParams {
    target_offset: [f32; 2],
    target_size: [f32; 2],
    mip_level: f32,
    unused: u32,
}

#[derive(blade_macros::ShaderData)]
struct DebugBlitData {
    input: blade_graphics::TextureView,
    samp: blade_graphics::Sampler,
    params: DebugBlitParams,
}

#[repr(C)]
#[derive(Debug)]
struct HitEntry {
    index_buf: u32,
    vertex_buf: u32,
    geometry_to_world_rotation: [i8; 4],
    unused: u32,
    //Note: it's technically `mat4x3` on WGSL side,
    // but it's aligned and sized the same way as `mat4`.
    geometry_to_object: mint::ColumnMatrix4<f32>,
    base_color_texture: u32,
    base_color_factor: [u8; 4],
    // make sure the end of the struct is aligned
    finish_pad: [u32; 2],
}

#[derive(Clone, PartialEq)]
pub struct Shaders {
    fill_gbuf: blade_asset::Handle<crate::Shader>,
    ray_trace: blade_asset::Handle<crate::Shader>,
    post_proc: blade_asset::Handle<crate::Shader>,
    debug_draw: blade_asset::Handle<crate::Shader>,
    debug_blit: blade_asset::Handle<crate::Shader>,
}

impl Shaders {
    pub fn load(path: &Path, asset_hub: &crate::AssetHub) -> (Self, choir::RunningTask) {
        let mut ctx = asset_hub.open_context(path, "shader finish");
        let shaders = Self {
            fill_gbuf: ctx.load_shader("fill-gbuf.wgsl"),
            ray_trace: ctx.load_shader("ray-trace.wgsl"),
            post_proc: ctx.load_shader("post-proc.wgsl"),
            debug_draw: ctx.load_shader("debug-draw.wgsl"),
            debug_blit: ctx.load_shader("debug-blit.wgsl"),
        };
        (shaders, ctx.close())
    }
}

struct ShaderPipelines {
    fill: blade_graphics::ComputePipeline,
    main: blade_graphics::ComputePipeline,
    post_proc: blade_graphics::RenderPipeline,
    debug_draw: blade_graphics::RenderPipeline,
    debug_blit: blade_graphics::RenderPipeline,
    env_preproc: blade_graphics::ComputePipeline,
    debug_line_size: u32,
    debug_buffer_size: u32,
    reservoir_size: u32,
}

impl ShaderPipelines {
    fn create_gbuf_fill(
        shader: &blade_graphics::Shader,
        gpu: &blade_graphics::Context,
    ) -> blade_graphics::ComputePipeline {
        shader.check_struct_size::<crate::Vertex>();
        shader.check_struct_size::<HitEntry>();
        let layout = <FillData as blade_graphics::ShaderData>::layout();
        gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "fill-gbuf",
            data_layouts: &[&layout],
            compute: shader.at("main"),
        })
    }
    fn create_ray_trace(
        shader: &blade_graphics::Shader,
        gpu: &blade_graphics::Context,
    ) -> blade_graphics::ComputePipeline {
        shader.check_struct_size::<CameraParams>();
        shader.check_struct_size::<DebugParams>();
        shader.check_struct_size::<MainParams>();
        shader.check_struct_size::<DebugVariance>();
        let layout = <MainData as blade_graphics::ShaderData>::layout();
        gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "ray-trace",
            data_layouts: &[&layout],
            compute: shader.at("main"),
        })
    }
    fn create_post_proc(
        shader: &blade_graphics::Shader,
        format: blade_graphics::TextureFormat,
        gpu: &blade_graphics::Context,
    ) -> blade_graphics::RenderPipeline {
        let layout = <BlitData as blade_graphics::ShaderData>::layout();
        gpu.create_render_pipeline(blade_graphics::RenderPipelineDesc {
            name: "main",
            data_layouts: &[&layout],
            primitive: blade_graphics::PrimitiveState {
                topology: blade_graphics::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            vertex: shader.at("blit_vs"),
            fragment: shader.at("blit_fs"),
            color_targets: &[format.into()],
            depth_stencil: None,
        })
    }
    fn create_debug_draw(
        shader: &blade_graphics::Shader,
        format: blade_graphics::TextureFormat,
        gpu: &blade_graphics::Context,
    ) -> blade_graphics::RenderPipeline {
        let layout = <DebugDrawData as blade_graphics::ShaderData>::layout();
        gpu.create_render_pipeline(blade_graphics::RenderPipelineDesc {
            name: "debug-draw",
            data_layouts: &[&layout],
            vertex: shader.at("debug_vs"),
            primitive: blade_graphics::PrimitiveState {
                topology: blade_graphics::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            fragment: shader.at("debug_fs"),
            color_targets: &[blade_graphics::ColorTargetState {
                format,
                blend: Some(blade_graphics::BlendState::ALPHA_BLENDING),
                write_mask: blade_graphics::ColorWrites::all(),
            }],
        })
    }
    fn create_debug_blit(
        shader: &blade_graphics::Shader,
        format: blade_graphics::TextureFormat,
        gpu: &blade_graphics::Context,
    ) -> blade_graphics::RenderPipeline {
        shader.check_struct_size::<DebugBlitParams>();
        let layout = <DebugBlitData as blade_graphics::ShaderData>::layout();
        gpu.create_render_pipeline(blade_graphics::RenderPipelineDesc {
            name: "debug-blit",
            data_layouts: &[&layout],
            vertex: shader.at("blit_vs"),
            primitive: blade_graphics::PrimitiveState {
                topology: blade_graphics::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            fragment: shader.at("blit_fs"),
            color_targets: &[format.into()],
        })
    }

    fn init(
        shaders: &Shaders,
        config: &RenderConfig,
        gpu: &blade_graphics::Context,
        shader_man: &blade_asset::AssetManager<crate::shader::Baker>,
    ) -> Result<Self, &'static str> {
        let sh_main = &shader_man[shaders.ray_trace].raw;
        Ok(Self {
            fill: Self::create_gbuf_fill(&shader_man[shaders.fill_gbuf].raw, gpu),
            main: Self::create_ray_trace(sh_main, gpu),
            post_proc: Self::create_post_proc(
                &shader_man[shaders.post_proc].raw,
                config.surface_format,
                gpu,
            ),
            debug_draw: Self::create_debug_draw(
                &shader_man[shaders.debug_draw].raw,
                config.surface_format,
                gpu,
            ),
            debug_blit: Self::create_debug_blit(
                &shader_man[shaders.debug_blit].raw,
                config.surface_format,
                gpu,
            ),
            env_preproc: EnvironmentMap::init_pipeline(gpu)?,
            debug_line_size: sh_main.get_struct_size("DebugLine"),
            debug_buffer_size: sh_main.get_struct_size("DebugBuffer"),
            reservoir_size: sh_main.get_struct_size("StoredReservoir"),
        })
    }
}

impl Renderer {
    /// Create a new renderer with a given configuration.
    ///
    /// Panics if the system is not compatible.
    /// Records initialization routines into the given command encoder.
    pub fn new(
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
        shaders: Shaders,
        shader_man: &blade_asset::AssetManager<crate::shader::Baker>,
        config: &RenderConfig,
    ) -> Self {
        let capabilities = gpu.capabilities();
        assert!(capabilities
            .ray_query
            .contains(blade_graphics::ShaderVisibility::COMPUTE));

        let sp = ShaderPipelines::init(&shaders, config, gpu, shader_man).unwrap();

        let debug_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "debug",
            size: (sp.debug_buffer_size + (config.max_debug_lines - 1) * sp.debug_line_size) as u64,
            memory: blade_graphics::Memory::Device,
        });
        let variance_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "variance",
            size: mem::size_of::<DebugVariance>() as u64,
            memory: blade_graphics::Memory::Shared,
        });
        unsafe {
            ptr::write_bytes(variance_buffer.data(), 0, mem::size_of::<DebugVariance>());
        }

        let (main_texture, main_view) = FrameData::create_target(
            "main",
            blade_graphics::TextureFormat::Rgba16Float,
            config.screen_size,
            gpu,
        );
        encoder.init_texture(main_texture);
        let frame_data = [
            FrameData::new(config.screen_size, sp.reservoir_size, encoder, gpu),
            FrameData::new(config.screen_size, sp.reservoir_size, encoder, gpu),
        ];
        let dummy = DummyResources::new(encoder, gpu);

        let samplers = Samplers {
            nearest: gpu.create_sampler(blade_graphics::SamplerDesc {
                name: "nearest",
                address_modes: [blade_graphics::AddressMode::ClampToEdge; 3],
                mag_filter: blade_graphics::FilterMode::Nearest,
                min_filter: blade_graphics::FilterMode::Nearest,
                mipmap_filter: blade_graphics::FilterMode::Nearest,
                ..Default::default()
            }),
            linear: gpu.create_sampler(blade_graphics::SamplerDesc {
                name: "linear",
                address_modes: [blade_graphics::AddressMode::Repeat; 3],
                mag_filter: blade_graphics::FilterMode::Linear,
                min_filter: blade_graphics::FilterMode::Linear,
                mipmap_filter: blade_graphics::FilterMode::Linear,
                ..Default::default()
            }),
        };

        Self {
            config: *config,
            shaders,
            frame_data,
            main_texture,
            main_view,
            scene: super::Scene::default(),
            fill_pipeline: sp.fill,
            main_pipeline: sp.main,
            blit_pipeline: sp.post_proc,
            acceleration_structure: blade_graphics::AccelerationStructure::default(),
            env_map: EnvironmentMap::with_pipeline(&dummy, sp.env_preproc),
            dummy,
            hit_buffer: blade_graphics::Buffer::default(),
            vertex_buffers: blade_graphics::BufferArray::new(),
            index_buffers: blade_graphics::BufferArray::new(),
            textures: blade_graphics::TextureArray::new(),
            samplers,
            reservoir_size: sp.reservoir_size,
            debug: DebugRender {
                capacity: config.max_debug_lines,
                buffer: debug_buffer,
                variance_buffer,
                draw_pipeline: sp.debug_draw,
                blit_pipeline: sp.debug_blit,
                line_size: sp.debug_line_size,
                buffer_size: sp.debug_buffer_size,
            },
            is_tlas_dirty: true,
            screen_size: config.screen_size,
            frame_index: 0,
        }
    }

    /// Destroy all internally managed GPU resources.
    pub fn destroy(&mut self, gpu: &blade_graphics::Context) {
        // internal resources
        for frame_data in self.frame_data.iter_mut() {
            frame_data.destroy(gpu);
        }
        gpu.destroy_texture_view(self.main_view);
        gpu.destroy_texture(self.main_texture);
        if self.hit_buffer != blade_graphics::Buffer::default() {
            gpu.destroy_buffer(self.hit_buffer);
        }
        gpu.destroy_acceleration_structure(self.acceleration_structure);
        // env map, dummy
        self.env_map.destroy(gpu);
        self.dummy.destroy(gpu);
        // samplers
        gpu.destroy_sampler(self.samplers.nearest);
        gpu.destroy_sampler(self.samplers.linear);
        // buffers
        gpu.destroy_buffer(self.debug.buffer);
        gpu.destroy_buffer(self.debug.variance_buffer);
    }

    pub fn merge_scene(&mut self, scene: super::Scene) {
        self.scene = scene;
        self.is_tlas_dirty = true;
    }

    pub fn hot_reload(
        &mut self,
        asset_hub: &crate::AssetHub,
        gpu: &blade_graphics::Context,
        sync_point: &blade_graphics::SyncPoint,
    ) -> bool {
        let mut tasks = Vec::new();
        let old = self.shaders.clone();

        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.fill_gbuf));
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.ray_trace));
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.post_proc));
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.debug_draw));
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.debug_blit));

        if tasks.is_empty() {
            return false;
        }

        log::info!("Hot reloading shaders");
        gpu.wait_for(sync_point, !0);
        for task in tasks {
            let _ = task.join();
        }

        if self.shaders.fill_gbuf != old.fill_gbuf {
            self.fill_pipeline = ShaderPipelines::create_gbuf_fill(
                &asset_hub.shaders[self.shaders.fill_gbuf].raw,
                gpu,
            );
        }
        if self.shaders.ray_trace != old.ray_trace {
            let shader = &asset_hub.shaders[self.shaders.ray_trace].raw;
            assert_eq!(shader.get_struct_size("DebugLine"), self.debug.line_size);
            assert_eq!(
                shader.get_struct_size("DebugBuffer"),
                self.debug.buffer_size
            );
            assert_eq!(
                shader.get_struct_size("StoredReservoir"),
                self.reservoir_size
            );
            self.main_pipeline = ShaderPipelines::create_ray_trace(shader, gpu);
        }
        if self.shaders.post_proc != old.post_proc {
            self.blit_pipeline = ShaderPipelines::create_post_proc(
                &asset_hub.shaders[self.shaders.post_proc].raw,
                self.config.surface_format,
                gpu,
            );
        }
        if self.shaders.debug_draw != old.debug_draw {
            self.debug.draw_pipeline = ShaderPipelines::create_debug_draw(
                &asset_hub.shaders[self.shaders.debug_draw].raw,
                self.config.surface_format,
                gpu,
            );
        }
        if self.shaders.debug_blit != old.debug_blit {
            self.debug.blit_pipeline = ShaderPipelines::create_debug_blit(
                &asset_hub.shaders[self.shaders.debug_blit].raw,
                self.config.surface_format,
                gpu,
            );
        }

        true
    }

    pub fn get_screen_size(&self) -> blade_graphics::Extent {
        self.screen_size
    }

    pub fn resize_screen(
        &mut self,
        size: blade_graphics::Extent,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
    ) {
        self.screen_size = size;
        for frame_data in self.frame_data.iter_mut() {
            frame_data.destroy(gpu);
            *frame_data = FrameData::new(size, self.reservoir_size, encoder, gpu);
        }

        gpu.destroy_texture(self.main_texture);
        gpu.destroy_texture_view(self.main_view);
        let (main_texture, main_view) = FrameData::create_target(
            "main",
            blade_graphics::TextureFormat::Rgba16Float,
            size,
            gpu,
        );
        encoder.init_texture(main_texture);
        self.main_texture = main_texture;
        self.main_view = main_view;
    }

    /// Prepare to render a frame.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        camera: &super::Camera,
        asset_hub: &crate::AssetHub,
        gpu: &blade_graphics::Context,
        temp_buffers: &mut Vec<blade_graphics::Buffer>,
        temp_acceleration_structures: &mut Vec<blade_graphics::AccelerationStructure>,
        enable_debug: bool,
        reset_accumulation: bool,
    ) {
        if self.is_tlas_dirty {
            self.is_tlas_dirty = false;
            if self.acceleration_structure != blade_graphics::AccelerationStructure::default() {
                temp_buffers.push(self.hit_buffer);
                temp_acceleration_structures.push(self.acceleration_structure);
            }

            let (tlas, geometry_count) = self.scene.build_top_level_acceleration_structure(
                command_encoder,
                &asset_hub.models,
                gpu,
                temp_buffers,
            );
            self.acceleration_structure = tlas;
            log::info!("Preparing ray tracing with {geometry_count} geometries in total");
            let mut transfers = command_encoder.transfer();

            {
                // init the debug buffer
                let data = [2, 0, 0, 0, self.debug.capacity, 0];
                let size = 4 * data.len() as u64;
                let staging = gpu.create_buffer(blade_graphics::BufferDesc {
                    name: "debug buf staging",
                    size,
                    memory: blade_graphics::Memory::Upload,
                });
                unsafe {
                    ptr::write(staging.data() as *mut _, data);
                }
                transfers.copy_buffer_to_buffer(staging.into(), self.debug.buffer.into(), size);
                temp_buffers.push(staging);
            }

            self.vertex_buffers.clear();
            self.index_buffers.clear();
            self.textures.clear();
            let dummy_white = self.textures.alloc(self.dummy.white_view);

            if geometry_count != 0 {
                let hit_staging = {
                    // init the hit buffer
                    let hit_size = (geometry_count as usize * mem::size_of::<HitEntry>()) as u64;
                    self.hit_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
                        name: "hit entries",
                        size: hit_size,
                        memory: blade_graphics::Memory::Device,
                    });
                    let staging = gpu.create_buffer(blade_graphics::BufferDesc {
                        name: "hit staging",
                        size: hit_size,
                        memory: blade_graphics::Memory::Upload,
                    });
                    temp_buffers.push(staging);
                    transfers.copy_buffer_to_buffer(staging.at(0), self.hit_buffer.at(0), hit_size);
                    staging
                };

                fn extract_matrix3(transform: blade_graphics::Transform) -> glam::Mat3 {
                    let col_mx = mint::ColumnMatrix3x4::from(transform);
                    glam::Mat3::from_cols(col_mx.x.into(), col_mx.y.into(), col_mx.z.into())
                }

                let mut texture_indices = HashMap::new();
                let mut geometry_index = 0;
                for object in self.scene.objects.iter() {
                    let m3_object = extract_matrix3(object.transform);
                    let model = &asset_hub.models[object.model];
                    for geometry in model.geometries.iter() {
                        let material = &model.materials[geometry.material_index];
                        let vertex_offset = geometry.vertex_range.start as u64
                            * mem::size_of::<crate::Vertex>() as u64;
                        let geometry_to_world_rotation = {
                            let m3_geo = extract_matrix3(geometry.transform);
                            let m3_normal = (m3_object * m3_geo).inverse().transpose();
                            let quat = glam::Quat::from_mat3(&m3_normal);
                            let qv = glam::Vec4::from(quat) * 127.0;
                            [qv.x as i8, qv.y as i8, qv.z as i8, qv.w as i8]
                        };
                        fn extend(v: mint::Vector3<f32>) -> mint::Vector4<f32> {
                            mint::Vector4 {
                                x: v.x,
                                y: v.y,
                                z: v.z,
                                w: 0.0,
                            }
                        }

                        let hit_entry = HitEntry {
                            index_buf: match geometry.index_type {
                                Some(_) => self
                                    .index_buffers
                                    .alloc(model.index_buffer.at(geometry.index_offset)),
                                None => !0,
                            },
                            vertex_buf: self
                                .vertex_buffers
                                .alloc(model.vertex_buffer.at(vertex_offset)),
                            geometry_to_world_rotation,
                            unused: 0,
                            geometry_to_object: {
                                let m = mint::ColumnMatrix3x4::from(geometry.transform);
                                mint::ColumnMatrix4 {
                                    x: extend(m.x),
                                    y: extend(m.y),
                                    z: extend(m.z),
                                    w: extend(m.w),
                                }
                            },
                            base_color_texture: match material.base_color_texture {
                                Some(handle) => {
                                    *texture_indices.entry(handle).or_insert_with(|| {
                                        let texture = &asset_hub.textures[handle];
                                        self.textures.alloc(texture.view)
                                    })
                                }
                                None => dummy_white,
                            },
                            base_color_factor: {
                                let c = material.base_color_factor;
                                [
                                    (c[0] * 255.0) as u8,
                                    (c[1] * 255.0) as u8,
                                    (c[2] * 255.0) as u8,
                                    (c[3] * 255.0) as u8,
                                ]
                            },
                            finish_pad: [0; 2],
                        };

                        log::debug!("Entry[{geometry_index}] = {hit_entry:?}");
                        unsafe {
                            ptr::write(
                                (hit_staging.data() as *mut HitEntry).add(geometry_index),
                                hit_entry,
                            );
                        }
                        geometry_index += 1;
                    }
                }
                assert_eq!(geometry_index, geometry_count as usize);
            } else {
                self.hit_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
                    name: "hit entries",
                    size: mem::size_of::<HitEntry>() as u64,
                    memory: blade_graphics::Memory::Device,
                });
            }
        }

        if let Some(handle) = self.scene.environment_map {
            let asset = &asset_hub.textures[handle];
            self.env_map
                .assign(asset.view, asset.extent, command_encoder, gpu);
        };

        let mut transfer = command_encoder.transfer();
        if enable_debug {
            // reset the debug line count
            transfer.fill_buffer(self.debug.buffer.at(4), 4, 0);
            transfer.fill_buffer(self.debug.buffer.at(20), 4, 1);
            if !reset_accumulation {
                // copy the previous frame variance
                transfer.copy_buffer_to_buffer(
                    self.debug.buffer.at(32),
                    self.debug.variance_buffer.into(),
                    mem::size_of::<DebugVariance>() as u64,
                );
            }
        } else {
            // reset the open bit
            transfer.fill_buffer(self.debug.buffer.at(20), 12, 0);
        }
        if reset_accumulation {
            // reset the open bit, variance accumulator
            transfer.fill_buffer(
                self.debug.buffer.at(32),
                mem::size_of::<DebugVariance>() as u64,
                0,
            );
            let total_reservoirs = self.screen_size.width as u64 * self.screen_size.height as u64;
            transfer.fill_buffer(
                self.frame_data[0].reservoir_buf.into(),
                total_reservoirs * self.reservoir_size as u64,
                0,
            );
        }

        self.frame_index += 1;
        self.frame_data.swap(0, 1);
        self.frame_data[0].camera_params = self.make_camera_params(camera);
    }

    fn make_camera_params(&self, camera: &super::Camera) -> CameraParams {
        let fov_x = 2.0
            * ((camera.fov_y * 0.5).tan() * self.screen_size.width as f32
                / self.screen_size.height as f32)
                .atan();
        CameraParams {
            position: camera.pos.into(),
            depth: camera.depth,
            orientation: camera.rot.into(),
            fov: [fov_x, camera.fov_y],
            target_size: [self.screen_size.width, self.screen_size.height],
        }
    }

    /// Ray trace the scene.
    ///
    /// The result is stored internally in an HDR render target.
    pub fn ray_trace(
        &self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        debug_config: DebugConfig,
        ray_config: RayConfig,
    ) {
        assert!(!self.is_tlas_dirty);
        let debug = DebugParams {
            view_mode: debug_config.view_mode as u32,
            flags: debug_config.flags.bits(),
            mouse_pos: match debug_config.mouse_pos {
                Some(p) => [p[0], self.screen_size.height as i32 - p[1]],
                None => [-1; 2],
            },
        };
        let cur = self.frame_data.first().unwrap();
        let prev = self.frame_data.last().unwrap();

        if let mut pass = command_encoder.compute() {
            let mut pc = pass.with(&self.fill_pipeline);
            let groups = self.fill_pipeline.get_dispatch_for(self.screen_size);
            pc.bind(
                0,
                &FillData {
                    camera: cur.camera_params,
                    debug,
                    acc_struct: self.acceleration_structure,
                    hit_entries: self.hit_buffer.into(),
                    index_buffers: &self.index_buffers,
                    vertex_buffers: &self.vertex_buffers,
                    textures: &self.textures,
                    sampler_linear: self.samplers.linear,
                    debug_buf: self.debug.buffer.into(),
                    out_depth: cur.depth_view,
                    out_basis: cur.basis_view,
                    out_albedo: cur.albedo_view,
                },
            );
            pc.dispatch(groups);
        }

        if let mut pass = command_encoder.compute() {
            let mut pc = pass.with(&self.main_pipeline);
            let groups = self.main_pipeline.get_dispatch_for(self.screen_size);
            pc.bind(
                0,
                &MainData {
                    camera: cur.camera_params,
                    prev_camera: prev.camera_params,
                    debug,
                    parameters: MainParams {
                        frame_index: self.frame_index,
                        num_environment_samples: ray_config.num_environment_samples,
                        environment_importance_sampling: ray_config.environment_importance_sampling
                            as u32,
                        temporal_history: ray_config.temporal_history,
                        spatial_taps: ray_config.spatial_taps,
                        spatial_tap_history: ray_config.spatial_tap_history,
                        spatial_radius: ray_config.spatial_radius,
                    },
                    acc_struct: self.acceleration_structure,
                    sampler_linear: self.samplers.linear,
                    sampler_nearest: self.samplers.nearest,
                    env_map: self.env_map.main_view,
                    env_weights: self.env_map.weight_view,
                    t_depth: cur.depth_view,
                    t_prev_depth: prev.depth_view,
                    t_basis: cur.basis_view,
                    t_prev_basis: prev.basis_view,
                    t_albedo: cur.albedo_view,
                    t_prev_albedo: prev.albedo_view,
                    debug_buf: self.debug.buffer.into(),
                    reservoirs: cur.reservoir_buf.into(),
                    prev_reservoirs: prev.reservoir_buf.into(),
                    output: self.main_view,
                },
            );
            pc.dispatch(groups);
        }
    }

    /// Blit the rendering result into a specified render pass.
    pub fn blit(&self, pass: &mut blade_graphics::RenderCommandEncoder, debug_blits: &[DebugBlit]) {
        let pp = &self.scene.post_processing;
        if let mut pc = pass.with(&self.blit_pipeline) {
            pc.bind(
                0,
                &BlitData {
                    input: self.main_view,
                    tone_map_params: ToneMapParams {
                        enabled: 1,
                        average_lum: pp.average_luminocity,
                        key_value: pp.exposure_key_value,
                        white_level: pp.white_level,
                    },
                },
            );
            pc.draw(0, 3, 0, 1);
        }
        if let mut pc = pass.with(&self.debug.draw_pipeline) {
            let cur = self.frame_data.first().unwrap();
            pc.bind(
                0,
                &DebugDrawData {
                    camera: cur.camera_params,
                    debug_buf: self.debug.buffer.into(),
                    depth: cur.depth_view,
                },
            );
            pc.draw_indirect(self.debug.buffer.at(0));
        }
        if let mut pc = pass.with(&self.debug.blit_pipeline) {
            fn scale(dim: u32, power: i32) -> u32 {
                if power >= 0 {
                    dim.max(1) << power
                } else {
                    (dim >> -power).max(1)
                }
            }
            for db in debug_blits {
                let (input, size) = match db.input {
                    DebugBlitInput::Dummy => {
                        (self.dummy.white_view, blade_graphics::Extent::default())
                    }
                    DebugBlitInput::Environment => (self.env_map.main_view, self.env_map.size),
                    DebugBlitInput::EnvironmentWeight => {
                        (self.env_map.weight_view, self.env_map.weight_size())
                    }
                };
                pc.bind(
                    0,
                    &DebugBlitData {
                        input,
                        samp: self.samplers.linear,
                        params: DebugBlitParams {
                            target_offset: [
                                db.offset[0] as f32 / self.screen_size.width as f32,
                                db.offset[1] as f32 / self.screen_size.height as f32,
                            ],
                            target_size: [
                                scale(size.width >> db.mip_level, db.scale_power) as f32
                                    / self.screen_size.width as f32,
                                scale(size.height >> db.mip_level, db.scale_power) as f32
                                    / self.screen_size.height as f32,
                            ],
                            mip_level: db.mip_level as f32,
                            unused: 0,
                        },
                    },
                );
                pc.draw(0, 4, 0, 1);
            }
        }
    }

    pub fn read_debug_std_deviation(&self) -> Option<mint::Vector3<f32>> {
        let dv = unsafe { &*(self.debug.variance_buffer.data() as *const DebugVariance) };
        if dv.count == 0 {
            return None;
        }
        let sum_avg = glam::Vec3::from(dv.color_sum) / (dv.count as f32);
        let sum2_avg = glam::Vec3::from(dv.color2_sum) / (dv.count as f32);
        let variance = sum2_avg - sum_avg * sum_avg;
        Some(mint::Vector3 {
            x: variance.x.sqrt(),
            y: variance.y.sqrt(),
            z: variance.z.sqrt(),
        })
    }

    pub fn configure_post_processing(&mut self) -> &mut crate::PostProcessing {
        &mut self.scene.post_processing
    }
}
