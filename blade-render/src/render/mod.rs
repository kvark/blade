mod dummy;
mod env_map;

pub use dummy::DummyResources;
pub use env_map::EnvironmentMap;

use std::{collections::HashMap, mem, num::NonZeroU32, path::Path, ptr};

const MAX_RESOURCES: u32 = 1000;
const RADIANCE_FORMAT: blade_graphics::TextureFormat = blade_graphics::TextureFormat::Rgba16Float;

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

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, blade_macros::AsPrimitive, strum::EnumIter)]
#[repr(u32)]
pub enum DebugMode {
    Final = 0,
    Depth = 1,
    Normal = 2,
    HitConsistency = 3,
    Variance = 4,
}

impl Default for DebugMode {
    fn default() -> Self {
        Self::Final
    }
}

bitflags::bitflags! {
    #[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq, PartialOrd)]
    pub struct DebugDrawFlags: u32 {
        const GEOMETRY = 1;
        const RESTIR = 2;
    }
}

bitflags::bitflags! {
    #[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq, PartialOrd)]
    pub struct DebugTextureFlags: u32 {
        const ALBEDO = 1;
        const NORMAL = 2;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DebugBlit {
    pub input: blade_graphics::TextureView,
    pub mip_level: u32,
    pub target_offset: [i32; 2],
    pub target_size: [u32; 2],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DebugConfig {
    pub view_mode: DebugMode,
    pub draw_flags: DebugDrawFlags,
    pub texture_flags: DebugTextureFlags,
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

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct DenoiserConfig {
    pub num_passes: u32,
    pub temporal_weight: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct PostProcConfig {
    //TODO: compute automatically
    pub average_luminocity: f32,
    pub exposure_key_value: f32,
    pub white_level: f32,
}
impl Default for PostProcConfig {
    fn default() -> Self {
        Self {
            average_luminocity: 1.0,
            exposure_key_value: 1.0,
            white_level: 1.0,
        }
    }
}

pub struct SelectionInfo {
    pub std_deviation: mint::Vector3<f32>,
    pub std_deviation_history: u32,
    pub custom_index: u32,
    pub depth: f32,
    pub tex_coords: mint::Vector2<f32>,
    pub base_color_texture: Option<blade_asset::Handle<crate::Texture>>,
    pub normal_texture: Option<blade_asset::Handle<crate::Texture>>,
}
impl Default for SelectionInfo {
    fn default() -> Self {
        Self {
            std_deviation: [0.0; 3].into(),
            std_deviation_history: 0,
            custom_index: 0,
            depth: 0.0,
            tex_coords: [0.0; 2].into(),
            base_color_texture: None,
            normal_texture: None,
        }
    }
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

// Has to match the shader!
#[repr(C)]
#[derive(Debug)]
struct DebugEntry {
    custom_index: u32,
    depth: f32,
    tex_coords: [f32; 2],
    base_color_texture: u32,
    normal_texture: u32,
}

struct DebugRender {
    _capacity: u32,
    buffer: blade_graphics::Buffer,
    variance_buffer: blade_graphics::Buffer,
    entry_buffer: blade_graphics::Buffer,
    draw_pipeline: blade_graphics::RenderPipeline,
    blit_pipeline: blade_graphics::RenderPipeline,
    line_size: u32,
    buffer_size: u32,
}

#[allow(dead_code)]
struct DoubleRenderTarget {
    texture: blade_graphics::Texture,
    views: [blade_graphics::TextureView; 2],
    active: usize,
}
#[allow(dead_code)]
impl DoubleRenderTarget {
    fn new(
        name: &str,
        format: blade_graphics::TextureFormat,
        size: blade_graphics::Extent,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
    ) -> Self {
        let texture = gpu.create_texture(blade_graphics::TextureDesc {
            name,
            format,
            size,
            dimension: blade_graphics::TextureDimension::D2,
            array_layer_count: 2,
            mip_level_count: 1,
            usage: blade_graphics::TextureUsage::RESOURCE | blade_graphics::TextureUsage::STORAGE,
        });
        encoder.init_texture(texture);
        let mut views = [blade_graphics::TextureView::default(); 2];
        for (i, view) in views.iter_mut().enumerate() {
            *view = gpu.create_texture_view(blade_graphics::TextureViewDesc {
                name: &format!("{name}-{i}"),
                texture,
                format,
                dimension: blade_graphics::ViewDimension::D2,
                subresources: &blade_graphics::TextureSubresources {
                    base_array_layer: i as u32,
                    array_layer_count: NonZeroU32::new(1),
                    ..Default::default()
                },
            });
        }

        Self {
            texture,
            views,
            active: 0,
        }
    }

    fn destroy(&mut self, gpu: &blade_graphics::Context) {
        gpu.destroy_texture(self.texture);
        for view in self.views.iter_mut() {
            gpu.destroy_texture_view(*view);
        }
    }

    fn swap(&mut self) {
        self.active = 1 - self.active;
    }

    fn cur(&self) -> blade_graphics::TextureView {
        self.views[self.active]
    }
    fn prev(&self) -> blade_graphics::TextureView {
        self.views[1 - self.active]
    }
}

struct FrameData {
    reservoir_buf: blade_graphics::Buffer,
    depth: blade_graphics::Texture,
    depth_view: blade_graphics::TextureView,
    basis: blade_graphics::Texture,
    basis_view: blade_graphics::TextureView,
    flat_normal: blade_graphics::Texture,
    flat_normal_view: blade_graphics::TextureView,
    albedo: blade_graphics::Texture,
    albedo_view: blade_graphics::TextureView,
    light_diffuse: blade_graphics::Texture,
    light_diffuse_view: blade_graphics::TextureView,
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

        let (flat_normal, flat_normal_view) = Self::create_target(
            "flat-normal",
            blade_graphics::TextureFormat::Rgba8Snorm,
            size,
            gpu,
        );
        encoder.init_texture(flat_normal);

        let (albedo, albedo_view) = Self::create_target(
            "basis",
            blade_graphics::TextureFormat::Rgba8Unorm,
            size,
            gpu,
        );
        encoder.init_texture(albedo);

        let (light_diffuse, light_diffuse_view) =
            Self::create_target("light-diffuse", RADIANCE_FORMAT, size, gpu);
        encoder.init_texture(light_diffuse);

        Self {
            reservoir_buf,
            depth,
            depth_view,
            basis,
            basis_view,
            flat_normal,
            flat_normal_view,
            albedo,
            albedo_view,
            light_diffuse,
            light_diffuse_view,
            camera_params: CameraParams::default(),
        }
    }

    fn destroy(&self, gpu: &blade_graphics::Context) {
        gpu.destroy_buffer(self.reservoir_buf);
        gpu.destroy_texture_view(self.depth_view);
        gpu.destroy_texture(self.depth);
        gpu.destroy_texture_view(self.basis_view);
        gpu.destroy_texture(self.basis);
        gpu.destroy_texture_view(self.flat_normal_view);
        gpu.destroy_texture(self.flat_normal);
        gpu.destroy_texture_view(self.albedo_view);
        gpu.destroy_texture(self.albedo);
        gpu.destroy_texture_view(self.light_diffuse_view);
        gpu.destroy_texture(self.light_diffuse);
    }
}

struct Blur {
    temporal_accum_pipeline: blade_graphics::ComputePipeline,
    atrous_pipeline: blade_graphics::ComputePipeline,
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
    light_temp_texture: blade_graphics::Texture,
    light_temp_view: blade_graphics::TextureView,
    post_proc_input: blade_graphics::TextureView,
    debug_texture: blade_graphics::Texture,
    debug_view: blade_graphics::TextureView,
    fill_pipeline: blade_graphics::ComputePipeline,
    main_pipeline: blade_graphics::ComputePipeline,
    post_proc_pipeline: blade_graphics::RenderPipeline,
    blur: Blur,
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
    screen_size: blade_graphics::Extent,
    frame_index: u32,
    //TODO: refactor `ResourceArray` to not carry the freelist logic
    // This way we can embed user info into the allocator.
    texture_resource_lookup:
        HashMap<blade_graphics::ResourceIndex, blade_asset::Handle<crate::Texture>>,
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
    draw_flags: u32,
    texture_flags: u32,
    unused: u32,
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
    out_flat_normal: blade_graphics::TextureView,
    out_albedo: blade_graphics::TextureView,
    out_debug: blade_graphics::TextureView,
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
    t_flat_normal: blade_graphics::TextureView,
    t_prev_flat_normal: blade_graphics::TextureView,
    debug_buf: blade_graphics::BufferPiece,
    reservoirs: blade_graphics::BufferPiece,
    prev_reservoirs: blade_graphics::BufferPiece,
    out_diffuse: blade_graphics::TextureView,
    out_debug: blade_graphics::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct BlurParams {
    extent: [u32; 2],
    temporal_weight: f32,
    iteration: i32,
}

#[derive(blade_macros::ShaderData)]
struct TemporalAccumData {
    camera: CameraParams,
    prev_camera: CameraParams,
    params: BlurParams,
    input: blade_graphics::TextureView,
    prev_input: blade_graphics::TextureView,
    t_depth: blade_graphics::TextureView,
    t_prev_depth: blade_graphics::TextureView,
    t_flat_normal: blade_graphics::TextureView,
    t_prev_flat_normal: blade_graphics::TextureView,
    output: blade_graphics::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct AtrousData {
    params: BlurParams,
    input: blade_graphics::TextureView,
    t_flat_normal: blade_graphics::TextureView,
    t_depth: blade_graphics::TextureView,
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
struct PostProcData {
    t_albedo: blade_graphics::TextureView,
    light_diffuse: blade_graphics::TextureView,
    t_debug: blade_graphics::TextureView,
    tone_map_params: ToneMapParams,
    debug_params: DebugParams,
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
    normal_texture: u32,
    // make sure the end of the struct is aligned
    finish_pad: [u32; 1],
}

#[derive(Clone, PartialEq)]
pub struct Shaders {
    env_prepare: blade_asset::Handle<crate::Shader>,
    fill_gbuf: blade_asset::Handle<crate::Shader>,
    ray_trace: blade_asset::Handle<crate::Shader>,
    blur: blade_asset::Handle<crate::Shader>,
    post_proc: blade_asset::Handle<crate::Shader>,
    debug_draw: blade_asset::Handle<crate::Shader>,
    debug_blit: blade_asset::Handle<crate::Shader>,
}

impl Shaders {
    pub fn load(path: &Path, asset_hub: &crate::AssetHub) -> (Self, choir::RunningTask) {
        let mut ctx = asset_hub.open_context(path, "shader finish");
        let shaders = Self {
            env_prepare: ctx.load_shader("env-prepare.wgsl"),
            fill_gbuf: ctx.load_shader("fill-gbuf.wgsl"),
            ray_trace: ctx.load_shader("ray-trace.wgsl"),
            blur: ctx.load_shader("blur.wgsl"),
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
    temporal_accum: blade_graphics::ComputePipeline,
    atrous: blade_graphics::ComputePipeline,
    post_proc: blade_graphics::RenderPipeline,
    debug_draw: blade_graphics::RenderPipeline,
    debug_blit: blade_graphics::RenderPipeline,
    env_prepare: blade_graphics::ComputePipeline,
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
        shader.check_struct_size::<DebugEntry>();
        let layout = <MainData as blade_graphics::ShaderData>::layout();
        gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "ray-trace",
            data_layouts: &[&layout],
            compute: shader.at("main"),
        })
    }

    fn create_temporal_accum(
        shader: &blade_graphics::Shader,
        gpu: &blade_graphics::Context,
    ) -> blade_graphics::ComputePipeline {
        let layout = <TemporalAccumData as blade_graphics::ShaderData>::layout();
        gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "temporal-accum",
            data_layouts: &[&layout],
            compute: shader.at("temporal_accum"),
        })
    }

    fn create_atrous(
        shader: &blade_graphics::Shader,
        gpu: &blade_graphics::Context,
    ) -> blade_graphics::ComputePipeline {
        let layout = <AtrousData as blade_graphics::ShaderData>::layout();
        gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "atrous",
            data_layouts: &[&layout],
            compute: shader.at("atrous3x3"),
        })
    }

    fn create_post_proc(
        shader: &blade_graphics::Shader,
        format: blade_graphics::TextureFormat,
        gpu: &blade_graphics::Context,
    ) -> blade_graphics::RenderPipeline {
        let layout = <PostProcData as blade_graphics::ShaderData>::layout();
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
        let sh_main = shader_man[shaders.ray_trace].raw.as_ref().unwrap();
        let sh_blur = shader_man[shaders.blur].raw.as_ref().unwrap();
        Ok(Self {
            fill: Self::create_gbuf_fill(shader_man[shaders.fill_gbuf].raw.as_ref().unwrap(), gpu),
            main: Self::create_ray_trace(sh_main, gpu),
            temporal_accum: Self::create_temporal_accum(sh_blur, gpu),
            atrous: Self::create_atrous(sh_blur, gpu),
            post_proc: Self::create_post_proc(
                shader_man[shaders.post_proc].raw.as_ref().unwrap(),
                config.surface_format,
                gpu,
            ),
            debug_draw: Self::create_debug_draw(
                shader_man[shaders.debug_draw].raw.as_ref().unwrap(),
                config.surface_format,
                gpu,
            ),
            debug_blit: Self::create_debug_blit(
                shader_man[shaders.debug_blit].raw.as_ref().unwrap(),
                config.surface_format,
                gpu,
            ),
            env_prepare: EnvironmentMap::init_pipeline(
                shader_man[shaders.env_prepare].raw.as_ref().unwrap(),
                gpu,
            )?,
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
    #[profiling::function]
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

        let debug = DebugRender {
            _capacity: config.max_debug_lines,
            buffer: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "debug",
                size: (sp.debug_buffer_size + (config.max_debug_lines - 1) * sp.debug_line_size)
                    as u64,
                memory: blade_graphics::Memory::Device,
            }),
            variance_buffer: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "variance",
                size: mem::size_of::<DebugVariance>() as u64,
                memory: blade_graphics::Memory::Shared,
            }),
            entry_buffer: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "debug entry",
                size: mem::size_of::<DebugEntry>() as u64,
                memory: blade_graphics::Memory::Shared,
            }),
            draw_pipeline: sp.debug_draw,
            blit_pipeline: sp.debug_blit,
            line_size: sp.debug_line_size,
            buffer_size: sp.debug_buffer_size,
        };

        let debug_init_data = [2u32, 0, 0, config.max_debug_lines];
        let debug_init_size = debug_init_data.len() * mem::size_of::<u32>();
        assert!(debug_init_size <= mem::size_of::<DebugEntry>());
        unsafe {
            ptr::write_bytes(
                debug.variance_buffer.data(),
                0,
                mem::size_of::<DebugVariance>(),
            );
            ptr::write_bytes(debug.entry_buffer.data(), 0, mem::size_of::<DebugEntry>());
            // piggyback on the staging buffers to upload the data
            ptr::copy_nonoverlapping(
                debug_init_data.as_ptr(),
                debug.entry_buffer.data() as *mut u32,
                debug_init_data.len(),
            );
        }
        {
            let mut transfers = encoder.transfer();
            transfers.copy_buffer_to_buffer(
                debug.entry_buffer.at(0),
                debug.buffer.at(0),
                debug_init_size as u64,
            );
        }

        let frame_data = [
            FrameData::new(config.screen_size, sp.reservoir_size, encoder, gpu),
            FrameData::new(config.screen_size, sp.reservoir_size, encoder, gpu),
        ];
        let (light_temp_texture, light_temp_view) =
            FrameData::create_target("light-temp", RADIANCE_FORMAT, config.screen_size, gpu);
        encoder.init_texture(light_temp_texture);

        let dummy = DummyResources::new(encoder, gpu);
        let (debug_texture, debug_view) = FrameData::create_target(
            "debug",
            blade_graphics::TextureFormat::Rgba8Unorm,
            config.screen_size,
            gpu,
        );
        encoder.init_texture(debug_texture);

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
            light_temp_texture,
            light_temp_view,
            post_proc_input: blade_graphics::TextureView::default(),
            debug_texture,
            debug_view,
            fill_pipeline: sp.fill,
            main_pipeline: sp.main,
            post_proc_pipeline: sp.post_proc,
            blur: Blur {
                temporal_accum_pipeline: sp.temporal_accum,
                atrous_pipeline: sp.atrous,
            },
            acceleration_structure: blade_graphics::AccelerationStructure::default(),
            env_map: EnvironmentMap::with_pipeline(&dummy, sp.env_prepare),
            dummy,
            hit_buffer: blade_graphics::Buffer::default(),
            vertex_buffers: blade_graphics::BufferArray::new(),
            index_buffers: blade_graphics::BufferArray::new(),
            textures: blade_graphics::TextureArray::new(),
            samplers,
            reservoir_size: sp.reservoir_size,
            debug,
            screen_size: config.screen_size,
            frame_index: 0,
            texture_resource_lookup: HashMap::default(),
        }
    }

    /// Destroy all internally managed GPU resources.
    pub fn destroy(&mut self, gpu: &blade_graphics::Context) {
        // internal resources
        for frame_data in self.frame_data.iter_mut() {
            frame_data.destroy(gpu);
        }
        gpu.destroy_texture(self.light_temp_texture);
        gpu.destroy_texture_view(self.light_temp_view);
        gpu.destroy_texture(self.debug_texture);
        gpu.destroy_texture_view(self.debug_view);
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
        gpu.destroy_buffer(self.debug.entry_buffer);
    }

    #[profiling::function]
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
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.blur));
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
            if let Ok(ref shader) = asset_hub.shaders[self.shaders.fill_gbuf].raw {
                self.fill_pipeline = ShaderPipelines::create_gbuf_fill(shader, gpu);
            }
        }
        if self.shaders.ray_trace != old.ray_trace {
            if let Ok(ref shader) = asset_hub.shaders[self.shaders.ray_trace].raw {
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
        }
        if self.shaders.blur != old.blur {
            if let Ok(ref shader) = asset_hub.shaders[self.shaders.blur].raw {
                self.blur.temporal_accum_pipeline =
                    ShaderPipelines::create_temporal_accum(shader, gpu);
                self.blur.atrous_pipeline = ShaderPipelines::create_atrous(shader, gpu);
            }
        }
        if self.shaders.post_proc != old.post_proc {
            if let Ok(ref shader) = asset_hub.shaders[self.shaders.post_proc].raw {
                self.post_proc_pipeline =
                    ShaderPipelines::create_post_proc(shader, self.config.surface_format, gpu);
            }
        }
        if self.shaders.debug_draw != old.debug_draw {
            if let Ok(ref shader) = asset_hub.shaders[self.shaders.debug_draw].raw {
                self.debug.draw_pipeline =
                    ShaderPipelines::create_debug_draw(shader, self.config.surface_format, gpu);
            }
        }
        if self.shaders.debug_blit != old.debug_blit {
            if let Ok(ref shader) = asset_hub.shaders[self.shaders.debug_blit].raw {
                self.debug.blit_pipeline =
                    ShaderPipelines::create_debug_blit(shader, self.config.surface_format, gpu);
            }
        }

        true
    }

    pub fn get_screen_size(&self) -> blade_graphics::Extent {
        self.screen_size
    }

    pub fn view_dummy_white(&self) -> blade_graphics::TextureView {
        self.dummy.white_view
    }
    pub fn view_environment_main(&self) -> blade_graphics::TextureView {
        self.env_map.main_view
    }
    pub fn view_environment_weight(&self) -> blade_graphics::TextureView {
        self.env_map.weight_view
    }

    #[profiling::function]
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

        gpu.destroy_texture(self.light_temp_texture);
        gpu.destroy_texture_view(self.light_temp_view);
        let (light_temp_texture, light_temp_view) =
            FrameData::create_target("light-temp", RADIANCE_FORMAT, size, gpu);
        encoder.init_texture(light_temp_texture);
        self.light_temp_texture = light_temp_texture;
        self.light_temp_view = light_temp_view;

        gpu.destroy_texture(self.debug_texture);
        gpu.destroy_texture_view(self.debug_view);
        let (debug_texture, debug_view) = FrameData::create_target(
            "debug",
            blade_graphics::TextureFormat::Rgba8Unorm,
            size,
            gpu,
        );
        encoder.init_texture(debug_texture);
        self.debug_texture = debug_texture;
        self.debug_view = debug_view;
    }

    #[profiling::function]
    pub fn build_scene(
        &mut self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        objects: &[crate::Object],
        env_map: Option<blade_asset::Handle<crate::Texture>>,
        asset_hub: &crate::AssetHub,
        gpu: &blade_graphics::Context,
        temp_buffers: &mut Vec<blade_graphics::Buffer>,
        temp_acceleration_structures: &mut Vec<blade_graphics::AccelerationStructure>,
    ) {
        let (env_view, env_extent) = match env_map {
            Some(handle) => {
                let asset = &asset_hub.textures[handle];
                (asset.view, asset.extent)
            }
            None => (self.dummy.white_view, blade_graphics::Extent::default()),
        };
        self.env_map
            .assign(env_view, env_extent, command_encoder, gpu);

        if self.acceleration_structure != blade_graphics::AccelerationStructure::default() {
            temp_acceleration_structures.push(self.acceleration_structure);
        }

        let geometry_count = objects
            .iter()
            .map(|object| {
                let model = &asset_hub.models[object.model];
                model.geometries.len()
            })
            .sum::<usize>();
        let hit_size = (geometry_count.max(1) * mem::size_of::<HitEntry>()) as u64;
        //TODO: reuse the hit buffer
        if self.hit_buffer != blade_graphics::Buffer::default() {
            temp_buffers.push(self.hit_buffer);
        }
        self.hit_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "hit entries",
            size: hit_size,
            memory: blade_graphics::Memory::Device,
        });
        let hit_staging = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "hit staging",
            size: hit_size,
            memory: blade_graphics::Memory::Upload,
        });
        temp_buffers.push(hit_staging);
        {
            let mut transfers = command_encoder.transfer();
            transfers.copy_buffer_to_buffer(hit_staging.at(0), self.hit_buffer.at(0), hit_size);
        }

        self.vertex_buffers.clear();
        self.index_buffers.clear();
        self.textures.clear();
        let dummy_white = self.textures.alloc(self.dummy.white_view);
        let dummy_black = self.textures.alloc(self.dummy.black_view);

        let mut geometry_index = 0;
        let mut instances = Vec::with_capacity(objects.len());
        let mut blases = Vec::with_capacity(objects.len());
        let mut texture_indices = HashMap::new();

        for object in objects {
            let m3_object = glam::Mat3 {
                x_axis: glam::Vec4::from(object.transform.x).truncate(),
                y_axis: glam::Vec4::from(object.transform.y).truncate(),
                z_axis: glam::Vec4::from(object.transform.z).truncate(),
            };

            let model = &asset_hub.models[object.model];
            instances.push(blade_graphics::AccelerationStructureInstance {
                acceleration_structure_index: blases.len() as u32,
                transform: object.transform,
                mask: 0xFF,
                custom_index: geometry_index as u32,
            });
            blases.push(model.acceleration_structure);

            for geometry in model.geometries.iter() {
                let material = &model.materials[geometry.material_index];
                let vertex_offset =
                    geometry.vertex_range.start as u64 * mem::size_of::<crate::Vertex>() as u64;
                let geometry_to_world_rotation = {
                    let colm = mint::ColumnMatrix3x4::from(geometry.transform);
                    let m3_geo = glam::Mat3 {
                        x_axis: colm.x.into(),
                        y_axis: colm.y.into(),
                        z_axis: colm.z.into(),
                    };
                    let m3_normal = (m3_object * m3_geo).inverse().transpose();
                    let quat = glam::Quat::from_mat3(&m3_normal);
                    let qv = glam::Vec4::from(quat) * 127.0;
                    [qv.x as i8, qv.y as i8, qv.z as i8, qv.w as i8]
                };

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
                    geometry_to_object: mint::ColumnMatrix4::from(mint::RowMatrix4 {
                        x: geometry.transform.x,
                        y: geometry.transform.y,
                        z: geometry.transform.z,
                        w: [0.0, 0.0, 0.0, 1.0].into(),
                    }),
                    base_color_texture: match material.base_color_texture {
                        Some(handle) => *texture_indices.entry(handle).or_insert_with(|| {
                            let texture = &asset_hub.textures[handle];
                            self.textures.alloc(texture.view)
                        }),
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
                    normal_texture: match material.normal_texture {
                        Some(handle) => *texture_indices.entry(handle).or_insert_with(|| {
                            let texture = &asset_hub.textures[handle];
                            self.textures.alloc(texture.view)
                        }),
                        None => dummy_black,
                    },
                    finish_pad: [0; 1],
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

        self.texture_resource_lookup.clear();
        for (handle, res_id) in texture_indices {
            self.texture_resource_lookup.insert(res_id, handle);
        }

        assert_eq!(geometry_index, geometry_count);
        log::info!(
            "Preparing ray tracing with {} geometries in total",
            geometry_count
        );

        // Needs to be a separate encoder in order to force synchronization
        let sizes = gpu.get_top_level_acceleration_structure_sizes(instances.len() as u32);
        self.acceleration_structure =
            gpu.create_acceleration_structure(blade_graphics::AccelerationStructureDesc {
                name: "TLAS",
                ty: blade_graphics::AccelerationStructureType::TopLevel,
                size: sizes.data,
            });
        let instance_buf = gpu.create_acceleration_structure_instance_buffer(&instances, &blases);
        let scratch_buf = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "TLAS scratch",
            size: sizes.scratch,
            memory: blade_graphics::Memory::Device,
        });

        let mut tlas_encoder = command_encoder.acceleration_structure();
        tlas_encoder.build_top_level(
            self.acceleration_structure,
            &blases,
            instances.len() as u32,
            instance_buf.at(0),
            scratch_buf.at(0),
        );

        temp_buffers.push(instance_buf);
        temp_buffers.push(scratch_buf);
    }

    /// Prepare to render a frame.
    #[profiling::function]
    pub fn prepare(
        &mut self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        camera: &crate::Camera,
        enable_debug_draw: bool,
        accumulate_variance: bool,
        reset_reservoirs: bool,
    ) {
        let mut transfer = command_encoder.transfer();

        if enable_debug_draw {
            // reset the debug line count
            transfer.fill_buffer(self.debug.buffer.at(4), 4, 0);
            transfer.fill_buffer(self.debug.buffer.at(20), 4, 1);
        } else {
            transfer.fill_buffer(self.debug.buffer.at(20), 4, 0);
        }

        if reset_reservoirs || !accumulate_variance {
            transfer.fill_buffer(
                self.debug.buffer.at(32),
                mem::size_of::<DebugVariance>() as u64,
                0,
            );
        } else {
            // copy the previous frame variance
            transfer.copy_buffer_to_buffer(
                self.debug.buffer.at(32),
                self.debug.variance_buffer.into(),
                mem::size_of::<DebugVariance>() as u64,
            );
        }
        transfer.copy_buffer_to_buffer(
            self.debug
                .buffer
                .at(32 + mem::size_of::<DebugVariance>() as u64),
            self.debug.entry_buffer.into(),
            mem::size_of::<DebugEntry>() as u64,
        );

        if reset_reservoirs {
            if !enable_debug_draw {
                transfer.fill_buffer(self.debug.buffer.at(4), 4, 0);
            }
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
        self.post_proc_input = self.frame_data[0].light_diffuse_view;
    }

    fn make_debug_params(&self, config: &DebugConfig) -> DebugParams {
        DebugParams {
            view_mode: config.view_mode as u32,
            draw_flags: config.draw_flags.bits(),
            texture_flags: config.texture_flags.bits(),
            unused: 0,
            mouse_pos: match config.mouse_pos {
                Some(p) => [p[0], self.screen_size.height as i32 - p[1]],
                None => [-1; 2],
            },
        }
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
    #[profiling::function]
    pub fn ray_trace(
        &self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        debug_config: DebugConfig,
        ray_config: RayConfig,
    ) {
        let debug = self.make_debug_params(&debug_config);
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
                    out_flat_normal: cur.flat_normal_view,
                    out_albedo: cur.albedo_view,
                    out_debug: self.debug_view,
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
                    t_flat_normal: cur.flat_normal_view,
                    t_prev_flat_normal: prev.flat_normal_view,
                    debug_buf: self.debug.buffer.into(),
                    reservoirs: cur.reservoir_buf.into(),
                    prev_reservoirs: prev.reservoir_buf.into(),
                    out_diffuse: cur.light_diffuse_view,
                    out_debug: self.debug_view,
                },
            );
            pc.dispatch(groups);
        }
    }

    /// Perform noise reduction using SVGF.
    #[profiling::function]
    pub fn denoise(
        &mut self, //TODO: borrow immutably
        command_encoder: &mut blade_graphics::CommandEncoder,
        denoiser_config: DenoiserConfig,
    ) {
        let mut params = BlurParams {
            extent: [self.screen_size.width, self.screen_size.height],
            temporal_weight: denoiser_config.temporal_weight,
            iteration: 0,
        };
        if denoiser_config.temporal_weight < 1.0 {
            let cur = self.frame_data.first().unwrap();
            let prev = self.frame_data.last().unwrap();
            if let mut pass = command_encoder.compute() {
                let mut pc = pass.with(&self.blur.temporal_accum_pipeline);
                let groups = self.blur.atrous_pipeline.get_dispatch_for(self.screen_size);
                pc.bind(
                    0,
                    &TemporalAccumData {
                        camera: cur.camera_params,
                        prev_camera: prev.camera_params,
                        params,
                        input: cur.light_diffuse_view,
                        prev_input: prev.light_diffuse_view,
                        t_depth: cur.depth_view,
                        t_prev_depth: prev.depth_view,
                        t_flat_normal: cur.flat_normal_view,
                        t_prev_flat_normal: prev.flat_normal_view,
                        output: self.light_temp_view,
                    },
                );
                pc.dispatch(groups);
            }

            // make it so `cur.light_diffuse_view` always contains the fresh reprojection result
            let cur_mut = self.frame_data.first_mut().unwrap();
            mem::swap(&mut self.light_temp_view, &mut cur_mut.light_diffuse_view);
            mem::swap(&mut self.light_temp_texture, &mut cur_mut.light_diffuse);
        }

        {
            let cur = self.frame_data.first().unwrap();
            let prev = self.frame_data.last().unwrap();
            self.post_proc_input = cur.light_diffuse_view;
            //Note: we no longer need `prev.light_diffuse_view` so reusing it here
            let mut targets = [self.light_temp_view, prev.light_diffuse_view];
            for _ in 0..denoiser_config.num_passes {
                if let mut pass = command_encoder.compute() {
                    let mut pc = pass.with(&self.blur.atrous_pipeline);
                    let groups = self.blur.atrous_pipeline.get_dispatch_for(self.screen_size);
                    pc.bind(
                        0,
                        &AtrousData {
                            params,
                            input: self.post_proc_input,
                            t_flat_normal: cur.flat_normal_view,
                            t_depth: cur.depth_view,
                            output: targets[0],
                        },
                    );
                    pc.dispatch(groups);
                    self.post_proc_input = targets[0];
                    targets.swap(0, 1); // rotate the views
                    params.iteration += 1;
                }
            }
        }
    }

    /// Blit the rendering result into a specified render pass.
    #[profiling::function]
    pub fn post_proc(
        &self,
        pass: &mut blade_graphics::RenderCommandEncoder,
        debug_config: DebugConfig,
        pp_config: PostProcConfig,
        debug_blits: &[DebugBlit],
    ) {
        let cur = self.frame_data.first().unwrap();
        if let mut pc = pass.with(&self.post_proc_pipeline) {
            let debug_params = self.make_debug_params(&debug_config);
            pc.bind(
                0,
                &PostProcData {
                    t_albedo: cur.albedo_view,
                    light_diffuse: self.post_proc_input,
                    t_debug: self.debug_view,
                    tone_map_params: ToneMapParams {
                        enabled: 1,
                        average_lum: pp_config.average_luminocity,
                        key_value: pp_config.exposure_key_value,
                        white_level: pp_config.white_level,
                    },
                    debug_params,
                },
            );
            pc.draw(0, 3, 0, 1);
        }
        if let mut pc = pass.with(&self.debug.draw_pipeline) {
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
            for db in debug_blits {
                pc.bind(
                    0,
                    &DebugBlitData {
                        input: db.input,
                        samp: self.samplers.linear,
                        params: DebugBlitParams {
                            target_offset: [
                                db.target_offset[0] as f32 / self.screen_size.width as f32,
                                db.target_offset[1] as f32 / self.screen_size.height as f32,
                            ],
                            target_size: [
                                db.target_size[0] as f32 / self.screen_size.width as f32,
                                db.target_size[1] as f32 / self.screen_size.height as f32,
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

    #[profiling::function]
    pub fn read_debug_selection_info(&self) -> SelectionInfo {
        let db_v = unsafe { &*(self.debug.variance_buffer.data() as *const DebugVariance) };
        let db_e = unsafe { &*(self.debug.entry_buffer.data() as *const DebugEntry) };
        SelectionInfo {
            std_deviation: if db_v.count == 0 {
                [0.0; 3].into()
            } else {
                let sum_avg = glam::Vec3::from(db_v.color_sum) / (db_v.count as f32);
                let sum2_avg = glam::Vec3::from(db_v.color2_sum) / (db_v.count as f32);
                let variance = sum2_avg - sum_avg * sum_avg;
                mint::Vector3 {
                    x: variance.x.sqrt(),
                    y: variance.y.sqrt(),
                    z: variance.z.sqrt(),
                }
            },
            std_deviation_history: db_v.count,
            custom_index: db_e.custom_index,
            depth: db_e.depth,
            tex_coords: db_e.tex_coords.into(),
            base_color_texture: self
                .texture_resource_lookup
                .get(&db_e.base_color_texture)
                .cloned(),
            normal_texture: self
                .texture_resource_lookup
                .get(&db_e.normal_texture)
                .cloned(),
        }
    }
}
