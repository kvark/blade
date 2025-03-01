mod debug;
mod dummy;
mod env_map;

use debug::{DebugEntry, DebugRender, DebugVariance};

pub use debug::{DebugBlit, DebugLine, DebugPoint};
pub use dummy::DummyResources;
pub use env_map::EnvironmentMap;

use blade_graphics::Memory;
use std::{collections::HashMap, mem, num::NonZeroU32, path::Path, ptr};

const MAX_RESOURCES: u32 = 8192;
const RADIANCE_FORMAT: blade_graphics::TextureFormat = blade_graphics::TextureFormat::Rgba16Float;

fn mat4_transform(t: &blade_graphics::Transform) -> glam::Mat4 {
    glam::Mat4 {
        x_axis: t.x.into(),
        y_axis: t.y.into(),
        z_axis: t.z.into(),
        w_axis: glam::Vec4::W,
    }
    .transpose()
}
fn mat3_transform(t_orig: &blade_graphics::Transform) -> glam::Mat3 {
    let t = mint::ColumnMatrix3x4::from(*t_orig);
    glam::Mat3 {
        x_axis: t.x.into(),
        y_axis: t.y.into(),
        z_axis: t.z.into(),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    pub surface_size: blade_graphics::Extent,
    pub surface_info: blade_graphics::SurfaceInfo,
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
    DiffuseAlbedoTexture = 2,
    DiffuseAlbedoFactor = 3,
    NormalTexture = 4,
    NormalScale = 5,
    GeometryNormal = 6,
    ShadingNormal = 7,
    Motion = 8,
    HitConsistency = 9,
    SampleReuse = 10,
    Variance = 15,
}

impl Default for DebugMode {
    fn default() -> Self {
        Self::Final
    }
}

bitflags::bitflags! {
    #[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq, PartialOrd)]
    pub struct DebugDrawFlags: u32 {
        const SPACE = 1;
        const GEOMETRY = 2;
        const RESTIR = 4;
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
    pub tap_count: u32,
    pub tap_radius: u32,
    pub tap_confidence_near: u32,
    pub tap_confidence_far: u32,
    pub t_start: f32,
    /// Evaluate MIS factor for ReSTIR in a pair-wise fashion.
    /// Adds 2 extra visibility rays per reused sample.
    pub pairwise_mis: bool,
    /// Defensive MIS factor for the canonical sample.
    /// Can be between 0 and 1.
    pub defensive_mis: f32,
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
    pub position: mint::Vector3<f32>,
    pub normal: mint::Vector3<f32>,
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
            position: [0.0; 3].into(),
            normal: [0.0; 3].into(),
            tex_coords: [0.0; 2].into(),
            base_color_texture: None,
            normal_texture: None,
        }
    }
}

struct RenderTarget<const N: usize> {
    texture: blade_graphics::Texture,
    views: [blade_graphics::TextureView; N],
}
impl<const N: usize> RenderTarget<N> {
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
            array_layer_count: N as u32,
            mip_level_count: 1,
            usage: blade_graphics::TextureUsage::RESOURCE | blade_graphics::TextureUsage::STORAGE,
            sample_count: 1,
            external: None,
        });
        encoder.init_texture(texture);

        let mut views = [blade_graphics::TextureView::default(); N];
        for (i, view) in views.iter_mut().enumerate() {
            *view = gpu.create_texture_view(
                texture,
                blade_graphics::TextureViewDesc {
                    name: &format!("{name}{i}"),
                    format,
                    dimension: blade_graphics::ViewDimension::D2,
                    subresources: &blade_graphics::TextureSubresources {
                        base_array_layer: i as u32,
                        array_layer_count: NonZeroU32::new(1),
                        ..Default::default()
                    },
                },
            );
        }

        Self { texture, views }
    }

    fn destroy(&self, gpu: &blade_graphics::Context) {
        gpu.destroy_texture(self.texture);
        for view in self.views.iter() {
            gpu.destroy_texture_view(*view);
        }
    }
}

struct RestirTargets {
    reservoir_buf: [blade_graphics::Buffer; 2],
    debug: RenderTarget<1>,
    depth: RenderTarget<2>,
    basis: RenderTarget<2>,
    flat_normal: RenderTarget<2>,
    albedo: RenderTarget<1>,
    motion: RenderTarget<1>,
    light_diffuse: RenderTarget<3>,
    camera_params: [CameraParams; 2],
}

impl RestirTargets {
    fn new(
        size: blade_graphics::Extent,
        reservoir_size: u32,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
    ) -> Self {
        let total_reservoirs = size.width as usize * size.height as usize;
        let mut reservoir_buf = [blade_graphics::Buffer::default(); 2];
        for (i, rb) in reservoir_buf.iter_mut().enumerate() {
            *rb = gpu.create_buffer(blade_graphics::BufferDesc {
                name: &format!("reservoirs{i}"),
                size: reservoir_size as u64 * total_reservoirs as u64,
                memory: blade_graphics::Memory::Device,
            });
        }

        Self {
            reservoir_buf,
            debug: RenderTarget::new(
                "debug",
                blade_graphics::TextureFormat::Rgba8Unorm,
                size,
                encoder,
                gpu,
            ),
            depth: RenderTarget::new(
                "depth",
                blade_graphics::TextureFormat::R32Float,
                size,
                encoder,
                gpu,
            ),
            basis: RenderTarget::new(
                "basis",
                blade_graphics::TextureFormat::Rgba8Snorm,
                size,
                encoder,
                gpu,
            ),
            flat_normal: RenderTarget::new(
                "flat-normal",
                blade_graphics::TextureFormat::Rgba8Snorm,
                size,
                encoder,
                gpu,
            ),
            albedo: RenderTarget::new(
                "albedo",
                blade_graphics::TextureFormat::Rgba8Unorm,
                size,
                encoder,
                gpu,
            ),
            motion: RenderTarget::new(
                "motion",
                blade_graphics::TextureFormat::Rg8Snorm,
                size,
                encoder,
                gpu,
            ),
            light_diffuse: RenderTarget::new("light-diffuse", RADIANCE_FORMAT, size, encoder, gpu),
            camera_params: [CameraParams::default(); 2],
        }
    }

    fn destroy(&self, gpu: &blade_graphics::Context) {
        for rb in self.reservoir_buf.iter() {
            gpu.destroy_buffer(*rb);
        }
        self.debug.destroy(gpu);
        self.depth.destroy(gpu);
        self.basis.destroy(gpu);
        self.flat_normal.destroy(gpu);
        self.albedo.destroy(gpu);
        self.motion.destroy(gpu);
        self.light_diffuse.destroy(gpu);
    }
}

struct Blur {
    temporal_accum_pipeline: blade_graphics::ComputePipeline,
    a_trous_pipeline: blade_graphics::ComputePipeline,
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
    shaders: Shaders,
    targets: RestirTargets,
    post_proc_input_index: usize,
    fill_pipeline: blade_graphics::ComputePipeline,
    main_pipeline: blade_graphics::ComputePipeline,
    post_proc_pipeline: blade_graphics::RenderPipeline,
    blur: Blur,
    acceleration_structure: blade_graphics::AccelerationStructure,
    prev_acceleration_structure: blade_graphics::AccelerationStructure,
    env_map: EnvironmentMap,
    dummy: DummyResources,
    hit_buffer: blade_graphics::Buffer,
    vertex_buffers: blade_graphics::BufferArray<MAX_RESOURCES>,
    index_buffers: blade_graphics::BufferArray<MAX_RESOURCES>,
    textures: blade_graphics::TextureArray<MAX_RESOURCES>,
    samplers: Samplers,
    reservoir_size: u32,
    debug: DebugRender,
    surface_size: blade_graphics::Extent,
    surface_info: blade_graphics::SurfaceInfo,
    frame_index: usize,
    frame_scene_built: usize,
    is_frozen: bool,
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
    tap_count: u32,
    tap_radius: f32,
    tap_confidence_near: f32,
    tap_confidence_far: f32,
    t_start: f32,
    use_pairwise_mis: u32,
    defensive_mis: f32,
    use_motion_vectors: u32,
}

#[derive(blade_macros::ShaderData)]
struct FillData<'a> {
    camera: CameraParams,
    prev_camera: CameraParams,
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
    out_motion: blade_graphics::TextureView,
    out_debug: blade_graphics::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct MainData {
    camera: CameraParams,
    prev_camera: CameraParams,
    debug: DebugParams,
    parameters: MainParams,
    acc_struct: blade_graphics::AccelerationStructure,
    prev_acc_struct: blade_graphics::AccelerationStructure,
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
    t_motion: blade_graphics::TextureView,
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
    use_motion_vectors: u32,
    pad: u32,
}

#[derive(blade_macros::ShaderData)]
struct TemporalAccumData {
    camera: CameraParams,
    prev_camera: CameraParams,
    params: BlurParams,
    input: blade_graphics::TextureView,
    t_depth: blade_graphics::TextureView,
    t_prev_depth: blade_graphics::TextureView,
    t_flat_normal: blade_graphics::TextureView,
    t_prev_flat_normal: blade_graphics::TextureView,
    t_motion: blade_graphics::TextureView,
    output: blade_graphics::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct ATrousData {
    params: BlurParams,
    input: blade_graphics::TextureView,
    t_depth: blade_graphics::TextureView,
    t_flat_normal: blade_graphics::TextureView,
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

#[repr(C)]
#[derive(Debug)]
struct HitEntry {
    index_buf: u32,
    vertex_buf: u32,
    winding: f32,
    geometry_to_world_rotation: [i8; 4],
    //Note: it's technically `mat4x3` on WGSL side,
    // but it's aligned and sized the same way as `mat4`.
    geometry_to_object: mint::ColumnMatrix4<f32>,
    prev_object_to_world: mint::ColumnMatrix4<f32>,
    base_color_texture: u32,
    base_color_factor: [u8; 4],
    normal_texture: u32,
    normal_scale: f32,
}

#[derive(Clone, PartialEq)]
pub struct Shaders {
    env_prepare: blade_asset::Handle<crate::Shader>,
    fill_gbuf: blade_asset::Handle<crate::Shader>,
    ray_trace: blade_asset::Handle<crate::Shader>,
    a_trous: blade_asset::Handle<crate::Shader>,
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
            a_trous: ctx.load_shader("a-trous.wgsl"),
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
    a_trous: blade_graphics::ComputePipeline,
    post_proc: blade_graphics::RenderPipeline,
    env_prepare: blade_graphics::ComputePipeline,
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

    fn create_a_trous(
        shader: &blade_graphics::Shader,
        gpu: &blade_graphics::Context,
    ) -> blade_graphics::ComputePipeline {
        let layout = <ATrousData as blade_graphics::ShaderData>::layout();
        gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "a-trous",
            data_layouts: &[&layout],
            compute: shader.at("atrous3x3"),
        })
    }

    fn create_post_proc(
        shader: &blade_graphics::Shader,
        info: blade_graphics::SurfaceInfo,
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
            vertex: shader.at("postfx_vs"),
            vertex_fetches: &[],
            fragment: Some(shader.at("postfx_fs")),
            color_targets: &[info.format.into()],
            depth_stencil: None,
            multisample_state: blade_graphics::MultisampleState::default(),
        })
    }

    fn init(
        shaders: &Shaders,
        config: &RenderConfig,
        gpu: &blade_graphics::Context,
        shader_man: &blade_asset::AssetManager<crate::shader::Baker>,
    ) -> Result<Self, &'static str> {
        let sh_main = shader_man[shaders.ray_trace].raw.as_ref().unwrap();
        let sh_a_trous = shader_man[shaders.a_trous].raw.as_ref().unwrap();
        Ok(Self {
            fill: Self::create_gbuf_fill(shader_man[shaders.fill_gbuf].raw.as_ref().unwrap(), gpu),
            main: Self::create_ray_trace(sh_main, gpu),
            temporal_accum: Self::create_temporal_accum(sh_a_trous, gpu),
            a_trous: Self::create_a_trous(sh_a_trous, gpu),
            post_proc: Self::create_post_proc(
                shader_man[shaders.post_proc].raw.as_ref().unwrap(),
                config.surface_info,
                gpu,
            ),
            env_prepare: EnvironmentMap::init_pipeline(
                shader_man[shaders.env_prepare].raw.as_ref().unwrap(),
                gpu,
            )?,
            reservoir_size: sh_main.get_struct_size("StoredReservoir"),
        })
    }
}

#[derive(Clone, Copy, Default)]
pub struct FrameConfig {
    pub frozen: bool,
    pub debug_draw: bool,
    pub reset_variance: bool,
    pub reset_reservoirs: bool,
}

/// Temporary resources associated with a GPU frame.
#[derive(Default)]
pub struct FrameResources {
    pub buffers: Vec<blade_graphics::Buffer>,
    pub acceleration_structures: Vec<blade_graphics::AccelerationStructure>,
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
        let debug = {
            let sh_draw = shader_man[shaders.debug_draw].raw.as_ref().unwrap();
            let sh_blit = shader_man[shaders.debug_blit].raw.as_ref().unwrap();
            DebugRender::init(
                encoder,
                gpu,
                sh_draw,
                sh_blit,
                config.max_debug_lines,
                config.surface_info,
            )
        };

        let targets = RestirTargets::new(config.surface_size, sp.reservoir_size, encoder, gpu);
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
            shaders,
            targets,
            post_proc_input_index: 0,
            fill_pipeline: sp.fill,
            main_pipeline: sp.main,
            post_proc_pipeline: sp.post_proc,
            blur: Blur {
                temporal_accum_pipeline: sp.temporal_accum,
                a_trous_pipeline: sp.a_trous,
            },
            acceleration_structure: blade_graphics::AccelerationStructure::default(),
            prev_acceleration_structure: blade_graphics::AccelerationStructure::default(),
            env_map: EnvironmentMap::with_pipeline(&dummy, sp.env_prepare),
            dummy,
            hit_buffer: blade_graphics::Buffer::default(),
            vertex_buffers: blade_graphics::BufferArray::new(),
            index_buffers: blade_graphics::BufferArray::new(),
            textures: blade_graphics::TextureArray::new(),
            samplers,
            reservoir_size: sp.reservoir_size,
            debug,
            surface_size: config.surface_size,
            surface_info: config.surface_info,
            frame_index: 0,
            frame_scene_built: 0,
            is_frozen: false,
            texture_resource_lookup: HashMap::default(),
        }
    }

    /// Destroy all internally managed GPU resources.
    pub fn destroy(&mut self, gpu: &blade_graphics::Context) {
        // internal resources
        self.targets.destroy(gpu);
        if self.hit_buffer != blade_graphics::Buffer::default() {
            gpu.destroy_buffer(self.hit_buffer);
        }
        gpu.destroy_acceleration_structure(self.acceleration_structure);
        if self.prev_acceleration_structure != blade_graphics::AccelerationStructure::default() {
            gpu.destroy_acceleration_structure(self.prev_acceleration_structure);
        }
        // env map, dummy, and debug
        self.env_map.destroy(gpu);
        self.dummy.destroy(gpu);
        self.debug.destroy(gpu);
        // samplers
        gpu.destroy_sampler(self.samplers.nearest);
        gpu.destroy_sampler(self.samplers.linear);
        // pipelines
        gpu.destroy_compute_pipeline(&mut self.blur.temporal_accum_pipeline);
        gpu.destroy_compute_pipeline(&mut self.blur.a_trous_pipeline);
        gpu.destroy_compute_pipeline(&mut self.fill_pipeline);
        gpu.destroy_compute_pipeline(&mut self.main_pipeline);
        gpu.destroy_render_pipeline(&mut self.post_proc_pipeline);
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
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.a_trous));
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
                assert_eq!(
                    shader.get_struct_size("StoredReservoir"),
                    self.reservoir_size
                );
                self.main_pipeline = ShaderPipelines::create_ray_trace(shader, gpu);
            }
        }
        if self.shaders.a_trous != old.a_trous {
            if let Ok(ref shader) = asset_hub.shaders[self.shaders.a_trous].raw {
                self.blur.temporal_accum_pipeline =
                    ShaderPipelines::create_temporal_accum(shader, gpu);
                self.blur.a_trous_pipeline = ShaderPipelines::create_a_trous(shader, gpu);
            }
        }
        if self.shaders.post_proc != old.post_proc {
            if let Ok(ref shader) = asset_hub.shaders[self.shaders.post_proc].raw {
                self.post_proc_pipeline =
                    ShaderPipelines::create_post_proc(shader, self.surface_info, gpu);
            }
        }
        if self.shaders.debug_draw != old.debug_draw {
            if let Ok(ref shader) = asset_hub.shaders[self.shaders.debug_draw].raw {
                self.debug.recreate_draw_pipeline(shader, gpu);
            }
        }
        if self.shaders.debug_blit != old.debug_blit {
            if let Ok(ref shader) = asset_hub.shaders[self.shaders.debug_blit].raw {
                self.debug.recreate_blit_pipeline(shader, gpu);
            }
        }

        true
    }

    pub fn get_surface_size(&self) -> blade_graphics::Extent {
        self.surface_size
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
        self.surface_size = size;
        self.targets.destroy(gpu);
        self.targets = RestirTargets::new(size, self.reservoir_size, encoder, gpu);
    }

    #[profiling::function]
    pub fn build_scene(
        &mut self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        objects: &[crate::Object],
        env_map: Option<blade_asset::Handle<crate::Texture>>,
        asset_hub: &crate::AssetHub,
        gpu: &blade_graphics::Context,
        temp: &mut FrameResources,
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

        if self.prev_acceleration_structure != blade_graphics::AccelerationStructure::default() {
            temp.acceleration_structures
                .push(self.prev_acceleration_structure);
        }
        self.prev_acceleration_structure = self.acceleration_structure;

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
            temp.buffers.push(self.hit_buffer);
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
        temp.buffers.push(hit_staging);
        {
            let mut transfers = command_encoder.transfer("build-scene");
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
            let m3_object = mat3_transform(&object.transform);
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
                    let m3_geo = mat3_transform(&geometry.transform);
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
                    winding: model.winding,
                    geometry_to_world_rotation,
                    geometry_to_object: mint::ColumnMatrix4::from(mint::RowMatrix4 {
                        x: geometry.transform.x,
                        y: geometry.transform.y,
                        z: geometry.transform.z,
                        w: [0.0, 0.0, 0.0, 1.0].into(),
                    }),
                    prev_object_to_world: mat4_transform(&object.prev_transform).into(),
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
                    normal_scale: material.normal_scale,
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

        let mut tlas_encoder = command_encoder.acceleration_structure("TLAS");
        tlas_encoder.build_top_level(
            self.acceleration_structure,
            &blases,
            instances.len() as u32,
            instance_buf.at(0),
            scratch_buf.at(0),
        );

        temp.buffers.push(instance_buf);
        temp.buffers.push(scratch_buf);
        self.frame_scene_built = self.frame_index + 1;
    }

    fn make_debug_params(&self, config: &DebugConfig) -> DebugParams {
        DebugParams {
            view_mode: config.view_mode as u32,
            draw_flags: config.draw_flags.bits(),
            texture_flags: config.texture_flags.bits(),
            unused: 0,
            mouse_pos: config.mouse_pos.unwrap_or([-1; 2]),
        }
    }

    fn make_camera_params(&self, camera: &super::Camera) -> CameraParams {
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

    fn work_indices(&self) -> (usize, usize) {
        let cur = self.frame_index & 1;
        let prev = if cur < self.frame_index { cur ^ 1 } else { cur };
        (cur, prev)
    }

    /// Prepare to render a frame.
    #[profiling::function]
    pub fn prepare(
        &mut self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        camera: &crate::Camera,
        config: FrameConfig,
    ) {
        let mut transfer = command_encoder.transfer("prepare");

        if config.debug_draw {
            self.debug.reset_lines(&mut transfer);
            self.debug.enable_draw(&mut transfer, true);
        } else {
            self.debug.enable_draw(&mut transfer, false);
        }

        if config.reset_reservoirs || config.reset_variance {
            self.debug.reset_variance(&mut transfer);
        } else {
            self.debug.update_variance(&mut transfer);
        }
        self.debug.update_entry(&mut transfer);

        if config.reset_reservoirs {
            if !config.debug_draw {
                self.debug.reset_lines(&mut transfer);
            }
            let total_reservoirs = self.surface_size.width as u64 * self.surface_size.height as u64;
            for reservoir_buf in self.targets.reservoir_buf.iter() {
                transfer.fill_buffer(
                    reservoir_buf.at(0),
                    total_reservoirs * self.reservoir_size as u64,
                    0,
                );
            }
        }

        if !config.frozen {
            self.frame_index += 1;
        }
        self.is_frozen = config.frozen;
        self.targets.camera_params[self.frame_index % 2] = self.make_camera_params(camera);
        self.post_proc_input_index = self.frame_index % 2;
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
        let (cur, prev) = self.work_indices();
        assert_eq!(cur, self.post_proc_input_index);

        if let mut pass = command_encoder.compute("fill-gbuf") {
            let mut pc = pass.with(&self.fill_pipeline);
            let groups = self.fill_pipeline.get_dispatch_for(self.surface_size);
            pc.bind(
                0,
                &FillData {
                    camera: self.targets.camera_params[cur],
                    prev_camera: self.targets.camera_params[prev],
                    debug,
                    acc_struct: self.acceleration_structure,
                    hit_entries: self.hit_buffer.into(),
                    index_buffers: &self.index_buffers,
                    vertex_buffers: &self.vertex_buffers,
                    textures: &self.textures,
                    sampler_linear: self.samplers.linear,
                    debug_buf: self.debug.buffer_resource(),
                    out_depth: self.targets.depth.views[cur],
                    out_basis: self.targets.basis.views[cur],
                    out_flat_normal: self.targets.flat_normal.views[cur],
                    out_albedo: self.targets.albedo.views[0],
                    out_motion: self.targets.motion.views[0],
                    out_debug: self.targets.debug.views[0],
                },
            );
            pc.dispatch(groups);
        }

        if let mut pass = command_encoder.compute("ray-trace") {
            let mut pc = pass.with(&self.main_pipeline);
            let groups = self.main_pipeline.get_dispatch_for(self.surface_size);
            pc.bind(
                0,
                &MainData {
                    camera: self.targets.camera_params[cur],
                    prev_camera: self.targets.camera_params[prev],
                    debug,
                    parameters: MainParams {
                        frame_index: self.frame_index as u32,
                        num_environment_samples: ray_config.num_environment_samples,
                        environment_importance_sampling: ray_config.environment_importance_sampling
                            as u32,
                        tap_count: ray_config.tap_count,
                        tap_radius: ray_config.tap_radius as f32,
                        tap_confidence_near: ray_config.tap_confidence_near as f32,
                        tap_confidence_far: ray_config.tap_confidence_far as f32,
                        t_start: ray_config.t_start,
                        use_pairwise_mis: ray_config.pairwise_mis as u32,
                        defensive_mis: ray_config.defensive_mis,
                        use_motion_vectors: (self.frame_scene_built >= self.frame_index) as u32,
                    },
                    acc_struct: self.acceleration_structure,
                    prev_acc_struct: if self.frame_scene_built < self.frame_index
                        || self.prev_acceleration_structure
                            == blade_graphics::AccelerationStructure::default()
                    {
                        self.acceleration_structure
                    } else {
                        self.prev_acceleration_structure
                    },
                    sampler_linear: self.samplers.linear,
                    sampler_nearest: self.samplers.nearest,
                    env_map: self.env_map.main_view,
                    env_weights: self.env_map.weight_view,
                    t_depth: self.targets.depth.views[cur],
                    t_prev_depth: self.targets.depth.views[prev],
                    t_basis: self.targets.basis.views[cur],
                    t_prev_basis: self.targets.basis.views[prev],
                    t_flat_normal: self.targets.flat_normal.views[cur],
                    t_prev_flat_normal: self.targets.flat_normal.views[prev],
                    t_motion: self.targets.motion.views[0],
                    debug_buf: self.debug.buffer_resource(),
                    reservoirs: self.targets.reservoir_buf[cur].into(),
                    prev_reservoirs: self.targets.reservoir_buf[prev].into(),
                    out_diffuse: self.targets.light_diffuse.views[cur],
                    out_debug: self.targets.debug.views[0],
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
            extent: [self.surface_size.width, self.surface_size.height],
            temporal_weight: denoiser_config.temporal_weight,
            iteration: 0,
            use_motion_vectors: (self.frame_scene_built >= self.frame_index) as u32,
            pad: 0,
        };
        let (cur, prev) = self.work_indices();

        if denoiser_config.temporal_weight < 1.0 {
            let mut pass = command_encoder.compute("temporal-accum");
            let mut pc = pass.with(&self.blur.temporal_accum_pipeline);
            let groups = self
                .blur
                .a_trous_pipeline
                .get_dispatch_for(self.surface_size);
            pc.bind(
                0,
                &TemporalAccumData {
                    camera: self.targets.camera_params[cur],
                    prev_camera: self.targets.camera_params[prev],
                    params,
                    input: self.targets.light_diffuse.views[prev],
                    t_depth: self.targets.depth.views[cur],
                    t_prev_depth: self.targets.depth.views[prev],
                    t_flat_normal: self.targets.flat_normal.views[cur],
                    t_prev_flat_normal: self.targets.flat_normal.views[prev],
                    t_motion: self.targets.motion.views[0],
                    output: self.targets.light_diffuse.views[cur],
                },
            );
            pc.dispatch(groups);
        }

        assert_eq!(cur, self.post_proc_input_index);
        let mut ping_pong = [2, if self.is_frozen { cur } else { prev }];
        for _ in 0..denoiser_config.num_passes {
            let mut pass = command_encoder.compute("a-trous");
            let mut pc = pass.with(&self.blur.a_trous_pipeline);
            let groups = self
                .blur
                .a_trous_pipeline
                .get_dispatch_for(self.surface_size);
            pc.bind(
                0,
                &ATrousData {
                    params,
                    input: self.targets.light_diffuse.views[self.post_proc_input_index],
                    t_depth: self.targets.depth.views[cur],
                    t_flat_normal: self.targets.flat_normal.views[cur],
                    output: self.targets.light_diffuse.views[ping_pong[0]],
                },
            );
            pc.dispatch(groups);
            self.post_proc_input_index = ping_pong[0];
            ping_pong.swap(0, 1);
            params.iteration += 1;
        }
    }

    /// Blit the rendering result into a specified render pass.
    #[profiling::function]
    pub fn post_proc(
        &self,
        pass: &mut blade_graphics::RenderCommandEncoder,
        debug_config: DebugConfig,
        pp_config: PostProcConfig,
        debug_lines: &[DebugLine],
        debug_blits: &[DebugBlit],
    ) {
        let cur = self.frame_index % 2;
        if let mut pc = pass.with(&self.post_proc_pipeline) {
            let debug_params = self.make_debug_params(&debug_config);
            pc.bind(
                0,
                &PostProcData {
                    t_albedo: self.targets.albedo.views[0],
                    light_diffuse: self.targets.light_diffuse.views[self.post_proc_input_index],
                    t_debug: self.targets.debug.views[0],
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

        self.debug.render_lines(
            debug_lines,
            self.targets.camera_params[cur],
            self.targets.depth.views[cur],
            pass,
        );
        self.debug
            .render_blits(debug_blits, self.samplers.linear, self.surface_size, pass);
    }

    #[profiling::function]
    pub fn read_debug_selection_info(&self) -> SelectionInfo {
        let (db_v, db_e) = self.debug.read_shared_data();
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
            position: db_e.position.into(),
            normal: db_e.normal.into(),
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
