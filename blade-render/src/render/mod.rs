mod debug;

use crate::{
    CameraParams, DebugLine, DummyResources, EnvironmentMap, RenderConfig, Shaders,
    skin::{self, SkinPass},
};
use debug::{DebugEntry, DebugVariance};

pub use debug::DebugBlit;
pub(crate) use debug::DebugRender;

use std::{collections::HashMap, mem, num::NonZeroU32, ptr};

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
struct Samplers {
    nearest: blade_graphics::Sampler,
    linear: blade_graphics::Sampler,
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, PartialOrd, blade_macros::AsPrimitive, strum::EnumIter,
)]
#[repr(u32)]
pub enum DebugMode {
    #[default]
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
    Roughness = 11,
    SpecularF0 = 12,
    Emissive = 13,
    Variance = 15,
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
        const METALLIC_ROUGHNESS = 4;
        const EMISSIVE = 8;
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
    /// Light samples taken at every shading point.
    pub num_environment_samples: u32,
    /// Material samples taken at every shading point.
    ///
    /// The canonical mode continues the path along each of them,
    /// so this is also its number of paths per pixel.
    pub num_brdf_samples: u32,
    /// Randomize the primary ray within each pixel in canonical mode.
    ///
    /// Disable this when radiance must correspond to a separately rasterized
    /// center-sampled G-buffer. Converged reference renders should leave it on
    /// for antialiasing.
    pub jitter_primary_rays: bool,
    /// Sample the environment map by importance rather than uniformly.
    pub environment_importance_sampling: bool,
    /// Number of secondary surfaces a canonical path is allowed to hit.
    ///
    /// The real-time mode ignores this field: it shades the primary surface
    /// from directly visible environment or first-hit emission and does not
    /// estimate indirect transport.
    pub max_bounces: u32,
    pub t_start: f32,
    /// Number of samples to accumulate before going idle, or 0 for no limit.
    ///
    /// Note: only used by the canonical mode.
    pub max_accumulated_samples: u32,
    /// Number of the neighbors to reuse the samples of.
    ///
    /// Note: reuse is only done by the real-time mode.
    pub tap_count: u32,
    pub tap_radius: u32,
    pub tap_confidence_near: u32,
    pub tap_confidence_far: u32,
    /// Evaluate MIS factors for ReSTIR reuse pair-wise.
    ///
    /// This is the energy-preserving path when samples cross surfaces with
    /// different target distributions. It adds two visibility rays per reused
    /// sample. Clearing it selects a faster biased approximation and can
    /// darken occluded geometry; it is suitable for experiments, not a
    /// reference comparison.
    pub pairwise_mis: bool,
    /// Defensive MIS factor for the canonical sample.
    /// Can be between 0 and 1.
    pub defensive_mis: f32,
}

/// What the ray tracer does with the scene.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, PartialOrd, blade_macros::AsPrimitive, strum::EnumIter,
)]
#[repr(u32)]
pub enum RenderMode {
    /// Direct illumination at the primary surface, with samples reused between
    /// neighboring pixels and frames before the result is denoised.
    ///
    /// This is not a one-bounce global-illumination estimator: geometry hit by
    /// a candidate contributes emission, but a non-emissive hit terminates the
    /// candidate. Compare its energy to canonical mode with `max_bounces = 0`;
    /// a multi-bounce canonical image also contains indirect fill.
    #[default]
    RealTime = 0,
    /// Reference: full paths with no reuse and no denoising, accumulated
    /// over the frames until the camera moves. Converges to the ground truth.
    Canonical = 1,
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
    /// Compress the radiance into displayable range.
    ///
    /// Clearing this leaves the composed linear radiance as it is, which is
    /// what a target that can hold it wants: a floating point offscreen
    /// texture for capture, analysis, or a neural post-process. The exposure
    /// and white level above are then unused. Note that an unbounded value
    /// written to a fixed point target still clamps, so the target format has
    /// to be a floating point one for this to mean anything.
    pub tone_map: bool,
}
impl Default for PostProcConfig {
    fn default() -> Self {
        Self {
            average_luminocity: 1.0,
            exposure_key_value: 1.0,
            white_level: 1.0,
            tone_map: true,
        }
    }
}

/// Views of the ray tracer's geometry and material buffer.
///
/// Handed out by [`RayTracer::view_gbuffer`]. Every one is readable as a
/// sampled texture; the renderer owns them, so nothing here needs destroying.
#[derive(Clone, Copy, Debug)]
pub struct GBufferViews {
    /// `R32Float`. Distance from the camera along the ray, not a projected
    /// depth, so it is in world units and needs no unprojection.
    pub depth: blade_graphics::TextureView,
    /// `Rgba8Snorm`. The shading tangent frame as a quaternion, which is where
    /// normal mapping ends up. The shading normal is the quaternion applied to
    /// `+Z`: `v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)` for
    /// `v = (0, 0, 1)`, matching `qrot` in `quaternion.inc.wgsl`.
    pub basis: blade_graphics::TextureView,
    /// `Rgba8Snorm`. The geometric normal in XYZ, straight from the triangle,
    /// with no normal map applied. Cheaper to consume than [`basis`] when the
    /// mapped detail is not wanted.
    ///
    /// [`basis`]: Self::basis
    pub flat_normal: blade_graphics::TextureView,
    /// `Rgba8Unorm`. Base color with the specularly reflected part taken out.
    pub diffuse_albedo: blade_graphics::TextureView,
    /// `Rgba8Unorm`. Specular reflectance at normal incidence in RGB, and the
    /// roughness in alpha.
    pub specular_f0: blade_graphics::TextureView,
    /// Emitted radiance, in the renderer's radiance format.
    pub emissive: blade_graphics::TextureView,
    /// `Rg16Float`. Screen space motion since the previous frame, scaled by
    /// `MOTION_SCALE` from `gbuf.inc.wgsl`. Half precision keeps subpixel
    /// reprojection accurate enough for temporal upscaling; `Rg8Snorm` stepped
    /// by roughly 0.4 pixels after decoding.
    pub motion: blade_graphics::TextureView,
}

/// Views of the current real-time lighting estimate.
///
/// Handed out by [`RayTracer::view_radiance`], primarily for an external
/// denoiser or upscaler. These are the same two inputs selected for
/// [`RayTracer::post_proc`]: if the built-in SVGF denoiser ran, they are its
/// latest output; otherwise they are the raw ReSTIR estimate.
///
/// The diffuse lobe is demodulated. Compose the final linear radiance as
/// `gbuffer.diffuse_albedo * diffuse + specular + gbuffer.emissive`.
#[derive(Clone, Copy, Debug)]
pub struct RadianceViews {
    /// Demodulated diffuse illumination, in the renderer's radiance format.
    pub diffuse: blade_graphics::TextureView,
    /// Specular radiance, already tinted by the Fresnel reflectance.
    pub specular: blade_graphics::TextureView,
}

/// Accumulated canonical path-tracing output, split by the primary response.
///
/// Every view is `Rgba32Float`: RGB is a radiance sum and alpha is the common
/// sample count. `diffuse` has the primary albedo divided out, `specular`
/// retains Fresnel tint, and `emissive` contains light seen directly by the
/// camera. `total` is accumulated separately so stochastic primary coverage
/// remains exact at material boundaries.
#[derive(Clone, Copy, Debug)]
pub struct AccumulatedRadianceViews {
    pub total: blade_graphics::TextureView,
    pub diffuse: blade_graphics::TextureView,
    pub specular: blade_graphics::TextureView,
    pub emissive: blade_graphics::TextureView,
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
    /// The base color with the specularly reflected part taken out.
    diffuse_albedo: RenderTarget<2>,
    /// RGB is the specular reflectance at normal incidence, alpha is the roughness.
    specular_f0: RenderTarget<2>,
    emissive: RenderTarget<1>,
    motion: RenderTarget<1>,
    light_diffuse: RenderTarget<3>,
    light_specular: RenderTarget<3>,
    /// Sum of the radiance of the canonical renderer, with the
    /// number of the accumulated samples in the alpha channel.
    accumulation: RenderTarget<1>,
    /// Canonical primary-lobe sums. Alpha repeats the common sample count.
    accumulation_diffuse: RenderTarget<1>,
    accumulation_specular: RenderTarget<1>,
    accumulation_emissive: RenderTarget<1>,
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
                transient: false,
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
            diffuse_albedo: RenderTarget::new(
                "diffuse-albedo",
                blade_graphics::TextureFormat::Rgba8Unorm,
                size,
                encoder,
                gpu,
            ),
            specular_f0: RenderTarget::new(
                "specular-f0",
                blade_graphics::TextureFormat::Rgba8Unorm,
                size,
                encoder,
                gpu,
            ),
            emissive: RenderTarget::new("emissive", RADIANCE_FORMAT, size, encoder, gpu),
            motion: RenderTarget::new(
                "motion",
                blade_graphics::TextureFormat::Rg16Float,
                size,
                encoder,
                gpu,
            ),
            light_diffuse: RenderTarget::new("light-diffuse", RADIANCE_FORMAT, size, encoder, gpu),
            light_specular: RenderTarget::new(
                "light-specular",
                RADIANCE_FORMAT,
                size,
                encoder,
                gpu,
            ),
            accumulation: RenderTarget::new(
                "accumulation",
                blade_graphics::TextureFormat::Rgba32Float,
                size,
                encoder,
                gpu,
            ),
            accumulation_diffuse: RenderTarget::new(
                "accumulation-diffuse",
                blade_graphics::TextureFormat::Rgba32Float,
                size,
                encoder,
                gpu,
            ),
            accumulation_specular: RenderTarget::new(
                "accumulation-specular",
                blade_graphics::TextureFormat::Rgba32Float,
                size,
                encoder,
                gpu,
            ),
            accumulation_emissive: RenderTarget::new(
                "accumulation-emissive",
                blade_graphics::TextureFormat::Rgba32Float,
                size,
                encoder,
                gpu,
            ),
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
        self.diffuse_albedo.destroy(gpu);
        self.specular_f0.destroy(gpu);
        self.emissive.destroy(gpu);
        self.motion.destroy(gpu);
        self.light_diffuse.destroy(gpu);
        self.light_specular.destroy(gpu);
        self.accumulation.destroy(gpu);
        self.accumulation_diffuse.destroy(gpu);
        self.accumulation_specular.destroy(gpu);
        self.accumulation_emissive.destroy(gpu);
    }
}

struct Blur {
    temporal_accum_pipeline: blade_graphics::ComputePipeline,
    a_trous_pipeline: blade_graphics::ComputePipeline,
}

/// Blade RayTracer is a comprehensive rendering solution for
/// end user applications.
///
/// It takes care of the shaders, geometry buffers, acceleration structures,
/// dummy resources, and debug drawing.
///
/// It doesn't:
///   - manage or submit any command encoders
///   - know about the window to display on
pub struct RayTracer {
    shaders: Shaders,
    targets: RestirTargets,
    post_proc_input_index: usize,
    fill_pipeline: blade_graphics::ComputePipeline,
    main_pipeline: blade_graphics::ComputePipeline,
    path_trace_pipeline: blade_graphics::ComputePipeline,
    skin: SkinPass,
    post_proc_pipeline: blade_graphics::RenderPipeline,
    blur: Blur,
    /// Owned by the ray tracer until the next scene build.
    acceleration_structure: blade_graphics::AccelerationStructure,
    /// Non-owning handle to the old scene used for this frame's motion queries.
    /// Its ownership is transferred to the current `FrameResources` in
    /// `build_scene` and it must never be destroyed directly by the ray tracer.
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
    color_space: blade_graphics::ColorSpace,
    frame_index: usize,
    frame_scene_built: usize,
    scene_models: Vec<blade_asset::Handle<crate::Model>>,
    scene_topology_changed: bool,
    is_frozen: bool,
    reset_accumulation: bool,
    show_accumulation: bool,
    //TODO: refactor `ResourceArray` to not carry the freelist logic
    // This way we can embed user info into the allocator.
    texture_resource_lookup:
        HashMap<blade_graphics::ResourceIndex, blade_asset::Handle<crate::Texture>>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub(crate) struct DebugParams {
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
    num_brdf_samples: u32,
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
    out_diffuse_albedo: blade_graphics::TextureView,
    out_specular_f0: blade_graphics::TextureView,
    out_emissive: blade_graphics::TextureView,
    out_motion: blade_graphics::TextureView,
    out_debug: blade_graphics::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct MainData<'a> {
    camera: CameraParams,
    prev_camera: CameraParams,
    debug: DebugParams,
    parameters: MainParams,
    acc_struct: blade_graphics::AccelerationStructure,
    prev_acc_struct: blade_graphics::AccelerationStructure,
    hit_entries: blade_graphics::BufferPiece,
    index_buffers: &'a blade_graphics::BufferArray<MAX_RESOURCES>,
    vertex_buffers: &'a blade_graphics::BufferArray<MAX_RESOURCES>,
    textures: &'a blade_graphics::TextureArray<MAX_RESOURCES>,
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
    t_diffuse_albedo: blade_graphics::TextureView,
    t_prev_diffuse_albedo: blade_graphics::TextureView,
    t_specular_f0: blade_graphics::TextureView,
    t_prev_specular_f0: blade_graphics::TextureView,
    t_motion: blade_graphics::TextureView,
    debug_buf: blade_graphics::BufferPiece,
    reservoirs: blade_graphics::BufferPiece,
    prev_reservoirs: blade_graphics::BufferPiece,
    out_diffuse: blade_graphics::TextureView,
    out_specular: blade_graphics::TextureView,
    out_debug: blade_graphics::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PathTraceParams {
    frame_index: u32,
    num_environment_samples: u32,
    num_brdf_samples: u32,
    max_bounces: u32,
    max_accumulated_samples: u32,
    t_start: f32,
    environment_importance_sampling: u32,
    reset_accumulation: u32,
    jitter_primary_rays: u32,
    _pad: [u32; 3],
}

#[derive(blade_macros::ShaderData)]
struct PathTraceData<'a> {
    camera: CameraParams,
    parameters: PathTraceParams,
    acc_struct: blade_graphics::AccelerationStructure,
    hit_entries: blade_graphics::BufferPiece,
    index_buffers: &'a blade_graphics::BufferArray<MAX_RESOURCES>,
    vertex_buffers: &'a blade_graphics::BufferArray<MAX_RESOURCES>,
    textures: &'a blade_graphics::TextureArray<MAX_RESOURCES>,
    sampler_linear: blade_graphics::Sampler,
    sampler_nearest: blade_graphics::Sampler,
    env_map: blade_graphics::TextureView,
    env_weights: blade_graphics::TextureView,
    accumulator: blade_graphics::TextureView,
    accumulator_diffuse: blade_graphics::TextureView,
    accumulator_specular: blade_graphics::TextureView,
    accumulator_emissive: blade_graphics::TextureView,
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
struct PostProcParams {
    tone_map_enabled: u32,
    average_lum: f32,
    key_value: f32,
    white_level: f32,
    accumulated: u32,
    encode_srgb: u32,
    external_input: u32,
    _pad: u32,
}

#[derive(blade_macros::ShaderData)]
struct PostProcData {
    t_diffuse_albedo: blade_graphics::TextureView,
    t_emissive: blade_graphics::TextureView,
    light_diffuse: blade_graphics::TextureView,
    light_specular: blade_graphics::TextureView,
    t_accumulation: blade_graphics::TextureView,
    t_debug: blade_graphics::TextureView,
    t_external: blade_graphics::TextureView,
    post_proc_params: PostProcParams,
    debug_params: DebugParams,
}

#[repr(C)]
#[derive(Debug)]
struct HitEntry {
    index_buf: u32,
    vertex_buf: u32,
    prev_vertex_buf: u32,
    flags: u32,
    //Note: it's technically `mat4x3` on WGSL side,
    // but it's aligned and sized the same way as `mat4`.
    geometry_to_object: mint::ColumnMatrix4<f32>,
    prev_geometry_to_object: mint::ColumnMatrix4<f32>,
    prev_object_to_world: mint::ColumnMatrix4<f32>,
    base_color_texture: u32,
    base_color_factor: [u8; 4],
    normal_texture: u32,
    normal_scale: f32,
    metallic_roughness_texture: u32,
    metalness: f32,
    roughness: f32,
    emissive_texture: u32,
    //Note: aligned to 16 bytes, matching `vec4` on the WGSL side
    emissive_factor: [f32; 4],
}

struct AnimatedBlasWork {
    acceleration_structure: blade_graphics::AccelerationStructure,
    meshes: Vec<blade_graphics::AccelerationStructureMesh>,
    scratch: blade_graphics::Buffer,
    update: bool,
}

struct AnimatedBlasSlot {
    vertex_buffer: blade_graphics::Buffer,
    acceleration_structure: blade_graphics::AccelerationStructure,
    scratch: blade_graphics::Buffer,
    model: blade_asset::Handle<crate::Model>,
    geometries: Box<[AnimatedGeometrySignature]>,
}

#[derive(PartialEq)]
struct AnimatedGeometrySignature {
    vertex_range: std::ops::Range<u32>,
    index_buffer: blade_graphics::Buffer,
    index_offset: u64,
    index_type: Option<blade_graphics::IndexType>,
    triangle_count: u32,
    opaque: bool,
}

impl AnimatedGeometrySignature {
    fn matches(&self, model: &crate::Model, geometry: &crate::model::Geometry) -> bool {
        self.vertex_range == geometry.vertex_range
            && self.index_buffer == model.index_buffer
            && self.index_offset == geometry.index_offset
            && self.index_type == geometry.index_type
            && self.triangle_count == geometry.triangle_count
            && self.opaque != model.materials[geometry.material_index].transparent
    }
}

impl AnimatedBlasSlot {
    fn matches(&self, handle: blade_asset::Handle<crate::Model>, model: &crate::Model) -> bool {
        self.model == handle
            && self.geometries.len() == model.geometries.len()
            && self
                .geometries
                .iter()
                .zip(&model.geometries)
                .all(|(signature, geometry)| signature.matches(model, geometry))
    }

    fn destroy(self, gpu: &blade_graphics::Context) {
        gpu.destroy_buffer(self.vertex_buffer);
        gpu.destroy_acceleration_structure(self.acceleration_structure);
        gpu.destroy_buffer(self.scratch);
    }

    fn retire(self, temp: &mut FrameResources) {
        temp.buffers.push(self.vertex_buffer);
        temp.acceleration_structures
            .push(self.acceleration_structure);
        temp.buffers.push(self.scratch);
    }
}

/// Per-object GPU storage for a posed or skinned instance.
///
/// Persist this with the [`crate::Object`] across frames so BLASes can be refit.
#[derive(Default)]
pub struct ObjectBlas {
    generations: crate::util::RollingBuffer<AnimatedBlasSlot, 3>,
    /// Dynamic vertex generation used by the immediately preceding scene.
    previous_generation: Option<usize>,
}

impl ObjectBlas {
    /// Release the per-instance resources after prior GPU work has completed.
    pub fn destroy(&mut self, gpu: &blade_graphics::Context) {
        for slot in self.generations.drain() {
            slot.destroy(gpu);
        }
        self.previous_generation = None;
    }

    pub(crate) fn retire(&mut self, temp: &mut FrameResources) {
        for slot in self.generations.drain() {
            slot.retire(temp);
        }
        self.previous_generation = None;
    }
}

struct ShaderPipelines {
    fill: blade_graphics::ComputePipeline,
    main: blade_graphics::ComputePipeline,
    path_trace: blade_graphics::ComputePipeline,
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

    fn create_path_trace(
        shader: &blade_graphics::Shader,
        gpu: &blade_graphics::Context,
    ) -> blade_graphics::ComputePipeline {
        shader.check_struct_size::<crate::Vertex>();
        shader.check_struct_size::<HitEntry>();
        shader.check_struct_size::<CameraParams>();
        shader.check_struct_size::<PathTraceParams>();
        let layout = <PathTraceData as blade_graphics::ShaderData>::layout();
        gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "path-trace",
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
            path_trace: Self::create_path_trace(
                shader_man[shaders.path_trace].raw.as_ref().unwrap(),
                gpu,
            ),
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
    /// Throw away what the canonical renderer has accumulated so far.
    ///
    /// Note: this also happens automatically when the camera moves.
    pub reset_accumulation: bool,
}

/// Temporary resources associated with a GPU frame.
pub use crate::util::FrameResources;

impl RayTracer {
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
        assert!(
            capabilities
                .ray_query
                .contains(blade_graphics::ShaderVisibility::COMPUTE)
        );

        let sp = ShaderPipelines::init(&shaders, config, gpu, shader_man).unwrap();
        let skin = SkinPass::new(shader_man[shaders.skin].raw.as_ref().unwrap(), gpu)
            .expect("compute skinning is required for the ray tracer");
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
            path_trace_pipeline: sp.path_trace,
            skin,
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
            color_space: config.color_space,
            frame_index: 0,
            frame_scene_built: 0,
            scene_models: Vec::new(),
            scene_topology_changed: false,
            is_frozen: false,
            reset_accumulation: true,
            show_accumulation: false,
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
        gpu.destroy_compute_pipeline(&mut self.path_trace_pipeline);
        self.skin.destroy(gpu);
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
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.path_trace));
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.skin));
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.a_trous));
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.post_proc));
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.debug_draw));
        tasks.extend(asset_hub.shaders.hot_reload(&mut self.shaders.debug_blit));

        if tasks.is_empty() {
            return false;
        }

        log::info!("Hot reloading shaders");
        let _ = gpu.wait_for(sync_point, !0);
        for task in tasks {
            let _ = task.join();
        }

        if self.shaders.fill_gbuf != old.fill_gbuf
            && let Ok(ref shader) = asset_hub.shaders[self.shaders.fill_gbuf].raw
        {
            self.fill_pipeline = ShaderPipelines::create_gbuf_fill(shader, gpu);
        }
        if self.shaders.ray_trace != old.ray_trace
            && let Ok(ref shader) = asset_hub.shaders[self.shaders.ray_trace].raw
        {
            assert_eq!(
                shader.get_struct_size("StoredReservoir"),
                self.reservoir_size
            );
            self.main_pipeline = ShaderPipelines::create_ray_trace(shader, gpu);
        }
        if self.shaders.path_trace != old.path_trace
            && let Ok(ref shader) = asset_hub.shaders[self.shaders.path_trace].raw
        {
            self.path_trace_pipeline = ShaderPipelines::create_path_trace(shader, gpu);
        }
        if self.shaders.skin != old.skin
            && let Ok(ref shader) = asset_hub.shaders[self.shaders.skin].raw
        {
            self.skin.recreate(shader, gpu);
        }
        if self.shaders.a_trous != old.a_trous
            && let Ok(ref shader) = asset_hub.shaders[self.shaders.a_trous].raw
        {
            self.blur.temporal_accum_pipeline = ShaderPipelines::create_temporal_accum(shader, gpu);
            self.blur.a_trous_pipeline = ShaderPipelines::create_a_trous(shader, gpu);
        }
        if self.shaders.post_proc != old.post_proc
            && let Ok(ref shader) = asset_hub.shaders[self.shaders.post_proc].raw
        {
            self.post_proc_pipeline =
                ShaderPipelines::create_post_proc(shader, self.surface_info, gpu);
        }
        if self.shaders.debug_draw != old.debug_draw
            && let Ok(ref shader) = asset_hub.shaders[self.shaders.debug_draw].raw
        {
            self.debug.recreate_draw_pipeline(shader, gpu);
        }
        if self.shaders.debug_blit != old.debug_blit
            && let Ok(ref shader) = asset_hub.shaders[self.shaders.debug_blit].raw
        {
            self.debug.recreate_blit_pipeline(shader, gpu);
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

    /// The geometry and material buffer of the frame that was last prepared.
    ///
    /// A post process that knows what the renderer knows can do things a post
    /// process working from the color alone cannot: exact silhouettes from the
    /// depth and the normals, texture detail separated from lighting by the
    /// albedo, and the width of a specular highlight from the roughness. This
    /// hands those out so a consumer outside the renderer — a neural upscaler,
    /// a capture tool, an analysis pass — can read them.
    ///
    /// The views are only valid until the next [`resize_screen`], and they
    /// describe the frame that [`prepare`] last filled: call it after
    /// `prepare`, and read before the next one overwrites them.
    ///
    /// [`resize_screen`]: Self::resize_screen
    /// [`prepare`]: Self::prepare
    pub fn view_gbuffer(&self) -> GBufferViews {
        // The geometry targets are double buffered for temporal reuse, and
        // `prepare` advanced the frame index, so the current one is here.
        let cur = self.frame_index % 2;
        GBufferViews {
            depth: self.targets.depth.views[cur],
            basis: self.targets.basis.views[cur],
            flat_normal: self.targets.flat_normal.views[cur],
            diffuse_albedo: self.targets.diffuse_albedo.views[cur],
            specular_f0: self.targets.specular_f0.views[cur],
            emissive: self.targets.emissive.views[0],
            motion: self.targets.motion.views[0],
        }
    }

    /// The real-time radiance that [`post_proc`](Self::post_proc) would use.
    ///
    /// The views remain valid until the next [`resize_screen`](Self::resize_screen).
    /// Call this after [`render`](Self::render), so `post_proc_input_index`
    /// identifies either the raw estimate or the last built-in denoising pass.
    pub fn view_radiance(&self) -> RadianceViews {
        RadianceViews {
            diffuse: self.targets.light_diffuse.views[self.post_proc_input_index],
            specular: self.targets.light_specular.views[self.post_proc_input_index],
        }
    }

    /// Canonical path-tracing sums after [`render`](Self::render) in
    /// [`RenderMode::Canonical`].
    ///
    /// Unlike [`view_radiance`](Self::view_radiance), these are unnormalised
    /// accumulators. Divide RGB by alpha before consuming a view.
    pub fn view_accumulated_radiance(&self) -> AccumulatedRadianceViews {
        AccumulatedRadianceViews {
            total: self.targets.accumulation.views[0],
            diffuse: self.targets.accumulation_diffuse.views[0],
            specular: self.targets.accumulation_specular.views[0],
            emissive: self.targets.accumulation_emissive.views[0],
        }
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
        objects: &mut [crate::Object],
        env_map: Option<blade_asset::Handle<crate::Texture>>,
        asset_hub: &crate::AssetHub,
        gpu: &blade_graphics::Context,
        temp: &mut FrameResources,
    ) {
        let scene_models: Vec<_> = objects.iter().map(|object| object.model).collect();
        self.scene_topology_changed = self.acceleration_structure
            != blade_graphics::AccelerationStructure::default()
            && scene_models != self.scene_models;
        self.scene_models = scene_models;

        let (env_view, env_extent) = match env_map {
            Some(handle) => {
                let asset = &asset_hub.textures[handle];
                (asset.view, asset.extent)
            }
            None => (self.dummy.white_view, blade_graphics::Extent::default()),
        };
        self.env_map
            .assign(env_view, env_extent, command_encoder, gpu);

        self.prev_acceleration_structure = std::mem::take(&mut self.acceleration_structure);
        if self.prev_acceleration_structure != blade_graphics::AccelerationStructure::default() {
            // The old current scene becomes the previous scene used by this
            // frame. Transfer ownership to this frame so FramePacer destroys
            // it only after this frame's fence signals.
            temp.acceleration_structures
                .push(self.prev_acceleration_structure);
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
            temp.buffers.push(self.hit_buffer);
        }
        self.hit_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "hit entries",
            size: hit_size,
            memory: blade_graphics::Memory::Device,
            transient: false,
        });
        let hit_staging = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "hit staging",
            size: hit_size,
            memory: blade_graphics::Memory::Upload,
            transient: false,
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
        let mut dynamic_blas_work = Vec::new();
        let mut skin_jobs = Vec::new();
        let mut vertex_copies = Vec::new();
        let mut texture_indices =
            HashMap::<blade_asset::Handle<crate::Texture>, blade_graphics::ResourceIndex>::new();
        // Note: this only borrows `self.textures`, so the buffer arrays stay available.
        let mut alloc_texture =
            |handle: Option<blade_asset::Handle<crate::Texture>>,
             dummy: blade_graphics::ResourceIndex| {
                match handle {
                    Some(handle) => *texture_indices.entry(handle).or_insert_with(|| {
                        let texture = &asset_hub.textures[handle];
                        self.textures.alloc(texture.view)
                    }),
                    None => dummy,
                }
            };

        for object in objects.iter_mut() {
            let transform = object.transform;
            let prev_transform = object.prev_transform;
            let pose = object.pose.clone();
            let prev_pose = object.prev_pose.clone();
            let model = &asset_hub.models[object.model];
            let pose = pose.as_ref();
            let prev_pose = prev_pose.as_ref();
            let dynamic = model
                .geometries
                .iter()
                .any(|geometry| geometry.skin_index.is_some())
                || pose.is_some();
            let geometry_transforms: Vec<_> = model
                .geometries
                .iter()
                .map(|geometry| model.geometry_transform(geometry, pose))
                .collect();
            let prev_geometry_transforms: Vec<_> = model
                .geometries
                .iter()
                .map(|geometry| model.geometry_transform(geometry, prev_pose))
                .collect();

            let (
                active_vertex_buffer,
                previous_vertex_buffer,
                current_baked,
                previous_baked,
                current_generation,
            ) = if dynamic {
                let vertex_count = model.vertex_count();
                let vertex_size = (vertex_count * mem::size_of::<crate::Vertex>()) as u64;
                let previous_vertex_buffer = object
                    .blas
                    .previous_generation
                    .and_then(|index| object.blas.generations.get(index))
                    .filter(|slot| slot.matches(object.model, model))
                    .map(|slot| slot.vertex_buffer);
                let (generation, slot) = object.blas.generations.next_indexed();
                let reuse = slot
                    .as_ref()
                    .is_some_and(|slot| slot.matches(object.model, model));
                if !reuse {
                    if let Some(old) = slot.take() {
                        old.retire(temp);
                    }
                    let vertex_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
                        name: "animated vertices",
                        size: vertex_size.max(1),
                        memory: blade_graphics::Memory::Device,
                        transient: false,
                    });
                    let vertex_stride = mem::size_of::<crate::Vertex>() as u32;
                    let size_meshes: Vec<_> = model
                        .geometries
                        .iter()
                        .map(|geometry| blade_graphics::AccelerationStructureMesh {
                            vertex_data: vertex_buffer
                                .at(geometry.vertex_range.start as u64 * vertex_stride as u64),
                            vertex_format: blade_graphics::VertexFormat::F32Vec3,
                            vertex_stride,
                            vertex_count: geometry.vertex_range.end - geometry.vertex_range.start,
                            index_data: model.index_buffer.at(geometry.index_offset),
                            index_type: geometry.index_type,
                            triangle_count: geometry.triangle_count,
                            transform_data: blade_graphics::Buffer::default().into(),
                            is_opaque: !model.materials[geometry.material_index].transparent,
                        })
                        .collect();
                    let sizes = gpu.get_bottom_level_acceleration_structure_sizes(&size_meshes);
                    let acceleration_structure = gpu.create_acceleration_structure(
                        blade_graphics::AccelerationStructureDesc {
                            name: "animated BLAS",
                            ty: blade_graphics::AccelerationStructureType::BottomLevel,
                            size: sizes.data,
                            updatable: true,
                        },
                    );
                    let scratch = gpu.create_buffer(blade_graphics::BufferDesc {
                        name: "animated BLAS scratch",
                        size: sizes.scratch,
                        memory: blade_graphics::Memory::Device,
                        transient: false,
                    });
                    *slot = Some(AnimatedBlasSlot {
                        vertex_buffer,
                        acceleration_structure,
                        scratch,
                        model: object.model,
                        geometries: model
                            .geometries
                            .iter()
                            .map(|geometry| AnimatedGeometrySignature {
                                vertex_range: geometry.vertex_range.clone(),
                                index_buffer: model.index_buffer,
                                index_offset: geometry.index_offset,
                                index_type: geometry.index_type,
                                triangle_count: geometry.triangle_count,
                                opaque: !model.materials[geometry.material_index].transparent,
                            })
                            .collect(),
                    });
                }
                let slot = slot.as_ref().unwrap();
                let vertex_buffer = slot.vertex_buffer;
                skin::queue_model(
                    model,
                    pose,
                    Some(&geometry_transforms),
                    vertex_buffer,
                    0,
                    &mut skin_jobs,
                    &mut vertex_copies,
                );

                let vertex_stride = mem::size_of::<crate::Vertex>() as u32;
                let meshes: Vec<_> = model
                    .geometries
                    .iter()
                    .map(|geometry| blade_graphics::AccelerationStructureMesh {
                        vertex_data: vertex_buffer
                            .at(geometry.vertex_range.start as u64 * vertex_stride as u64),
                        vertex_format: blade_graphics::VertexFormat::F32Vec3,
                        vertex_stride,
                        vertex_count: geometry.vertex_range.end - geometry.vertex_range.start,
                        index_data: model.index_buffer.at(geometry.index_offset),
                        index_type: geometry.index_type,
                        triangle_count: geometry.triangle_count,
                        transform_data: blade_graphics::Buffer::default().into(),
                        is_opaque: !model.materials[geometry.material_index].transparent,
                    })
                    .collect();
                dynamic_blas_work.push(AnimatedBlasWork {
                    acceleration_structure: slot.acceleration_structure,
                    meshes,
                    scratch: slot.scratch,
                    update: reuse,
                });
                let (previous_vertex_buffer, previous_baked) = match previous_vertex_buffer {
                    Some(buffer) => (buffer, true),
                    None if skin::model_needs_vertex_skin(model) => (vertex_buffer, true),
                    None => (model.vertex_buffer, false),
                };
                (
                    vertex_buffer,
                    previous_vertex_buffer,
                    true,
                    previous_baked,
                    Some(generation),
                )
            } else {
                (model.vertex_buffer, model.vertex_buffer, false, false, None)
            };
            object.blas.previous_generation = current_generation;

            instances.push(blade_graphics::AccelerationStructureInstance {
                acceleration_structure_index: blases.len() as u32,
                transform,
                mask: 0xFF,
                custom_index: geometry_index as u32,
            });
            blases.push(if dynamic {
                dynamic_blas_work.last().unwrap().acceleration_structure
            } else {
                model.acceleration_structure
            });

            for (geometry_index_in_model, geometry) in model.geometries.iter().enumerate() {
                let geometry_transform = if current_baked {
                    blade_graphics::IDENTITY_TRANSFORM
                } else {
                    geometry_transforms[geometry_index_in_model]
                };
                let prev_geometry_transform = if previous_baked {
                    blade_graphics::IDENTITY_TRANSFORM
                } else {
                    prev_geometry_transforms[geometry_index_in_model]
                };
                let material = &model.materials[geometry.material_index];
                let vertex_offset =
                    geometry.vertex_range.start as u64 * mem::size_of::<crate::Vertex>() as u64;

                let hit_entry = HitEntry {
                    index_buf: match geometry.index_type {
                        Some(_) => self
                            .index_buffers
                            .alloc(model.index_buffer.at(geometry.index_offset)),
                        None => !0,
                    },
                    vertex_buf: self
                        .vertex_buffers
                        .alloc(active_vertex_buffer.at(vertex_offset)),
                    prev_vertex_buf: self
                        .vertex_buffers
                        .alloc(previous_vertex_buffer.at(vertex_offset)),
                    flags: u32::from(model.winding < 0.0),
                    geometry_to_object: mint::ColumnMatrix4::from(mint::RowMatrix4 {
                        x: geometry_transform.x,
                        y: geometry_transform.y,
                        z: geometry_transform.z,
                        w: [0.0, 0.0, 0.0, 1.0].into(),
                    }),
                    prev_geometry_to_object: mint::ColumnMatrix4::from(mint::RowMatrix4 {
                        x: prev_geometry_transform.x,
                        y: prev_geometry_transform.y,
                        z: prev_geometry_transform.z,
                        w: [0.0, 0.0, 0.0, 1.0].into(),
                    }),
                    prev_object_to_world: mat4_transform(&prev_transform).into(),
                    base_color_texture: alloc_texture(material.base_color_texture, dummy_white),
                    base_color_factor: {
                        let c = material.base_color_factor;
                        [
                            (c[0] * 255.0) as u8,
                            (c[1] * 255.0) as u8,
                            (c[2] * 255.0) as u8,
                            (c[3] * 255.0) as u8,
                        ]
                    },
                    normal_texture: alloc_texture(material.normal_texture, dummy_black),
                    normal_scale: material.normal_scale,
                    //Note: the dummy is white, so that the factors are unaffected
                    metallic_roughness_texture: alloc_texture(
                        material.metallic_roughness_texture,
                        dummy_white,
                    ),
                    metalness: material.metalness,
                    roughness: material.roughness,
                    emissive_texture: alloc_texture(material.emissive_texture, dummy_white),
                    emissive_factor: {
                        let c = material.emissive_factor;
                        [c[0], c[1], c[2], 0.0]
                    },
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

        skin::encode(
            command_encoder,
            Some(&self.skin),
            &vertex_copies,
            &skin_jobs,
        );

        if !dynamic_blas_work.is_empty() {
            let mut encoder = command_encoder.acceleration_structure("animated BLAS");
            for work in &dynamic_blas_work {
                if work.update {
                    encoder.update_bottom_level(
                        work.acceleration_structure,
                        &work.meshes,
                        work.scratch.into(),
                    );
                } else {
                    encoder.build_bottom_level(
                        work.acceleration_structure,
                        &work.meshes,
                        work.scratch.into(),
                    );
                }
            }
        }

        // Needs to be a separate encoder in order to force synchronization
        let sizes = gpu.get_top_level_acceleration_structure_sizes(instances.len() as u32);
        self.acceleration_structure =
            gpu.create_acceleration_structure(blade_graphics::AccelerationStructureDesc {
                name: "TLAS",
                ty: blade_graphics::AccelerationStructureType::TopLevel,
                size: sizes.data,
                updatable: false,
            });
        let instance_buf = gpu.create_acceleration_structure_instance_buffer(&instances, &blases);
        let scratch_buf = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "TLAS scratch",
            size: sizes.scratch,
            memory: blade_graphics::Memory::Device,
            transient: false,
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
        CameraParams::new(camera, [self.surface_size.width, self.surface_size.height])
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
        let reset_reservoirs = config.reset_reservoirs || self.scene_topology_changed;
        let mut transfer = command_encoder.transfer("prepare");

        if config.debug_draw {
            self.debug.reset_lines(&mut transfer);
            self.debug.enable_draw(&mut transfer, true);
        } else {
            self.debug.enable_draw(&mut transfer, false);
        }

        if reset_reservoirs || config.reset_variance {
            self.debug.reset_variance(&mut transfer);
        } else {
            self.debug.update_variance(&mut transfer);
        }
        self.debug.update_entry(&mut transfer);

        if reset_reservoirs {
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
        let cur = self.frame_index % 2;
        let camera_params = self.make_camera_params(camera);
        // A moving camera invalidates the accumulation of the canonical renderer.
        self.reset_accumulation = config.reset_accumulation
            || self.scene_topology_changed
            || camera_params != self.targets.camera_params[cur ^ 1];
        self.show_accumulation = false;
        self.targets.camera_params[cur] = camera_params;
        self.post_proc_input_index = cur;
    }

    /// Render a frame in the given mode.
    ///
    /// The result is stored internally in an HDR render target, to be
    /// brought to the screen by `post_proc`. The denoiser configuration
    /// is only used by the real-time mode.
    #[profiling::function]
    pub fn render(
        &mut self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        mode: RenderMode,
        debug_config: DebugConfig,
        ray_config: RayConfig,
        denoiser_config: Option<DenoiserConfig>,
    ) {
        match mode {
            RenderMode::RealTime => {
                self.ray_trace(command_encoder, debug_config, ray_config);
                if let Some(config) = denoiser_config {
                    self.denoise(command_encoder, config);
                }
            }
            RenderMode::Canonical => {
                self.path_trace(command_encoder, ray_config);
            }
        }
    }

    /// Trace full paths with no reuse and no denoising, accumulating
    /// on top of what the previous frames have produced.
    #[profiling::function]
    fn path_trace(
        &mut self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        config: RayConfig,
    ) {
        let cur = self.frame_index % 2;
        let mut pass = command_encoder.compute("path-trace");
        let mut pc = pass.with(&self.path_trace_pipeline);
        let groups = self.path_trace_pipeline.get_dispatch_for(self.surface_size);
        pc.bind(
            0,
            &PathTraceData {
                camera: self.targets.camera_params[cur],
                parameters: PathTraceParams {
                    frame_index: self.frame_index as u32,
                    num_environment_samples: config.num_environment_samples,
                    num_brdf_samples: config.num_brdf_samples,
                    max_bounces: config.max_bounces,
                    max_accumulated_samples: config.max_accumulated_samples,
                    t_start: config.t_start,
                    environment_importance_sampling: config.environment_importance_sampling as u32,
                    reset_accumulation: self.reset_accumulation as u32,
                    jitter_primary_rays: config.jitter_primary_rays as u32,
                    _pad: [0; 3],
                },
                acc_struct: self.acceleration_structure,
                hit_entries: self.hit_buffer.into(),
                index_buffers: &self.index_buffers,
                vertex_buffers: &self.vertex_buffers,
                textures: &self.textures,
                sampler_linear: self.samplers.linear,
                sampler_nearest: self.samplers.nearest,
                env_map: self.env_map.main_view,
                env_weights: self.env_map.weight_view,
                accumulator: self.targets.accumulation.views[0],
                accumulator_diffuse: self.targets.accumulation_diffuse.views[0],
                accumulator_specular: self.targets.accumulation_specular.views[0],
                accumulator_emissive: self.targets.accumulation_emissive.views[0],
            },
        );
        pc.dispatch(groups);
        // The following frames add to what this one has produced.
        self.reset_accumulation = false;
        self.show_accumulation = true;
    }

    /// Record the primary-surface geometry and material buffers for the
    /// current camera and scene.
    ///
    /// [`RenderMode::RealTime`] does this as part of rendering. A sparse path
    /// tracer followed by an external denoiser can call it once after path
    /// accumulation instead of paying for the same primary rays on every
    /// accumulated sample.
    pub fn fill_gbuffer(
        &self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        debug_config: DebugConfig,
    ) {
        let debug = self.make_debug_params(&debug_config);
        let (cur, prev) = self.work_indices();
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
                    out_diffuse_albedo: self.targets.diffuse_albedo.views[cur],
                    out_specular_f0: self.targets.specular_f0.views[cur],
                    out_emissive: self.targets.emissive.views[0],
                    out_motion: self.targets.motion.views[0],
                    out_debug: self.targets.debug.views[0],
                },
            );
            pc.dispatch(groups);
        }
    }

    /// Estimate the lighting with ReSTIR, reusing the samples of
    /// the neighbors and of the previous frame.
    #[profiling::function]
    fn ray_trace(
        &self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        debug_config: DebugConfig,
        ray_config: RayConfig,
    ) {
        let debug = self.make_debug_params(&debug_config);
        let (cur, prev) = self.work_indices();
        assert_eq!(cur, self.post_proc_input_index);

        self.fill_gbuffer(command_encoder, debug_config);

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
                        num_brdf_samples: ray_config.num_brdf_samples,
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
                        || self.scene_topology_changed
                        || self.prev_acceleration_structure
                            == blade_graphics::AccelerationStructure::default()
                    {
                        self.acceleration_structure
                    } else {
                        self.prev_acceleration_structure
                    },
                    hit_entries: self.hit_buffer.into(),
                    index_buffers: &self.index_buffers,
                    vertex_buffers: &self.vertex_buffers,
                    textures: &self.textures,
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
                    t_diffuse_albedo: self.targets.diffuse_albedo.views[cur],
                    t_prev_diffuse_albedo: self.targets.diffuse_albedo.views[prev],
                    t_specular_f0: self.targets.specular_f0.views[cur],
                    t_prev_specular_f0: self.targets.specular_f0.views[prev],
                    t_motion: self.targets.motion.views[0],
                    debug_buf: self.debug.buffer_resource(),
                    reservoirs: self.targets.reservoir_buf[cur].into(),
                    prev_reservoirs: self.targets.reservoir_buf[prev].into(),
                    out_diffuse: self.targets.light_diffuse.views[cur],
                    out_specular: self.targets.light_specular.views[cur],
                    out_debug: self.targets.debug.views[0],
                },
            );
            pc.dispatch(groups);
        }
    }

    /// Perform noise reduction using SVGF.
    #[profiling::function]
    fn denoise(
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
        // Both of the lighting lobes are filtered the same way.
        let radiance_views = [
            self.targets.light_diffuse.views,
            self.targets.light_specular.views,
        ];

        if denoiser_config.temporal_weight < 1.0 {
            let mut pass = command_encoder.compute("temporal-accum");
            let mut pc = pass.with(&self.blur.temporal_accum_pipeline);
            let groups = self
                .blur
                .a_trous_pipeline
                .get_dispatch_for(self.surface_size);
            for views in radiance_views.iter() {
                pc.bind(
                    0,
                    &TemporalAccumData {
                        camera: self.targets.camera_params[cur],
                        prev_camera: self.targets.camera_params[prev],
                        params,
                        input: views[prev],
                        t_depth: self.targets.depth.views[cur],
                        t_prev_depth: self.targets.depth.views[prev],
                        t_flat_normal: self.targets.flat_normal.views[cur],
                        t_prev_flat_normal: self.targets.flat_normal.views[prev],
                        t_motion: self.targets.motion.views[0],
                        output: views[cur],
                    },
                );
                pc.dispatch(groups);
            }
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
            for views in radiance_views.iter() {
                pc.bind(
                    0,
                    &ATrousData {
                        params,
                        input: views[self.post_proc_input_index],
                        t_depth: self.targets.depth.views[cur],
                        t_flat_normal: self.targets.flat_normal.views[cur],
                        output: views[ping_pong[0]],
                    },
                );
                pc.dispatch(groups);
            }
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
        self.post_proc_impl(
            pass,
            None,
            debug_config,
            pp_config,
            debug_lines,
            debug_blits,
        );
    }

    /// Bring an external linear-radiance texture to the screen.
    ///
    /// This is the output side of an external denoiser/upscaler integration.
    /// In the final debug view, `color` replaces the renderer's internally
    /// composed radiance and receives the same tone mapping and surface color
    /// encoding as [`Self::post_proc`]. Other debug views remain available.
    #[profiling::function]
    pub fn post_proc_external(
        &self,
        pass: &mut blade_graphics::RenderCommandEncoder,
        color: blade_graphics::TextureView,
        debug_config: DebugConfig,
        pp_config: PostProcConfig,
        debug_lines: &[DebugLine],
        debug_blits: &[DebugBlit],
    ) {
        self.post_proc_impl(
            pass,
            Some(color),
            debug_config,
            pp_config,
            debug_lines,
            debug_blits,
        );
    }

    fn post_proc_impl(
        &self,
        pass: &mut blade_graphics::RenderCommandEncoder,
        external: Option<blade_graphics::TextureView>,
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
                    t_diffuse_albedo: self.targets.diffuse_albedo.views[cur],
                    t_emissive: self.targets.emissive.views[0],
                    light_diffuse: self.targets.light_diffuse.views[self.post_proc_input_index],
                    light_specular: self.targets.light_specular.views[self.post_proc_input_index],
                    t_accumulation: self.targets.accumulation.views[0],
                    t_debug: self.targets.debug.views[0],
                    t_external: external.unwrap_or(self.dummy.white_view),
                    post_proc_params: PostProcParams {
                        tone_map_enabled: pp_config.tone_map as u32,
                        average_lum: pp_config.average_luminocity,
                        key_value: pp_config.exposure_key_value,
                        white_level: pp_config.white_level,
                        accumulated: self.show_accumulation as u32,
                        encode_srgb: (self.color_space == blade_graphics::ColorSpace::Srgb) as u32,
                        external_input: external.is_some() as u32,
                        _pad: 0,
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

#[cfg(test)]
mod layout_tests {
    #[test]
    fn animated_hit_entry_matches_shader_layout() {
        assert_eq!(std::mem::size_of::<super::HitEntry>(), 256);
        assert_eq!(std::mem::size_of::<crate::Vertex>(), 32);
        assert_eq!(std::mem::size_of::<crate::SkinVertex>(), 8);
    }
}
