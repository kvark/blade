//! Differentiable rendering of the material factors and the environment.
//!
//! [`DiffTracer`] renders the scene of a [`RayTracer`](super::RayTracer)
//! with parameters it keeps in its own buffers, compares the image to a
//! target, and propagates the gradient of the loss back to the parameters,
//! which an Adam optimizer then updates. See "diff-path-trace.wgsl" for
//! the estimator.
//!
//! What is differentiated:
//!   - the glTF factors of every hit entry, which multiply its textures
//!     (see [`MaterialParams`]),
//!   - the radiance of every environment map texel.
//!
//! Geometry, the camera, and the textures themselves are fixed.

use std::{mem, path::Path, slice};

/// Longest path the differentiable tracer follows.
pub const MAX_DIFF_BOUNCES: u32 = 8;

/// The factors of the material of one hit entry, i.e. one geometry of
/// one object in the scene, in the order [`RayTracer::build_scene`]
/// (super::RayTracer::build_scene) visits them.
///
/// They multiply the textures of the material, like in glTF.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct MaterialParams {
    pub base_color: [f32; 3],
    pub metalness: f32,
    pub emissive: [f32; 3],
    pub roughness: f32,
}

const MATERIAL_PARAM_COUNT: u32 = (mem::size_of::<MaterialParams>() / 4) as u32;
// Floats per environment texel.
const ENV_PARAM_COUNT: u32 = 4;

bitflags::bitflags! {
    /// Which of the [`MaterialParams`] an optimizer step changes.
    #[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
    pub struct MaterialParamFlags: u32 {
        const BASE_COLOR = 0x07;
        const METALNESS = 0x08;
        const EMISSIVE = 0x70;
        const ROUGHNESS = 0x80;
    }
}

/// Parameters that live in the unit interval.
const BOUNDED_MATERIAL_PARAMS: MaterialParamFlags = MaterialParamFlags::BASE_COLOR
    .union(MaterialParamFlags::METALNESS)
    .union(MaterialParamFlags::ROUGHNESS);

#[derive(Clone, Copy, Debug)]
pub struct DiffConfig {
    /// Environment samples at every vertex of a path.
    pub num_environment_samples: u32,
    /// Paths traced through every pixel by each pass.
    pub samples_per_pixel: u32,
    /// Bounces after the primary hit, up to [`MAX_DIFF_BOUNCES`].
    pub max_bounces: u32,
    pub t_start: f32,
    /// Sample the environment by the weights of the original map. The
    /// gradient stays unbiased as the environment changes, since only the
    /// sampling, and not the estimate, depends on them.
    pub environment_importance_sampling: bool,
    pub jitter_primary_rays: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            num_environment_samples: 1,
            samples_per_pixel: 4,
            max_bounces: 3,
            t_start: 0.01,
            environment_importance_sampling: false,
            jitter_primary_rays: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdamConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct DiffShaders {
    path_trace: blade_asset::Handle<crate::Shader>,
    optimize: blade_asset::Handle<crate::Shader>,
}

impl DiffShaders {
    pub fn load(path: &Path, asset_hub: &crate::AssetHub) -> (Self, choir::RunningTask) {
        let mut ctx = asset_hub.open_context(path, "diff shader finish");
        let shaders = Self {
            path_trace: ctx.load_shader("diff-path-trace.wgsl"),
            optimize: ctx.load_shader("diff-optimize.wgsl"),
        };
        (shaders, ctx.close())
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct DiffParams {
    frame_index: u32,
    num_environment_samples: u32,
    samples_per_pixel: u32,
    max_bounces: u32,
    t_start: f32,
    environment_importance_sampling: u32,
    accumulated_frames: u32,
    jitter_primary_rays: u32,
}

#[derive(blade_macros::ShaderData)]
struct DiffTraceData<'a> {
    camera: crate::CameraParams,
    parameters: DiffParams,
    acc_struct: blade_graphics::AccelerationStructure,
    hit_entries: blade_graphics::BufferPiece,
    index_buffers: &'a blade_graphics::BufferArray<{ super::MAX_RESOURCES }>,
    vertex_buffers: &'a blade_graphics::BufferArray<{ super::MAX_RESOURCES }>,
    textures: &'a blade_graphics::TextureArray<{ super::MAX_RESOURCES }>,
    sampler_linear: blade_graphics::Sampler,
    sampler_nearest: blade_graphics::Sampler,
    env_map: blade_graphics::TextureView,
    env_weights: blade_graphics::TextureView,
    material_params: blade_graphics::BufferPiece,
    env_params: blade_graphics::BufferPiece,
    image: blade_graphics::BufferPiece,
    image_adjoint: blade_graphics::BufferPiece,
    material_grads: blade_graphics::BufferPiece,
    env_grads: blade_graphics::BufferPiece,
    init_material_params: blade_graphics::BufferPiece,
    init_env_params: blade_graphics::BufferPiece,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct LossParams {
    pixel_count: u32,
    _pad: [u32; 3],
}

#[derive(blade_macros::ShaderData)]
struct LossData {
    loss_params: LossParams,
    image: blade_graphics::BufferPiece,
    target_image: blade_graphics::BufferPiece,
    image_adjoint: blade_graphics::BufferPiece,
    loss: blade_graphics::BufferPiece,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct AdamParams {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    lower: f32,
    upper: f32,
    count: u32,
    period: u32,
    mask: u32,
    upper_mask: u32,
}

#[derive(blade_macros::ShaderData)]
struct AdamData {
    adam: AdamParams,
    params: blade_graphics::BufferPiece,
    grads: blade_graphics::BufferPiece,
    moments: blade_graphics::BufferPiece,
}

/// Parameters of one kind, with what the optimizer needs for them.
struct ParamGroup {
    params: blade_graphics::Buffer,
    grads: blade_graphics::Buffer,
    moments: blade_graphics::Buffer,
    /// Number of floats.
    count: u32,
    /// Adam steps taken so far.
    step: u32,
}

impl ParamGroup {
    fn new(name: &str, count: u32, host_visible: bool, gpu: &blade_graphics::Context) -> Self {
        let size = u64::from(count.max(1)) * 4;
        let memory = if host_visible {
            blade_graphics::Memory::Shared
        } else {
            blade_graphics::Memory::Device
        };
        Self {
            params: gpu.create_buffer(blade_graphics::BufferDesc {
                name: &format!("{name} params"),
                size,
                memory,
            }),
            grads: gpu.create_buffer(blade_graphics::BufferDesc {
                name: &format!("{name} grads"),
                size,
                memory,
            }),
            moments: gpu.create_buffer(blade_graphics::BufferDesc {
                name: &format!("{name} moments"),
                size: 2 * size,
                memory: blade_graphics::Memory::Device,
            }),
            count,
            step: 0,
        }
    }

    fn clear(&self, encoder: &mut blade_graphics::CommandEncoder) {
        let mut transfer = encoder.transfer("diff-clear");
        transfer.fill_buffer(self.grads.into(), self.grads.size(), 0);
        transfer.fill_buffer(self.moments.into(), self.moments.size(), 0);
    }

    fn clear_grads(&self, encoder: &mut blade_graphics::CommandEncoder) {
        let mut transfer = encoder.transfer("diff-clear-grads");
        transfer.fill_buffer(self.grads.into(), self.grads.size(), 0);
    }

    fn destroy(&mut self, gpu: &blade_graphics::Context) {
        gpu.destroy_buffer(self.params);
        gpu.destroy_buffer(self.grads);
        gpu.destroy_buffer(self.moments);
    }
}

/// Differentiable path tracer of the scene of a [`RayTracer`](super::RayTracer).
///
/// Its parameters start from the scene that was built when it was created,
/// and it has to be recreated when the scene is built with other geometry.
/// Every pass reads the scene from the ray tracer, which has to stay as it is.
///
/// A step of optimization goes through [`Self::render`],
/// [`Self::backward`], and [`Self::step`]. The primal image that the loss
/// is computed on must come from other random numbers than the gradient,
/// which is what passing different `frame_index` values does.
pub struct DiffTracer {
    primal_pipeline: blade_graphics::ComputePipeline,
    adjoint_pipeline: blade_graphics::ComputePipeline,
    init_materials_pipeline: blade_graphics::ComputePipeline,
    init_environment_pipeline: blade_graphics::ComputePipeline,
    loss_pipeline: blade_graphics::ComputePipeline,
    adam_pipeline: blade_graphics::ComputePipeline,
    materials: ParamGroup,
    environment: ParamGroup,
    environment_size: blade_graphics::Extent,
    image_size: blade_graphics::Extent,
    image: blade_graphics::Buffer,
    target: blade_graphics::Buffer,
    image_adjoint: blade_graphics::Buffer,
    loss: blade_graphics::Buffer,
    accumulated_frames: u32,
}

impl DiffTracer {
    /// Start from the scene of `ray_tracer`, which has to be built already,
    /// possibly earlier in the same `encoder`.
    ///
    /// Requires [`blade_graphics::Capabilities::shader_float32_atomic`].
    pub fn new(
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
        shaders: &DiffShaders,
        shader_man: &blade_asset::AssetManager<crate::shader::Baker>,
        ray_tracer: &super::RayTracer,
    ) -> Self {
        assert!(
            gpu.capabilities().shader_float32_atomic,
            "differentiable rendering needs float32 atomics"
        );
        let sh_trace = shader_man[shaders.path_trace].raw.as_ref().unwrap();
        sh_trace.check_struct_size::<crate::CameraParams>();
        sh_trace.check_struct_size::<DiffParams>();
        sh_trace.check_struct_size::<MaterialParams>();
        sh_trace.check_struct_size::<super::HitEntry>();
        let sh_optimize = shader_man[shaders.optimize].raw.as_ref().unwrap();
        sh_optimize.check_struct_size::<LossParams>();
        sh_optimize.check_struct_size::<AdamParams>();

        let trace_layout = super::ray_data_layout::<DiffTraceData>(gpu);
        let trace_pipeline = |name, entry| {
            gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
                name,
                data_layouts: &[&trace_layout],
                compute: sh_trace.at(entry),
            })
        };
        let primal_pipeline = trace_pipeline("diff-primal", "primal");
        let adjoint_pipeline = trace_pipeline("diff-adjoint", "adjoint");
        let init_materials_pipeline = trace_pipeline("diff-init-materials", "init_materials");
        let init_environment_pipeline = trace_pipeline("diff-init-environment", "init_environment");
        let loss_layout = <LossData as blade_graphics::ShaderData>::layout();
        let loss_pipeline = gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "diff-loss",
            data_layouts: &[&loss_layout],
            compute: sh_optimize.at("loss_l2"),
        });
        let adam_layout = <AdamData as blade_graphics::ShaderData>::layout();
        let adam_pipeline = gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "diff-adam",
            data_layouts: &[&adam_layout],
            compute: sh_optimize.at("adam_step"),
        });

        let material_count =
            (ray_tracer.hit_buffer.size() as usize / mem::size_of::<super::HitEntry>()) as u32;
        let materials = ParamGroup::new(
            "diff materials",
            material_count * MATERIAL_PARAM_COUNT,
            true,
            gpu,
        );
        // Without a map, the environment is the 1x1 dummy texture.
        let environment_size = match ray_tracer.env_map.size {
            size if size.width == 0 || size.height == 0 => blade_graphics::Extent {
                width: 1,
                height: 1,
                depth: 1,
            },
            size => size,
        };
        let environment = ParamGroup::new(
            "diff environment",
            environment_size.width * environment_size.height * ENV_PARAM_COUNT,
            false,
            gpu,
        );

        let image_size = ray_tracer.surface_size;
        let image_bytes = u64::from(image_size.width * image_size.height) * 16;
        let image_buffer = |name| {
            gpu.create_buffer(blade_graphics::BufferDesc {
                name,
                size: image_bytes,
                memory: blade_graphics::Memory::Device,
            })
        };
        let this = Self {
            primal_pipeline,
            adjoint_pipeline,
            init_materials_pipeline,
            init_environment_pipeline,
            loss_pipeline,
            adam_pipeline,
            materials,
            environment,
            environment_size,
            image_size,
            image: image_buffer("diff image"),
            target: image_buffer("diff target"),
            image_adjoint: image_buffer("diff image adjoint"),
            loss: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "diff loss",
                size: 4,
                memory: blade_graphics::Memory::Shared,
            }),
            accumulated_frames: 0,
        };

        this.materials.clear(encoder);
        this.environment.clear(encoder);
        {
            let mut transfer = encoder.transfer("diff-init");
            transfer.fill_buffer(this.image_adjoint.into(), image_bytes, 0);
        }
        let camera = crate::CameraParams::default();
        let parameters = bytemuck::Zeroable::zeroed();
        let mut pass = encoder.compute("diff-init");
        {
            let mut pc = pass.with(&this.init_materials_pipeline);
            pc.bind(0, &this.trace_data(ray_tracer, camera, parameters));
            let wg = this.init_materials_pipeline.get_workgroup_size();
            pc.dispatch([material_count.div_ceil(wg[0]).max(1), 1, 1]);
        }
        {
            let mut pc = pass.with(&this.init_environment_pipeline);
            pc.bind(0, &this.trace_data(ray_tracer, camera, parameters));
            pc.dispatch(
                this.init_environment_pipeline
                    .get_dispatch_for(environment_size),
            );
        }
        drop(pass);
        this
    }

    fn trace_data<'a>(
        &self,
        ray_tracer: &'a super::RayTracer,
        camera: crate::CameraParams,
        parameters: DiffParams,
    ) -> DiffTraceData<'a> {
        DiffTraceData {
            camera,
            parameters,
            acc_struct: ray_tracer.acceleration_structure,
            hit_entries: ray_tracer.hit_buffer.into(),
            index_buffers: &ray_tracer.index_buffers,
            vertex_buffers: &ray_tracer.vertex_buffers,
            textures: &ray_tracer.textures,
            sampler_linear: ray_tracer.samplers.linear,
            sampler_nearest: ray_tracer.samplers.nearest,
            env_map: ray_tracer.env_map.main_view,
            env_weights: ray_tracer.env_map.weight_view,
            material_params: self.materials.params.into(),
            env_params: self.environment.params.into(),
            image: self.image.into(),
            image_adjoint: self.image_adjoint.into(),
            material_grads: self.materials.grads.into(),
            env_grads: self.environment.grads.into(),
            init_material_params: self.materials.params.into(),
            init_env_params: self.environment.params.into(),
        }
    }

    fn make_params(&self, config: &DiffConfig, frame_index: u32) -> DiffParams {
        assert!(
            config.max_bounces <= MAX_DIFF_BOUNCES,
            "at most {MAX_DIFF_BOUNCES} bounces are supported"
        );
        DiffParams {
            frame_index,
            num_environment_samples: config.num_environment_samples,
            samples_per_pixel: config.samples_per_pixel,
            max_bounces: config.max_bounces,
            t_start: config.t_start,
            environment_importance_sampling: config.environment_importance_sampling as u32,
            accumulated_frames: self.accumulated_frames,
            jitter_primary_rays: config.jitter_primary_rays as u32,
        }
    }

    fn make_camera(&self, camera: &crate::Camera) -> crate::CameraParams {
        crate::CameraParams::new(camera, [self.image_size.width, self.image_size.height])
    }

    /// Number of hit entries, each with its own [`MaterialParams`].
    pub fn material_count(&self) -> usize {
        self.materials.count as usize / MATERIAL_PARAM_COUNT as usize
    }

    pub fn environment_size(&self) -> blade_graphics::Extent {
        self.environment_size
    }

    pub fn image_size(&self) -> blade_graphics::Extent {
        self.image_size
    }

    /// The material parameters, as they are after the submitted work.
    ///
    /// Only meaningful while no work that changes them is in flight.
    pub fn material_params(&self) -> &[MaterialParams] {
        unsafe {
            slice::from_raw_parts(
                self.materials.params.data() as *const MaterialParams,
                self.material_count(),
            )
        }
    }

    /// Change the material parameters for the work submitted from now on.
    ///
    /// Only valid while no work that uses them is in flight.
    pub fn material_params_mut(&mut self) -> &mut [MaterialParams] {
        unsafe {
            slice::from_raw_parts_mut(
                self.materials.params.data() as *mut MaterialParams,
                self.material_count(),
            )
        }
    }

    /// Derivative of the loss by the material parameters, accumulated by
    /// [`Self::backward`] since the last [`Self::step`].
    ///
    /// Only meaningful while no work that changes them is in flight.
    pub fn material_grads(&self) -> &[MaterialParams] {
        unsafe {
            slice::from_raw_parts(
                self.materials.grads.data() as *const MaterialParams,
                self.material_count(),
            )
        }
    }

    /// The loss computed by the last [`Self::backward`], once it's done.
    pub fn loss(&self) -> f32 {
        unsafe { *(self.loss.data() as *const f32) }
    }

    /// The rendered image: `Rgba32Float` radiance, row by row.
    pub fn image(&self) -> blade_graphics::BufferPiece {
        self.image.into()
    }

    /// The target that [`Self::backward`] compares the image with,
    /// in the same layout as [`Self::image`], for filling it in.
    pub fn target(&self) -> blade_graphics::BufferPiece {
        self.target.into()
    }

    /// Environment radiance as `Rgba32Float` texels, row by row.
    pub fn environment_params(&self) -> blade_graphics::BufferPiece {
        self.environment.params.into()
    }

    /// Render the image with the current parameters.
    ///
    /// With `accumulate`, the image becomes the average of this frame and
    /// those accumulated before, which only makes sense for the same
    /// parameters. Otherwise it starts over.
    pub fn render(
        &mut self,
        encoder: &mut blade_graphics::CommandEncoder,
        ray_tracer: &super::RayTracer,
        camera: &crate::Camera,
        config: &DiffConfig,
        frame_index: u32,
        accumulate: bool,
    ) {
        if !accumulate {
            self.accumulated_frames = 0;
        }
        let data = self.trace_data(
            ray_tracer,
            self.make_camera(camera),
            self.make_params(config, frame_index),
        );
        let mut pass = encoder.compute("diff-primal");
        let mut pc = pass.with(&self.primal_pipeline);
        pc.bind(0, &data);
        pc.dispatch(self.primal_pipeline.get_dispatch_for(self.image_size));
        self.accumulated_frames += 1;
    }

    /// Make the current image the target.
    pub fn capture_target(&self, encoder: &mut blade_graphics::CommandEncoder) {
        let mut transfer = encoder.transfer("diff-capture-target");
        transfer.copy_buffer_to_buffer(self.image.into(), self.target.into(), self.image.size());
    }

    /// Compute the loss of the current image against the target, along with
    /// its derivative by the image, which [`Self::backward`] starts from.
    pub fn evaluate_loss(&mut self, encoder: &mut blade_graphics::CommandEncoder) {
        {
            let mut transfer = encoder.transfer("diff-loss-clear");
            transfer.fill_buffer(self.loss.into(), 4, 0);
        }
        let pixel_count = self.image_size.width * self.image_size.height;
        let mut pass = encoder.compute("diff-loss");
        let mut pc = pass.with(&self.loss_pipeline);
        pc.bind(
            0,
            &LossData {
                loss_params: LossParams {
                    pixel_count,
                    _pad: [0; 3],
                },
                image: self.image.into(),
                target_image: self.target.into(),
                image_adjoint: self.image_adjoint.into(),
                loss: self.loss.into(),
            },
        );
        let wg = self.loss_pipeline.get_workgroup_size();
        pc.dispatch([pixel_count.div_ceil(wg[0]), 1, 1]);
    }

    /// Compute the loss of the current image against the target, and add
    /// its derivative by the parameters to the gradients.
    ///
    /// `frame_index` must differ from the one the image was rendered with,
    /// or the gradient would be correlated with the image and biased.
    pub fn backward(
        &mut self,
        encoder: &mut blade_graphics::CommandEncoder,
        ray_tracer: &super::RayTracer,
        camera: &crate::Camera,
        config: &DiffConfig,
        frame_index: u32,
    ) {
        self.evaluate_loss(encoder);
        let data = self.trace_data(
            ray_tracer,
            self.make_camera(camera),
            self.make_params(config, frame_index),
        );
        let mut pass = encoder.compute("diff-adjoint");
        let mut pc = pass.with(&self.adjoint_pipeline);
        pc.bind(0, &data);
        pc.dispatch(self.adjoint_pipeline.get_dispatch_for(self.image_size));
    }

    /// Update the parameters along the gradients accumulated since the last
    /// step, and clear them. `None` keeps a kind of parameters as it is.
    ///
    /// Every parameter stays non-negative, and those of the materials that
    /// are fractions stay in the unit interval.
    pub fn step(
        &mut self,
        encoder: &mut blade_graphics::CommandEncoder,
        materials: Option<(AdamConfig, MaterialParamFlags)>,
        environment: Option<AdamConfig>,
    ) {
        match materials {
            Some((config, flags)) => adam_step(
                &self.adam_pipeline,
                encoder,
                &mut self.materials,
                &config,
                MATERIAL_PARAM_COUNT,
                flags.bits(),
                BOUNDED_MATERIAL_PARAMS.bits(),
            ),
            None => self.materials.clear_grads(encoder),
        }
        match environment {
            Some(config) => adam_step(
                &self.adam_pipeline,
                encoder,
                &mut self.environment,
                &config,
                ENV_PARAM_COUNT,
                0x7,
                0,
            ),
            None => self.environment.clear_grads(encoder),
        }
    }

    pub fn destroy(&mut self, gpu: &blade_graphics::Context) {
        gpu.destroy_compute_pipeline(&mut self.primal_pipeline);
        gpu.destroy_compute_pipeline(&mut self.adjoint_pipeline);
        gpu.destroy_compute_pipeline(&mut self.init_materials_pipeline);
        gpu.destroy_compute_pipeline(&mut self.init_environment_pipeline);
        gpu.destroy_compute_pipeline(&mut self.loss_pipeline);
        gpu.destroy_compute_pipeline(&mut self.adam_pipeline);
        self.materials.destroy(gpu);
        self.environment.destroy(gpu);
        gpu.destroy_buffer(self.image);
        gpu.destroy_buffer(self.target);
        gpu.destroy_buffer(self.image_adjoint);
        gpu.destroy_buffer(self.loss);
    }
}

fn adam_step(
    pipeline: &blade_graphics::ComputePipeline,
    encoder: &mut blade_graphics::CommandEncoder,
    group: &mut ParamGroup,
    config: &AdamConfig,
    period: u32,
    mask: u32,
    upper_mask: u32,
) {
    group.step += 1;
    let step = group.step as i32;
    let mut pass = encoder.compute("diff-adam");
    let mut pc = pass.with(pipeline);
    pc.bind(
        0,
        &AdamData {
            adam: AdamParams {
                learning_rate: config.learning_rate,
                beta1: config.beta1,
                beta2: config.beta2,
                epsilon: config.epsilon,
                bias_correction1: 1.0 - config.beta1.powi(step),
                bias_correction2: 1.0 - config.beta2.powi(step),
                lower: 0.0,
                upper: 1.0,
                count: group.count,
                period,
                mask,
                upper_mask,
            },
            params: group.params.into(),
            grads: group.grads.into(),
            moments: group.moments.into(),
        },
    );
    let wg = pipeline.get_workgroup_size();
    pc.dispatch([group.count.div_ceil(wg[0]).max(1), 1, 1]);
}
