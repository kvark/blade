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
//!   - the radiance of every environment map texel,
//!   - optionally, the texels of the base color, metallic-roughness, and
//!     emissive textures (see [`DiffTracerDesc::textures`]).
//!
//! Geometry, the camera, and normal maps are fixed.

use std::{collections::HashMap, mem, path::Path, slice};

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
// Floats per material texture texel.
const TEXTURE_PARAM_COUNT: u32 = 4;
const NO_TEXTURE_REGION: u32 = !0;
// Workgroup size of the passes over flat arrays.
const FLAT_WORKGROUP_SIZE: u32 = 64;
// Rows of workgroups stay within the dispatch limit of every backend.
const MAX_FLAT_WORKGROUPS_PER_ROW: u32 = 32768;

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

#[derive(Clone, Copy, Debug, Default)]
pub struct DiffTracerDesc {
    /// Make the texels of the base color, metallic-roughness, and emissive
    /// textures of the scene parameters, at their top mip level.
    ///
    /// Each texel takes four floats in device memory, for the parameter,
    /// its gradient, and the two moments of the optimizer.
    pub textures: bool,
}

/// A texture whose texels are parameters, see [`DiffTracer::textures`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffTexture {
    pub texture: blade_asset::Handle<crate::Texture>,
    /// Offset of its texels in [`DiffTracer::texture_params`].
    pub offset: u32,
    pub width: u32,
    pub height: u32,
}

/// What an optimizer step changes. Every kind of parameters left as `None`
/// stays as it is.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepConfig {
    pub materials: Option<(AdamConfig, MaterialParamFlags)>,
    pub environment: Option<AdamConfig>,
    /// RGB of the texture texels, which stay in the unit interval.
    pub textures: Option<AdamConfig>,
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
    texture_regions: blade_graphics::BufferPiece,
    texture_params: blade_graphics::BufferPiece,
    texture_grads: blade_graphics::BufferPiece,
    init_material_params: blade_graphics::BufferPiece,
    init_env_params: blade_graphics::BufferPiece,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct TextureRegion {
    offset: u32,
    width: u32,
    height: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct TextureInit {
    texture_index: u32,
    offset: u32,
    width: u32,
    height: u32,
}

#[derive(blade_macros::ShaderData)]
struct TextureInitData<'a> {
    texture_init: TextureInit,
    textures: &'a blade_graphics::TextureArray<{ super::MAX_RESOURCES }>,
    init_texture_params: blade_graphics::BufferPiece,
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
    fn new(name: &str, count: u32, gpu: &blade_graphics::Context) -> Self {
        let size = u64::from(count.max(1)) * 4;
        let memory = blade_graphics::Memory::Device;
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
    init_texture_pipeline: blade_graphics::ComputePipeline,
    loss_pipeline: blade_graphics::ComputePipeline,
    adam_pipeline: blade_graphics::ComputePipeline,
    materials: ParamGroup,
    environment: ParamGroup,
    environment_size: blade_graphics::Extent,
    textures: ParamGroup,
    texture_list: Vec<DiffTexture>,
    texture_regions: blade_graphics::Buffer,
    /// Kept until the regions are uploaded, i.e. until destruction.
    texture_regions_staging: blade_graphics::Buffer,
    image_size: blade_graphics::Extent,
    image: blade_graphics::Buffer,
    target: blade_graphics::Buffer,
    image_adjoint: blade_graphics::Buffer,
    loss: blade_graphics::Buffer,
    /// Material parameters and gradients, for the host to read.
    readback: blade_graphics::Buffer,
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
        asset_hub: &crate::AssetHub,
        ray_tracer: &super::RayTracer,
        desc: &DiffTracerDesc,
    ) -> Self {
        let shader_man = &asset_hub.shaders;
        assert!(
            gpu.capabilities().shader_float32_atomic,
            "differentiable rendering needs float32 atomics"
        );
        let sh_trace = shader_man[shaders.path_trace].raw.as_ref().unwrap();
        sh_trace.check_struct_size::<crate::CameraParams>();
        sh_trace.check_struct_size::<DiffParams>();
        sh_trace.check_struct_size::<MaterialParams>();
        sh_trace.check_struct_size::<super::HitEntry>();
        sh_trace.check_struct_size::<TextureRegion>();
        sh_trace.check_struct_size::<TextureInit>();
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
        let texture_init_layout = super::ray_data_layout::<TextureInitData>(gpu);
        let init_texture_pipeline =
            gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
                name: "diff-init-texture",
                data_layouts: &[&texture_init_layout],
                compute: sh_trace.at("init_texture"),
            });
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
        let materials =
            ParamGroup::new("diff materials", material_count * MATERIAL_PARAM_COUNT, gpu);
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
            gpu,
        );

        // The textures that the materials of the scene sample, by their
        // index in the binding array of the ray tracer.
        let mut texture_list = Vec::new();
        let mut texture_count = 0;
        let mut region_table = vec![
            TextureRegion {
                offset: NO_TEXTURE_REGION,
                width: 0,
                height: 0,
                _pad: 0,
            };
            1
        ];
        if desc.textures {
            let resource_indices = ray_tracer
                .texture_resource_lookup
                .iter()
                .map(|(&index, &handle)| (handle, index))
                .collect::<HashMap<_, _>>();
            for &model in &ray_tracer.scene_models {
                for material in &asset_hub.models[model].materials {
                    for handle in [
                        material.base_color_texture,
                        material.metallic_roughness_texture,
                        material.emissive_texture,
                    ]
                    .into_iter()
                    .flatten()
                    {
                        let index = resource_indices[&handle] as usize;
                        if index >= region_table.len() {
                            region_table.resize(index + 1, region_table[0]);
                        }
                        if region_table[index].offset != NO_TEXTURE_REGION {
                            continue;
                        }
                        let extent = asset_hub.textures[handle].extent;
                        let texture = DiffTexture {
                            texture: handle,
                            offset: texture_count,
                            width: extent.width,
                            height: extent.height,
                        };
                        region_table[index] = TextureRegion {
                            offset: texture.offset,
                            width: texture.width,
                            height: texture.height,
                            _pad: 0,
                        };
                        texture_count += texture.width * texture.height;
                        texture_list.push((index as u32, texture));
                    }
                }
            }
        }
        let textures = ParamGroup::new("diff textures", texture_count * TEXTURE_PARAM_COUNT, gpu);
        let region_bytes = (region_table.len() * mem::size_of::<TextureRegion>()) as u64;
        let texture_regions = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "diff texture regions",
            size: region_bytes,
            memory: blade_graphics::Memory::Device,
        });
        let texture_regions_staging = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "diff texture regions staging",
            size: region_bytes,
            memory: blade_graphics::Memory::Upload,
        });
        unsafe {
            slice::from_raw_parts_mut(
                texture_regions_staging.data() as *mut TextureRegion,
                region_table.len(),
            )
            .copy_from_slice(&region_table);
        }

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
            init_texture_pipeline,
            loss_pipeline,
            adam_pipeline,
            materials,
            environment,
            environment_size,
            textures,
            texture_list: texture_list.iter().map(|&(_, texture)| texture).collect(),
            texture_regions,
            texture_regions_staging,
            image_size,
            image: image_buffer("diff image"),
            target: image_buffer("diff target"),
            image_adjoint: image_buffer("diff image adjoint"),
            loss: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "diff loss",
                size: 4,
                memory: blade_graphics::Memory::Shared,
            }),
            readback: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "diff readback",
                size: 2
                    * u64::from(material_count.max(1))
                    * mem::size_of::<MaterialParams>() as u64,
                memory: blade_graphics::Memory::Download,
            }),
            accumulated_frames: 0,
        };

        this.materials.clear(encoder);
        this.environment.clear(encoder);
        this.textures.clear(encoder);
        {
            let mut transfer = encoder.transfer("diff-init");
            transfer.fill_buffer(this.image_adjoint.into(), image_bytes, 0);
            transfer.copy_buffer_to_buffer(
                texture_regions_staging.into(),
                texture_regions.into(),
                region_bytes,
            );
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
        if !texture_list.is_empty() {
            let mut pc = pass.with(&this.init_texture_pipeline);
            for &(texture_index, texture) in &texture_list {
                pc.bind(
                    0,
                    &TextureInitData {
                        texture_init: TextureInit {
                            texture_index,
                            offset: texture.offset,
                            width: texture.width,
                            height: texture.height,
                        },
                        textures: &ray_tracer.textures,
                        init_texture_params: this.textures.params.into(),
                    },
                );
                pc.dispatch(
                    this.init_texture_pipeline
                        .get_dispatch_for(blade_graphics::Extent {
                            width: texture.width,
                            height: texture.height,
                            depth: 1,
                        }),
                );
            }
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
            texture_regions: self.texture_regions.into(),
            texture_params: self.textures.params.into(),
            texture_grads: self.textures.grads.into(),
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

    /// Copy the material parameters and their gradients, as they are at
    /// this point of the `encoder`, to where [`Self::material_params`] and
    /// [`Self::material_grads`] read them once the work is done.
    pub fn read_back_materials(&self, encoder: &mut blade_graphics::CommandEncoder) {
        let bytes = self.material_bytes();
        let mut transfer = encoder.transfer("diff-read-back-materials");
        transfer.copy_buffer_to_buffer(self.materials.params.into(), self.readback.at(0), bytes);
        transfer.copy_buffer_to_buffer(self.materials.grads.into(), self.readback.at(bytes), bytes);
    }

    fn material_bytes(&self) -> u64 {
        (self.material_count() * mem::size_of::<MaterialParams>()) as u64
    }

    fn read_back(&self, offset: u64) -> &[MaterialParams] {
        unsafe {
            slice::from_raw_parts(
                self.readback.data().add(offset as usize) as *const MaterialParams,
                self.material_count(),
            )
        }
    }

    /// The material parameters, as of the last [`Self::read_back_materials`],
    /// once its work is done.
    pub fn material_params(&self) -> &[MaterialParams] {
        self.read_back(0)
    }

    /// Derivative of the loss by the material parameters, accumulated by
    /// [`Self::backward`] since the last [`Self::step`], as of the last
    /// [`Self::read_back_materials`], once its work is done.
    pub fn material_grads(&self) -> &[MaterialParams] {
        self.read_back(self.material_bytes())
    }

    /// Replace the material parameters at this point of the `encoder`.
    ///
    /// The staging buffer goes to `temp`, to be destroyed with it.
    pub fn write_materials(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
        params: &[MaterialParams],
        temp: &mut crate::FrameResources,
    ) {
        assert_eq!(params.len(), self.material_count());
        let bytes = self.material_bytes();
        let staging = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "diff material staging",
            size: bytes,
            memory: blade_graphics::Memory::Upload,
        });
        unsafe {
            slice::from_raw_parts_mut(staging.data() as *mut MaterialParams, params.len())
                .copy_from_slice(params);
        }
        encoder
            .transfer("diff-write-materials")
            .copy_buffer_to_buffer(staging.into(), self.materials.params.into(), bytes);
        temp.buffers.push(staging);
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

    /// The textures whose texels are parameters, which
    /// [`DiffTracerDesc::textures`] asks for.
    pub fn textures(&self) -> &[DiffTexture] {
        &self.texture_list
    }

    /// Texels of all the [`Self::textures`] as `Rgba32Float`, each texture
    /// row by row, starting at its [`DiffTexture::offset`]. Like the
    /// textures, they are linear: sRGB textures are decoded.
    pub fn texture_params(&self) -> blade_graphics::BufferPiece {
        self.textures.params.into()
    }

    /// Derivative of the loss by [`Self::texture_params`], accumulated by
    /// [`Self::backward`] since the last [`Self::step`].
    pub fn texture_grads(&self) -> blade_graphics::BufferPiece {
        self.textures.grads.into()
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
        pc.dispatch(flat_dispatch(pixel_count));
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
    /// step, and clear the gradients.
    ///
    /// Every parameter stays non-negative, and those that are fractions
    /// stay in the unit interval.
    pub fn step(&mut self, encoder: &mut blade_graphics::CommandEncoder, config: &StepConfig) {
        match config.materials {
            Some((adam, flags)) => adam_step(
                &self.adam_pipeline,
                encoder,
                &mut self.materials,
                &adam,
                MATERIAL_PARAM_COUNT,
                flags.bits(),
                BOUNDED_MATERIAL_PARAMS.bits(),
            ),
            None => self.materials.clear_grads(encoder),
        }
        match config.environment {
            Some(adam) => adam_step(
                &self.adam_pipeline,
                encoder,
                &mut self.environment,
                &adam,
                ENV_PARAM_COUNT,
                0x7,
                0,
            ),
            None => self.environment.clear_grads(encoder),
        }
        match config.textures {
            Some(adam) => adam_step(
                &self.adam_pipeline,
                encoder,
                &mut self.textures,
                &adam,
                TEXTURE_PARAM_COUNT,
                0x7,
                0x7,
            ),
            None => self.textures.clear_grads(encoder),
        }
    }

    pub fn destroy(&mut self, gpu: &blade_graphics::Context) {
        gpu.destroy_compute_pipeline(&mut self.primal_pipeline);
        gpu.destroy_compute_pipeline(&mut self.adjoint_pipeline);
        gpu.destroy_compute_pipeline(&mut self.init_materials_pipeline);
        gpu.destroy_compute_pipeline(&mut self.init_environment_pipeline);
        gpu.destroy_compute_pipeline(&mut self.init_texture_pipeline);
        gpu.destroy_compute_pipeline(&mut self.loss_pipeline);
        gpu.destroy_compute_pipeline(&mut self.adam_pipeline);
        self.materials.destroy(gpu);
        self.environment.destroy(gpu);
        self.textures.destroy(gpu);
        gpu.destroy_buffer(self.texture_regions);
        gpu.destroy_buffer(self.texture_regions_staging);
        gpu.destroy_buffer(self.image);
        gpu.destroy_buffer(self.target);
        gpu.destroy_buffer(self.image_adjoint);
        gpu.destroy_buffer(self.loss);
        gpu.destroy_buffer(self.readback);
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
    pc.dispatch(flat_dispatch(group.count));
}

/// Workgroups covering `count` elements of a flat array, in rows.
fn flat_dispatch(count: u32) -> [u32; 3] {
    let groups = count.div_ceil(FLAT_WORKGROUP_SIZE).max(1);
    let row = groups.min(MAX_FLAT_WORKGROUPS_PER_ROW);
    [row, groups.div_ceil(row), 1]
}

#[cfg(test)]
mod tests {
    #[test]
    fn flat_dispatch_covers_long_arrays() {
        for count in [0, 1, 64, 65, 64 * 32768, 64 * 32768 + 1, 4096 * 4096 * 4] {
            let [x, y, z] = super::flat_dispatch(count);
            assert!(x <= super::MAX_FLAT_WORKGROUPS_PER_ROW && y <= 65535 && z == 1);
            let covered = u64::from(x) * u64::from(y) * u64::from(super::FLAT_WORKGROUP_SIZE);
            assert!(covered >= u64::from(count), "{count} elements");
            // Only the last row is partial.
            assert!(covered - u64::from(count) <= u64::from(x * super::FLAT_WORKGROUP_SIZE));
        }
    }
}
