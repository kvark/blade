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
//!   - optionally, the texels of the base color, metallic-roughness,
//!     emissive, and normal textures (see [`DiffTracerDesc::textures`]).
//!
//!   - optionally, the translations of the objects and of the camera (see
//!     [`DiffTracerDesc::geometry`]).
//!
//! Shading normals that face away from the view are bent towards it, which
//! is the only difference from the reference path tracer. That one cuts the
//! BSDF off there, which would be a discontinuity for the gradient to miss.

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
    /// The part of the diffuse albedo that goes through the surface
    /// instead, as through a thin wall, like `KHR_materials_transmission`
    /// with its base color tint. Only the differentiable tracer has it, and
    /// it starts at zero.
    pub transmission: f32,
    /// The index of refraction of a solid that the transmission goes
    /// through, refracted at its surface by a rough dielectric microfacet
    /// model, or zero for a thin wall. It starts at zero.
    pub ior: f32,
    _pad: [f32; 2],
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
        const TRANSMISSION = 0x100;
        const IOR = 0x200;
    }
}

/// Parameters that live in the unit interval.
const BOUNDED_MATERIAL_PARAMS: MaterialParamFlags = MaterialParamFlags::BASE_COLOR
    .union(MaterialParamFlags::METALNESS)
    .union(MaterialParamFlags::ROUGHNESS)
    .union(MaterialParamFlags::TRANSMISSION);

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
    /// Has to be on for the gradient of the geometry.
    pub jitter_primary_rays: bool,
    /// Samples along every edge of every triangle, for the gradient of
    /// the geometry.
    pub edge_samples: u32,
    /// Edges drawn at random for every primary hit, for the gradient of the
    /// geometry by the light it receives: shadows and reflections move with
    /// the silhouettes that cast them. Zero leaves it out.
    ///
    /// The edges come from the whole scene, and few of them are silhouettes
    /// from a given point, so this takes many samples.
    pub secondary_edge_samples: u32,
    /// The hits of a path, from the primary one, at which
    /// [`Self::secondary_edge_samples`] are drawn: 1 for the shadows and
    /// reflections that the camera sees, and more for those that are seen
    /// in reflections themselves.
    pub secondary_edge_depth: u32,
    /// Draw the secondary edges by how much they brought before, once
    /// [`DiffTracer::refresh_edge_guide`] made a guide of it, mixed with
    /// drawing them uniformly, which keeps the gradient unbiased. This
    /// also records what they bring for the next guide.
    pub guided_edges: bool,
    /// Make every other path of the gradient the twin of the one before,
    /// with the direction that it takes from the first hit turned by half a
    /// turn about the normal there. Their gradients by the normal mostly
    /// cancel, which takes much of the noise out of those by normal maps,
    /// rotations and vertex positions on glossy surfaces.
    pub antithetic: bool,
    /// Points drawn on the triangles whose materials emit, at every vertex
    /// of a path, for their light, along with finding them by the BSDF.
    /// The emitters are those with any emissive factor, as of every pass.
    pub emitter_samples: u32,
    /// Weight of the recursive control variate of the image that the loss
    /// is taken on, from 0 for none to below 1, see
    /// [`DiffTracer::render`]. It takes another pass for every image.
    pub control_variate: f32,
    /// Sample the edges that the camera sees by carrying the hits of the
    /// paths of the gradient onto them, instead of sampling every edge,
    /// which takes as long for any number of triangles, and those that the
    /// first [`Self::secondary_edge_depth`] hits see by carrying the hits
    /// of the directions that the paths take from them, instead of drawing
    /// [`Self::secondary_edge_samples`], see `sample_projective_edge` and
    /// `sample_projective_secondary` in the shader.
    pub projective_edges: bool,
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
            edge_samples: 1,
            secondary_edge_samples: 0,
            secondary_edge_depth: 1,
            guided_edges: false,
            antithetic: false,
            emitter_samples: 1,
            control_variate: 0.0,
            projective_edges: false,
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
    /// Make the texels of the base color, metallic-roughness, emissive, and
    /// normal textures of the scene parameters, at their top mip level.
    ///
    /// Each texel takes four floats in device memory, for the parameter,
    /// its gradient, and the two moments of the optimizer.
    pub textures: bool,
    /// Differentiate translations of every object and of the camera.
    ///
    /// They are offsets that the caller applies, by adding them to the
    /// object transforms and the camera position, building the scene again,
    /// and passing it to [`DiffTracer::update_geometry`].
    ///
    /// The gradient accounts for edges of triangles moving across the image,
    /// and for the interpolated normal of the primary hits moving with it.
    /// With [`DiffConfig::secondary_edge_samples`], it also accounts for
    /// shadows and reflected silhouettes moving on the surfaces that the
    /// camera sees. It leaves out how they move past those, how the lighting
    /// changes between them, and how texture coordinates move. Edges are told apart by the positions of their
    /// ends, rounded to 1/65536 in the space of their object, so that copies
    /// of a vertex along a seam count as one.
    pub geometry: bool,
    /// Differentiate the positions of the vertices of every model in the
    /// scene, as offsets from where they were loaded, which
    /// [`DiffTracer::apply_vertices`] writes into the models. Copies of a
    /// vertex along seams share an offset. Requires `geometry`.
    ///
    /// The gradient goes through what moves with the triangles, like that of
    /// the translations, and takes the shading normals as they are. Those
    /// follow the positions as they are applied, as the normalized sums of
    /// the normals of the triangles around each vertex.
    pub vertices: bool,
    /// Number of views, each with its own camera, image, and target, which
    /// [`DiffTracer::select_view`] switches between. Zero counts as one.
    ///
    /// The gradients of every view add up until the step, which is how the
    /// views constrain the scene together.
    pub views: u32,
}

/// Offsets of the geometry from where an optimization started, as
/// [`DiffTracer::geometry_offsets`] reads them back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeometryOffsets {
    /// Translation of every object.
    pub translations: Vec<[f32; 3]>,
    /// Rotation of every object about its origin, as an axis times the
    /// angle in radians.
    pub rotations: Vec<[f32; 3]>,
    /// Translation of the camera.
    pub camera: [f32; 3],
}

/// Where the objects and the camera were when an optimization started,
/// which the [`GeometryOffsets`] are relative to.
///
/// A step of optimizing the geometry goes like this:
///   1. [`DiffTracer::step`], then [`DiffTracer::read_back`], and waiting
///      for the work,
///   2. [`GeometryBase::apply`] of [`DiffTracer::geometry_offsets`],
///   3. [`DiffTracer::rebuild_scene`] with the moved objects.
#[derive(Clone, Debug)]
pub struct GeometryBase {
    pub transforms: Vec<blade_graphics::Transform>,
    pub camera_position: mint::Vector3<f32>,
}

impl GeometryBase {
    pub fn new(objects: &[crate::Object], camera: &crate::Camera) -> Self {
        Self {
            transforms: objects.iter().map(|object| object.transform).collect(),
            camera_position: camera.pos,
        }
    }

    /// Move `objects` and `camera` by `offsets` from where they started.
    pub fn apply(
        &self,
        offsets: &GeometryOffsets,
        objects: &mut [crate::Object],
        camera: &mut crate::Camera,
    ) {
        assert_eq!(objects.len(), self.transforms.len());
        for (i, (object, base)) in objects.iter_mut().zip(&self.transforms).enumerate() {
            let translation = offsets.translations.get(i).copied().unwrap_or_default();
            let rotation = offsets.rotations.get(i).copied().unwrap_or_default();
            object.transform = offset_transform(base, translation, rotation);
        }
        let p = self.camera_position;
        let c = offsets.camera;
        camera.pos = [p.x + c[0], p.y + c[1], p.z + c[2]].into();
    }
}

/// Turn `base` by `rotation` about its origin, then translate it.
pub fn offset_transform(
    base: &blade_graphics::Transform,
    translation: [f32; 3],
    rotation: [f32; 3],
) -> blade_graphics::Transform {
    let rotation = glam::Vec3::from(rotation);
    let angle = rotation.length();
    let turn = if angle > 0.0 {
        glam::Mat3::from_axis_angle(rotation / angle, angle)
    } else {
        glam::Mat3::IDENTITY
    };
    let rows = [base.x, base.y, base.z];
    let linear = glam::Mat3::from_cols(
        glam::Vec3::new(rows[0].x, rows[1].x, rows[2].x),
        glam::Vec3::new(rows[0].y, rows[1].y, rows[2].y),
        glam::Vec3::new(rows[0].z, rows[1].z, rows[2].z),
    );
    let turned = turn * linear;
    let row = |i: usize, origin: f32| -> mint::Vector4<f32> {
        let r = turned.row(i);
        [r.x, r.y, r.z, origin + translation[i]].into()
    };
    blade_graphics::Transform {
        x: row(0, rows[0].w),
        y: row(1, rows[1].w),
        z: row(2, rows[2].w),
    }
}

/// A geometry of a model, as [`DiffTracer::current_meshes`] reads it back.
#[derive(Clone, Debug)]
pub struct CurrentMesh {
    pub model: blade_asset::Handle<crate::Model>,
    /// Index of the geometry in the model.
    pub geometry: usize,
    pub mesh: super::remesh::Mesh,
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
    /// RGB of the texture texels, which stay in the unit interval. Normal
    /// maps only use red and green, so blue doesn't get a gradient.
    pub textures: Option<AdamConfig>,
    /// Translations of the objects, which are unbounded.
    pub translations: Option<AdamConfig>,
    /// Rotations of the objects, which are unbounded.
    pub rotations: Option<AdamConfig>,
    /// Offsets of the vertices, which are unbounded, with the weight of
    /// pulling the offset of every vertex towards the mean of those of its
    /// neighbors, which keeps the surface from crumpling.
    pub vertices: Option<(AdamConfig, f32)>,
    /// Translation of the camera, which is unbounded.
    pub camera_position: Option<AdamConfig>,
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
    geometry: u32,
    instance_count: u32,
    triangle_count: u32,
    edge_samples: u32,
    secondary_edge_samples: u32,
    secondary_edge_depth: u32,
    guided_edges: u32,
    antithetic: u32,
    emitter_samples: u32,
    projective_edges: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct EdgeEntry {
    hit_entry: u32,
    instance: u32,
    first_triangle: u32,
    triangle_count: u32,
    vertex_base: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct VertexPass {
    first_vertex: u32,
    vertex_count: u32,
    model: u32,
    first_index: u32,
    geometry_vertex_start: u32,
    triangle_count: u32,
    smoothness: f32,
    slot_count: u32,
}

#[derive(blade_macros::ShaderData)]
struct VertexPassData {
    vertex_pass: VertexPass,
    mesh_vertices: blade_graphics::BufferPiece,
    base_vertices: blade_graphics::BufferPiece,
    mesh_indices: blade_graphics::BufferPiece,
    weld_keys: blade_graphics::BufferPiece,
    slot_positions: blade_graphics::BufferPiece,
    pass_vertex_slots: blade_graphics::BufferPiece,
    vertex_offsets: blade_graphics::BufferPiece,
    normal_sums: blade_graphics::BufferPiece,
    neighbor_sums: blade_graphics::BufferPiece,
    smooth_grads: blade_graphics::BufferPiece,
    vertex_pass_grads: blade_graphics::BufferPiece,
    pass_normal_grads: blade_graphics::BufferPiece,
}

/// A model whose vertex positions are parameters.
struct VertexModel {
    model: blade_asset::Handle<crate::Model>,
    /// Where its indices start in the mesh readback, and their count.
    index_readback: (u64, u64),
    vertex_buffer: blade_graphics::Buffer,
    index_buffer: blade_graphics::Buffer,
    acceleration_structure: blade_graphics::AccelerationStructure,
    meshes: Vec<blade_graphics::AccelerationStructureMesh>,
    /// First index, first vertex, and triangle count of every geometry.
    geometries: Vec<(u32, u32, u32)>,
    /// Where its vertices start in the slot table.
    first_vertex: u32,
    vertex_count: u32,
    /// Its vertices as they were loaded.
    base: blade_graphics::Buffer,
    /// For building its acceleration structure again.
    scratch: blade_graphics::Buffer,
}

/// The positions of the vertices of models, as parameters.
struct VertexParams {
    models: Vec<VertexModel>,
    /// Offsets of the slots, which copies of a vertex share.
    group: ParamGroup,
    slot_count: u32,
    /// The slot of every vertex.
    vertex_slots: blade_graphics::Buffer,
    weld_keys: blade_graphics::Buffer,
    /// Where the vertex of every slot was loaded, with 1 in W, or zeros.
    slot_positions: blade_graphics::Buffer,
    normal_sums: blade_graphics::Buffer,
    /// By the normal of every slot, until the step carries it to positions.
    normal_grads: blade_graphics::Buffer,
    neighbor_sums: blade_graphics::Buffer,
    /// The positions and offsets of the slots, for the host to read.
    readback: blade_graphics::Buffer,
    /// The slots of the vertices, then the indices of every model.
    mesh_readback: blade_graphics::Buffer,
    vertex_count: u32,
}

impl VertexParams {
    fn pass_data(&self, vertex_pass: VertexPass, model: Option<&VertexModel>) -> VertexPassData {
        let (mesh_vertices, base_vertices, mesh_indices) = match model {
            Some(vm) => (
                vm.vertex_buffer.into(),
                vm.base.into(),
                vm.index_buffer.into(),
            ),
            None => (
                self.slot_positions.into(),
                self.slot_positions.into(),
                self.vertex_slots.into(),
            ),
        };
        VertexPassData {
            vertex_pass,
            mesh_vertices,
            base_vertices,
            mesh_indices,
            weld_keys: self.weld_keys.into(),
            slot_positions: self.slot_positions.into(),
            pass_vertex_slots: self.vertex_slots.into(),
            vertex_offsets: self.group.params.into(),
            normal_sums: self.normal_sums.into(),
            neighbor_sums: self.neighbor_sums.into(),
            smooth_grads: self.group.grads.into(),
            vertex_pass_grads: self.group.grads.into(),
            pass_normal_grads: self.normal_grads.into(),
        }
    }

    /// Run `pipeline` over the vertices of every model, or with
    /// `per_geometry`, over the triangles of every geometry.
    fn dispatch(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        label: &str,
        pipeline: &blade_graphics::ComputePipeline,
        per_geometry: bool,
        smoothness: f32,
    ) {
        let mut pass = encoder.compute(label);
        let mut pc = pass.with(pipeline);
        for (index, vm) in self.models.iter().enumerate() {
            let base = VertexPass {
                first_vertex: vm.first_vertex,
                vertex_count: vm.vertex_count,
                model: index as u32,
                smoothness,
                slot_count: self.slot_count,
                ..Default::default()
            };
            if per_geometry {
                for &(first_index, geometry_vertex_start, triangle_count) in &vm.geometries {
                    let vertex_pass = VertexPass {
                        first_index,
                        geometry_vertex_start,
                        triangle_count,
                        ..base
                    };
                    pc.bind(0, &self.pass_data(vertex_pass, Some(vm)));
                    pc.dispatch(flat_dispatch(triangle_count));
                }
            } else {
                pc.bind(0, &self.pass_data(base, Some(vm)));
                pc.dispatch(flat_dispatch(vm.vertex_count));
            }
        }
    }

    fn model_meshes(model: &crate::Model) -> Vec<blade_graphics::AccelerationStructureMesh> {
        let stride = mem::size_of::<crate::Vertex>() as u32;
        model
            .geometries
            .iter()
            .enumerate()
            .map(
                |(index, geometry)| blade_graphics::AccelerationStructureMesh {
                    vertex_data: model
                        .vertex_buffer
                        .at(u64::from(geometry.vertex_range.start) * u64::from(stride)),
                    vertex_format: blade_graphics::VertexFormat::F32Vec3,
                    vertex_stride: stride,
                    vertex_count: geometry.vertex_range.end - geometry.vertex_range.start,
                    index_data: model.index_buffer.at(geometry.index_offset),
                    index_type: geometry.index_type,
                    triangle_count: geometry.triangle_count,
                    transform_data: model.transform_buffer.at((index
                        * mem::size_of::<blade_graphics::AccelerationStructureTransform>())
                        as u64),
                    is_opaque: !model.materials[geometry.material_index].transparent,
                },
            )
            .collect()
    }

    fn destroy(&mut self, gpu: &blade_graphics::Context) {
        for vm in self.models.drain(..) {
            gpu.destroy_buffer(vm.base);
            gpu.destroy_buffer(vm.scratch);
        }
        self.group.destroy(gpu);
        for buffer in [
            self.vertex_slots,
            self.weld_keys,
            self.slot_positions,
            self.normal_sums,
            self.normal_grads,
            self.neighbor_sums,
            self.readback,
            self.mesh_readback,
        ] {
            gpu.destroy_buffer(buffer);
        }
    }
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
    edge_entries: blade_graphics::BufferPiece,
    instance_transforms: blade_graphics::BufferPiece,
    translation_grads: blade_graphics::BufferPiece,
    rotation_grads: blade_graphics::BufferPiece,
    rotation_params: blade_graphics::BufferPiece,
    camera_grads: blade_graphics::BufferPiece,
    edge_keys: blade_graphics::BufferPiece,
    edge_counts: blade_graphics::BufferPiece,
    edge_normals: blade_graphics::BufferPiece,
    edge_triangles: blade_graphics::BufferPiece,
    edge_hits: blade_graphics::BufferPiece,
    edge_importance: blade_graphics::BufferPiece,
    edge_guide: blade_graphics::BufferPiece,
    emitters: blade_graphics::BufferPiece,
    vertex_slots: blade_graphics::BufferPiece,
    vertex_grads: blade_graphics::BufferPiece,
    normal_grads: blade_graphics::BufferPiece,
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct ControlParams {
    pixel_count: u32,
    valid: u32,
    weight: f32,
    _pad: u32,
}

#[derive(blade_macros::ShaderData)]
struct ControlData {
    control: ControlParams,
    control_image: blade_graphics::BufferPiece,
    control_previous: blade_graphics::BufferPiece,
    control_estimate: blade_graphics::BufferPiece,
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

/// The compiled passes of [`DiffTracer`], which outlive it through
/// [`DiffTracer::rebuild`].
struct DiffPipelines {
    control: blade_graphics::ComputePipeline,
    collect_emitters: blade_graphics::ComputePipeline,
    primal: blade_graphics::ComputePipeline,
    adjoint: blade_graphics::ComputePipeline,
    init_materials: blade_graphics::ComputePipeline,
    init_environment: blade_graphics::ComputePipeline,
    init_texture: blade_graphics::ComputePipeline,
    loss: blade_graphics::ComputePipeline,
    adam: blade_graphics::ComputePipeline,
    edges: blade_graphics::ComputePipeline,
    count_edges: blade_graphics::ComputePipeline,
    guide_edges: blade_graphics::ComputePipeline,
    weld: blade_graphics::ComputePipeline,
    apply: blade_graphics::ComputePipeline,
    sum_normals: blade_graphics::ComputePipeline,
    write_normals: blade_graphics::ComputePipeline,
    sum_neighbors: blade_graphics::ComputePipeline,
    smoothness: blade_graphics::ComputePipeline,
    normals_to_positions: blade_graphics::ComputePipeline,
}

impl DiffPipelines {
    fn new(
        gpu: &blade_graphics::Context,
        sh_trace: &blade_graphics::Shader,
        sh_optimize: &blade_graphics::Shader,
    ) -> Self {
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
        let edges_pipeline = trace_pipeline("diff-edges", "edges");
        let count_edges_pipeline = trace_pipeline("diff-count-edges", "count_edges");
        let guide_edges_pipeline = trace_pipeline("diff-guide-edges", "guide_edges");
        let collect_emitters_pipeline = trace_pipeline("diff-collect-emitters", "collect_emitters");
        let texture_init_layout = super::ray_data_layout::<TextureInitData>(gpu);
        let init_texture_pipeline =
            gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
                name: "diff-init-texture",
                data_layouts: &[&texture_init_layout],
                compute: sh_trace.at("init_texture"),
            });
        let loss_layout = <LossData as blade_graphics::ShaderData>::layout();
        let control_layout = <ControlData as blade_graphics::ShaderData>::layout();
        let control_pipeline = gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
            name: "diff-control-variate",
            data_layouts: &[&control_layout],
            compute: sh_optimize.at("control_variate"),
        });
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

        let vertex_layout = <VertexPassData as blade_graphics::ShaderData>::layout();
        let vertex_pipeline = |name, entry| {
            gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
                name,
                data_layouts: &[&vertex_layout],
                compute: sh_trace.at(entry),
            })
        };
        Self {
            primal: primal_pipeline,
            adjoint: adjoint_pipeline,
            init_materials: init_materials_pipeline,
            init_environment: init_environment_pipeline,
            init_texture: init_texture_pipeline,
            loss: loss_pipeline,
            control: control_pipeline,
            adam: adam_pipeline,
            edges: edges_pipeline,
            count_edges: count_edges_pipeline,
            guide_edges: guide_edges_pipeline,
            collect_emitters: collect_emitters_pipeline,
            weld: vertex_pipeline("diff-weld-vertices", "weld_vertices"),
            apply: vertex_pipeline("diff-apply-vertices", "apply_vertices"),
            sum_normals: vertex_pipeline("diff-sum-normals", "sum_normals"),
            write_normals: vertex_pipeline("diff-write-normals", "write_normals"),
            sum_neighbors: vertex_pipeline("diff-sum-neighbors", "sum_neighbors"),
            smoothness: vertex_pipeline("diff-add-smoothness", "add_smoothness"),
            normals_to_positions: vertex_pipeline(
                "diff-normals-to-positions",
                "normals_to_positions",
            ),
        }
    }

    fn destroy(&mut self, gpu: &blade_graphics::Context) {
        gpu.destroy_compute_pipeline(&mut self.collect_emitters);
        gpu.destroy_compute_pipeline(&mut self.control);
        gpu.destroy_compute_pipeline(&mut self.primal);
        gpu.destroy_compute_pipeline(&mut self.adjoint);
        gpu.destroy_compute_pipeline(&mut self.init_materials);
        gpu.destroy_compute_pipeline(&mut self.init_environment);
        gpu.destroy_compute_pipeline(&mut self.init_texture);
        gpu.destroy_compute_pipeline(&mut self.loss);
        gpu.destroy_compute_pipeline(&mut self.adam);
        gpu.destroy_compute_pipeline(&mut self.edges);
        gpu.destroy_compute_pipeline(&mut self.count_edges);
        gpu.destroy_compute_pipeline(&mut self.guide_edges);
        gpu.destroy_compute_pipeline(&mut self.weld);
        gpu.destroy_compute_pipeline(&mut self.apply);
        gpu.destroy_compute_pipeline(&mut self.sum_normals);
        gpu.destroy_compute_pipeline(&mut self.write_normals);
        gpu.destroy_compute_pipeline(&mut self.sum_neighbors);
        gpu.destroy_compute_pipeline(&mut self.smoothness);
        gpu.destroy_compute_pipeline(&mut self.normals_to_positions);
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
    /// Only `None` while [`Self::rebuild`] moves them on.
    pipelines: Option<DiffPipelines>,
    materials: ParamGroup,
    environment: ParamGroup,
    environment_size: blade_graphics::Extent,
    textures: ParamGroup,
    texture_list: Vec<DiffTexture>,
    texture_regions: blade_graphics::Buffer,
    vertex_params: VertexParams,
    geometry: bool,
    /// Hash table of the edges, with the triangles along each.
    edge_keys: blade_graphics::Buffer,
    edge_counts: blade_graphics::Buffer,
    edge_normals: blade_graphics::Buffer,
    edge_triangles: blade_graphics::Buffer,
    /// What the secondary edges drawn from every triangle corner brought,
    /// their importance, and its cumulative distribution.
    edge_hits: blade_graphics::Buffer,
    edge_importance: blade_graphics::Buffer,
    edge_guide: blade_graphics::Buffer,
    edge_guide_ready: bool,
    emitters: blade_graphics::Buffer,
    /// For the control variate: the parameters of the last estimate of
    /// every view, the image of them, and the estimate.
    control: ControlState,
    /// Translations of the instances, in XYZ of 4 floats.
    translations: ParamGroup,
    /// Rotations of the instances about their origins, as axis times angle.
    rotations: ParamGroup,
    camera_position: ParamGroup,
    edge_entries: blade_graphics::Buffer,
    instance_transforms: blade_graphics::Buffer,
    instance_count: u32,
    triangle_count: u32,
    /// Uploaded at creation, and kept until destruction.
    staging: Vec<blade_graphics::Buffer>,
    image_size: blade_graphics::Extent,
    image: blade_graphics::Buffer,
    target: blade_graphics::Buffer,
    image_adjoint: blade_graphics::Buffer,
    loss: blade_graphics::Buffer,
    /// Parameters and gradients of the materials and the geometry,
    /// for the host to read.
    readback: blade_graphics::Buffer,
    /// Frames already averaged into the image of every view.
    accumulated_frames: Vec<u32>,
    view: u32,
    view_count: u32,
    /// Bytes between the images of two views.
    view_stride: u64,
}

/// Bytes between the losses of two views, for any storage offset alignment.
const LOSS_STRIDE: u64 = 256;

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
        sh_trace.check_struct_size::<EdgeEntry>();
        let sh_optimize = shader_man[shaders.optimize].raw.as_ref().unwrap();
        sh_optimize.check_struct_size::<LossParams>();
        sh_optimize.check_struct_size::<AdamParams>();
        let pipelines = DiffPipelines::new(gpu, sh_trace, sh_optimize);
        Self::with_pipelines(pipelines, encoder, gpu, asset_hub, ray_tracer, desc)
    }

    /// Start again from the scene of `ray_tracer`, which can have other
    /// geometry, keeping the compiled passes, which takes much less than
    /// [`Self::new`]. The targets stay, if the images are of the same size
    /// and as many, and everything else starts over, including the
    /// parameters and the moments of the optimizer.
    ///
    /// The work submitted with this tracer has to be done.
    pub fn rebuild(
        &mut self,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
        asset_hub: &crate::AssetHub,
        ray_tracer: &super::RayTracer,
        desc: &DiffTracerDesc,
    ) {
        let target = self.target;
        let layout = (self.image_size, self.view_count);
        self.destroy_resources(gpu, false);
        let pipelines = self.pipelines.take().unwrap();
        let mut this = Self::with_pipelines(pipelines, encoder, gpu, asset_hub, ray_tracer, desc);
        if (this.image_size, this.view_count) == layout {
            gpu.destroy_buffer(this.target);
            this.target = target;
            this.view = self.view;
        } else {
            gpu.destroy_buffer(target);
        }
        *self = this;
    }

    fn with_pipelines(
        pipelines: DiffPipelines,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
        asset_hub: &crate::AssetHub,
        ray_tracer: &super::RayTracer,
        desc: &DiffTracerDesc,
    ) -> Self {
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
                        material.normal_texture,
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

        assert!(
            desc.geometry || !desc.vertices,
            "vertex positions need the gradient of the geometry"
        );
        // The models whose vertices are parameters, once each.
        let mut vertex_models = Vec::<(blade_asset::Handle<crate::Model>, u32, u32)>::new();
        let mut vertex_count = 0;
        if desc.vertices {
            for &handle in &ray_tracer.scene_models {
                if vertex_models.iter().all(|&(h, _, _)| h != handle) {
                    let count = asset_hub.models[handle]
                        .geometries
                        .iter()
                        .map(|g| g.vertex_range.end)
                        .max()
                        .unwrap_or(0);
                    vertex_models.push((handle, vertex_count, count));
                    vertex_count += count;
                }
            }
        }
        // Every triangle of the scene, in the order of the hit entries.
        let mut edge_table = Vec::new();
        let mut triangle_count = 0;
        // For the emitters, as well as the edges.
        {
            for (instance, &model) in ray_tracer.scene_models.iter().enumerate() {
                let first_vertex = vertex_models
                    .iter()
                    .find(|&&(h, _, _)| h == model)
                    .map(|&(_, first, _)| first);
                for geometry in &asset_hub.models[model].geometries {
                    edge_table.push(EdgeEntry {
                        hit_entry: edge_table.len() as u32,
                        instance: instance as u32,
                        first_triangle: triangle_count,
                        triangle_count: geometry.triangle_count,
                        vertex_base: first_vertex
                            .map_or(!0, |first| first + geometry.vertex_range.start),
                        _pad: [0; 3],
                    });
                    triangle_count += geometry.triangle_count;
                }
            }
        }
        let slot_count = (2 * vertex_count).max(64).next_power_of_two();
        let device_buffer = |name: &str, size: u64| {
            gpu.create_buffer(blade_graphics::BufferDesc {
                name,
                size,
                memory: blade_graphics::Memory::Device,
            })
        };
        let mut index_readback_end = 4 * u64::from(vertex_count);
        let vertex_params = VertexParams {
            models: vertex_models
                .iter()
                .map(|&(handle, first_vertex, count)| {
                    let model = &asset_hub.models[handle];
                    let meshes = VertexParams::model_meshes(model);
                    let sizes = gpu.get_bottom_level_acceleration_structure_sizes(&meshes);
                    let index_bytes = model
                        .geometries
                        .iter()
                        .map(|g| g.index_offset + 12 * u64::from(g.triangle_count))
                        .max()
                        .unwrap_or(0);
                    let index_readback = (index_readback_end, index_bytes);
                    index_readback_end += index_bytes;
                    VertexModel {
                        model: handle,
                        index_readback,
                        vertex_buffer: model.vertex_buffer,
                        index_buffer: model.index_buffer,
                        acceleration_structure: model.acceleration_structure,
                        meshes,
                        geometries: model
                            .geometries
                            .iter()
                            .map(|g| {
                                assert_eq!(
                                    g.index_type,
                                    Some(blade_graphics::IndexType::U32),
                                    "vertex positions need 32-bit indices"
                                );
                                (
                                    (g.index_offset / 4) as u32,
                                    g.vertex_range.start,
                                    g.triangle_count,
                                )
                            })
                            .collect(),
                        first_vertex,
                        vertex_count: count,
                        base: device_buffer(
                            "diff base vertices",
                            (u64::from(count) * mem::size_of::<crate::Vertex>() as u64).max(16),
                        ),
                        scratch: device_buffer("diff BLAS scratch", sizes.scratch.max(16)),
                    }
                })
                .collect(),
            group: ParamGroup::new("diff vertices", slot_count * 4, gpu),
            slot_count,
            vertex_slots: device_buffer("diff vertex slots", 4 * u64::from(vertex_count.max(4))),
            weld_keys: device_buffer("diff weld keys", 4 * u64::from(slot_count)),
            slot_positions: device_buffer("diff slot positions", 16 * u64::from(slot_count)),
            normal_sums: device_buffer("diff normal sums", 16 * u64::from(slot_count)),
            normal_grads: device_buffer("diff normal grads", 16 * u64::from(slot_count)),
            neighbor_sums: device_buffer("diff neighbor sums", 16 * u64::from(slot_count)),
            readback: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "diff vertex readback",
                size: 48 * u64::from(slot_count),
                memory: blade_graphics::Memory::Download,
            }),
            mesh_readback: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "diff mesh readback",
                size: index_readback_end.max(16),
                memory: blade_graphics::Memory::Download,
            }),
            vertex_count,
        };
        let instance_count = if desc.geometry {
            ray_tracer.scene_models.len() as u32
        } else {
            0
        };
        let upload = |name: &str, bytes: &[u8]| {
            let size = bytes.len().max(16) as u64;
            let buffer = gpu.create_buffer(blade_graphics::BufferDesc {
                name,
                size,
                memory: blade_graphics::Memory::Device,
            });
            let staging = gpu.create_buffer(blade_graphics::BufferDesc {
                name: &format!("{name} staging"),
                size,
                memory: blade_graphics::Memory::Upload,
            });
            unsafe {
                slice::from_raw_parts_mut(staging.data(), bytes.len()).copy_from_slice(bytes);
            }
            (buffer, staging, size)
        };
        let (edge_entries, edge_staging, edge_bytes) =
            upload("diff edge entries", bytemuck::cast_slice(&edge_table));
        let transforms = instance_transform_data(ray_tracer);
        let (instance_transforms, transform_staging, transform_bytes) = upload(
            "diff instance transforms",
            bytemuck::cast_slice(&transforms),
        );
        // Half full at most, so that probing stays short.
        let edge_slots = (6 * u64::from(triangle_count)).max(64).next_power_of_two();
        let edge_table = |name| {
            gpu.create_buffer(blade_graphics::BufferDesc {
                name,
                size: 4 * edge_slots,
                memory: blade_graphics::Memory::Device,
            })
        };
        let edge_keys = edge_table("diff edge keys");
        let edge_counts = edge_table("diff edge counts");
        let edge_normals = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "diff edge normals",
            size: 32 * edge_slots,
            memory: blade_graphics::Memory::Device,
        });
        let edge_triangles = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "diff edge triangles",
            size: 8 * edge_slots,
            memory: blade_graphics::Memory::Device,
        });
        let edge_array = |name| {
            gpu.create_buffer(blade_graphics::BufferDesc {
                name,
                size: 4 * u64::from(3 * triangle_count).max(1),
                memory: blade_graphics::Memory::Device,
            })
        };
        let edge_hits = edge_array("diff edge hits");
        let edge_importance = edge_array("diff edge importance");
        let edge_guide = edge_array("diff edge guide");
        let emitters = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "diff emitters",
            size: 4 * (1 + u64::from(triangle_count)),
            memory: blade_graphics::Memory::Device,
        });
        let translations = ParamGroup::new("diff translations", instance_count * 4, gpu);
        let rotations = ParamGroup::new("diff rotations", instance_count * 4, gpu);
        let camera_position = ParamGroup::new("diff camera position", 4, gpu);

        let image_size = ray_tracer.surface_size;
        let image_bytes = u64::from(image_size.width * image_size.height) * 16;
        let view_count = desc.views.max(1);
        let view_stride = image_bytes.next_multiple_of(256);
        let image_buffer = |name| {
            gpu.create_buffer(blade_graphics::BufferDesc {
                name,
                size: view_stride * u64::from(view_count),
                memory: blade_graphics::Memory::Device,
            })
        };
        let control = ControlState::new(
            gpu,
            view_count,
            view_stride,
            [
                materials.params.size(),
                environment.params.size(),
                textures.params.size(),
            ],
        );
        let this = Self {
            pipelines: Some(pipelines),
            materials,
            environment,
            environment_size,
            textures,
            texture_list: texture_list.iter().map(|&(_, texture)| texture).collect(),
            texture_regions,
            vertex_params,
            geometry: desc.geometry,
            edge_keys,
            edge_counts,
            edge_normals,
            edge_triangles,
            edge_hits,
            edge_importance,
            edge_guide,
            edge_guide_ready: false,
            emitters,
            control,
            translations,
            rotations,
            camera_position,
            edge_entries,
            instance_transforms,
            instance_count,
            triangle_count,
            staging: vec![texture_regions_staging, edge_staging, transform_staging],
            image_size,
            image: image_buffer("diff image"),
            target: image_buffer("diff target"),
            image_adjoint: image_buffer("diff image adjoint"),
            loss: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "diff loss",
                size: LOSS_STRIDE * u64::from(view_count),
                memory: blade_graphics::Memory::Shared,
            }),
            readback: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "diff readback",
                size: 2
                    * (u64::from(material_count) * mem::size_of::<MaterialParams>() as u64
                        + u64::from(2 * instance_count + 1) * 16),
                memory: blade_graphics::Memory::Download,
            }),
            accumulated_frames: vec![0; view_count as usize],
            view: 0,
            view_count,
            view_stride,
        };

        this.materials.clear(encoder);
        this.environment.clear(encoder);
        this.textures.clear(encoder);
        this.translations.clear(encoder);
        this.rotations.clear(encoder);
        this.camera_position.clear(encoder);
        this.vertex_params.group.clear(encoder);
        {
            let mut transfer = encoder.transfer("diff-init");
            transfer.fill_buffer(this.image_adjoint.into(), this.image_adjoint.size(), 0);
            transfer.fill_buffer(this.loss.into(), this.loss.size(), 0);
            transfer.copy_buffer_to_buffer(
                texture_regions_staging.into(),
                texture_regions.into(),
                region_bytes,
            );
            transfer.copy_buffer_to_buffer(edge_staging.into(), edge_entries.into(), edge_bytes);
            transfer.fill_buffer(edge_keys.into(), edge_keys.size(), 0);
            let vp = &this.vertex_params;
            for vm in &vp.models {
                let bytes = u64::from(vm.vertex_count) * mem::size_of::<crate::Vertex>() as u64;
                if bytes != 0 {
                    transfer.copy_buffer_to_buffer(vm.vertex_buffer.at(0), vm.base.at(0), bytes);
                }
            }
            for buffer in [
                vp.weld_keys,
                vp.slot_positions,
                vp.group.params,
                vp.normal_grads,
                vp.normal_sums,
            ] {
                transfer.fill_buffer(buffer.into(), buffer.size(), 0);
            }
            transfer.fill_buffer(edge_counts.into(), edge_counts.size(), 0);
            for buffer in [edge_hits, edge_importance, edge_guide] {
                transfer.fill_buffer(buffer.into(), buffer.size(), 0);
            }
            transfer.copy_buffer_to_buffer(
                transform_staging.into(),
                instance_transforms.into(),
                transform_bytes,
            );
            // The offsets start at zero.
            for group in [&this.translations, &this.rotations] {
                transfer.fill_buffer(group.params.into(), group.params.size(), 0);
            }
            transfer.fill_buffer(
                this.camera_position.params.into(),
                this.camera_position.params.size(),
                0,
            );
        }
        let camera = crate::CameraParams::default();
        let parameters = bytemuck::Zeroable::zeroed();
        let mut pass = encoder.compute("diff-init");
        {
            let mut pc = pass.with(&this.pipelines.as_ref().unwrap().init_materials);
            pc.bind(0, &this.trace_data(ray_tracer, camera, parameters));
            let wg = this
                .pipelines
                .as_ref()
                .unwrap()
                .init_materials
                .get_workgroup_size();
            pc.dispatch([material_count.div_ceil(wg[0]).max(1), 1, 1]);
        }
        {
            let mut pc = pass.with(&this.pipelines.as_ref().unwrap().init_environment);
            pc.bind(0, &this.trace_data(ray_tracer, camera, parameters));
            pc.dispatch(
                this.pipelines
                    .as_ref()
                    .unwrap()
                    .init_environment
                    .get_dispatch_for(environment_size),
            );
        }
        if this.geometry && this.triangle_count != 0 {
            let mut pc = pass.with(&this.pipelines.as_ref().unwrap().count_edges);
            let mut parameters: DiffParams = bytemuck::Zeroable::zeroed();
            parameters.triangle_count = this.triangle_count;
            pc.bind(0, &this.trace_data(ray_tracer, camera, parameters));
            pc.dispatch(flat_dispatch(this.triangle_count));
        }
        if !texture_list.is_empty() {
            let mut pc = pass.with(&this.pipelines.as_ref().unwrap().init_texture);
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
                    this.pipelines
                        .as_ref()
                        .unwrap()
                        .init_texture
                        .get_dispatch_for(blade_graphics::Extent {
                            width: texture.width,
                            height: texture.height,
                            depth: 1,
                        }),
                );
            }
        }
        drop(pass);
        this.vertex_params.dispatch(
            encoder,
            "diff-weld-vertices",
            &this.pipelines.as_ref().unwrap().weld,
            false,
            0.0,
        );
        // The step reads the normals of the slots, which the vertices only
        // get as offsets are applied, so start them from where they are.
        this.vertex_params.dispatch(
            encoder,
            "diff-sum-normals",
            &this.pipelines.as_ref().unwrap().sum_normals,
            true,
            0.0,
        );
        this
    }

    /// Take the instance transforms of the scene that `ray_tracer` built
    /// last, which has to have the same objects and models as when this was
    /// created. The staging buffer goes to `temp`.
    pub fn update_geometry(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
        ray_tracer: &super::RayTracer,
        temp: &mut crate::FrameResources,
    ) {
        self.control.invalidate();
        if !self.geometry {
            return;
        }
        assert_eq!(
            ray_tracer.instance_transforms.len(),
            self.instance_count as usize
        );
        let transforms = instance_transform_data(ray_tracer);
        let bytes = (transforms.len() * mem::size_of::<[f32; 16]>()) as u64;
        let staging = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "diff instance transforms staging",
            size: bytes,
            memory: blade_graphics::Memory::Upload,
        });
        unsafe {
            slice::from_raw_parts_mut(staging.data() as *mut [f32; 16], transforms.len())
                .copy_from_slice(&transforms);
        }
        encoder
            .transfer("diff-instance-transforms")
            .copy_buffer_to_buffer(staging.into(), self.instance_transforms.into(), bytes);
        temp.buffers.push(staging);
        // The edges are told apart by their positions, which the vertices
        // move, so they are counted again for where they are now.
        if !self.vertex_params.models.is_empty() && self.triangle_count != 0 {
            let mut transfer = encoder.transfer("diff-clear-edges");
            transfer.fill_buffer(self.edge_keys.into(), self.edge_keys.size(), 0);
            transfer.fill_buffer(self.edge_counts.into(), self.edge_counts.size(), 0);
            drop(transfer);
            let mut pass = encoder.compute("diff-count-edges");
            let mut pc = pass.with(&self.pipelines.as_ref().unwrap().count_edges);
            let mut parameters: DiffParams = bytemuck::Zeroable::zeroed();
            parameters.triangle_count = self.triangle_count;
            pc.bind(
                0,
                &self.trace_data(ray_tracer, crate::CameraParams::default(), parameters),
            );
            pc.dispatch(flat_dispatch(self.triangle_count));
        }
    }

    /// Make a guide for drawing secondary edges from what they brought since
    /// the last one, with [`DiffConfig::guided_edges`], along with half of
    /// the importance of the edges of that one. Calling this every few
    /// steps keeps the guide up with the geometry.
    pub fn refresh_edge_guide(
        &mut self,
        encoder: &mut blade_graphics::CommandEncoder,
        ray_tracer: &super::RayTracer,
    ) {
        if !self.geometry || self.triangle_count == 0 {
            return;
        }
        let mut pass = encoder.compute("diff-guide-edges");
        let mut pc = pass.with(&self.pipelines.as_ref().unwrap().guide_edges);
        let mut parameters: DiffParams = bytemuck::Zeroable::zeroed();
        parameters.triangle_count = self.triangle_count;
        pc.bind(
            0,
            &self.trace_data(ray_tracer, crate::CameraParams::default(), parameters),
        );
        pc.dispatch([1, 1, 1]);
        self.edge_guide_ready = true;
    }

    /// List the emitting triangles for the passes that follow, by the
    /// current material parameters.
    fn collect_emitters(&self, encoder: &mut blade_graphics::CommandEncoder, data: &DiffTraceData) {
        encoder
            .transfer("diff-clear-emitters")
            .fill_buffer(self.emitters.at(0), 4, 0);
        if self.triangle_count == 0 {
            return;
        }
        let mut pass = encoder.compute("diff-collect-emitters");
        let mut pc = pass.with(&self.pipelines.as_ref().unwrap().collect_emitters);
        pc.bind(0, data);
        pc.dispatch(flat_dispatch(self.triangle_count));
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
            image: self.view_piece(self.image),
            image_adjoint: self.view_piece(self.image_adjoint),
            material_grads: self.materials.grads.into(),
            env_grads: self.environment.grads.into(),
            texture_regions: self.texture_regions.into(),
            texture_params: self.textures.params.into(),
            texture_grads: self.textures.grads.into(),
            init_material_params: self.materials.params.into(),
            init_env_params: self.environment.params.into(),
            edge_entries: self.edge_entries.into(),
            instance_transforms: self.instance_transforms.into(),
            translation_grads: self.translations.grads.into(),
            rotation_grads: self.rotations.grads.into(),
            rotation_params: self.rotations.params.into(),
            camera_grads: self.camera_position.grads.into(),
            edge_keys: self.edge_keys.into(),
            edge_counts: self.edge_counts.into(),
            edge_normals: self.edge_normals.into(),
            edge_triangles: self.edge_triangles.into(),
            edge_hits: self.edge_hits.into(),
            edge_importance: self.edge_importance.into(),
            edge_guide: self.edge_guide.into(),
            emitters: self.emitters.into(),
            vertex_slots: self.vertex_params.vertex_slots.into(),
            vertex_grads: self.vertex_params.group.grads.into(),
            normal_grads: self.vertex_params.normal_grads.into(),
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
            accumulated_frames: self.accumulated_frames[self.view as usize],
            jitter_primary_rays: config.jitter_primary_rays as u32,
            geometry: self.geometry as u32,
            instance_count: self.instance_count,
            triangle_count: self.triangle_count,
            edge_samples: config.edge_samples,
            secondary_edge_samples: if self.geometry {
                config.secondary_edge_samples
            } else {
                0
            },
            secondary_edge_depth: config.secondary_edge_depth,
            guided_edges: (config.guided_edges && self.edge_guide_ready) as u32,
            antithetic: config.antithetic as u32,
            emitter_samples: config.emitter_samples,
            projective_edges: config.projective_edges as u32,
            _pad: [0; 2],
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

    /// Copy the parameters and the gradients of the materials and of the
    /// geometry, as they are at this point of the `encoder`, to where
    /// [`Self::material_params`] and the like read them once the work is done.
    pub fn read_back(&self, encoder: &mut blade_graphics::CommandEncoder) {
        let mut transfer = encoder.transfer("diff-read-back");
        let mut offset = 0;
        for group in [
            &self.materials,
            &self.translations,
            &self.rotations,
            &self.camera_position,
        ] {
            let bytes = u64::from(group.count) * 4;
            for buffer in [group.params, group.grads] {
                if bytes != 0 {
                    transfer.copy_buffer_to_buffer(buffer.into(), self.readback.at(offset), bytes);
                }
                offset += bytes;
            }
        }
    }

    fn material_bytes(&self) -> u64 {
        (self.material_count() * mem::size_of::<MaterialParams>()) as u64
    }

    fn read_back_slice<T: bytemuck::Pod>(&self, offset: u64, count: usize) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.readback.data().add(offset as usize) as *const T, count)
        }
    }

    /// The material parameters, as of the last [`Self::read_back`], once its
    /// work is done.
    pub fn material_params(&self) -> &[MaterialParams] {
        self.read_back_slice(0, self.material_count())
    }

    /// Derivative of the loss by the material parameters, accumulated by
    /// [`Self::backward`] since the last [`Self::step`], as of the last
    /// [`Self::read_back`], once its work is done.
    pub fn material_grads(&self) -> &[MaterialParams] {
        self.read_back_slice(self.material_bytes(), self.material_count())
    }

    fn geometry_offset(&self) -> u64 {
        2 * self.material_bytes()
    }

    /// Translation offset of every object, in XYZ, as of the last
    /// [`Self::read_back`]. See [`DiffTracerDesc::geometry`].
    pub fn translations(&self) -> &[[f32; 4]] {
        self.read_back_slice(self.geometry_offset(), self.instance_count as usize)
    }

    /// Derivative of the loss by [`Self::translations`].
    pub fn translation_grads(&self) -> &[[f32; 4]] {
        let offset = self.geometry_offset() + u64::from(self.instance_count) * 16;
        self.read_back_slice(offset, self.instance_count as usize)
    }

    /// Rotation offset of every object about its origin, as an axis times
    /// the angle in radians, as of the last [`Self::read_back`].
    pub fn rotations(&self) -> &[[f32; 4]] {
        let offset = self.geometry_offset() + u64::from(self.instance_count) * 32;
        self.read_back_slice(offset, self.instance_count as usize)
    }

    /// Derivative of the loss by [`Self::rotations`].
    pub fn rotation_grads(&self) -> &[[f32; 4]] {
        let offset = self.geometry_offset() + u64::from(self.instance_count) * 48;
        self.read_back_slice(offset, self.instance_count as usize)
    }

    /// Translation offset of the camera, as of the last [`Self::read_back`].
    pub fn camera_offset(&self) -> [f32; 3] {
        let offset = self.geometry_offset() + u64::from(self.instance_count) * 64;
        let [x, y, z, _] = self.read_back_slice::<[f32; 4]>(offset, 1)[0];
        [x, y, z]
    }

    /// Derivative of the loss by [`Self::camera_offset`].
    pub fn camera_offset_grad(&self) -> [f32; 3] {
        let offset = self.geometry_offset() + u64::from(self.instance_count) * 64 + 16;
        let [x, y, z, _] = self.read_back_slice::<[f32; 4]>(offset, 1)[0];
        [x, y, z]
    }

    /// All the offsets of the geometry, as of the last [`Self::read_back`].
    pub fn geometry_offsets(&self) -> GeometryOffsets {
        let xyz = |v: &[f32; 4]| [v[0], v[1], v[2]];
        GeometryOffsets {
            translations: self.translations().iter().map(xyz).collect(),
            rotations: self.rotations().iter().map(xyz).collect(),
            camera: self.camera_offset(),
        }
    }

    /// Replace the offsets of the geometry at this point of the `encoder`,
    /// such as to start from a guess. The staging buffer goes to `temp`.
    pub fn write_geometry_offsets(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
        offsets: &GeometryOffsets,
        temp: &mut crate::FrameResources,
    ) {
        self.control.invalidate();
        let count = self.instance_count as usize;
        assert_eq!(offsets.translations.len(), count);
        assert_eq!(offsets.rotations.len(), count);
        let xyzw = |v: &[f32; 3]| [v[0], v[1], v[2], 0.0];
        let mut data = Vec::with_capacity(2 * count + 1);
        data.extend(offsets.translations.iter().map(xyzw));
        data.extend(offsets.rotations.iter().map(xyzw));
        data.push(xyzw(&offsets.camera));
        let staging = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "diff geometry staging",
            size: (data.len() * 16) as u64,
            memory: blade_graphics::Memory::Upload,
        });
        unsafe {
            slice::from_raw_parts_mut(staging.data() as *mut [f32; 4], data.len())
                .copy_from_slice(&data);
        }
        let bytes = (count * 16) as u64;
        let mut transfer = encoder.transfer("diff-write-geometry");
        if count != 0 {
            transfer.copy_buffer_to_buffer(staging.at(0), self.translations.params.into(), bytes);
            transfer.copy_buffer_to_buffer(staging.at(bytes), self.rotations.params.into(), bytes);
        }
        transfer.copy_buffer_to_buffer(
            staging.at(2 * bytes),
            self.camera_position.params.into(),
            16,
        );
        temp.buffers.push(staging);
    }

    /// Write the positions of the vertices, offset by their parameters, and
    /// the normals that follow from them, into the models, and build their
    /// acceleration structures again. [`Self::rebuild_scene`] has to follow
    /// before tracing, for the scene to take them.
    pub fn apply_vertices(&self, encoder: &mut blade_graphics::CommandEncoder) {
        self.control.invalidate();
        let vp = &self.vertex_params;
        if vp.models.is_empty() {
            return;
        }
        vp.dispatch(
            encoder,
            "diff-apply-vertices",
            &self.pipelines.as_ref().unwrap().apply,
            false,
            0.0,
        );
        encoder.transfer("diff-clear-normals").fill_buffer(
            vp.normal_sums.into(),
            vp.normal_sums.size(),
            0,
        );
        vp.dispatch(
            encoder,
            "diff-sum-normals",
            &self.pipelines.as_ref().unwrap().sum_normals,
            true,
            0.0,
        );
        vp.dispatch(
            encoder,
            "diff-write-normals",
            &self.pipelines.as_ref().unwrap().write_normals,
            false,
            0.0,
        );
        let mut pass = encoder.acceleration_structure("diff-vertex-BLAS");
        for vm in &vp.models {
            pass.build_bottom_level(vm.acceleration_structure, &vm.meshes, vm.scratch.into());
        }
    }

    /// Number of vertex slots, which [`Self::vertex_bases`] and
    /// [`Self::vertex_offsets`] have, used or not.
    pub fn vertex_slot_count(&self) -> usize {
        if self.vertex_params.models.is_empty() {
            0
        } else {
            self.vertex_params.slot_count as usize
        }
    }

    /// Carry the gradients by the normals of the vertices to their positions,
    /// which [`Self::step`] does by itself.
    pub fn gather_vertex_grads(&self, encoder: &mut blade_graphics::CommandEncoder) {
        let vp = &self.vertex_params;
        if vp.models.is_empty() {
            return;
        }
        vp.dispatch(
            encoder,
            "diff-normals-to-positions",
            &self.pipelines.as_ref().unwrap().normals_to_positions,
            true,
            0.0,
        );
        encoder.transfer("diff-clear-normal-grads").fill_buffer(
            vp.normal_grads.into(),
            vp.normal_grads.size(),
            0,
        );
    }

    /// Copy the positions, the offsets, and the gradients of the vertex
    /// slots to where [`Self::vertex_bases`] and the like read them.
    pub fn read_back_vertices(&self, encoder: &mut blade_graphics::CommandEncoder) {
        let vp = &self.vertex_params;
        let bytes = 16 * u64::from(vp.slot_count);
        let mut transfer = encoder.transfer("diff-read-back-vertices");
        transfer.copy_buffer_to_buffer(vp.slot_positions.into(), vp.readback.at(0), bytes);
        transfer.copy_buffer_to_buffer(vp.group.params.into(), vp.readback.at(bytes), bytes);
        transfer.copy_buffer_to_buffer(vp.group.grads.into(), vp.readback.at(2 * bytes), bytes);
    }

    /// Copy what [`Self::current_meshes`] needs to the host, which includes
    /// [`Self::read_back_vertices`].
    pub fn read_back_meshes(&self, encoder: &mut blade_graphics::CommandEncoder) {
        self.read_back_vertices(encoder);
        let vp = &self.vertex_params;
        let mut transfer = encoder.transfer("diff-read-back-meshes");
        if vp.vertex_count != 0 {
            transfer.copy_buffer_to_buffer(
                vp.vertex_slots.into(),
                vp.mesh_readback.at(0),
                4 * u64::from(vp.vertex_count),
            );
        }
        for vm in &vp.models {
            let (offset, bytes) = vm.index_readback;
            if bytes != 0 {
                transfer.copy_buffer_to_buffer(
                    vm.index_buffer.at(0),
                    vp.mesh_readback.at(offset),
                    bytes,
                );
            }
        }
    }

    /// The geometries of the models whose vertices are parameters, as they
    /// are now, with the copies of every vertex as one, as of the last
    /// [`Self::read_back_meshes`]. They are what remeshing starts from, see
    /// [`super::remesh`].
    pub fn current_meshes(&self, asset_hub: &crate::AssetHub) -> Vec<CurrentMesh> {
        let vp = &self.vertex_params;
        let data = vp.mesh_readback.data();
        let slots = unsafe { slice::from_raw_parts(data as *const u32, vp.vertex_count as usize) };
        let bases = self.vertex_bases();
        let offsets = self.vertex_offsets();
        let mut meshes = Vec::new();
        for vm in &vp.models {
            let (offset, bytes) = vm.index_readback;
            let indices = unsafe {
                slice::from_raw_parts(data.add(offset as usize) as *const u32, bytes as usize / 4)
            };
            for (geometry_index, geometry) in
                asset_hub.models[vm.model].geometries.iter().enumerate()
            {
                let mut compact = HashMap::<u32, u32>::new();
                let mut mesh = super::remesh::Mesh::default();
                let first = (geometry.index_offset / 4) as usize;
                for triangle in 0..geometry.triangle_count as usize {
                    let corners = [0, 1, 2].map(|k| {
                        let local = indices[first + 3 * triangle + k];
                        let slot =
                            slots[(vm.first_vertex + geometry.vertex_range.start + local) as usize];
                        *compact.entry(slot).or_insert_with(|| {
                            let (b, o) = (bases[slot as usize], offsets[slot as usize]);
                            mesh.positions.push([b[0] + o[0], b[1] + o[1], b[2] + o[2]]);
                            (mesh.positions.len() - 1) as u32
                        })
                    });
                    mesh.triangles.push(corners);
                }
                meshes.push(CurrentMesh {
                    model: vm.model,
                    geometry: geometry_index,
                    mesh,
                });
            }
        }
        meshes
    }

    /// The triangles of all the models whose vertices are parameters, by the
    /// vertex slots of their corners, as of the last
    /// [`Self::read_back_meshes`]. They tell which slots neighbor which, for
    /// optimizing the offsets on the host, such as with
    /// [`super::large_steps::LargeSteps`].
    pub fn slot_triangles(&self, asset_hub: &crate::AssetHub) -> Vec<[u32; 3]> {
        let vp = &self.vertex_params;
        let data = vp.mesh_readback.data();
        let slots = unsafe { slice::from_raw_parts(data as *const u32, vp.vertex_count as usize) };
        let mut triangles = Vec::new();
        for vm in &vp.models {
            let (offset, bytes) = vm.index_readback;
            let indices = unsafe {
                slice::from_raw_parts(data.add(offset as usize) as *const u32, bytes as usize / 4)
            };
            for geometry in asset_hub.models[vm.model].geometries.iter() {
                let first = (geometry.index_offset / 4) as usize;
                for triangle in 0..geometry.triangle_count as usize {
                    triangles.push([0, 1, 2].map(|k| {
                        let local = indices[first + 3 * triangle + k];
                        slots[(vm.first_vertex + geometry.vertex_range.start + local) as usize]
                    }));
                }
            }
        }
        triangles
    }

    /// Derivative of the loss by [`Self::vertex_offsets`], accumulated since
    /// the last [`Self::step`], as of [`Self::read_back_vertices`]. The part
    /// through the normals only joins at the step, or with
    /// [`Self::gather_vertex_grads`].
    pub fn vertex_grads(&self) -> &[[f32; 4]] {
        self.vertex_readback(32 * u64::from(self.vertex_params.slot_count))
    }

    fn vertex_readback(&self, offset: u64) -> &[[f32; 4]] {
        unsafe {
            slice::from_raw_parts(
                self.vertex_params.readback.data().add(offset as usize) as *const [f32; 4],
                self.vertex_slot_count(),
            )
        }
    }

    /// Where the vertex of every slot was loaded, in the space of its model,
    /// with 1 in W for the slots in use, as of [`Self::read_back_vertices`].
    pub fn vertex_bases(&self) -> &[[f32; 4]] {
        self.vertex_readback(0)
    }

    /// Offset of the vertex of every slot, as of [`Self::read_back_vertices`].
    pub fn vertex_offsets(&self) -> &[[f32; 4]] {
        self.vertex_readback(16 * u64::from(self.vertex_params.slot_count))
    }

    /// Replace the offsets of the vertex slots, one for each of
    /// [`Self::vertex_slot_count`]. The staging buffer goes to `temp`.
    pub fn write_vertex_offsets(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
        offsets: &[[f32; 4]],
        temp: &mut crate::FrameResources,
    ) {
        self.control.invalidate();
        assert_eq!(offsets.len(), self.vertex_slot_count());
        let bytes = (offsets.len() * 16) as u64;
        let staging = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "diff vertex staging",
            size: bytes,
            memory: blade_graphics::Memory::Upload,
        });
        unsafe {
            slice::from_raw_parts_mut(staging.data() as *mut [f32; 4], offsets.len())
                .copy_from_slice(offsets);
        }
        encoder
            .transfer("diff-write-vertices")
            .copy_buffer_to_buffer(
                staging.into(),
                self.vertex_params.group.params.into(),
                bytes,
            );
        temp.buffers.push(staging);
    }

    /// Build the scene of `ray_tracer` again, with `objects` as they are now,
    /// and take their transforms for the geometry. This is the way to move
    /// the objects between the steps of an optimization, see [`GeometryBase`].
    #[allow(clippy::too_many_arguments)]
    pub fn rebuild_scene(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
        ray_tracer: &mut super::RayTracer,
        objects: &mut [crate::Object],
        env_map: Option<blade_asset::Handle<crate::Texture>>,
        asset_hub: &crate::AssetHub,
        temp: &mut crate::FrameResources,
    ) {
        ray_tracer.build_scene(encoder, objects, env_map, asset_hub, gpu, temp);
        self.update_geometry(encoder, gpu, ray_tracer, temp);
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
        self.control.invalidate();
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

    /// Sum of the losses of the views, each as of its last
    /// [`Self::evaluate_loss`], which [`Self::backward`] includes, once the
    /// work is done.
    pub fn loss(&self) -> f32 {
        (0..self.view_count).map(|view| self.loss_of(view)).sum()
    }

    /// The loss of the selected view alone, see [`Self::loss`].
    pub fn view_loss(&self) -> f32 {
        self.loss_of(self.view)
    }

    fn loss_of(&self, view: u32) -> f32 {
        unsafe {
            *(self
                .loss
                .data()
                .add((LOSS_STRIDE * u64::from(view)) as usize) as *const f32)
        }
    }

    /// Make `view` the one that the following calls render, compare, and
    /// differentiate, and whose image and target they use.
    pub fn select_view(&mut self, view: u32) {
        assert!(
            view < self.view_count,
            "there are {} views",
            self.view_count
        );
        self.view = view;
    }

    /// The view that [`Self::select_view`] selected last.
    pub fn selected_view(&self) -> u32 {
        self.view
    }

    pub fn view_count(&self) -> u32 {
        self.view_count
    }

    fn view_piece(&self, buffer: blade_graphics::Buffer) -> blade_graphics::BufferPiece {
        buffer.at(self.view_stride * u64::from(self.view))
    }

    fn loss_piece(&self) -> blade_graphics::BufferPiece {
        self.loss.at(LOSS_STRIDE * u64::from(self.view))
    }

    fn image_bytes(&self) -> u64 {
        u64::from(self.image_size.width * self.image_size.height) * 16
    }

    /// The rendered image of the view: `Rgba32Float` radiance, row by row.
    pub fn image(&self) -> blade_graphics::BufferPiece {
        self.view_piece(self.image)
    }

    /// Derivative of the loss by [`Self::image`], in the same layout, which
    /// [`Self::evaluate_loss`] fills in and [`Self::backpropagate`] reads.
    pub fn image_adjoint(&self) -> blade_graphics::BufferPiece {
        self.view_piece(self.image_adjoint)
    }

    /// The target of the view that [`Self::backward`] compares the image
    /// with, in the same layout as [`Self::image`], for filling it in.
    pub fn target(&self) -> blade_graphics::BufferPiece {
        self.view_piece(self.target)
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
            self.accumulated_frames[self.view as usize] = 0;
        }
        let data = self.trace_data(
            ray_tracer,
            self.make_camera(camera),
            self.make_params(config, frame_index),
        );
        self.collect_emitters(encoder, &data);
        let dispatch = self
            .pipelines
            .as_ref()
            .unwrap()
            .primal
            .get_dispatch_for(self.image_size);
        {
            let mut pass = encoder.compute("diff-primal");
            let mut pc = pass.with(&self.pipelines.as_ref().unwrap().primal);
            pc.bind(0, &data);
            pc.dispatch(dispatch);
        }
        self.accumulated_frames[self.view as usize] += 1;
        if config.control_variate > 0.0 && !accumulate {
            self.apply_control_variate(encoder, &data, dispatch, config.control_variate);
        }
    }

    /// Make the image the recursive control variate of it, from the image
    /// of the parameters of the last estimate of the view with the same
    /// random numbers, and keep the parameters for the next one.
    fn apply_control_variate(
        &mut self,
        encoder: &mut blade_graphics::CommandEncoder,
        data: &DiffTraceData,
        dispatch: [u32; 3],
        weight: f32,
    ) {
        let view = self.view as usize;
        let generation = self.control.generation.get();
        let valid = self.control.view_generations[view] == Some(generation);
        let previous =
            |i: usize| self.control.previous_params[i].at(self.control.strides[i] * view as u64);
        if valid {
            let mut previous_data = DiffTraceData {
                material_params: previous(0),
                env_params: previous(1),
                texture_params: previous(2),
                image: self.control.previous_image.into(),
                ..*data
            };
            // Rendered anew, not accumulated.
            previous_data.parameters.accumulated_frames = 0;
            let mut pass = encoder.compute("diff-primal-previous");
            let mut pc = pass.with(&self.pipelines.as_ref().unwrap().primal);
            pc.bind(0, &previous_data);
            pc.dispatch(dispatch);
        }
        let pixel_count = self.image_size.width * self.image_size.height;
        {
            let mut pass = encoder.compute("diff-control-variate");
            let mut pc = pass.with(&self.pipelines.as_ref().unwrap().control);
            pc.bind(
                0,
                &ControlData {
                    control: ControlParams {
                        pixel_count,
                        valid: valid as u32,
                        weight,
                        _pad: 0,
                    },
                    control_image: self.view_piece(self.image),
                    control_previous: self.control.previous_image.into(),
                    control_estimate: self.view_piece(self.control.estimate),
                },
            );
            pc.dispatch(flat_dispatch(pixel_count));
        }
        let mut transfer = encoder.transfer("diff-keep-parameters");
        for (i, current) in [
            self.materials.params,
            self.environment.params,
            self.textures.params,
        ]
        .into_iter()
        .enumerate()
        {
            if current.size() != 0 {
                transfer.copy_buffer_to_buffer(current.into(), previous(i), current.size());
            }
        }
        self.control.view_generations[view] = Some(generation);
    }

    /// Make the current image the target.
    pub fn capture_target(&self, encoder: &mut blade_graphics::CommandEncoder) {
        let mut transfer = encoder.transfer("diff-capture-target");
        transfer.copy_buffer_to_buffer(self.image(), self.target(), self.image_bytes());
    }

    /// Compute the loss of the current image against the target, along with
    /// its derivative by the image, which [`Self::backward`] starts from.
    pub fn evaluate_loss(&mut self, encoder: &mut blade_graphics::CommandEncoder) {
        {
            let mut transfer = encoder.transfer("diff-loss-clear");
            transfer.fill_buffer(self.loss_piece(), 4, 0);
        }
        let pixel_count = self.image_size.width * self.image_size.height;
        let mut pass = encoder.compute("diff-loss");
        let mut pc = pass.with(&self.pipelines.as_ref().unwrap().loss);
        pc.bind(
            0,
            &LossData {
                loss_params: LossParams {
                    pixel_count,
                    _pad: [0; 3],
                },
                image: self.image(),
                target_image: self.target(),
                image_adjoint: self.image_adjoint(),
                loss: self.loss_piece(),
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
        self.backpropagate(encoder, ray_tracer, camera, config, frame_index);
    }

    /// Add the derivative by the parameters of whatever loss
    /// [`Self::image_adjoint`] is the derivative of, by the image, to the
    /// gradients. [`Self::backward`] does this for the L2 loss against the
    /// target, and this is the way to use another loss.
    pub fn backpropagate(
        &mut self,
        encoder: &mut blade_graphics::CommandEncoder,
        ray_tracer: &super::RayTracer,
        camera: &crate::Camera,
        config: &DiffConfig,
        frame_index: u32,
    ) {
        let data = self.trace_data(
            ray_tracer,
            self.make_camera(camera),
            self.make_params(config, frame_index),
        );
        self.collect_emitters(encoder, &data);
        let mut pass = encoder.compute("diff-adjoint");
        {
            let mut pc = pass.with(&self.pipelines.as_ref().unwrap().adjoint);
            pc.bind(0, &data);
            pc.dispatch(
                self.pipelines
                    .as_ref()
                    .unwrap()
                    .adjoint
                    .get_dispatch_for(self.image_size),
            );
        }
        if self.geometry && self.triangle_count != 0 && !config.projective_edges {
            assert!(
                config.jitter_primary_rays,
                "the gradient of the geometry needs jittered primary rays"
            );
            let mut pc = pass.with(&self.pipelines.as_ref().unwrap().edges);
            pc.bind(0, &data);
            pc.dispatch(flat_dispatch(self.triangle_count));
        }
    }

    /// Update the parameters along the gradients accumulated since the last
    /// step, and clear the gradients.
    ///
    /// Every parameter stays non-negative, and those that are fractions
    /// stay in the unit interval.
    pub fn step(&mut self, encoder: &mut blade_graphics::CommandEncoder, config: &StepConfig) {
        match config.materials {
            Some((adam, flags)) => adam_step(
                &self.pipelines.as_ref().unwrap().adam,
                encoder,
                &mut self.materials,
                &adam,
                MATERIAL_PARAM_COUNT,
                flags.bits(),
                Bounds::NonNegative {
                    upper_mask: BOUNDED_MATERIAL_PARAMS.bits(),
                },
            ),
            None => self.materials.clear_grads(encoder),
        }
        match config.environment {
            Some(adam) => adam_step(
                &self.pipelines.as_ref().unwrap().adam,
                encoder,
                &mut self.environment,
                &adam,
                ENV_PARAM_COUNT,
                0x7,
                Bounds::NonNegative { upper_mask: 0 },
            ),
            None => self.environment.clear_grads(encoder),
        }
        match config.textures {
            Some(adam) => adam_step(
                &self.pipelines.as_ref().unwrap().adam,
                encoder,
                &mut self.textures,
                &adam,
                TEXTURE_PARAM_COUNT,
                0x7,
                Bounds::NonNegative { upper_mask: 0x7 },
            ),
            None => self.textures.clear_grads(encoder),
        }
        match config.vertices {
            Some((adam, smoothness)) => {
                let vp = &mut self.vertex_params;
                if !vp.models.is_empty() {
                    vp.dispatch(
                        encoder,
                        "diff-normals-to-positions",
                        &self.pipelines.as_ref().unwrap().normals_to_positions,
                        true,
                        0.0,
                    );
                    encoder.transfer("diff-clear-normal-grads").fill_buffer(
                        vp.normal_grads.into(),
                        vp.normal_grads.size(),
                        0,
                    );
                }
                if smoothness > 0.0 && !vp.models.is_empty() {
                    encoder.transfer("diff-clear-neighbors").fill_buffer(
                        vp.neighbor_sums.into(),
                        vp.neighbor_sums.size(),
                        0,
                    );
                    vp.dispatch(
                        encoder,
                        "diff-sum-neighbors",
                        &self.pipelines.as_ref().unwrap().sum_neighbors,
                        true,
                        smoothness,
                    );
                    let mut pass = encoder.compute("diff-add-smoothness");
                    let mut pc = pass.with(&self.pipelines.as_ref().unwrap().smoothness);
                    let vertex_pass = VertexPass {
                        smoothness,
                        slot_count: vp.slot_count,
                        ..Default::default()
                    };
                    pc.bind(0, &vp.pass_data(vertex_pass, None));
                    pc.dispatch(flat_dispatch(vp.slot_count));
                }
                adam_step(
                    &self.pipelines.as_ref().unwrap().adam,
                    encoder,
                    &mut vp.group,
                    &adam,
                    4,
                    0x7,
                    Bounds::Unbounded,
                );
            }
            None => {
                let vp = &self.vertex_params;
                vp.group.clear_grads(encoder);
                encoder.transfer("diff-clear-normal-grads").fill_buffer(
                    vp.normal_grads.into(),
                    vp.normal_grads.size(),
                    0,
                );
            }
        }
        for (group, adam) in [
            (&mut self.translations, config.translations),
            (&mut self.rotations, config.rotations),
            (&mut self.camera_position, config.camera_position),
        ] {
            match adam {
                Some(adam) => adam_step(
                    &self.pipelines.as_ref().unwrap().adam,
                    encoder,
                    group,
                    &adam,
                    4,
                    0x7,
                    Bounds::Unbounded,
                ),
                None => group.clear_grads(encoder),
            }
        }
    }

    pub fn destroy(&mut self, gpu: &blade_graphics::Context) {
        if let Some(mut pipelines) = self.pipelines.take() {
            pipelines.destroy(gpu);
        }
        self.destroy_resources(gpu, true);
    }

    fn destroy_resources(&mut self, gpu: &blade_graphics::Context, with_target: bool) {
        gpu.destroy_buffer(self.edge_keys);
        gpu.destroy_buffer(self.edge_counts);
        gpu.destroy_buffer(self.edge_normals);
        gpu.destroy_buffer(self.edge_triangles);
        gpu.destroy_buffer(self.edge_hits);
        gpu.destroy_buffer(self.edge_importance);
        gpu.destroy_buffer(self.edge_guide);
        gpu.destroy_buffer(self.emitters);
        self.control.destroy(gpu);
        self.vertex_params.destroy(gpu);
        self.translations.destroy(gpu);
        self.rotations.destroy(gpu);
        self.camera_position.destroy(gpu);
        gpu.destroy_buffer(self.edge_entries);
        gpu.destroy_buffer(self.instance_transforms);
        for buffer in self.staging.drain(..) {
            gpu.destroy_buffer(buffer);
        }
        self.materials.destroy(gpu);
        self.environment.destroy(gpu);
        self.textures.destroy(gpu);
        gpu.destroy_buffer(self.texture_regions);
        gpu.destroy_buffer(self.image);
        if with_target {
            gpu.destroy_buffer(self.target);
        }
        gpu.destroy_buffer(self.image_adjoint);
        gpu.destroy_buffer(self.loss);
        gpu.destroy_buffer(self.readback);
    }
}

/// What the recursive control variate of the images keeps, see
/// [`DiffConfig::control_variate`].
struct ControlState {
    /// The material, environment and texture parameters of the last
    /// estimate of every view, each at its multiple of `strides`.
    previous_params: [blade_graphics::Buffer; 3],
    strides: [u64; 3],
    /// The image of those, with the random numbers of the current one.
    previous_image: blade_graphics::Buffer,
    /// The estimate of every view.
    estimate: blade_graphics::Buffer,
    /// Bumped by every change that the parameters of the last estimates
    /// don't account for, such as of the geometry.
    generation: std::cell::Cell<u32>,
    /// The generation of the last estimate of every view, if there is one.
    view_generations: Vec<Option<u32>>,
}

impl ControlState {
    fn new(
        gpu: &blade_graphics::Context,
        view_count: u32,
        view_stride: u64,
        param_sizes: [u64; 3],
    ) -> Self {
        let strides = param_sizes.map(|size| size.max(4).next_multiple_of(256));
        let names = [
            "diff previous materials",
            "diff previous environment",
            "diff previous textures",
        ];
        let device = |name, size| {
            gpu.create_buffer(blade_graphics::BufferDesc {
                name,
                size,
                memory: blade_graphics::Memory::Device,
            })
        };
        Self {
            previous_params: [0, 1, 2]
                .map(|i| device(names[i], strides[i] * u64::from(view_count))),
            strides,
            previous_image: device("diff previous image", view_stride),
            estimate: device("diff estimate", view_stride * u64::from(view_count)),
            generation: std::cell::Cell::new(0),
            view_generations: vec![None; view_count as usize],
        }
    }

    fn invalidate(&self) {
        self.generation.set(self.generation.get().wrapping_add(1));
    }

    fn destroy(&mut self, gpu: &blade_graphics::Context) {
        for buffer in self.previous_params {
            gpu.destroy_buffer(buffer);
        }
        gpu.destroy_buffer(self.previous_image);
        gpu.destroy_buffer(self.estimate);
    }
}

/// Bounds of the parameters in [`adam_step`].
#[derive(Clone, Copy)]
enum Bounds {
    /// Non-negative, and those with a bit in the mask at most 1.
    NonNegative {
        upper_mask: u32,
    },
    Unbounded,
}

fn adam_step(
    pipeline: &blade_graphics::ComputePipeline,
    encoder: &mut blade_graphics::CommandEncoder,
    group: &mut ParamGroup,
    config: &AdamConfig,
    period: u32,
    mask: u32,
    bounds: Bounds,
) {
    let (lower, upper_mask) = match bounds {
        Bounds::NonNegative { upper_mask } => (0.0, upper_mask),
        Bounds::Unbounded => (f32::MIN, 0),
    };
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
                lower,
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

/// Instance transforms laid out like `mat4x3<f32>`.
fn instance_transform_data(ray_tracer: &super::RayTracer) -> Vec<[f32; 16]> {
    ray_tracer
        .instance_transforms
        .iter()
        .map(|t| super::mat4_transform(t).to_cols_array())
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn offset_transforms_turn_about_the_origin() {
        let mut base = blade_graphics::IDENTITY_TRANSFORM;
        base.x.w = 1.0;
        base.y.w = 2.0;
        let quarter = std::f32::consts::FRAC_PI_2;
        let t = super::offset_transform(&base, [0.0, 0.0, 3.0], [0.0, 0.0, quarter]);
        // The X axis turns into Y about Z, and the origin only moves.
        let m = super::super::mat4_transform(&t);
        let p = m.transform_point3(glam::Vec3::new(1.0, 0.0, 0.0));
        assert!(
            (p - glam::Vec3::new(1.0, 3.0, 3.0)).length() < 1.0e-5,
            "{p}"
        );
        let o = m.transform_point3(glam::Vec3::ZERO);
        assert!(
            (o - glam::Vec3::new(1.0, 2.0, 3.0)).length() < 1.0e-5,
            "{o}"
        );
    }

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
