use std::{
    borrow::Cow,
    fmt, mem,
    ops::Range,
    ptr, str,
    sync::{Arc, Mutex},
};

#[cfg(feature = "asset")]
use std::{
    collections::hash_map::{Entry, HashMap},
    hash,
};

mod skinning;
use skinning::{CookedNode, CookedSkin, Skin, SkinNode};
pub use skinning::{MAX_JOINTS_PER_DRAW, Pose};

const PRELOAD_TEXTURES: bool = false;

const fn texture_format(
    compressed: blade_graphics::TextureFormat,
    uncompressed: blade_graphics::TextureFormat,
) -> blade_graphics::TextureFormat {
    if cfg!(target_arch = "wasm32") {
        uncompressed
    } else {
        compressed
    }
}

const META_BASE_COLOR: crate::texture::Meta = crate::texture::Meta {
    format: texture_format(
        blade_graphics::TextureFormat::Bc1UnormSrgb,
        blade_graphics::TextureFormat::Rgba8UnormSrgb,
    ),
    generate_mips: true,
    y_flip: false,
};
const META_NORMAL: crate::texture::Meta = crate::texture::Meta {
    // "texpresso" doesn't know how to produce signed normalized.
    format: texture_format(
        blade_graphics::TextureFormat::Bc5Unorm,
        blade_graphics::TextureFormat::Rgba8Unorm,
    ),
    generate_mips: false,
    y_flip: false,
};
const META_METALLIC_ROUGHNESS: crate::texture::Meta = crate::texture::Meta {
    // Metallic-roughness values are linear, so no sRGB here.
    format: texture_format(
        blade_graphics::TextureFormat::Bc1Unorm,
        blade_graphics::TextureFormat::Rgba8Unorm,
    ),
    generate_mips: true,
    y_flip: false,
};
const META_EMISSIVE: crate::texture::Meta = crate::texture::Meta {
    format: texture_format(
        blade_graphics::TextureFormat::Bc1UnormSrgb,
        blade_graphics::TextureFormat::Rgba8UnormSrgb,
    ),
    generate_mips: true,
    y_flip: false,
};

fn pack4x8snorm(v: [f32; 4]) -> u32 {
    v.iter().rev().fold(0u32, |u, f| {
        (u << 8) | (f.clamp(-1.0, 1.0) * 127.0 + 0.5) as i8 as u8 as u32
    })
}

fn encode_normal(v: [f32; 3]) -> u32 {
    pack4x8snorm([v[0], v[1], v[2], 0.0])
}

pub(crate) fn mat4_to_transform(matrix: glam::Mat4) -> blade_graphics::Transform {
    let columns = mint::ColumnMatrix3x4 {
        x: matrix.x_axis.truncate().into(),
        y: matrix.y_axis.truncate().into(),
        z: matrix.z_axis.truncate().into(),
        w: matrix.w_axis.truncate().into(),
    };
    mint::RowMatrix3x4::from(columns)
}

pub(crate) fn mat4_from_transform(transform: &blade_graphics::Transform) -> glam::Mat4 {
    glam::Mat4 {
        x_axis: transform.x.into(),
        y_axis: transform.y.into(),
        z_axis: transform.z.into(),
        w_axis: glam::Vec4::W,
    }
    .transpose()
}

pub struct Geometry {
    pub name: String,
    pub vertex_range: Range<u32>,
    pub index_offset: u64,
    pub index_type: Option<blade_graphics::IndexType>,
    pub triangle_count: u32,
    pub transform: blade_graphics::Transform,
    pub material_index: usize,
    pub(crate) node_index: usize,
    pub(crate) skin_index: Option<usize>,
    pub(crate) joint_palette: Vec<u32>,
}

/// Surface appearance, following the glTF 2.0 metallic-roughness model.
///
/// Each of the textures is modulated by the corresponding factor,
/// so a material without textures is fully described by the factors.
//TODO: move out into a separate asset type
pub struct Material {
    pub base_color_texture: Option<blade_asset::Handle<crate::Texture>>,
    pub base_color_factor: [f32; 4],
    pub normal_texture: Option<blade_asset::Handle<crate::Texture>>,
    pub normal_scale: f32,
    /// Green channel is roughness, blue channel is metalness.
    pub metallic_roughness_texture: Option<blade_asset::Handle<crate::Texture>>,
    pub metalness: f32,
    pub roughness: f32,
    pub emissive_texture: Option<blade_asset::Handle<crate::Texture>>,
    /// Emitted radiance, with `KHR_materials_emissive_strength` folded in.
    pub emissive_factor: [f32; 3],
    pub transparent: bool,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color_texture: None,
            base_color_factor: [1.0; 4],
            normal_texture: None,
            normal_scale: 0.0,
            metallic_roughness_texture: None,
            metalness: 0.0,
            roughness: 0.5,
            emissive_texture: None,
            emissive_factor: [0.0; 3],
            transparent: false,
        }
    }
}

pub struct Model {
    pub name: String,
    pub winding: f32,
    pub geometries: Vec<Geometry>,
    pub materials: Vec<Material>,
    pub vertex_buffer: blade_graphics::Buffer,
    pub skin_vertex_buffer: blade_graphics::Buffer,
    pub index_buffer: blade_graphics::Buffer,
    pub transform_buffer: blade_graphics::Buffer,
    pub acceleration_structure: blade_graphics::AccelerationStructure,
    pub(crate) nodes: Vec<SkinNode>,
    pub(crate) bind_pose: Pose,
    pub(crate) skins: Vec<Skin>,
}

impl Model {
    pub(crate) fn vertex_count(&self) -> usize {
        self.geometries
            .last()
            .map(|geometry| geometry.vertex_range.end as usize)
            .unwrap_or(0)
    }
}

#[derive(blade_macros::Flat, Default)]
struct TextureReference<'a> {
    path: Cow<'a, [u8]>,
    embedded_data: Cow<'a, [u8]>,
    //Note: this isn't used for anything during deserialization
    source_index: usize,
}

#[derive(blade_macros::Flat)]
struct CookedMaterial<'a> {
    base_color: TextureReference<'a>,
    base_color_factor: [f32; 4],
    normal: TextureReference<'a>,
    normal_scale: f32,
    metallic_roughness: TextureReference<'a>,
    metalness: f32,
    roughness: f32,
    emissive: TextureReference<'a>,
    emissive_factor: [f32; 3],
    transparent: bool,
}

#[derive(blade_macros::Flat)]
struct CookedGeometry<'a> {
    name: Cow<'a, [u8]>,
    vertices: Cow<'a, [crate::Vertex]>,
    skin_vertices: Cow<'a, [crate::SkinVertex]>,
    indices: Cow<'a, [u32]>,
    transform: [f32; 12],
    material_index: u32,
    node_index: u32,
    skin_index: u32,
    joint_palette: Cow<'a, [u32]>,
}

#[cfg(feature = "asset")]
#[derive(Clone, PartialEq)]
struct GltfVertex {
    position: [f32; 3],
    normal: [f32; 3],
    tangent: [f32; 4],
    tex_coords: [f32; 2],
    joints: [u32; 4],
    weights: [f32; 4],
}
#[cfg(feature = "asset")]
impl Default for GltfVertex {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 0.0],
            tex_coords: [0.0; 2],
            joints: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
        }
    }
}
#[cfg(feature = "asset")]
impl Eq for GltfVertex {}
#[cfg(feature = "asset")]
impl hash::Hash for GltfVertex {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        for f in self.position.iter() {
            f.to_bits().hash(state);
        }
        for f in self.normal.iter() {
            f.to_bits().hash(state);
        }
        for f in self.tangent.iter() {
            f.to_bits().hash(state);
        }
        for f in self.tex_coords.iter() {
            f.to_bits().hash(state);
        }
        self.joints.hash(state);
        for f in self.weights.iter() {
            f.to_bits().hash(state);
        }
    }
}

#[cfg(feature = "asset")]
struct FlattenedGeometry(Box<[GltfVertex]>);
#[cfg(feature = "asset")]
impl mikktspace::Geometry for FlattenedGeometry {
    fn num_faces(&self) -> usize {
        self.0.len() / 3
    }
    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }
    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.0[face * 3 + vert].position
    }
    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.0[face * 3 + vert].normal
    }
    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.0[face * 3 + vert].tex_coords
    }
    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.0[face * 3 + vert].tangent = tangent;
    }
}
#[cfg(feature = "asset")]
impl FlattenedGeometry {
    #[profiling::function]
    fn reconstruct_indices(
        self,
        skin_joint_count: Option<usize>,
    ) -> (
        Vec<u32>,
        Vec<crate::Vertex>,
        Vec<crate::SkinVertex>,
        Vec<u32>,
    ) {
        let mut indices = Vec::with_capacity(self.0.len());
        let mut unique = Vec::new();
        let mut cache = HashMap::new();
        for v in self.0.iter() {
            let i = match cache.entry(v.clone()) {
                Entry::Occupied(e) => *e.get(),
                Entry::Vacant(e) => {
                    let i = unique.len() as u32;
                    unique.push(v.clone());
                    *e.insert(i)
                }
            };
            indices.push(i);
        }
        let mut joint_palette = Vec::new();
        if let Some(joint_count) = skin_joint_count {
            let mut palette_lookup = HashMap::new();
            for vertex in &mut unique {
                let weight_sum: f32 = vertex.weights.iter().sum();
                if weight_sum > 0.0 {
                    for weight in &mut vertex.weights {
                        *weight /= weight_sum;
                    }
                } else {
                    vertex.weights = [1.0, 0.0, 0.0, 0.0];
                }
                for influence in 0..4 {
                    if vertex.weights[influence] == 0.0 {
                        vertex.joints[influence] = 0;
                        continue;
                    }
                    let source_joint = vertex.joints[influence];
                    assert!(
                        (source_joint as usize) < joint_count,
                        "vertex references joint {source_joint}, but its skin has only {joint_count} joints"
                    );
                    let palette_index = match palette_lookup.entry(source_joint) {
                        Entry::Occupied(entry) => *entry.get(),
                        Entry::Vacant(entry) => {
                            let index = joint_palette.len() as u32;
                            joint_palette.push(source_joint);
                            *entry.insert(index)
                        }
                    };
                    vertex.joints[influence] = palette_index;
                }
            }
            assert!(
                joint_palette.len() <= skinning::MAX_JOINTS_PER_DRAW,
                "a mesh primitive uses {} joints; at most {} are supported per draw",
                joint_palette.len(),
                skinning::MAX_JOINTS_PER_DRAW,
            );
        }
        let vertices: Vec<_> = unique
            .iter()
            .map(|v| {
                let t = v.tangent;
                crate::Vertex {
                    position: v.position,
                    bitangent_sign: t[3],
                    tex_coords: v.tex_coords,
                    normal: encode_normal(v.normal),
                    tangent: encode_normal([t[0], t[1], t[2]]),
                }
            })
            .collect();
        let skin_vertices: Vec<_> = unique
            .iter()
            .map(|v| crate::SkinVertex::packed_skin(v.joints, v.weights))
            .collect();
        log::debug!("Compacted {}->{}", self.0.len(), vertices.len());
        (indices, vertices, skin_vertices, joint_palette)
    }
}

#[derive(blade_macros::Flat)]
pub struct CookedModel<'a> {
    name: &'a [u8],
    winding: f32,
    materials: Vec<CookedMaterial<'a>>,
    geometries: Vec<CookedGeometry<'a>>,
    nodes: Vec<CookedNode>,
    skins: Vec<CookedSkin<'a>>,
}

#[cfg(feature = "asset")]
impl CookedModel<'_> {
    fn populate_gltf(
        &mut self,
        g_node: gltf::Node,
        parent_transform: glam::Mat4,
        data_buffers: &[Vec<u8>],
        flattened_geos: &mut Vec<FlattenedGeometry>,
        default_material_index: Option<u32>,
    ) {
        let local_transform = glam::Mat4::from_cols_array_2d(&g_node.transform().matrix());
        let global_transform = parent_transform * local_transform;

        if let Some(g_mesh) = g_node.mesh() {
            let name = g_node.name().unwrap_or("");
            let node_index = g_node.index() as u32;
            let skin_index = g_node.skin().map_or(u32::MAX, |skin| skin.index() as u32);
            let col_matrix = mint::ColumnMatrix3x4 {
                x: global_transform.x_axis.truncate().into(),
                y: global_transform.y_axis.truncate().into(),
                z: global_transform.z_axis.truncate().into(),
                w: global_transform.w_axis.truncate().into(),
            };
            let transform = mint::RowMatrix3x4::from(col_matrix).into();

            for (prim_index, g_primitive) in g_mesh.primitives().enumerate() {
                if g_primitive.mode() != gltf::mesh::Mode::Triangles {
                    log::warn!(
                        "Skipping primitive '{}'[{}] for having mesh mode {:?}",
                        name,
                        prim_index,
                        g_primitive.mode()
                    );
                    continue;
                }
                let material_index = match g_primitive.material().index() {
                    Some(index) => index as u32,
                    None => default_material_index.expect("missing glTF default material"),
                };

                let reader = g_primitive.reader(|buffer| Some(&data_buffers[buffer.index()]));
                let vertex_count = g_primitive.get(&gltf::Semantic::Positions).unwrap().count();

                // Read the vertices into memory
                flattened_geos.push({
                    profiling::scope!("Read data");
                    let mut pre_vertices = vec![GltfVertex::default(); vertex_count];

                    for (v, pos) in pre_vertices
                        .iter_mut()
                        .zip(reader.read_positions().unwrap())
                    {
                        for component in pos {
                            assert!(component.is_finite());
                        }
                        v.position = pos;
                    }
                    if let Some(iter) = reader.read_tex_coords(0) {
                        for (v, tc) in pre_vertices.iter_mut().zip(iter.into_f32()) {
                            v.tex_coords = tc;
                        }
                    } else {
                        log::warn!("No tex coords in {name}");
                    }
                    if let Some(iter) = reader.read_normals() {
                        assert_eq!(
                            pre_vertices.len(),
                            iter.len(),
                            "geometry {name} doesn't have enough normals"
                        );
                        for (v, normal) in pre_vertices.iter_mut().zip(iter) {
                            v.normal = normal;
                            assert_ne!(encode_normal(normal), 0);
                        }
                    } else {
                        log::warn!("No normals in {name}");
                    }
                    if let Some(iter) = reader.read_tangents() {
                        assert_eq!(pre_vertices.len(), iter.len());
                        for (v, tangent) in pre_vertices.iter_mut().zip(iter) {
                            v.tangent = tangent;
                        }
                    }
                    if let Some(iter) = reader.read_joints(0) {
                        for (v, joints) in pre_vertices.iter_mut().zip(iter.into_u16()) {
                            v.joints = joints.map(u32::from);
                        }
                    } else if skin_index != u32::MAX {
                        log::warn!("Skinned geometry {name} has no JOINTS_0 attribute");
                    }
                    if let Some(iter) = reader.read_weights(0) {
                        for (v, weights) in pre_vertices.iter_mut().zip(iter.into_f32()) {
                            v.weights = weights;
                        }
                    } else if skin_index != u32::MAX {
                        log::warn!("Skinned geometry {name} has no WEIGHTS_0 attribute");
                    }

                    // Untangle from the index buffer
                    match reader.read_indices() {
                        Some(read) => FlattenedGeometry(
                            read.into_u32()
                                .map(|i| pre_vertices[i as usize].clone())
                                .collect(),
                        ),
                        None => FlattenedGeometry(pre_vertices.into_boxed_slice()),
                    }
                });

                self.geometries.push(CookedGeometry {
                    name: Cow::Owned(name.as_bytes().to_owned()),
                    vertices: Cow::Borrowed(&[]),
                    skin_vertices: Cow::Borrowed(&[]),
                    indices: Cow::Borrowed(&[]),
                    transform,
                    material_index,
                    node_index,
                    skin_index,
                    joint_palette: Cow::Borrowed(&[]),
                });
            }
        }

        for child in g_node.children() {
            self.populate_gltf(
                child,
                global_transform,
                data_buffers,
                flattened_geos,
                default_material_index,
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FrontFace {
    Clockwise,
    #[default]
    CounterClockwise,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Meta {
    pub generate_tangents: bool,
    pub front_face: FrontFace,
}

impl fmt::Display for Meta {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        Ok(()) //TODO
    }
}

#[derive(Debug)]
struct Transfer {
    stage: blade_graphics::Buffer,
    dst: blade_graphics::Buffer,
    size: u64,
}

struct BufferUpload {
    stage: blade_graphics::Buffer,
    dst: blade_graphics::Buffer,
    size: u64,
    target: blade_graphics::BufferTarget,
}

impl BufferUpload {
    fn new(
        gpu: &blade_graphics::Context,
        name: &'static str,
        size: u64,
        target: blade_graphics::BufferTarget,
    ) -> Self {
        let dst = gpu.create_buffer(blade_graphics::BufferDesc {
            name,
            size,
            memory: blade_graphics::Memory::Device,
        });
        let stage = if cfg!(target_arch = "wasm32") {
            dst
        } else {
            gpu.create_buffer(blade_graphics::BufferDesc {
                name: "model staging",
                size,
                memory: blade_graphics::Memory::Upload,
            })
        };

        Self {
            stage,
            dst,
            size,
            target,
        }
    }

    fn finish(self, baker: &Baker) {
        if cfg!(target_arch = "wasm32") {
            baker.gpu_context.sync_buffer(self.dst, self.target);
        } else {
            baker
                .pending_operations
                .lock()
                .unwrap()
                .transfers
                .push(Transfer {
                    stage: self.stage,
                    dst: self.dst,
                    size: self.size,
                });
        }
    }
}

#[derive(Debug)]
struct BlasConstruct {
    meshes: Vec<blade_graphics::AccelerationStructureMesh>,
    scratch: blade_graphics::Buffer,
    dst: blade_graphics::AccelerationStructure,
}

#[derive(Default)]
struct PendingOperations {
    transfers: Vec<Transfer>,
    blas_constructs: Vec<BlasConstruct>,
}

enum TextureSource {
    Path(String),
    Embedded(
        Option<choir::IdleTask>,
        Arc<blade_asset::Cooker<super::texture::Baker>>,
    ),
}

#[cfg(feature = "asset")]
impl TextureReference<'_> {
    fn complete(&mut self, sources: &slab::Slab<TextureSource>) {
        match sources.get(self.source_index) {
            Some(&TextureSource::Embedded(ref _task, ref sub_cooker)) => {
                self.embedded_data = Cow::Owned(sub_cooker.extract_embedded());
            }
            Some(&TextureSource::Path(ref full)) => {
                self.path = Cow::Owned(full.as_bytes().to_owned());
            }
            None => {}
        }
    }
}

pub struct Baker {
    gpu_context: Arc<blade_graphics::Context>,
    pending_operations: Mutex<PendingOperations>,
    //TODO: change to asset materials
    asset_textures: Arc<blade_asset::AssetManager<crate::texture::Baker>>,
}

impl Baker {
    pub fn new(
        gpu_context: &Arc<blade_graphics::Context>,
        asset_textures: &Arc<blade_asset::AssetManager<crate::texture::Baker>>,
    ) -> Self {
        Self {
            gpu_context: Arc::clone(gpu_context),
            pending_operations: Mutex::new(PendingOperations::default()),
            asset_textures: Arc::clone(asset_textures),
        }
    }

    pub fn flush(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        temp_buffers: &mut Vec<blade_graphics::Buffer>,
    ) {
        let mut pending_ops = self.pending_operations.lock().unwrap();
        if !pending_ops.transfers.is_empty() {
            let mut pass = encoder.transfer("init models");
            for transfer in pending_ops.transfers.drain(..) {
                pass.copy_buffer_to_buffer(
                    transfer.stage.into(),
                    transfer.dst.into(),
                    transfer.size,
                );
                temp_buffers.push(transfer.stage);
            }
        }
        // Skip when `build_blas` queued nothing (`ray_query` empty).
        if !pending_ops.blas_constructs.is_empty() {
            let mut pass = encoder.acceleration_structure("BLAS");
            for construct in pending_ops.blas_constructs.drain(..) {
                pass.build_bottom_level(construct.dst, &construct.meshes, construct.scratch.into());
                temp_buffers.push(construct.scratch);
            }
        }
    }

    #[cfg(feature = "asset")]
    fn cook_texture(
        &self,
        texture: gltf::texture::Texture,
        meta: super::texture::Meta,
        parent_cooker: &blade_asset::Cooker<Baker>,
        data_buffers: &[Vec<u8>],
    ) -> TextureSource {
        match texture.source().source() {
            gltf::image::Source::View { view, mime_type } => {
                let sub_cooker = Arc::new(blade_asset::Cooker::new_embedded());
                let cooker = Arc::clone(&sub_cooker);
                let baker = Arc::clone(&self.asset_textures.baker);
                let buffer = &data_buffers[view.buffer().index()];
                let data = buffer[view.offset()..view.offset() + view.length()].to_vec();
                let extension = mime_type.split_once('/').unwrap().1.to_string();
                let task =
                    self.asset_textures
                        .choir
                        .spawn("embedded cook")
                        .init(move |exe_ontext| {
                            blade_asset::Baker::cook(
                                baker.as_ref(),
                                &data,
                                &extension,
                                meta,
                                cooker,
                                &exe_ontext,
                            );
                        });
                TextureSource::Embedded(Some(task), sub_cooker)
            }
            gltf::image::Source::Uri { uri, mime_type: _ } => {
                let relative = if let Some(_rest) = uri.strip_prefix("data:") {
                    panic!("Data URL isn't supported for textures yet");
                } else if let Some(rest) = uri.strip_prefix("file://") {
                    rest
                } else if let Some(rest) = uri.strip_prefix("file:") {
                    rest
                } else {
                    uri
                };
                let full = parent_cooker.base_path().join(relative);
                if PRELOAD_TEXTURES {
                    self.asset_textures.load(&full, meta);
                }
                TextureSource::Path(full.to_str().unwrap().to_string())
            }
        }
    }

    fn serve_texture(
        &self,
        texture_ref: &TextureReference,
        meta: super::texture::Meta,
        exe_context: &choir::ExecutionContext,
    ) -> Option<blade_asset::Handle<super::texture::Texture>> {
        if !texture_ref.path.is_empty() {
            let path_str = str::from_utf8(&texture_ref.path).unwrap();
            let (handle, task) = self.asset_textures.load(path_str, meta);
            exe_context.add_fork(&task);
            Some(handle)
        } else if !texture_ref.embedded_data.is_empty() {
            let cooked = unsafe {
                <super::texture::CookedImage<'_> as blade_asset::Flat>::read(
                    texture_ref.embedded_data.as_ptr(),
                )
            };
            Some(
                self.asset_textures
                    .load_cooked_inside_task(cooked, exe_context),
            )
        } else {
            None
        }
    }
}

// SAFETY: GLES asset tasks are executed inline on the context's owning thread.
#[cfg(any(gles, target_arch = "wasm32"))]
unsafe impl Send for Baker {}
#[cfg(any(gles, target_arch = "wasm32"))]
unsafe impl Sync for Baker {}

/// Description of a procedural model geometry.
///
/// Each geometry gets a texture-less material of its own,
/// described by the PBR factors here.
pub struct ProceduralGeometry {
    pub name: String,
    pub vertices: Vec<crate::Vertex>,
    pub indices: Vec<u32>,
    pub base_color_factor: [f32; 4],
    pub metalness: f32,
    pub roughness: f32,
    pub emissive_factor: [f32; 3],
}

impl Default for ProceduralGeometry {
    fn default() -> Self {
        let material = Material::default();
        Self {
            name: String::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            base_color_factor: material.base_color_factor,
            metalness: material.metalness,
            roughness: material.roughness,
            emissive_factor: material.emissive_factor,
        }
    }
}

impl Baker {
    /// Create a model from procedural geometry data, bypassing the asset cooking pipeline.
    pub fn create_model(&self, name: &str, geometries: Vec<ProceduralGeometry>) -> Model {
        assert!(!geometries.is_empty(), "Need at least one geometry");

        let total_vertices: usize = geometries.iter().map(|g| g.vertices.len()).sum();
        let total_vertex_size = (total_vertices * mem::size_of::<crate::Vertex>()) as u64;
        let vertex_buffer = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "proc vertex",
            size: total_vertex_size,
            memory: blade_graphics::Memory::Shared,
        });

        // Filled with the identity skin data. It's only read for posed
        // models, so that the skinning pass can bake the pose.
        let total_skin_vertex_size = (total_vertices * mem::size_of::<crate::SkinVertex>()) as u64;
        let skin_vertex_buffer = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "proc skin vertex",
            size: total_skin_vertex_size,
            memory: blade_graphics::Memory::Shared,
        });
        unsafe {
            let skin_ptr = skin_vertex_buffer.data() as *mut crate::SkinVertex;
            for index in 0..total_vertices {
                ptr::write(skin_ptr.add(index), crate::SkinVertex::default());
            }
        }

        let total_indices: usize = geometries.iter().map(|g| g.indices.len()).sum();
        let total_index_size = total_indices as u64 * 4
            + geometries.len() as u64 * blade_graphics::limits::STORAGE_BUFFER_ALIGNMENT;
        let index_buffer = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "proc index",
            size: total_index_size,
            memory: blade_graphics::Memory::Shared,
        });

        let total_transform_size =
            (geometries.len() * mem::size_of::<blade_graphics::Transform>()) as u64;
        let transform_buffer = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "proc transform",
            size: total_transform_size,
            memory: blade_graphics::Memory::Shared,
        });

        let mut start_vertex = 0u32;
        let mut index_offset = 0u64;
        let mut transform_offset = 0u64;
        let mut model_geometries = Vec::with_capacity(geometries.len());
        let mut materials = Vec::with_capacity(geometries.len());
        let mut meshes = Vec::with_capacity(geometries.len());
        let vertex_stride = mem::size_of::<crate::Vertex>() as u32;

        for geo in geometries.iter() {
            index_offset = crate::util::align_to(
                index_offset,
                blade_graphics::limits::STORAGE_BUFFER_ALIGNMENT,
            );

            unsafe {
                ptr::copy_nonoverlapping(
                    geo.vertices.as_ptr(),
                    (vertex_buffer.data() as *mut crate::Vertex).add(start_vertex as usize),
                    geo.vertices.len(),
                );
                ptr::copy_nonoverlapping(
                    geo.indices.as_ptr(),
                    index_buffer.data().add(index_offset as usize) as *mut u32,
                    geo.indices.len(),
                );
            }
            let transform = blade_graphics::IDENTITY_TRANSFORM;
            unsafe {
                ptr::copy_nonoverlapping(
                    ptr::from_ref(&transform).cast::<u8>(),
                    transform_buffer.data().add(transform_offset as usize),
                    mem::size_of::<blade_graphics::Transform>(),
                );
            }

            let index_type = if geo.indices.is_empty() {
                None
            } else {
                Some(blade_graphics::IndexType::U32)
            };
            let triangle_count = if geo.indices.is_empty() {
                geo.vertices.len() as u32 / 3
            } else {
                geo.indices.len() as u32 / 3
            };

            let material_index = materials.len();
            materials.push(Material {
                base_color_factor: geo.base_color_factor,
                metalness: geo.metalness,
                roughness: geo.roughness,
                emissive_factor: geo.emissive_factor,
                ..Material::default()
            });

            meshes.push(blade_graphics::AccelerationStructureMesh {
                vertex_data: vertex_buffer.at(start_vertex as u64 * vertex_stride as u64),
                vertex_format: blade_graphics::VertexFormat::F32Vec3,
                vertex_stride,
                vertex_count: geo.vertices.len() as u32,
                index_data: index_buffer.at(index_offset),
                index_type,
                triangle_count,
                transform_data: transform_buffer.at(transform_offset),
                is_opaque: true,
            });

            model_geometries.push(Geometry {
                name: geo.name.clone(),
                vertex_range: start_vertex..start_vertex + geo.vertices.len() as u32,
                index_offset,
                index_type,
                triangle_count,
                transform,
                material_index,
                node_index: 0,
                skin_index: None,
                joint_palette: Vec::new(),
            });

            start_vertex += geo.vertices.len() as u32;
            index_offset += geo.indices.len() as u64 * 4;
            transform_offset += mem::size_of::<blade_graphics::Transform>() as u64;
        }

        self.gpu_context
            .sync_buffer(vertex_buffer, blade_graphics::BufferTarget::Data);
        self.gpu_context
            .sync_buffer(skin_vertex_buffer, blade_graphics::BufferTarget::Data);
        self.gpu_context
            .sync_buffer(index_buffer, blade_graphics::BufferTarget::Index);
        self.gpu_context
            .sync_buffer(transform_buffer, blade_graphics::BufferTarget::Data);

        Model {
            name: name.to_string(),
            winding: 1.0,
            geometries: model_geometries,
            materials,
            vertex_buffer,
            skin_vertex_buffer,
            index_buffer,
            transform_buffer,
            acceleration_structure: self.build_blas(name, meshes),
            nodes: vec![SkinNode {
                parent_index: u32::MAX,
                translation: glam::Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            }],
            bind_pose: Pose::from_node_matrices(vec![glam::Mat4::IDENTITY]),
            skins: Vec::new(),
        }
    }

    /// Schedule building of a bottom level acceleration structure for the given meshes.
    ///
    /// Returns a null acceleration structure if ray tracing isn't supported.
    fn build_blas(
        &self,
        name: &str,
        meshes: Vec<blade_graphics::AccelerationStructureMesh>,
    ) -> blade_graphics::AccelerationStructure {
        if self.gpu_context.capabilities().ray_query.is_empty() {
            return blade_graphics::AccelerationStructure::default();
        }

        let sizes = self
            .gpu_context
            .get_bottom_level_acceleration_structure_sizes(&meshes);
        let acceleration_structure = self.gpu_context.create_acceleration_structure(
            blade_graphics::AccelerationStructureDesc {
                name,
                ty: blade_graphics::AccelerationStructureType::BottomLevel,
                size: sizes.data,
                updatable: false,
            },
        );
        let scratch = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "BLAS scratch",
            size: sizes.scratch,
            memory: blade_graphics::Memory::Device,
        });

        self.pending_operations
            .lock()
            .unwrap()
            .blas_constructs
            .push(BlasConstruct {
                meshes,
                scratch,
                dst: acceleration_structure,
            });
        acceleration_structure
    }
}

impl blade_asset::Baker for Baker {
    type Meta = Meta;
    type Data<'a> = CookedModel<'a>;
    type Output = Model;

    fn cook(
        &self,
        source: &[u8],
        extension: &str,
        meta: Meta,
        cooker: Arc<blade_asset::Cooker<Self>>,
        exe_context: &choir::ExecutionContext,
    ) {
        match extension {
            #[cfg(feature = "asset")]
            "gltf" | "glb" => {
                use base64::engine::{Engine as _, general_purpose::STANDARD as ENCODING_ENGINE};

                let gltf::Gltf { document, mut blob } = gltf::Gltf::from_slice(source).unwrap();
                // extract buffers
                let mut buffers = Vec::new();
                for buffer in document.buffers() {
                    let mut data = match buffer.source() {
                        gltf::buffer::Source::Uri(uri) => {
                            if let Some(rest) = uri.strip_prefix("data:") {
                                let (_before, after) = rest.split_once(";base64,").unwrap();
                                ENCODING_ENGINE.decode(after).unwrap()
                            } else if let Some(rest) = uri.strip_prefix("file://") {
                                cooker.add_dependency(rest.as_ref())
                            } else if let Some(rest) = uri.strip_prefix("file:") {
                                cooker.add_dependency(rest.as_ref())
                            } else {
                                cooker.add_dependency(uri.as_ref())
                            }
                        }
                        gltf::buffer::Source::Bin => blob.take().unwrap(),
                    };
                    assert!(data.len() >= buffer.length());
                    while data.len() % 4 != 0 {
                        data.push(0);
                    }
                    buffers.push(data);
                }

                let mut sources = slab::Slab::new();
                let mut model = CookedModel {
                    name: &[],
                    winding: match meta.front_face {
                        FrontFace::Clockwise => -1.0,
                        FrontFace::CounterClockwise => 1.0,
                    },
                    materials: Vec::new(),
                    geometries: Vec::new(),
                    nodes: skinning::cook_nodes(&document),
                    skins: skinning::cook_skins(&document, &buffers),
                };
                for g_material in document.materials() {
                    let pbr = g_material.pbr_metallic_roughness();
                    let emissive_strength = g_material.emissive_strength().unwrap_or(1.0);
                    model.materials.push(CookedMaterial {
                        base_color: TextureReference {
                            source_index: match pbr.base_color_texture() {
                                Some(info) => sources.insert(self.cook_texture(
                                    info.texture(),
                                    META_BASE_COLOR,
                                    &cooker,
                                    &buffers,
                                )),
                                None => !0,
                            },
                            ..Default::default()
                        },
                        base_color_factor: pbr.base_color_factor(),
                        normal: TextureReference {
                            source_index: match g_material.normal_texture() {
                                Some(info) => sources.insert(self.cook_texture(
                                    info.texture(),
                                    META_NORMAL,
                                    &cooker,
                                    &buffers,
                                )),
                                None => !0,
                            },
                            ..Default::default()
                        },
                        normal_scale: g_material.normal_texture().map_or(0.0, |info| info.scale()),
                        metallic_roughness: TextureReference {
                            source_index: match pbr.metallic_roughness_texture() {
                                Some(info) => sources.insert(self.cook_texture(
                                    info.texture(),
                                    META_METALLIC_ROUGHNESS,
                                    &cooker,
                                    &buffers,
                                )),
                                None => !0,
                            },
                            ..Default::default()
                        },
                        metalness: pbr.metallic_factor(),
                        roughness: pbr.roughness_factor(),
                        emissive: TextureReference {
                            source_index: match g_material.emissive_texture() {
                                Some(info) => sources.insert(self.cook_texture(
                                    info.texture(),
                                    META_EMISSIVE,
                                    &cooker,
                                    &buffers,
                                )),
                                None => !0,
                            },
                            ..Default::default()
                        },
                        emissive_factor: g_material
                            .emissive_factor()
                            .map(|c| c * emissive_strength),
                        transparent: g_material.alpha_mode() != gltf::material::AlphaMode::Opaque,
                    });
                }

                let default_material_index = document
                    .meshes()
                    .flat_map(|mesh| mesh.primitives())
                    .any(|primitive| primitive.material().index().is_none())
                    .then(|| {
                        let index = model.materials.len() as u32;
                        model.materials.push(CookedMaterial {
                            base_color: TextureReference {
                                source_index: !0,
                                ..Default::default()
                            },
                            base_color_factor: [1.0; 4],
                            normal: TextureReference {
                                source_index: !0,
                                ..Default::default()
                            },
                            normal_scale: 0.0,
                            metallic_roughness: TextureReference {
                                source_index: !0,
                                ..Default::default()
                            },
                            metalness: 1.0,
                            roughness: 1.0,
                            emissive: TextureReference {
                                source_index: !0,
                                ..Default::default()
                            },
                            emissive_factor: [0.0; 3],
                            transparent: false,
                        });
                        index
                    });

                let mut flattened_geos = Vec::new();
                for g_scene in document.scenes() {
                    for g_node in g_scene.nodes() {
                        model.populate_gltf(
                            g_node,
                            glam::Mat4::IDENTITY,
                            &buffers,
                            &mut flattened_geos,
                            default_material_index,
                        );
                    }
                }

                assert!(
                    !model.geometries.is_empty(),
                    "Empty models are not supported yet"
                );
                let skin_joint_counts: Vec<_> =
                    document.skins().map(|skin| skin.joints().len()).collect();
                let geometry_skin_joint_counts: Vec<_> = model
                    .geometries
                    .iter()
                    .map(|geometry| {
                        (geometry.skin_index != u32::MAX)
                            .then(|| skin_joint_counts[geometry.skin_index as usize])
                    })
                    .collect();
                let model_shared = Arc::new(Mutex::new(model));
                let model_clone = Arc::clone(&model_shared);
                let gen_tangents = exe_context.choir().spawn("generate tangents").init_iter(
                    flattened_geos.into_iter().enumerate(),
                    move |_, (index, mut fg)| {
                        if meta.generate_tangents {
                            let ok = mikktspace::generate_tangents(&mut fg);
                            assert!(ok, "MikkTSpace failed");
                        }
                        let (indices, vertices, skin_vertices, joint_palette) =
                            fg.reconstruct_indices(geometry_skin_joint_counts[index]);
                        let mut model = model_clone.lock().unwrap();
                        let geo = &mut model.geometries[index];
                        geo.vertices = Cow::Owned(vertices);
                        geo.skin_vertices = Cow::Owned(skin_vertices);
                        geo.indices = Cow::Owned(indices);
                        geo.joint_palette = Cow::Owned(joint_palette);
                    },
                );

                let mut dependencies = vec![gen_tangents];
                for (_, source) in sources.iter_mut() {
                    if let TextureSource::Embedded(ref mut task, _) = *source {
                        dependencies.push(task.take().unwrap())
                    }
                }

                let mut finish = exe_context.fork("finish").init(move |_| {
                    let mut model = Arc::into_inner(model_shared).unwrap().into_inner().unwrap();
                    for material in model.materials.iter_mut() {
                        material.base_color.complete(&sources);
                        material.normal.complete(&sources);
                        material.metallic_roughness.complete(&sources);
                        material.emissive.complete(&sources);
                    }
                    cooker.finish(model);
                });
                for dependency in dependencies {
                    finish.depend_on(&dependency);
                }
            }
            other => panic!("Unknown model extension: {}", other),
        }
    }

    fn serve(&self, model: CookedModel<'_>, exe_context: &choir::ExecutionContext) -> Self::Output {
        let nodes: Vec<_> = model.nodes.into_iter().map(SkinNode::from).collect();
        let bind_pose = skinning::bind_pose(&nodes);
        let skins: Vec<_> = model.skins.into_iter().map(Skin::from).collect();
        let mut materials = Vec::with_capacity(model.materials.len());
        for material in model.materials.iter() {
            materials.push(Material {
                base_color_texture: self.serve_texture(
                    &material.base_color,
                    META_BASE_COLOR,
                    exe_context,
                ),
                base_color_factor: material.base_color_factor,
                normal_texture: self.serve_texture(&material.normal, META_NORMAL, exe_context),
                normal_scale: material.normal_scale,
                metallic_roughness_texture: self.serve_texture(
                    &material.metallic_roughness,
                    META_METALLIC_ROUGHNESS,
                    exe_context,
                ),
                metalness: material.metalness,
                roughness: material.roughness,
                emissive_texture: self.serve_texture(
                    &material.emissive,
                    META_EMISSIVE,
                    exe_context,
                ),
                emissive_factor: material.emissive_factor,
                transparent: material.transparent,
            });
        }

        let total_vertices = model
            .geometries
            .iter()
            .map(|geo| geo.vertices.len())
            .sum::<usize>();
        let total_vertex_size = (total_vertices * mem::size_of::<crate::Vertex>()) as u64;
        let vertex_upload = BufferUpload::new(
            &self.gpu_context,
            "vertex",
            total_vertex_size,
            blade_graphics::BufferTarget::Data,
        );
        let vertex_buffer = vertex_upload.dst;
        let vertex_stage = vertex_upload.stage;

        let total_skin_vertex_size = (total_vertices * mem::size_of::<crate::SkinVertex>()) as u64;
        let skin_vertex_upload = BufferUpload::new(
            &self.gpu_context,
            "skin vertex",
            total_skin_vertex_size,
            blade_graphics::BufferTarget::Data,
        );
        let skin_vertex_buffer = skin_vertex_upload.dst;
        let skin_vertex_stage = skin_vertex_upload.stage;

        let total_indices = model
            .geometries
            .iter()
            .map(|geo| geo.indices.len())
            .sum::<usize>();
        let total_index_size = total_indices as u64 * 4
            + model.geometries.len() as u64 * blade_graphics::limits::STORAGE_BUFFER_ALIGNMENT;
        let index_upload = BufferUpload::new(
            &self.gpu_context,
            "index",
            total_index_size,
            blade_graphics::BufferTarget::Index,
        );
        let index_buffer = index_upload.dst;
        let index_stage = index_upload.stage;

        let total_transform_size =
            (model.geometries.len() * mem::size_of::<blade_graphics::Transform>()) as u64;
        let transform_upload = BufferUpload::new(
            &self.gpu_context,
            "transform",
            total_transform_size,
            blade_graphics::BufferTarget::Data,
        );
        let transform_buffer = transform_upload.dst;
        let transform_stage = transform_upload.stage;

        let mut meshes = Vec::with_capacity(model.geometries.len());
        let vertex_stride = mem::size_of::<super::Vertex>() as u32;
        let mut start_vertex = 0;
        let mut index_offset = 0;
        let mut transform_offset = 0;
        let mut geometries = Vec::with_capacity(model.geometries.len());
        for geometry in model.geometries.iter() {
            index_offset = crate::util::align_to(
                index_offset,
                blade_graphics::limits::STORAGE_BUFFER_ALIGNMENT,
            );
            let material = &model.materials[geometry.material_index as usize];
            unsafe {
                ptr::copy_nonoverlapping(
                    geometry.vertices.as_ptr(),
                    (vertex_stage.data() as *mut crate::Vertex).add(start_vertex as usize),
                    geometry.vertices.len(),
                );
                ptr::copy_nonoverlapping(
                    geometry.skin_vertices.as_ptr(),
                    (skin_vertex_stage.data() as *mut crate::SkinVertex).add(start_vertex as usize),
                    geometry.skin_vertices.len(),
                );
                ptr::copy_nonoverlapping(
                    geometry.indices.as_ptr(),
                    index_stage.data().add(index_offset as usize) as *mut u32,
                    geometry.indices.len(),
                );
                ptr::copy_nonoverlapping(
                    geometry.transform.as_ptr() as *const u8,
                    transform_stage.data().add(transform_offset as usize),
                    mem::size_of::<blade_graphics::Transform>(),
                );
            }
            let index_type = if geometry.indices.is_empty() {
                None
            } else {
                Some(blade_graphics::IndexType::U32)
            };
            let triangle_count = if geometry.indices.is_empty() {
                geometry.vertices.len() as u32 / 3
            } else {
                geometry.indices.len() as u32 / 3
            };
            meshes.push(blade_graphics::AccelerationStructureMesh {
                vertex_data: vertex_buffer.at(start_vertex as u64 * vertex_stride as u64),
                vertex_format: blade_graphics::VertexFormat::F32Vec3,
                vertex_stride,
                vertex_count: geometry.vertices.len() as u32,
                index_data: index_buffer.at(index_offset),
                index_type,
                triangle_count,
                transform_data: transform_buffer.at(transform_offset), //TODO
                is_opaque: !material.transparent,
            });
            geometries.push(Geometry {
                name: String::from_utf8_lossy(geometry.name.as_ref()).into_owned(),
                vertex_range: start_vertex..start_vertex + geometry.vertices.len() as u32,
                index_offset,
                index_type,
                triangle_count,
                transform: geometry.transform.into(),
                material_index: geometry.material_index as usize,
                node_index: geometry.node_index as usize,
                skin_index: (geometry.skin_index != u32::MAX)
                    .then_some(geometry.skin_index as usize),
                joint_palette: geometry.joint_palette.to_vec(),
            });
            start_vertex += geometry.vertices.len() as u32;
            index_offset += geometry.indices.len() as u64 * 4;
            transform_offset += mem::size_of::<blade_graphics::Transform>() as u64;
        }
        assert_eq!(start_vertex as usize, total_vertices);
        assert!(index_offset <= total_index_size);
        assert_eq!(transform_offset, total_transform_size);

        vertex_upload.finish(self);
        skin_vertex_upload.finish(self);
        index_upload.finish(self);
        transform_upload.finish(self);
        let acceleration_structure = self.build_blas(str::from_utf8(model.name).unwrap(), meshes);

        Model {
            name: String::from_utf8_lossy(model.name).into_owned(),
            winding: model.winding,
            geometries,
            materials,
            vertex_buffer,
            skin_vertex_buffer,
            index_buffer,
            transform_buffer,
            acceleration_structure,
            nodes,
            bind_pose,
            skins,
        }
    }

    fn delete(&self, model: Self::Output) {
        if model.acceleration_structure != blade_graphics::AccelerationStructure::default() {
            self.gpu_context
                .destroy_acceleration_structure(model.acceleration_structure);
        }
        self.gpu_context.destroy_buffer(model.vertex_buffer);
        if model.skin_vertex_buffer != blade_graphics::Buffer::default() {
            self.gpu_context.destroy_buffer(model.skin_vertex_buffer);
        }
        self.gpu_context.destroy_buffer(model.index_buffer);
        self.gpu_context.destroy_buffer(model.transform_buffer);
    }
}

#[cfg(all(test, feature = "asset"))]
mod tests {
    use super::*;

    #[test]
    fn compacts_and_normalizes_joint_palettes() {
        let vertex = GltfVertex {
            joints: [5, 2, 0, 0],
            weights: [0.75, 0.25, 0.0, 0.0],
            ..Default::default()
        };
        let geometry = FlattenedGeometry(vec![vertex; 3].into_boxed_slice());
        let (_, _, skin_vertices, palette) = geometry.reconstruct_indices(Some(6));
        assert_eq!(palette, [5, 2]);
        assert_eq!(skin_vertices[0].skin_joints(), [0, 1, 0, 0]);
        let weights = skin_vertices[0].skin_weights();
        assert!((weights[0] - 0.75).abs() < 1.0 / 255.0);
        assert!((weights[1] - 0.25).abs() < 1.0 / 255.0);
        assert_eq!(weights[2], 0.0);
        assert_eq!(weights[3], 0.0);
    }
}
