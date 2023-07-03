use std::{
    borrow::Cow,
    collections::hash_map::{Entry, HashMap},
    fmt, hash, mem,
    ops::Range,
    ptr, str,
    sync::{Arc, Mutex},
};

fn pack4x8snorm(v: [f32; 4]) -> u32 {
    v.iter().rev().fold(0u32, |u, f| {
        (u << 8) | (f.clamp(-1.0, 1.0) * 127.0 + 0.5) as i8 as u8 as u32
    })
}

fn encode_normal(v: [f32; 3]) -> u32 {
    pack4x8snorm([v[0], v[1], v[2], 0.0])
}

pub struct Geometry {
    pub name: String,
    pub vertex_range: Range<u32>,
    pub index_offset: u64,
    pub index_type: Option<blade_graphics::IndexType>,
    pub triangle_count: u32,
    pub transform: blade_graphics::Transform,
    pub material_index: usize,
}

//TODO: move out into a separate asset type
pub struct Material {
    pub base_color_texture: Option<blade_asset::Handle<crate::Texture>>,
    pub base_color_factor: [f32; 4],
    pub transparent: bool,
}

pub struct Model {
    pub name: String,
    pub geometries: Vec<Geometry>,
    pub materials: Vec<Material>,
    pub vertex_buffer: blade_graphics::Buffer,
    pub index_buffer: blade_graphics::Buffer,
    pub transform_buffer: blade_graphics::Buffer,
    pub acceleration_structure: blade_graphics::AccelerationStructure,
}

#[derive(blade_macros::Flat)]
struct CookedMaterial<'a> {
    base_color_path: Cow<'a, [u8]>,
    base_color_factor: [f32; 4],
    transparent: bool,
}

#[derive(blade_macros::Flat)]
struct CookedGeometry<'a> {
    name: Cow<'a, [u8]>,
    vertices: Cow<'a, [crate::Vertex]>,
    indices: Cow<'a, [u32]>,
    transform: [f32; 12],
    material_index: u32,
}

#[derive(Clone, Default, PartialEq)]
struct GltfVertex {
    position: [f32; 3],
    normal: [f32; 3],
    tangent: [f32; 4],
    tex_coords: [f32; 2],
}
impl Eq for GltfVertex {}
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
    fn reconstruct_indices(self) -> (Vec<u32>, Vec<crate::Vertex>) {
        let mut indices = Vec::with_capacity(self.0.len());
        let mut vertices = Vec::new();
        let mut cache = HashMap::new();
        for v in self.0.iter() {
            let i = match cache.entry(v.clone()) {
                Entry::Occupied(e) => *e.get(),
                Entry::Vacant(e) => {
                    let i = vertices.len() as u32;
                    vertices.push(crate::Vertex {
                        position: v.position,
                        bitangent_sign: v.tangent[3],
                        tex_coords: v.tex_coords,
                        normal: encode_normal(v.normal),
                        tangent: encode_normal([v.tangent[0], v.tangent[1], v.tangent[2]]),
                    });
                    *e.insert(i)
                }
            };
            indices.push(i);
        }
        log::info!("Compacted {}->{}", self.0.len(), vertices.len());
        (indices, vertices)
    }
}

#[derive(blade_macros::Flat)]
pub struct CookedModel<'a> {
    name: &'a [u8],
    materials: Vec<CookedMaterial<'a>>,
    geometries: Vec<CookedGeometry<'a>>,
}

#[cfg(feature = "asset")]
impl CookedModel<'_> {
    fn populate_gltf(
        &mut self,
        g_node: gltf::Node,
        parent_transform: glam::Mat4,
        data_buffers: &[Vec<u8>],
        flattened_geos: &mut Vec<FlattenedGeometry>,
    ) {
        let local_transform = glam::Mat4::from_cols_array_2d(&g_node.transform().matrix());
        let global_transform = parent_transform * local_transform;

        if let Some(g_mesh) = g_node.mesh() {
            let name = g_node.name().unwrap_or("");
            let col_matrix = mint::ColumnMatrix3x4 {
                x: global_transform.x_axis.truncate().into(),
                y: global_transform.y_axis.truncate().into(),
                z: global_transform.z_axis.truncate().into(),
                w: global_transform.w_axis.truncate().into(),
            };
            let transform = mint::RowMatrix3x4::from(col_matrix).into();

            for g_primitive in g_mesh.primitives() {
                if g_primitive.mode() != gltf::mesh::Mode::Triangles {
                    log::warn!(
                        "Skipping primitive for having mesh mode {:?}",
                        g_primitive.mode()
                    );
                    continue;
                }
                let material_index = match g_primitive.material().index() {
                    Some(index) => index as u32,
                    None => {
                        log::warn!("Skipping primitive for having default material");
                        continue;
                    }
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
                        }
                    } else {
                        log::warn!("No normals in {name}");
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
                    indices: Cow::Borrowed(&[]),
                    transform,
                    material_index,
                });
            }
        }

        for child in g_node.children() {
            self.populate_gltf(child, global_transform, data_buffers, flattened_geos);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Meta;

impl fmt::Display for Meta {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

struct Transfer {
    stage: blade_graphics::Buffer,
    dst: blade_graphics::Buffer,
    size: u64,
}

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
            let mut pass = encoder.transfer();
            for transfer in pending_ops.transfers.drain(..) {
                pass.copy_buffer_to_buffer(
                    transfer.stage.into(),
                    transfer.dst.into(),
                    transfer.size,
                );
                temp_buffers.push(transfer.stage);
            }
        }
        if !pending_ops.blas_constructs.is_empty() {
            let mut pass = encoder.acceleration_structure();
            for construct in pending_ops.blas_constructs.drain(..) {
                pass.build_bottom_level(construct.dst, &construct.meshes, construct.scratch.into());
                temp_buffers.push(construct.scratch);
            }
        }
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
        _meta: Meta,
        cooker: Arc<blade_asset::Cooker<CookedModel<'_>>>,
        exe_context: choir::ExecutionContext,
    ) {
        match extension {
            #[cfg(feature = "asset")]
            "gltf" => {
                use base64::engine::{general_purpose::URL_SAFE as ENCODING_ENGINE, Engine as _};

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
                let mut texture_paths = Vec::new();
                for texture in document.textures() {
                    let relative = match texture.source().source() {
                        gltf::image::Source::Uri { uri, .. } => {
                            if let Some(rest) = uri.strip_prefix("data:") {
                                let (_before, after) = rest.split_once(";base64,").unwrap();
                                let _data = ENCODING_ENGINE.decode(after).unwrap();
                                panic!("Data URL isn't supported here yet");
                            } else if let Some(rest) = uri.strip_prefix("file://") {
                                rest
                            } else if let Some(rest) = uri.strip_prefix("file:") {
                                rest
                            } else {
                                uri
                            }
                        }
                        gltf::image::Source::View { .. } => {
                            panic!("Embedded images are not supported yet")
                        }
                    };
                    let full = cooker.base_path().join(relative);
                    texture_paths.push(full.to_str().unwrap().to_string());
                }

                let mut model = CookedModel {
                    name: &[],
                    materials: Vec::new(),
                    geometries: Vec::new(),
                };
                for g_material in document.materials() {
                    let pbr = g_material.pbr_metallic_roughness();
                    model.materials.push(CookedMaterial {
                        base_color_path: Cow::Owned(match pbr.base_color_texture() {
                            Some(info) => texture_paths[info.texture().index()].as_bytes().to_vec(),
                            None => Vec::new(),
                        }),
                        base_color_factor: pbr.base_color_factor(),
                        transparent: g_material.alpha_mode() != gltf::material::AlphaMode::Opaque,
                    });
                }
                let mut flattened_geos = Vec::new();
                for g_scene in document.scenes() {
                    for g_node in g_scene.nodes() {
                        model.populate_gltf(
                            g_node,
                            glam::Mat4::IDENTITY,
                            &buffers,
                            &mut flattened_geos,
                        );
                    }
                }

                let model_shared = Arc::new(Mutex::new(model));
                let model_clone = Arc::clone(&model_shared);
                let gen_tangents = exe_context.choir().spawn("generate tangents").init_iter(
                    flattened_geos.into_iter().enumerate(),
                    move |(index, mut fg)| {
                        let ok = mikktspace::generate_tangents(&mut fg);
                        assert!(ok, "MillTSpace failed");
                        let (indices, vertices) = fg.reconstruct_indices();
                        let mut model = model_clone.lock().unwrap();
                        let geo = &mut model.geometries[index];
                        geo.vertices = Cow::Owned(vertices);
                        geo.indices = Cow::Owned(indices);
                    },
                );
                let mut finish = exe_context.fork("finish").init(move |_| {
                    let model = Arc::into_inner(model_shared).unwrap().into_inner().unwrap();
                    cooker.finish(model);
                });
                finish.depend_on(&gen_tangents);
            }
            other => panic!("Unknown model extension: {}", other),
        }
    }

    fn serve(&self, model: CookedModel<'_>, exe_context: choir::ExecutionContext) -> Self::Output {
        let mut materials = Vec::with_capacity(model.materials.len());
        for material in model.materials.iter() {
            let base_color_texture = if material.base_color_path.is_empty() {
                None
            } else {
                let path_str = str::from_utf8(&material.base_color_path).unwrap();
                let (handle, task) = self.asset_textures.load(
                    path_str,
                    crate::texture::Meta {
                        format: blade_graphics::TextureFormat::Bc1UnormSrgb,
                        generate_mips: true,
                        y_flip: true,
                    },
                );
                exe_context.add_fork(&task);
                Some(handle)
            };
            materials.push(Material {
                base_color_texture,
                base_color_factor: material.base_color_factor,
                transparent: material.transparent,
            });
        }

        let total_vertices = model
            .geometries
            .iter()
            .map(|geo| geo.vertices.len())
            .sum::<usize>();
        let total_vertex_size = (total_vertices * mem::size_of::<crate::Vertex>()) as u64;
        let vertex_buffer = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "vertex",
            size: total_vertex_size,
            memory: blade_graphics::Memory::Device,
        });
        let vertex_stage = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "vertex stage",
            size: total_vertex_size,
            memory: blade_graphics::Memory::Upload,
        });

        let total_indices = model
            .geometries
            .iter()
            .map(|geo| geo.indices.len())
            .sum::<usize>();
        let total_index_size = total_indices as u64 * 4;
        let index_buffer = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "index",
            size: total_index_size,
            memory: blade_graphics::Memory::Device,
        });
        let index_stage = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "index stage",
            size: total_index_size,
            memory: blade_graphics::Memory::Upload,
        });

        let total_transform_size =
            (model.geometries.len() * mem::size_of::<blade_graphics::Transform>()) as u64;
        let transform_buffer = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "transform",
            size: total_transform_size,
            memory: blade_graphics::Memory::Device,
        });
        let transform_stage = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "transform stage",
            size: total_transform_size,
            memory: blade_graphics::Memory::Upload,
        });

        let mut meshes = Vec::with_capacity(model.geometries.len());
        let vertex_stride = mem::size_of::<super::Vertex>() as u32;
        let mut start_vertex = 0;
        let mut index_offset = 0;
        let mut transform_offset = 0;
        let mut geometries = Vec::with_capacity(model.geometries.len());
        for geometry in model.geometries.iter() {
            let material = &model.materials[geometry.material_index as usize];
            unsafe {
                ptr::copy_nonoverlapping(
                    geometry.vertices.as_ptr(),
                    (vertex_stage.data() as *mut crate::Vertex).add(start_vertex as usize),
                    geometry.vertices.len(),
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
            });
            start_vertex += geometry.vertices.len() as u32;
            index_offset += geometry.indices.len() as u64 * 4;
            transform_offset += mem::size_of::<blade_graphics::Transform>() as u64;
        }
        assert_eq!(start_vertex as usize, total_vertices);
        assert_eq!(index_offset, total_index_size);
        assert_eq!(transform_offset, total_transform_size);

        let sizes = self
            .gpu_context
            .get_bottom_level_acceleration_structure_sizes(&meshes);
        let acceleration_structure = self.gpu_context.create_acceleration_structure(
            blade_graphics::AccelerationStructureDesc {
                name: str::from_utf8(model.name).unwrap(),
                ty: blade_graphics::AccelerationStructureType::BottomLevel,
                size: sizes.data,
            },
        );
        let scratch = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: "BLAS scratch",
            size: sizes.scratch,
            memory: blade_graphics::Memory::Device,
        });

        let mut pending_ops = self.pending_operations.lock().unwrap();
        pending_ops.transfers.push(Transfer {
            stage: vertex_stage,
            dst: vertex_buffer,
            size: total_vertex_size,
        });
        pending_ops.transfers.push(Transfer {
            stage: index_stage,
            dst: index_buffer,
            size: total_index_size,
        });
        pending_ops.transfers.push(Transfer {
            stage: transform_stage,
            dst: transform_buffer,
            size: total_transform_size,
        });
        pending_ops.blas_constructs.push(BlasConstruct {
            meshes,
            scratch,
            dst: acceleration_structure,
        });

        Model {
            name: String::from_utf8_lossy(model.name).into_owned(),
            geometries,
            materials,
            vertex_buffer,
            index_buffer,
            transform_buffer,
            acceleration_structure,
        }
    }

    fn delete(&self, model: Self::Output) {
        self.gpu_context
            .destroy_acceleration_structure(model.acceleration_structure);
        self.gpu_context.destroy_buffer(model.vertex_buffer);
        self.gpu_context.destroy_buffer(model.index_buffer);
        self.gpu_context.destroy_buffer(model.transform_buffer);
    }
}
