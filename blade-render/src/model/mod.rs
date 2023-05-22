use std::{
    borrow::Cow,
    mem,
    ops::Range,
    path::PathBuf,
    ptr, str,
    sync::{Arc, Mutex},
};

pub struct Geometry {
    pub name: String,
    pub vertex_range: Range<u32>,
    pub index_offset: u64,
    pub index_type: Option<blade::IndexType>,
    pub triangle_count: u32,
    pub transform: blade::Transform,
    pub material_index: usize,
}

//TODO: move out into a separate asset type
pub struct Material {
    pub base_color_texture: Option<blade_asset::Handle<crate::Texture>>,
    pub base_color_factor: [f32; 4],
}

pub struct Model {
    pub name: String,
    pub geometries: Vec<Geometry>,
    pub materials: Vec<Material>,
    pub vertex_buffer: blade::Buffer,
    pub index_buffer: blade::Buffer,
    pub transform_buffer: blade::Buffer,
    pub acceleration_structure: blade::AccelerationStructure,
}

#[derive(blade_macros::Flat)]
struct CookedMaterial<'a> {
    base_color_path: &'a [u8],
    base_color_factor: [f32; 4],
}

#[derive(blade_macros::Flat)]
struct CookedGeometry<'a> {
    name: Cow<'a, [u8]>,
    vertices: Cow<'a, [crate::Vertex]>,
    indices: Cow<'a, [u32]>,
    transform: [f32; 12],
    material_index: u32,
}

#[derive(blade_macros::Flat)]
pub struct CookedModel<'a> {
    name: &'a [u8],
    materials: Vec<CookedMaterial<'a>>,
    geometries: Vec<CookedGeometry<'a>>,
}

struct Transfer {
    stage: blade::Buffer,
    dst: blade::Buffer,
    size: u64,
}

struct BlasConstruct {
    meshes: Vec<blade::AccelerationStructureMesh>,
    scratch: blade::Buffer,
    dst: blade::AccelerationStructure,
}

#[derive(Default)]
struct PendingOperations {
    transfers: Vec<Transfer>,
    blas_constructs: Vec<BlasConstruct>,
}

pub struct Baker {
    gpu_context: Arc<blade::Context>,
    pending_operations: Mutex<PendingOperations>,
    //TODO: change to asset materials
    asset_textures: Arc<blade_asset::AssetManager<crate::texture::Baker>>,
}

impl Baker {
    pub fn new(
        gpu_context: &Arc<blade::Context>,
        asset_textures: &Arc<blade_asset::AssetManager<crate::texture::Baker>>,
    ) -> Self {
        Self {
            gpu_context: Arc::clone(gpu_context),
            pending_operations: Mutex::new(PendingOperations::default()),
            asset_textures: Arc::clone(asset_textures),
        }
    }

    #[cfg(feature = "asset")]
    fn populate_gltf(
        &self,
        model: &mut CookedModel,
        g_node: gltf::Node,
        parent_transform: glam::Mat4,
        data_buffers: &[Vec<u8>],
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
                let indices = match reader.read_indices() {
                    Some(read) => read.into_u32().collect(),
                    None => Vec::new(),
                };
                let vertex_count = g_primitive.get(&gltf::Semantic::Positions).unwrap().count();
                let mut vertices = vec![crate::Vertex::default(); vertex_count];
                for (v, pos) in vertices.iter_mut().zip(reader.read_positions().unwrap()) {
                    v.position = pos;
                }
                if let Some(iter) = reader.read_tex_coords(0) {
                    for (v, tc) in vertices.iter_mut().zip(iter.into_f32()) {
                        v.tex_coords = tc;
                    }
                }
                if let Some(iter) = reader.read_normals() {
                    for (v, normal) in vertices.iter_mut().zip(iter) {
                        // convert floating point to i16 normalized
                        v.normal = [
                            (normal[0] * i16::MAX as f32) as i16,
                            (normal[1] * i16::MAX as f32) as i16,
                        ];
                    }
                }
                model.geometries.push(CookedGeometry {
                    name: Cow::Owned(name.as_bytes().to_owned()),
                    vertices: Cow::Owned(vertices),
                    indices: Cow::Owned(indices),
                    transform,
                    material_index,
                });
            }
        }

        for child in g_node.children() {
            self.populate_gltf(model, child, global_transform, data_buffers);
        }
    }

    pub fn flush(
        &self,
        encoder: &mut blade::CommandEncoder,
        temp_buffers: &mut Vec<blade::Buffer>,
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

    fn get_full_path(&self, relative: &str) -> PathBuf {
        //HACK: using textures' root. Instead, need a mechanism
        // for tracking source dependencies, which are relative.
        self.asset_textures.root.join(relative)
    }
}

impl blade_asset::Baker for Baker {
    type Meta = ();
    type Data<'a> = CookedModel<'a>;
    type Output = Model;

    fn cook(
        &self,
        source: &[u8],
        extension: &str,
        _meta: (),
        result: Arc<blade_asset::Cooked<CookedModel<'_>>>,
        _exe_context: choir::ExecutionContext,
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
                                std::fs::read(rest).unwrap()
                            } else if let Some(rest) = uri.strip_prefix("file:") {
                                std::fs::read(rest).unwrap()
                            } else {
                                let path = self.get_full_path(uri);
                                std::fs::read(path).unwrap()
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
                    texture_paths.push(match texture.source().source() {
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
                    });
                }

                let mut model = CookedModel {
                    name: &[],
                    materials: Vec::new(),
                    geometries: Vec::new(),
                };
                for g_material in document.materials() {
                    let pbr = g_material.pbr_metallic_roughness();
                    model.materials.push(CookedMaterial {
                        base_color_path: match pbr.base_color_texture() {
                            Some(info) => texture_paths[info.texture().index()].as_bytes(),
                            None => &[],
                        },
                        base_color_factor: pbr.base_color_factor(),
                    });
                }
                for g_scene in document.scenes() {
                    for g_node in g_scene.nodes() {
                        self.populate_gltf(&mut model, g_node, glam::Mat4::IDENTITY, &buffers);
                    }
                }
                result.put(model);
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
                let path_str = str::from_utf8(material.base_color_path).unwrap();
                let (handle, task) = self.asset_textures.load(
                    path_str.as_ref(),
                    crate::texture::Meta {
                        format: blade::TextureFormat::Bc1UnormSrgb,
                    },
                );
                exe_context.add_fork(&task);
                Some(handle)
            };
            materials.push(Material {
                base_color_texture,
                base_color_factor: material.base_color_factor,
            });
        }

        let total_vertices = model
            .geometries
            .iter()
            .map(|geo| geo.vertices.len())
            .sum::<usize>();
        let total_vertex_size = (total_vertices * mem::size_of::<crate::Vertex>()) as u64;
        let vertex_buffer = self.gpu_context.create_buffer(blade::BufferDesc {
            name: "vertex",
            size: total_vertex_size,
            memory: blade::Memory::Device,
        });
        let vertex_stage = self.gpu_context.create_buffer(blade::BufferDesc {
            name: "vertex stage",
            size: total_vertex_size,
            memory: blade::Memory::Upload,
        });

        let total_indices = model
            .geometries
            .iter()
            .map(|geo| geo.indices.len())
            .sum::<usize>();
        let total_index_size = total_indices as u64 * 4;
        let index_buffer = self.gpu_context.create_buffer(blade::BufferDesc {
            name: "index",
            size: total_index_size,
            memory: blade::Memory::Device,
        });
        let index_stage = self.gpu_context.create_buffer(blade::BufferDesc {
            name: "index stage",
            size: total_index_size,
            memory: blade::Memory::Upload,
        });

        let total_transform_size =
            (model.geometries.len() * mem::size_of::<blade::Transform>()) as u64;
        let transform_buffer = self.gpu_context.create_buffer(blade::BufferDesc {
            name: "transform",
            size: total_transform_size,
            memory: blade::Memory::Device,
        });
        let transform_stage = self.gpu_context.create_buffer(blade::BufferDesc {
            name: "transform stage",
            size: total_transform_size,
            memory: blade::Memory::Upload,
        });

        let mut meshes = Vec::with_capacity(model.geometries.len());
        let vertex_stride = mem::size_of::<super::Vertex>() as u32;
        let mut start_vertex = 0;
        let mut index_offset = 0;
        let mut transform_offset = 0;
        let mut geometries = Vec::with_capacity(model.geometries.len());
        for geometry in model.geometries.iter() {
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
                    mem::size_of::<blade::Transform>(),
                );
            }
            let index_type = if geometry.indices.is_empty() {
                None
            } else {
                Some(blade::IndexType::U32)
            };
            let triangle_count = if geometry.indices.is_empty() {
                geometry.vertices.len() as u32 / 3
            } else {
                geometry.indices.len() as u32 / 3
            };
            meshes.push(blade::AccelerationStructureMesh {
                vertex_data: vertex_buffer.at(start_vertex as u64 * vertex_stride as u64),
                vertex_format: blade::VertexFormat::F32Vec3,
                vertex_stride,
                vertex_count: geometry.vertices.len() as u32,
                index_data: index_buffer.at(index_offset),
                index_type,
                triangle_count,
                transform_data: transform_buffer.at(transform_offset), //TODO
                is_opaque: true,                                       //TODO
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
            transform_offset += mem::size_of::<blade::Transform>() as u64;
        }
        assert_eq!(start_vertex as usize, total_vertices);
        assert_eq!(index_offset, total_index_size);
        assert_eq!(transform_offset, total_transform_size);

        let sizes = self
            .gpu_context
            .get_bottom_level_acceleration_structure_sizes(&meshes);
        let acceleration_structure =
            self.gpu_context
                .create_acceleration_structure(blade::AccelerationStructureDesc {
                    name: str::from_utf8(model.name).unwrap(),
                    ty: blade::AccelerationStructureType::BottomLevel,
                    size: sizes.data,
                });
        let scratch = self.gpu_context.create_buffer(blade::BufferDesc {
            name: "BLAS scratch",
            size: sizes.scratch,
            memory: blade::Memory::Device,
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
            name: String::from_utf8_lossy(model.name.as_ref()).into_owned(),
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
