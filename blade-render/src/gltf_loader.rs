use std::{mem, path::Path, ptr};

struct LoadContext<'a> {
    gltf_buffers: &'a [gltf::buffer::Data],
    gpu: &'a blade::Context,
    encoder: blade::TransferCommandEncoder<'a>,
    temp_buffers: Vec<blade::Buffer>,
    scene: &'a mut super::Scene,
}

impl LoadContext<'_> {
    fn populate(&mut self, g_node: gltf::Node, parent_transform: glam::Mat4) {
        let local_transform = glam::Mat4::from_cols_array_2d(&g_node.transform().matrix());
        let transform = parent_transform * local_transform;

        if let Some(g_mesh) = g_node.mesh() {
            let col_matrix = mint::ColumnMatrix3x4 {
                x: transform.x_axis.truncate().into(),
                y: transform.y_axis.truncate().into(),
                z: transform.z_axis.truncate().into(),
                w: transform.w_axis.truncate().into(),
            };
            let mut object = super::Object {
                name: g_node.name().map_or(String::new(), str::to_string),
                geometries: Vec::new(),
                transform: mint::RowMatrix3x4::from(col_matrix).into(),
                acceleration_structure: blade::AccelerationStructure::default(),
            };

            for g_primitive in g_mesh.primitives() {
                let mut geometry = super::Geometry {
                    vertex_buf: blade::Buffer::default(),
                    vertex_count: 0,
                    index_buf: blade::Buffer::default(),
                    index_type: None,
                    triangle_count: 0,
                };
                let vertex_count = g_primitive.get(&gltf::Semantic::Positions).unwrap().count();
                let vertex_buf_size = mem::size_of::<super::Vertex>() * vertex_count;
                geometry.vertex_count = vertex_count as u32;
                geometry.vertex_buf = self.gpu.create_buffer(blade::BufferDesc {
                    name: "vertex",
                    size: vertex_buf_size as u64,
                    memory: blade::Memory::Device,
                });

                let reader = g_primitive.reader(|buffer| Some(&self.gltf_buffers[buffer.index()]));
                let mut indices = Vec::new();
                if let Some(read) = reader.read_indices() {
                    indices.extend(read.into_u32());
                    geometry.index_buf = self.gpu.create_buffer(blade::BufferDesc {
                        name: "index",
                        size: indices.len() as u64 * 4,
                        memory: blade::Memory::Device,
                    });
                    geometry.index_type = Some(blade::IndexType::U32);
                    geometry.triangle_count = indices.len() as u32 / 3;
                } else {
                    geometry.triangle_count = vertex_count as u32 / 3;
                }

                let staging_size = vertex_buf_size + indices.len() * 4;
                let staging_buf = self.gpu.create_buffer(blade::BufferDesc {
                    name: "mesh staging",
                    size: staging_size as u64,
                    memory: blade::Memory::Upload,
                });
                for (i, pos) in reader.read_positions().unwrap().enumerate() {
                    unsafe {
                        (*(staging_buf.data() as *mut super::Vertex).add(i)).position = pos;
                    }
                }
                self.encoder.copy_buffer_to_buffer(
                    staging_buf.at(0),
                    geometry.vertex_buf.at(0),
                    vertex_buf_size as u64,
                );
                if !indices.is_empty() {
                    unsafe {
                        ptr::copy_nonoverlapping(
                            indices.as_ptr(),
                            staging_buf.data().add(vertex_buf_size) as *mut u32,
                            indices.len(),
                        );
                    }
                    self.encoder.copy_buffer_to_buffer(
                        staging_buf.at(vertex_buf_size as u64),
                        geometry.index_buf.at(0),
                        indices.len() as u64 * 4,
                    );
                }
                self.temp_buffers.push(staging_buf);

                object.geometries.push(geometry);
            }

            self.scene.objects.push(object);
        }

        for child in g_node.children() {
            self.populate(child, transform);
        }
    }
}

impl super::Scene {
    pub fn load_gltf(
        path: &Path,
        encoder: &mut blade::CommandEncoder,
        gpu: &blade::Context,
    ) -> (Self, Vec<blade::Buffer>) {
        let (doc, buffers, _images) = gltf::import(path).unwrap();
        let mut scene = super::Scene::default();
        let g_scene = doc.scenes().next().unwrap();

        let mut temp_buffers = {
            let mut loader = LoadContext {
                gltf_buffers: &buffers,
                gpu,
                encoder: encoder.transfer(),
                temp_buffers: Vec::new(),
                scene: &mut scene,
            };
            for g_node in g_scene.nodes() {
                loader.populate(g_node, glam::Mat4::IDENTITY);
            }
            loader.temp_buffers
        };

        scene.populate_bottom_level_acceleration_structures(gpu, encoder, &mut temp_buffers);
        (scene, temp_buffers)
    }
}
