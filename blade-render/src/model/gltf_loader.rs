use core::slice;
use std::{mem, path::Path, ptr};

struct LoadContext<'a> {
    gltf_buffers: &'a [gltf::buffer::Data],
    gpu: &'a blade::Context,
    encoder: blade::TransferCommandEncoder<'a>,
    temp_buffers: Vec<blade::Buffer>,
    scene: &'a mut crate::Scene,
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
            let mut object = crate::Object {
                name: g_node.name().map_or(String::new(), str::to_string),
                geometries: Vec::new(),
                transform: mint::RowMatrix3x4::from(col_matrix).into(),
                acceleration_structure: blade::AccelerationStructure::default(),
            };

            for g_primitive in g_mesh.primitives() {
                if g_primitive.mode() != gltf::mesh::Mode::Triangles {
                    log::warn!(
                        "Skipping primitive for having mesh mode {:?}",
                        g_primitive.mode()
                    );
                    continue;
                }
                let material_index = match g_primitive.material().index() {
                    Some(index) => index,
                    None => {
                        log::warn!("Skipping primitive for having default material");
                        continue;
                    }
                };

                let mut geometry = crate::Geometry {
                    vertex_buf: blade::Buffer::default(),
                    vertex_count: 0,
                    index_buf: blade::Buffer::default(),
                    index_type: None,
                    triangle_count: 0,
                    material_index,
                };
                let vertex_count = g_primitive.get(&gltf::Semantic::Positions).unwrap().count();
                let vertex_buf_size = mem::size_of::<crate::Vertex>() * vertex_count;
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
                unsafe {
                    ptr::write_bytes(staging_buf.data(), 0, staging_size);
                }
                for (i, pos) in reader.read_positions().unwrap().enumerate() {
                    unsafe {
                        (*(staging_buf.data() as *mut crate::Vertex).add(i)).position = pos;
                    }
                }
                if let Some(iter) = reader.read_tex_coords(0) {
                    for (i, tc) in iter.into_f32().enumerate() {
                        unsafe {
                            (*(staging_buf.data() as *mut crate::Vertex).add(i)).tex_coords = tc;
                        }
                    }
                }
                if let Some(iter) = reader.read_normals() {
                    for (i, normal) in iter.enumerate() {
                        // convert floating point to i16 normalized
                        let nu = [
                            (normal[0] * i16::MAX as f32) as i16,
                            (normal[1] * i16::MAX as f32) as i16,
                        ];
                        unsafe {
                            (*(staging_buf.data() as *mut crate::Vertex).add(i)).normal = nu;
                        }
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

impl crate::Scene {
    pub fn load_gltf(
        path: &Path,
        encoder: &mut blade::CommandEncoder,
        gpu: &blade::Context,
    ) -> (Self, Vec<blade::Buffer>) {
        let (doc, buffers, images) = gltf::import(path).unwrap();
        let mut scene = crate::Scene::default();
        let g_scene = doc.scenes().next().unwrap();
        let mut temp_buffers = Vec::new();

        for g_texture in doc.textures() {
            let img_data = &images[g_texture.source().index()];
            let (format, source_bytes_pp) = match img_data.format {
                gltf::image::Format::R8G8B8 => (blade::TextureFormat::Rgba8UnormSrgb, 3),
                gltf::image::Format::R8G8B8A8 => (blade::TextureFormat::Rgba8UnormSrgb, 4),
                other => panic!("Unsupported image format {:?}", other),
            };
            let size = blade::Extent {
                width: img_data.width,
                height: img_data.height,
                depth: 1,
            };
            let name = g_texture.name().unwrap_or_default();
            let texture = gpu.create_texture(blade::TextureDesc {
                name,
                format,
                size,
                array_layer_count: 1,
                mip_level_count: 1, //TODO
                dimension: blade::TextureDimension::D2,
                usage: blade::TextureUsage::COPY | blade::TextureUsage::RESOURCE,
            });
            let view = gpu.create_texture_view(blade::TextureViewDesc {
                name,
                format,
                texture,
                dimension: blade::ViewDimension::D2,
                subresources: &blade::TextureSubresources::default(),
            });
            let bytes_pp = format.block_info().size as u32;
            let stage_size = (img_data.width * img_data.height * bytes_pp) as u64;
            let staging = gpu.create_buffer(blade::BufferDesc {
                name: "staging",
                size: stage_size,
                memory: blade::Memory::Upload,
            });
            let staging_slice =
                unsafe { slice::from_raw_parts_mut(staging.data(), stage_size as usize) };
            for (src_bytes, dst_bytes) in img_data
                .pixels
                .chunks(source_bytes_pp)
                .zip(staging_slice.chunks_mut(bytes_pp as usize))
            {
                let (dst_copy, dst_fill) = dst_bytes.split_at_mut(source_bytes_pp);
                dst_copy.copy_from_slice(src_bytes);
                dst_fill.fill(!0);
            }

            encoder.init_texture(texture);
            encoder.transfer().copy_buffer_to_texture(
                staging.into(),
                img_data.width * bytes_pp,
                texture.into(),
                size,
            );
            temp_buffers.push(staging);
            scene.textures.push(crate::Texture { texture, view });
        }

        for g_material in doc.materials() {
            let pbr = g_material.pbr_metallic_roughness();
            scene.materials.push(crate::Material {
                base_color_texture_index: match pbr.base_color_texture() {
                    Some(info) => info.texture().index(),
                    None => !0,
                },
                base_color_factor: pbr.base_color_factor(),
            });
        }

        let mut temp_buffers = {
            let mut loader = LoadContext {
                gltf_buffers: &buffers,
                gpu,
                encoder: encoder.transfer(),
                temp_buffers,
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
