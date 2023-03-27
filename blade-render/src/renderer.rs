use std::{mem, ptr};

const TARGET_FORMAT: blade::TextureFormat = blade::TextureFormat::Rgba16Float;
const MAX_RESOURCES: u32 = 1000;

struct DummyResources {
    size: blade::Extent,
    white_texture: blade::Texture,
    white_view: blade::TextureView,
}

struct Samplers {
    linear: blade::Sampler,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(u32)]
pub enum DebugMode {
    None = 0,
    Depth = 1,
    Normal = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct RayConfig {
    pub num_environment_samples: u32,
}

pub struct Renderer {
    target: blade::Texture,
    target_view: blade::TextureView,
    rt_pipeline: blade::ComputePipeline,
    draw_pipeline: blade::RenderPipeline,
    scene: super::Scene,
    acceleration_structure: blade::AccelerationStructure,
    dummy: DummyResources,
    hit_buffer: blade::Buffer,
    vertex_buffers: blade::BufferArray<MAX_RESOURCES>,
    index_buffers: blade::BufferArray<MAX_RESOURCES>,
    textures: blade::TextureArray<MAX_RESOURCES>,
    samplers: Samplers,
    is_tlas_dirty: bool,
    screen_size: blade::Extent,
    frame_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Parameters {
    cam_position: [f32; 3],
    depth: f32,
    cam_orientation: [f32; 4],
    fov: [f32; 2],
    frame_index: u32,
    debug_mode: u32,
    num_environment_samples: u32,
    pad: [u32; 3],
}

#[derive(blade_macros::ShaderData)]
struct ShaderData<'a> {
    parameters: Parameters,
    acc_struct: blade::AccelerationStructure,
    hit_entries: blade::BufferPiece,
    index_buffers: &'a blade::BufferArray<MAX_RESOURCES>,
    vertex_buffers: &'a blade::BufferArray<MAX_RESOURCES>,
    textures: &'a blade::TextureArray<MAX_RESOURCES>,
    sampler_linear: blade::Sampler,
    output: blade::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct DrawData {
    input: blade::TextureView,
}

#[repr(C)]
#[derive(Debug)]
struct HitEntry {
    index_buf: u32,
    vertex_buf: u32,
    rotation: [i8; 4],
    //geometry_to_object: mint::RowMatrix3x4<f32>,
    base_color_texture: u32,
    base_color_factor: [u8; 4],
}

impl super::Scene {
    pub(super) fn populate_bottom_level_acceleration_structures(
        &mut self,
        gpu: &blade::Context,
        encoder: &mut blade::CommandEncoder,
        temp_buffers: &mut Vec<blade::Buffer>,
    ) {
        let mut blas_encoder = encoder.acceleration_structure();
        let mut meshes = Vec::new();

        for object in self.objects.iter_mut() {
            log::debug!("Object {}", object.name);
            meshes.clear();
            for geometry in object.geometries.iter() {
                log::debug!(
                    "Geomtry vertices {}, trianges {}",
                    geometry.vertex_count,
                    geometry.triangle_count
                );
                meshes.push(blade::AccelerationStructureMesh {
                    vertex_data: geometry.vertex_buf.at(0),
                    vertex_format: blade::VertexFormat::F32Vec3,
                    vertex_stride: mem::size_of::<super::Vertex>() as u32,
                    vertex_count: geometry.vertex_count,
                    index_data: geometry.index_buf.at(0),
                    index_type: geometry.index_type,
                    triangle_count: geometry.triangle_count,
                    transform_data: blade::Buffer::default().at(0),
                    is_opaque: true,
                });
            }
            let sizes = gpu.get_bottom_level_acceleration_structure_sizes(&meshes);
            object.acceleration_structure =
                gpu.create_acceleration_structure(blade::AccelerationStructureDesc {
                    name: &object.name,
                    ty: blade::AccelerationStructureType::BottomLevel,
                    size: sizes.data,
                });
            let scratch_buffer = gpu.create_buffer(blade::BufferDesc {
                name: "BLAS scratch",
                size: sizes.scratch,
                memory: blade::Memory::Device,
            });
            blas_encoder.build_bottom_level(
                object.acceleration_structure,
                &meshes,
                scratch_buffer.at(0),
            );
            temp_buffers.push(scratch_buffer);
        }
    }

    fn build_top_level_acceleration_structure(
        &self,
        command_encoder: &mut blade::CommandEncoder,
        gpu: &blade::Context,
        temp_buffers: &mut Vec<blade::Buffer>,
    ) -> (blade::AccelerationStructure, u32) {
        let mut instances = Vec::with_capacity(self.objects.len());
        let mut blases = Vec::with_capacity(self.objects.len());
        let mut custom_index = 0;

        for object in self.objects.iter() {
            instances.push(blade::AccelerationStructureInstance {
                acceleration_structure_index: blases.len() as u32,
                transform: object.transform.into(),
                mask: 0xFF,
                custom_index,
            });
            blases.push(object.acceleration_structure);
            custom_index += object.geometries.len() as u32;
        }

        // Needs to be a separate encoder in order to force synchronization
        let sizes = gpu.get_top_level_acceleration_structure_sizes(instances.len() as u32);
        let acceleration_structure =
            gpu.create_acceleration_structure(blade::AccelerationStructureDesc {
                name: "TLAS",
                ty: blade::AccelerationStructureType::TopLevel,
                size: sizes.data,
            });
        let instance_buf = gpu.create_acceleration_structure_instance_buffer(&instances, &blases);
        let scratch_buf = gpu.create_buffer(blade::BufferDesc {
            name: "TLAS scratch",
            size: sizes.scratch,
            memory: blade::Memory::Device,
        });

        let mut tlas_encoder = command_encoder.acceleration_structure();
        tlas_encoder.build_top_level(
            acceleration_structure,
            &blases,
            instances.len() as u32,
            instance_buf.at(0),
            scratch_buf.at(0),
        );

        temp_buffers.push(instance_buf);
        temp_buffers.push(scratch_buf);
        (acceleration_structure, custom_index)
    }
}

impl Renderer {
    pub fn new(
        encoder: &mut blade::CommandEncoder,
        context: &blade::Context,
        screen_size: blade::Extent,
        surface_format: blade::TextureFormat,
    ) -> Self {
        let capabilities = context.capabilities();
        assert!(capabilities
            .ray_query
            .contains(blade::ShaderVisibility::COMPUTE));

        let target = context.create_texture(blade::TextureDesc {
            name: "main",
            format: TARGET_FORMAT,
            size: screen_size,
            dimension: blade::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: blade::TextureUsage::RESOURCE | blade::TextureUsage::STORAGE,
        });
        let target_view = context.create_texture_view(blade::TextureViewDesc {
            name: "main",
            texture: target,
            format: TARGET_FORMAT,
            dimension: blade::ViewDimension::D2,
            subresources: &blade::TextureSubresources::default(),
        });

        let source = std::fs::read_to_string("blade-render/shader.wgsl").unwrap();
        let shader = context.create_shader(blade::ShaderDesc { source: &source });
        let rt_layout = <ShaderData as blade::ShaderData>::layout();
        let rt_pipeline = context.create_compute_pipeline(blade::ComputePipelineDesc {
            name: "ray-trace",
            data_layouts: &[&rt_layout],
            compute: shader.at("main"),
        });
        let draw_layout = <DrawData as blade::ShaderData>::layout();
        let draw_pipeline = context.create_render_pipeline(blade::RenderPipelineDesc {
            name: "main",
            data_layouts: &[&draw_layout],
            primitive: blade::PrimitiveState {
                topology: blade::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            vertex: shader.at("draw_vs"),
            fragment: shader.at("draw_fs"),
            color_targets: &[surface_format.into()],
            depth_stencil: None,
        });

        encoder.init_texture(target);

        let dummy = {
            let size = blade::Extent {
                width: 1,
                height: 1,
                depth: 1,
            };
            let white_texture = context.create_texture(blade::TextureDesc {
                name: "dummy/white",
                format: blade::TextureFormat::Rgba8Unorm,
                size,
                array_layer_count: 1,
                mip_level_count: 1,
                dimension: blade::TextureDimension::D2,
                usage: blade::TextureUsage::COPY | blade::TextureUsage::RESOURCE,
            });
            let white_view = context.create_texture_view(blade::TextureViewDesc {
                name: "dummy/white",
                texture: white_texture,
                format: blade::TextureFormat::Rgba8Unorm,
                dimension: blade::ViewDimension::D2,
                subresources: &blade::TextureSubresources::default(),
            });
            encoder.init_texture(white_texture);
            DummyResources {
                size,
                white_texture,
                white_view,
            }
        };

        let samplers = Samplers {
            linear: context.create_sampler(blade::SamplerDesc {
                name: "linear",
                address_modes: [blade::AddressMode::ClampToEdge; 3],
                mag_filter: blade::FilterMode::Linear,
                min_filter: blade::FilterMode::Linear,
                mipmap_filter: blade::FilterMode::Linear,
                ..Default::default()
            }),
        };

        Self {
            target,
            target_view,
            scene: super::Scene::default(),
            rt_pipeline,
            draw_pipeline,
            acceleration_structure: blade::AccelerationStructure::default(),
            dummy,
            hit_buffer: blade::Buffer::default(),
            vertex_buffers: blade::BufferArray::new(),
            index_buffers: blade::BufferArray::new(),
            textures: blade::TextureArray::new(),
            samplers,
            is_tlas_dirty: true,
            screen_size,
            frame_index: 0,
        }
    }

    pub fn destroy(&mut self, gpu: &blade::Context) {
        // scene
        for texture in self.scene.textures.drain(..) {
            gpu.destroy_texture_view(texture.view);
            gpu.destroy_texture(texture.texture);
        }
        for mut object in self.scene.objects.drain(..) {
            for geometry in object.geometries.drain(..) {
                gpu.destroy_buffer(geometry.vertex_buf);
                if geometry.index_type.is_some() {
                    gpu.destroy_buffer(geometry.index_buf);
                }
            }
            gpu.destroy_acceleration_structure(object.acceleration_structure);
        }
        // internal resources
        gpu.destroy_texture_view(self.target_view);
        gpu.destroy_texture(self.target);
        if self.hit_buffer != blade::Buffer::default() {
            gpu.destroy_buffer(self.hit_buffer);
        }
        gpu.destroy_acceleration_structure(self.acceleration_structure);
        // dummy resources
        gpu.destroy_texture_view(self.dummy.white_view);
        gpu.destroy_texture(self.dummy.white_texture);
        // samplers
        gpu.destroy_sampler(self.samplers.linear);
    }

    pub fn merge_scene(&mut self, scene: super::Scene) {
        self.scene = scene;
    }

    pub fn prepare(
        &mut self,
        command_encoder: &mut blade::CommandEncoder,
        gpu: &blade::Context,
        temp_buffers: &mut Vec<blade::Buffer>,
    ) {
        self.frame_index += 1;
        if self.is_tlas_dirty {
            self.is_tlas_dirty = false;
            if self.acceleration_structure != blade::AccelerationStructure::default() {
                temp_buffers.push(self.hit_buffer);
                //TODO: delay this or stall the GPU
                gpu.destroy_acceleration_structure(self.acceleration_structure);
            }

            let (tlas, geometry_count) = self.scene.build_top_level_acceleration_structure(
                command_encoder,
                gpu,
                temp_buffers,
            );
            self.acceleration_structure = tlas;
            log::info!("Preparing ray tracing with {geometry_count} geometries in total");
            let mut transfers = command_encoder.transfer();

            {
                // init the dummy
                let staging = gpu.create_buffer(blade::BufferDesc {
                    name: "dummy staging",
                    size: 4,
                    memory: blade::Memory::Upload,
                });
                unsafe {
                    ptr::write(staging.data() as *mut _, [!0u8; 4]);
                }
                transfers.copy_buffer_to_texture(
                    staging.into(),
                    4,
                    self.dummy.white_texture.into(),
                    self.dummy.size,
                );
                temp_buffers.push(staging);
            }
            let hit_staging = {
                // init the hit buffer
                let hit_size = (geometry_count as usize * mem::size_of::<HitEntry>()) as u64;
                self.hit_buffer = gpu.create_buffer(blade::BufferDesc {
                    name: "hit entries",
                    size: hit_size,
                    memory: blade::Memory::Device,
                });
                let staging = gpu.create_buffer(blade::BufferDesc {
                    name: "hit staging",
                    size: hit_size,
                    memory: blade::Memory::Upload,
                });
                temp_buffers.push(staging);
                transfers.copy_buffer_to_buffer(staging.at(0), self.hit_buffer.at(0), hit_size);
                staging
            };

            self.textures.clear();
            let dummy_white = self.textures.alloc(self.dummy.white_view);
            let mut texture_indices = Vec::with_capacity(self.scene.textures.len());
            for texture in self.scene.textures.iter() {
                texture_indices.push(self.textures.alloc(texture.view));
            }

            self.vertex_buffers.clear();
            self.index_buffers.clear();
            let mut geometry_index = 0;
            for object in self.scene.objects.iter() {
                let rotation = {
                    let col_matrix = mint::ColumnMatrix3x4::from(object.transform);
                    let m3 = glam::Mat3::from_cols(
                        col_matrix.x.into(),
                        col_matrix.y.into(),
                        col_matrix.z.into(),
                    );
                    let m3_normal = m3.inverse().transpose();
                    let quat = glam::Quat::from_mat3(&m3_normal);
                    let qv = glam::Vec4::from(quat) * 127.0;
                    [qv.x as i8, qv.y as i8, qv.z as i8, qv.w as i8]
                };
                for geometry in object.geometries.iter() {
                    let material = &self.scene.materials[geometry.material_index];
                    let hit_entry = HitEntry {
                        index_buf: match geometry.index_type {
                            Some(_) => self.index_buffers.alloc(geometry.index_buf.at(0)),
                            None => !0,
                        },
                        vertex_buf: self.vertex_buffers.alloc(geometry.vertex_buf.at(0)),
                        rotation,
                        base_color_texture: if material.base_color_texture_index == !0 {
                            dummy_white
                        } else {
                            texture_indices[material.base_color_texture_index]
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
            assert_eq!(geometry_index, geometry_count as usize);
        }
    }

    pub fn ray_trace(
        &self,
        command_encoder: &mut blade::CommandEncoder,
        camera: &super::Camera,
        debug_mode: DebugMode,
        ray_config: RayConfig,
    ) {
        assert!(!self.is_tlas_dirty);

        let mut pass = command_encoder.compute();
        let mut pc = pass.with(&self.rt_pipeline);
        let wg_size = self.rt_pipeline.get_workgroup_size();
        let group_count = [
            (self.screen_size.width + wg_size[0] - 1) / wg_size[0],
            (self.screen_size.height + wg_size[1] - 1) / wg_size[1],
            1,
        ];
        let fov_x = camera.fov_y * self.screen_size.width as f32 / self.screen_size.height as f32;

        pc.bind(
            0,
            &ShaderData {
                parameters: Parameters {
                    cam_position: camera.pos.into(),
                    depth: camera.depth,
                    cam_orientation: camera.rot.into(),
                    fov: [fov_x, camera.fov_y],
                    frame_index: self.frame_index,
                    debug_mode: debug_mode as u32,
                    num_environment_samples: ray_config.num_environment_samples,
                    pad: [0; 3],
                },
                acc_struct: self.acceleration_structure,
                hit_entries: self.hit_buffer.at(0),
                index_buffers: &self.index_buffers,
                vertex_buffers: &self.vertex_buffers,
                textures: &self.textures,
                sampler_linear: self.samplers.linear,
                output: self.target_view,
            },
        );
        pc.dispatch(group_count);
    }

    pub fn blit(&self, pass: &mut blade::RenderCommandEncoder) {
        let mut pc = pass.with(&self.draw_pipeline);
        pc.bind(
            0,
            &DrawData {
                input: self.target_view,
            },
        );
        pc.draw(0, 3, 0, 1);
    }
}
