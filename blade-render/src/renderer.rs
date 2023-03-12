use std::mem;

const TARGET_FORMAT: blade::TextureFormat = blade::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Parameters {
    cam_position: [f32; 3],
    depth: f32,
    cam_orientation: [f32; 4],
    fov: [f32; 2],
    pad: [f32; 2],
}

#[derive(blade_macros::ShaderData)]
struct ShaderData {
    parameters: Parameters,
    acc_struct: blade::AccelerationStructure,
    output: blade::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct DrawData {
    input: blade::TextureView,
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
    ) -> blade::AccelerationStructure {
        let mut instances = Vec::with_capacity(self.objects.len());
        let mut blases = Vec::with_capacity(self.objects.len());

        for object in self.objects.iter() {
            instances.push(blade::AccelerationStructureInstance {
                acceleration_structure_index: blases.len() as u32,
                transform: object.transform.into(),
                mask: 0xFF,
            });
            blases.push(object.acceleration_structure);
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
        acceleration_structure
    }
}

impl super::Renderer {
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

        Self {
            target,
            target_view,
            scene: super::Scene::default(),
            rt_pipeline,
            draw_pipeline,
            acceleration_structure: blade::AccelerationStructure::default(),
            is_tlas_dirty: true,
            screen_size,
        }
    }

    pub fn destroy(&mut self, gpu: &blade::Context) {
        for mut object in self.scene.objects.drain(..) {
            for geometry in object.geometries.drain(..) {
                gpu.destroy_buffer(geometry.vertex_buf);
                if geometry.index_type.is_some() {
                    gpu.destroy_buffer(geometry.index_buf);
                }
            }
            gpu.destroy_acceleration_structure(object.acceleration_structure);
        }
        gpu.destroy_texture_view(self.target_view);
        gpu.destroy_texture(self.target);
        gpu.destroy_acceleration_structure(self.acceleration_structure);
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
        if self.is_tlas_dirty {
            self.acceleration_structure = self.scene.build_top_level_acceleration_structure(
                command_encoder,
                gpu,
                temp_buffers,
            );
            self.is_tlas_dirty = false;
        }
    }

    pub fn ray_trace(&self, command_encoder: &mut blade::CommandEncoder, camera: &super::Camera) {
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
                    depth: 100.0,
                    cam_orientation: camera.rot.into(),
                    fov: [fov_x, camera.fov_y],
                    pad: [0.0; 2],
                },
                acc_struct: self.acceleration_structure,
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
