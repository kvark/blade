use std::mem;

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

    pub(super) fn build_top_level_acceleration_structure(
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
