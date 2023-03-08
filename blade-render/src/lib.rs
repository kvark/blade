mod gltf_loader;

use std::mem;

pub struct Vertex {
    pub position: [f32; 3],
}

pub struct Geometry {
    pub vertex_buf: bg::Buffer,
    pub vertex_count: u32,
    pub index_buf: bg::Buffer,
    pub index_type: Option<bg::IndexType>,
    pub triangle_count: u32,
}

pub struct Object {
    pub name: String,
    pub geometries: Vec<Geometry>,
    pub transform: bg::Transform,
    pub acceleration_structure: bg::AccelerationStructure,
}

#[derive(Default)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub acceleration_structure: bg::AccelerationStructure,
}

impl Scene {
    fn populate_acceleration_structures(
        &mut self,
        gpu: &bg::Context,
        encoder: &mut bg::CommandEncoder,
    ) -> Vec<bg::Buffer> {
        let mut temp_buffers = Vec::new();
        let mut blas_encoder = encoder.acceleration_structure();
        let mut meshes = Vec::new();
        let mut blases = Vec::new();
        let mut instances = Vec::new();

        for object in self.objects.iter_mut() {
            log::debug!("Object {}", object.name);
            meshes.clear();
            for geometry in object.geometries.iter() {
                log::debug!(
                    "Geomtry vertices {}, trianges {}",
                    geometry.vertex_count,
                    geometry.triangle_count
                );
                meshes.push(bg::AccelerationStructureMesh {
                    vertex_data: geometry.vertex_buf.at(0),
                    vertex_format: bg::VertexFormat::F32Vec3,
                    vertex_stride: mem::size_of::<Vertex>() as u32,
                    vertex_count: geometry.vertex_count,
                    index_data: geometry.index_buf.at(0),
                    index_type: geometry.index_type,
                    triangle_count: geometry.triangle_count,
                    transform_data: bg::Buffer::default().at(0),
                    is_opaque: true,
                });
            }
            let sizes = gpu.get_bottom_level_acceleration_structure_sizes(&meshes);
            object.acceleration_structure =
                gpu.create_acceleration_structure(bg::AccelerationStructureDesc {
                    name: &object.name,
                    ty: bg::AccelerationStructureType::BottomLevel,
                    size: sizes.data,
                });
            let scratch_buffer = gpu.create_buffer(bg::BufferDesc {
                name: "BLAS scratch",
                size: sizes.scratch,
                memory: bg::Memory::Device,
            });
            blas_encoder.build_bottom_level(
                object.acceleration_structure,
                &meshes,
                scratch_buffer.at(0),
            );
            temp_buffers.push(scratch_buffer);
            instances.push(bg::AccelerationStructureInstance {
                acceleration_structure_index: blases.len() as u32,
                transform: object.transform.into(),
                mask: 0xFF,
            });
            blases.push(object.acceleration_structure);
        }

        // Needs to be a separate encoder in order to force synchronization
        let sizes = gpu.get_top_level_acceleration_structure_sizes(instances.len() as u32);
        self.acceleration_structure =
            gpu.create_acceleration_structure(bg::AccelerationStructureDesc {
                name: "TLAS",
                ty: bg::AccelerationStructureType::TopLevel,
                size: sizes.data,
            });
        let instance_buf = gpu.create_acceleration_structure_instance_buffer(&instances, &blases);
        let scratch_buf = gpu.create_buffer(bg::BufferDesc {
            name: "TLAS scratch",
            size: sizes.scratch,
            memory: bg::Memory::Device,
        });

        let mut tlas_encoder = encoder.acceleration_structure();
        tlas_encoder.build_top_level(
            self.acceleration_structure,
            &blases,
            instances.len() as u32,
            instance_buf.at(0),
            scratch_buf.at(0),
        );

        temp_buffers.push(instance_buf);
        temp_buffers.push(scratch_buf);
        temp_buffers
    }

    pub fn destroy(&mut self, gpu: &bg::Context) {
        for mut object in self.objects.drain(..) {
            for geometry in object.geometries.drain(..) {
                gpu.destroy_buffer(geometry.vertex_buf);
                if geometry.index_type.is_some() {
                    gpu.destroy_buffer(geometry.index_buf);
                }
            }
            gpu.destroy_acceleration_structure(object.acceleration_structure);
        }
        gpu.destroy_acceleration_structure(self.acceleration_structure);
    }
}
