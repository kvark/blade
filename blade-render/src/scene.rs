impl super::Scene {
    pub(super) fn build_top_level_acceleration_structure(
        &self,
        command_encoder: &mut blade::CommandEncoder,
        asset_models: &blade_asset::AssetManager<crate::model::Baker>,
        gpu: &blade::Context,
        temp_buffers: &mut Vec<blade::Buffer>,
    ) -> (blade::AccelerationStructure, u32) {
        let mut instances = Vec::with_capacity(self.objects.len());
        let mut blases = Vec::with_capacity(self.objects.len());
        let mut custom_index = 0;

        for object in self.objects.iter() {
            let model = &asset_models[object.model];
            instances.push(blade::AccelerationStructureInstance {
                acceleration_structure_index: blases.len() as u32,
                transform: object.transform.into(),
                mask: 0xFF,
                custom_index,
            });
            blases.push(model.acceleration_structure);
            custom_index += model.geometries.len() as u32;
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
