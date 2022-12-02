use ash::vk;
use naga::back::spv;
use std::{ffi, mem, ptr};

struct CompiledShader {
    vk_module: vk::ShaderModule,
    _entry_point: ffi::CString,
    create_info: vk::PipelineShaderStageCreateInfo,
    wg_size: [u32; 3],
}

fn map_shader_visibility(visibility: crate::ShaderVisibility) -> vk::ShaderStageFlags {
    use crate::ShaderVisibility as Sv;
    use vk::ShaderStageFlags as Flags;

    let mut flags = Flags::empty();
    if visibility.contains(Sv::COMPUTE) {
        flags |= Flags::COMPUTE;
    }
    if visibility.contains(Sv::VERTEX) {
        flags |= Flags::VERTEX;
    }
    if visibility.contains(Sv::FRAGMENT) {
        flags |= Flags::FRAGMENT;
    }

    flags
}

impl super::Context {
    fn load_shader(
        &self,
        sf: crate::ShaderFunction,
        naga_stage: naga::ShaderStage,
        naga_options: &spv::Options,
    ) -> CompiledShader {
        let pipeline_options = spv::PipelineOptions {
            shader_stage: naga_stage,
            entry_point: sf.entry_point.to_string(),
        };

        let spv = spv::write_vec(
            &sf.shader.module,
            &sf.shader.info,
            naga_options,
            Some(&pipeline_options),
        )
        .unwrap();

        let ep = sf
            .shader
            .module
            .entry_points
            .iter()
            .find(|ep| ep.stage == naga_stage && ep.name == sf.entry_point)
            .expect("Entry point not found in the shader");

        let vk_info = vk::ShaderModuleCreateInfo::builder().code(&spv);

        let vk_module = unsafe {
            self.device
                .core
                .create_shader_module(&vk_info, None)
                .unwrap()
        };

        let vk_stage = match naga_stage {
            naga::ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
            naga::ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            naga::ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        };

        let entry_point = ffi::CString::new(sf.entry_point).unwrap();
        let create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk_stage)
            .module(vk_module)
            .name(&entry_point)
            .build();

        CompiledShader {
            vk_module,
            _entry_point: entry_point,
            create_info,
            wg_size: ep.workgroup_size,
        }
    }

    fn create_descriptor_set_layout(
        &self,
        layout: &crate::ShaderDataLayout,
        visibility: crate::ShaderVisibility,
    ) -> super::DescriptorSetLayout {
        if visibility.is_empty() {
            return super::DescriptorSetLayout::default();
        }
        let stage_flags = map_shader_visibility(visibility);
        let mut vk_bindings = Vec::new();
        let mut binding_index = 1;
        let mut plain_data_size = 0u32;
        let mut template_entries = Vec::new();
        let mut template_offsets = Vec::with_capacity(layout.bindings.len());
        let mut update_offset = 0;
        for &(_, binding) in layout.bindings.iter() {
            let (descriptor_type, descriptor_size) = match binding {
                crate::ShaderBinding::Texture { .. } => (
                    vk::DescriptorType::SAMPLED_IMAGE,
                    mem::size_of::<vk::DescriptorImageInfo>(),
                ),
                crate::ShaderBinding::TextureStorage { .. } => (
                    vk::DescriptorType::STORAGE_IMAGE,
                    mem::size_of::<vk::DescriptorImageInfo>(),
                ),
                crate::ShaderBinding::Sampler { .. } => (
                    vk::DescriptorType::SAMPLER,
                    mem::size_of::<vk::DescriptorImageInfo>(),
                ),
                crate::ShaderBinding::Buffer { .. } => (
                    vk::DescriptorType::STORAGE_BUFFER,
                    mem::size_of::<vk::DescriptorBufferInfo>(),
                ),
                crate::ShaderBinding::Plain { ty, container } => {
                    let elem_size = match ty {
                        crate::PlainType::F32 | crate::PlainType::I32 | crate::PlainType::U32 => 4,
                    };
                    //TODO: alignment
                    let count = match container {
                        crate::PlainContainer::Scalar => 1,
                        crate::PlainContainer::Vector(size) => size as u32,
                        crate::PlainContainer::Matrix(rows, cols) => rows as u32 * cols as u32,
                    };
                    template_offsets.push(plain_data_size);
                    plain_data_size += elem_size * count;
                    continue;
                }
            };
            vk_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: binding_index,
                descriptor_type,
                descriptor_count: 1,
                stage_flags,
                p_immutable_samplers: ptr::null(),
            });
            template_entries.push(vk::DescriptorUpdateTemplateEntryKHR {
                dst_binding: binding_index,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type,
                offset: update_offset,
                stride: 0,
            });
            binding_index += 1;
            template_offsets.push(update_offset as u32);
            update_offset += descriptor_size;
        }

        let handle_section_size = update_offset as u32;
        if plain_data_size != 0 {
            vk_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                descriptor_count: plain_data_size,
                stage_flags,
                p_immutable_samplers: ptr::null(),
            });
            template_entries.push(vk::DescriptorUpdateTemplateEntryKHR {
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: plain_data_size,
                descriptor_type: vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                offset: update_offset,
                stride: 0,
            });
            for (template_offset, &(_, binding)) in
                template_offsets.iter_mut().zip(layout.bindings.iter())
            {
                if let crate::ShaderBinding::Plain { .. } = binding {
                    *template_offset += handle_section_size;
                }
            }
        }

        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&vk_bindings);
        let raw = unsafe {
            self.device
                .core
                .create_descriptor_set_layout(&set_layout_info, None)
                .unwrap()
        };

        let template_create_info = vk::DescriptorUpdateTemplateCreateInfo::builder()
            .descriptor_update_entries(&template_entries)
            .template_type(vk::DescriptorUpdateTemplateTypeKHR::DESCRIPTOR_SET)
            .descriptor_set_layout(raw);
        let update_template = unsafe {
            self.device
                .core
                .create_descriptor_update_template(&template_create_info, None)
                .unwrap()
        };

        super::DescriptorSetLayout {
            raw,
            update_template,
            template_size: handle_section_size + plain_data_size,
            template_offsets: template_offsets.into_boxed_slice(),
        }
    }

    fn create_pipeline_layout(
        &self,
        combined: &[(&crate::ShaderDataLayout, crate::ShaderVisibility)],
    ) -> super::PipelineLayout {
        let mut descriptor_set_layouts = Vec::with_capacity(combined.len());
        let mut vk_set_layouts = Vec::with_capacity(combined.len());
        for &(layout, visibility) in combined {
            let dsl = self.create_descriptor_set_layout(layout, visibility);
            vk_set_layouts.push(dsl.raw);
            descriptor_set_layouts.push(dsl);
        }

        let vk_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&vk_set_layouts);
        let raw = unsafe {
            self.device
                .core
                .create_pipeline_layout(&vk_info, None)
                .unwrap()
        };

        super::PipelineLayout {
            raw,
            descriptor_set_layouts,
        }
    }

    pub fn create_compute_pipeline(
        &self,
        desc: crate::ComputePipelineDesc,
    ) -> super::ComputePipeline {
        let combined =
            crate::merge_shader_layouts(&[(desc.compute.shader, crate::ShaderVisibility::COMPUTE)]);
        let layout = self.create_pipeline_layout(&combined);

        let options = spv::Options {
            lang_version: (1, 3),
            flags: self.naga_flags,
            binding_map: spv::BindingMap::default(),
            capabilities: None,
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        };

        let cs = self.load_shader(desc.compute, naga::ShaderStage::Compute, &options);

        let vk_infos = [{
            vk::ComputePipelineCreateInfo::builder()
                .layout(layout.raw)
                .stage(cs.create_info)
                .build()
        }];

        let mut raw_vec = unsafe {
            self.device
                .core
                .create_compute_pipelines(vk::PipelineCache::null(), &vk_infos, None)
                .unwrap()
        };
        let raw = raw_vec.pop().unwrap();

        unsafe { self.device.core.destroy_shader_module(cs.vk_module, None) };

        if !desc.name.is_empty() {
            self.set_object_name(vk::ObjectType::PIPELINE, raw, desc.name);
        }
        super::ComputePipeline {
            raw,
            layout,
            wg_size: cs.wg_size,
        }
    }

    pub fn create_render_pipeline(&self, desc: crate::RenderPipelineDesc) -> super::RenderPipeline {
        unimplemented!()
    }
}
