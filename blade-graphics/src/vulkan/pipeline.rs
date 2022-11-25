use ash::vk;
use naga::back::spv;
use std::{ffi, ptr};

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

        let vk_module = unsafe { self.device.create_shader_module(&vk_info, None).unwrap() };

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
        let stage_flags = map_shader_visibility(visibility);
        let mut vk_bindings = Vec::new();
        let mut binding_index = 1;
        for &(_, binding) in layout.bindings.iter() {
            let descriptor_type = match binding {
                crate::ShaderBinding::Texture { .. } => vk::DescriptorType::SAMPLED_IMAGE,
                crate::ShaderBinding::TextureStorage { .. } => vk::DescriptorType::STORAGE_IMAGE,
                crate::ShaderBinding::Sampler { .. } => vk::DescriptorType::SAMPLER,
                crate::ShaderBinding::Buffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
                crate::ShaderBinding::Plain { .. } => continue,
            };
            vk_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: binding_index,
                descriptor_type,
                descriptor_count: 1,
                stage_flags,
                p_immutable_samplers: ptr::null(),
            });
            binding_index += 1;
        }

        let vk_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&vk_bindings);
        let raw = unsafe {
            self.device
                .create_descriptor_set_layout(&vk_info, None)
                .unwrap()
        };

        super::DescriptorSetLayout {
            raw,
            update_template: unimplemented!(),
        }
    }

    fn create_pipeline_layout(
        &self,
        combined: &[(Option<&crate::ShaderDataLayout>, crate::ShaderVisibility)],
    ) -> super::PipelineLayout {
        let mut descriptor_set_layouts = Vec::new();
        for &(layout_maybe, visibility) in combined {
            let dsl =
                layout_maybe.map(|layout| self.create_descriptor_set_layout(layout, visibility));
            descriptor_set_layouts.push(dsl);
        }

        let vk_set_layouts = descriptor_set_layouts
            .iter()
            .map(|dsl| match dsl {
                Some(ref dsl) => dsl.raw,
                None => vk::DescriptorSetLayout::null(),
            })
            .collect::<Vec<_>>();
        let vk_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&vk_set_layouts);
        let raw = unsafe { self.device.create_pipeline_layout(&vk_info, None).unwrap() };

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
            lang_version: (1, 4),
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
                .create_compute_pipelines(vk::PipelineCache::null(), &vk_infos, None)
        }
        .unwrap();
        let raw = raw_vec.pop().unwrap();

        unsafe { self.device.destroy_shader_module(cs.vk_module, None) };

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
