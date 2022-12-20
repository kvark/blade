use glow::HasContext as _;
use naga::back::glsl;
use std::mem;

impl super::Context {
    unsafe fn create_pipeline(
        &self,
        shaders: &[crate::ShaderFunction],
        group_layouts: &[&crate::ShaderDataLayout],
        name: &str,
    ) -> super::PipelineInner {
        let gl = self.lock();

        let program = gl.create_program().unwrap();
        if !name.is_empty() && gl.supports_debug() {
            gl.object_label(glow::PROGRAM, mem::transmute(program), Some(name));
        }

        let mut naga_options = glsl::Options::default();

        let mut shaders_to_delete = Vec::new();

        for &sf in shaders {
            let ep_index = sf.entry_point_index();
            let ep = &sf.shader.module.entry_points[ep_index];
            let pipeline_options = glsl::PipelineOptions {
                shader_stage: ep.stage,
                entry_point: sf.entry_point.to_string(),
                multiview: None,
            };
            let mut source = String::new();
            let mut writer = glsl::Writer::new(
                &mut source,
                &sf.shader.module,
                &sf.shader.info,
                &naga_options,
                &pipeline_options,
                Default::default(),
            )
            .unwrap();

            log::debug!(
                "Naga generated shader for entry point '{}' and stage {:?}\n{}",
                sf.entry_point,
                ep.stage,
                &source
            );

            let target = match ep.stage {
                naga::ShaderStage::Vertex => glow::VERTEX_SHADER,
                naga::ShaderStage::Fragment => glow::FRAGMENT_SHADER,
                naga::ShaderStage::Compute => glow::COMPUTE_SHADER,
            };
            let shader = gl.create_shader(target).unwrap();
            gl.shader_source(shader, &source);
            gl.compile_shader(shader);

            let compiled_ok = gl.get_shader_compile_status(shader);
            let msg = gl.get_shader_info_log(shader);
            assert!(compiled_ok, "Compile: {}", msg);

            shaders_to_delete.push(shader);
        }

        for &shader in shaders_to_delete.iter() {
            gl.attach_shader(program, shader);
        }
        gl.link_program(program);

        for shader in shaders_to_delete {
            gl.delete_shader(shader);
        }

        log::info!("\tLinked program {:?}", program);

        let linked_ok = gl.get_program_link_status(program);
        let msg = gl.get_program_info_log(program);
        assert!(linked_ok, "Link: {}", msg);

        super::PipelineInner { program }
    }

    pub fn create_compute_pipeline(
        &self,
        desc: crate::ComputePipelineDesc,
    ) -> super::ComputePipeline {
        let wg_size = desc.compute.shader.module.entry_points[desc.compute.entry_point_index()]
            .workgroup_size;
        let inner = unsafe { self.create_pipeline(&[desc.compute], desc.data_layouts, desc.name) };
        super::ComputePipeline { inner, wg_size }
    }
}
