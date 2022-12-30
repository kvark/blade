use glow::HasContext as _;
use naga::back::glsl;

impl super::Context {
    unsafe fn create_pipeline(
        &self,
        shaders: &[crate::ShaderFunction],
        group_layouts: &[&crate::ShaderDataLayout],
        name: &str,
    ) -> super::PipelineInner {
        let gl = self.lock();

        let program = gl.create_program().unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        if !name.is_empty() && gl.supports_debug() {
            gl.object_label(glow::PROGRAM, std::mem::transmute(program), Some(name));
        }

        let naga_options = glsl::Options {
            version: glsl::Version::Embedded {
                version: 300,
                is_webgl: cfg!(target_arch = "wasm32"),
            },
            ..Default::default()
        };

        let mut baked_shaders = Vec::with_capacity(shaders.len());

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
            let reflection = writer.write().unwrap();

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

            gl.attach_shader(program, shader);
            baked_shaders.push((shader, reflection));
        }

        gl.link_program(program);
        log::info!("\tLinked program {:?}", program);

        let linked_ok = gl.get_program_link_status(program);
        let msg = gl.get_program_info_log(program);
        assert!(linked_ok, "Link: {}", msg);
        gl.use_program(Some(program));

        //type NameList = Vec<String>;
        //type BindingNames = Vec<NameList>;
        let mut bind_group_infos = group_layouts
            .iter()
            .map(|layout| super::BindGroupInfo {
                targets: vec![Vec::new(); layout.bindings.len()].into_boxed_slice(),
            })
            .collect::<Box<[_]>>();

        let force_uniform_block_assignment = true;
        let mut variables_to_bind = Vec::new();
        for (sf, &(shader, ref reflection)) in shaders.iter().zip(baked_shaders.iter()) {
            for (glsl_name, mapping) in reflection.texture_mapping.iter() {
                variables_to_bind.push((glsl_name, mapping.texture));
                if let Some(handle) = mapping.sampler {
                    variables_to_bind.push((glsl_name, handle));
                }
            }
            for (&handle, glsl_name) in reflection.uniforms.iter() {
                variables_to_bind.push((glsl_name, handle));
            }

            for (glsl_name, var_handle) in variables_to_bind.drain(..) {
                let var = &sf.shader.module.global_variables[var_handle];
                let var_name = var.name.as_ref().unwrap().as_str();
                let (group_index, binding_index) = group_layouts
                    .iter()
                    .enumerate()
                    .find_map(|(group_index, layout)| {
                        layout
                            .bindings
                            .iter()
                            .position(|&(name, _)| name == var_name)
                            .map(|binding_index| (group_index, binding_index))
                    })
                    .unwrap_or_else(|| {
                        panic!("Shader variable {} is not found in the bindings", var_name)
                    });

                let targets = &mut bind_group_infos[group_index].targets[binding_index];
                match group_layouts[group_index].bindings[binding_index].1 {
                    crate::ShaderBinding::Texture | crate::ShaderBinding::Sampler => {
                        if let Some(ref location) = gl.get_uniform_location(program, glsl_name) {
                            let mut slots = [0i32];
                            gl.get_uniform_i32(program, location, &mut slots);
                            targets.push(slots[0] as u32);
                        }
                    }
                    crate::ShaderBinding::Buffer => {
                        if let Some(index) = gl.get_shader_storage_block_index(program, glsl_name) {
                            let params = gl.get_program_resource_i32(
                                program,
                                glow::SHADER_STORAGE_BLOCK,
                                index,
                                &[glow::BUFFER_BINDING],
                            );
                            targets.push(params[0] as u32);
                        }
                    }
                    crate::ShaderBinding::Plain { size } => {
                        if let Some(index) = gl.get_uniform_block_index(program, glsl_name) {
                            let expected_size = gl.get_active_uniform_block_parameter_i32(
                                program,
                                index,
                                glow::UNIFORM_BLOCK_DATA_SIZE,
                            ) as u32;
                            let rounded_up_size = super::round_up_uniform_size(size);
                            assert!(
                                expected_size <= rounded_up_size,
                                "Shader expects block[{}] size {}, but data has size of {} (rounded up to {})",
                                index,
                                expected_size,
                                size,
                                rounded_up_size,
                            );
                            let slot = if force_uniform_block_assignment {
                                gl.uniform_block_binding(program, index, index);
                                index
                            } else {
                                gl.get_active_uniform_block_parameter_i32(
                                    program,
                                    index,
                                    glow::UNIFORM_BLOCK_BINDING,
                                ) as u32
                            };
                            targets.push(slot);
                        }
                    }
                }
            }

            gl.delete_shader(shader);
        }
        gl.use_program(None);

        super::PipelineInner {
            program,
            bind_group_infos,
        }
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

    pub fn create_render_pipeline(&self, desc: crate::RenderPipelineDesc) -> super::RenderPipeline {
        let inner = unsafe {
            self.create_pipeline(&[desc.vertex, desc.fragment], desc.data_layouts, desc.name)
        };
        super::RenderPipeline {
            inner,
            topology: desc.primitive.topology,
        }
    }
}
