use blade::RenderCommandEncoder;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Parameters {
    pub life: f32,
    pub velocity: f32,
    pub scale: f32,
}

pub struct System {
    capacity: usize,
    pub params: Parameters,
    particle_buf: blade::Buffer,
    free_list_buf: blade::Buffer,
    reset_pipeline: blade::ComputePipeline,
    emit_pipeline: blade::ComputePipeline,
    update_pipeline: blade::ComputePipeline,
    draw_pipeline: blade::RenderPipeline,
}

pub struct SystemDesc<'a> {
    pub name: &'a str,
    pub capacity: usize,
    pub draw_format: blade::TextureFormat,
}

#[derive(blade::ShaderData)]
struct MainData {
    particles: blade::BufferPiece,
    free_list: blade::BufferPiece,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct UpdateParams {
    time_delta: f32,
}

#[derive(blade::ShaderData)]
struct UpdateData {
    update_params: UpdateParams,
    parameters: Parameters,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Transform2D {
    pos: [f32; 2],
    scale: f32,
    rot: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct DrawParams {
    t_emitter: Transform2D,
    screen_center: [f32; 2],
    screen_extent: [f32; 2],
}

#[derive(blade::ShaderData)]
struct DrawData {
    particles: blade::BufferPiece,
    draw_params: DrawParams,
}

impl System {
    pub fn new(context: &blade::Context, desc: SystemDesc) -> Self {
        let source = std::fs::read_to_string("examples/particle/particle.wgsl").unwrap();
        let shader = context.create_shader(blade::ShaderDesc { source: &source });
        let particle_size = shader.get_struct_size("Particle");

        let main_layout = <MainData as blade::ShaderData>::layout();
        let update_layout = <UpdateData as blade::ShaderData>::layout();
        let draw_layout = <DrawData as blade::ShaderData>::layout();

        let reset_pipeline = context.create_compute_pipeline(blade::ComputePipelineDesc {
            name: &format!("{} - reset", desc.name),
            data_layouts: &[&main_layout],
            compute: shader.at("reset"),
        });
        let emit_pipeline = context.create_compute_pipeline(blade::ComputePipelineDesc {
            name: &format!("{} - emit", desc.name),
            data_layouts: &[&main_layout, &update_layout],
            compute: shader.at("emit"),
        });
        let update_pipeline = context.create_compute_pipeline(blade::ComputePipelineDesc {
            name: &format!("{} - update", desc.name),
            data_layouts: &[&main_layout, &update_layout],
            compute: shader.at("update"),
        });
        let draw_pipeline = context.create_render_pipeline(blade::RenderPipelineDesc {
            name: &format!("{} - draw", desc.name),
            data_layouts: &[&draw_layout],
            primitive: blade::PrimitiveState {
                topology: blade::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            vertex: shader.at("draw_vs"),
            fragment: shader.at("draw_fs"),
            color_targets: &[blade::ColorTargetState {
                format: desc.draw_format,
                blend: Some(blade::BlendState::ALPHA_BLENDING),
                write_mask: blade::ColorWrites::default(),
            }],
            depth_stencil: None,
        });

        let wg_width = reset_pipeline.get_workgroup_size()[0] as usize;
        let capacity = ((desc.capacity - 1) | (wg_width - 1)) + 1;
        let particle_buf = context.create_buffer(blade::BufferDesc {
            name: desc.name,
            size: capacity as u64 * particle_size as u64,
            memory: blade::Memory::Device,
        });
        let free_list_buf = context.create_buffer(blade::BufferDesc {
            name: &format!("{} - free list", desc.name),
            size: 4 + capacity as u64 * 4,
            memory: blade::Memory::Device,
        });

        Self {
            capacity,
            params: Parameters {
                life: 5.0,
                velocity: 250.0,
                scale: 15.0,
            },
            particle_buf,
            free_list_buf,
            reset_pipeline,
            emit_pipeline,
            update_pipeline,
            draw_pipeline,
        }
    }

    pub fn delete(self, context: &blade::Context) {
        context.destroy_buffer(self.particle_buf);
        context.destroy_buffer(self.free_list_buf);
    }

    fn main_data(&self) -> MainData {
        MainData {
            particles: self.particle_buf.into(),
            free_list: self.free_list_buf.into(),
        }
    }

    pub fn reset(&self, encoder: &mut blade::CommandEncoder) {
        let mut pass = encoder.compute();
        let mut pc = pass.with(&self.reset_pipeline);
        pc.bind(0, &self.main_data());
        let group_size = self.reset_pipeline.get_workgroup_size();
        let group_count = (self.capacity as u32 + group_size[0] - 1) / group_size[0];
        pc.dispatch([group_count, 1, 1]);
    }

    pub fn update(&self, encoder: &mut blade::CommandEncoder) {
        let main_data = self.main_data();
        if let mut pass = encoder.compute() {
            let mut pc = pass.with(&self.update_pipeline);
            pc.bind(0, &main_data);
            pc.bind(
                1,
                &UpdateData {
                    update_params: UpdateParams { time_delta: 0.01 },
                    parameters: self.params,
                },
            );
            let group_size = self.update_pipeline.get_workgroup_size();
            let group_count = self.capacity as u32 / group_size[0];
            pc.dispatch([group_count, 1, 1]);
        }
        // new pass because both pipelines use the free list
        if let mut pass = encoder.compute() {
            let mut pc = pass.with(&self.emit_pipeline);
            pc.bind(0, &main_data);
            pc.dispatch([1, 1, 1]);
        }
    }

    pub fn draw(&self, pass: &mut RenderCommandEncoder) {
        let mut pc = pass.with(&self.draw_pipeline);
        pc.bind(
            0,
            &DrawData {
                particles: self.particle_buf.into(),
                draw_params: DrawParams {
                    t_emitter: Transform2D {
                        pos: [0.0; 2],
                        rot: 0.0,
                        scale: 1.0,
                    },
                    screen_center: [0.0; 2],
                    screen_extent: [1000.0; 2],
                },
            },
        );
        pc.draw(0, 4, 0, self.capacity as u32);
    }

    pub fn add_gui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.params.life, 0.1..=10.0).text("life"));
        ui.add(egui::Slider::new(&mut self.params.velocity, 1.0..=500.0).text("velocity"));
        ui.add(egui::Slider::new(&mut self.params.scale, 0.1..=50.0).text("scale"));
    }
}
