use blade_graphics as gpu;

use crate::{CameraParams, ColorConfig, EmitterShape, ParticleEffect};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct EmitParams {
    origin: [f32; 3],
    emitter_radius: f32,
    direction: [f32; 3],
    cone_half_angle_cos: f32,
    colors: [u32; 4],
    color_count: u32,
    emit_count: u32,
    life_min: f32,
    life_max: f32,
    speed_min: f32,
    speed_max: f32,
    scale_min: f32,
    scale_max: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct UpdateParams {
    time_delta: f32,
}

#[derive(blade_macros::ShaderData)]
struct MainData {
    particles: gpu::BufferPiece,
    free_list: gpu::BufferPiece,
}

#[derive(blade_macros::ShaderData)]
struct EmitData {
    emit_params: EmitParams,
}

#[derive(blade_macros::ShaderData)]
struct UpdateData {
    update_params: UpdateParams,
}

#[derive(blade_macros::ShaderData)]
struct DrawData {
    draw_particles: gpu::BufferPiece,
    camera: CameraParams,
}

pub struct PipelineDesc<'a> {
    pub name: &'a str,
    pub draw_format: gpu::TextureFormat,
    pub sample_count: u32,
}

/// Shared GPU pipelines for particle simulation and rendering.
/// Create once, then spawn many `ParticleSystem` instances from it.
pub struct ParticlePipeline {
    reset_pipeline: gpu::ComputePipeline,
    emit_pipeline: gpu::ComputePipeline,
    update_pipeline: gpu::ComputePipeline,
    draw_pipeline: gpu::RenderPipeline,
    particle_size: u32,
}

impl ParticlePipeline {
    pub fn new(context: &gpu::Context, desc: PipelineDesc) -> Self {
        let source = include_str!("particle.wgsl");
        let shader = context.create_shader(gpu::ShaderDesc {
            source,
            naga_module: None,
        });

        let particle_size = shader.get_struct_size("Particle");

        let main_layout = <MainData as gpu::ShaderData>::layout();
        let emit_layout = <EmitData as gpu::ShaderData>::layout();
        let update_layout = <UpdateData as gpu::ShaderData>::layout();
        let draw_layout = <DrawData as gpu::ShaderData>::layout();

        let reset_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: &format!("{} - reset", desc.name),
            data_layouts: &[&main_layout],
            compute: shader.at("reset"),
        });
        let emit_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: &format!("{} - emit", desc.name),
            data_layouts: &[&main_layout, &emit_layout],
            compute: shader.at("emit"),
        });
        let update_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: &format!("{} - update", desc.name),
            data_layouts: &[&main_layout, &update_layout],
            compute: shader.at("update"),
        });
        let draw_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: &format!("{} - draw", desc.name),
            data_layouts: &[&draw_layout],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            vertex: shader.at("draw_vs"),
            vertex_fetches: &[],
            fragment: Some(shader.at("draw_fs")),
            color_targets: &[gpu::ColorTargetState {
                format: desc.draw_format,
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::default(),
            }],
            depth_stencil: None,
            multisample_state: gpu::MultisampleState {
                sample_count: desc.sample_count,
                ..Default::default()
            },
        });

        Self {
            reset_pipeline,
            emit_pipeline,
            update_pipeline,
            draw_pipeline,
            particle_size,
        }
    }

    pub fn destroy(&mut self, context: &gpu::Context) {
        context.destroy_compute_pipeline(&mut self.reset_pipeline);
        context.destroy_compute_pipeline(&mut self.emit_pipeline);
        context.destroy_compute_pipeline(&mut self.update_pipeline);
        context.destroy_render_pipeline(&mut self.draw_pipeline);
    }

    /// Create a new particle system using this pipeline.
    pub fn create_system(
        &self,
        context: &gpu::Context,
        name: &str,
        effect: &ParticleEffect,
    ) -> ParticleSystem {
        let wg_width = self.reset_pipeline.get_workgroup_size()[0] as usize;
        let capacity = ((effect.capacity as usize - 1) | (wg_width - 1)) + 1;
        let particle_buf = context.create_buffer(gpu::BufferDesc {
            name,
            size: capacity as u64 * self.particle_size as u64,
            memory: gpu::Memory::Device,
        });
        let free_list_buf = context.create_buffer(gpu::BufferDesc {
            name: &format!("{} - free list", name),
            size: 4 + capacity as u64 * 4,
            memory: gpu::Memory::Device,
        });

        ParticleSystem {
            capacity,
            effect: effect.clone(),
            origin: [0.0; 3],
            axis: [0.0, 1.0, 0.0],
            particle_buf,
            free_list_buf,
            emit_accumulator: 0.0,
            pending_bursts: Vec::new(),
            needs_reset: true,
        }
    }
}

struct PendingBurst {
    count: u32,
    position: [f32; 3],
}

/// A single particle system instance with its own buffers and emitter state.
/// Uses a shared `ParticlePipeline` for GPU dispatch.
pub struct ParticleSystem {
    capacity: usize,
    pub effect: ParticleEffect,
    /// Emitter position in world space.
    pub origin: [f32; 3],
    /// Emitter facing axis (unit vector). Newly emitted particles spray
    /// in a cone around this axis. Already-emitted particles are unaffected.
    /// Default is [0, 1, 0] (upward).
    pub axis: [f32; 3],
    particle_buf: gpu::Buffer,
    free_list_buf: gpu::Buffer,
    emit_accumulator: f32,
    pending_bursts: Vec<PendingBurst>,
    needs_reset: bool,
}

impl ParticleSystem {
    pub fn destroy(&mut self, context: &gpu::Context) {
        context.destroy_buffer(self.particle_buf);
        context.destroy_buffer(self.free_list_buf);
    }

    fn main_data(&self) -> MainData {
        MainData {
            particles: self.particle_buf.into(),
            free_list: self.free_list_buf.into(),
        }
    }

    fn make_emit_params(&self, count: u32, position: [f32; 3]) -> EmitParams {
        let emitter_radius = match self.effect.emitter.shape {
            EmitterShape::Point => 0.0,
            EmitterShape::Sphere { radius } => radius,
        };

        let (colors, color_count) = match &self.effect.particle.color {
            ColorConfig::Solid(c) => {
                let packed = pack_color(*c);
                ([packed, packed, packed, packed], 1u32)
            }
            ColorConfig::Palette(palette) => {
                let mut colors = [0u32; 4];
                let count = palette.len().min(4);
                for i in 0..count {
                    colors[i] = pack_color(palette[i]);
                }
                for i in count..4 {
                    colors[i] = colors[0];
                }
                (colors, count as u32)
            }
        };

        EmitParams {
            origin: position,
            emitter_radius,
            direction: self.axis,
            cone_half_angle_cos: self.effect.emitter.cone_angle.cos(),
            colors,
            color_count,
            emit_count: count,
            life_min: self.effect.particle.life[0],
            life_max: self.effect.particle.life[1],
            speed_min: self.effect.particle.speed[0],
            speed_max: self.effect.particle.speed[1],
            scale_min: self.effect.particle.scale[0],
            scale_max: self.effect.particle.scale[1],
        }
    }

    /// Queue a burst of particles at a world position.
    pub fn burst(&mut self, count: u32, position: [f32; 3]) {
        self.pending_bursts.push(PendingBurst { count, position });
    }

    /// Update particle simulation and emit new particles.
    pub fn update(
        &mut self,
        pipeline: &ParticlePipeline,
        encoder: &mut gpu::CommandEncoder,
        dt: f32,
    ) {
        if self.needs_reset {
            let mut pass = encoder.compute("particle reset");
            let mut pc = pass.with(&pipeline.reset_pipeline);
            pc.bind(0, &self.main_data());
            let group_size = pipeline.reset_pipeline.get_workgroup_size();
            let group_count = (self.capacity as u32 + group_size[0] - 1) / group_size[0];
            pc.dispatch([group_count, 1, 1]);
            self.needs_reset = false;
        }

        let main_data = self.main_data();

        // Update existing particles
        {
            let mut pass = encoder.compute("particle update");
            let mut pc = pass.with(&pipeline.update_pipeline);
            pc.bind(0, &main_data);
            pc.bind(
                1,
                &UpdateData {
                    update_params: UpdateParams { time_delta: dt },
                },
            );
            let group_size = pipeline.update_pipeline.get_workgroup_size();
            let group_count = self.capacity as u32 / group_size[0];
            pc.dispatch([group_count, 1, 1]);
        }

        // Continuous emission
        if self.effect.emitter.rate > 0.0 {
            self.emit_accumulator += self.effect.emitter.rate * dt;
            let emit_count = self.emit_accumulator as u32;
            if emit_count > 0 {
                self.emit_accumulator -= emit_count as f32;
                let params = self.make_emit_params(emit_count, self.origin);
                let wg_size = pipeline.emit_pipeline.get_workgroup_size()[0];
                let groups = (emit_count + wg_size - 1) / wg_size;
                let mut pass = encoder.compute("particle emit continuous");
                let mut pc = pass.with(&pipeline.emit_pipeline);
                pc.bind(0, &main_data);
                pc.bind(
                    1,
                    &EmitData {
                        emit_params: params,
                    },
                );
                pc.dispatch([groups, 1, 1]);
            }
        }

        // Process burst emissions
        let bursts: Vec<_> = self.pending_bursts.drain(..).collect();
        for burst in bursts {
            let params = self.make_emit_params(burst.count, burst.position);
            let wg_size = pipeline.emit_pipeline.get_workgroup_size()[0];
            let groups = (burst.count + wg_size - 1) / wg_size;
            let mut pass = encoder.compute("particle emit burst");
            let mut pc = pass.with(&pipeline.emit_pipeline);
            pc.bind(0, &main_data);
            pc.bind(
                1,
                &EmitData {
                    emit_params: params,
                },
            );
            pc.dispatch([groups, 1, 1]);
        }
    }

    /// Draw particles with 3D camera projection.
    pub fn draw(
        &self,
        pipeline: &ParticlePipeline,
        pass: &mut gpu::RenderCommandEncoder,
        camera: &CameraParams,
    ) {
        let mut pc = pass.with(&pipeline.draw_pipeline);
        pc.bind(
            0,
            &DrawData {
                draw_particles: self.particle_buf.into(),
                camera: *camera,
            },
        );
        pc.draw(0, 4, 0, self.capacity as u32);
    }
}

fn pack_color(c: [u8; 4]) -> u32 {
    (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16) | ((c[3] as u32) << 24)
}
