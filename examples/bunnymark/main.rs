#![allow(irrefutable_let_patterns)]

use bytemuck::{Pod, Zeroable};
use std::{ptr, time};

const BUNNY_SIZE: f32 = 0.15 * 256.0;
const GRAVITY: f32 = -9.8 * 100.0;
const MAX_VELOCITY: f32 = 750.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    mvp_transform: [[f32; 4]; 4],
    sprite_size: [f32; 2],
    pad: [f32; 2],
}

#[derive(blade::ShaderData)]
struct Params {
    globals: Globals,
    sprite_texture: blade::TextureView,
    sprite_sampler: blade::Sampler,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Locals {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
    pad: u32,
}

#[derive(blade::ShaderData)]
struct Sprite {
    locals: Locals,
}

struct Example {
    pipeline: blade::RenderPipeline,
    command_encoder: blade::CommandEncoder,
    prev_sync_point: Option<blade::SyncPoint>,
    texture: blade::Texture,
    view: blade::TextureView,
    sampler: blade::Sampler,
    window_size: winit::dpi::PhysicalSize<u32>,
    bunnies: Vec<Sprite>,
    rng: rand::rngs::ThreadRng,
    context: blade::Context,
}

impl Example {
    fn new(window: &winit::window::Window) -> Self {
        let window_size = window.inner_size();
        let context = unsafe {
            blade::Context::init_windowed(
                window,
                blade::ContextDesc {
                    validation: cfg!(debug_assertions),
                    capture: false,
                },
            )
            .unwrap()
        };

        let surface_format = context.resize(blade::SurfaceConfig {
            size: blade::Extent {
                width: window_size.width,
                height: window_size.height,
                depth: 1,
            },
            usage: blade::TextureUsage::TARGET,
            frame_count: 3,
        });

        let global_layout = <Params as blade::ShaderData>::layout();
        let local_layout = <Sprite as blade::ShaderData>::layout();
        let shader_source = std::fs::read_to_string("examples/bunnymark/shader.wgsl").unwrap();
        let shader = context.create_shader(blade::ShaderDesc {
            source: &shader_source,
        });

        let pipeline = context.create_render_pipeline(blade::RenderPipelineDesc {
            name: "main",
            data_layouts: &[&global_layout, &local_layout],
            vertex: shader.at("vs_main"),
            primitive: blade::PrimitiveState {
                topology: blade::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            fragment: shader.at("fs_main"),
            color_targets: &[blade::ColorTargetState {
                format: surface_format,
                blend: Some(blade::BlendState::ALPHA_BLENDING),
                write_mask: blade::ColorWrites::default(),
            }],
        });

        let extent = blade::Extent {
            width: 1,
            height: 1,
            depth: 1,
        };
        let texture = context.create_texture(blade::TextureDesc {
            name: "texutre",
            format: blade::TextureFormat::Rgba8Unorm,
            size: extent,
            dimension: blade::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: blade::TextureUsage::RESOURCE | blade::TextureUsage::COPY,
        });
        let view = context.create_texture_view(blade::TextureViewDesc {
            name: "view",
            texture,
            format: blade::TextureFormat::Rgba8Unorm,
            dimension: blade::ViewDimension::D2,
            subresources: &Default::default(),
        });

        let upload_buffer = context.create_buffer(blade::BufferDesc {
            name: "staging",
            size: (extent.width * extent.height) as u64 * 4,
            memory: blade::Memory::Upload,
        });
        let texture_data = vec![0xFFu8; 4];
        unsafe {
            ptr::copy_nonoverlapping(
                texture_data.as_ptr(),
                upload_buffer.data(),
                texture_data.len(),
            );
        }
        context.sync_buffer(upload_buffer);

        let sampler = context.create_sampler(blade::SamplerDesc {
            name: "main",
            ..Default::default()
        });

        let mut bunnies = Vec::new();
        bunnies.push(Sprite {
            locals: Locals {
                position: [-100.0, 100.0],
                velocity: [10.0, 0.0],
                color: 0xFFFFFFFF,
                pad: 0,
            },
        });

        let mut command_encoder = context.create_command_encoder(blade::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();
        command_encoder.init_texture(texture);
        if let mut transfer = command_encoder.transfer() {
            transfer.copy_buffer_to_texture(upload_buffer.into(), 4, texture.into(), extent);
        }
        let sync_point = context.submit(&mut command_encoder);
        context.wait_for(&sync_point, !0);

        context.destroy_buffer(upload_buffer);

        Self {
            pipeline,
            command_encoder,
            prev_sync_point: None,
            texture,
            view,
            sampler,
            window_size,
            bunnies,
            rng: rand::thread_rng(),
            context,
        }
    }

    fn increase(&mut self) {
        use rand::{Rng as _, RngCore as _};
        let spawn_count = 64 + self.bunnies.len() / 2;
        for _ in 0..spawn_count {
            let speed = self.rng.gen_range(-1.0..=1.0) * MAX_VELOCITY;
            self.bunnies.push(Sprite {
                locals: Locals {
                    position: [0.0, 0.5 * (self.window_size.height as f32)],
                    velocity: [speed, 0.0],
                    color: self.rng.next_u32(),
                    pad: 0,
                },
            });
        }
        println!("Population: {} bunnies", self.bunnies.len());
    }

    fn step(&mut self, delta: f32) {
        for bunny in self.bunnies.iter_mut() {
            let Locals {
                position: ref mut pos,
                velocity: ref mut vel,
                ..
            } = bunny.locals;

            pos[0] += vel[0] * delta;
            pos[1] += vel[1] * delta;
            vel[1] += GRAVITY * delta;
            if (vel[0] > 0.0 && pos[0] + 0.5 * BUNNY_SIZE > self.window_size.width as f32)
                || (vel[0] < 0.0 && pos[0] - 0.5 * BUNNY_SIZE < 0.0)
            {
                vel[0] *= -1.0;
            }
            if vel[1] < 0.0 && pos[1] < 0.5 * BUNNY_SIZE {
                vel[1] *= -1.0;
            }
        }
    }

    fn render(&mut self) {
        let frame = self.context.acquire_frame();

        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        if let mut pass = self.command_encoder.render(blade::RenderTargetSet {
            colors: &[blade::RenderTarget {
                view: frame.texture_view(),
                init_op: blade::InitOp::Clear(blade::TextureColor::TransparentBlack),
                finish_op: blade::FinishOp::Store,
            }],
            depth_stencil: None,
        }) {
            let mut rc = pass.with(&self.pipeline);
            rc.bind(
                0,
                &Params {
                    globals: Globals {
                        mvp_transform: [
                            [2.0 / self.window_size.width as f32, 0.0, 0.0, 0.0],
                            [0.0, 2.0 / self.window_size.height as f32, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [-1.0, -1.0, 0.0, 1.0],
                        ],
                        sprite_size: [BUNNY_SIZE; 2],
                        pad: [0.0; 2],
                    },
                    sprite_texture: self.view,
                    sprite_sampler: self.sampler,
                },
            );

            for sprite in self.bunnies.iter() {
                rc.bind(1, sprite);
                rc.draw(0, 4, 0, 1);
            }
        }
        self.command_encoder.present(frame);
        let sync_point = self.context.submit(&mut self.command_encoder);
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        self.prev_sync_point = Some(sync_point);
    }

    fn deinit(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        self.context.destroy_texture(self.texture);
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("blade-bunnymark")
        .build(&event_loop)
        .unwrap();

    let mut example = Example::new(&window);
    let mut last_snapshot = time::Instant::now();
    let mut frame_count = 0;

    event_loop.run(move |event, _, control_flow| {
        let _ = &window; // force ownership by the closure
        *control_flow = winit::event_loop::ControlFlow::Poll;
        match event {
            winit::event::Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(key_code),
                            state: winit::event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => match key_code {
                    winit::event::VirtualKeyCode::Escape => {
                        *control_flow = winit::event_loop::ControlFlow::Exit;
                    }
                    winit::event::VirtualKeyCode::Space => {
                        example.increase();
                    }
                    _ => {}
                },
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                _ => {}
            },
            winit::event::Event::RedrawRequested(_) => {
                frame_count += 1;
                if frame_count == 100 {
                    let accum_time = last_snapshot.elapsed().as_secs_f32();
                    println!(
                        "Avg frame time {}ms",
                        accum_time * 1000.0 / frame_count as f32
                    );
                    last_snapshot = time::Instant::now();
                    frame_count = 0;
                }
                example.step(0.01);
                example.render();
            }
            winit::event::Event::LoopDestroyed => {
                example.deinit();
            }
            _ => {}
        }
    })
}
