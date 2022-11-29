#![allow(irrefutable_let_patterns)]

use std::{ptr, time};

const BUNNY_SIZE: f32 = 0.15 * 256.0;
const GRAVITY: f32 = -9.8 * 100.0;
const MAX_VELOCITY: f32 = 750.0;

struct Globals {
    mvp_transform: [[f32; 4]; 4],
    sprite_size: [f32; 2],
    sprite_texture: blade::TextureView,
    sprite_sampler: blade::Sampler,
}

struct Locals {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
}

//TEMP
impl blade::ShaderData for Globals {
    fn layout() -> blade::ShaderDataLayout {
        blade::ShaderDataLayout {
            bindings: vec![
                (
                    "mvp_transform".to_string(),
                    blade::ShaderBinding::Plain {
                        ty: blade::PlainType::F32,
                        container: blade::PlainContainer::Matrix(
                            blade::VectorSize::Quad,
                            blade::VectorSize::Quad,
                        ),
                    },
                ),
                (
                    "sprite_size".to_string(),
                    blade::ShaderBinding::Plain {
                        ty: blade::PlainType::F32,
                        container: blade::PlainContainer::Vector(blade::VectorSize::Bi),
                    },
                ),
                (
                    "sprite_texture".to_string(),
                    blade::ShaderBinding::Texture {
                        dimension: blade::TextureViewDimension::D2,
                    },
                ),
                (
                    "sprite_sampler".to_string(),
                    blade::ShaderBinding::Sampler { comparison: false },
                ),
            ],
        }
    }
    fn fill<E: blade::ShaderDataEncoder>(&self, mut encoder: E) {
        encoder.set_plain(0, self.mvp_transform);
        encoder.set_plain(1, self.sprite_size);
        encoder.set_texture(2, self.sprite_texture);
        encoder.set_sampler(3, self.sprite_sampler);
    }
}

impl blade::ShaderData for Locals {
    fn layout() -> blade::ShaderDataLayout {
        blade::ShaderDataLayout {
            bindings: vec![
                (
                    "position".to_string(),
                    blade::ShaderBinding::Plain {
                        ty: blade::PlainType::F32,
                        container: blade::PlainContainer::Vector(blade::VectorSize::Bi),
                    },
                ),
                (
                    "velocity".to_string(),
                    blade::ShaderBinding::Plain {
                        ty: blade::PlainType::F32,
                        container: blade::PlainContainer::Vector(blade::VectorSize::Bi),
                    },
                ),
                (
                    "color".to_string(),
                    blade::ShaderBinding::Plain {
                        ty: blade::PlainType::U32,
                        container: blade::PlainContainer::Scalar,
                    },
                ),
            ],
        }
    }
    fn fill<E: blade::ShaderDataEncoder>(&self, mut encoder: E) {
        encoder.set_plain(0, self.position);
        encoder.set_plain(1, self.velocity);
        encoder.set_plain(2, self.color);
    }
}

struct Example {
    pipeline: blade::RenderPipeline,
    command_encoder: blade::CommandEncoder,
    prev_sync_point: Option<blade::SyncPoint>,
    _texture: blade::Texture,
    view: blade::TextureView,
    sampler: blade::Sampler,
    window_size: winit::dpi::PhysicalSize<u32>,
    bunnies: Vec<Locals>,
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
                    validation: true,
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
            frame_count: 2,
        });

        let global_layout = <Globals as blade::ShaderData>::layout();
        let local_layout = <Locals as blade::ShaderData>::layout();
        let shader_source = std::fs::read_to_string("examples/bunnymark.wgsl").unwrap();
        let shader = context.create_shader(blade::ShaderDesc {
            source: &shader_source,
            data_layouts: &[Some(&global_layout), Some(&local_layout)],
        });

        let pipeline = context.create_render_pipeline(blade::RenderPipelineDesc {
            name: "main",
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
            dimension: blade::TextureViewDimension::D2,
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

        let sampler = context.create_sampler(blade::SamplerDesc {
            name: "main",
            ..Default::default()
        });

        let mut bunnies = Vec::new();
        bunnies.push(Locals {
            position: [-100.0, 100.0],
            velocity: [10.0, 0.0],
            color: 0xFFFFFFFF,
        });

        let mut command_encoder = context.create_command_encoder(blade::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();
        if let mut transfer = command_encoder.transfer() {
            transfer.copy_buffer_to_texture(upload_buffer.into(), 4, texture.into(), extent);
        }
        context.submit(&mut command_encoder);

        Self {
            pipeline,
            command_encoder,
            prev_sync_point: None,
            _texture: texture,
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
            self.bunnies.push(Locals {
                position: [0.0, 0.5 * (self.window_size.height as f32)],
                velocity: [speed, 0.0],
                color: self.rng.next_u32(),
            });
        }
        println!("Population: {} bunnies", self.bunnies.len());
    }

    fn step(&mut self, delta: f32) {
        for bunny in self.bunnies.iter_mut() {
            bunny.position[0] += bunny.velocity[0] * delta;
            bunny.position[1] += bunny.velocity[1] * delta;
            bunny.velocity[1] += GRAVITY * delta;
            if (bunny.velocity[0] > 0.0
                && bunny.position[0] + 0.5 * BUNNY_SIZE > self.window_size.width as f32)
                || (bunny.velocity[0] < 0.0 && bunny.position[0] - 0.5 * BUNNY_SIZE < 0.0)
            {
                bunny.velocity[0] *= -1.0;
            }
            if bunny.velocity[1] < 0.0 && bunny.position[1] < 0.5 * BUNNY_SIZE {
                bunny.velocity[1] *= -1.0;
            }
        }
    }

    fn render(&mut self) {
        let frame = self.context.acquire_frame();

        self.command_encoder.start();
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
                &Globals {
                    mvp_transform: [
                        [2.0 / self.window_size.width as f32, 0.0, 0.0, 0.0],
                        [0.0, 2.0 / self.window_size.height as f32, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [-1.0, -1.0, 0.0, 1.0],
                    ],
                    sprite_size: [BUNNY_SIZE; 2],
                    sprite_texture: self.view,
                    sprite_sampler: self.sampler,
                },
            );

            for local in self.bunnies.iter() {
                rc.bind(1, local);
                rc.draw(0, 4, 0, 1);
            }
        }

        let sync_point = self.context.submit(&mut self.command_encoder);
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(sp, !0);
        }
        self.context.present(frame);
        self.prev_sync_point = Some(sync_point);
    }

    fn deinit(&mut self) {
        //TODO
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("hal-bunnymark")
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
