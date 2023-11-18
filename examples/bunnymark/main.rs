#![allow(irrefutable_let_patterns)]

use blade::graphics as gpu;
use bytemuck::{Pod, Zeroable};
use std::ptr;

const BUNNY_SIZE: f32 = 0.15 * 256.0;
const GRAVITY: f32 = -9.8 * 100.0;
const MAX_VELOCITY: i32 = 750;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    mvp_transform: [[f32; 4]; 4],
    sprite_size: [f32; 2],
    pad: [f32; 2],
}

#[derive(blade::macros::ShaderData)]
struct Params {
    globals: Globals,
    sprite_texture: gpu::TextureView,
    sprite_sampler: gpu::Sampler,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Locals {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
    pad: u32,
}

#[derive(blade::macros::ShaderData)]
struct Sprite {
    locals: Locals,
}

struct Example {
    pipeline: gpu::RenderPipeline,
    command_encoder: Option<gpu::CommandEncoder>,
    prev_sync_point: Option<gpu::SyncPoint>,
    texture: gpu::Texture,
    view: gpu::TextureView,
    sampler: gpu::Sampler,
    window_size: winit::dpi::PhysicalSize<u32>,
    bunnies: Vec<Sprite>,
    rng: nanorand::WyRand,
    context: gpu::Context,
}

impl Example {
    fn new(window: &winit::window::Window) -> Self {
        let window_size = window.inner_size();
        let context = unsafe {
            gpu::Context::init_windowed(
                window,
                gpu::ContextDesc {
                    validation: cfg!(debug_assertions),
                    capture: false,
                },
            )
            .unwrap()
        };

        let surface_format = context.resize(gpu::SurfaceConfig {
            size: gpu::Extent {
                width: window_size.width,
                height: window_size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            frame_count: 3,
        });

        let global_layout = <Params as gpu::ShaderData>::layout();
        let local_layout = <Sprite as gpu::ShaderData>::layout();
        #[cfg(target_arch = "wasm32")]
        let shader_source = include_str!("shader.wgsl");
        #[cfg(not(target_arch = "wasm32"))]
        let shader_source = std::fs::read_to_string("examples/bunnymark/shader.wgsl").unwrap();
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &shader_source,
        });

        let pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "main",
            data_layouts: &[&global_layout, &local_layout],
            vertex: shader.at("vs_main"),
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            fragment: shader.at("fs_main"),
            color_targets: &[gpu::ColorTargetState {
                format: surface_format,
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::default(),
            }],
        });

        let extent = gpu::Extent {
            width: 1,
            height: 1,
            depth: 1,
        };
        let texture = context.create_texture(gpu::TextureDesc {
            name: "texutre",
            format: gpu::TextureFormat::Rgba8Unorm,
            size: extent,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::RESOURCE | gpu::TextureUsage::COPY,
        });
        let view = context.create_texture_view(gpu::TextureViewDesc {
            name: "view",
            texture,
            format: gpu::TextureFormat::Rgba8Unorm,
            dimension: gpu::ViewDimension::D2,
            subresources: &Default::default(),
        });

        let upload_buffer = context.create_buffer(gpu::BufferDesc {
            name: "staging",
            size: (extent.width * extent.height) as u64 * 4,
            memory: gpu::Memory::Upload,
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

        let sampler = context.create_sampler(gpu::SamplerDesc {
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

        let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
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
            command_encoder: Some(command_encoder),
            prev_sync_point: None,
            texture,
            view,
            sampler,
            window_size,
            bunnies,
            rng: nanorand::WyRand::new_seed(73),
            context,
        }
    }

    fn increase(&mut self) {
        use nanorand::Rng as _;
        let spawn_count = 64 + self.bunnies.len() / 2;
        for _ in 0..spawn_count {
            let speed = self.rng.generate_range(-MAX_VELOCITY..=MAX_VELOCITY) as f32;
            self.bunnies.push(Sprite {
                locals: Locals {
                    position: [0.0, 0.5 * (self.window_size.height as f32)],
                    velocity: [speed, 0.0],
                    color: self.rng.generate::<u32>(),
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

        let encoder = self.command_encoder.as_mut().unwrap();
        encoder.start();
        encoder.init_texture(frame.texture());

        if let mut pass = encoder.render(gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: frame.texture_view(),
                init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                finish_op: gpu::FinishOp::Store,
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
        encoder.present(frame);
        let sync_point = self.context.submit(encoder);
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
        self.context
            .destroy_command_encoder(self.command_encoder.take().unwrap());
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("blade-bunnymark")
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys as _;

        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
    }

    let mut example = Example::new(&window);
    #[cfg(not(target_arch = "wasm32"))]
    let mut last_snapshot = std::time::Instant::now();
    #[cfg(target_arch = "wasm32")]
    {
        example.increase();
        example.increase();
    }
    let mut frame_count = 0;

    event_loop.run(move |event, _, control_flow| {
        let _ = &window; // force ownership by the closure
        *control_flow = winit::event_loop::ControlFlow::Poll;
        match event {
            winit::event::Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            winit::event::Event::WindowEvent { event, .. } => match event {
                #[cfg(not(target_arch = "wasm32"))]
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
                #[cfg(not(target_arch = "wasm32"))]
                if frame_count == 100 {
                    let accum_time = last_snapshot.elapsed().as_secs_f32();
                    println!(
                        "Avg frame time {}ms",
                        accum_time * 1000.0 / frame_count as f32
                    );
                    last_snapshot = std::time::Instant::now();
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
