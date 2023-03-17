#![allow(irrefutable_let_patterns)]

use blade_render::{Camera, Renderer};
use std::time;

struct Example {
    _start_time: time::Instant,
    prev_temp_buffers: Vec<blade::Buffer>,
    prev_sync_point: Option<blade::SyncPoint>,
    renderer: Renderer,
    command_encoder: blade::CommandEncoder,
    context: blade::Context,
    camera: blade_render::Camera,
}

impl Example {
    fn new(window: &winit::window::Window, gltf_path: &str, camera: Camera) -> Self {
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

        let screen_size = blade::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        };
        let surface_format = context.resize(blade::SurfaceConfig {
            size: screen_size,
            usage: blade::TextureUsage::TARGET,
            frame_count: 3,
        });
        let mut command_encoder = context.create_command_encoder(blade::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();

        let mut renderer =
            Renderer::new(&mut command_encoder, &context, screen_size, surface_format);

        let (scene, prev_temp_buffers) =
            blade_render::Scene::load_gltf(gltf_path.as_ref(), &mut command_encoder, &context);
        renderer.merge_scene(scene);
        let sync_point = context.submit(&mut command_encoder);

        Self {
            _start_time: time::Instant::now(),
            prev_temp_buffers,
            prev_sync_point: Some(sync_point),
            renderer,
            command_encoder,
            context,
            camera,
        }
    }

    fn destroy(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        for buffer in self.prev_temp_buffers.drain(..) {
            self.context.destroy_buffer(buffer);
        }
        self.renderer.destroy(&self.context);
    }

    fn render(&mut self) {
        self.command_encoder.start();

        let mut temp_buffers = Vec::new();
        self.renderer
            .prepare(&mut self.command_encoder, &self.context, &mut temp_buffers);
        self.renderer
            .ray_trace(&mut self.command_encoder, &self.camera);

        let frame = self.context.acquire_frame();
        self.command_encoder.init_texture(frame.texture());

        if let mut pass = self.command_encoder.render(blade::RenderTargetSet {
            colors: &[blade::RenderTarget {
                view: frame.texture_view(),
                init_op: blade::InitOp::Clear(blade::TextureColor::TransparentBlack),
                finish_op: blade::FinishOp::Store,
            }],
            depth_stencil: None,
        }) {
            self.renderer.blit(&mut pass);
        }

        self.command_encoder.present(frame);
        let sync_point = self.context.submit(&mut self.command_encoder);

        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
            for buffer in self.prev_temp_buffers.drain(..) {
                self.context.destroy_buffer(buffer);
            }
        }
        self.prev_sync_point = Some(sync_point);
        self.prev_temp_buffers.extend(temp_buffers);
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("blade-scene")
        .build(&event_loop)
        .unwrap();

    let camera = Camera {
        pos: [0.0, 1.0, 5.0].into(),
        rot: [0.0, 1.0, 0.0, 0.0].into(),
        fov_y: 0.3,
        depth: 100.0,
    };
    let mut example = Example::new(&window, "examples/scene/data/cornellBox.gltf", camera);

    event_loop.run(move |event, _, control_flow| {
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
                    _ => {}
                },
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                _ => {}
            },
            winit::event::Event::RedrawRequested(_) => {
                *control_flow = winit::event_loop::ControlFlow::Wait;
                example.render();
            }
            winit::event::Event::LoopDestroyed => {
                example.destroy();
            }
            _ => {}
        }
    })
}
