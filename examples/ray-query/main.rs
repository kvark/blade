#![allow(irrefutable_let_patterns)]

mod example;

use blade_graphics as gpu;
use std::time;

struct App {
    example: Option<example::Example>,
    command_encoder: Option<gpu::CommandEncoder>,
    prev_sync_point: Option<gpu::SyncPoint>,
    surface: Option<gpu::Surface>,
    context: Option<gpu::Context>,
    window: Option<winit::window::Window>,
    start_time: time::Instant,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes =
            winit::window::Window::default_attributes().with_title("blade-ray-query");
        let window = event_loop.create_window(window_attributes).unwrap();

        let context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                ray_tracing: true,
                validation: cfg!(debug_assertions),
                ..Default::default()
            })
            .unwrap()
        };
        let capabilities = context.capabilities();
        assert!(
            capabilities
                .ray_query
                .contains(gpu::ShaderVisibility::COMPUTE)
        );

        let window_size = window.inner_size();
        let screen_size = gpu::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        };
        let surface_config = gpu::SurfaceConfig {
            size: screen_size,
            usage: gpu::TextureUsage::TARGET,
            transparent: true,
            ..Default::default()
        };
        let surface = context
            .create_surface_configured(&window, surface_config)
            .unwrap();

        let example = example::Example::new(&context, screen_size, surface.info().format);

        let command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });

        self.example = Some(example);
        self.command_encoder = Some(command_encoder);
        self.surface = Some(surface);
        self.context = Some(context);
        self.window = Some(window);
        self.start_time = time::Instant::now();
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if key_code == winit::keyboard::KeyCode::Escape {
                    event_loop.exit();
                }
            }
            winit::event::WindowEvent::RedrawRequested => {
                let example = self.example.as_mut().unwrap();
                let context = self.context.as_ref().unwrap();
                let surface = self.surface.as_mut().unwrap();
                let command_encoder = self.command_encoder.as_mut().unwrap();

                let rotation_angle = self.start_time.elapsed().as_secs_f32() * 0.4;
                let frame = surface.acquire_frame();

                command_encoder.start();
                command_encoder.init_texture(frame.texture());
                example.render(command_encoder, frame.texture_view(), rotation_angle);
                command_encoder.present(frame);
                let sync_point = context.submit(command_encoder);

                if let Some(sp) = self.prev_sync_point.take() {
                    let _ = context.wait_for(&sp, !0);
                }
                self.prev_sync_point = Some(sync_point);
            }
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = App {
        example: None,
        command_encoder: None,
        prev_sync_point: None,
        surface: None,
        context: None,
        window: None,
        start_time: time::Instant::now(),
    };
    event_loop.run_app(&mut app).unwrap();

    let context = app.context.as_ref().unwrap();
    if let Some(sp) = app.prev_sync_point.take() {
        let _ = context.wait_for(&sp, !0);
    }
    if let Some(example) = app.example.take() {
        example.deinit(context);
    }
    if let Some(mut command_encoder) = app.command_encoder.take() {
        context.destroy_command_encoder(&mut command_encoder);
    }
    if let Some(mut surface) = app.surface.take() {
        context.destroy_surface(&mut surface);
    }
}
