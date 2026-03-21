#![allow(irrefutable_let_patterns)]

mod example;

use blade_graphics as gpu;

fn make_surface_config(size: winit::dpi::PhysicalSize<u32>) -> gpu::SurfaceConfig {
    log::info!("Window size: {:?}", size);
    gpu::SurfaceConfig {
        size: gpu::Extent {
            width: size.width,
            height: size.height,
            depth: 1,
        },
        usage: gpu::TextureUsage::TARGET,
        display_sync: gpu::DisplaySync::Recent,
        ..Default::default()
    }
}

struct App {
    example: Option<example::Example>,
    command_encoder: Option<gpu::CommandEncoder>,
    prev_sync_point: Option<gpu::SyncPoint>,
    surface: Option<gpu::Surface>,
    context: Option<gpu::Context>,
    window: Option<winit::window::Window>,
    #[cfg(not(target_arch = "wasm32"))]
    last_snapshot: std::time::Instant,
    frame_count: u32,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes =
            winit::window::Window::default_attributes().with_title("blade-bunnymark");
        let window = event_loop.create_window(window_attributes).unwrap();

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys as _;

            console_error_panic_hook::set_once();
            console_log::init().expect("could not initialize logger");
            // On wasm, append the canvas to the document body
            let canvas = window.canvas().unwrap();
            canvas.set_id(gpu::CANVAS_ID);
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.body())
                .and_then(|body| body.append_child(&web_sys::Element::from(canvas)).ok())
                .expect("couldn't append canvas to document body");
        }

        let context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                overlay: true,
                ..Default::default()
            })
            .unwrap()
        };
        println!("{:?}", context.device_information());

        let window_size = window.inner_size();
        let surface = context
            .create_surface_configured(&window, make_surface_config(window_size))
            .unwrap();

        let screen_size = gpu::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        };

        #[allow(unused_mut)]
        let mut example = example::Example::new(&context, screen_size, surface.info().format);
        #[cfg(target_arch = "wasm32")]
        {
            example.increase();
            example.increase();
        }

        let command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });

        self.example = Some(example);
        self.command_encoder = Some(command_encoder);
        self.surface = Some(surface);
        self.context = Some(context);
        self.window = Some(window);
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
        let example = self.example.as_mut().unwrap();
        let context = self.context.as_ref().unwrap();
        match event {
            winit::event::WindowEvent::Resized(size) => {
                let screen_size = gpu::Extent {
                    width: size.width,
                    height: size.height,
                    depth: 1,
                };
                example.set_screen_size(screen_size);
                let config = make_surface_config(size);
                context.reconfigure_surface(self.surface.as_mut().unwrap(), config);
            }
            #[cfg(not(target_arch = "wasm32"))]
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => match key_code {
                winit::keyboard::KeyCode::Escape => {
                    event_loop.exit();
                }
                winit::keyboard::KeyCode::Space => {
                    example.increase();
                }
                _ => {}
            },
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                self.frame_count += 1;
                #[cfg(not(target_arch = "wasm32"))]
                if self.frame_count == 100 {
                    let accum_time = self.last_snapshot.elapsed().as_secs_f32();
                    println!(
                        "Avg frame time {}ms",
                        accum_time * 1000.0 / self.frame_count as f32
                    );
                    self.last_snapshot = std::time::Instant::now();
                    self.frame_count = 0;
                }

                if example.screen_size()
                    == (gpu::Extent {
                        width: 0,
                        height: 0,
                        depth: 1,
                    })
                {
                    return;
                }

                let surface = self.surface.as_mut().unwrap();
                let command_encoder = self.command_encoder.as_mut().unwrap();
                let frame = surface.acquire_frame();

                example.step(0.01);

                command_encoder.start();
                command_encoder.init_texture(frame.texture());
                example.render(command_encoder, frame.texture_view());
                command_encoder.present(frame);
                let sync_point = context.submit(command_encoder);
                if let Some(sp) = self.prev_sync_point.take() {
                    let _ = context.wait_for(&sp, !0);
                }
                self.prev_sync_point = Some(sync_point);
            }
            _ => {}
        }
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = App {
        example: None,
        command_encoder: None,
        prev_sync_point: None,
        surface: None,
        context: None,
        window: None,
        #[cfg(not(target_arch = "wasm32"))]
        last_snapshot: std::time::Instant::now(),
        frame_count: 0,
    };
    event_loop.run_app(&mut app).unwrap();

    let context = app.context.as_ref().unwrap();
    if let Some(sp) = app.prev_sync_point.take() {
        let _ = context.wait_for(&sp, !0);
    }
    if let Some(mut example) = app.example.take() {
        example.deinit(context);
    }
    if let Some(mut command_encoder) = app.command_encoder.take() {
        context.destroy_command_encoder(&mut command_encoder);
    }
    if let Some(mut surface) = app.surface.take() {
        context.destroy_surface(&mut surface);
    }
}
