#![allow(irrefutable_let_patterns)]

pub mod belt;
mod gui;

struct Example {
    command_encoder: blade::CommandEncoder,
    prev_sync_point: Option<blade::SyncPoint>,
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

        let _surface_format = context.resize(blade::SurfaceConfig {
            size: blade::Extent {
                width: window_size.width,
                height: window_size.height,
                depth: 1,
            },
            usage: blade::TextureUsage::TARGET,
            frame_count: 3,
        });

        let command_encoder = context.create_command_encoder(blade::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });

        Self {
            command_encoder,
            prev_sync_point: None,
            context,
        }
    }

    fn render(&mut self) {
        let frame = self.context.acquire_frame();

        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        if let _ = self.command_encoder.render(blade::RenderTargetSet {
            colors: &[blade::RenderTarget {
                view: frame.texture_view(),
                init_op: blade::InitOp::Clear(blade::TextureColor::TransparentBlack),
                finish_op: blade::FinishOp::Store,
            }],
            depth_stencil: None,
        }) {}
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
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("blade-particle")
        .build(&event_loop)
        .unwrap();

    let mut example = Example::new(&window);

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
                    _ => {}
                },
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                _ => {}
            },
            winit::event::Event::RedrawRequested(_) => {
                example.render();
            }
            winit::event::Event::LoopDestroyed => {
                example.deinit();
            }
            _ => {}
        }
    })
}
