#![allow(irrefutable_let_patterns)]

use blade_graphics as gpu;

mod particle;

struct Example {
    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,
    context: gpu::Context,
    gui_painter: blade_egui::GuiPainter,
    particle_system: particle::System,
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
            color_space: gpu::ColorSpace::Linear,
        });
        let gui_painter = blade_egui::GuiPainter::new(surface_format, &context);
        let particle_system = particle::System::new(
            &context,
            particle::SystemDesc {
                name: "particle system",
                capacity: 100_000,
                draw_format: surface_format,
            },
        );

        let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();
        particle_system.reset(&mut command_encoder);
        let sync_point = context.submit(&mut command_encoder);

        Self {
            command_encoder,
            prev_sync_point: Some(sync_point),
            context,
            gui_painter,
            particle_system,
        }
    }

    fn destroy(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        self.context
            .destroy_command_encoder(&mut self.command_encoder);
        self.gui_painter.destroy(&self.context);
        self.particle_system.destroy(&self.context);
    }

    fn render(
        &mut self,
        gui_primitives: &[egui::ClippedPrimitive],
        gui_textures: &egui::TexturesDelta,
        screen_desc: &blade_egui::ScreenDescriptor,
    ) {
        let frame = self.context.acquire_frame();
        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        self.gui_painter
            .update_textures(&mut self.command_encoder, gui_textures, &self.context);

        self.particle_system.update(&mut self.command_encoder);

        if let mut pass = self.command_encoder.render(gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: frame.texture_view(),
                init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        }) {
            self.particle_system.draw(&mut pass);
            self.gui_painter
                .paint(&mut pass, gui_primitives, screen_desc, &self.context);
        }
        self.command_encoder.present(frame);
        let sync_point = self.context.submit(&mut self.command_encoder);
        self.gui_painter.after_submit(&sync_point);

        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        self.prev_sync_point = Some(sync_point);
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("blade-particle")
        .build(&event_loop)
        .unwrap();

    let egui_ctx = egui::Context::default();
    let mut egui_winit = egui_winit::State::new(&event_loop);

    let mut example = Example::new(&window);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
        match event {
            winit::event::Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            winit::event::Event::WindowEvent { event, .. } => {
                let response = egui_winit.on_event(&egui_ctx, &event);
                if response.consumed {
                    return;
                }
                if response.repaint {
                    window.request_redraw();
                }

                match event {
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
                }
            }
            winit::event::Event::RedrawRequested(_) => {
                let mut quit = false;
                let raw_input = egui_winit.take_egui_input(&window);
                let egui_output = egui_ctx.run(raw_input, |egui_ctx| {
                    egui::SidePanel::left("my_side_panel").show(egui_ctx, |ui| {
                        ui.heading("Particle System");
                        example.particle_system.add_gui(ui);
                        if ui.button("Quit").clicked() {
                            quit = true;
                        }
                    });
                });

                egui_winit.handle_platform_output(&window, &egui_ctx, egui_output.platform_output);

                let primitives = egui_ctx.tessellate(egui_output.shapes);

                *control_flow = if quit {
                    winit::event_loop::ControlFlow::Exit
                } else if let Some(repaint_after_instant) =
                    std::time::Instant::now().checked_add(egui_output.repaint_after)
                {
                    winit::event_loop::ControlFlow::WaitUntil(repaint_after_instant)
                } else {
                    winit::event_loop::ControlFlow::Wait
                };

                //Note: this will probably look different with proper support for resizing
                let window_size = window.inner_size();
                let screen_desc = blade_egui::ScreenDescriptor {
                    physical_size: (window_size.width, window_size.height),
                    scale_factor: egui_ctx.pixels_per_point(),
                };

                example.render(&primitives, &egui_output.textures_delta, &screen_desc);
            }
            winit::event::Event::LoopDestroyed => {
                example.destroy();
            }
            _ => {}
        }
    })
}
