#![allow(irrefutable_let_patterns)]

use blade_graphics as gpu;

mod particle;

struct Example {
    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,
    context: gpu::Context,
    surface: gpu::Surface,
    gui_painter: blade_egui::GuiPainter,
    particle_system: particle::System,
}

impl Example {
    fn new(window: &winit::window::Window) -> Self {
        let window_size = window.inner_size();
        let context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                timing: true,
                capture: true,
                overlay: false,
            })
            .unwrap()
        };
        let surface_config = gpu::SurfaceConfig {
            size: gpu::Extent {
                width: window_size.width,
                height: window_size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Block,
            ..Default::default()
        };
        let surface = context.create_surface(window, surface_config).unwrap();
        let surface_info = surface.info();

        let gui_painter = blade_egui::GuiPainter::new(surface_info, &context);
        let particle_system = particle::System::new(
            &context,
            particle::SystemDesc {
                name: "particle system",
                capacity: 100_000,
                draw_format: surface_info.format,
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
            surface,
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
        self.context.destroy_surface(&mut self.surface);
    }

    fn render(
        &mut self,
        gui_primitives: &[egui::ClippedPrimitive],
        gui_textures: &egui::TexturesDelta,
        screen_desc: &blade_egui::ScreenDescriptor,
    ) {
        let frame = self.context.acquire_frame(&mut self.surface);
        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        self.gui_painter
            .update_textures(&mut self.command_encoder, gui_textures, &self.context);

        self.particle_system.update(&mut self.command_encoder);

        if let mut pass = self.command_encoder.render(
            "draw",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: frame.texture_view(),
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
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

    fn add_gui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Particle System");
        self.particle_system.add_gui(ui);
        ui.heading("Timings");
        for (name, time) in self.command_encoder.timings() {
            let millis = time.as_secs_f32() * 1000.0;
            ui.horizontal(|ui| {
                ui.label(name);
                ui.colored_label(egui::Color32::WHITE, format!("{:.2} ms", millis));
            });
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window_attributes =
        winit::window::Window::default_attributes().with_title("blade-particle");

    let window = event_loop.create_window(window_attributes).unwrap();

    let egui_ctx = egui::Context::default();
    let viewport_id = egui_ctx.viewport_id();
    let mut egui_winit = egui_winit::State::new(egui_ctx, viewport_id, &window, None, None, None);

    let mut example = Example::new(&window);

    event_loop
        .run(|event, target| {
            target.set_control_flow(winit::event_loop::ControlFlow::Poll);
            match event {
                winit::event::Event::AboutToWait => {
                    window.request_redraw();
                }
                winit::event::Event::WindowEvent { event, .. } => {
                    let response = egui_winit.on_window_event(&window, &event);
                    if response.consumed {
                        return;
                    }
                    if response.repaint {
                        window.request_redraw();
                    }

                    match event {
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
                                target.exit();
                            }
                            _ => {}
                        },
                        winit::event::WindowEvent::CloseRequested => {
                            target.exit();
                        }
                        winit::event::WindowEvent::RedrawRequested => {
                            let raw_input = egui_winit.take_egui_input(&window);
                            let egui_output = egui_winit.egui_ctx().run(raw_input, |egui_ctx| {
                                egui::SidePanel::left("info").show(egui_ctx, |ui| {
                                    example.add_gui(ui);
                                    if ui.button("Quit").clicked() {
                                        target.exit();
                                    }
                                });
                            });

                            egui_winit.handle_platform_output(&window, egui_output.platform_output);
                            let repaint_delay =
                                egui_output.viewport_output[&viewport_id].repaint_delay;

                            let pixels_per_point =
                                egui_winit::pixels_per_point(egui_winit.egui_ctx(), &window);
                            let primitives = egui_winit
                                .egui_ctx()
                                .tessellate(egui_output.shapes, pixels_per_point);

                            let control_flow = if let Some(repaint_after_instant) =
                                std::time::Instant::now().checked_add(repaint_delay)
                            {
                                winit::event_loop::ControlFlow::WaitUntil(repaint_after_instant)
                            } else {
                                winit::event_loop::ControlFlow::Wait
                            };
                            target.set_control_flow(control_flow);

                            //Note: this will probably look different with proper support for resizing
                            let window_size = window.inner_size();
                            let screen_desc = blade_egui::ScreenDescriptor {
                                physical_size: (window_size.width, window_size.height),
                                scale_factor: pixels_per_point,
                            };

                            example.render(&primitives, &egui_output.textures_delta, &screen_desc);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        })
        .unwrap();

    example.destroy();
}
