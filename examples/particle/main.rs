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

    sample_count: u32,
    msaa_texture: Option<gpu::Texture>,
    msaa_view: Option<gpu::TextureView>,
}

// pub const INITIAL_SAMPLE_COUNT: u32 = 1;
// pub const INITIAL_SAMPLE_COUNT: u32 = 2;
pub const INITIAL_SAMPLE_COUNT: u32 = 4;
// pub const INITIAL_SAMPLE_COUNT: u32 = 8;

impl Example {
    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        let config = Self::make_surface_config(size);
        self.context.reconfigure_surface(&mut self.surface, config);

        let surface_info = self.surface.info();

        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }

        if let Some(msaa_view) = self.msaa_view.take() {
            self.context.destroy_texture_view(msaa_view);
        }
        if let Some(msaa_texture) = self.msaa_texture.take() {
            self.context.destroy_texture(msaa_texture);
        }
        self.recreate_msaa_texutres_if_needed((size.width, size.height), surface_info.format);
    }

    fn recreate_msaa_texutres_if_needed(
        &mut self,
        (width, height): (u32, u32),
        format: gpu::TextureFormat,
    ) {
        if self.sample_count > 1 && self.msaa_texture.is_none() {
            let msaa_texture = self.context.create_texture(gpu::TextureDesc {
                name: "msaa texture",
                format,
                size: gpu::Extent {
                    width,
                    height,
                    depth: 1,
                },
                sample_count: self.sample_count,
                dimension: gpu::TextureDimension::D2,
                usage: gpu::TextureUsage::TARGET,
                array_layer_count: 1,
                mip_level_count: 1,
            });
            let msaa_view = self.context.create_texture_view(
                msaa_texture,
                gpu::TextureViewDesc {
                    name: "msaa texture view",
                    format,
                    dimension: gpu::ViewDimension::D2,
                    subresources: &Default::default(),
                },
            );

            self.msaa_texture = Some(msaa_texture);
            self.msaa_view = Some(msaa_view);
        }
    }

    fn make_surface_config(size: winit::dpi::PhysicalSize<u32>) -> gpu::SurfaceConfig {
        log::info!("Window size: {:?}", size);
        gpu::SurfaceConfig {
            size: gpu::Extent {
                width: size.width,
                height: size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Block,
            ..Default::default()
        }
    }

    fn new(window: &winit::window::Window) -> Self {
        let window_size = window.inner_size();
        let context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                timing: true,
                capture: false,

                ..Default::default()
            })
            .unwrap()
        };
        let surface = context
            .create_surface_configured(window, Self::make_surface_config(window_size))
            .unwrap();
        let surface_info = surface.info();

        let gui_painter = blade_egui::GuiPainter::new(surface_info, &context);
        let particle_system = particle::System::new(
            &context,
            particle::SystemDesc {
                name: "particle system",
                capacity: 100_000,
                draw_format: surface_info.format,
            },
            INITIAL_SAMPLE_COUNT,
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
            sample_count: INITIAL_SAMPLE_COUNT,
            msaa_texture: None,
            msaa_view: None,
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

        if let Some(msaa_view) = self.msaa_view.take() {
            self.context.destroy_texture_view(msaa_view);
        }
        if let Some(msaa_texture) = self.msaa_texture.take() {
            self.context.destroy_texture(msaa_texture);
        }
    }

    fn render(
        &mut self,
        gui_primitives: &[egui::ClippedPrimitive],
        gui_textures: &egui::TexturesDelta,
        screen_desc: &blade_egui::ScreenDescriptor,
    ) {
        self.recreate_msaa_texutres_if_needed(
            screen_desc.physical_size,
            self.surface.info().format,
        );

        let frame = self.surface.acquire_frame();
        let frame_view = frame.texture_view();
        self.command_encoder.start();
        if let Some(msaa_texture) = self.msaa_texture {
            self.command_encoder.init_texture(msaa_texture);
        }
        self.command_encoder.init_texture(frame.texture());

        self.gui_painter
            .update_textures(&mut self.command_encoder, gui_textures, &self.context);
        self.particle_system.update(&mut self.command_encoder);

        if self.sample_count <= 1 {
            if let mut pass = self.command_encoder.render(
                "draw particles and ui",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: frame_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            ) {
                self.particle_system
                    .draw(&mut pass, screen_desc.physical_size);
                self.gui_painter
                    .paint(&mut pass, gui_primitives, screen_desc, &self.context);
            }
        } else {
            if let mut pass = self.command_encoder.render(
                "draw particles with msaa resolve",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: self.msaa_view.unwrap(),
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                        finish_op: gpu::FinishOp::ResolveTo(frame_view),
                    }],
                    depth_stencil: None,
                },
            ) {
                self.particle_system
                    .draw(&mut pass, screen_desc.physical_size);
            }
            if let mut pass = self.command_encoder.render(
                "draw ui",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: frame_view,
                        init_op: gpu::InitOp::Load,
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            ) {
                self.gui_painter
                    .paint(&mut pass, gui_primitives, screen_desc, &self.context);
            }
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
        ui.add_space(5.0);
        ui.heading("Particle System");
        self.particle_system.add_gui(ui);

        ui.add_space(5.0);
        ui.heading("Rendering Settings");
        egui::ComboBox::new("msaa dropdown", "MSAA samples")
            .selected_text(format!("x{}", self.sample_count))
            .show_ui(ui, |ui| {
                for i in [1, 2, 4] {
                    if ui
                        .selectable_value(&mut self.sample_count, i, format!("x{i}"))
                        .changed()
                    {
                        if let Some(sp) = self.prev_sync_point.take() {
                            self.context.wait_for(&sp, !0);
                        }

                        self.particle_system.destroy(&self.context);
                        self.particle_system = particle::System::new(
                            &self.context,
                            particle::SystemDesc {
                                name: "particle system",
                                capacity: 100_000,
                                draw_format: self.surface.info().format,
                            },
                            self.sample_count,
                        );
                        if let Some(msaa_view) = self.msaa_view.take() {
                            self.context.destroy_texture_view(msaa_view);
                        }
                        if let Some(msaa_texture) = self.msaa_texture.take() {
                            self.context.destroy_texture(msaa_texture);
                        }
                    }
                }
            });

        ui.add_space(5.0);
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

                        winit::event::WindowEvent::Resized(size) => {
                            example.resize(size);
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
                                winit::event_loop::ControlFlow::WaitUntil(
                                    repaint_after_instant.into(),
                                )
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
