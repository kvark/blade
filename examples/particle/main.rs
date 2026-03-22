#![allow(irrefutable_let_patterns)]

use blade_graphics as gpu;

struct Example {
    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,
    context: gpu::Context,
    surface: gpu::Surface,
    gui_painter: blade_egui::GuiPainter,

    particle_pipeline: blade_particle::ParticlePipeline,
    particle_system: blade_particle::ParticleSystem,
    time: f32,

    sample_count: u32,
    msaa_texture: Option<gpu::Texture>,
    msaa_view: Option<gpu::TextureView>,

    export_image: bool,
}

impl Example {
    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        let config = Self::make_surface_config(size);
        self.context.reconfigure_surface(&mut self.surface, config);

        let surface_info = self.surface.info();

        if let Some(sp) = self.prev_sync_point.take() {
            let _ = self.context.wait_for(&sp, !0);
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
        if (self.sample_count > 1 || self.export_image) && self.msaa_texture.is_none() {
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
                usage: gpu::TextureUsage::TARGET
                    | gpu::TextureUsage::RESOURCE
                    | gpu::TextureUsage::COPY,
                array_layer_count: 1,
                mip_level_count: 1,
                external: if self.export_image {
                    #[cfg(target_os = "windows")]
                    {
                        Some(gpu::ExternalMemorySource::Win32KMT(None))
                    }
                    #[cfg(not(target_os = "windows"))]
                    {
                        Some(gpu::ExternalMemorySource::Fd(None))
                    }
                } else {
                    None
                },
            });

            #[cfg(not(target_os = "macos"))]
            if self.export_image {
                println!(
                    "msaa_texture_fd: {:?}",
                    self.context.get_external_texture_source(msaa_texture)
                );
            }

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

        let caps = context.capabilities();
        let sample_count = [4, 2, 1]
            .into_iter()
            .find(|&n| (caps.sample_count_mask & n) != 0)
            .unwrap();

        let gui_painter = blade_egui::GuiPainter::new(surface_info, &context);

        let particle_pipeline = blade_particle::ParticlePipeline::new(
            &context,
            blade_particle::PipelineDesc {
                name: "particle",
                draw_format: surface_info.format,
                depth_format: None,
                sample_count,
            },
        );
        let effect = blade_particle::ParticleEffect {
            capacity: 100_000,
            emitter: blade_particle::Emitter {
                rate: 6400.0,
                burst_count: 0,
                shape: blade_particle::EmitterShape::Point,
                cone_angle: 0.5,
            },
            particle: blade_particle::ParticleConfig {
                life: [1.0, 5.0],
                speed: [50.0, 250.0],
                scale: [1.0, 15.0],
                color: blade_particle::ColorConfig::Palette(vec![
                    [255, 100, 50, 255],
                    [50, 200, 255, 255],
                    [255, 220, 60, 255],
                    [100, 255, 120, 255],
                ]),
            },
        };
        let particle_system = particle_pipeline.create_system(&context, "particle system", &effect);

        let command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });

        Self {
            command_encoder,
            prev_sync_point: None,
            context,
            surface,
            gui_painter,
            particle_pipeline,
            particle_system,
            time: 0.0,
            sample_count,
            msaa_texture: None,
            msaa_view: None,
            export_image: false,
        }
    }

    fn destroy(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            let _ = self.context.wait_for(&sp, !0);
        }
        self.context
            .destroy_command_encoder(&mut self.command_encoder);
        self.gui_painter.destroy(&self.context);
        self.particle_system.destroy(&self.context);
        self.particle_pipeline.destroy(&self.context);
        self.context.destroy_surface(&mut self.surface);

        if let Some(msaa_view) = self.msaa_view.take() {
            self.context.destroy_texture_view(msaa_view);
        }
        if let Some(msaa_texture) = self.msaa_texture.take() {
            self.context.destroy_texture(msaa_texture);
        }
    }

    fn make_camera(&self, size: (u32, u32)) -> blade_particle::CameraParams {
        // Camera at +Z looking toward origin along -Z.
        let distance = 1000.0_f32;
        let fov_y = 2.0 * (500.0_f32 / distance).atan();
        let aspect = size.0 as f32 / size.1.max(1) as f32;
        let near = 0.01_f32;
        let far = distance * 2.0;
        let pos = glam::Vec3::new(0.0, 0.0, distance);
        let view = glam::Mat4::look_at_rh(pos, glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(fov_y, aspect, near, far);
        let view_proj = proj * view;
        // Camera looks along -Z, so right=+X, up=+Y (identity orientation)
        blade_particle::CameraParams {
            view_proj: view_proj.to_cols_array(),
            camera_right: [1.0, 0.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0, 0.0],
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

        // Orbit the emitter and rotate its axis
        self.time += 0.01;
        let orbit_radius = 60.0;
        self.particle_system.origin = [
            orbit_radius * self.time.cos(),
            orbit_radius * self.time.sin(),
            0.0,
        ];
        let spray_angle = self.time * 3.0;
        self.particle_system.axis = [spray_angle.cos(), spray_angle.sin(), 0.0];
        self.particle_system
            .update(&self.particle_pipeline, &mut self.command_encoder, 0.01);

        let camera = self.make_camera(screen_desc.physical_size);

        if self.sample_count <= 1 && !self.export_image {
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
                    .draw(&self.particle_pipeline, &mut pass, &camera);
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
                        finish_op: if self.export_image {
                            gpu::FinishOp::Store
                        } else {
                            gpu::FinishOp::ResolveTo(frame_view)
                        },
                    }],
                    depth_stencil: None,
                },
            ) {
                self.particle_system
                    .draw(&self.particle_pipeline, &mut pass, &camera);
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
            let _ = self.context.wait_for(&sp, !0);
        }
        self.prev_sync_point = Some(sync_point);
    }

    fn add_gui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Particle System");
        let p = &mut self.particle_system.effect.particle;
        ui.add(egui::Slider::new(&mut p.life[1], 0.1..=20.0).text("life"));
        ui.add(egui::Slider::new(&mut p.speed[1], 1.0..=500.0).text("speed"));
        ui.add(egui::Slider::new(&mut p.scale[1], 0.1..=70.0).text("scale"));
        ui.add(
            egui::Slider::new(&mut self.particle_system.effect.emitter.rate, 0.0..=20000.0)
                .text("rate"),
        );

        ui.add_space(5.0);
        ui.heading("Rendering Settings");
        egui::ComboBox::new("msaa dropdown", "MSAA samples")
            .selected_text(format!("x{}", self.sample_count))
            .show_ui(ui, |ui| {
                let caps = self.context.capabilities();
                let supported_samples = [1, 2, 4]
                    .into_iter()
                    .filter(|&n| (caps.sample_count_mask & n) != 0)
                    .collect::<Vec<_>>();
                for i in supported_samples {
                    if ui
                        .selectable_value(&mut self.sample_count, i, format!("x{i}"))
                        .changed()
                    {
                        if let Some(sp) = self.prev_sync_point.take() {
                            let _ = self.context.wait_for(&sp, !0);
                        }

                        let old_effect = self.particle_system.effect.clone();
                        self.particle_system.destroy(&self.context);
                        self.particle_pipeline.destroy(&self.context);
                        self.particle_pipeline = blade_particle::ParticlePipeline::new(
                            &self.context,
                            blade_particle::PipelineDesc {
                                name: "particle",
                                draw_format: self.surface.info().format,
                                depth_format: None,
                                sample_count: self.sample_count,
                            },
                        );
                        self.particle_system = self.particle_pipeline.create_system(
                            &self.context,
                            "particle system",
                            &old_effect,
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

struct App {
    example: Option<Example>,
    window: Option<winit::window::Window>,
    egui_winit: Option<egui_winit::State>,
    viewport_id: egui::ViewportId,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes =
            winit::window::Window::default_attributes().with_title("blade-particle");
        let window = event_loop.create_window(window_attributes).unwrap();

        let egui_ctx = egui::Context::default();
        self.viewport_id = egui_ctx.viewport_id();
        self.egui_winit = Some(egui_winit::State::new(
            egui_ctx,
            self.viewport_id,
            &window,
            None,
            None,
            None,
        ));

        self.example = Some(Example::new(&window));
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
        let window = self.window.as_ref().unwrap();
        let egui_winit = self.egui_winit.as_mut().unwrap();

        let response = egui_winit.on_window_event(window, &event);
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
            } => {
                if key_code == winit::keyboard::KeyCode::Escape {
                    event_loop.exit();
                }
            }
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            winit::event::WindowEvent::Resized(size) => {
                example.resize(size);
            }

            winit::event::WindowEvent::RedrawRequested => {
                let raw_input = egui_winit.take_egui_input(window);
                let egui_output = egui_winit.egui_ctx().run(raw_input, |egui_ctx| {
                    egui::SidePanel::left("info").show(egui_ctx, |ui| {
                        ui.add_space(5.0);
                        example.add_gui(ui);

                        ui.add_space(5.0);
                        if ui.button("Quit").clicked() {
                            event_loop.exit();
                        }
                    });
                });

                egui_winit.handle_platform_output(window, egui_output.platform_output);
                let repaint_delay = egui_output.viewport_output[&self.viewport_id].repaint_delay;

                let pixels_per_point = egui_winit::pixels_per_point(egui_winit.egui_ctx(), window);
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
                event_loop.set_control_flow(control_flow);

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
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = App {
        example: None,
        window: None,
        egui_winit: None,
        viewport_id: egui::ViewportId::ROOT,
    };
    event_loop.run_app(&mut app).unwrap();

    if let Some(mut example) = app.example.take() {
        example.destroy();
    }
}
