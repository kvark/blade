#![allow(irrefutable_let_patterns)]

use blade_graphics as gpu;
use std::collections::VecDeque;

const TIMING_HISTORY_SIZE: usize = 120;

struct Example {
    command_encoder: gpu::CommandEncoder,
    compute_encoder: gpu::CommandEncoder,
    prev_compute_sync: gpu::SyncPoint,
    prev_render_sync: gpu::SyncPoint,
    context: gpu::Context,
    surface: gpu::Surface,
    gui_painter: blade_egui::GuiPainter,

    particle_pipeline: blade_particle::ParticlePipeline,
    /// Double-buffered particle systems for async compute overlap.
    /// `particle_systems[frame_index % 2]` is being computed,
    /// while the other is being rendered (from the previous frame's compute).
    particle_systems: [blade_particle::ParticleSystem; 2],
    frame_index: usize,
    async_compute: bool,
    parallel_update: bool,
    timing_history: VecDeque<[f32; 2]>,
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

        let _ = self.context.wait_for(&self.prev_render_sync, !0);

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

    fn make_effect() -> blade_particle::ParticleEffect {
        blade_particle::ParticleEffect {
            capacity: 1000_000,
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
                multi_queue: true,
                ..Default::default()
            })
            .unwrap()
        };
        let surface = context
            .create_surface_configured(window, Self::make_surface_config(window_size))
            .unwrap();
        let surface_info = surface.info();

        let caps = context.capabilities();
        let async_compute = caps.queues.contains(&gpu::QueueType::AsyncCompute);
        log::info!("Async compute: {async_compute}");

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
        let effect = Self::make_effect();
        let particle_systems = [
            particle_pipeline.create_system(&context, "particles-A", &effect),
            particle_pipeline.create_system(&context, "particles-B", &effect),
        ];

        let command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
            queue: gpu::QueueType::Main,
        });
        let compute_queue = if async_compute {
            gpu::QueueType::AsyncCompute
        } else {
            gpu::QueueType::Main
        };
        let compute_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "compute",
            buffer_count: 2,
            queue: compute_queue,
        });

        Self {
            command_encoder,
            compute_encoder,
            prev_compute_sync: gpu::SyncPoint::default(),
            prev_render_sync: gpu::SyncPoint::default(),
            context,
            surface,
            gui_painter,
            particle_pipeline,
            particle_systems,
            frame_index: 0,
            async_compute,
            parallel_update: async_compute,
            timing_history: VecDeque::with_capacity(TIMING_HISTORY_SIZE),
            time: 0.0,
            sample_count,
            msaa_texture: None,
            msaa_view: None,
            export_image: false,
        }
    }

    fn destroy(&mut self) {
        let _ = self.context.wait_for(&self.prev_render_sync, !0);
        let _ = self.context.wait_for(&self.prev_compute_sync, !0);
        self.context
            .destroy_command_encoder(&mut self.command_encoder);
        self.context
            .destroy_command_encoder(&mut self.compute_encoder);
        self.gui_painter.destroy(&self.context);
        for ps in &mut self.particle_systems {
            ps.destroy(&self.context);
        }
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

    fn render_particles(
        &mut self,
        frame_view: gpu::TextureView,
        draw_idx: usize,
        gui_primitives: &[egui::ClippedPrimitive],
        screen_desc: &blade_egui::ScreenDescriptor,
    ) {
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
                self.particle_systems[draw_idx].draw(&self.particle_pipeline, &mut pass, &camera);
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
                self.particle_systems[draw_idx].draw(&self.particle_pipeline, &mut pass, &camera);
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
    }

    fn render(
        &mut self,
        gui_primitives: &[egui::ClippedPrimitive],
        gui_textures: &egui::TexturesDelta,
        screen_desc: &blade_egui::ScreenDescriptor,
    ) {
        profiling::scope!("frame");
        self.recreate_msaa_texutres_if_needed(
            screen_desc.physical_size,
            self.surface.info().format,
        );

        let frame = self.surface.acquire_frame();
        let frame_view = frame.texture_view();

        // Orbit the emitter and rotate its axis
        self.time += 0.01;
        let orbit_radius = 60.0;
        let origin = [
            orbit_radius * self.time.cos(),
            orbit_radius * self.time.sin(),
            0.0,
        ];
        let spray_angle = self.time * 3.0;
        let axis = [spray_angle.cos(), spray_angle.sin(), 0.0];

        // Pick which particle system to compute and which to draw.
        // In parallel mode, they are different (double-buffered).
        // In serial mode, both are the same (index 0).
        let compute_idx = if self.parallel_update {
            self.frame_index % 2
        } else {
            0
        };
        let draw_idx = if self.parallel_update {
            (self.frame_index + 1) % 2
        } else {
            0
        };

        // Update emitter state on the system we're about to compute
        self.particle_systems[compute_idx].origin = origin;
        self.particle_systems[compute_idx].axis = axis;

        if self.parallel_update {
            // === PARALLEL PATH ===
            // Compute and render touch different buffers, so they overlap on GPU.

            // Submit compute on async queue. Only waits for previous compute.
            let compute_sync = {
                profiling::scope!("submit compute");
                self.compute_encoder.start();
                self.particle_systems[compute_idx].update(
                    &self.particle_pipeline,
                    &mut self.compute_encoder,
                    0.01,
                );
                self.context
                    .submit(&mut self.compute_encoder, &[self.prev_compute_sync.clone()])
            };

            // Submit render on main queue. Only waits for previous render.
            let render_sync = {
                profiling::scope!("submit render");
                self.command_encoder.start();
                if let Some(msaa_texture) = self.msaa_texture {
                    self.command_encoder.init_texture(msaa_texture);
                }
                self.command_encoder.init_texture(frame.texture());
                self.gui_painter.update_textures(
                    &mut self.command_encoder,
                    gui_textures,
                    &self.context,
                );
                self.render_particles(frame_view, draw_idx, gui_primitives, screen_desc);
                self.command_encoder.present(frame);
                self.context
                    .submit(&mut self.command_encoder, &[self.prev_render_sync.clone()])
            };

            self.gui_painter.after_submit(&render_sync);
            let _ = self.context.wait_for(&self.prev_compute_sync, !0);
            let _ = self.context.wait_for(&self.prev_render_sync, !0);
            self.prev_compute_sync = compute_sync;
            self.prev_render_sync = render_sync;
        } else {
            // === SERIAL PATH ===
            // Everything on the main encoder, single particle system.

            self.command_encoder.start();
            if let Some(msaa_texture) = self.msaa_texture {
                self.command_encoder.init_texture(msaa_texture);
            }
            self.command_encoder.init_texture(frame.texture());
            self.gui_painter.update_textures(
                &mut self.command_encoder,
                gui_textures,
                &self.context,
            );
            self.particle_systems[0].update(
                &self.particle_pipeline,
                &mut self.command_encoder,
                0.01,
            );
            self.render_particles(frame_view, 0, gui_primitives, screen_desc);
            self.command_encoder.present(frame);
            let render_sync = self
                .context
                .submit(&mut self.command_encoder, &[self.prev_render_sync.clone()]);

            self.gui_painter.after_submit(&render_sync);
            let _ = self.context.wait_for(&self.prev_render_sync, !0);
            self.prev_render_sync = render_sync;
        }

        // Record timing history
        let compute_ms: f32 = self
            .compute_encoder
            .timings()
            .iter()
            .map(|(_, d)| d.as_secs_f32() * 1000.0)
            .sum();
        let render_ms: f32 = self
            .command_encoder
            .timings()
            .iter()
            .map(|(_, d)| d.as_secs_f32() * 1000.0)
            .sum();
        if self.timing_history.len() >= TIMING_HISTORY_SIZE {
            self.timing_history.pop_front();
        }
        self.timing_history.push_back([compute_ms, render_ms]);

        self.frame_index += 1;
        profiling::finish_frame!();
    }

    fn add_gui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Particle System");
        let compute_idx = self.frame_index % 2;
        let p = &mut self.particle_systems[compute_idx].effect.particle;
        ui.add(egui::Slider::new(&mut p.life[1], 0.1..=20.0).text("life"));
        ui.add(egui::Slider::new(&mut p.speed[1], 1.0..=500.0).text("speed"));
        ui.add(egui::Slider::new(&mut p.scale[1], 0.1..=70.0).text("scale"));
        ui.add(
            egui::Slider::new(
                &mut self.particle_systems[compute_idx].effect.emitter.rate,
                0.0..=20000.0,
            )
            .text("rate"),
        );

        ui.add_space(5.0);
        ui.heading("Rendering Settings");
        ui.add_enabled(
            self.async_compute,
            egui::Checkbox::new(&mut self.parallel_update, "Parallel update"),
        );
        if !self.async_compute {
            ui.label("(no dedicated compute queue)");
        }
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
                        let _ = self.context.wait_for(&self.prev_render_sync, !0);
                        let _ = self.context.wait_for(&self.prev_compute_sync, !0);

                        let effect = Self::make_effect();
                        for ps in &mut self.particle_systems {
                            ps.destroy(&self.context);
                        }
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
                        self.particle_systems = [
                            self.particle_pipeline.create_system(
                                &self.context,
                                "particles-A",
                                &effect,
                            ),
                            self.particle_pipeline.create_system(
                                &self.context,
                                "particles-B",
                                &effect,
                            ),
                        ];

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
        ui.heading("GPU Timings");
        // Current frame numbers
        let compute_ms: f32 = self
            .compute_encoder
            .timings()
            .iter()
            .map(|(_, d)| d.as_secs_f32() * 1000.0)
            .sum();
        let render_ms: f32 = self
            .command_encoder
            .timings()
            .iter()
            .map(|(_, d)| d.as_secs_f32() * 1000.0)
            .sum();
        if self.parallel_update {
            ui.label(format!(
                "compute: {compute_ms:.2} ms  render: {render_ms:.2} ms"
            ));
        } else {
            ui.label(format!("frame: {:.2} ms", compute_ms + render_ms));
        }

        // Stacked bar chart of timing history
        let plot_height = 80.0;
        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), plot_height),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, egui::Color32::from_gray(20));

        if !self.timing_history.is_empty() {
            let max_ms = self
                .timing_history
                .iter()
                .map(|[c, r]| c + r)
                .fold(0.1_f32, f32::max);
            let bar_w = rect.width() / TIMING_HISTORY_SIZE as f32;
            let offset = TIMING_HISTORY_SIZE - self.timing_history.len();
            for (i, &[c_ms, r_ms]) in self.timing_history.iter().enumerate() {
                let x = rect.left() + (offset + i) as f32 * bar_w;
                // Compute bar (bottom)
                let c_h = (c_ms / max_ms) * rect.height();
                let c_rect = egui::Rect::from_min_size(
                    egui::pos2(x, rect.bottom() - c_h),
                    egui::vec2(bar_w - 1.0, c_h),
                );
                painter.rect_filled(c_rect, 0.0, egui::Color32::from_rgb(100, 200, 255));
                // Render bar (stacked on top)
                let r_h = (r_ms / max_ms) * rect.height();
                let r_rect = egui::Rect::from_min_size(
                    egui::pos2(x, rect.bottom() - c_h - r_h),
                    egui::vec2(bar_w - 1.0, r_h),
                );
                painter.rect_filled(r_rect, 0.0, egui::Color32::from_rgb(255, 150, 50));
            }
            // Legend
            let legend_y = rect.top() + 2.0;
            painter.text(
                egui::pos2(rect.left() + 4.0, legend_y),
                egui::Align2::LEFT_TOP,
                format!("{max_ms:.1} ms"),
                egui::FontId::proportional(10.0),
                egui::Color32::GRAY,
            );
            painter.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(rect.right() - 90.0, legend_y),
                    egui::vec2(8.0, 8.0),
                ),
                0.0,
                egui::Color32::from_rgb(255, 150, 50),
            );
            painter.text(
                egui::pos2(rect.right() - 78.0, legend_y),
                egui::Align2::LEFT_TOP,
                "render",
                egui::FontId::proportional(10.0),
                egui::Color32::GRAY,
            );
            painter.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(rect.right() - 44.0, legend_y),
                    egui::vec2(8.0, 8.0),
                ),
                0.0,
                egui::Color32::from_rgb(100, 200, 255),
            );
            painter.text(
                egui::pos2(rect.right() - 32.0, legend_y),
                egui::Align2::LEFT_TOP,
                "compute",
                egui::FontId::proportional(10.0),
                egui::Color32::GRAY,
            );
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
                let egui_output = egui_winit.egui_ctx().run_ui(raw_input, |egui_ctx| {
                    egui::Panel::left("info").show_inside(egui_ctx, |ui| {
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
