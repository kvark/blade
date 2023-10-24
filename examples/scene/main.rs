#![allow(irrefutable_let_patterns)]
#![cfg(not(target_arch = "wasm32"))]

use blade_graphics as gpu;
use blade_render::{AssetHub, Camera, RenderConfig, Renderer};
use std::{collections::VecDeque, fmt, fs, path::Path, sync::Arc, time};

const FRAME_TIME_HISTORY: usize = 30;

#[derive(Clone, Copy, PartialEq, strum::EnumIter)]
enum DebugBlitInput {
    None,
    SelectedBaseColor,
    SelectedNormal,
    Environment,
    EnvironmentWeight,
}
impl fmt::Display for DebugBlitInput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let desc = match *self {
            Self::None => "-",
            Self::SelectedBaseColor => "selected base color",
            Self::SelectedNormal => "selected normal",
            Self::Environment => "environment map",
            Self::EnvironmentWeight => "environment weight",
        };
        desc.fmt(f)
    }
}

#[derive(serde::Deserialize)]
struct ConfigCamera {
    position: mint::Vector3<f32>,
    orientation: mint::Quaternion<f32>,
    fov_y: f32,
    max_depth: f32,
    speed: f32,
}

fn default_transform() -> mint::RowMatrix3x4<f32> {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    .into()
}
fn default_luminocity() -> f32 {
    1.0
}

#[derive(serde::Deserialize)]
struct ConfigModel {
    path: String,
    #[serde(default = "default_transform")]
    transform: mint::RowMatrix3x4<f32>,
}

#[derive(serde::Deserialize)]
struct ConfigScene {
    camera: ConfigCamera,
    #[serde(default)]
    environment_map: String,
    #[serde(default = "default_luminocity")]
    average_luminocity: f32,
    models: Vec<ConfigModel>,
}

struct Example {
    prev_temp_buffers: Vec<gpu::Buffer>,
    prev_acceleration_structures: Vec<gpu::AccelerationStructure>,
    prev_sync_point: Option<gpu::SyncPoint>,
    renderer: Renderer,
    pending_scene: Option<(choir::RunningTask, blade_render::Scene)>,
    gui_painter: blade_egui::GuiPainter,
    command_encoder: Option<gpu::CommandEncoder>,
    asset_hub: AssetHub,
    context: Arc<gpu::Context>,
    camera: blade_render::Camera,
    fly_speed: f32,
    debug: blade_render::DebugConfig,
    need_accumulation_reset: bool,
    is_debug_drawing: bool,
    last_render_time: time::Instant,
    render_times: VecDeque<u32>,
    ray_config: blade_render::RayConfig,
    denoiser_config: blade_render::DenoiserConfig,
    debug_blit: Option<blade_render::DebugBlit>,
    debug_blit_input: DebugBlitInput,
    workers: Vec<choir::WorkerHandle>,
}

impl Example {
    fn make_surface_config(
        physical_size: winit::dpi::PhysicalSize<u32>,
    ) -> blade_graphics::SurfaceConfig {
        blade_graphics::SurfaceConfig {
            size: blade_graphics::Extent {
                width: physical_size.width,
                height: physical_size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            frame_count: 3,
        }
    }

    fn new(window: &winit::window::Window, scene_path: &Path) -> Self {
        log::info!("Initializing");
        //let _ = profiling::tracy_client::Client::start();

        let context = Arc::new(unsafe {
            gpu::Context::init_windowed(
                window,
                gpu::ContextDesc {
                    validation: cfg!(debug_assertions),
                    capture: false,
                },
            )
            .unwrap()
        });

        let surface_config = Self::make_surface_config(window.inner_size());
        let screen_size = surface_config.size;
        let surface_format = context.resize(surface_config);

        let num_workers = num_cpus::get_physical().max((num_cpus::get() * 3 + 2) / 4);
        log::info!("Initializing Choir with {} workers", num_workers);
        let choir = Arc::new(choir::Choir::new());
        let workers = (0..num_workers)
            .map(|i| choir.add_worker(&format!("Worker-{}", i)))
            .collect();

        let asset_hub = AssetHub::new(Path::new("asset-cache"), &choir, &context);
        let (shaders, shader_task) =
            blade_render::Shaders::load("blade-render/code/".as_ref(), &asset_hub);

        let config_scene: ConfigScene =
            ron::de::from_bytes(&fs::read(scene_path).expect("Unable to open the scene file"))
                .expect("Unable to parse the scene file");

        let camera = Camera {
            pos: config_scene.camera.position,
            rot: glam::Quat::from(config_scene.camera.orientation)
                .normalize()
                .into(),
            fov_y: config_scene.camera.fov_y,
            depth: config_scene.camera.max_depth,
        };

        let mut scene = blade_render::Scene::default();
        scene.post_processing = blade_render::PostProcessing {
            average_luminocity: config_scene.average_luminocity,
            exposure_key_value: 1.0 / 9.6,
            white_level: 1.0,
        };

        let parent = scene_path.parent().unwrap();
        let mut load_finish = choir.spawn("load finish").init_dummy();
        if !config_scene.environment_map.is_empty() {
            let meta = blade_render::texture::Meta {
                format: blade_graphics::TextureFormat::Rgba32Float,
                generate_mips: false,
                y_flip: false,
            };
            let (texture, texture_task) = asset_hub
                .textures
                .load(parent.join(&config_scene.environment_map), meta);
            load_finish.depend_on(texture_task);
            scene.environment_map = Some(texture);
        }
        for config_model in config_scene.models {
            let (model, model_task) = asset_hub.models.load(
                parent.join(&config_model.path),
                blade_render::model::Meta {
                    generate_tangents: true,
                },
            );
            load_finish.depend_on(model_task);
            scene.objects.push(blade_render::Object {
                model,
                transform: config_model.transform,
            });
        }

        log::info!("Spinning up the renderer");
        shader_task.join();
        let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();
        let render_config = RenderConfig {
            screen_size,
            surface_format,
            max_debug_lines: 1000,
        };
        let renderer = Renderer::new(
            &mut command_encoder,
            &context,
            shaders,
            &asset_hub.shaders,
            &render_config,
        );
        let sync_point = context.submit(&mut command_encoder);

        let gui_painter = blade_egui::GuiPainter::new(&context, surface_format);

        Self {
            prev_temp_buffers: Vec::new(),
            prev_acceleration_structures: Vec::new(),
            prev_sync_point: Some(sync_point),
            renderer,
            pending_scene: Some((load_finish.run(), scene)),
            gui_painter,
            command_encoder: Some(command_encoder),
            asset_hub,
            context,
            camera,
            fly_speed: config_scene.camera.speed,
            debug: blade_render::DebugConfig::default(),
            need_accumulation_reset: true,
            is_debug_drawing: false,
            last_render_time: time::Instant::now(),
            render_times: VecDeque::with_capacity(FRAME_TIME_HISTORY),
            ray_config: blade_render::RayConfig {
                num_environment_samples: 1,
                environment_importance_sampling: !config_scene.environment_map.is_empty(),
                temporal_history: 10,
                spatial_taps: 1,
                spatial_tap_history: 5,
                spatial_radius: 10,
            },
            denoiser_config: blade_render::DenoiserConfig { num_passes: 5 },
            debug_blit: None,
            debug_blit_input: DebugBlitInput::None,
            workers,
        }
    }

    fn destroy(&mut self) {
        self.workers.clear();
        self.wait_for_previous_frame();
        self.gui_painter.destroy(&self.context);
        self.renderer.destroy(&self.context);
        self.asset_hub.destroy();
        self.context
            .destroy_command_encoder(self.command_encoder.take().unwrap());
    }

    fn wait_for_previous_frame(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        for buffer in self.prev_temp_buffers.drain(..) {
            self.context.destroy_buffer(buffer);
        }
        for accel_structure in self.prev_acceleration_structures.drain(..) {
            self.context.destroy_acceleration_structure(accel_structure);
        }
    }

    fn render(
        &mut self,
        gui_primitives: &[egui::ClippedPrimitive],
        gui_textures: &egui::TexturesDelta,
        physical_size: winit::dpi::PhysicalSize<u32>,
        scale_factor: f32,
    ) {
        if let Some((ref task, _)) = self.pending_scene {
            if task.is_done() {
                log::info!("Scene is loaded");
                let (_, scene) = self.pending_scene.take().unwrap();
                self.renderer.merge_scene(scene);
            }
        }

        self.renderer.hot_reload(
            &self.asset_hub,
            &self.context,
            self.prev_sync_point.as_ref().unwrap(),
        );

        // Note: the resize is split in 2 parts because `wait_for_previous_frame`
        // wants to borrow `self` mutably, and `command_encoder` blocks that.
        let surface_config = Self::make_surface_config(physical_size);
        let new_render_size = surface_config.size;
        if new_render_size != self.renderer.get_screen_size() {
            log::info!("Resizing to {}", new_render_size);
            self.wait_for_previous_frame();
            self.context.resize(surface_config);
        }

        let command_encoder = self.command_encoder.as_mut().unwrap();
        command_encoder.start();
        if new_render_size != self.renderer.get_screen_size() {
            self.renderer
                .resize_screen(new_render_size, command_encoder, &self.context);
            self.need_accumulation_reset = true;
        }

        self.gui_painter
            .update_textures(command_encoder, gui_textures, &self.context);

        let mut temp_buffers = Vec::new();
        let mut temp_acceleration_structures = Vec::new();
        self.asset_hub.flush(command_encoder, &mut temp_buffers);
        //TODO: remove these checks.
        // We should be able to update TLAS and render content
        // even while it's still being loaded.
        if self.pending_scene.is_none() {
            self.renderer.prepare(
                command_encoder,
                &self.camera,
                &self.asset_hub,
                &self.context,
                &mut temp_buffers,
                &mut temp_acceleration_structures,
                self.is_debug_drawing,
                self.debug.mouse_pos.is_some(),
                self.need_accumulation_reset,
            );
            self.need_accumulation_reset = false;
            self.renderer
                .ray_trace(command_encoder, self.debug, self.ray_config);
            self.renderer.denoise(command_encoder, self.denoiser_config);
        }

        let frame = self.context.acquire_frame();
        command_encoder.init_texture(frame.texture());

        if let mut pass = command_encoder.render(gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: frame.texture_view(),
                init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        }) {
            let screen_desc = blade_egui::ScreenDescriptor {
                physical_size: (physical_size.width, physical_size.height),
                scale_factor,
            };
            if self.pending_scene.is_none() {
                let mut debug_blit_array = [blade_render::DebugBlit::default()];
                let debug_blits = match self.debug_blit {
                    Some(ref blit) => {
                        debug_blit_array[0] = *blit;
                        &debug_blit_array[..]
                    }
                    None => &[],
                };
                self.renderer.blit(&mut pass, debug_blits);
            }
            self.gui_painter
                .paint(&mut pass, gui_primitives, &screen_desc, &self.context);
        }

        command_encoder.present(frame);
        let sync_point = self.context.submit(command_encoder);
        self.gui_painter.after_submit(sync_point.clone());

        self.wait_for_previous_frame();
        self.prev_sync_point = Some(sync_point);
        self.prev_temp_buffers.extend(temp_buffers);
        self.prev_acceleration_structures
            .extend(temp_acceleration_structures);
    }

    fn add_gui(&mut self, ui: &mut egui::Ui) {
        use strum::IntoEnumIterator as _;

        let delta = self.last_render_time.elapsed();
        self.last_render_time += delta;
        while self.render_times.len() >= FRAME_TIME_HISTORY {
            self.render_times.pop_back();
        }
        self.render_times.push_front(delta.as_millis() as u32);

        if self.pending_scene.is_some() {
            ui.horizontal(|ui| {
                ui.label("Loading...");
                ui.spinner();
            });
            //TODO: seeing GPU Device Lost issues without this
            for task in self.asset_hub.list_running_tasks() {
                ui.label(format!("{}", task.as_ref()));
            }
            return;
        }

        egui::CollapsingHeader::new("Camera").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Position:");
                ui.add(egui::DragValue::new(&mut self.camera.pos.x));
                ui.add(egui::DragValue::new(&mut self.camera.pos.y));
                ui.add(egui::DragValue::new(&mut self.camera.pos.z));
            });
            ui.horizontal(|ui| {
                ui.label("Rotation:");
                ui.add(egui::DragValue::new(&mut self.camera.rot.v.x));
                ui.add(egui::DragValue::new(&mut self.camera.rot.v.y));
                ui.add(egui::DragValue::new(&mut self.camera.rot.v.z));
                ui.add(egui::DragValue::new(&mut self.camera.rot.s));
            });
            ui.add(egui::Slider::new(&mut self.camera.fov_y, 0.5f32..=2.0f32).text("FOV"));
            ui.add(
                egui::Slider::new(&mut self.camera.depth, 1f32..=1_000_000f32)
                    .text("depth")
                    .logarithmic(true),
            );
        });

        egui::CollapsingHeader::new("Debug")
            .default_open(true)
            .show(ui, |ui| {
                // debug mode
                egui::ComboBox::from_label("View mode")
                    .selected_text(format!("{:?}", self.debug.view_mode))
                    .show_ui(ui, |ui| {
                        for value in blade_render::DebugMode::iter() {
                            ui.selectable_value(
                                &mut self.debug.view_mode,
                                value,
                                format!("{value:?}"),
                            );
                        }
                    });
                // debug flags
                ui.label("Draw debug:");
                for (name, bit) in blade_render::DebugDrawFlags::all().iter_names() {
                    let mut enabled = self.debug.draw_flags.contains(bit);
                    ui.checkbox(&mut enabled, name);
                    self.debug.draw_flags.set(bit, enabled);
                }
                ui.label("Ignore textures:");
                for (name, bit) in blade_render::DebugTextureFlags::all().iter_names() {
                    let mut enabled = self.debug.texture_flags.contains(bit);
                    ui.checkbox(&mut enabled, name);
                    self.debug.texture_flags.set(bit, enabled);
                }
                // reset accumulation
                self.need_accumulation_reset |= ui.button("reset").clicked();
                // selection info
                let mut selection = blade_render::SelectionInfo::default();
                if let Some(screen_pos) = self.debug.mouse_pos {
                    selection = self.renderer.read_debug_selection_info();
                    let sd = selection.std_deviation.unwrap_or([0.0; 3].into());
                    let style = ui.style();
                    egui::Frame::group(style).show(ui, |ui| {
                        ui.label(format!("Pixel: {screen_pos:?}"));
                        ui.horizontal(|ui| {
                            ui.label("Std Deviation:");
                            ui.colored_label(
                                egui::Color32::WHITE,
                                format!("{:.2} {:.2} {:.2}", sd.x, sd.y, sd.z),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Texture coords:");
                            ui.colored_label(
                                egui::Color32::WHITE,
                                format!(
                                    "{:.2} {:.2}",
                                    selection.tex_coords.x, selection.tex_coords.y
                                ),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Base color:");
                            if let Some(handle) = selection.base_color_texture {
                                let name = self
                                    .asset_hub
                                    .textures
                                    .get_main_source_path(handle)
                                    .display()
                                    .to_string();
                                ui.colored_label(egui::Color32::WHITE, name);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Normal:");
                            if let Some(handle) = selection.normal_texture {
                                let name = self
                                    .asset_hub
                                    .textures
                                    .get_main_source_path(handle)
                                    .display()
                                    .to_string();
                                ui.colored_label(egui::Color32::WHITE, name);
                            }
                        });
                        if ui.button("Unselect").clicked() {
                            self.debug.mouse_pos = None;
                        }
                    });
                }
                // blits
                ui.label("Debug blit:");
                egui::ComboBox::from_label("Input")
                    .selected_text(format!("{}", self.debug_blit_input))
                    .show_ui(ui, |ui| {
                        for value in DebugBlitInput::iter() {
                            ui.selectable_value(
                                &mut self.debug_blit_input,
                                value,
                                format!("{value}"),
                            );
                        }
                    });
                let blit_view = match self.debug_blit_input {
                    DebugBlitInput::None => None,
                    DebugBlitInput::SelectedBaseColor => selection
                        .base_color_texture
                        .map(|handle| self.asset_hub.textures[handle].view),
                    DebugBlitInput::SelectedNormal => selection
                        .normal_texture
                        .map(|handle| self.asset_hub.textures[handle].view),
                    DebugBlitInput::Environment => Some(self.renderer.view_environment_main()),
                    DebugBlitInput::EnvironmentWeight => {
                        Some(self.renderer.view_environment_weight())
                    }
                };
                let min_size = 64u32;
                self.debug_blit = if let Some(view) = blit_view {
                    let mut db = match self.debug_blit.take() {
                        Some(db) => db,
                        None => {
                            let mut db = blade_render::DebugBlit::default();
                            db.target_size = [min_size, min_size];
                            db
                        }
                    };
                    db.input = view;
                    let style = ui.style();
                    egui::Frame::group(style).show(ui, |ui| {
                        ui.add(egui::Slider::new(&mut db.mip_level, 0u32..=15u32).text("Mip"));
                        ui.add(
                            egui::Slider::new(&mut db.target_size[0], min_size..=1024u32)
                                .text("Target size"),
                        );
                        db.target_size[1] = db.target_size[0];
                    });
                    Some(db)
                } else {
                    None
                };
            });

        let old_ray_config = self.ray_config;
        egui::CollapsingHeader::new("Ray Trace")
            .default_open(true)
            .show(ui, |ui| {
                let rc = &mut self.ray_config;
                ui.add(
                    egui::Slider::new(&mut rc.num_environment_samples, 1..=100u32)
                        .text("Num env samples")
                        .logarithmic(true),
                );
                ui.checkbox(
                    &mut rc.environment_importance_sampling,
                    "Env importance sampling",
                );
                ui.add(
                    egui::widgets::Slider::new(&mut rc.temporal_history, 0..=50)
                        .text("Temporal history"),
                );
                ui.add(
                    egui::widgets::Slider::new(&mut rc.spatial_taps, 0..=10).text("Spatial taps"),
                );
                ui.add(
                    egui::widgets::Slider::new(&mut rc.spatial_tap_history, 0..=50)
                        .text("Spatial tap history"),
                );
                ui.add(
                    egui::widgets::Slider::new(&mut rc.spatial_radius, 1..=50)
                        .text("Spatial radius (px)"),
                );
            });
        self.need_accumulation_reset |= self.ray_config != old_ray_config;

        egui::CollapsingHeader::new("Denoise")
            .default_open(true)
            .show(ui, |ui| {
                let dc = &mut self.denoiser_config;
                ui.add(egui::Slider::new(&mut dc.num_passes, 0..=15u32).text("A-trous passes"));
            });

        egui::CollapsingHeader::new("Tone Map").show(ui, |ui| {
            let pp = self.renderer.configure_post_processing();
            ui.add(
                egui::Slider::new(&mut pp.average_luminocity, 0.1f32..=1_000f32)
                    .text("Average luminocity")
                    .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut pp.exposure_key_value, 0.01f32..=10f32)
                    .text("Key value")
                    .logarithmic(true),
            );
            ui.add(egui::Slider::new(&mut pp.white_level, 0.1f32..=2f32).text("White level"));
        });

        egui::CollapsingHeader::new("Performance").show(ui, |ui| {
            let times = self.render_times.as_slices();
            let fd_points = egui::plot::PlotPoints::from_iter(
                times
                    .0
                    .iter()
                    .chain(times.1.iter())
                    .enumerate()
                    .map(|(x, &y)| [x as f64, y as f64]),
            );
            let fd_line = egui::plot::Line::new(fd_points).name("last");
            egui::plot::Plot::new("Frame time")
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_drag(false)
                .show_x(false)
                .include_y(0.0)
                .show_axes([false, true])
                .show(ui, |plot_ui| {
                    plot_ui.line(fd_line);
                    plot_ui.hline(egui::plot::HLine::new(1000.0 / 60.0).name("smooth"));
                });
        });
    }

    fn move_camera_by(&mut self, offset: glam::Vec3) {
        let dir = glam::Quat::from(self.camera.rot) * offset;
        self.camera.pos = (glam::Vec3::from(self.camera.pos) + dir).into();
        self.debug.mouse_pos = None;
    }
    fn rotate_camera_z_by(&mut self, angle: f32) -> glam::Quat {
        let quat = glam::Quat::from(self.camera.rot);
        let rotation = glam::Quat::from_rotation_z(angle);
        self.camera.rot = (quat * rotation).into();
        self.debug.mouse_pos = None;
        rotation
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("blade-scene")
        .build(&event_loop)
        .unwrap();

    let egui_ctx = egui::Context::default();
    let mut egui_winit = egui_winit::State::new(&event_loop);

    let mut args = std::env::args();
    let path_to_scene = args
        .nth(1)
        .unwrap_or("examples/scene/data/scene.ron".to_string());

    let mut example = Example::new(&window, Path::new(&path_to_scene));

    struct Drag {
        screen_pos: glam::IVec2,
        rotation: glam::Quat,
    }
    let mut drag_start = None::<Drag>;
    let mut last_event = time::Instant::now();
    let mut last_mouse_pos = [0i32; 2];

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
        let delta = last_event.elapsed().as_secs_f32();
        last_event = time::Instant::now();
        let move_speed = example.fly_speed * delta;
        let rotate_speed = 0.01f32;
        let rotate_speed_z = 1000.0 * delta;

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
                        winit::event::VirtualKeyCode::W => {
                            example.move_camera_by(glam::Vec3::new(0.0, 0.0, -move_speed));
                        }
                        winit::event::VirtualKeyCode::S => {
                            example.move_camera_by(glam::Vec3::new(0.0, 0.0, move_speed));
                        }
                        winit::event::VirtualKeyCode::A => {
                            example.move_camera_by(glam::Vec3::new(-move_speed, 0.0, 0.0));
                        }
                        winit::event::VirtualKeyCode::D => {
                            example.move_camera_by(glam::Vec3::new(move_speed, 0.0, 0.0));
                        }
                        winit::event::VirtualKeyCode::Z => {
                            example.move_camera_by(glam::Vec3::new(0.0, -move_speed, 0.0));
                        }
                        winit::event::VirtualKeyCode::X => {
                            example.move_camera_by(glam::Vec3::new(0.0, move_speed, 0.0));
                        }
                        winit::event::VirtualKeyCode::Q => {
                            let rot = example.rotate_camera_z_by(rotate_speed_z);
                            if let Some(ref mut drag) = drag_start {
                                drag.rotation *= rot;
                            }
                        }
                        winit::event::VirtualKeyCode::E => {
                            let rot = example.rotate_camera_z_by(-rotate_speed_z);
                            if let Some(ref mut drag) = drag_start {
                                drag.rotation *= rot;
                            }
                        }
                        _ => {}
                    },
                    winit::event::WindowEvent::CloseRequested => {
                        *control_flow = winit::event_loop::ControlFlow::Exit;
                    }
                    winit::event::WindowEvent::MouseInput {
                        state,
                        button: winit::event::MouseButton::Left,
                        ..
                    } => {
                        drag_start = match state {
                            winit::event::ElementState::Pressed => Some(Drag {
                                screen_pos: last_mouse_pos.into(),
                                rotation: example.camera.rot.into(),
                            }),
                            winit::event::ElementState::Released => None,
                        };
                    }
                    winit::event::WindowEvent::MouseInput {
                        state: winit::event::ElementState::Pressed,
                        button: winit::event::MouseButton::Right,
                        ..
                    } => {
                        example.debug.mouse_pos = Some(last_mouse_pos);
                        example.is_debug_drawing = true;
                    }
                    winit::event::WindowEvent::MouseInput {
                        state: winit::event::ElementState::Released,
                        button: winit::event::MouseButton::Right,
                        ..
                    } => {
                        example.is_debug_drawing = false;
                    }
                    winit::event::WindowEvent::CursorMoved { position, .. } => {
                        last_mouse_pos = [position.x as i32, position.y as i32];
                        if let Some(ref mut drag) = drag_start {
                            // This is rotation around the world UP, which is assumed to be Y
                            let qx = glam::Quat::from_rotation_y(
                                (drag.screen_pos.x - last_mouse_pos[0]) as f32 * rotate_speed,
                            );
                            let qy = glam::Quat::from_rotation_x(
                                (drag.screen_pos.y - last_mouse_pos[1]) as f32 * rotate_speed,
                            );
                            example.camera.rot = (qx * drag.rotation * qy).into();
                            example.debug.mouse_pos = None;
                        }
                    }
                    _ => {}
                }
            }
            winit::event::Event::RedrawRequested(_) => {
                let mut quit = false;
                let raw_input = egui_winit.take_egui_input(&window);
                let egui_output = egui_ctx.run(raw_input, |egui_ctx| {
                    let frame = {
                        let mut frame = egui::Frame::side_top_panel(&egui_ctx.style());
                        let mut fill = frame.fill.to_array();
                        for f in fill.iter_mut() {
                            *f = (*f as u32 * 7 / 8) as u8;
                        }
                        frame.fill = egui::Color32::from_rgba_premultiplied(
                            fill[0], fill[1], fill[2], fill[3],
                        );
                        frame
                    };
                    egui::SidePanel::right("control_panel")
                        .frame(frame)
                        .show(egui_ctx, |ui| {
                            example.add_gui(ui);
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

                example.render(
                    &primitives,
                    &egui_output.textures_delta,
                    window.inner_size(),
                    egui_ctx.pixels_per_point(),
                );
            }
            winit::event::Event::LoopDestroyed => {
                example.destroy();
            }
            _ => {}
        }
    })
}
