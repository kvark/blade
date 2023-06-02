#![allow(irrefutable_let_patterns)]
#![cfg(not(target_arch = "wasm32"))]

use blade_graphics as gpu;
use blade_render::{AssetHub, Camera, RenderConfig, Renderer};
use std::{fs, path::Path, sync::Arc, time};

#[derive(serde::Deserialize)]
struct ConfigCamera {
    position: [f32; 3],
    orientation: [f32; 4],
    fov_y: f32,
    max_depth: f32,
}

fn default_transform() -> [[f32; 4]; 3] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
}
fn default_luminocity() -> f32 {
    1.0
}

#[derive(serde::Deserialize)]
struct ConfigModel {
    path: String,
    #[serde(default = "default_transform")]
    transform: [[f32; 4]; 3],
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
    prev_sync_point: Option<gpu::SyncPoint>,
    renderer: Renderer,
    gui_painter: blade_egui::GuiPainter,
    command_encoder: gpu::CommandEncoder,
    asset_hub: AssetHub,
    context: Arc<gpu::Context>,
    camera: blade_render::Camera,
    debug: blade_render::DebugConfig,
    ray_config: blade_render::RayConfig,
    debug_blits: Vec<blade_render::DebugBlit>,
    workers: Vec<choir::WorkerHandle>,
}

impl Example {
    fn new(window: &winit::window::Window, scene_path: &Path) -> Self {
        log::info!("Initializing");
        //let _ = profiling::tracy_client::Client::start();

        let window_size = window.inner_size();
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

        let screen_size = gpu::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        };
        let surface_format = context.resize(gpu::SurfaceConfig {
            size: screen_size,
            usage: gpu::TextureUsage::TARGET,
            frame_count: 3,
        });

        let num_workers = num_cpus::get_physical().max((num_cpus::get() * 3 + 2) / 4);
        log::info!("Initializing Choir with {} workers", num_workers);
        let choir = Arc::new(choir::Choir::new());
        let workers = (0..num_workers)
            .map(|i| choir.add_worker(&format!("Worker-{}", i)))
            .collect();

        let asset_hub = AssetHub::new(
            scene_path.parent().unwrap(),
            Path::new("asset-cache"),
            &choir,
            &context,
        );

        let config_scene: ConfigScene =
            ron::de::from_bytes(&fs::read(scene_path).expect("Unable to open the scene file"))
                .expect("Unable to parse the scene file");

        let camera = Camera {
            pos: config_scene.camera.position.into(),
            rot: {
                let [x, y, z, w] = config_scene.camera.orientation;
                glam::Quat::from_xyzw(x, y, z, w).normalize().into()
            },
            fov_y: config_scene.camera.fov_y,
            depth: config_scene.camera.max_depth,
        };

        let mut scene = blade_render::Scene::default();
        let time_start = time::Instant::now();
        scene.post_processing = blade_render::PostProcessing {
            average_luminocity: config_scene.average_luminocity,
            exposure_key_value: 1.0 / 9.6,
            white_level: 1.0,
        };

        let mut load_finish = choir.spawn("load finish").init_dummy();
        if !config_scene.environment_map.is_empty() {
            let meta = blade_render::texture::Meta {
                format: blade_graphics::TextureFormat::Rgba32Float,
            };
            let (texture, texture_task) = asset_hub
                .textures
                .load(config_scene.environment_map.as_ref(), meta);
            load_finish.depend_on(texture_task);
            scene.environment_map = Some(texture);
        }
        for config_model in config_scene.models {
            let (model, model_task) = asset_hub
                .models
                .load(config_model.path.as_ref(), blade_render::model::Meta);
            load_finish.depend_on(model_task);
            scene.objects.push(blade_render::Object {
                model,
                transform: config_model.transform.into(),
            });
        }
        log::info!("Waiting for scene to load");
        let _ = load_finish.run().join();
        println!("Scene loaded in {} ms", time_start.elapsed().as_millis());

        log::info!("Spinning up the renderer");
        let mut prev_temp_buffers = Vec::new();
        let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();
        asset_hub.flush(&mut command_encoder, &mut prev_temp_buffers);

        let render_config = RenderConfig {
            screen_size,
            surface_format,
            max_debug_lines: 1000,
        };
        let mut renderer = Renderer::new(&mut command_encoder, &context, &render_config);
        renderer.merge_scene(scene);
        let sync_point = context.submit(&mut command_encoder);

        let gui_painter = blade_egui::GuiPainter::new(&context, surface_format);

        Self {
            prev_temp_buffers,
            prev_sync_point: Some(sync_point),
            renderer,
            gui_painter,
            command_encoder,
            asset_hub,
            context,
            camera,
            debug: blade_render::DebugConfig::default(),
            ray_config: blade_render::RayConfig {
                num_environment_samples: 1,
                environment_importance_sampling: true,
                temporal_history: 10,
            },
            debug_blits: Vec::new(),
            workers,
        }
    }

    fn destroy(&mut self) {
        self.workers.clear();
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        for buffer in self.prev_temp_buffers.drain(..) {
            self.context.destroy_buffer(buffer);
        }
        self.gui_painter.destroy(&self.context);
        self.renderer.destroy(&self.context);
        self.asset_hub.destroy();
    }

    fn render(
        &mut self,
        gui_primitives: &[egui::ClippedPrimitive],
        gui_textures: &egui::TexturesDelta,
        screen_desc: &blade_egui::ScreenDescriptor,
    ) {
        self.renderer
            .hot_reload(&self.context, self.prev_sync_point.as_ref().unwrap());

        self.command_encoder.start();

        self.gui_painter
            .update_textures(&mut self.command_encoder, gui_textures, &self.context);

        let mut temp_buffers = Vec::new();
        self.asset_hub
            .flush(&mut self.command_encoder, &mut temp_buffers);
        self.renderer.prepare(
            &mut self.command_encoder,
            &self.asset_hub,
            &self.context,
            &mut temp_buffers,
            self.debug.mouse_pos.is_some(),
        );
        self.renderer.ray_trace(
            &mut self.command_encoder,
            &self.camera,
            self.debug,
            self.ray_config,
        );

        let frame = self.context.acquire_frame();
        self.command_encoder.init_texture(frame.texture());

        if let mut pass = self.command_encoder.render(gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: frame.texture_view(),
                init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        }) {
            self.renderer
                .blit(&mut pass, &self.camera, &self.debug_blits);
            self.gui_painter
                .paint(&mut pass, gui_primitives, screen_desc, &self.context);
        }

        self.command_encoder.present(frame);
        let sync_point = self.context.submit(&mut self.command_encoder);
        self.gui_painter.after_submit(sync_point.clone());

        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
            for buffer in self.prev_temp_buffers.drain(..) {
                self.context.destroy_buffer(buffer);
            }
        }
        self.prev_sync_point = Some(sync_point);
        self.prev_temp_buffers.extend(temp_buffers);
    }

    fn add_gui(&mut self, ui: &mut egui::Ui) {
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
                        use blade_render::DebugMode as Dm;
                        for value in [Dm::None, Dm::Depth, Dm::Normal] {
                            ui.selectable_value(
                                &mut self.debug.view_mode,
                                value,
                                format!("{value:?}"),
                            );
                        }
                    });
                // debug flags
                for (name, bit) in blade_render::DebugFlags::all().iter_names() {
                    let mut enabled = self.debug.flags.contains(bit);
                    ui.checkbox(&mut enabled, name);
                    self.debug.flags.set(bit, enabled);
                }
                // blits
                let mut blits_to_remove = Vec::new();
                for (i, db) in self.debug_blits.iter_mut().enumerate() {
                    let style = ui.style();
                    egui::Frame::group(style).show(ui, |ui| {
                        if ui.button("-remove").clicked() {
                            blits_to_remove.push(i);
                        }
                        egui::ComboBox::from_label("Input")
                            .selected_text(format!("{:?}", db.input))
                            .show_ui(ui, |ui| {
                                use blade_render::DebugBlitInput as Dbi;
                                for value in [Dbi::Dummy, Dbi::Environment, Dbi::EnvironmentWeight]
                                {
                                    ui.selectable_value(&mut db.input, value, format!("{value:?}"));
                                }
                            });
                        ui.add(egui::Slider::new(&mut db.mip_level, 0u32..=10u32).text("Mip"));
                        ui.add(
                            egui::Slider::new(&mut db.scale_power, -5i32..=5i32).text("Scale Pow"),
                        );
                    });
                }
                for i in blits_to_remove.into_iter().rev() {
                    self.debug_blits.remove(i);
                }
                if ui.button("+add blit").clicked() {
                    self.debug_blits.push(blade_render::DebugBlit::default());
                }
                // variance
                if self.debug.mouse_pos.is_some() {
                    let sd = self
                        .renderer
                        .read_debug_std_deviation()
                        .unwrap_or([0.0; 3].into());
                    ui.horizontal(|ui| {
                        ui.label("Std Deviation:");
                        ui.label(format!("{:.2}", sd.x));
                        ui.label(format!("{:.2}", sd.y));
                        ui.label(format!("{:.2}", sd.z));
                    });
                }
            });
        egui::CollapsingHeader::new("Ray Trace")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.ray_config.num_environment_samples, 1..=100u32)
                        .text("Num env samples")
                        .logarithmic(true),
                );
                ui.checkbox(
                    &mut self.ray_config.environment_importance_sampling,
                    "Env importance sampling",
                );
                ui.add(
                    egui::widgets::Slider::new(&mut self.ray_config.temporal_history, 0..=50)
                        .text("Temporal reuse"),
                );
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
    }

    fn move_camera_by(&mut self, offset: glam::Vec3) {
        let dir = glam::Quat::from(self.camera.rot) * offset;
        self.camera.pos = (glam::Vec3::from(self.camera.pos) + dir).into();
    }
    fn rotate_camera_z_by(&mut self, angle: f32) {
        let quat = glam::Quat::from(self.camera.rot);
        self.camera.rot = (quat * glam::Quat::from_rotation_z(angle)).into();
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
    let mut drag_start = None;
    let mut last_event = time::Instant::now();
    let mut last_mouse_pos = [0i32; 2];

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
        let delta = last_event.elapsed().as_secs_f32();
        last_event = time::Instant::now();
        let move_speed = 1000.0 * delta;
        let rotate_speed = 0.01f32;
        let rotate_speed_z = 200.0 * delta;

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
                            example.move_camera_by(glam::Vec3::new(0.0, 0.0, move_speed));
                        }
                        winit::event::VirtualKeyCode::S => {
                            example.move_camera_by(glam::Vec3::new(0.0, 0.0, -move_speed));
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
                            example.rotate_camera_z_by(rotate_speed_z);
                        }
                        winit::event::VirtualKeyCode::E => {
                            example.rotate_camera_z_by(-rotate_speed_z);
                        }
                        _ => {}
                    },
                    winit::event::WindowEvent::CloseRequested => {
                        *control_flow = winit::event_loop::ControlFlow::Exit;
                    }
                    winit::event::WindowEvent::MouseInput {
                        state,
                        button: winit::event::MouseButton::Right,
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
                        state,
                        button: winit::event::MouseButton::Left,
                        ..
                    } => {
                        example.debug.mouse_pos = match state {
                            winit::event::ElementState::Pressed => Some(last_mouse_pos),
                            winit::event::ElementState::Released => None,
                        };
                    }
                    winit::event::WindowEvent::CursorMoved { position, .. } => {
                        last_mouse_pos = [position.x as i32, position.y as i32];
                        if let Some(ref mut drag) = drag_start {
                            let qx = glam::Quat::from_rotation_y(
                                (last_mouse_pos[0] - drag.screen_pos.x) as f32 * rotate_speed,
                            );
                            let qy = glam::Quat::from_rotation_x(
                                (last_mouse_pos[1] - drag.screen_pos.y) as f32 * rotate_speed,
                            );
                            example.camera.rot = (qx * drag.rotation * qy).into();
                        }
                        if let Some(ref mut pos) = example.debug.mouse_pos {
                            *pos = last_mouse_pos;
                        }
                    }
                    _ => {}
                }
            }
            winit::event::Event::RedrawRequested(_) => {
                let mut quit = false;
                let raw_input = egui_winit.take_egui_input(&window);
                let egui_output = egui_ctx.run(raw_input, |egui_ctx| {
                    egui::SidePanel::right("control_panel").show(egui_ctx, |ui| {
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
