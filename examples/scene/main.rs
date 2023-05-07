#![allow(irrefutable_let_patterns)]
#![cfg(not(target_arch = "wasm32"))]

use blade_render::{AssetHub, Camera, RenderConfig, Renderer};
use std::{path::Path, sync::Arc, time};

struct Example {
    prev_temp_buffers: Vec<blade::Buffer>,
    prev_sync_point: Option<blade::SyncPoint>,
    renderer: Renderer,
    gui_painter: blade_egui::GuiPainter,
    command_encoder: blade::CommandEncoder,
    asset_hub: AssetHub,
    context: Arc<blade::Context>,
    camera: blade_render::Camera,
    debug_mode: blade_render::DebugMode,
    ray_config: blade_render::RayConfig,
    debug_mouse_pos: Option<glam::IVec2>,
}

impl Example {
    fn new(window: &winit::window::Window, gltf_path: &str, camera: Camera) -> Self {
        let window_size = window.inner_size();
        let context = Arc::new(unsafe {
            blade::Context::init_windowed(
                window,
                blade::ContextDesc {
                    validation: cfg!(debug_assertions),
                    capture: false,
                },
            )
            .unwrap()
        });

        let screen_size = blade::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        };
        let surface_format = context.resize(blade::SurfaceConfig {
            size: screen_size,
            usage: blade::TextureUsage::TARGET,
            frame_count: 3,
        });
        let mut command_encoder = context.create_command_encoder(blade::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();

        let render_config = RenderConfig {
            screen_size,
            surface_format,
            max_debug_lines: 1000,
        };
        let mut renderer = Renderer::new(&mut command_encoder, &context, &render_config);

        let gui_painter = blade_egui::GuiPainter::new(&context, surface_format);

        let (scene, prev_temp_buffers) =
            blade_render::Scene::load_gltf(gltf_path.as_ref(), &mut command_encoder, &context);
        renderer.merge_scene(scene);
        let sync_point = context.submit(&mut command_encoder);

        let choir = Arc::new(choir::Choir::new());

        Self {
            prev_temp_buffers,
            prev_sync_point: Some(sync_point),
            renderer,
            gui_painter,
            command_encoder,
            asset_hub: AssetHub::new(
                &Path::new("examples/scene/data"),
                &Path::new("asset-cache"),
                &choir,
                &context,
            ),
            context,
            camera,
            debug_mode: blade_render::DebugMode::None,
            ray_config: blade_render::RayConfig {
                num_environment_samples: 1,
                temporal_history: 10,
            },
            debug_mouse_pos: None,
        }
    }

    fn destroy(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        for buffer in self.prev_temp_buffers.drain(..) {
            self.context.destroy_buffer(buffer);
        }
        self.gui_painter.destroy(&self.context);
        self.renderer.destroy(&self.context);
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
            &self.context,
            &mut temp_buffers,
            self.debug_mouse_pos.is_some(),
        );
        self.renderer.ray_trace(
            &mut self.command_encoder,
            &self.camera,
            self.debug_mode,
            self.debug_mouse_pos.map(|p| p.into()),
            self.ray_config,
        );

        let frame = self.context.acquire_frame();
        self.command_encoder.init_texture(frame.texture());

        if let mut pass = self.command_encoder.render(blade::RenderTargetSet {
            colors: &[blade::RenderTarget {
                view: frame.texture_view(),
                init_op: blade::InitOp::Clear(blade::TextureColor::TransparentBlack),
                finish_op: blade::FinishOp::Store,
            }],
            depth_stencil: None,
        }) {
            self.renderer.blit(&mut pass, &self.camera);
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
        egui::CollapsingHeader::new("Eye").show(ui, |ui| {
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
            ui.add(egui::DragValue::new(&mut self.camera.depth));
        });
        egui::CollapsingHeader::new("Debug")
            .default_open(true)
            .show(ui, |ui| {
                egui::ComboBox::from_label("View mode")
                    .selected_text(format!("{:?}", self.debug_mode))
                    .show_ui(ui, |ui| {
                        use blade_render::DebugMode as Dm;
                        for value in [Dm::None, Dm::Depth, Dm::Normal] {
                            ui.selectable_value(&mut self.debug_mode, value, format!("{value:?}"));
                        }
                    });
                if self.debug_mouse_pos.is_some() {
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
                ui.add(
                    egui::widgets::Slider::new(&mut self.ray_config.temporal_history, 0..=50)
                        .text("Temporal reuse"),
                );
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
        .unwrap_or("examples/scene/data/monkey.gltf".to_string());

    let camera = Camera {
        pos: [2.7, 1.6, 2.1].into(),
        rot: glam::Quat::from_xyzw(-0.04, 0.92, -0.05, -0.37)
            .normalize()
            .into(),
        fov_y: 0.8,
        depth: 100.0,
    };
    let mut example = Example::new(&window, &path_to_scene, camera);

    struct Drag {
        screen_pos: glam::IVec2,
        rotation: glam::Quat,
    }
    let mut drag_start = None;
    let mut last_event = time::Instant::now();
    let mut last_mouse_pos = glam::IVec2::new(0, 0);

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
                                screen_pos: last_mouse_pos,
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
                        example.debug_mouse_pos = match state {
                            winit::event::ElementState::Pressed => Some(last_mouse_pos),
                            winit::event::ElementState::Released => None,
                        };
                    }
                    winit::event::WindowEvent::CursorMoved { position, .. } => {
                        last_mouse_pos = glam::IVec2::new(position.x as i32, position.y as i32);
                        if let Some(ref mut drag) = drag_start {
                            let qx = glam::Quat::from_rotation_y(
                                (last_mouse_pos.x - drag.screen_pos.x) as f32 * rotate_speed,
                            );
                            let qy = glam::Quat::from_rotation_x(
                                (last_mouse_pos.y - drag.screen_pos.y) as f32 * rotate_speed,
                            );
                            example.camera.rot = (qx * drag.rotation * qy).into();
                        }
                        if let Some(ref mut pos) = example.debug_mouse_pos {
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
                    egui::SidePanel::left("my_side_panel").show(egui_ctx, |ui| {
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
