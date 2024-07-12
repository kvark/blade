use blade_helpers::{Camera, ControlledCamera};
use std::{f32::consts, path::PathBuf, time};

struct Game {
    // engine stuff
    engine: blade::Engine,
    last_update: time::Instant,
    is_paused: bool,
    camera: ControlledCamera,
    // windowing
    window: winit::window::Window,
    egui_state: egui_winit::State,
    egui_viewport_id: egui::ViewportId,
    // game data
    _ground_handle: blade::ObjectHandle,
    object_handle: blade::ObjectHandle,
    angle: f32,
    last_event: time::Instant,
    last_mouse_pos: [i32; 2],
    is_point_selected: bool,
    is_debug_active: bool,
}

#[derive(Debug)]
struct QuitEvent;

impl Drop for Game {
    fn drop(&mut self) {
        self.engine.destroy();
    }
}

impl Game {
    fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        log::info!("Initializing");

        let window = winit::window::WindowBuilder::new()
            .with_title("Move")
            .build(event_loop)
            .unwrap();

        let camera = ControlledCamera {
            inner: Camera {
                pos: glam::Vec3::new(0.0, 1.0, 10.0).into(),
                rot: glam::Quat::from_rotation_x(consts::PI * 0.0).into(),
                fov_y: 1.0,
                depth: 0.0,
            },
            fly_speed: 10.0,
        };

        let data_path = PathBuf::from("examples/move/data");
        let mut engine = blade::Engine::new(
            &window,
            &blade::config::Engine {
                shader_path: "blade-render/code".to_string(),
                data_path: data_path.as_os_str().to_string_lossy().into_owned(),
                time_step: 0.01,
            },
        );

        let ground_handle = engine.add_object(
            &blade::config::Object {
                name: "ground".to_string(),
                visuals: vec![blade::config::Visual {
                    model: "plane.glb".to_string(),
                    ..Default::default()
                }],
                colliders: vec![],
                additional_mass: None,
            },
            blade::Transform::default(),
            blade::DynamicInput::Empty,
        );
        let object_handle = engine.add_object(
            &blade::config::Object {
                name: "object".to_string(),
                visuals: vec![blade::config::Visual {
                    model: "sphere.glb".to_string(),
                    ..Default::default()
                }],
                colliders: vec![],
                additional_mass: None,
            },
            blade::Transform::default(),
            blade::DynamicInput::SetPosition,
        );

        let egui_context = egui::Context::default();
        let egui_viewport_id = egui_context.viewport_id();
        let egui_state =
            egui_winit::State::new(egui_context, egui_viewport_id, &window, None, None);

        Self {
            engine,
            last_update: time::Instant::now(),
            is_paused: false,
            camera,
            window,
            egui_state,
            egui_viewport_id,
            _ground_handle: ground_handle,
            object_handle,
            angle: 0.0,
            last_event: time::Instant::now(),
            last_mouse_pos: [0; 2],
            is_point_selected: false,
            is_debug_active: false,
        }
    }

    fn update_time(&mut self) {
        let engine_dt = self.last_update.elapsed().as_secs_f32();
        self.last_update = time::Instant::now();
        if !self.is_paused {
            self.engine.teleport_object(
                self.object_handle,
                blade::Transform {
                    position: (glam::Vec3::new(0.0, 1.0, 0.0)
                        + 5.0 * glam::Vec3::new(self.angle.sin(), 0.0, self.angle.cos()))
                    .into(),
                    orientation: (glam::Quat::from_rotation_y(self.angle)
                        * glam::Quat::from_rotation_z(-10.0 * self.angle))
                    .into(),
                },
            );
            self.angle += 1.0 * engine_dt;
            self.engine.update(engine_dt);
        }
    }

    fn on_event(
        &mut self,
        event: &winit::event::WindowEvent,
    ) -> Result<winit::event_loop::ControlFlow, QuitEvent> {
        let response = self.egui_state.on_window_event(&self.window, event);
        if response.repaint {
            self.window.request_redraw();
        }
        if response.consumed {
            return Ok(winit::event_loop::ControlFlow::Poll);
        }

        let delta = self.last_event.elapsed().as_secs_f32().min(0.1);
        self.last_event = time::Instant::now();

        match *event {
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
                    return Err(QuitEvent);
                }
                if self.camera.on_key(key_code, delta) {
                    self.is_debug_active = false;
                }
            }
            winit::event::WindowEvent::CloseRequested => {
                return Err(QuitEvent);
            }
            winit::event::WindowEvent::RedrawRequested => {
                let wait = self.on_draw();

                return Ok(
                    if let Some(repaint_after_instant) = std::time::Instant::now().checked_add(wait)
                    {
                        winit::event_loop::ControlFlow::WaitUntil(repaint_after_instant)
                    } else {
                        winit::event_loop::ControlFlow::Wait
                    },
                );
            }
            winit::event::WindowEvent::MouseInput {
                state,
                button: winit::event::MouseButton::Right,
                ..
            } => {
                self.is_point_selected = match state {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
                self.is_debug_active |= self.is_point_selected;
            }
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                self.last_mouse_pos = [position.x as i32, position.y as i32];
            }
            _ => {}
        }

        Ok(winit::event_loop::ControlFlow::Poll)
    }

    fn populate_hud(&mut self, ui: &mut egui::Ui) {
        use blade_helpers::ExposeHud as _;

        egui::CollapsingHeader::new("Game")
            .default_open(true)
            .show(ui, |ui| {
                ui.toggle_value(&mut self.is_paused, "Pause");
            });
        egui::CollapsingHeader::new("Camera")
            .default_open(true)
            .show(ui, |ui| {
                self.camera.populate_hud(ui);
            });
        self.engine.populate_hud(ui);
    }

    fn on_draw(&mut self) -> time::Duration {
        self.update_time();

        self.engine.set_debug(
            self.is_point_selected,
            if self.is_debug_active {
                Some(self.last_mouse_pos)
            } else {
                None
            },
        );

        let raw_input = self.egui_state.take_egui_input(&self.window);
        let egui_context = self.egui_state.egui_ctx().clone();
        let egui_output = egui_context.run(raw_input, |egui_ctx| {
            let frame = {
                let mut frame = egui::Frame::side_top_panel(&egui_ctx.style());
                let mut fill = frame.fill.to_array();
                for f in fill.iter_mut() {
                    *f = (*f as u32 * 7 / 8) as u8;
                }
                frame.fill =
                    egui::Color32::from_rgba_premultiplied(fill[0], fill[1], fill[2], fill[3]);
                frame
            };
            egui::SidePanel::right("engine")
                .frame(frame)
                .show(egui_ctx, |ui| self.populate_hud(ui));
        });

        self.egui_state
            .handle_platform_output(&self.window, egui_output.platform_output);

        let primitives = self
            .egui_state
            .egui_ctx()
            .tessellate(egui_output.shapes, egui_output.pixels_per_point);
        self.engine.render(
            &self.camera.inner.into(),
            &primitives,
            &egui_output.textures_delta,
            self.window.inner_size(),
            self.window.scale_factor() as f32,
        );

        egui_output.viewport_output[&self.egui_viewport_id].repaint_delay
    }
}

fn main() {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut game = Game::new(&event_loop);
    let mut last_event = time::Instant::now();

    event_loop
        .run(|event, target| {
            let _delta = last_event.elapsed().as_secs_f32();
            last_event = time::Instant::now();

            match event {
                winit::event::Event::AboutToWait => {
                    game.window.request_redraw();
                }
                winit::event::Event::WindowEvent { event, .. } => match game.on_event(&event) {
                    Ok(control_flow) => {
                        target.set_control_flow(control_flow);
                    }
                    Err(QuitEvent) => {
                        target.exit();
                    }
                },
                _ => {}
            }
        })
        .unwrap();
}
