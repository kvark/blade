mod config;

use std::{f32::consts, fs, mem, ops, path::PathBuf, time};

#[derive(Clone)]
struct Wheel {
    object: blade::ObjectHandle,
    spin_joint: blade::JointHandle,
    suspender: Option<blade::ObjectHandle>,
    steer_joint: Option<blade::JointHandle>,
}

struct Vehicle {
    body_handle: blade::ObjectHandle,
    jump_impulse: f32,
    roll_impulse: f32,
    wheels: Vec<Wheel>,
}

struct Game {
    // engine stuff
    engine: blade::Engine,
    last_physics_update: time::Instant,
    last_camera_update: time::Instant,
    last_camera_base_quat: glam::Quat,
    is_paused: bool,
    // windowing
    window: winit::window::Window,
    egui_state: egui_winit::State,
    egui_viewport_id: egui::ViewportId,
    // game data
    _ground_handle: blade::ObjectHandle,
    vehicle: Vehicle,
    cam_config: config::Camera,
    spawn_pos: glam::Vec3,
}

#[derive(Clone, Debug, PartialEq)]
struct Isometry {
    position: glam::Vec3,
    orientation: glam::Quat,
}
impl From<blade::Transform> for Isometry {
    fn from(transform: blade::Transform) -> Self {
        Self {
            position: transform.position.into(),
            orientation: transform.orientation.into(),
        }
    }
}
impl Isometry {
    fn inverse(&self) -> Self {
        let orientation = self.orientation.inverse();
        Self {
            position: orientation * -self.position,
            orientation,
        }
    }

    fn to_blade(&self) -> blade::Transform {
        blade::Transform {
            position: self.position.into(),
            orientation: self.orientation.into(),
        }
    }
}
impl ops::Mul<Isometry> for Isometry {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            position: self.orientation * other.position + self.position,
            orientation: self.orientation * other.orientation,
        }
    }
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

        let window_attributes = winit::window::Window::default_attributes().with_title("RayCraft");

        let window = event_loop.create_window(window_attributes).unwrap();

        let cam_config = config::Camera {
            distance: 5.0,
            azimuth: 0.0,
            altitude: 0.5,
            speed: 0.05,
            target: [0.0, 1.0, 0.0],
            fov: 1.0,
        };

        let data_path = PathBuf::from("examples/vehicle/data");
        let mut engine = blade::Engine::new(
            &window,
            &blade::config::Engine {
                shader_path: "blade-render/code".to_string(),
                data_path: data_path.as_os_str().to_string_lossy().into_owned(),
                time_step: 0.01,
            },
        );

        let lev_config: config::Level = ron::de::from_bytes(
            &fs::read(data_path.join("level.ron")).expect("Unable to open the level config"),
        )
        .expect("Unable to parse the level config");
        engine.set_environment_map(&lev_config.environment);
        engine.set_gravity(lev_config.gravity);
        engine.set_average_luminosity(lev_config.average_luminocity);

        let ground_handle = engine.add_object(
            &lev_config.ground,
            blade::Transform::default(),
            blade::DynamicInput::Empty,
        );

        let veh_config: config::Vehicle = ron::de::from_bytes(
            &fs::read(data_path.join("raceFuture.ron")).expect("Unable to open the vehicle config"),
        )
        .expect("Unable to parse the vehicle config");
        let body_config = blade::config::Object {
            name: "vehicle/body".to_string(),
            visuals: vec![veh_config.body.visual],
            colliders: vec![veh_config.body.collider],
            additional_mass: None,
        };
        let spawn_pos = glam::Vec3::from(lev_config.spawn_pos);
        let mut vehicle = Vehicle {
            body_handle: engine.add_object(
                &body_config,
                blade::Transform {
                    position: spawn_pos.into(),
                    ..Default::default()
                },
                blade::DynamicInput::Full,
            ),
            jump_impulse: veh_config.jump_impulse,
            roll_impulse: veh_config.roll_impulse,
            wheels: Vec::new(),
        };
        let wheel_config = blade::config::Object {
            name: "vehicle/wheel".to_string(),
            visuals: vec![veh_config.wheel.visual],
            colliders: vec![veh_config.wheel.collider],
            additional_mass: None,
        };
        let suspender_config = blade::config::Object {
            name: "vehicle/suspender".to_string(),
            visuals: vec![],
            colliders: vec![],
            additional_mass: Some(veh_config.suspender),
        };

        //Note: in the vehicle coordinate system X=left, Y=up, Z=forward
        for ac in veh_config.axles {
            for wheel_x in ac.x_wheels {
                let offset = glam::Vec3::new(wheel_x, ac.y, ac.z);
                let rotation = if wheel_x > 0.0 {
                    glam::Quat::from_rotation_y(consts::PI)
                } else {
                    glam::Quat::IDENTITY
                };

                let wheel_handle = engine.add_object(
                    &wheel_config,
                    blade::Transform {
                        position: (spawn_pos + offset).into(),
                        orientation: rotation.into(),
                    },
                    blade::DynamicInput::Full,
                );
                let wheel_angular_freedoms = mint::Vector3 {
                    x: Some(blade::FreedomAxis {
                        limits: None,
                        motor: Some(blade::config::Motor {
                            stiffness: 0.0,
                            damping: veh_config.drive_factor,
                            max_force: 1000.0,
                        }),
                    }),
                    y: None,
                    z: None,
                };

                vehicle.wheels.push(
                    if ac.max_steering_angle > 0.0 || ac.max_suspension_offset > 0.0 {
                        let max_angle = ac.max_steering_angle.to_radians();
                        let suspender_handle = engine.add_object(
                            &suspender_config,
                            blade::Transform {
                                position: (spawn_pos + offset).into(),
                                ..Default::default()
                            },
                            blade::DynamicInput::Full,
                        );

                        let suspension_joint = engine.add_joint(
                            vehicle.body_handle,
                            suspender_handle,
                            blade::JointDesc {
                                parent_anchor: blade::Transform {
                                    position: offset.into(),
                                    ..Default::default()
                                },
                                linear: mint::Vector3 {
                                    x: None,
                                    y: if ac.max_suspension_offset > 0.0 {
                                        Some(blade::FreedomAxis {
                                            limits: Some(0.0..ac.max_suspension_offset),
                                            motor: Some(ac.suspension),
                                        })
                                    } else {
                                        None
                                    },
                                    z: None,
                                },
                                angular: mint::Vector3 {
                                    x: None,
                                    y: if ac.max_steering_angle > 0.0 {
                                        Some(blade::FreedomAxis {
                                            limits: Some(-max_angle..max_angle),
                                            motor: Some(ac.steering),
                                        })
                                    } else {
                                        None
                                    },
                                    z: None,
                                },
                                ..Default::default()
                            },
                        );

                        let wheel_joint = engine.add_joint(
                            suspender_handle,
                            wheel_handle,
                            blade::JointDesc {
                                child_anchor: blade::Transform {
                                    orientation: rotation.into(),
                                    ..Default::default()
                                },
                                angular: wheel_angular_freedoms,
                                ..Default::default()
                            },
                        );

                        let _extra_joint = engine.add_joint(
                            vehicle.body_handle,
                            wheel_handle,
                            blade::JointDesc {
                                linear: blade::FreedomAxis::ALL_FREE,
                                angular: blade::FreedomAxis::ALL_FREE,
                                ..Default::default()
                            },
                        );

                        Wheel {
                            object: wheel_handle,
                            spin_joint: wheel_joint,
                            suspender: Some(suspender_handle),
                            steer_joint: if ac.max_steering_angle > 0.0 {
                                Some(suspension_joint)
                            } else {
                                None
                            },
                        }
                    } else {
                        let wheel_joint = engine.add_joint(
                            vehicle.body_handle,
                            wheel_handle,
                            blade::JointDesc {
                                parent_anchor: blade::Transform {
                                    position: offset.into(),
                                    ..Default::default()
                                },
                                child_anchor: blade::Transform {
                                    orientation: rotation.into(),
                                    ..Default::default()
                                },
                                angular: wheel_angular_freedoms,
                                ..Default::default()
                            },
                        );

                        Wheel {
                            object: wheel_handle,
                            spin_joint: wheel_joint,
                            suspender: None,
                            steer_joint: None,
                        }
                    },
                );
            }
        }

        let egui_context = egui::Context::default();
        let egui_viewport_id = egui_context.viewport_id();
        let egui_state =
            egui_winit::State::new(egui_context, egui_viewport_id, &window, None, None, None);

        Self {
            engine,
            last_physics_update: time::Instant::now(),
            last_camera_update: time::Instant::now(),
            last_camera_base_quat: Default::default(),
            is_paused: false,
            window,
            egui_state,
            egui_viewport_id,
            _ground_handle: ground_handle,
            vehicle,
            cam_config,
            spawn_pos,
        }
    }

    fn set_velocity(&mut self, velocity: f32) {
        self.engine.wake_up(self.vehicle.body_handle);
        self.update_time();
        for wheel in self.vehicle.wheels.iter() {
            self.engine.set_joint_motor(
                wheel.spin_joint,
                blade::JointAxis::AngularX,
                0.0,
                velocity,
            );
        }
    }

    fn set_steering(&mut self, angle_rad: f32) {
        self.update_time();
        for wheel in self.vehicle.wheels.iter() {
            if let Some(handle) = wheel.steer_joint {
                self.engine
                    .set_joint_motor(handle, blade::JointAxis::AngularY, angle_rad, 0.0);
            }
        }
    }

    fn teleport_object_rel(&mut self, handle: blade::ObjectHandle, isometry: &Isometry) {
        let prev_transform = self
            .engine
            .get_object_transform(handle, blade::Prediction::LastKnown);
        let next = isometry.clone() * Isometry::from(prev_transform);
        self.engine.teleport_object(handle, next.to_blade());
    }

    fn teleport(&mut self, position: glam::Vec3) {
        let old_transform = self
            .engine
            .get_object_transform(self.vehicle.body_handle, blade::Prediction::LastKnown);
        let old_isometry_inv = Isometry::from(old_transform).inverse();
        let new_transform = blade::Transform {
            position: position.into(),
            ..Default::default()
        };
        self.engine
            .teleport_object(self.vehicle.body_handle, new_transform.clone());

        let relative = Isometry::from(new_transform) * old_isometry_inv;
        let wheels = mem::take(&mut self.vehicle.wheels);
        for wheel in wheels.iter() {
            if let Some(suspender) = wheel.suspender {
                self.teleport_object_rel(suspender, &relative);
            }
            self.teleport_object_rel(wheel.object, &relative);
        }
        self.vehicle.wheels = wheels;
    }

    fn update_time(&mut self) {
        let engine_dt = self.last_physics_update.elapsed().as_secs_f32();
        self.last_physics_update = time::Instant::now();
        if !self.is_paused {
            //self.align_wheels();
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

        match *event {
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
                    return Err(QuitEvent);
                }
                winit::keyboard::KeyCode::ArrowUp => {
                    self.set_velocity(100.0);
                }
                winit::keyboard::KeyCode::ArrowDown => {
                    self.set_velocity(-20.0);
                }
                winit::keyboard::KeyCode::ArrowLeft => {
                    self.set_steering(1.0);
                }
                winit::keyboard::KeyCode::ArrowRight => {
                    self.set_steering(-1.0);
                }
                winit::keyboard::KeyCode::Comma => {
                    let transform = self.engine.get_object_transform(
                        self.vehicle.body_handle,
                        blade::Prediction::LastKnown,
                    );
                    let forward = glam::Quat::from(transform.orientation) * glam::Vec3::Z;
                    self.engine.apply_angular_impulse(
                        self.vehicle.body_handle,
                        (-self.vehicle.roll_impulse * forward).into(),
                    );
                }
                winit::keyboard::KeyCode::Period => {
                    let transform = self.engine.get_object_transform(
                        self.vehicle.body_handle,
                        blade::Prediction::LastKnown,
                    );
                    let forward = glam::Quat::from(transform.orientation) * glam::Vec3::Z;
                    self.engine.apply_angular_impulse(
                        self.vehicle.body_handle,
                        (self.vehicle.roll_impulse * forward).into(),
                    );
                }
                winit::keyboard::KeyCode::Space => {
                    let transform = self.engine.get_object_transform(
                        self.vehicle.body_handle,
                        blade::Prediction::LastKnown,
                    );
                    let mut up = glam::Quat::from(transform.orientation) * glam::Vec3::Y;
                    up.y = up.y.abs();
                    self.engine.apply_linear_impulse(
                        self.vehicle.body_handle,
                        (self.vehicle.jump_impulse * up).into(),
                    );
                }
                _ => {}
            },
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                        state: winit::event::ElementState::Released,
                        ..
                    },
                ..
            } => match key_code {
                winit::keyboard::KeyCode::ArrowUp | winit::keyboard::KeyCode::ArrowDown => {
                    self.set_velocity(0.0);
                }
                winit::keyboard::KeyCode::ArrowLeft | winit::keyboard::KeyCode::ArrowRight => {
                    self.set_steering(0.0);
                }
                _ => {}
            },
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
            _ => {}
        }

        Ok(winit::event_loop::ControlFlow::Poll)
    }

    fn populate_hud(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Camera")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.cam_config.distance, 1.0..=1000.0)
                        .text("Distance")
                        .logarithmic(true),
                );
                ui.horizontal(|ui| {
                    ui.label("Target");
                    ui.add(egui::DragValue::new(&mut self.cam_config.target[1]));
                    ui.add(egui::DragValue::new(&mut self.cam_config.target[2]));
                });
                ui.horizontal(|ui| {
                    let eps = 0.01;
                    ui.label("Angle");
                    ui.add(
                        egui::DragValue::new(&mut self.cam_config.azimuth)
                            .range(-consts::PI..=consts::PI)
                            .speed(0.1),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.cam_config.altitude)
                            .range(eps..=consts::FRAC_PI_2 - eps)
                            .speed(0.1),
                    );
                });
                ui.add(egui::Slider::new(&mut self.cam_config.fov, 0.5f32..=2.0f32).text("FOV"));
                ui.add(
                    egui::Slider::new(&mut self.cam_config.speed, 0.0..=1.0).text("Rotate speed"),
                );
                ui.toggle_value(&mut self.is_paused, "Pause");
            });

        egui::CollapsingHeader::new("Dynamics")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Spawn pos");
                    ui.add(egui::DragValue::new(&mut self.spawn_pos.x));
                    ui.add(egui::DragValue::new(&mut self.spawn_pos.y));
                    ui.add(egui::DragValue::new(&mut self.spawn_pos.z));
                });
                ui.horizontal(|ui| {
                    if ui.button("Recover").clicked() {
                        let transform = self.engine.get_object_transform(
                            self.vehicle.body_handle,
                            blade::Prediction::LastKnown,
                        );
                        let pos = glam::Vec3::from(transform.position);
                        let bounds = self.engine.get_object_bounds(self.vehicle.body_handle);
                        self.teleport(pos + glam::Vec3::from(bounds.half) * glam::Vec3::Y);
                    }
                    if ui.button("Respawn").clicked() {
                        self.teleport(self.spawn_pos);
                    }
                });
                ui.add(
                    egui::DragValue::new(&mut self.vehicle.jump_impulse).prefix("Jump impulse: "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.vehicle.roll_impulse).prefix("Roll impulse: "),
                );
            });

        self.engine.populate_hud(ui);
    }

    fn on_draw(&mut self) -> time::Duration {
        self.update_time();

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

        let camera = {
            let veh_transform = self.engine.get_object_transform(
                self.vehicle.body_handle,
                blade::Prediction::IntegrateVelocityAndForces,
            );
            let veh_isometry = Isometry::from(veh_transform);
            // Projection of the rotation of the vehicle on the Y axis
            let projection = veh_isometry.orientation.xyz().dot(glam::Vec3::Y);
            let base_quat_nonorm =
                glam::Quat::from_xyzw(0.0, projection, 0.0, veh_isometry.orientation.w);
            let validity = base_quat_nonorm.length();
            let base_quat = base_quat_nonorm / validity;

            let camera_dt = self.last_camera_update.elapsed().as_secs_f32();
            self.last_physics_update = time::Instant::now();

            let cc = &self.cam_config;
            let smooth_t = (-camera_dt * cc.speed * validity).exp();
            let smooth_quat = base_quat.lerp(self.last_camera_base_quat, smooth_t);
            let base = Isometry {
                position: veh_isometry.position,
                orientation: smooth_quat,
            };
            self.last_camera_base_quat = smooth_quat;

            let source = glam::Vec3::from(cc.target)
                + cc.distance
                    * glam::Vec3::new(
                        -cc.azimuth.sin() * cc.altitude.cos(),
                        cc.altitude.sin(),
                        -cc.azimuth.cos() * cc.altitude.cos(),
                    );
            let local_affine = glam::Affine3A::look_at_rh(source, cc.target.into(), glam::Vec3::Y);
            let local = Isometry {
                position: local_affine.translation.into(),
                orientation: glam::Quat::from_affine3(&local_affine),
            };
            blade::FrameCamera {
                transform: (base * local.inverse()).to_blade(),
                fov_y: cc.fov,
            }
        };

        let primitives = self
            .egui_state
            .egui_ctx()
            .tessellate(egui_output.shapes, egui_output.pixels_per_point);
        self.engine.render(
            &camera,
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
