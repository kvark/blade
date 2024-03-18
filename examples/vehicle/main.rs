mod config;

use std::{f32::consts, fs, mem, path::PathBuf, time};

#[derive(Clone)]
struct Wheel {
    object: blade::ObjectHandle,
    spin_joint: blade::JointHandle,
    suspender: Option<blade::ObjectHandle>,
    steer_joint: Option<blade::JointHandle>,
}

struct WheelAxle {
    wheels: Vec<Wheel>,
    steer: Option<config::Motor>,
}

struct Vehicle {
    body_handle: blade::ObjectHandle,
    drive_factor: f32,
    jump_impulse: f32,
    roll_impulse: f32,
    wheel_axles: Vec<WheelAxle>,
}

struct Game {
    // engine stuff
    engine: blade::Engine,
    last_physics_update: time::Instant,
    last_camera_update: time::Instant,
    last_camera_base_quat: nalgebra::UnitQuaternion<f32>,
    is_paused: bool,
    // windowing
    window: winit::window::Window,
    egui_state: egui_winit::State,
    egui_viewport_id: egui::ViewportId,
    // game data
    _ground_handle: blade::ObjectHandle,
    vehicle: Vehicle,
    cam_config: config::Camera,
    spawn_pos: nalgebra::Vector3<f32>,
}

const SUSPENSION_AXIS: rapier3d::dynamics::JointAxis = rapier3d::dynamics::JointAxis::Y;
const STEERING_AXIS: rapier3d::dynamics::JointAxis = rapier3d::dynamics::JointAxis::AngY;
const SPIN_AXIS: rapier3d::dynamics::JointAxis = rapier3d::dynamics::JointAxis::AngX;

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
            .with_title("RayCraft")
            .build(event_loop)
            .unwrap();

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
            nalgebra::Isometry3::default(),
            blade::BodyType::Fixed,
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
        let spawn_pos = nalgebra::Vector3::from(lev_config.spawn_pos);
        let mut vehicle = Vehicle {
            body_handle: engine.add_object(
                &body_config,
                nalgebra::Isometry3::new(spawn_pos, nalgebra::Vector3::zeros()),
                blade::BodyType::Dynamic,
            ),
            drive_factor: veh_config.drive_factor,
            jump_impulse: veh_config.jump_impulse,
            roll_impulse: veh_config.roll_impulse,
            wheel_axles: Vec::new(),
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
            let joint_kind = blade::JointKind::Soft;
            let mut wheels = Vec::new();

            for wheel_x in ac.x_wheels {
                let offset = nalgebra::Vector3::new(wheel_x, ac.y, ac.z);
                let rotation = if wheel_x > 0.0 {
                    nalgebra::Vector3::y_axis().scale(consts::PI)
                } else {
                    nalgebra::Vector3::zeros()
                };

                let wheel_handle = engine.add_object(
                    &wheel_config,
                    nalgebra::Isometry3::new(spawn_pos + offset, rotation),
                    blade::BodyType::Dynamic,
                );

                wheels.push(if ac.steering.limit > 0.0 || ac.suspension.limit > 0.0 {
                    let max_angle = ac.steering.limit.to_radians();
                    let mut locked_axes = rapier3d::dynamics::JointAxesMask::LOCKED_FIXED_AXES;
                    if ac.steering.limit > 0.0 {
                        locked_axes ^= STEERING_AXIS.into();
                    }
                    if ac.suspension.limit > 0.0 {
                        locked_axes ^= SUSPENSION_AXIS.into();
                    }

                    let suspender_handle = engine.add_object(
                        &suspender_config,
                        nalgebra::Isometry3::new(spawn_pos + offset, nalgebra::Vector3::zeros()),
                        blade::BodyType::Dynamic,
                    );

                    let suspension_joint = engine.add_joint(
                        vehicle.body_handle,
                        suspender_handle,
                        rapier3d::dynamics::GenericJointBuilder::new(locked_axes)
                            .contacts_enabled(false)
                            .local_anchor1(offset.into())
                            .limits(SUSPENSION_AXIS, [0.0, ac.suspension.limit])
                            .motor_position(
                                SUSPENSION_AXIS,
                                0.0,
                                ac.suspension.stiffness,
                                ac.suspension.damping,
                            )
                            .limits(STEERING_AXIS, [-max_angle, max_angle])
                            .motor_position(
                                STEERING_AXIS,
                                0.0,
                                ac.steering.stiffness,
                                ac.steering.damping,
                            )
                            .build(),
                        joint_kind,
                    );

                    let wheel_joint = engine.add_joint(
                        suspender_handle,
                        wheel_handle,
                        rapier3d::dynamics::GenericJointBuilder::new(
                            rapier3d::dynamics::JointAxesMask::LOCKED_REVOLUTE_AXES,
                        )
                        .contacts_enabled(false)
                        .local_frame2(nalgebra::Isometry3::rotation(rotation))
                        .build(),
                        joint_kind,
                    );

                    let _extra_joint = engine.add_joint(
                        vehicle.body_handle,
                        wheel_handle,
                        rapier3d::dynamics::GenericJoint {
                            contacts_enabled: false,
                            ..Default::default()
                        },
                        joint_kind,
                    );

                    Wheel {
                        object: wheel_handle,
                        spin_joint: wheel_joint,
                        suspender: Some(suspender_handle),
                        steer_joint: Some(suspension_joint),
                    }
                } else {
                    let locked_axes =
                        rapier3d::dynamics::JointAxesMask::LOCKED_FIXED_AXES ^ SPIN_AXIS.into();

                    let wheel_joint = engine.add_joint(
                        vehicle.body_handle,
                        wheel_handle,
                        rapier3d::dynamics::GenericJointBuilder::new(locked_axes)
                            .contacts_enabled(false)
                            .local_anchor1(offset.into())
                            .local_frame2(nalgebra::Isometry3::rotation(rotation))
                            .build(),
                        joint_kind,
                    );

                    Wheel {
                        object: wheel_handle,
                        spin_joint: wheel_joint,
                        suspender: None,
                        steer_joint: None,
                    }
                });
            }

            vehicle.wheel_axles.push(WheelAxle {
                wheels,
                steer: if ac.steering.limit > 0.0 {
                    Some(ac.steering)
                } else {
                    None
                },
            });
        }

        let egui_context = egui::Context::default();
        let egui_viewport_id = egui_context.viewport_id();
        let egui_state =
            egui_winit::State::new(egui_context, egui_viewport_id, &window, None, None);

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
        for wax in self.vehicle.wheel_axles.iter() {
            for wheel in wax.wheels.iter() {
                self.engine[wheel.spin_joint].set_motor_velocity(
                    SPIN_AXIS,
                    velocity,
                    self.vehicle.drive_factor,
                );
            }
        }
    }

    fn set_steering(&mut self, angle_rad: f32) {
        self.update_time();
        for wax in self.vehicle.wheel_axles.iter() {
            let steer = match wax.steer {
                Some(ref steer) => steer,
                None => continue,
            };
            for wheel in wax.wheels.iter() {
                if let Some(handle) = wheel.steer_joint {
                    self.engine[handle].set_motor_position(
                        STEERING_AXIS,
                        angle_rad,
                        steer.stiffness,
                        steer.damping,
                    );
                }
            }
        }
    }

    fn teleport_object_rel(
        &mut self,
        handle: blade::ObjectHandle,
        transform: &nalgebra::Isometry3<f32>,
    ) {
        let prev = self.engine.get_object_isometry_approx(handle);
        let next = transform * prev;
        self.engine.teleport_object(handle, next);
    }

    fn teleport(&mut self, position: nalgebra::Vector3<f32>) {
        let old_isometry_inv = self
            .engine
            .get_object_isometry_approx(self.vehicle.body_handle)
            .inverse();
        let new_isometry = nalgebra::Isometry3 {
            rotation: Default::default(),
            translation: position.into(),
        };
        self.engine
            .teleport_object(self.vehicle.body_handle, new_isometry);

        let relative = new_isometry * old_isometry_inv;
        let waxes = mem::take(&mut self.vehicle.wheel_axles);
        for wax in waxes.iter() {
            for wheel in wax.wheels.iter() {
                if let Some(suspender) = wheel.suspender {
                    self.teleport_object_rel(suspender, &relative);
                }
                self.teleport_object_rel(wheel.object, &relative);
            }
        }
        self.vehicle.wheel_axles = waxes;
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
                    let forward = self
                        .engine
                        .get_object_isometry_approx(self.vehicle.body_handle)
                        .transform_vector(&nalgebra::Vector3::z_axis());
                    self.engine.apply_torque_impulse(
                        self.vehicle.body_handle,
                        -self.vehicle.roll_impulse * forward,
                    );
                }
                winit::keyboard::KeyCode::Period => {
                    let forward = self
                        .engine
                        .get_object_isometry_approx(self.vehicle.body_handle)
                        .transform_vector(&nalgebra::Vector3::z_axis());
                    self.engine.apply_torque_impulse(
                        self.vehicle.body_handle,
                        self.vehicle.roll_impulse * forward,
                    );
                }
                winit::keyboard::KeyCode::Space => {
                    let mut up = self
                        .engine
                        .get_object_isometry_approx(self.vehicle.body_handle)
                        .transform_vector(&nalgebra::Vector3::y_axis());
                    up.y = up.y.abs();
                    self.engine
                        .apply_impulse(self.vehicle.body_handle, self.vehicle.jump_impulse * up);
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
                            .clamp_range(-consts::PI..=consts::PI)
                            .speed(0.1),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.cam_config.altitude)
                            .clamp_range(eps..=consts::FRAC_PI_2 - eps)
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
                        let pos = self
                            .engine
                            .get_object_isometry_approx(self.vehicle.body_handle)
                            .translation
                            .vector;
                        let bounds = self.engine.get_object_bounds(self.vehicle.body_handle);
                        self.teleport(
                            pos + bounds
                                .half_extents()
                                .component_mul(&nalgebra::Vector3::y_axis()),
                        );
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
            let veh_isometry = self.engine.get_object_isometry(self.vehicle.body_handle);
            // Projection of the rotation of the vehicle on the Y axis
            let projection = veh_isometry
                .rotation
                .quaternion()
                .imag()
                .dot(&nalgebra::Vector3::y_axis());
            let base_quat_nonorm = nalgebra::Quaternion::from_parts(
                veh_isometry.rotation.quaternion().w,
                nalgebra::Vector3::y_axis().scale(projection),
            );
            let validity = base_quat_nonorm.norm();
            let base_quat = nalgebra::UnitQuaternion::new_normalize(base_quat_nonorm);

            let camera_dt = self.last_camera_update.elapsed().as_secs_f32();
            self.last_physics_update = time::Instant::now();

            let cc = &self.cam_config;
            let smooth_t = (-camera_dt * cc.speed * validity).exp();
            let smooth_quat = nalgebra::UnitQuaternion::new_normalize(
                base_quat.lerp(&self.last_camera_base_quat, smooth_t),
            );
            let base =
                nalgebra::geometry::Isometry3::from_parts(veh_isometry.translation, smooth_quat);
            self.last_camera_base_quat = smooth_quat;

            //TODO: `nalgebra::Point3::from(mint::Vector3)` doesn't exist?
            let source = nalgebra::Vector3::from(cc.target)
                + nalgebra::Vector3::new(
                    -cc.azimuth.sin() * cc.altitude.cos(),
                    cc.altitude.sin(),
                    -cc.azimuth.cos() * cc.altitude.cos(),
                )
                .scale(cc.distance);
            let local = nalgebra::geometry::Isometry3::look_at_rh(
                &source.into(),
                &nalgebra::Vector3::from(cc.target).into(),
                &nalgebra::Vector3::y_axis(),
            );
            blade::Camera {
                isometry: base * local.inverse(),
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
