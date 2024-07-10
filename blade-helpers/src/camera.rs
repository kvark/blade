pub struct ControlledCamera {
    pub inner: blade_render::Camera,
    pub fly_speed: f32,
}

impl Default for ControlledCamera {
    fn default() -> Self {
        Self {
            inner: blade_render::Camera {
                pos: mint::Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                rot: mint::Quaternion {
                    v: mint::Vector3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    s: 1.0,
                },
                fov_y: 0.0,
                depth: 0.0,
            },
            fly_speed: 0.0,
        }
    }
}

impl ControlledCamera {
    pub fn get_view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_rotation_translation(self.inner.rot.into(), self.inner.pos.into())
            .inverse()
    }

    pub fn get_projection_matrix(&self, aspect: f32) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.inner.fov_y, aspect, 1.0, self.inner.depth)
    }

    pub fn populate_hud(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Position:");
            ui.add(egui::DragValue::new(&mut self.inner.pos.x));
            ui.add(egui::DragValue::new(&mut self.inner.pos.y));
            ui.add(egui::DragValue::new(&mut self.inner.pos.z));
        });
        ui.horizontal(|ui| {
            ui.label("Rotation:");
            ui.add(egui::DragValue::new(&mut self.inner.rot.v.x));
            ui.add(egui::DragValue::new(&mut self.inner.rot.v.y));
            ui.add(egui::DragValue::new(&mut self.inner.rot.v.z));
            ui.add(egui::DragValue::new(&mut self.inner.rot.s));
        });
        ui.add(egui::Slider::new(&mut self.inner.fov_y, 0.5f32..=2.0f32).text("FOV"));
        ui.add(
            egui::Slider::new(&mut self.fly_speed, 1f32..=100000f32)
                .text("Fly speed")
                .logarithmic(true),
        );
    }

    pub fn move_by(&mut self, offset: glam::Vec3) {
        let dir = glam::Quat::from(self.inner.rot) * offset;
        self.inner.pos = (glam::Vec3::from(self.inner.pos) + dir).into();
    }

    pub fn rotate_z_by(&mut self, angle: f32) {
        let quat = glam::Quat::from(self.inner.rot);
        let rotation = glam::Quat::from_rotation_z(angle);
        self.inner.rot = (quat * rotation).into();
    }

    pub fn on_key(&mut self, code: winit::keyboard::KeyCode, delta: f32) -> bool {
        use winit::keyboard::KeyCode as Kc;

        let move_offset = self.fly_speed * delta;
        let rotate_offset_z = 1000.0 * delta;
        match code {
            Kc::KeyW => {
                self.move_by(glam::Vec3::new(0.0, 0.0, -move_offset));
            }
            Kc::KeyS => {
                self.move_by(glam::Vec3::new(0.0, 0.0, move_offset));
            }
            Kc::KeyA => {
                self.move_by(glam::Vec3::new(-move_offset, 0.0, 0.0));
            }
            Kc::KeyD => {
                self.move_by(glam::Vec3::new(move_offset, 0.0, 0.0));
            }
            Kc::KeyZ => {
                self.move_by(glam::Vec3::new(0.0, -move_offset, 0.0));
            }
            Kc::KeyX => {
                self.move_by(glam::Vec3::new(0.0, move_offset, 0.0));
            }
            Kc::KeyQ => {
                self.rotate_z_by(rotate_offset_z);
            }
            Kc::KeyE => {
                self.rotate_z_by(-rotate_offset_z);
            }
            _ => return false,
        }

        true
    }
}
