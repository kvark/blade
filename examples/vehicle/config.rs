#[derive(serde::Deserialize)]
pub struct Body {
    pub visual: blade_engine::config::Visual,
    pub collider: blade_engine::config::Collider,
}

#[derive(serde::Deserialize)]
pub struct Wheel {
    pub visual: blade_engine::config::Visual,
    pub collider: blade_engine::config::Collider,
}

#[derive(serde::Deserialize)]
pub struct Axle {
    /// Side offset for each wheel.
    pub x_wheels: Vec<f32>,
    /// Height offset from the body.
    pub y: f32,
    /// Forward offset from the body.
    pub z: f32,
    #[serde(default)]
    pub max_steering_angle: f32,
    #[serde(default)]
    pub max_suspension_offset: f32,
    #[serde(default)]
    pub suspension: blade_engine::config::Motor,
    #[serde(default)]
    pub steering: blade_engine::config::Motor,
}

fn default_additional_mass() -> blade_engine::config::AdditionalMass {
    blade_engine::config::AdditionalMass {
        density: 0.0,
        shape: blade_engine::config::Shape::Ball { radius: 0.0 },
    }
}

#[derive(serde::Deserialize)]
pub struct Vehicle {
    pub body: Body,
    pub wheel: Wheel,
    #[serde(default = "default_additional_mass")]
    pub suspender: blade_engine::config::AdditionalMass,
    pub drive_factor: f32,
    pub jump_impulse: f32,
    pub roll_impulse: f32,
    pub axles: Vec<Axle>,
}

#[derive(serde::Deserialize)]
pub struct Level {
    #[serde(default)]
    pub environment: String,
    pub gravity: f32,
    pub average_luminocity: f32,
    pub spawn_pos: [f32; 3],
    pub ground: blade_engine::config::Object,
}

#[derive(serde::Deserialize)]
pub struct Camera {
    pub azimuth: f32,
    pub altitude: f32,
    pub distance: f32,
    pub speed: f32,
    pub target: [f32; 3],
    pub fov: f32,
}
