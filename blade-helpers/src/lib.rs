#![cfg(not(any(gles, target_arch = "wasm32")))]

mod camera;
mod hud;

pub use blade_render::Camera;
pub use camera::ControlledCamera;
pub use hud::{populate_debug_selection, ExposeHud};

pub fn default_ray_config() -> blade_render::RayConfig {
    blade_render::RayConfig {
        num_environment_samples: 1,
        environment_importance_sampling: false,
        tap_count: 2,
        tap_radius: 20,
        tap_confidence_near: 15,
        tap_confidence_far: 10,
        t_start: 0.01,
        pairwise_mis: true,
        defensive_mis: 0.1,
    }
}
