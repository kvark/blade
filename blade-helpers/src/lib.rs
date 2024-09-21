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
        temporal_tap: true,
        temporal_history: 10,
        spatial_taps: 1,
        spatial_tap_history: 10,
        spatial_radius: 20,
        t_start: 0.01,
        pairwise_mis: true,
        defensive_mis: 0.1,
    }
}
