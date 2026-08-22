mod camera;
mod hud;

pub use blade_render::Camera;
pub use camera::ControlledCamera;
pub use hud::ExposeHud;
pub use hud::{populate_debug_selection, populate_render_mode};

pub fn default_ray_config() -> blade_render::RayConfig {
    blade_render::RayConfig {
        num_environment_samples: 1,
        num_brdf_samples: 1,
        jitter_primary_rays: true,
        environment_importance_sampling: true,
        max_bounces: 3,
        max_accumulated_samples: 0,
        tap_count: 2,
        tap_radius: 20,
        tap_confidence_near: 15,
        tap_confidence_far: 10,
        t_start: 0.01,
        pairwise_mis: true,
        defensive_mis: 0.1,
    }
}
