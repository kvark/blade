//! The asset hub and the shaders of the renderer, for the tests that
//! render scenes, shared by the test targets that include this module.
#![allow(dead_code)]

use super::{SHADER_DIR, pbr_scene};
use blade_graphics as gpu;

pub(crate) struct PbrHarness {
    pub(crate) context: std::sync::Arc<gpu::Context>,
    pub(crate) choir: std::sync::Arc<choir::Choir>,
    pub(crate) workers: Vec<choir::WorkerHandle>,
    pub(crate) asset_hub: blade_render::AssetHub,
    pub(crate) shaders: blade_render::Shaders,
}

impl PbrHarness {
    /// Bring up the asset hub and cook the renderer shaders.
    pub(crate) fn new(context: gpu::Context, cache_name: &str, ray_tracing: bool) -> Self {
        let context = std::sync::Arc::new(context);
        let choir = choir::Choir::new();
        #[cfg(not(any(gles, target_arch = "wasm32")))]
        let workers = (0..2)
            .map(|i| choir.add_worker(&format!("{cache_name}-{i}")))
            .collect::<Vec<_>>();
        #[cfg(any(gles, target_arch = "wasm32"))]
        let workers = Vec::new();
        let cache_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-assets")
            .join(cache_name);
        let asset_hub = blade_render::AssetHub::new(&cache_path, &choir, &context);
        let (shaders, shader_task) =
            blade_render::Shaders::load(SHADER_DIR.as_ref(), &asset_hub, ray_tracing);
        if workers.is_empty() {
            shader_task.join_active();
        } else {
            shader_task.join();
        }
        Self {
            context,
            choir,
            workers,
            asset_hub,
            shaders,
        }
    }

    pub(crate) fn create_grid_model(
        &self,
        roughness_range: [f32; 2],
    ) -> blade_asset::Handle<blade_render::Model> {
        let geometries = pbr_scene::material_grid(roughness_range);
        let model = self
            .asset_hub
            .models
            .baker
            .create_model("pbr-material-grid", geometries);
        self.asset_hub.models.insert(model)
    }

    pub(crate) fn destroy(mut self) {
        self.asset_hub.destroy();
        // let the workers finish before the choir goes away
        self.workers.clear();
        drop(self.choir);
    }
}
