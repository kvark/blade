use blade_asset::AssetManager;
use std::{path::Path, sync::Arc};

/// A single hub to manage all assets.
pub struct AssetHub {
    pub textures: Arc<AssetManager<crate::texture::Baker>>,
    pub models: AssetManager<crate::model::Baker>,
    pub shaders: AssetManager<crate::shader::Baker>,
}

impl AssetHub {
    /// Create a new hub.
    pub fn new(
        root: &Path,
        target: &Path,
        choir: &Arc<choir::Choir>,
        gpu_context: &Arc<blade_graphics::Context>,
    ) -> Self {
        let _ = std::fs::create_dir_all(target);
        let textures = Arc::new(AssetManager::new(
            root,
            target,
            choir,
            crate::texture::Baker::new(gpu_context),
        ));
        let models = AssetManager::new(
            root,
            target,
            choir,
            crate::model::Baker::new(gpu_context, &textures),
        );
        let shaders = AssetManager::new(
            "blade-render/code/".as_ref(),
            target,
            choir,
            crate::shader::Baker::new(gpu_context),
        );
        Self {
            textures,
            models,
            shaders,
        }
    }

    /// Flush the GPU state updates into the specified command encoder.
    ///
    /// Populates the list of temporary buffers that can be freed when the
    /// relevant submission is completely retired.
    pub fn flush(
        &self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        temp_buffers: &mut Vec<blade_graphics::Buffer>,
    ) {
        self.textures.baker.flush(command_encoder, temp_buffers);
        self.models.baker.flush(command_encoder, temp_buffers);
    }

    /// Destroy the hub contents.
    pub fn destroy(&mut self) {
        self.textures.clear();
        self.models.clear();
        self.shaders.clear();
    }
}
