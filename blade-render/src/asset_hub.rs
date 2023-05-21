use blade_asset::AssetManager;
use std::{path::Path, sync::Arc};

pub struct AssetHub {
    pub textures: Arc<AssetManager<crate::texture::Baker>>,
    pub models: Arc<AssetManager<crate::model::Baker>>,
}

impl AssetHub {
    pub fn new(
        root: &Path,
        target: &Path,
        choir: &Arc<choir::Choir>,
        gpu_context: &Arc<blade::Context>,
    ) -> Self {
        let _ = std::fs::create_dir_all(target);
        let textures = Arc::new(AssetManager::new(
            root,
            target,
            choir,
            crate::texture::Baker::new(gpu_context),
        ));
        let models = Arc::new(AssetManager::new(
            root,
            target,
            choir,
            crate::model::Baker::new(gpu_context, &textures),
        ));
        Self { textures, models }
    }

    pub fn flush(
        &self,
        command_encoder: &mut blade::CommandEncoder,
        temp_buffers: &mut Vec<blade::Buffer>,
    ) {
        self.textures.baker.flush(command_encoder, temp_buffers);
        self.models.baker.flush(command_encoder, temp_buffers);
    }

    pub fn destroy(&mut self) {
        self.textures.clear();
        self.models.clear();
    }
}
