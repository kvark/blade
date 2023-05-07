use blade_asset::AssetManager;
use std::{path::Path, sync::Arc};

pub struct AssetHub {
    pub textures: AssetManager<crate::texture::Baker>,
}

impl AssetHub {
    pub fn new(
        root: &Path,
        target: &Path,
        choir: &Arc<choir::Choir>,
        gpu_context: &Arc<blade::Context>,
    ) -> Self {
        let _ = std::fs::create_dir_all(target);
        Self {
            textures: AssetManager::new(
                root,
                target,
                choir,
                crate::texture::Baker::new(gpu_context),
            ),
        }
    }

    pub fn flush(
        &self,
        command_encoder: &mut blade::CommandEncoder,
        temp_buffers: &mut Vec<blade::Buffer>,
    ) {
        let mut transfers = command_encoder.transfer();
        self.textures.baker.flush(&mut transfers, temp_buffers);
    }
}
