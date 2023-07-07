use blade_asset::AssetManager;
use std::{path::Path, sync::Arc};

/// A single hub to manage all assets.
pub struct AssetHub {
    pub textures: Arc<AssetManager<crate::texture::Baker>>,
    pub models: AssetManager<crate::model::Baker>,
    pub shaders: AssetManager<crate::shader::Baker>,
}

pub struct LoadContext<'a> {
    asset_hub: &'a AssetHub,
    base_path: &'a Path,
    finish_task: choir::IdleTask,
}

impl AssetHub {
    /// Create a new hub.
    pub fn new(
        target: &Path,
        choir: &Arc<choir::Choir>,
        gpu_context: &Arc<blade_graphics::Context>,
    ) -> Self {
        let _ = std::fs::create_dir_all(target);
        let textures = Arc::new(AssetManager::new(
            target,
            choir,
            crate::texture::Baker::new(gpu_context),
        ));
        let models = AssetManager::new(
            target,
            choir,
            crate::model::Baker::new(gpu_context, &textures),
        );

        let mut sh_baker = crate::shader::Baker::new(gpu_context);
        sh_baker.register_enum::<crate::render::DebugMode>();
        sh_baker.register_bitflags::<crate::render::DebugFlags>();
        let shaders = AssetManager::new(target, choir, sh_baker);

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

    pub fn open_context<'a, N: Into<choir::Name>>(
        &'a self,
        base_path: &'a Path,
        name: N,
    ) -> LoadContext {
        LoadContext {
            asset_hub: self,
            base_path,
            finish_task: self.shaders.choir.spawn(name).init_dummy(),
        }
    }
}

impl LoadContext<'_> {
    pub fn load_shader(&mut self, path: &str) -> blade_asset::Handle<crate::Shader> {
        let (handle, task) = self
            .asset_hub
            .shaders
            .load(self.base_path.join(path), crate::shader::Meta);
        self.finish_task.depend_on(task);
        handle
    }

    pub fn close(self) -> choir::RunningTask {
        self.finish_task.run()
    }
}
