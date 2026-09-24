use std::path::Path;

use crate::AssetHub;

/// Configuration shared by raster and ray-traced renderers.
#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    pub surface_size: blade_graphics::Extent,
    pub surface_info: blade_graphics::SurfaceInfo,
    /// Color space to produce the image in, matching the one the
    /// surface was configured with.
    ///
    /// `Linear` leaves the encoding to the platform, which is what an sRGB
    /// surface format does for us. `Srgb` means the values are passed to the
    /// display as they are, so we have to encode them ourselves.
    pub color_space: blade_graphics::ColorSpace,
    pub max_debug_lines: u32,
}

#[derive(Clone, PartialEq)]
pub struct Shaders {
    pub(crate) env_prepare: blade_asset::Handle<crate::Shader>,
    pub(crate) fill_gbuf: blade_asset::Handle<crate::Shader>,
    pub(crate) ray_trace: blade_asset::Handle<crate::Shader>,
    pub(crate) path_trace: blade_asset::Handle<crate::Shader>,
    pub(crate) a_trous: blade_asset::Handle<crate::Shader>,
    pub(crate) post_proc: blade_asset::Handle<crate::Shader>,
    pub(crate) raster: blade_asset::Handle<crate::Shader>,
    pub(crate) skin: blade_asset::Handle<crate::Shader>,
    pub(crate) debug_draw: blade_asset::Handle<crate::Shader>,
    pub(crate) debug_blit: blade_asset::Handle<crate::Shader>,
}

impl Shaders {
    /// Load the stock shaders embedded in this crate.
    ///
    /// Ray-traced passes are replaced with a no-op shader when `ray_tracing` is false.
    pub fn load(asset_hub: &AssetHub, ray_tracing: bool) -> (Self, choir::RunningTask) {
        let mut ctx = asset_hub.open_context(Path::new("."), "shader finish");
        let noop = if ray_tracing {
            None
        } else {
            Some(ctx.load_shader_ir("noop.json", crate::ir::NOOP))
        };
        let shaders = Self {
            env_prepare: noop
                .unwrap_or_else(|| ctx.load_shader_ir("env_prepare.json", crate::ir::ENV_PREPARE)),
            fill_gbuf: noop
                .unwrap_or_else(|| ctx.load_shader_ir("fill_gbuf.json", crate::ir::FILL_GBUF)),
            ray_trace: noop
                .unwrap_or_else(|| ctx.load_shader_ir("ray_trace.json", crate::ir::RAY_TRACE)),
            path_trace: noop
                .unwrap_or_else(|| ctx.load_shader_ir("path_trace.json", crate::ir::PATH_TRACE)),
            a_trous: noop.unwrap_or_else(|| ctx.load_shader_ir("a_trous.json", crate::ir::A_TROUS)),
            post_proc: noop
                .unwrap_or_else(|| ctx.load_shader_ir("post_proc.json", crate::ir::POST_PROC)),
            raster: ctx.load_shader_ir("raster.json", crate::ir::RASTER),
            // GLES/WebGL keep vertex-stage skinning; compute skin is native-only.
            // `cfg!(gles)` is not set for wasm32 git dependents unless they
            // pass RUSTFLAGS, so match blade-graphics: wasm32 == GLES profile.
            skin: if cfg!(any(gles, target_arch = "wasm32")) {
                ctx.load_shader_ir("noop.json", crate::ir::NOOP)
            } else {
                ctx.load_shader_ir("skin.json", crate::ir::SKIN)
            },
            debug_draw: ctx.load_shader_ir("debug_draw.json", crate::ir::DEBUG_DRAW),
            debug_blit: ctx.load_shader_ir("debug_blit.json", crate::ir::DEBUG_BLIT),
        };
        (shaders, ctx.close())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn stock_ir_deserializes() {
        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        let caps = naga::valid::Capabilities::RAY_QUERY
            | naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY
            | naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING
            | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
            | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING;
        for &(name, bytes) in crate::ir::ALL {
            let module: naga::Module = serde_json::from_slice(bytes)
                .unwrap_or_else(|err| panic!("{name} did not deserialize: {err}"));
            naga::valid::Validator::new(flags, caps)
                .validate(&module)
                .unwrap_or_else(|err| panic!("{name} failed validation: {err}"));
        }
    }
}
