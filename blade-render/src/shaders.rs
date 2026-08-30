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
    pub fn load(
        path: &Path,
        asset_hub: &AssetHub,
        ray_tracing: bool,
    ) -> (Self, choir::RunningTask) {
        let mut ctx = asset_hub.open_context(path, "shader finish");
        let noop = if ray_tracing {
            None
        } else {
            Some(ctx.load_shader("noop.wgsl"))
        };
        let shaders = Self {
            env_prepare: noop.unwrap_or_else(|| ctx.load_shader("env-prepare.wgsl")),
            fill_gbuf: noop.unwrap_or_else(|| ctx.load_shader("fill-gbuf.wgsl")),
            ray_trace: noop.unwrap_or_else(|| ctx.load_shader("ray-trace.wgsl")),
            path_trace: noop.unwrap_or_else(|| ctx.load_shader("path-trace.wgsl")),
            a_trous: noop.unwrap_or_else(|| ctx.load_shader("a-trous.wgsl")),
            post_proc: noop.unwrap_or_else(|| ctx.load_shader("post-proc.wgsl")),
            raster: ctx.load_shader("raster.wgsl"),
            // GLES/WebGL keep vertex-stage skinning; compute skin is native-only.
            // `cfg!(gles)` is not set for wasm32 git dependents unless they
            // pass RUSTFLAGS, so match blade-graphics: wasm32 == GLES profile.
            skin: if cfg!(any(gles, target_arch = "wasm32")) {
                ctx.load_shader("noop.wgsl")
            } else {
                ctx.load_shader("skin.wgsl")
            },
            debug_draw: ctx.load_shader("debug-draw.wgsl"),
            debug_blit: ctx.load_shader("debug-blit.wgsl"),
        };
        (shaders, ctx.close())
    }
}
