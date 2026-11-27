//! VR and AR through OpenXR: the description of the session that a
//! `Context` takes, the swapchains, frames and composition layers of
//! `Context::xr_begin_frame` and `Context::xr_end_frame`, and, on Meta Quest,
//! passthrough and the depth of the environment.

use std::fmt;

#[cfg(all(
    not(gles),
    any(
        vulkan,
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    )
))]
pub use crate::hal::xr::{
    DepthImage, EnvironmentDepth, Frame, Image, Layer, Passthrough, PassthroughColorMap,
    PassthroughStyle, ProjectionLayer, QuadLayer, RawLayer, Surface,
};

#[cfg(not(any(
    vulkan,
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
)))]
mod openxr {
    #[derive(Clone, Debug)]
    pub enum Instance {}
    #[derive(Clone, Debug)]
    pub enum SystemId {}
}

#[derive(Clone)]
pub struct Desc {
    pub instance: openxr::Instance,
    pub system_id: openxr::SystemId,
}

impl fmt::Debug for Desc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Desc")
            .field("system_id", &self.system_id)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceConfig {
    pub size: crate::Extent,
    pub usage: crate::TextureUsage,
    pub color_space: crate::ColorSpace,
    pub view_count: u32,
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            size: crate::Extent::default(),
            usage: crate::TextureUsage::TARGET,
            color_space: crate::ColorSpace::Linear,
            view_count: 2,
        }
    }
}

/// How the XR compositor blends the layers of a frame with the real world,
/// see `Context::xr_end_frame`.
#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
pub enum BlendMode {
    /// The layers replace the view of the world.
    #[default]
    Opaque,
    /// The layers add up to the light of the world, as on optical see-through
    /// displays.
    Additive,
    /// The layers are blended over the world by their alpha, as for video
    /// passthrough where the runtime does it without a passthrough layer.
    AlphaBlend,
}

bitflags::bitflags! {
    /// How the compositor treats one XR composition layer.
    #[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
    pub struct LayerFlags: u32 {
        /// Blend the layer over those below it by the alpha of its texture,
        /// instead of replacing them.
        const BLEND_SOURCE_ALPHA = 1 << 0;
        /// The colors of the texture aren't multiplied by its alpha.
        const UNPREMULTIPLIED_ALPHA = 1 << 1;
        /// Ask the compositor to correct the chromatic aberration of the
        /// lenses for this layer.
        const CORRECT_CHROMATIC_ABERRATION = 1 << 2;
    }
}

/// Which eyes see an XR layer that isn't per eye, such as a quad.
#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
pub enum EyeVisibility {
    #[default]
    Both,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Pose {
    pub orientation: [f32; 4],
    pub position: [f32; 3],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Fov {
    pub angle_left: f32,
    pub angle_right: f32,
    pub angle_up: f32,
    pub angle_down: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct View {
    pub pose: Pose,
    pub fov: Fov,
}
