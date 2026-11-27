//! The XR types of the Vulkan backend, which `crate::xr` exports.

use openxr as xr;

pub use super::xr_ext::{
    DepthImage, EnvironmentDepth, Passthrough, PassthroughColorMap, PassthroughStyle,
};

pub struct Surface {
    pub(super) raw: openxr::Swapchain<openxr::Vulkan>,
    pub(super) frames: Vec<super::InternalFrame>,
    pub(super) swapchain: super::Swapchain,
    pub(super) view_count: u32,
}

pub(crate) struct SessionState {
    pub instance: xr::Instance,
    pub system_id: xr::SystemId,
    pub session: xr::Session<xr::Vulkan>,
    pub frame_wait: xr::FrameWaiter,
    pub frame_stream: xr::FrameStream<xr::Vulkan>,
    pub view_type: xr::ViewConfigurationType,
    /// The blend mode of the frames that `Surface::acquire_frame` begins,
    /// the first that the runtime lists.
    pub environment_blend_mode: xr::EnvironmentBlendMode,
    /// Every blend mode that the runtime supports, in its order of
    /// preference.
    pub environment_blend_modes: Vec<xr::EnvironmentBlendMode>,
    /// The base space of the views, and of the layers that don't name one.
    pub space: Option<xr::Space>,
    pub predicted_display_time: Option<xr::Time>,
}

/// An XR frame, from `Context::xr_begin_frame` to `Context::xr_end_frame`,
/// which takes it, with the composition layers that make it up.
#[must_use = "an XR frame has to be ended with `Context::xr_end_frame`"]
#[derive(Debug)]
pub struct Frame {
    pub(super) display_time: xr::Time,
    pub(super) view_count: u32,
    pub(super) views: [crate::xr::View; super::MAX_XR_VIEWS],
}

impl Frame {
    /// When the frame is expected to be displayed, which the views are
    /// predicted for.
    pub fn display_time(&self) -> xr::Time {
        self.display_time
    }

    /// The views of the eyes at the display time, in the base space.
    pub fn views(&self) -> &[crate::xr::View] {
        &self.views[..self.view_count as usize]
    }
}

/// An image of an XR swapchain, acquired with `Surface::acquire_image` and
/// held until `Surface::release_image`.
#[derive(Clone, Copy, Debug)]
pub struct Image {
    pub(super) index: u32,
    pub(super) internal: super::InternalFrame,
    pub(super) swapchain: super::Swapchain,
    pub(super) layer_count: u32,
}

impl Image {
    /// The whole image, with all of its array layers.
    pub fn texture(&self) -> super::Texture {
        super::Texture {
            raw: self.internal.image,
            memory_handle: !0,
            target_size: self.swapchain.target_size,
            format: self.swapchain.format,
            external: None,
        }
    }

    /// A view of all of the array layers.
    pub fn texture_view(&self) -> super::TextureView {
        super::TextureView {
            raw: self.internal.view,
            target_size: self.swapchain.target_size,
            aspects: crate::TexelAspects::COLOR,
        }
    }

    /// A view of one array layer, such as that of one eye.
    pub fn layer_view(&self, layer: u32) -> super::TextureView {
        assert!(
            layer < self.layer_count,
            "XR swapchain layer {layer} is out of range"
        );
        super::TextureView {
            raw: self.internal.xr_views[layer as usize],
            target_size: self.swapchain.target_size,
            aspects: crate::TexelAspects::COLOR,
        }
    }
}

/// A projection layer: one view per array layer of the swapchain, rendered
/// from those views.
pub struct ProjectionLayer<'a> {
    pub surface: &'a Surface,
    /// The views it was rendered from, or those of the frame it ends.
    pub views: Option<&'a [crate::xr::View]>,
    /// The space of the views, or the base space.
    pub space: Option<&'a xr::Space>,
    pub flags: crate::xr::LayerFlags,
}

/// A quad layer: a flat rectangle placed in a space, such as a HUD in the
/// view space, which the compositor samples directly.
pub struct QuadLayer<'a> {
    pub surface: &'a Surface,
    /// The array layer of the swapchain that it shows.
    pub array_index: u32,
    /// The space of the pose, or the base space.
    pub space: Option<&'a xr::Space>,
    /// The center of the quad, which faces along +Z of its orientation.
    pub pose: crate::xr::Pose,
    /// Width and height, in meters.
    pub size: [f32; 2],
    pub eye_visibility: crate::xr::EyeVisibility,
    pub flags: crate::xr::LayerFlags,
}

/// Any other composition layer that the runtime takes, such as that of
/// `XR_FB_passthrough`, as its raw OpenXR structure.
#[derive(Clone, Copy)]
pub struct RawLayer<'a> {
    pub(super) header: &'a xr::sys::CompositionLayerBaseHeader,
}

impl<'a> RawLayer<'a> {
    /// # Safety
    ///
    /// `header` has to be the start of a complete OpenXR composition layer
    /// structure, whose handles and chained structures stay valid until
    /// `Context::xr_end_frame` returns.
    pub unsafe fn new(header: &'a xr::sys::CompositionLayerBaseHeader) -> Self {
        Self { header }
    }
}

/// One composition layer of an XR frame. The layers of a frame are
/// composited in order, the first at the back.
pub enum Layer<'a> {
    Projection(ProjectionLayer<'a>),
    Quad(QuadLayer<'a>),
    Raw(RawLayer<'a>),
}
