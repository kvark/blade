//! OpenXR extensions of Meta Quest: passthrough (`XR_FB_passthrough`),
//! the depth of the environment (`XR_META_environment_depth`), and the clock
//! of the system (`XR_KHR_convert_timespec_time`), along with locating spaces
//! at any time. Each needs its extension enabled on the `openxr::Instance`
//! that the context is made with, and is `None` without it.

use ash::vk::{self, Handle as _};
use openxr::{self as xr, sys::Handle as _};
use std::{mem, ptr};

/// Colors that passthrough maps the brightness of the cameras to.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum XrPassthroughColorMap {
    /// The colors of the cameras.
    #[default]
    None,
    /// One color, with alpha, for each of the 256 levels of brightness.
    MonoToRgba(Box<[[f32; 4]; 256]>),
    /// One level for each of the 256 levels of brightness.
    MonoToMono(Box<[u8; 256]>),
    /// Brightness from -100 to 100, contrast and saturation from 0, with
    /// 1 for neither.
    BrightnessContrastSaturation {
        brightness: f32,
        contrast: f32,
        saturation: f32,
    },
}

impl XrPassthroughColorMap {
    /// The colors from black to `color`, by brightness, such as a red tint.
    pub fn tint(color: [f32; 4]) -> Self {
        let mut map = Box::new([[0.0; 4]; 256]);
        for (level, entry) in map.iter_mut().enumerate() {
            let t = level as f32 / 255.0;
            *entry = [color[0] * t, color[1] * t, color[2] * t, color[3]];
        }
        Self::MonoToRgba(map)
    }
}

/// How the passthrough layer draws the world.
#[derive(Clone, Debug, PartialEq)]
pub struct XrPassthroughStyle {
    /// From 0, transparent, to 1.
    pub opacity: f32,
    /// The color of the edges that it finds in the images, or with an alpha
    /// of 0, no edges.
    pub edge_color: [f32; 4],
    pub color_map: XrPassthroughColorMap,
}

impl Default for XrPassthroughStyle {
    fn default() -> Self {
        Self {
            opacity: 1.0,
            edge_color: [0.0; 4],
            color_map: XrPassthroughColorMap::None,
        }
    }
}

/// Passthrough of the cameras of the headset, as a layer of the frames, see
/// `layer`, which usually comes first, below the scene drawn with alpha.
pub struct XrPassthrough {
    session: xr::Session<xr::Vulkan>,
    passthrough: xr::sys::PassthroughFB,
    layer: xr::sys::PassthroughLayerFB,
    composition: xr::sys::CompositionLayerPassthroughFB,
}

// The handles belong to the session, which is shared between threads.
unsafe impl Send for XrPassthrough {}
unsafe impl Sync for XrPassthrough {}

impl XrPassthrough {
    fn fp(&self) -> &xr::raw::PassthroughFB {
        self.session.instance().exts().fb_passthrough.as_ref().unwrap()
    }

    /// Change how the world is drawn, from the next frame.
    pub fn set_style(&self, style: &XrPassthroughStyle) -> xr::Result<()> {
        let color = |c: [f32; 4]| xr::Color4f {
            r: c[0],
            g: c[1],
            b: c[2],
            a: c[3],
        };
        let mono_to_rgba;
        let mono_to_mono;
        let bcs;
        let next: *const std::ffi::c_void = match style.color_map {
            XrPassthroughColorMap::None => ptr::null(),
            XrPassthroughColorMap::MonoToRgba(ref map) => {
                mono_to_rgba = xr::sys::PassthroughColorMapMonoToRgbaFB {
                    ty: xr::sys::PassthroughColorMapMonoToRgbaFB::TYPE,
                    next: ptr::null(),
                    texture_color_map: map.map(color),
                };
                &mono_to_rgba as *const _ as *const _
            }
            XrPassthroughColorMap::MonoToMono(ref map) => {
                mono_to_mono = xr::sys::PassthroughColorMapMonoToMonoFB {
                    ty: xr::sys::PassthroughColorMapMonoToMonoFB::TYPE,
                    next: ptr::null(),
                    texture_color_map: **map,
                };
                &mono_to_mono as *const _ as *const _
            }
            XrPassthroughColorMap::BrightnessContrastSaturation {
                brightness,
                contrast,
                saturation,
            } => {
                bcs = xr::sys::PassthroughBrightnessContrastSaturationFB {
                    ty: xr::sys::PassthroughBrightnessContrastSaturationFB::TYPE,
                    next: ptr::null(),
                    brightness,
                    contrast,
                    saturation,
                };
                &bcs as *const _ as *const _
            }
        };
        let info = xr::sys::PassthroughStyleFB {
            ty: xr::sys::PassthroughStyleFB::TYPE,
            next,
            texture_opacity_factor: style.opacity,
            edge_color: color(style.edge_color),
        };
        cvt(unsafe { (self.fp().passthrough_layer_set_style)(self.layer, &info) })
    }

    /// Stop the cameras, which also saves power, until `resume`.
    pub fn pause(&self) -> xr::Result<()> {
        cvt(unsafe { (self.fp().passthrough_pause)(self.passthrough) })
    }

    pub fn resume(&self) -> xr::Result<()> {
        cvt(unsafe { (self.fp().passthrough_start)(self.passthrough) })
    }

    /// The layer to give `Context::xr_end_frame`. Its alpha blends it over
    /// the layers below it.
    pub fn layer(&self) -> super::XrLayer<'_> {
        // The structure starts with the header, and holds the handles of
        // `self`, which outlive the borrow.
        super::XrLayer::Raw(unsafe {
            super::XrRawLayer::new(
                &*(&self.composition as *const xr::sys::CompositionLayerPassthroughFB
                    as *const xr::sys::CompositionLayerBaseHeader),
            )
        })
    }
}

impl Drop for XrPassthrough {
    fn drop(&mut self) {
        let fp = self.fp();
        unsafe {
            (fp.destroy_passthrough_layer)(self.layer);
            (fp.destroy_passthrough)(self.passthrough);
        }
    }
}

/// The depth of the environment, as the headset senses it, in images that
/// the runtime fills, see `XrEnvironmentDepth::acquire`.
pub struct XrEnvironmentDepth {
    session: xr::Session<xr::Vulkan>,
    provider: xr::sys::EnvironmentDepthProviderMETA,
    swapchain: xr::sys::EnvironmentDepthSwapchainMETA,
    textures: Vec<super::Texture>,
    views: Vec<super::TextureView>,
}

unsafe impl Send for XrEnvironmentDepth {}
unsafe impl Sync for XrEnvironmentDepth {}

/// One depth image of the environment, with the views it was taken from,
/// which are not those of the frame.
#[derive(Clone, Copy, Debug)]
pub struct XrDepthImage {
    /// A 2D array of one layer for each eye, of `TextureFormat::Depth16Unorm`
    /// that a shader reads.
    pub texture: super::Texture,
    /// A view of all of its layers, as a 2D array.
    pub view: super::TextureView,
    /// The views of the eyes that the depth is projected for, in the base
    /// space, which reprojecting the depth to the views of the frame takes.
    pub views: [super::XrView; 2],
    /// The planes of the projection of the depth, where `far` may be
    /// infinite, in which case the depth is `1 - near / z`.
    pub near: f32,
    pub far: f32,
}

impl XrEnvironmentDepth {
    fn fp(&self) -> &xr::raw::EnvironmentDepthMETA {
        self.session
            .instance()
            .exts()
            .meta_environment_depth
            .as_ref()
            .unwrap()
    }

    pub fn start(&self) -> xr::Result<()> {
        cvt(unsafe { (self.fp().start_environment_depth_provider)(self.provider) })
    }

    pub fn stop(&self) -> xr::Result<()> {
        cvt(unsafe { (self.fp().stop_environment_depth_provider)(self.provider) })
    }

    /// Take the hands out of the depth, so that they don't hide what's drawn
    /// right in front of them.
    pub fn set_hand_removal(&self, enabled: bool) -> xr::Result<()> {
        let info = xr::sys::EnvironmentDepthHandRemovalSetInfoMETA {
            ty: xr::sys::EnvironmentDepthHandRemovalSetInfoMETA::TYPE,
            next: ptr::null(),
            enabled: enabled.into(),
        };
        cvt(unsafe { (self.fp().set_environment_depth_hand_removal)(self.provider, &info) })
    }

    /// The latest depth image, with its views located for the display time of
    /// `frame`, or `None` until the provider has one.
    pub fn acquire(
        &self,
        context: &super::Context,
        frame: &super::XrFrame,
    ) -> Option<XrDepthImage> {
        let xr_state = context.xr.as_ref()?.lock().unwrap();
        let space = xr_state.space.as_ref()?;
        let info = xr::sys::EnvironmentDepthImageAcquireInfoMETA {
            ty: xr::sys::EnvironmentDepthImageAcquireInfoMETA::TYPE,
            next: ptr::null(),
            space: space.as_raw(),
            display_time: frame.display_time,
        };
        let mut image: xr::sys::EnvironmentDepthImageMETA = unsafe { mem::zeroed() };
        image.ty = xr::sys::EnvironmentDepthImageMETA::TYPE;
        for view in image.views.iter_mut() {
            view.ty = xr::sys::EnvironmentDepthImageViewMETA::TYPE;
        }
        let result = unsafe {
            (self.fp().acquire_environment_depth_image)(self.provider, &info, &mut image)
        };
        if result != xr::sys::Result::SUCCESS {
            // Not ready yet is the usual answer of the first frames.
            if result != xr::sys::Result::ENVIRONMENT_DEPTH_NOT_AVAILABLE_META {
                log::warn!("XR environment depth acquire: {result}");
            }
            return None;
        }
        let index = image.swapchain_index as usize;
        let view_of = |view: &xr::sys::EnvironmentDepthImageViewMETA| super::XrView {
            pose: super::surface::pose_from_xr(view.pose),
            fov: super::XrFov {
                angle_left: view.fov.angle_left,
                angle_right: view.fov.angle_right,
                angle_up: view.fov.angle_up,
                angle_down: view.fov.angle_down,
            },
        };
        Some(XrDepthImage {
            texture: *self.textures.get(index)?,
            view: *self.views.get(index)?,
            views: [view_of(&image.views[0]), view_of(&image.views[1])],
            near: image.near_z,
            far: image.far_z,
        })
    }
}

impl super::Context {
    fn xr_instance(&self) -> Option<xr::Instance> {
        Some(self.xr.as_ref()?.lock().unwrap().instance.clone())
    }

    /// Start passthrough of the cameras, drawn in `style`. Needs
    /// `XR_FB_passthrough`.
    pub fn create_xr_passthrough(&self, style: &XrPassthroughStyle) -> Option<XrPassthrough> {
        let session = self.xr_session()?;
        let fp = session.instance().exts().fb_passthrough.as_ref()?;
        let mut passthrough = xr::sys::PassthroughFB::NULL;
        let info = xr::sys::PassthroughCreateInfoFB {
            ty: xr::sys::PassthroughCreateInfoFB::TYPE,
            next: ptr::null(),
            flags: xr::sys::PassthroughFlagsFB::IS_RUNNING_AT_CREATION,
        };
        if let Err(e) =
            cvt(unsafe { (fp.create_passthrough)(session.as_raw(), &info, &mut passthrough) })
        {
            log::error!("XR passthrough: {e}");
            return None;
        }
        let mut layer = xr::sys::PassthroughLayerFB::NULL;
        let layer_info = xr::sys::PassthroughLayerCreateInfoFB {
            ty: xr::sys::PassthroughLayerCreateInfoFB::TYPE,
            next: ptr::null(),
            passthrough,
            flags: xr::sys::PassthroughFlagsFB::IS_RUNNING_AT_CREATION,
            purpose: xr::sys::PassthroughLayerPurposeFB::RECONSTRUCTION,
        };
        if let Err(e) = cvt(unsafe {
            (fp.create_passthrough_layer)(session.as_raw(), &layer_info, &mut layer)
        }) {
            log::error!("XR passthrough layer: {e}");
            unsafe { (fp.destroy_passthrough)(passthrough) };
            return None;
        }
        let composition = xr::sys::CompositionLayerPassthroughFB {
            ty: xr::sys::CompositionLayerPassthroughFB::TYPE,
            next: ptr::null(),
            flags: xr::CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA,
            space: xr::sys::Space::NULL,
            layer_handle: layer,
        };
        let passthrough = XrPassthrough {
            session,
            passthrough,
            layer,
            composition,
        };
        if let Err(e) = passthrough.set_style(style) {
            log::warn!("XR passthrough style: {e}");
        }
        Some(passthrough)
    }

    /// Provide the depth of the environment. Needs
    /// `XR_META_environment_depth`, and on Quest the
    /// `com.oculus.permission.USE_SCENE` permission. It starts stopped.
    pub fn create_xr_environment_depth(&self) -> Option<XrEnvironmentDepth> {
        let session = self.xr_session()?;
        let fp = *session.instance().exts().meta_environment_depth.as_ref()?;
        let mut provider = xr::sys::EnvironmentDepthProviderMETA::NULL;
        let info = xr::sys::EnvironmentDepthProviderCreateInfoMETA {
            ty: xr::sys::EnvironmentDepthProviderCreateInfoMETA::TYPE,
            next: ptr::null(),
            create_flags: xr::sys::EnvironmentDepthProviderCreateFlagsMETA::EMPTY,
        };
        if let Err(e) = cvt(unsafe {
            (fp.create_environment_depth_provider)(session.as_raw(), &info, &mut provider)
        }) {
            log::error!("XR environment depth provider: {e}");
            return None;
        }
        let mut swapchain = xr::sys::EnvironmentDepthSwapchainMETA::NULL;
        let swapchain_info = xr::sys::EnvironmentDepthSwapchainCreateInfoMETA {
            ty: xr::sys::EnvironmentDepthSwapchainCreateInfoMETA::TYPE,
            next: ptr::null(),
            create_flags: xr::sys::EnvironmentDepthSwapchainCreateFlagsMETA::EMPTY,
        };
        let mut depth = XrEnvironmentDepth {
            session,
            provider,
            swapchain,
            textures: Vec::new(),
            views: Vec::new(),
        };
        if let Err(e) = cvt(unsafe {
            (fp.create_environment_depth_swapchain)(provider, &swapchain_info, &mut swapchain)
        }) {
            log::error!("XR environment depth swapchain: {e}");
            self.destroy_xr_environment_depth(&mut depth);
            return None;
        }
        depth.swapchain = swapchain;

        let mut state = xr::sys::EnvironmentDepthSwapchainStateMETA {
            ty: xr::sys::EnvironmentDepthSwapchainStateMETA::TYPE,
            next: ptr::null_mut(),
            width: 0,
            height: 0,
        };
        let mut count = 0;
        let images = cvt(unsafe { (fp.get_environment_depth_swapchain_state)(swapchain, &mut state) })
            .and_then(|()| {
                cvt(unsafe {
                    (fp.enumerate_environment_depth_swapchain_images)(
                        swapchain,
                        0,
                        &mut count,
                        ptr::null_mut(),
                    )
                })
            })
            .and_then(|()| {
                let mut images = vec![
                    xr::sys::SwapchainImageVulkanKHR {
                        ty: xr::sys::SwapchainImageVulkanKHR::TYPE,
                        next: ptr::null_mut(),
                        image: 0,
                    };
                    count as usize
                ];
                cvt(unsafe {
                    (fp.enumerate_environment_depth_swapchain_images)(
                        swapchain,
                        count,
                        &mut count,
                        images.as_mut_ptr() as *mut xr::sys::SwapchainImageBaseHeader,
                    )
                })
                .map(|()| images)
            });
        let images = match images {
            Ok(images) => images,
            Err(e) => {
                log::error!("XR environment depth images: {e}");
                self.destroy_xr_environment_depth(&mut depth);
                return None;
            }
        };
        let target_size = [state.width as u16, state.height as u16];
        for image in images.iter().take(count as usize) {
            let texture = super::Texture {
                raw: vk::Image::from_raw(image.image),
                memory_handle: !0,
                target_size,
                format: crate::TextureFormat::Depth16Unorm,
                external: None,
            };
            let view_info = vk::ImageViewCreateInfo {
                image: texture.raw,
                view_type: vk::ImageViewType::TYPE_2D_ARRAY,
                format: vk::Format::D16_UNORM,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 2,
                },
                ..Default::default()
            };
            let raw = unsafe { self.device.core.create_image_view(&view_info, None) }.unwrap();
            depth.textures.push(texture);
            depth.views.push(super::TextureView {
                raw,
                target_size,
                aspects: crate::TexelAspects::DEPTH,
            });
        }
        Some(depth)
    }

    pub fn destroy_xr_environment_depth(&self, depth: &mut XrEnvironmentDepth) {
        for view in depth.views.drain(..) {
            unsafe { self.device.core.destroy_image_view(view.raw, None) };
        }
        depth.textures.clear();
        let fp = *depth.fp();
        unsafe {
            if depth.swapchain != xr::sys::EnvironmentDepthSwapchainMETA::NULL {
                (fp.destroy_environment_depth_swapchain)(depth.swapchain);
                depth.swapchain = xr::sys::EnvironmentDepthSwapchainMETA::NULL;
            }
            if depth.provider != xr::sys::EnvironmentDepthProviderMETA::NULL {
                (fp.destroy_environment_depth_provider)(depth.provider);
                depth.provider = xr::sys::EnvironmentDepthProviderMETA::NULL;
            }
        }
    }

    /// A reference space of the session, such as `VIEW`, the head, which a
    /// HUD quad stays in front of.
    pub fn xr_create_reference_space(&self, ty: xr::ReferenceSpaceType) -> Option<xr::Space> {
        self.xr_session()?
            .create_reference_space(ty, xr::Posef::IDENTITY)
            .ok()
    }

    /// Where `space` is at `time`, in the base space, such as the head (the
    /// `VIEW` space) when a camera took an image. The runtime keeps a short
    /// history of the past.
    pub fn xr_locate(&self, space: &xr::Space, time: xr::Time) -> Option<super::XrPose> {
        let xr = self.xr.as_ref()?.lock().unwrap();
        let base = xr.space.as_ref()?;
        let location = space.locate(base, time).ok()?;
        let valid =
            xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID;
        location
            .location_flags
            .contains(valid)
            .then(|| super::surface::pose_from_xr(location.pose))
    }

    /// The XR time of a time of `CLOCK_MONOTONIC`. Needs
    /// `XR_KHR_convert_timespec_time`.
    pub fn xr_time_from_timespec(&self, monotonic: libc::timespec) -> Option<xr::Time> {
        let instance = self.xr_instance()?;
        let fp = instance.exts().khr_convert_timespec_time.as_ref()?;
        let mut time = xr::Time::from_nanos(0);
        cvt(unsafe {
            (fp.convert_timespec_time_to_time)(instance.as_raw(), &monotonic, &mut time)
        })
        .ok()?;
        Some(time)
    }

    /// The XR time of `nanos` of the system clock `clock`, such as
    /// `CLOCK_BOOTTIME` of the timestamps of Android cameras whose source
    /// is real time, or `CLOCK_MONOTONIC`. Needs `XR_KHR_convert_timespec_time`.
    // `timespec` is narrower on 32-bit targets.
    #[cfg(unix)]
    #[allow(trivial_numeric_casts, clippy::unnecessary_cast)]
    pub fn xr_time_from_clock(&self, clock: libc::clockid_t, nanos: i64) -> Option<xr::Time> {
        let now = |clock| {
            let mut ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            unsafe { libc::clock_gettime(clock, &mut ts) };
            ts.tv_sec as i64 * 1_000_000_000 + ts.tv_nsec as i64
        };
        // The offset of the clock from the monotonic one, which the extension
        // takes, now.
        let monotonic = if clock == libc::CLOCK_MONOTONIC {
            nanos
        } else {
            nanos - (now(clock) - now(libc::CLOCK_MONOTONIC))
        };
        self.xr_time_from_timespec(libc::timespec {
            tv_sec: (monotonic / 1_000_000_000) as _,
            tv_nsec: (monotonic % 1_000_000_000) as _,
        })
    }
}

fn cvt(result: xr::sys::Result) -> xr::Result<()> {
    if result.into_raw() >= 0 {
        Ok(())
    } else {
        Err(result)
    }
}
