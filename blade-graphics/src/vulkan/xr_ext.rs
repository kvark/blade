//! OpenXR extensions of Meta Quest: passthrough (`XR_FB_passthrough`),
//! the depth of the environment (`XR_META_environment_depth`), and the clock
//! of the runtime (`XR_KHR_convert_timespec_time`), along with locating spaces
//! at any time. Each needs its extension enabled on the `openxr::Instance`
//! that the context is made with, and is `None` without it.

use ash::vk::{self, Handle as _};
use openxr::{self as xr, sys::Handle as _};
use std::{mem, ptr};

/// Colors that passthrough maps the brightness of the cameras to.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum PassthroughColorMap {
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

impl PassthroughColorMap {
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
pub struct PassthroughStyle {
    /// From 0, transparent, to 1.
    pub opacity: f32,
    /// The color of the edges that it finds in the images, or with an alpha
    /// of 0, no edges.
    pub edge_color: [f32; 4],
    pub color_map: PassthroughColorMap,
}

impl Default for PassthroughStyle {
    fn default() -> Self {
        Self {
            opacity: 1.0,
            edge_color: [0.0; 4],
            color_map: PassthroughColorMap::None,
        }
    }
}

/// Passthrough of the cameras of the headset, as a layer of the frames, see
/// `layer`, which usually comes first, below the scene drawn with alpha.
pub struct Passthrough {
    session: xr::Session<xr::Vulkan>,
    passthrough: xr::sys::PassthroughFB,
    layer: xr::sys::PassthroughLayerFB,
    composition: xr::sys::CompositionLayerPassthroughFB,
}

// The handles belong to the session, which is shared between threads.
unsafe impl Send for Passthrough {}
unsafe impl Sync for Passthrough {}

impl Passthrough {
    fn fp(&self) -> &xr::raw::PassthroughFB {
        self.session
            .instance()
            .exts()
            .fb_passthrough
            .as_ref()
            .unwrap()
    }

    /// Change how the world is drawn, from the next frame.
    pub fn set_style(&self, style: &PassthroughStyle) -> xr::Result<()> {
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
            PassthroughColorMap::None => ptr::null(),
            PassthroughColorMap::MonoToRgba(ref map) => {
                mono_to_rgba = xr::sys::PassthroughColorMapMonoToRgbaFB {
                    ty: xr::sys::PassthroughColorMapMonoToRgbaFB::TYPE,
                    next: ptr::null(),
                    texture_color_map: map.map(color),
                };
                &mono_to_rgba as *const _ as *const _
            }
            PassthroughColorMap::MonoToMono(ref map) => {
                mono_to_mono = xr::sys::PassthroughColorMapMonoToMonoFB {
                    ty: xr::sys::PassthroughColorMapMonoToMonoFB::TYPE,
                    next: ptr::null(),
                    texture_color_map: **map,
                };
                &mono_to_mono as *const _ as *const _
            }
            PassthroughColorMap::BrightnessContrastSaturation {
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
    pub fn layer(&self) -> crate::xr::Layer<'_> {
        // The structure starts with the header, and holds the handles of
        // `self`, which outlive the borrow.
        crate::xr::Layer::Raw(unsafe {
            crate::xr::RawLayer::new(
                &*(&self.composition as *const xr::sys::CompositionLayerPassthroughFB
                    as *const xr::sys::CompositionLayerBaseHeader),
            )
        })
    }
}

impl Drop for Passthrough {
    fn drop(&mut self) {
        let fp = self.fp();
        unsafe {
            (fp.destroy_passthrough_layer)(self.layer);
            (fp.destroy_passthrough)(self.passthrough);
        }
    }
}

/// The depth of the environment, as the headset senses it, in images that
/// the runtime fills, see `EnvironmentDepth::acquire`.
pub struct EnvironmentDepth {
    session: xr::Session<xr::Vulkan>,
    provider: xr::sys::EnvironmentDepthProviderMETA,
    swapchain: xr::sys::EnvironmentDepthSwapchainMETA,
    textures: Vec<super::Texture>,
    views: Vec<super::TextureView>,
}

unsafe impl Send for EnvironmentDepth {}
unsafe impl Sync for EnvironmentDepth {}

/// One depth image of the environment, with the views it was taken from,
/// which are not those of the frame.
#[derive(Clone, Copy, Debug)]
pub struct DepthImage {
    /// A 2D array of one layer for each eye, of `TextureFormat::Depth16Unorm`
    /// that a shader reads.
    pub texture: super::Texture,
    /// A view of all of its layers, as a 2D array.
    pub view: super::TextureView,
    /// The views of the eyes that the depth is projected for, in the base
    /// space, which reprojecting the depth to the views of the frame takes.
    pub views: [crate::xr::View; 2],
    /// The planes of the projection that the depth is in, where `far` may
    /// be infinite.
    pub near: f32,
    pub far: f32,
}

impl EnvironmentDepth {
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
        frame: &crate::xr::Frame,
    ) -> Option<DepthImage> {
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
        let view_of = |view: &xr::sys::EnvironmentDepthImageViewMETA| crate::xr::View {
            pose: super::surface::pose_from_xr(view.pose),
            fov: crate::xr::Fov {
                angle_left: view.fov.angle_left,
                angle_right: view.fov.angle_right,
                angle_up: view.fov.angle_up,
                angle_down: view.fov.angle_down,
            },
        };
        Some(DepthImage {
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
    pub fn create_xr_passthrough(&self, style: &PassthroughStyle) -> Option<Passthrough> {
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
        if let Err(e) =
            cvt(unsafe { (fp.create_passthrough_layer)(session.as_raw(), &layer_info, &mut layer) })
        {
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
        let passthrough = Passthrough {
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
    pub fn create_xr_environment_depth(&self) -> Option<EnvironmentDepth> {
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
        let mut depth = EnvironmentDepth {
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
        let images =
            cvt(unsafe { (fp.get_environment_depth_swapchain_state)(swapchain, &mut state) })
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
            let raw = match unsafe { self.device.core.create_image_view(&view_info, None) } {
                Ok(raw) => raw,
                Err(e) => {
                    log::error!("XR environment depth view: {e}");
                    self.destroy_xr_environment_depth(&mut depth);
                    return None;
                }
            };
            depth.textures.push(texture);
            depth.views.push(super::TextureView {
                raw,
                target_size,
                aspects: crate::TexelAspects::DEPTH,
            });
        }
        Some(depth)
    }

    pub fn destroy_xr_environment_depth(&self, depth: &mut EnvironmentDepth) {
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
    pub fn xr_locate(&self, space: &xr::Space, time: xr::Time) -> Option<crate::xr::Pose> {
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

    /// The XR time `age` before now, such as when a camera took an image:
    /// its age is the time since its timestamp, in the camera's own clock.
    /// Needs `XR_KHR_convert_timespec_time`, or on Windows
    /// `XR_KHR_win32_convert_performance_counter_time`.
    pub fn xr_time_ago(&self, age: std::time::Duration) -> Option<xr::Time> {
        let instance = self.xr_instance()?;
        #[cfg(not(windows))]
        let loaded = instance.exts().khr_convert_timespec_time.is_some();
        #[cfg(windows)]
        let loaded = instance
            .exts()
            .khr_win32_convert_performance_counter_time
            .is_some();
        if !loaded {
            return None;
        }
        let now = instance.now().ok()?;
        let age = i64::try_from(age.as_nanos()).ok()?;
        Some(xr::Time::from_nanos(now.as_nanos().checked_sub(age)?))
    }
}

fn cvt(result: xr::sys::Result) -> xr::Result<()> {
    if result.into_raw() >= 0 {
        Ok(())
    } else {
        Err(result)
    }
}

// The layouts that the runtime hands out and takes back its images in,
// against the `GENERAL` layout that Blade keeps all textures in.
impl super::CommandEncoder {
    fn xr_layout_barrier(
        &mut self,
        image: vk::Image,
        aspect_mask: vk::ImageAspectFlags,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let barrier = vk::ImageMemoryBarrier {
            old_layout,
            new_layout,
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            src_access_mask: vk::AccessFlags::MEMORY_WRITE
                | self.device.workarounds.extra_sync_src_access,
            dst_access_mask: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            ..Default::default()
        };
        unsafe {
            self.device.core.cmd_pipeline_barrier(
                self.buffers[0].raw,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Make an acquired XR swapchain image ready to render to, keeping what
    /// it holds, from the layout that the runtime hands it out in.
    /// `init_texture` does the same, discarding it.
    pub fn begin_xr_image(&mut self, image: &crate::xr::Image) {
        self.xr_layout_barrier(
            image.internal.image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::GENERAL,
        );
    }

    /// Give an XR swapchain image back the layout that the runtime takes it
    /// in, after the work that renders to it, before it is released.
    /// `present` does it for the image of a `Frame`.
    pub fn end_xr_image(&mut self, image: &crate::xr::Image) {
        self.end_xr_raw_image(image.internal.image);
    }

    pub(super) fn end_xr_raw_image(&mut self, image: vk::Image) {
        self.xr_layout_barrier(
            image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
    }

    /// Make a depth image of the environment readable by shaders, from the
    /// layout that the runtime hands it out in, until `end_xr_depth`.
    pub fn begin_xr_depth(&mut self, depth: &DepthImage) {
        self.xr_layout_barrier(
            depth.texture.raw,
            vk::ImageAspectFlags::DEPTH,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::GENERAL,
        );
    }

    /// Give a depth image of the environment back to the runtime, in the
    /// layout it handed it out in, after the work that reads it.
    pub fn end_xr_depth(&mut self, depth: &DepthImage) {
        self.xr_layout_barrier(
            depth.texture.raw,
            vk::ImageAspectFlags::DEPTH,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }
}
