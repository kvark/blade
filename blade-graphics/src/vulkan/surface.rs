use ash::vk::{self, Handle as _};
use openxr as xr;
use std::mem;

impl super::Surface {
    pub fn info(&self) -> crate::SurfaceInfo {
        crate::SurfaceInfo {
            format: self.swapchain.format,
            alpha: self.swapchain.alpha,
        }
    }

    unsafe fn deinit_swapchain(&mut self, raw_device: &ash::Device) {
        let _ = raw_device.device_wait_idle();
        self.device
            .destroy_swapchain(mem::take(&mut self.swapchain.raw), None);
        for frame in self.frames.drain(..) {
            for view in frame.xr_views {
                if view != vk::ImageView::null() {
                    raw_device.destroy_image_view(view, None);
                }
            }
            raw_device.destroy_image_view(frame.view, None);
            raw_device.destroy_semaphore(frame.acquire_semaphore, None);
            raw_device.destroy_semaphore(frame.present_semaphore, None);
        }
    }

    pub fn acquire_frame(&mut self) -> super::Frame {
        let acquire_semaphore = self.next_semaphore;
        match unsafe {
            self.device.acquire_next_image(
                self.swapchain.raw,
                !0,
                acquire_semaphore,
                vk::Fence::null(),
            )
        } {
            Ok((index, _suboptimal)) => {
                self.next_semaphore = mem::replace(
                    &mut self.frames[index as usize].acquire_semaphore,
                    acquire_semaphore,
                );
                super::Frame {
                    internal: self.frames[index as usize],
                    swapchain: self.swapchain,
                    image_index: Some(index),
                    xr_swapchain: 0,
                    xr_view_count: 0,
                    xr_views: [super::XrView::default(); super::MAX_XR_EYES],
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                log::warn!("Acquire failed because the surface is out of date");
                super::Frame {
                    internal: self.frames[0],
                    swapchain: self.swapchain,
                    image_index: None,
                    xr_swapchain: 0,
                    xr_view_count: 0,
                    xr_views: [super::XrView::default(); super::MAX_XR_EYES],
                }
            }
            Err(other) => panic!("Aquire image error {}", other),
        }
    }
}

impl super::XrSurface {
    pub fn acquire_frame(&mut self, context: &super::Context) -> Option<super::Frame> {
        let xr_state = context.xr.as_ref()?;
        {
            let mut xr = xr_state.lock().unwrap();
            let frame_state = xr.frame_wait.wait().ok()?;
            xr.frame_stream.begin().ok()?;
            xr.predicted_display_time = Some(frame_state.predicted_display_time);
            if !frame_state.should_render {
                xr.predicted_display_time = None;
                let environment_blend_mode = xr.environment_blend_mode;
                xr.frame_stream
                    .end(
                        frame_state.predicted_display_time,
                        environment_blend_mode,
                        &[],
                    )
                    .ok()?;
                return None;
            }
        }

        let image_index = self.raw.acquire_image().ok()?;
        self.raw.wait_image(xr::Duration::INFINITE).ok()?;
        let mut xr_views = [super::XrView::default(); super::MAX_XR_EYES];
        let xr_view_count = {
            let xr = xr_state.lock().unwrap();
            let predicted_display_time = xr.predicted_display_time?;
            let space = xr.space.as_ref()?;
            let (_, views) = xr
                .session
                .locate_views(xr.view_type, predicted_display_time, space)
                .ok()?;
            let count = views.len().min(self.view_count as usize);
            if views.len() > self.view_count as usize {
                log::warn!(
                    "OpenXR returned {} views, truncating to {}",
                    views.len(),
                    self.view_count
                );
            }
            for (i, view) in views.iter().take(count).enumerate() {
                xr_views[i] = super::XrView {
                    pose: super::XrPose {
                        orientation: [
                            view.pose.orientation.x,
                            view.pose.orientation.y,
                            view.pose.orientation.z,
                            view.pose.orientation.w,
                        ],
                        position: [
                            view.pose.position.x,
                            view.pose.position.y,
                            view.pose.position.z,
                        ],
                    },
                    fov: super::XrFov {
                        angle_left: view.fov.angle_left,
                        angle_right: view.fov.angle_right,
                        angle_up: view.fov.angle_up,
                        angle_down: view.fov.angle_down,
                    },
                };
            }
            count as u32
        };
        Some(super::Frame {
            internal: self.frames[image_index as usize],
            swapchain: self.swapchain,
            image_index: Some(image_index),
            xr_swapchain: (&mut self.raw as *mut xr::Swapchain<xr::Vulkan>) as usize,
            xr_view_count,
            xr_views,
        })
    }

    pub fn release_frame(&mut self) {
        self.raw.release_image().unwrap();
    }

    pub fn extent(&self) -> crate::Extent {
        crate::Extent {
            width: self.swapchain.target_size[0] as u32,
            height: self.swapchain.target_size[1] as u32,
            depth: 1,
        }
    }

    pub fn view_count(&self) -> u32 {
        self.view_count
    }

    pub fn format(&self) -> crate::TextureFormat {
        self.swapchain.format
    }

    pub fn swapchain(&self) -> &xr::Swapchain<xr::Vulkan> {
        &self.raw
    }
}

impl super::Context {
    pub fn create_surface<
        I: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
    >(
        &self,
        window: &I,
    ) -> Result<super::Surface, crate::NotSupportedError> {
        let khr_swapchain = self
            .device
            .swapchain
            .clone()
            .ok_or(crate::NotSupportedError::NoSupportedDeviceFound)?;

        let raw = unsafe {
            ash_window::create_surface(
                &self.entry,
                &self.instance.core,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .map_err(super::PlatformError::Init)?
        };

        let khr_surface = self
            .instance
            .surface
            .as_ref()
            .ok_or(crate::NotSupportedError::PlatformNotSupported)?;
        if unsafe {
            khr_surface.get_physical_device_surface_support(
                self.physical_device,
                self.queue_family_index,
                raw,
            ) != Ok(true)
        } {
            log::warn!("Rejected for not presenting to the window surface");
            return Err(crate::NotSupportedError::PlatformNotSupported);
        }

        let mut surface_info = vk::PhysicalDeviceSurfaceInfo2KHR {
            surface: raw,
            ..Default::default()
        };
        let mut fullscreen_exclusive_win32 = vk::SurfaceFullScreenExclusiveWin32InfoEXT::default();
        surface_info = surface_info.push_next(&mut fullscreen_exclusive_win32);
        let mut fullscreen_exclusive_ext = vk::SurfaceCapabilitiesFullScreenExclusiveEXT::default();
        let mut capabilities2_khr =
            vk::SurfaceCapabilities2KHR::default().push_next(&mut fullscreen_exclusive_ext);
        let _ = unsafe {
            self.instance
                .get_surface_capabilities2
                .as_ref()
                .unwrap()
                .get_physical_device_surface_capabilities2(
                    self.physical_device,
                    &surface_info,
                    &mut capabilities2_khr,
                )
        };
        log::debug!("{:?}", capabilities2_khr.surface_capabilities);

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let next_semaphore = unsafe {
            self.device
                .core
                .create_semaphore(&semaphore_create_info, None)
                .unwrap()
        };

        Ok(super::Surface {
            device: khr_swapchain,
            raw,
            frames: Vec::new(),
            next_semaphore,
            swapchain: super::Swapchain {
                raw: vk::SwapchainKHR::null(),
                format: crate::TextureFormat::Rgba8Unorm,
                alpha: crate::AlphaMode::Ignored,
                target_size: [0; 2],
            },
            full_screen_exclusive: fullscreen_exclusive_ext.full_screen_exclusive_supported != 0,
        })
    }

    pub fn destroy_surface(&self, surface: &mut super::Surface) {
        unsafe {
            surface.deinit_swapchain(&self.device.core);
            self.device
                .core
                .destroy_semaphore(surface.next_semaphore, None)
        };
        if let Some(ref surface_instance) = self.instance.surface {
            unsafe { surface_instance.destroy_surface(surface.raw, None) };
        }
    }

    pub fn reconfigure_surface(&self, surface: &mut super::Surface, config: crate::SurfaceConfig) {
        let khr_surface = self.instance.surface.as_ref().unwrap();

        let capabilities = unsafe {
            khr_surface
                .get_physical_device_surface_capabilities(self.physical_device, surface.raw)
                .unwrap()
        };
        if config.size.width < capabilities.min_image_extent.width
            || config.size.width > capabilities.max_image_extent.width
            || config.size.height < capabilities.min_image_extent.height
            || config.size.height > capabilities.max_image_extent.height
        {
            log::warn!(
                "Requested size {}x{} is outside of surface capabilities",
                config.size.width,
                config.size.height
            );
        }

        let (alpha, composite_alpha) = if config.transparent {
            if capabilities
                .supported_composite_alpha
                .contains(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED)
            {
                (
                    crate::AlphaMode::PostMultiplied,
                    vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
                )
            } else if capabilities
                .supported_composite_alpha
                .contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
            {
                (
                    crate::AlphaMode::PreMultiplied,
                    vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
                )
            } else {
                log::error!(
                    "No composite alpha flag for transparency: {:?}",
                    capabilities.supported_composite_alpha
                );
                (
                    crate::AlphaMode::Ignored,
                    vk::CompositeAlphaFlagsKHR::OPAQUE,
                )
            }
        } else {
            (
                crate::AlphaMode::Ignored,
                vk::CompositeAlphaFlagsKHR::OPAQUE,
            )
        };

        let (requested_frame_count, mode_preferences) = match config.display_sync {
            crate::DisplaySync::Block => (3, [vk::PresentModeKHR::FIFO].as_slice()),
            crate::DisplaySync::Recent => (
                3,
                [
                    vk::PresentModeKHR::MAILBOX,
                    vk::PresentModeKHR::FIFO_RELAXED,
                    vk::PresentModeKHR::IMMEDIATE,
                ]
                .as_slice(),
            ),
            crate::DisplaySync::Tear => (2, [vk::PresentModeKHR::IMMEDIATE].as_slice()),
        };
        let effective_frame_count = requested_frame_count.max(capabilities.min_image_count);

        let present_modes = unsafe {
            khr_surface
                .get_physical_device_surface_present_modes(self.physical_device, surface.raw)
                .unwrap()
        };
        let present_mode = *mode_preferences
            .iter()
            .find(|mode| present_modes.contains(mode))
            .unwrap();
        log::info!("Using surface present mode {:?}", present_mode);

        let queue_families = [self.queue_family_index];

        let mut supported_formats = Vec::new();
        let (format, surface_format) = if surface.swapchain.target_size[0] > 0 {
            let format = surface.swapchain.format;
            log::info!("Retaining current format: {:?}", format);
            let vk_color_space = match (format, config.color_space) {
                (crate::TextureFormat::Bgra8Unorm, crate::ColorSpace::Srgb) => {
                    vk::ColorSpaceKHR::SRGB_NONLINEAR
                }
                (crate::TextureFormat::Bgra8Unorm, crate::ColorSpace::Linear) => {
                    vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT
                }
                (crate::TextureFormat::Bgra8UnormSrgb, crate::ColorSpace::Linear) => {
                    vk::ColorSpaceKHR::default()
                }
                _ => panic!(
                    "Unexpected format {:?} under color space {:?}",
                    format, config.color_space
                ),
            };
            (
                format,
                vk::SurfaceFormatKHR {
                    format: super::map_texture_format(format),
                    color_space: vk_color_space,
                },
            )
        } else {
            supported_formats = unsafe {
                khr_surface
                    .get_physical_device_surface_formats(self.physical_device, surface.raw)
                    .unwrap()
            };
            match config.color_space {
                crate::ColorSpace::Linear => {
                    let surface_format = vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8A8_UNORM,
                        color_space: vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
                    };
                    if supported_formats.contains(&surface_format) {
                        log::info!("Using linear SRGB color space");
                        (crate::TextureFormat::Bgra8Unorm, surface_format)
                    } else {
                        (
                            crate::TextureFormat::Bgra8UnormSrgb,
                            vk::SurfaceFormatKHR {
                                format: vk::Format::B8G8R8A8_SRGB,
                                color_space: vk::ColorSpaceKHR::default(),
                            },
                        )
                    }
                }
                crate::ColorSpace::Srgb => (
                    crate::TextureFormat::Bgra8Unorm,
                    vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8A8_UNORM,
                        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                    },
                ),
            }
        };
        if !supported_formats.is_empty() && !supported_formats.contains(&surface_format) {
            log::error!("Surface formats are incompatible: {:?}", supported_formats);
        }

        let vk_usage = super::resource::map_texture_usage(config.usage, crate::TexelAspects::COLOR);
        if !capabilities.supported_usage_flags.contains(vk_usage) {
            log::error!(
                "Surface usages are incompatible: {:?}",
                capabilities.supported_usage_flags
            );
        }

        let mut full_screen_exclusive_info = vk::SurfaceFullScreenExclusiveInfoEXT {
            full_screen_exclusive: if config.allow_exclusive_full_screen {
                vk::FullScreenExclusiveEXT::ALLOWED
            } else {
                vk::FullScreenExclusiveEXT::DISALLOWED
            },
            ..Default::default()
        };

        let mut create_info = vk::SwapchainCreateInfoKHR {
            surface: surface.raw,
            min_image_count: effective_frame_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: vk::Extent2D {
                width: config.size.width,
                height: config.size.height,
            },
            image_array_layers: 1,
            image_usage: vk_usage,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha,
            present_mode,
            old_swapchain: surface.swapchain.raw,
            ..Default::default()
        }
        .queue_family_indices(&queue_families);

        if surface.full_screen_exclusive {
            assert!(self.device.full_screen_exclusive.is_some());
            create_info = create_info.push_next(&mut full_screen_exclusive_info);
            log::info!(
                "Configuring exclusive full screen: {}",
                config.allow_exclusive_full_screen
            );
        }
        let raw_swapchain = unsafe { surface.device.create_swapchain(&create_info, None).unwrap() };

        unsafe {
            surface.deinit_swapchain(&self.device.core);
        }

        let images = unsafe { surface.device.get_swapchain_images(raw_swapchain).unwrap() };
        let target_size = [config.size.width as u16, config.size.height as u16];
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        for image in images {
            let view_create_info = vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: surface_format.format,
                subresource_range,
                ..Default::default()
            };
            let view = unsafe {
                self.device
                    .core
                    .create_image_view(&view_create_info, None)
                    .unwrap()
            };
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let acquire_semaphore = unsafe {
                self.device
                    .core
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            };
            let present_semaphore = unsafe {
                self.device
                    .core
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            };
            surface.frames.push(super::InternalFrame {
                acquire_semaphore,
                present_semaphore,
                image,
                view,
                xr_views: [vk::ImageView::null(); super::MAX_XR_EYES],
            });
        }
        surface.swapchain = super::Swapchain {
            raw: raw_swapchain,
            format,
            alpha,
            target_size,
        };
    }

    fn xr_recommended_surface_config(
        &self,
        view_type: xr::ViewConfigurationType,
    ) -> Option<crate::XrSurfaceConfig> {
        let xr = self.xr.as_ref()?;
        let xr = xr.lock().unwrap();
        let views = xr
            .instance
            .enumerate_view_configuration_views(xr.system_id, view_type)
            .ok()?;
        let first = *views.first()?;
        let view_count = (views.len() as u32).min(super::MAX_XR_EYES as u32);
        Some(crate::XrSurfaceConfig {
            size: crate::Extent {
                width: first.recommended_image_rect_width,
                height: first.recommended_image_rect_height,
                depth: 1,
            },
            usage: crate::TextureUsage::TARGET,
            color_space: crate::ColorSpace::Linear,
            view_count,
        })
    }

    pub fn create_xr_surface(&self) -> Option<super::XrSurface> {
        let config =
            self.xr_recommended_surface_config(xr::ViewConfigurationType::PRIMARY_STEREO)?;
        self.create_xr_surface_configured(config)
    }

    fn create_xr_surface_configured(
        &self,
        config: crate::XrSurfaceConfig,
    ) -> Option<super::XrSurface> {
        let xr = self.xr.as_ref()?;
        let mut surface = {
            let xr = xr.lock().unwrap();
            let (raw_format, format) = select_xr_swapchain_format(&xr.session, config.color_space);
            let raw = xr
                .session
                .create_swapchain(&xr::SwapchainCreateInfo {
                    create_flags: xr::SwapchainCreateFlags::EMPTY,
                    usage_flags: xr_swapchain_usage(config.usage),
                    format: raw_format,
                    sample_count: 1,
                    width: config.size.width,
                    height: config.size.height,
                    face_count: 1,
                    array_size: config.view_count.max(1),
                    mip_count: 1,
                })
                .ok()?;
            super::XrSurface {
                raw,
                frames: Vec::new(),
                swapchain: super::Swapchain {
                    raw: vk::SwapchainKHR::null(),
                    format,
                    alpha: crate::AlphaMode::Ignored,
                    target_size: [config.size.width as u16, config.size.height as u16],
                },
                view_count: config.view_count.max(1),
            }
        };
        self.reconfigure_xr_surface(&mut surface, config);
        Some(surface)
    }

    pub fn destroy_xr_surface(&self, surface: &mut super::XrSurface) {
        for frame in surface.frames.drain(..) {
            for view in frame.xr_views {
                if view != vk::ImageView::null() {
                    unsafe { self.device.core.destroy_image_view(view, None) };
                }
            }
            unsafe {
                self.device.core.destroy_image_view(frame.view, None);
                self.device
                    .core
                    .destroy_semaphore(frame.acquire_semaphore, None);
                self.device
                    .core
                    .destroy_semaphore(frame.present_semaphore, None);
            }
        }
    }

    fn reconfigure_xr_surface(
        &self,
        surface: &mut super::XrSurface,
        config: crate::XrSurfaceConfig,
    ) {
        self.destroy_xr_surface(surface);
        let xr = self.xr.as_ref().expect("XR is not enabled in this context");
        let xr = xr.lock().unwrap();
        assert!(
            config.view_count as usize <= super::MAX_XR_EYES,
            "XR view count {} exceeds MAX_XR_EYES={}",
            config.view_count,
            super::MAX_XR_EYES
        );
        let (raw_format, format) = select_xr_swapchain_format(&xr.session, config.color_space);

        let new_handle = xr
            .session
            .create_swapchain(&xr::SwapchainCreateInfo {
                create_flags: xr::SwapchainCreateFlags::EMPTY,
                usage_flags: xr_swapchain_usage(config.usage),
                format: raw_format,
                sample_count: 1,
                width: config.size.width,
                height: config.size.height,
                face_count: 1,
                array_size: config.view_count.max(1),
                mip_count: 1,
            })
            .unwrap();
        surface.raw = new_handle;

        let target_size = [config.size.width as u16, config.size.height as u16];
        let view_type = if config.view_count > 1 {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: config.view_count.max(1),
        };

        for raw_image in surface.raw.enumerate_images().unwrap() {
            let image = vk::Image::from_raw(raw_image);
            let view_create_info = vk::ImageViewCreateInfo {
                image,
                view_type,
                format: super::map_texture_format(format),
                subresource_range,
                ..Default::default()
            };
            let view = unsafe {
                self.device
                    .core
                    .create_image_view(&view_create_info, None)
                    .unwrap()
            };
            let mut xr_views = [vk::ImageView::null(); super::MAX_XR_EYES];
            for eye in 0..config.view_count.max(1) {
                let xr_view_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: super::map_texture_format(format),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: eye,
                        layer_count: 1,
                    },
                    ..Default::default()
                };
                let xr_view = unsafe {
                    self.device
                        .core
                        .create_image_view(&xr_view_info, None)
                        .unwrap()
                };
                xr_views[eye as usize] = xr_view;
            }
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let acquire_semaphore = unsafe {
                self.device
                    .core
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            };
            let present_semaphore = unsafe {
                self.device
                    .core
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            };
            surface.frames.push(super::InternalFrame {
                acquire_semaphore,
                present_semaphore,
                image,
                view,
                xr_views,
            });
        }

        surface.swapchain = super::Swapchain {
            raw: vk::SwapchainKHR::null(),
            format,
            alpha: crate::AlphaMode::Ignored,
            target_size,
        };
        surface.view_count = config.view_count.max(1);
    }
}

fn xr_swapchain_usage(usage: crate::TextureUsage) -> xr::SwapchainUsageFlags {
    let mut out = xr::SwapchainUsageFlags::EMPTY;
    if usage.contains(crate::TextureUsage::TARGET) {
        out |= xr::SwapchainUsageFlags::COLOR_ATTACHMENT;
    }
    if usage.contains(crate::TextureUsage::RESOURCE) {
        out |= xr::SwapchainUsageFlags::SAMPLED;
    }
    if usage.contains(crate::TextureUsage::STORAGE) {
        out |= xr::SwapchainUsageFlags::UNORDERED_ACCESS;
    }
    if out.is_empty() {
        out = xr::SwapchainUsageFlags::COLOR_ATTACHMENT;
    }
    out
}

fn texture_format_from_xr_raw(raw: u32) -> Option<crate::TextureFormat> {
    let format = vk::Format::from_raw(raw as i32);
    Some(match format {
        vk::Format::R8G8B8A8_UNORM => crate::TextureFormat::Rgba8Unorm,
        vk::Format::R8G8B8A8_SRGB => crate::TextureFormat::Rgba8UnormSrgb,
        vk::Format::B8G8R8A8_UNORM => crate::TextureFormat::Bgra8Unorm,
        vk::Format::B8G8R8A8_SRGB => crate::TextureFormat::Bgra8UnormSrgb,
        _ => return None,
    })
}

fn select_xr_swapchain_format(
    session: &xr::Session<xr::Vulkan>,
    color_space: crate::ColorSpace,
) -> (u32, crate::TextureFormat) {
    let formats = session.enumerate_swapchain_formats().unwrap();
    let mut linear_candidate = None;
    let mut srgb_candidate = None;
    for raw in formats {
        if let Some(format) = texture_format_from_xr_raw(raw) {
            match format {
                crate::TextureFormat::Rgba8Unorm | crate::TextureFormat::Bgra8Unorm => {
                    if linear_candidate.is_none() {
                        linear_candidate = Some((raw, format));
                    }
                }
                crate::TextureFormat::Rgba8UnormSrgb | crate::TextureFormat::Bgra8UnormSrgb => {
                    if srgb_candidate.is_none() {
                        srgb_candidate = Some((raw, format));
                    }
                }
                _ => {}
            }
        }
    }
    match color_space {
        crate::ColorSpace::Linear => linear_candidate.or(srgb_candidate),
        crate::ColorSpace::Srgb => srgb_candidate.or(linear_candidate),
    }
    .expect("No compatible XR swapchain format available")
}
