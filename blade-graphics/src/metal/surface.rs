use core_graphics_types::{
    base::CGFloat,
    geometry::{CGRect, CGSize},
};
use objc::{
    class, msg_send,
    runtime::{Object, BOOL, YES},
    sel, sel_impl,
};

use std::{mem, ptr};

#[cfg(target_os = "macos")]
#[link(name = "QuartzCore", kind = "framework")]
extern "C" {
    #[allow(non_upper_case_globals)]
    static kCAGravityTopLeft: *mut Object;
}

impl super::Surface {
    pub unsafe fn from_view(view: *mut Object) -> Self {
        let main_layer: *mut Object = msg_send![view, layer];
        let class = class!(CAMetalLayer);
        let is_valid_layer: BOOL = msg_send![main_layer, isKindOfClass: class];
        let raw_layer = if is_valid_layer == YES {
            main_layer
        } else {
            // If the main layer is not a CAMetalLayer, we create a CAMetalLayer and use it.
            let new_layer: *mut Object = msg_send![class, new];
            let frame: CGRect = msg_send![main_layer, bounds];
            let () = msg_send![new_layer, setFrame: frame];
            #[cfg(target_os = "ios")]
            {
                // Unlike NSView, UIView does not allow to replace main layer.
                let () = msg_send![main_layer, addSublayer: new_layer];
                let () = msg_send![main_layer, setAutoresizingMask: 0x1Fu64];
                let screen: *mut Object = msg_send![class!(UIScreen), mainScreen];
                let scale_factor: CGFloat = msg_send![screen, nativeScale];
                let () = msg_send![view, setContentScaleFactor: scale_factor];
            };
            #[cfg(target_os = "macos")]
            {
                let () = msg_send![view, setLayer: new_layer];
                let () = msg_send![view, setWantsLayer: YES];
                let () = msg_send![new_layer, setContentsGravity: kCAGravityTopLeft];
                let window: *mut Object = msg_send![view, window];
                if !window.is_null() {
                    let scale_factor: CGFloat = msg_send![window, backingScaleFactor];
                    let () = msg_send![new_layer, setContentsScale: scale_factor];
                }
            }
            new_layer
        };

        Self {
            view: msg_send![view, retain],
            render_layer: mem::transmute::<_, &metal::MetalLayerRef>(raw_layer).to_owned(),
            info: crate::SurfaceInfo {
                format: crate::TextureFormat::Rgba8Unorm,
                alpha: crate::AlphaMode::Ignored,
            },
        }
    }

    /// Get the CALayerMetal for this surface, if any.
    /// This is platform specific API.
    pub fn metal_layer(&self) -> metal::MetalLayer {
        self.render_layer.clone()
    }

    pub fn info(&self) -> crate::SurfaceInfo {
        self.info
    }

    pub fn acquire_frame(&self) -> super::Frame {
        let (drawable, texture) = objc::rc::autoreleasepool(|| {
            let drawable = self.render_layer.next_drawable().unwrap();
            (drawable.to_owned(), drawable.texture().to_owned())
        });
        super::Frame { drawable, texture }
    }
}

impl super::Context {
    pub fn create_surface<
        I: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
    >(
        &self,
        window: &I,
        config: crate::SurfaceConfig,
    ) -> Result<super::Surface, crate::NotSupportedError> {
        let mut surface = match window.window_handle().unwrap().as_raw() {
            #[cfg(target_os = "ios")]
            raw_window_handle::RawWindowHandle::UiKit(handle) => unsafe {
                super::Surface::from_view(handle.ui_view.as_ptr() as *mut _)
            },
            #[cfg(target_os = "macos")]
            raw_window_handle::RawWindowHandle::AppKit(handle) => unsafe {
                super::Surface::from_view(handle.ns_view.as_ptr() as *mut _)
            },
            _ => return Err(crate::NotSupportedError::PlatformNotSupported),
        };
        self.reconfigure_surface(&mut surface, config);
        Ok(surface)
    }

    pub fn reconfigure_surface(&self, surface: &mut super::Surface, config: crate::SurfaceConfig) {
        let device = self.device.lock().unwrap();
        surface.info = crate::SurfaceInfo {
            format: match config.color_space {
                crate::ColorSpace::Linear => crate::TextureFormat::Bgra8UnormSrgb,
                crate::ColorSpace::Srgb => crate::TextureFormat::Bgra8Unorm,
            },
            alpha: if config.transparent {
                crate::AlphaMode::PostMultiplied
            } else {
                //Warning: it's not really ignored! Instead, it's assumed to be 1:
                // https://developer.apple.com/documentation/quartzcore/calayer/1410763-isopaque
                crate::AlphaMode::Ignored
            },
        };
        let vsync = match config.display_sync {
            crate::DisplaySync::Block => true,
            crate::DisplaySync::Recent | crate::DisplaySync::Tear => false,
        };

        surface.render_layer.set_opaque(!config.transparent);
        surface.render_layer.set_device(&*device);
        surface
            .render_layer
            .set_pixel_format(super::map_texture_format(surface.info.format));
        surface
            .render_layer
            .set_framebuffer_only(config.usage == crate::TextureUsage::TARGET);
        surface.render_layer.set_maximum_drawable_count(3);
        surface.render_layer.set_drawable_size(CGSize::new(
            config.size.width as f64,
            config.size.height as f64,
        ));
        unsafe {
            let () = msg_send![surface.render_layer, setDisplaySyncEnabled: vsync];
        }
    }

    pub fn destroy_surface(&self, surface: &mut super::Surface) {
        unsafe {
            let () = msg_send![surface.view, release];
        }
        surface.view = ptr::null_mut();
    }
}
