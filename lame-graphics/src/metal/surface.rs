use core_graphics_types::{
    base::CGFloat,
    geometry::{CGRect, CGSize},
};
use objc::{class, msg_send, sel, sel_impl, runtime::{Object, BOOL, YES}};

use std::{mem};

#[cfg(target_os = "macos")]
#[link(name = "QuartzCore", kind = "framework")]
extern "C" {
    #[allow(non_upper_case_globals)]
    static kCAGravityTopLeft: *mut Object;
}

impl Drop for super::Surface {
    fn drop(&mut self) {
        unsafe {
            let () = msg_send![self.view, release];
        }
    }
}

impl super::Surface {
    pub unsafe fn from_view(
        view: *mut Object,
    ) -> Self {
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
            render_layer: mem::transmute::<_, &metal::MetalLayerRef>(raw_layer)
                .to_owned(),
            format: crate::TextureFormat::Rgba8Unorm,
        }
    }

    fn reconfigure(&mut self, device: &metal::DeviceRef, config: crate::SurfaceConfig) {
        self.render_layer.set_opaque(true);
        self.render_layer.set_device(device);
        self.render_layer.set_pixel_format(super::map_texture_format(self.format));
        self.render_layer.set_framebuffer_only(config.usage == crate::TextureUsage::TARGET);
        self.render_layer.set_maximum_drawable_count(config.frame_count as u64);
        self.render_layer.set_drawable_size(CGSize::new(config.size.width as f64, config.size.height as f64));
        unsafe {
            let () = msg_send![self.render_layer, setDisplaySyncEnabled: true];
        }
    }
}

impl super::Context {
    pub fn resize(&self, config: crate::SurfaceConfig) -> crate::TextureFormat {
        let mut surface = self.surface.as_ref().unwrap().lock().unwrap();
        surface.reconfigure(&*self.device.lock().unwrap(), config);
        surface.format
    }

    pub fn acquire_frame(&self) -> super::Frame {
        let surface = self.surface.as_ref().unwrap().lock().unwrap();
        let (drawable, texture) = objc::rc::autoreleasepool(|| {
            let drawable = surface.render_layer
                .next_drawable()
                .unwrap();
            (drawable.to_owned(), drawable.texture().to_owned())
        });
        super::Frame {
            drawable,
            texture,
        }
    }

    pub fn present(&self, frame: super::Frame) {
        let queue = self.queue.lock().unwrap();
        objc::rc::autoreleasepool(|| {
            let command_buffer = queue.new_command_buffer();
            command_buffer.set_label("_present");
            command_buffer.present_drawable(&frame.drawable);
            command_buffer.commit();
        });
    }
}
