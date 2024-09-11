use glow::HasContext as _;
use std::cell::Cell;
use wasm_bindgen::JsCast;

//TODO: consider sharing this struct with EGL
struct Swapchain {
    renderbuf: glow::Renderbuffer,
    framebuf: glow::Framebuffer,
    format: crate::TextureFormat,
    extent: Cell<crate::Extent>,
}

pub struct PlatformContext {
    #[allow(unused)]
    webgl2: web_sys::WebGl2RenderingContext,
    glow: glow::Context,
    swapchain: Swapchain,
}

impl super::Context {
    pub unsafe fn init(_desc: crate::ContextDesc) -> Result<Self, crate::NotSupportedError> {
        Err(crate::NotSupportedError::PlatformNotSupported)
    }

    pub unsafe fn init_windowed<
        I: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
    >(
        window: I,
        desc: crate::ContextDesc,
    ) -> Result<Self, crate::NotSupportedError> {
        let webgl2 = match window.window_handle().unwrap().as_raw() {
            raw_window_handle::RawWindowHandle::Web(handle) => {
                let canvas: web_sys::HtmlCanvasElement = web_sys::window()
                    .and_then(|win| win.document())
                    .expect("Cannot get document")
                    .query_selector(&format!("canvas[data-raw-handle=\"{}\"]", handle.id))
                    .expect("Cannot query for canvas")
                    .expect("Canvas is not found")
                    .dyn_into()
                    .expect("Failed to downcast to canvas type");

                let context_options = js_sys::Object::new();
                js_sys::Reflect::set(
                    &context_options,
                    &"antialias".into(),
                    &wasm_bindgen::JsValue::FALSE,
                )
                .expect("Cannot create context options");
                //Note: could also set: "alpha", "premultipliedAlpha"

                canvas
                    .get_context_with_context_options("webgl2", &context_options)
                    .expect("Cannot create WebGL2 context")
                    .and_then(|context| context.dyn_into::<web_sys::WebGl2RenderingContext>().ok())
                    .expect("Cannot convert into WebGL2 context")
            }
            _ => return Err(crate::NotSupportedError::PlatformNotSupported),
        };

        let glow = glow::Context::from_webgl2_context(webgl2.clone());
        let capabilities = super::Capabilities::empty();
        let limits = super::Limits {
            uniform_buffer_alignment: unsafe {
                glow.get_parameter_i32(glow::UNIFORM_BUFFER_OFFSET_ALIGNMENT) as u32
            },
        };
        let swapchain = Swapchain {
            renderbuf: unsafe { glow.create_renderbuffer().unwrap() },
            framebuf: unsafe { glow.create_framebuffer().unwrap() },
            format: crate::TextureFormat::Rgba8Unorm,
            extent: Cell::default(),
        };

        let device_information = crate::DeviceInformation {
            is_software_emulated: false,
            device_name: glow.get_parameter_string(glow::VENDOR),
            driver_name: glow.get_parameter_string(glow::RENDERER),
            driver_info: glow.get_parameter_string(glow::VERSION),
        };

        Ok(Self {
            platform: PlatformContext {
                webgl2,
                glow,
                swapchain,
            },
            capabilities,
            toggles: super::Toggles::default(),
            limits,
            device_information,
        })
    }

    pub fn resize(&self, config: crate::SurfaceConfig) -> crate::SurfaceInfo {
        //TODO: create WebGL context here
        let sc = &self.platform.swapchain;
        let format_desc = super::describe_texture_format(sc.format);
        let gl = &self.platform.glow;
        //Note: this code can be shared with EGL
        unsafe {
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(sc.renderbuf));
            gl.renderbuffer_storage(
                glow::RENDERBUFFER,
                format_desc.internal,
                config.size.width as _,
                config.size.height as _,
            );
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(sc.framebuf));
            gl.framebuffer_renderbuffer(
                glow::READ_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::RENDERBUFFER,
                Some(sc.renderbuf),
            );
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
        }
        sc.extent.set(config.size);

        crate::SurfaceInfo {
            format: sc.format,
            alpha: crate::AlphaMode::PreMultiplied,
        }
    }

    pub fn acquire_frame(&self) -> super::Frame {
        let sc = &self.platform.swapchain;
        let size = sc.extent.get();
        super::Frame {
            texture: super::Texture {
                inner: super::TextureInner::Renderbuffer { raw: sc.renderbuf },
                target_size: [size.width as u16, size.height as u16],
                format: sc.format,
            },
        }
    }

    /// Obtain a lock to the EGL context and get handle to the [`glow::Context`] that can be used to
    /// do rendering.
    pub(super) fn lock(&self) -> &glow::Context {
        &self.platform.glow
    }

    pub(super) fn present(&self) {
        let sc = &self.platform.swapchain;
        unsafe {
            super::present_blit(&self.platform.glow, sc.framebuf, sc.extent.get());
        }
    }
}
