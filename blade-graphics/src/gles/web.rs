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

pub struct Context {
    #[allow(unused)]
    webgl2: web_sys::WebGl2RenderingContext,
    glow: glow::Context,
    swapchain: Swapchain,
    pub(super) capabilities: super::Capabilities,
    pub(super) limits: super::Limits,
}

impl Context {
    pub unsafe fn init(_desc: crate::ContextDesc) -> Result<Self, crate::NotSupportedError> {
        Err(crate::NotSupportedError)
    }

    pub unsafe fn init_windowed<
        I: raw_window_handle::HasRawWindowHandle + raw_window_handle::HasRawDisplayHandle,
    >(
        window: I,
        desc: crate::ContextDesc,
    ) -> Result<Self, crate::NotSupportedError> {
        let webgl2 = match window.raw_window_handle() {
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

                canvas
                    .get_context_with_context_options("webgl2", &context_options)
                    .expect("Cannot create WebGL2 context")
                    .and_then(|context| context.dyn_into::<web_sys::WebGl2RenderingContext>().ok())
                    .expect("Cannot convert into WebGL2 context")
            }
            _ => return Err(crate::NotSupportedError),
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

        Ok(Self {
            webgl2,
            glow,
            swapchain,
            capabilities,
            limits,
        })
    }

    pub fn resize(&self, config: crate::SurfaceConfig) -> crate::TextureFormat {
        let sc = &self.swapchain;
        let format_desc = super::describe_texture_format(sc.format);
        let gl = &self.glow;
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
        sc.format
    }

    pub fn acquire_frame(&self) -> super::Frame {
        let sc = &self.swapchain;
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
        &self.glow
    }

    pub(super) fn present(&self) {
        let sc = &self.swapchain;
        let size = sc.extent.get();
        let gl = &self.glow;
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.color_mask(true, true, true, true);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(sc.framebuf));
            // Note the Y-flipping here. GL's presentation is not flipped,
            // but main rendering is. Therefore, we Y-flip the output positions
            // in the shader, and also this blit.
            gl.blit_framebuffer(
                0,
                size.height as i32,
                size.width as i32,
                0,
                0,
                0,
                size.width as i32,
                size.height as i32,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        }
    }
}
