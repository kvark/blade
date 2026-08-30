use glow::HasContext as _;
use wasm_bindgen::JsCast;

const PRESENT_VS: &str = r#"#version 300 es
void main() {
    vec2 pos[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
}
"#;

const PRESENT_FS: &str = r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D src;
out vec4 frag;
void main() {
    // Same Y-flip as `present_blit`. texelFetch is a raw copy: no filter,
    // no sRGB decode (that is part of texture filtering).
    ivec2 size = textureSize(src, 0);
    ivec2 p = ivec2(gl_FragCoord.xy);
    p.y = size.y - 1 - p.y;
    frag = texelFetch(src, p, 0);
}
"#;

struct PresentCopy {
    program: glow::Program,
    vao: glow::VertexArray,
    loc_src: glow::UniformLocation,
}

pub struct PlatformContext {
    #[allow(unused)]
    webgl2: web_sys::WebGl2RenderingContext,
    glow: glow::Context,
    present_copy: PresentCopy,
}

pub struct PlatformSurface {
    info: crate::SurfaceInfo,
    extent: crate::Extent,
}
#[derive(Debug)]
pub struct PlatformFrame {
    texture: glow::Texture,
    extent: crate::Extent,
}

impl super::Surface {
    pub fn info(&self) -> crate::SurfaceInfo {
        self.platform.info
    }
    pub fn acquire_frame(&self) -> super::Frame {
        let size = self.platform.extent;
        super::Frame {
            platform: PlatformFrame {
                texture: self.offscreen_texture,
                extent: self.platform.extent,
            },
            texture: super::Texture {
                inner: super::TextureInner::Texture {
                    raw: self.offscreen_texture,
                    target: glow::TEXTURE_2D,
                },
                target_size: [size.width as u16, size.height as u16],
                format: self.platform.info.format,
            },
        }
    }
}

impl PlatformContext {
    pub(super) fn present(&self, frame: PlatformFrame) {
        let gl = &self.glow;
        // The canvas drawing buffer is RGBA8. `blitFramebuffer` from an
        // sRGB offscreen decodes; a texelFetch draw copies stored bytes.
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, frame.extent.width as i32, frame.extent.height as i32);
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.color_mask(true, true, true, true);
            gl.use_program(Some(self.present_copy.program));
            gl.bind_vertex_array(Some(self.present_copy.vao));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(frame.texture));
            gl.uniform_1_i32(Some(&self.present_copy.loc_src), 0);
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }
}

impl super::Context {
    pub unsafe fn init(_desc: crate::ContextDesc) -> Result<Self, crate::NotSupportedError> {
        let canvas = web_sys::window()
            .and_then(|win| win.document())
            .expect("Cannot get document")
            .get_element_by_id("blade")
            .expect("Canvas is not found")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("Failed to downcast to canvas type");

        let context_options = js_sys::Object::new();
        js_sys::Reflect::set(
            &context_options,
            &"antialias".into(),
            &wasm_bindgen::JsValue::FALSE,
        )
        .expect("Cannot create context options");
        //Note: could also set: "alpha", "premultipliedAlpha"

        let webgl2 = canvas
            .get_context_with_context_options("webgl2", &context_options)
            .expect("Cannot create WebGL2 context")
            .and_then(|context| context.dyn_into::<web_sys::WebGl2RenderingContext>().ok())
            .expect("Cannot convert into WebGL2 context");

        let glow = glow::Context::from_webgl2_context(webgl2.clone());
        let present_copy = Self::compile_present_copy(&glow);

        let capabilities = super::Capabilities::empty();
        let limits = super::Limits {
            uniform_buffer_alignment: unsafe {
                glow.get_parameter_i32(glow::UNIFORM_BUFFER_OFFSET_ALIGNMENT) as u32
            },
        };
        let device_information = unsafe {
            crate::DeviceInformation {
                is_software_emulated: false,
                device_name: glow.get_parameter_string(glow::VENDOR),
                driver_name: glow.get_parameter_string(glow::RENDERER),
                driver_info: glow.get_parameter_string(glow::VERSION),
            }
        };

        Ok(super::Context {
            platform: PlatformContext {
                webgl2,
                glow,
                present_copy,
            },
            capabilities,
            toggles: super::Toggles::default(),
            limits,
            device_information,
        })
    }

    pub fn create_surface<I>(
        &self,
        _window: &I,
    ) -> Result<super::Surface, crate::NotSupportedError> {
        let platform = PlatformSurface {
            info: crate::SurfaceInfo {
                format: crate::TextureFormat::Rgba8Unorm,
                alpha: crate::AlphaMode::PreMultiplied,
            },
            extent: crate::Extent::default(),
        };
        Ok(unsafe {
            super::Surface {
                platform,
                offscreen_texture: self.platform.glow.create_texture().unwrap(),
                framebuf: self.platform.glow.create_framebuffer().unwrap(),
            }
        })
    }

    pub fn destroy_surface(&self, _surface: &mut super::Surface) {}

    pub fn reconfigure_surface(&self, surface: &mut super::Surface, config: crate::SurfaceConfig) {
        // Offscreen target the app renders into. Linear shaders get an sRGB
        // texture so the GPU encodes; present texelFetches those bytes onto
        // the RGBA8 canvas (a blit would decode and look dark).
        let format = match config.color_space {
            crate::ColorSpace::Linear => crate::TextureFormat::Rgba8UnormSrgb,
            crate::ColorSpace::Srgb => crate::TextureFormat::Rgba8Unorm,
        };
        let format_desc = super::describe_texture_format(format);
        let gl = &self.platform.glow;
        //Note: this code can be shared with EGL
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(surface.offscreen_texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                format_desc.internal as i32,
                config.size.width as _,
                config.size.height as _,
                0,
                format_desc.external,
                format_desc.data_type,
                glow::PixelUnpackData::Slice(None),
            );
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(surface.framebuf));
            gl.framebuffer_texture_2d(
                glow::READ_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(surface.offscreen_texture),
                0,
            );
            for filter in [glow::TEXTURE_MIN_FILTER, glow::TEXTURE_MAG_FILTER] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, filter, glow::NEAREST as i32);
            }
            for wrap in [glow::TEXTURE_WRAP_S, glow::TEXTURE_WRAP_T] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, wrap, glow::CLAMP_TO_EDGE as i32);
            }
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
        surface.platform.extent = config.size;
        surface.platform.info.format = format;
    }

    fn compile_present_copy(gl: &glow::Context) -> PresentCopy {
        unsafe {
            let vs = gl.create_shader(glow::VERTEX_SHADER).unwrap();
            gl.shader_source(vs, PRESENT_VS);
            gl.compile_shader(vs);
            assert!(
                gl.get_shader_compile_status(vs),
                "present vs: {}",
                gl.get_shader_info_log(vs)
            );
            let fs = gl.create_shader(glow::FRAGMENT_SHADER).unwrap();
            gl.shader_source(fs, PRESENT_FS);
            gl.compile_shader(fs);
            assert!(
                gl.get_shader_compile_status(fs),
                "present fs: {}",
                gl.get_shader_info_log(fs)
            );
            let program = gl.create_program().unwrap();
            gl.attach_shader(program, vs);
            gl.attach_shader(program, fs);
            gl.link_program(program);
            assert!(
                gl.get_program_link_status(program),
                "present link: {}",
                gl.get_program_info_log(program)
            );
            gl.delete_shader(vs);
            gl.delete_shader(fs);
            let loc_src = gl
                .get_uniform_location(program, "src")
                .expect("present src uniform");
            let vao = gl.create_vertex_array().unwrap();
            PresentCopy {
                program,
                vao,
                loc_src,
            }
        }
    }

    /// Obtain a lock to the EGL context and get handle to the [`glow::Context`] that can be used to
    /// do rendering.
    pub(super) fn lock(&self) -> &glow::Context {
        &self.platform.glow
    }
}
