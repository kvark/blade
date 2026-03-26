use glow::HasContext as _;
use std::{
    ffi,
    os::raw,
    ptr,
    sync::{Arc, Mutex, MutexGuard},
};

const EGL_CONTEXT_FLAGS_KHR: i32 = 0x30FC;
const EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR: i32 = 0x0001;
const EGL_PLATFORM_WAYLAND_KHR: u32 = 0x31D8;
const EGL_PLATFORM_X11_KHR: u32 = 0x31D5;
const _EGL_PLATFORM_XCB_EXT: u32 = 0x31DC;
const EGL_PLATFORM_ANGLE_ANGLE: u32 = 0x3202;
const EGL_PLATFORM_ANGLE_TYPE_ANGLE: u32 = 0x3203;
const EGL_PLATFORM_ANGLE_TYPE_DEFAULT_ANGLE: u32 = 0x3206;
const EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE: u32 = 0x3489;
const EGL_PLATFORM_ANGLE_NATIVE_PLATFORM_TYPE_ANGLE: u32 = 0x348F;
const EGL_PLATFORM_ANGLE_DEBUG_LAYERS_ENABLED: u32 = 0x3451;
const EGL_PLATFORM_SURFACELESS_MESA: u32 = 0x31DD;
const EGL_PLATFORM_GBM_MESA: u32 = 0x31D7;

const EGL_DEBUG_MSG_CRITICAL_KHR: u32 = 0x33B9;
const EGL_DEBUG_MSG_ERROR_KHR: u32 = 0x33BA;
const EGL_DEBUG_MSG_WARN_KHR: u32 = 0x33BB;
const EGL_DEBUG_MSG_INFO_KHR: u32 = 0x33BC;

// EGLImage / DMA-BUF constants
const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: egl::Attrib = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: egl::Attrib = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: egl::Attrib = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: egl::Attrib = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: egl::Attrib = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: egl::Attrib = 0x3444;

type _XOpenDisplayFun =
    unsafe extern "system" fn(display_name: *const raw::c_char) -> *mut raw::c_void;
type _WlDisplayConnectFun =
    unsafe extern "system" fn(display_name: *const raw::c_char) -> *mut raw::c_void;
type _WlDisplayDisconnectFun = unsafe extern "system" fn(display: *const raw::c_void);

type EglInstance = egl::DynamicInstance<egl::EGL1_4>;

type WlEglWindowCreateFun = unsafe extern "system" fn(
    surface: *const raw::c_void,
    width: raw::c_int,
    height: raw::c_int,
) -> *mut raw::c_void;

type _WlEglWindowResizeFun = unsafe extern "system" fn(
    window: *const raw::c_void,
    width: raw::c_int,
    height: raw::c_int,
    dx: raw::c_int,
    dy: raw::c_int,
);

type WlEglWindowDestroyFun = unsafe extern "system" fn(window: *const raw::c_void);

type EglLabel = *const raw::c_void;

type DebugProcKHR = Option<
    unsafe extern "system" fn(
        error: egl::Enum,
        command: *const raw::c_char,
        message_type: u32,
        thread_label: EglLabel,
        object_label: EglLabel,
        message: *const raw::c_char,
    ),
>;

type EglDebugMessageControlFun =
    unsafe extern "system" fn(proc: DebugProcKHR, attrib_list: *const egl::Attrib) -> raw::c_int;

// GBM function types (loaded dynamically from libgbm.so)
type GbmCreateDeviceFun = unsafe extern "C" fn(fd: raw::c_int) -> *mut ffi::c_void;
type GbmDeviceDestroyFun = unsafe extern "C" fn(gbm: *mut ffi::c_void);
type GbmBoCreateFun = unsafe extern "C" fn(
    gbm: *mut ffi::c_void,
    width: u32,
    height: u32,
    format: u32,
    flags: u32,
) -> *mut ffi::c_void;
type GbmBoDestroyFun = unsafe extern "C" fn(bo: *mut ffi::c_void);
type GbmBoGetFdFun = unsafe extern "C" fn(bo: *mut ffi::c_void) -> raw::c_int;
type GbmBoGetStrideFun = unsafe extern "C" fn(bo: *mut ffi::c_void) -> u32;
type GbmBoGetModifierFun = unsafe extern "C" fn(bo: *mut ffi::c_void) -> u64;

const GBM_FORMAT_ABGR8888: u32 = 0x34324241; // DRM_FORMAT_ABGR8888
const GBM_BO_USE_RENDERING: u32 = 1 << 2;
const GBM_BO_USE_LINEAR: u32 = 1 << 4;

// GL_OES_EGL_image
type GlEglImageTargetTexture2dOesFun =
    unsafe extern "system" fn(target: u32, image: *mut ffi::c_void);

#[derive(Clone, Copy, Debug)]
enum SrgbFrameBufferKind {
    /// No support for SRGB surface
    None,
    /// Using EGL 1.5's support for colorspaces
    Core,
    /// Using EGL_KHR_gl_colorspace
    Khr,
}

#[derive(Debug)]
struct EglContext {
    instance: EglInstance,
    display: egl::Display,
    raw: egl::Context,
    config: egl::Config,
    pbuffer: Option<egl::Surface>,
    srgb_kind: SrgbFrameBufferKind,
    /// If true, don't terminate the display on drop (shared display).
    shared_display: bool,
}

impl EglContext {
    fn make_current(&self) {
        self.instance
            .make_current(self.display, self.pbuffer, self.pbuffer, Some(self.raw))
            .unwrap();
    }
    fn unmake_current(&self) {
        self.instance
            .make_current(self.display, None, None, None)
            .unwrap();
    }
}

/// DMA-BUF / EGLImage function pointers for shared buffer import.
struct DmaBufFunctions {
    image_target_texture: GlEglImageTargetTexture2dOesFun,
}

/// Presentation context on a separate EGL display for window surface blitting.
/// Used when the main context is on a surfaceless display.
struct PresentationContext {
    egl: EglContext,
    glow: glow::Context,
    swapchain: Swapchain,
    /// GL texture backed by the imported DMA-BUF
    imported_texture: glow::Texture,
    /// Framebuffer with imported_texture attached for reading
    source_framebuf: glow::Framebuffer,
    /// EGLImage on the presentation display (from DMA-BUF import)
    imported_image: egl::Image,
}

/// GBM buffer object that backs the surface's off-screen texture.
/// The same DMA-BUF is shared between main and presentation GL contexts.
struct GbmBuffer {
    bo: *mut ffi::c_void,
    fd: raw::c_int,
    stride: u32,
    modifier: u64,
    /// EGLImage on the main display (imported from DMA-BUF)
    main_image: egl::Image,
}

#[derive(Clone, Debug)]
struct Swapchain {
    surface: egl::Surface,
    /// Wayland EGL window handle (must be destroyed after the EGL surface).
    wl_window: Option<*mut ffi::c_void>,
    extent: crate::Extent,
    info: crate::SurfaceInfo,
    swap_interval: i32,
}

unsafe impl Send for Swapchain {}
unsafe impl Sync for Swapchain {}

enum PresentMode {
    /// Direct presentation: main context creates window surfaces (ANGLE, default display).
    Direct(Swapchain),
    /// DMA-BUF sharing: separate presentation context blits from shared texture.
    DmaBuf(Arc<PresentationContext>),
}

unsafe impl Send for PresentationContext {}
unsafe impl Sync for PresentationContext {}

pub struct PlatformFrame {
    framebuf: glow::Framebuffer,
    present_mode: PresentMode,
}

impl std::fmt::Debug for PlatformFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlatformFrame")
            .field("framebuf", &self.framebuf)
            .finish()
    }
}

pub struct PlatformSurface {
    library: Option<libloading::Library>,
    window_handle: raw_window_handle::RawWindowHandle,
    display_handle: raw_window_handle::RawDisplayHandle,
    swapchain: Option<Swapchain>,
    /// Presentation context for DMA-BUF path (GBM-backed main display).
    presentation: Option<Arc<PresentationContext>>,
    /// GBM buffer object backing the off-screen texture.
    gbm_buffer: Option<GbmBuffer>,
}

pub(super) struct ContextInner {
    glow: glow::Context,
    egl: EglContext,
}

/// Holds the GBM device and library handle (if GBM-backed display is used).
struct GbmState {
    device: *mut ffi::c_void,
    drm_fd: raw::c_int,
    destroy_device: GbmDeviceDestroyFun,
    bo_create: GbmBoCreateFun,
    bo_destroy: GbmBoDestroyFun,
    bo_get_fd: GbmBoGetFdFun,
    bo_get_stride: GbmBoGetStrideFun,
    bo_get_modifier: GbmBoGetModifierFun,
    _lib: libloading::Library,
}

impl Drop for GbmState {
    fn drop(&mut self) {
        unsafe {
            (self.destroy_device)(self.device);
            libc::close(self.drm_fd);
        }
    }
}

pub struct PlatformContext {
    inner: Mutex<ContextInner>,
    /// DMA-BUF function pointers, present when the main display supports import.
    dmabuf_fn: Option<DmaBufFunctions>,
    /// GBM state for buffer allocation and display backing.
    gbm: Option<GbmState>,
}

pub struct ContextLock<'a> {
    guard: MutexGuard<'a, ContextInner>,
}
impl<'a> std::ops::Deref for ContextLock<'a> {
    type Target = glow::Context;
    fn deref(&self) -> &Self::Target {
        &self.guard.glow
    }
}
impl<'a> Drop for ContextLock<'a> {
    fn drop(&mut self) {
        self.guard.egl.unmake_current();
    }
}

impl super::Context {
    pub unsafe fn init(desc: crate::ContextDesc) -> Result<Self, crate::NotSupportedError> {
        unsafe {
            let egl = {
                let egl_result = if cfg!(windows) {
                    egl::DynamicInstance::<egl::EGL1_4>::load_required_from_filename("libEGL.dll")
                } else if cfg!(any(
                    target_os = "macos",
                    target_os = "ios",
                    target_os = "tvos"
                )) {
                    egl::DynamicInstance::<egl::EGL1_4>::load_required_from_filename("libEGL.dylib")
                } else {
                    egl::DynamicInstance::<egl::EGL1_4>::load_required()
                };
                egl_result.map_err(crate::PlatformError::loading)?
            };

            let client_extensions = match egl.query_string(None, egl::EXTENSIONS) {
                Ok(ext) => ext.to_string_lossy().into_owned(),
                Err(_) => String::new(),
            };
            log::debug!(
                "Client extensions: {:#?}",
                client_extensions.split_whitespace().collect::<Vec<_>>()
            );

            if desc.validation && client_extensions.contains("EGL_KHR_debug") {
                log::info!("Enabling EGL debug output");
                let function: EglDebugMessageControlFun = {
                    let addr = egl.get_proc_address("eglDebugMessageControlKHR").unwrap();
                    std::mem::transmute(addr)
                };
                let attributes = [
                    EGL_DEBUG_MSG_CRITICAL_KHR as egl::Attrib,
                    1,
                    EGL_DEBUG_MSG_ERROR_KHR as egl::Attrib,
                    1,
                    EGL_DEBUG_MSG_WARN_KHR as egl::Attrib,
                    1,
                    EGL_DEBUG_MSG_INFO_KHR as egl::Attrib,
                    1,
                    egl::ATTRIB_NONE,
                ];
                (function)(Some(egl_debug_proc), attributes.as_ptr());
            }

            let angle_display = if let Some(egl1_5) = egl.upcast::<egl::EGL1_5>() {
                if client_extensions.contains("EGL_ANGLE_platform_angle") {
                    log::info!("Using Angle");
                    let display_attributes = [
                        EGL_PLATFORM_ANGLE_TYPE_ANGLE as egl::Attrib,
                        if cfg!(any(
                            target_os = "macos",
                            target_os = "ios",
                            target_os = "tvos",
                        )) {
                            EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE
                        } else {
                            EGL_PLATFORM_ANGLE_TYPE_DEFAULT_ANGLE
                        } as egl::Attrib,
                        EGL_PLATFORM_ANGLE_NATIVE_PLATFORM_TYPE_ANGLE as egl::Attrib,
                        EGL_PLATFORM_SURFACELESS_MESA as egl::Attrib,
                        EGL_PLATFORM_ANGLE_DEBUG_LAYERS_ENABLED as egl::Attrib,
                        if desc.validation { 1 } else { 0 },
                        egl::ATTRIB_NONE,
                    ];
                    Some(
                        egl1_5
                            .get_platform_display(
                                EGL_PLATFORM_ANGLE_ANGLE,
                                ptr::null_mut(),
                                &display_attributes,
                            )
                            .unwrap(),
                    )
                } else {
                    None
                }
            } else {
                None
            };

            // Try GBM platform (gives us a real DRM device for DMA-BUF export)
            let (display, gbm_state) = if let Some(display) = angle_display {
                (display, None)
            } else if let Some(egl1_5) = egl.upcast::<egl::EGL1_5>() {
                if let Some(state) = try_create_gbm_display(egl1_5, &client_extensions) {
                    (state.0, Some(state.1))
                } else if client_extensions.contains("EGL_MESA_platform_surfaceless") {
                    log::info!("Using surfaceless platform");
                    let d = egl1_5
                        .get_platform_display(
                            EGL_PLATFORM_SURFACELESS_MESA,
                            ptr::null_mut(),
                            &[egl::ATTRIB_NONE],
                        )
                        .unwrap();
                    (d, None)
                } else {
                    log::info!("Using default platform");
                    let d = egl
                        .get_display(egl::DEFAULT_DISPLAY)
                        .ok_or(crate::NotSupportedError::NoSupportedDeviceFound)?;
                    (d, None)
                }
            } else {
                (egl.get_display(egl::DEFAULT_DISPLAY).unwrap(), None)
            };

            // Load DMA-BUF export functions if available
            let dmabuf_fn = load_dmabuf_functions(&egl);

            let egl_context = EglContext::init(&desc, egl, display)?;
            egl_context.make_current();
            let (glow, capabilities, toggles, device_information, limits) =
                egl_context.load_functions(&desc);
            egl_context.unmake_current();

            Ok(Self {
                platform: PlatformContext {
                    inner: Mutex::new(ContextInner {
                        glow,
                        egl: egl_context,
                    }),
                    dmabuf_fn,
                    gbm: gbm_state,
                },
                capabilities,
                toggles,
                limits,
                device_information,
            })
        }
    }

    pub fn create_surface<
        I: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
    >(
        &self,
        window: &I,
    ) -> Result<super::Surface, crate::NotSupportedError> {
        use raw_window_handle::RawWindowHandle as Rwh;

        let window_handle = window.window_handle().unwrap().as_raw();
        let display_handle = window.display_handle().unwrap().as_raw();
        let library = match window_handle {
            Rwh::Xlib(_) => Some(find_x_library().unwrap()),
            Rwh::Xcb(_) => Some(find_x_library().unwrap()),
            Rwh::Wayland(_) => Some(find_wayland_library().unwrap()),
            _ => None,
        };

        Ok(unsafe {
            let guard = self.lock();
            super::Surface {
                platform: PlatformSurface {
                    library,
                    window_handle,
                    display_handle,
                    swapchain: None,
                    presentation: None,
                    gbm_buffer: None,
                },
                offscreen_texture: guard.create_texture().unwrap(),
                framebuf: guard.create_framebuffer().unwrap(),
            }
        })
    }

    fn destroy_wl_egl_window(surface: &super::Surface, wl_window: *mut ffi::c_void) {
        unsafe {
            let wl_egl_window_destroy: libloading::Symbol<WlEglWindowDestroyFun> = surface
                .platform
                .library
                .as_ref()
                .unwrap()
                .get(b"wl_egl_window_destroy")
                .unwrap();
            wl_egl_window_destroy(wl_window);
        }
    }

    pub fn destroy_surface(&self, surface: &mut super::Surface) {
        let inner = self.platform.inner.lock().unwrap();

        // Clean up DMA-BUF presentation path
        if let Some(pres_arc) = surface.platform.presentation.take() {
            let pres = Arc::into_inner(pres_arc)
                .expect("presentation context still referenced by in-flight frames");
            pres.egl.make_current();
            unsafe {
                pres.glow.delete_texture(pres.imported_texture);
                pres.glow.delete_framebuffer(pres.source_framebuf);
            }
            if let Some(egl1_5) = pres.egl.instance.upcast::<egl::EGL1_5>() {
                let _ = egl1_5.destroy_image(pres.egl.display, pres.imported_image);
            }
            pres.egl
                .instance
                .destroy_surface(pres.egl.display, pres.swapchain.surface)
                .unwrap();
            if let Some(wl_window) = pres.swapchain.wl_window {
                Self::destroy_wl_egl_window(surface, wl_window);
            }
            pres.egl.unmake_current();
            // EglContext::drop handles context + display cleanup
        }
        if let Some(gbm_buf) = surface.platform.gbm_buffer.take() {
            if let Some(egl1_5) = inner.egl.instance.upcast::<egl::EGL1_5>() {
                inner.egl.make_current();
                let _ = egl1_5.destroy_image(inner.egl.display, gbm_buf.main_image);
                inner.egl.unmake_current();
            }
            unsafe { libc::close(gbm_buf.fd) };
            if let Some(gbm) = self.platform.gbm.as_ref() {
                unsafe { (gbm.bo_destroy)(gbm_buf.bo) };
            }
        }

        // Clean up direct presentation path
        if let Some(s) = surface.platform.swapchain.take() {
            inner
                .egl
                .instance
                .destroy_surface(inner.egl.display, s.surface)
                .unwrap();
            if let Some(wl_window) = s.wl_window {
                Self::destroy_wl_egl_window(surface, wl_window);
            }
        }
        inner.egl.make_current();
        unsafe {
            inner.glow.delete_texture(surface.offscreen_texture);
            inner.glow.delete_framebuffer(surface.framebuf);
        }
        inner.egl.unmake_current();
    }

    pub fn reconfigure_surface(&self, surface: &mut super::Surface, config: crate::SurfaceConfig) {
        if !config.allow_exclusive_full_screen {
            log::warn!("Unable to forbid exclusive full screen");
        }

        let alpha = if config.transparent {
            crate::AlphaMode::PreMultiplied //TODO: verify
        } else {
            crate::AlphaMode::Ignored
        };
        let format = match config.color_space {
            crate::ColorSpace::Linear => crate::TextureFormat::Rgba8UnormSrgb,
            crate::ColorSpace::Srgb => crate::TextureFormat::Rgba8Unorm,
        };
        let swap_interval = match config.display_sync {
            crate::DisplaySync::Block => 1,
            crate::DisplaySync::Recent | crate::DisplaySync::Tear => 0,
        };

        let inner = self.platform.inner.lock().unwrap();

        // Try GBM-backed DMA-BUF path if available
        if let (Some(gbm), Some(dmabuf_fn), Some(egl1_5)) = (
            self.platform.gbm.as_ref(),
            self.platform.dmabuf_fn.as_ref(),
            inner.egl.instance.upcast::<egl::EGL1_5>(),
        ) {
            // Clean up old GBM buffer
            if let Some(old_buf) = surface.platform.gbm_buffer.take() {
                inner.egl.make_current();
                let _ = egl1_5.destroy_image(inner.egl.display, old_buf.main_image);
                inner.egl.unmake_current();
                unsafe {
                    libc::close(old_buf.fd);
                    (gbm.bo_destroy)(old_buf.bo);
                }
            }

            // Allocate a GBM buffer object
            let bo = unsafe {
                (gbm.bo_create)(
                    gbm.device,
                    config.size.width,
                    config.size.height,
                    GBM_FORMAT_ABGR8888,
                    GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR,
                )
            };
            if !bo.is_null() {
                let fd = unsafe { (gbm.bo_get_fd)(bo) };
                let stride = unsafe { (gbm.bo_get_stride)(bo) };
                let modifier = unsafe { (gbm.bo_get_modifier)(bo) };

                log::debug!(
                    "GBM BO created: fd={}, stride={}, modifier=0x{:x}",
                    fd,
                    stride,
                    modifier,
                );

                if fd >= 0 {
                    // Import the DMA-BUF as an EGLImage on the main context
                    let main_image = import_dmabuf_as_image(
                        egl1_5,
                        inner.egl.display,
                        fd,
                        config.size.width,
                        config.size.height,
                        GBM_FORMAT_ABGR8888 as i32,
                        stride as i32,
                        0,
                        modifier,
                    );

                    if let Some(main_image) = main_image {
                        // Bind the EGLImage to our texture on the main context
                        inner.egl.make_current();
                        unsafe {
                            let gl = &inner.glow;
                            gl.bind_texture(glow::TEXTURE_2D, Some(surface.offscreen_texture));
                            (dmabuf_fn.image_target_texture)(glow::TEXTURE_2D, main_image.as_ptr());
                            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(surface.framebuf));
                            gl.framebuffer_texture_2d(
                                glow::READ_FRAMEBUFFER,
                                glow::COLOR_ATTACHMENT0,
                                glow::TEXTURE_2D,
                                Some(surface.offscreen_texture),
                                0,
                            );
                            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
                            gl.bind_texture(glow::TEXTURE_2D, None);
                        }
                        inner.egl.unmake_current();

                        let gbm_buf = GbmBuffer {
                            bo,
                            fd,
                            stride,
                            modifier,
                            main_image,
                        };

                        // Set up presentation context
                        let pres_result = self.setup_presentation_context(
                            surface,
                            &inner.egl.instance,
                            &gbm_buf,
                            config.size,
                            format,
                            alpha,
                            swap_interval,
                            dmabuf_fn,
                        );

                        match pres_result {
                            Ok(()) => {
                                surface.platform.gbm_buffer = Some(gbm_buf);
                                return;
                            }
                            Err(e) => {
                                log::warn!(
                                    "DMA-BUF presentation setup failed: {:?}, falling back",
                                    e
                                );
                                let _ = egl1_5.destroy_image(inner.egl.display, gbm_buf.main_image);
                                unsafe {
                                    libc::close(fd);
                                    (gbm.bo_destroy)(bo);
                                }
                            }
                        }
                    } else {
                        log::warn!("Failed to import DMA-BUF as EGLImage on main context");
                        unsafe {
                            libc::close(fd);
                            (gbm.bo_destroy)(bo);
                        }
                    }
                } else {
                    log::warn!("gbm_bo_get_fd failed");
                    unsafe { (gbm.bo_destroy)(bo) };
                }
            } else {
                log::warn!("gbm_bo_create failed");
            }

            // Fall through to direct path if GBM/DMA-BUF failed
        }

        // Set up the offscreen texture directly on the main context
        let format_desc = super::describe_texture_format(format);
        inner.egl.make_current();
        unsafe {
            let gl = &inner.glow;
            gl.delete_texture(surface.offscreen_texture);
            surface.offscreen_texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(surface.offscreen_texture));
            gl.tex_storage_2d(
                glow::TEXTURE_2D,
                1,
                format_desc.internal,
                config.size.width as _,
                config.size.height as _,
            );
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(surface.framebuf));
            gl.framebuffer_texture_2d(
                glow::READ_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(surface.offscreen_texture),
                0,
            );
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
        inner.egl.unmake_current();

        // Direct path: create window surface on the main display
        self.reconfigure_surface_direct(surface, &inner, config.size, format, alpha, swap_interval);
    }

    /// Direct presentation path: create a window surface on the main EGL display.
    fn reconfigure_surface_direct(
        &self,
        surface: &mut super::Surface,
        inner: &ContextInner,
        size: crate::Extent,
        format: crate::TextureFormat,
        alpha: crate::AlphaMode,
        swap_interval: i32,
    ) {
        use raw_window_handle::RawWindowHandle as Rwh;

        let (mut temp_xlib_handle, mut temp_xcb_handle);
        let mut new_wl_window = None;
        #[allow(trivial_casts)]
        let native_window_ptr = match surface.platform.window_handle {
            Rwh::Xlib(handle) if cfg!(windows) => handle.window as *mut ffi::c_void,
            Rwh::Xlib(handle) => {
                temp_xlib_handle = handle.window;
                &mut temp_xlib_handle as *mut _ as *mut ffi::c_void
            }
            Rwh::Xcb(handle) => {
                temp_xcb_handle = handle.window;
                &mut temp_xcb_handle as *mut _ as *mut ffi::c_void
            }
            Rwh::AndroidNdk(handle) => handle.a_native_window.as_ptr(),
            Rwh::Wayland(handle) => unsafe {
                let wl_egl_window_create: libloading::Symbol<WlEglWindowCreateFun> = surface
                    .platform
                    .library
                    .as_ref()
                    .unwrap()
                    .get(b"wl_egl_window_create")
                    .unwrap();
                let wl_window = wl_egl_window_create(
                    handle.surface.as_ptr(),
                    size.width as _,
                    size.height as _,
                );
                new_wl_window = Some(wl_window);
                wl_window
            },
            Rwh::Win32(handle) => handle.hwnd.get() as *mut ffi::c_void,
            Rwh::AppKit(handle) => {
                #[cfg(not(target_os = "macos"))]
                let window_ptr = handle.ns_view.as_ptr();
                #[cfg(target_os = "macos")]
                let window_ptr = unsafe {
                    use objc2::{msg_send, runtime::Object};
                    let layer: *mut Object =
                        msg_send![handle.ns_view.as_ptr() as *mut Object, layer];
                    layer as *mut ffi::c_void
                };
                window_ptr
            }
            Rwh::OhosNdk(handle) => handle.native_window.as_ptr(),
            other => {
                panic!("Unable to connect with RWH {:?}", other);
            }
        };

        if let Some(s) = surface.platform.swapchain.take() {
            inner
                .egl
                .instance
                .destroy_surface(inner.egl.display, s.surface)
                .unwrap();
            if let Some(wl_window) = s.wl_window {
                Self::destroy_wl_egl_window(surface, wl_window);
            }
        }

        let mut attributes = vec![
            egl::RENDER_BUFFER,
            if cfg!(any(
                target_os = "android",
                target_os = "macos",
                windows,
                target_env = "ohos"
            )) {
                egl::BACK_BUFFER
            } else {
                egl::SINGLE_BUFFER
            },
        ];
        match inner.egl.srgb_kind {
            SrgbFrameBufferKind::None => {}
            SrgbFrameBufferKind::Core | SrgbFrameBufferKind::Khr => {
                attributes.push(egl::GL_COLORSPACE);
                attributes.push(egl::GL_COLORSPACE_SRGB);
            }
        }
        attributes.push(egl::ATTRIB_NONE as i32);

        let surface_window = unsafe {
            inner
                .egl
                .instance
                .create_window_surface(
                    inner.egl.display,
                    inner.egl.config,
                    native_window_ptr,
                    Some(&attributes),
                )
                .unwrap()
        };

        surface.platform.swapchain = Some(Swapchain {
            surface: surface_window,
            wl_window: new_wl_window,
            extent: size,
            info: crate::SurfaceInfo { format, alpha },
            swap_interval,
        });
    }

    /// Set up or replace the presentation context for DMA-BUF sharing.
    #[allow(clippy::too_many_arguments)]
    fn setup_presentation_context(
        &self,
        surface: &mut super::Surface,
        main_egl_instance: &EglInstance,
        gbm_buf: &GbmBuffer,
        size: crate::Extent,
        format: crate::TextureFormat,
        alpha: crate::AlphaMode,
        swap_interval: i32,
        dmabuf_fn: &DmaBufFunctions,
    ) -> Result<(), crate::NotSupportedError> {
        use raw_window_handle::RawDisplayHandle as Rdh;
        use raw_window_handle::RawWindowHandle as Rwh;

        // Determine the native display for the presentation context
        let egl1_5 = main_egl_instance
            .upcast::<egl::EGL1_5>()
            .ok_or(crate::NotSupportedError::NoSupportedDeviceFound)?;

        let pres_display = match surface.platform.display_handle {
            Rdh::Wayland(handle) => unsafe {
                egl1_5.get_platform_display(
                    EGL_PLATFORM_WAYLAND_KHR,
                    handle.display.as_ptr(),
                    &[egl::ATTRIB_NONE],
                )
            }
            .map_err(crate::PlatformError::init)?,
            Rdh::Xlib(handle) => unsafe {
                let display_ptr = match handle.display {
                    Some(d) => d.as_ptr(),
                    None => ptr::null_mut(),
                };
                egl1_5.get_platform_display(EGL_PLATFORM_X11_KHR, display_ptr, &[egl::ATTRIB_NONE])
            }
            .map_err(crate::PlatformError::init)?,
            _ => {
                return Err(crate::NotSupportedError::NoSupportedDeviceFound);
            }
        };

        // Load a separate EGL instance for the presentation context so it
        // has its own library handle (EglContext takes ownership).
        let pres_egl_instance = unsafe { egl::DynamicInstance::<egl::EGL1_4>::load_required() }
            .map_err(crate::PlatformError::loading)?;

        let desc = crate::ContextDesc {
            presentation: true,
            ..Default::default()
        };
        let mut pres_egl_context = EglContext::init(&desc, pres_egl_instance, pres_display)?;
        // The Wayland/X11 display is shared with winit, so don't terminate it on drop
        pres_egl_context.shared_display = true;

        // Create the window surface on the presentation display
        let (mut temp_xlib_handle, mut temp_xcb_handle);
        let mut new_wl_window = None;
        #[allow(trivial_casts)]
        let native_window_ptr = match surface.platform.window_handle {
            Rwh::Xlib(handle) if cfg!(windows) => handle.window as *mut ffi::c_void,
            Rwh::Xlib(handle) => {
                temp_xlib_handle = handle.window;
                &mut temp_xlib_handle as *mut _ as *mut ffi::c_void
            }
            Rwh::Xcb(handle) => {
                temp_xcb_handle = handle.window;
                &mut temp_xcb_handle as *mut _ as *mut ffi::c_void
            }
            Rwh::Wayland(handle) => unsafe {
                let wl_egl_window_create: libloading::Symbol<WlEglWindowCreateFun> = surface
                    .platform
                    .library
                    .as_ref()
                    .unwrap()
                    .get(b"wl_egl_window_create")
                    .unwrap();
                let wl_window = wl_egl_window_create(
                    handle.surface.as_ptr(),
                    size.width as _,
                    size.height as _,
                );
                new_wl_window = Some(wl_window);
                wl_window
            },
            other => {
                panic!("Unable to connect with RWH {:?}", other);
            }
        };

        let mut attributes = vec![egl::RENDER_BUFFER, egl::SINGLE_BUFFER];
        match pres_egl_context.srgb_kind {
            SrgbFrameBufferKind::None => {}
            SrgbFrameBufferKind::Core | SrgbFrameBufferKind::Khr => {
                attributes.push(egl::GL_COLORSPACE);
                attributes.push(egl::GL_COLORSPACE_SRGB);
            }
        }
        attributes.push(egl::ATTRIB_NONE as i32);

        let window_surface = unsafe {
            pres_egl_context
                .instance
                .create_window_surface(
                    pres_egl_context.display,
                    pres_egl_context.config,
                    native_window_ptr,
                    Some(&attributes),
                )
                .map_err(|e| {
                    log::error!(
                        "Failed to create window surface on presentation display: {:?}",
                        e
                    );
                    crate::PlatformError::init(e)
                })?
        };

        let swapchain = Swapchain {
            surface: window_surface,
            wl_window: new_wl_window,
            extent: size,
            info: crate::SurfaceInfo { format, alpha },
            swap_interval,
        };

        // Import the DMA-BUF as a texture on the presentation context
        pres_egl_context.make_current();

        let pres_glow = unsafe {
            glow::Context::from_loader_function(|name| {
                pres_egl_context
                    .instance
                    .get_proc_address(name)
                    .map_or(ptr::null(), |p| p as *const _)
            })
        };

        let pres_egl1_5 = pres_egl_context
            .instance
            .upcast::<egl::EGL1_5>()
            .ok_or(crate::NotSupportedError::NoSupportedDeviceFound)?;

        let imported_image = import_dmabuf_as_image(
            pres_egl1_5,
            pres_egl_context.display,
            gbm_buf.fd,
            size.width,
            size.height,
            GBM_FORMAT_ABGR8888 as i32,
            gbm_buf.stride as i32,
            0,
            gbm_buf.modifier,
        )
        .ok_or_else(|| {
            log::error!("Failed to import DMA-BUF as EGLImage on presentation display");
            crate::NotSupportedError::NoSupportedDeviceFound
        })?;

        let (imported_texture, source_framebuf) = unsafe {
            let texture = pres_glow.create_texture().unwrap();
            pres_glow.bind_texture(glow::TEXTURE_2D, Some(texture));
            (dmabuf_fn.image_target_texture)(glow::TEXTURE_2D, imported_image.as_ptr());
            pres_glow.bind_texture(glow::TEXTURE_2D, None);

            let framebuf = pres_glow.create_framebuffer().unwrap();
            pres_glow.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(framebuf));
            pres_glow.framebuffer_texture_2d(
                glow::READ_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );
            pres_glow.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            (texture, framebuf)
        };

        pres_egl_context.unmake_current();

        // Clean up old presentation context if any
        if let Some(old_pres_arc) = surface.platform.presentation.take()
            && let Some(old_pres) = Arc::into_inner(old_pres_arc)
        {
            old_pres.egl.make_current();
            unsafe {
                old_pres.glow.delete_texture(old_pres.imported_texture);
                old_pres.glow.delete_framebuffer(old_pres.source_framebuf);
            }
            if let Some(old_egl1_5) = old_pres.egl.instance.upcast::<egl::EGL1_5>() {
                let _ = old_egl1_5.destroy_image(old_pres.egl.display, old_pres.imported_image);
            }
            old_pres
                .egl
                .instance
                .destroy_surface(old_pres.egl.display, old_pres.swapchain.surface)
                .unwrap();
            if let Some(wl_window) = old_pres.swapchain.wl_window {
                Self::destroy_wl_egl_window(surface, wl_window);
            }
            old_pres.egl.unmake_current();
        }

        surface.platform.presentation = Some(Arc::new(PresentationContext {
            egl: pres_egl_context,
            glow: pres_glow,
            swapchain,
            imported_texture,
            source_framebuf,
            imported_image,
        }));

        Ok(())
    }

    pub(super) fn lock(&self) -> ContextLock<'_> {
        let inner = self.platform.inner.lock().unwrap();
        inner.egl.make_current();
        ContextLock { guard: inner }
    }
}

impl PlatformContext {
    pub(super) fn present(&self, frame: PlatformFrame) {
        match frame.present_mode {
            PresentMode::Direct(sc) => {
                let inner = self.inner.lock().unwrap();
                inner
                    .egl
                    .instance
                    .make_current(
                        inner.egl.display,
                        Some(sc.surface),
                        Some(sc.surface),
                        Some(inner.egl.raw),
                    )
                    .unwrap();
                inner
                    .egl
                    .instance
                    .swap_interval(inner.egl.display, sc.swap_interval)
                    .unwrap();

                unsafe {
                    super::present_blit(&inner.glow, frame.framebuf, sc.extent);
                }

                inner
                    .egl
                    .instance
                    .swap_buffers(inner.egl.display, sc.surface)
                    .unwrap();
                inner
                    .egl
                    .instance
                    .make_current(inner.egl.display, None, None, None)
                    .unwrap();
            }
            PresentMode::DmaBuf(ref pres) => {
                // Ensure main context rendering is complete before reading shared memory
                {
                    let inner = self.inner.lock().unwrap();
                    inner.egl.make_current();
                    unsafe {
                        inner.glow.finish();
                    }
                    inner.egl.unmake_current();
                }

                let sc = &pres.swapchain;
                pres.egl
                    .instance
                    .make_current(
                        pres.egl.display,
                        Some(sc.surface),
                        Some(sc.surface),
                        Some(pres.egl.raw),
                    )
                    .unwrap();
                pres.egl
                    .instance
                    .swap_interval(pres.egl.display, sc.swap_interval)
                    .unwrap();

                unsafe {
                    super::present_blit(&pres.glow, pres.source_framebuf, sc.extent);
                }

                pres.egl
                    .instance
                    .swap_buffers(pres.egl.display, sc.surface)
                    .unwrap();
                pres.egl
                    .instance
                    .make_current(pres.egl.display, None, None, None)
                    .unwrap();
            }
        }
    }
}

impl super::Surface {
    pub fn info(&self) -> crate::SurfaceInfo {
        if let Some(pres) = self.platform.presentation.as_ref() {
            pres.swapchain.info
        } else {
            self.platform.swapchain.as_ref().unwrap().info
        }
    }

    pub fn acquire_frame(&mut self) -> super::Frame {
        let (present_mode, extent, info) = if let Some(pres) = self.platform.presentation.as_ref() {
            let sc = &pres.swapchain;
            (PresentMode::DmaBuf(Arc::clone(pres)), sc.extent, sc.info)
        } else {
            let sc = self.platform.swapchain.as_ref().unwrap();
            (PresentMode::Direct(sc.clone()), sc.extent, sc.info)
        };
        super::Frame {
            platform: PlatformFrame {
                framebuf: self.framebuf,
                present_mode,
            },
            texture: super::Texture {
                inner: super::TextureInner::Texture {
                    raw: self.offscreen_texture,
                    target: glow::TEXTURE_2D,
                },
                target_size: [extent.width as u16, extent.height as u16],
                format: info.format,
            },
        }
    }
}

unsafe fn find_library(paths: &[&str]) -> Option<libloading::Library> {
    unsafe {
        paths
            .iter()
            .find_map(|&path| libloading::Library::new(path).ok())
    }
}
fn find_x_library() -> Option<libloading::Library> {
    unsafe { libloading::Library::new("libX11.so").ok() }
}
fn find_wayland_library() -> Option<libloading::Library> {
    unsafe { find_library(&["libwayland-egl.so.1", "libwayland-egl.so"]) }
}

/// Load DMA-BUF export/import function pointers if available.
/// Try to create a GBM-backed EGL display from a DRM render node.
/// This gives us a real DRM device with full DMA-BUF export support.
fn try_create_gbm_display(
    egl1_5: &egl::DynamicInstance<egl::EGL1_5>,
    client_extensions: &str,
) -> Option<(egl::Display, GbmState)> {
    if !client_extensions.contains("EGL_MESA_platform_gbm") {
        return None;
    }

    // Try to load libgbm dynamically
    let gbm_lib = unsafe { libloading::Library::new("libgbm.so.1") }
        .or_else(|_| unsafe { libloading::Library::new("libgbm.so") })
        .ok()?;

    let gbm_create: GbmCreateDeviceFun = unsafe { *gbm_lib.get(b"gbm_create_device\0").ok()? };
    let gbm_destroy: GbmDeviceDestroyFun = unsafe { *gbm_lib.get(b"gbm_device_destroy\0").ok()? };
    let bo_create: GbmBoCreateFun = unsafe { *gbm_lib.get(b"gbm_bo_create\0").ok()? };
    let bo_destroy: GbmBoDestroyFun = unsafe { *gbm_lib.get(b"gbm_bo_destroy\0").ok()? };
    let bo_get_fd: GbmBoGetFdFun = unsafe { *gbm_lib.get(b"gbm_bo_get_fd\0").ok()? };
    let bo_get_stride: GbmBoGetStrideFun = unsafe { *gbm_lib.get(b"gbm_bo_get_stride\0").ok()? };
    let bo_get_modifier: GbmBoGetModifierFun =
        unsafe { *gbm_lib.get(b"gbm_bo_get_modifier\0").ok()? };

    // Open a DRM render node
    let drm_fd = unsafe { libc::open(c"/dev/dri/renderD128".as_ptr(), libc::O_RDWR) };
    if drm_fd < 0 {
        log::warn!("Failed to open /dev/dri/renderD128");
        return None;
    }

    let gbm_device = unsafe { gbm_create(drm_fd) };
    if gbm_device.is_null() {
        log::warn!("gbm_create_device failed");
        unsafe { libc::close(drm_fd) };
        return None;
    }

    let display = unsafe {
        egl1_5
            .get_platform_display(EGL_PLATFORM_GBM_MESA, gbm_device, &[egl::ATTRIB_NONE])
            .ok()?
    };

    // Verify the display can be initialized
    match egl1_5.initialize(display) {
        Ok(_) => {}
        Err(e) => {
            log::warn!("GBM display initialization failed: {:?}", e);
            unsafe {
                gbm_destroy(gbm_device);
                libc::close(drm_fd);
            }
            return None;
        }
    }

    log::info!("Using GBM platform (DRM render node)");

    Some((
        display,
        GbmState {
            device: gbm_device,
            drm_fd,
            destroy_device: gbm_destroy,
            bo_create,
            bo_destroy,
            bo_get_fd,
            bo_get_stride,
            bo_get_modifier,
            _lib: gbm_lib,
        },
    ))
}

fn load_dmabuf_functions(egl: &EglInstance) -> Option<DmaBufFunctions> {
    let image_target = egl.get_proc_address("glEGLImageTargetTexture2DOES")?;
    Some(DmaBufFunctions {
        image_target_texture: unsafe {
            std::mem::transmute::<
                extern "system" fn(),
                unsafe extern "system" fn(u32, *mut libc::c_void),
            >(image_target)
        },
    })
}

/// Import a DMA-BUF fd as an EGLImage on the given display.
fn import_dmabuf_as_image(
    egl1_5: &egl::DynamicInstance<egl::EGL1_5>,
    display: egl::Display,
    fd: raw::c_int,
    width: u32,
    height: u32,
    fourcc: i32,
    stride: i32,
    offset: i32,
    modifier: u64,
) -> Option<egl::Image> {
    let attribs = [
        egl::WIDTH as egl::Attrib,
        width as egl::Attrib,
        egl::HEIGHT as egl::Attrib,
        height as egl::Attrib,
        EGL_LINUX_DRM_FOURCC_EXT,
        fourcc as egl::Attrib,
        EGL_DMA_BUF_PLANE0_FD_EXT,
        fd as egl::Attrib,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT,
        offset as egl::Attrib,
        EGL_DMA_BUF_PLANE0_PITCH_EXT,
        stride as egl::Attrib,
        EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
        (modifier & 0xFFFFFFFF) as egl::Attrib,
        EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
        (modifier >> 32) as egl::Attrib,
        egl::ATTRIB_NONE,
    ];

    // DMA-BUF import uses EGL_NO_CONTEXT
    let image = unsafe {
        egl1_5
            .create_image(
                display,
                egl::Context::from_ptr(ptr::null_mut()),
                EGL_LINUX_DMA_BUF_EXT,
                egl::ClientBuffer::from_ptr(ptr::null_mut()),
                &attribs,
            )
            .ok()?
    };

    log::debug!(
        "Imported DMA-BUF fd={} as EGLImage {:?} on display {:?}",
        fd,
        image.as_ptr(),
        display.as_ptr(),
    );

    Some(image)
}

unsafe extern "system" fn egl_debug_proc(
    error: egl::Enum,
    command_raw: *const raw::c_char,
    message_type: u32,
    _thread_label: EglLabel,
    _object_label: EglLabel,
    message_raw: *const raw::c_char,
) {
    let log_severity = match message_type {
        EGL_DEBUG_MSG_CRITICAL_KHR | EGL_DEBUG_MSG_ERROR_KHR => log::Level::Error,
        EGL_DEBUG_MSG_WARN_KHR => log::Level::Warn,
        EGL_DEBUG_MSG_INFO_KHR => log::Level::Info,
        _ => log::Level::Debug,
    };
    let command = unsafe { ffi::CStr::from_ptr(command_raw) }.to_string_lossy();
    let message = if message_raw.is_null() {
        "".into()
    } else {
        unsafe { ffi::CStr::from_ptr(message_raw) }.to_string_lossy()
    };

    log::log!(
        log_severity,
        "EGL '{}' code 0x{:x}: {}",
        command,
        error,
        message,
    );
}

const LOG_LEVEL_SEVERITY: &[(log::Level, u32)] = &[
    (log::Level::Error, glow::DEBUG_SEVERITY_HIGH),
    (log::Level::Warn, glow::DEBUG_SEVERITY_MEDIUM),
    (log::Level::Info, glow::DEBUG_SEVERITY_LOW),
    (log::Level::Trace, glow::DEBUG_SEVERITY_NOTIFICATION),
];

fn gl_debug_message_callback(source: u32, gltype: u32, id: u32, severity: u32, message: &str) {
    let source_str = match source {
        glow::DEBUG_SOURCE_API => "API",
        glow::DEBUG_SOURCE_WINDOW_SYSTEM => "Window System",
        glow::DEBUG_SOURCE_SHADER_COMPILER => "ShaderCompiler",
        glow::DEBUG_SOURCE_THIRD_PARTY => "Third Party",
        glow::DEBUG_SOURCE_APPLICATION => "Application",
        glow::DEBUG_SOURCE_OTHER => "Other",
        _ => unreachable!(),
    };

    let &(log_severity, _) = LOG_LEVEL_SEVERITY
        .iter()
        .find(|&&(_level, sev)| sev == severity)
        .unwrap();

    let type_str = match gltype {
        glow::DEBUG_TYPE_DEPRECATED_BEHAVIOR => "Deprecated Behavior",
        glow::DEBUG_TYPE_ERROR => "Error",
        glow::DEBUG_TYPE_MARKER => "Marker",
        glow::DEBUG_TYPE_OTHER => "Other",
        glow::DEBUG_TYPE_PERFORMANCE => "Performance",
        glow::DEBUG_TYPE_POP_GROUP => "Pop Group",
        glow::DEBUG_TYPE_PORTABILITY => "Portability",
        glow::DEBUG_TYPE_PUSH_GROUP => "Push Group",
        glow::DEBUG_TYPE_UNDEFINED_BEHAVIOR => "Undefined Behavior",
        _ => unreachable!(),
    };

    log::log!(
        log_severity,
        "GLES: [{}/{}] ID {} : {}",
        source_str,
        type_str,
        id,
        message
    );
}

impl EglContext {
    fn init(
        desc: &crate::ContextDesc,
        egl: EglInstance,
        display: egl::Display,
    ) -> Result<Self, crate::NotSupportedError> {
        let version = egl
            .initialize(display)
            .map_err(crate::PlatformError::init)?;
        let vendor = egl.query_string(Some(display), egl::VENDOR).unwrap();
        let display_extensions = egl
            .query_string(Some(display), egl::EXTENSIONS)
            .unwrap()
            .to_string_lossy();
        log::info!("Display vendor {:?}, version {:?}", vendor, version,);
        log::debug!(
            "Display extensions: {:#?}",
            display_extensions.split_whitespace().collect::<Vec<_>>()
        );

        let srgb_kind = if version >= (1, 5) {
            log::info!("\tEGL surface: +srgb");
            SrgbFrameBufferKind::Core
        } else if display_extensions.contains("EGL_KHR_gl_colorspace") {
            log::info!("\tEGL surface: +srgb khr");
            SrgbFrameBufferKind::Khr
        } else {
            log::warn!("\tEGL surface: -srgb");
            SrgbFrameBufferKind::None
        };

        if log::max_level() >= log::LevelFilter::Trace {
            log::trace!("Configurations:");
            let config_count = egl.get_config_count(display).unwrap();
            let mut configurations = Vec::with_capacity(config_count);
            egl.get_configs(display, &mut configurations).unwrap();
            for &config in configurations.iter() {
                log::trace!(
                    "\tCONFORMANT=0x{:X}, RENDERABLE=0x{:X}, NATIVE_RENDERABLE=0x{:X}, SURFACE_TYPE=0x{:X}, ALPHA_SIZE={}",
                    egl.get_config_attrib(display, config, egl::CONFORMANT)
                        .unwrap(),
                    egl.get_config_attrib(display, config, egl::RENDERABLE_TYPE)
                        .unwrap(),
                    egl.get_config_attrib(display, config, egl::NATIVE_RENDERABLE)
                        .unwrap(),
                    egl.get_config_attrib(display, config, egl::SURFACE_TYPE)
                        .unwrap(),
                    egl.get_config_attrib(display, config, egl::ALPHA_SIZE)
                        .unwrap(),
                );
            }
        }

        let (config, _supports_presentation) = choose_config(&egl, display, srgb_kind)?;
        egl.bind_api(egl::OPENGL_ES_API).unwrap();

        let mut khr_context_flags = 0;
        let supports_khr_context = display_extensions.contains("EGL_KHR_create_context");

        //TODO: make it so `Device` == EGL Context
        let mut context_attributes = vec![
            egl::CONTEXT_CLIENT_VERSION,
            3, // Request GLES 3.0 or higher
        ];
        if desc.validation {
            if version >= (1, 5) {
                log::info!("\tEGL context: +debug");
                context_attributes.push(egl::CONTEXT_OPENGL_DEBUG);
                context_attributes.push(egl::TRUE as _);
            } else if supports_khr_context {
                log::info!("\tEGL context: +debug KHR");
                khr_context_flags |= EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR;
            } else {
                log::info!("\tEGL context: -debug");
            }
        }
        if khr_context_flags != 0 {
            context_attributes.push(EGL_CONTEXT_FLAGS_KHR);
            context_attributes.push(khr_context_flags);
        }
        context_attributes.push(egl::NONE);
        let context = match egl.create_context(display, config, None, &context_attributes) {
            Ok(context) => context,
            Err(e) => {
                log::warn!("unable to create GLES 3.x context: {:?}", e);
                return Err(crate::PlatformError::init(e).into());
            }
        };

        // Testing if context can be binded without surface
        // and creating dummy pbuffer surface if not.
        let pbuffer =
            if version >= (1, 5) || display_extensions.contains("EGL_KHR_surfaceless_context") {
                log::info!("\tEGL context: +surfaceless");
                None
            } else {
                let attributes = [egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE];
                egl.create_pbuffer_surface(display, config, &attributes)
                    .map(Some)
                    .map_err(|e| {
                        log::warn!("Error in create_pbuffer_surface: {:?}", e);
                        crate::PlatformError::init(e)
                    })?
            };

        Ok(Self {
            instance: egl,
            display,
            raw: context,
            config,
            pbuffer,
            srgb_kind,
            shared_display: false,
        })
    }

    unsafe fn load_functions(
        &self,
        desc: &crate::ContextDesc,
    ) -> (
        glow::Context,
        super::Capabilities,
        super::Toggles,
        crate::DeviceInformation,
        super::Limits,
    ) {
        unsafe {
            let mut gl = glow::Context::from_loader_function(|name| {
                self.instance
                    .get_proc_address(name)
                    .map_or(ptr::null(), |p| p as *const _)
            });
            if desc.validation {
                if gl.supports_debug() {
                    log::info!("Enabling GLES debug output");
                    gl.enable(glow::DEBUG_OUTPUT);
                    gl.debug_message_callback(gl_debug_message_callback);
                    for &(level, severity) in LOG_LEVEL_SEVERITY.iter() {
                        gl.debug_message_control(
                            glow::DONT_CARE,
                            glow::DONT_CARE,
                            severity,
                            &[],
                            level <= log::max_level(),
                        );
                    }
                } else {
                    log::warn!("Can't enable validation");
                }
            }

            let extensions = gl.supported_extensions();
            log::debug!("Extensions: {:#?}", extensions);
            let vendor = gl.get_parameter_string(glow::VENDOR);
            let renderer = gl.get_parameter_string(glow::RENDERER);
            let version = gl.get_parameter_string(glow::VERSION);
            log::info!("Vendor: {}", vendor);
            log::info!("Renderer: {}", renderer);
            log::info!("Version: {}", version);
            let device_information = crate::DeviceInformation {
                is_software_emulated: false,
                device_name: vendor,
                driver_name: renderer,
                driver_info: version,
            };

            let mut capabilities = super::Capabilities::empty();
            capabilities.set(
                super::Capabilities::BUFFER_STORAGE,
                extensions.contains("GL_EXT_buffer_storage"),
            );
            capabilities.set(
                super::Capabilities::DRAW_BUFFERS_INDEXED,
                if gl.version().is_embedded {
                    (gl.version().major, gl.version().minor) >= (3, 2)
                } else {
                    (gl.version().major, gl.version().minor) >= (3, 0)
                },
                // glow uses unsuffixed functions like glEnablei instead of glEnableiEXT.
                // Therefore, GL_EXT_draw_buffers_indexed is not sufficient.
            );

            let toggles = super::Toggles {
                scoping: desc.capture
                    && (gl.supports_debug() || {
                        log::warn!("Scoping is not supported");
                        false
                    }),
                timing: desc.timing
                    && (extensions.contains("GL_EXT_disjoint_timer_query") || {
                        log::warn!("Timing is not supported");
                        false
                    }),
            };

            let limits = super::Limits {
                uniform_buffer_alignment: gl
                    .get_parameter_i32(glow::UNIFORM_BUFFER_OFFSET_ALIGNMENT)
                    as u32,
            };
            (gl, capabilities, toggles, device_information, limits)
        }
    }
}

impl Drop for EglContext {
    fn drop(&mut self) {
        if let Err(e) = self.instance.destroy_context(self.display, self.raw) {
            log::warn!("Error in destroy_context: {:?}", e);
        }
        if !self.shared_display
            && let Err(e) = self.instance.terminate(self.display)
        {
            log::warn!("Error in terminate: {:?}", e);
        }
    }
}

/// Choose GLES framebuffer configuration.
fn choose_config(
    egl: &EglInstance,
    display: egl::Display,
    srgb_kind: SrgbFrameBufferKind,
) -> Result<(egl::Config, bool), crate::NotSupportedError> {
    //TODO: EGL_SLOW_CONFIG
    let tiers = [
        (
            "off-screen",
            &[
                egl::SURFACE_TYPE,
                egl::PBUFFER_BIT,
                egl::RENDERABLE_TYPE,
                egl::OPENGL_ES2_BIT,
            ][..],
        ),
        ("presentation", &[egl::SURFACE_TYPE, egl::WINDOW_BIT][..]),
        #[cfg(not(any(target_os = "android", target_env = "ohos")))]
        (
            "native-render",
            &[egl::NATIVE_RENDERABLE, egl::TRUE as _][..],
        ),
    ];

    let mut attributes = Vec::with_capacity(9);
    for tier_max in (0..tiers.len()).rev() {
        let name = tiers[tier_max].0;
        log::info!("\tTrying {}", name);

        attributes.clear();
        for &(_, tier_attr) in tiers[..=tier_max].iter() {
            attributes.extend_from_slice(tier_attr);
        }
        // make sure the Alpha is enough to support sRGB
        match srgb_kind {
            SrgbFrameBufferKind::None => {}
            _ => {
                attributes.push(egl::ALPHA_SIZE);
                attributes.push(8);
            }
        }
        attributes.push(egl::NONE);

        match egl.choose_first_config(display, &attributes) {
            Ok(Some(config)) => {
                return Ok((config, tier_max >= 1));
            }
            Ok(None) => {
                log::warn!("No config found!");
            }
            Err(e) => {
                log::error!("error in choose_first_config: {:?}", e);
            }
        }
    }

    Err(crate::NotSupportedError::NoSupportedDeviceFound)
}
