use std::{
    ffi,
    os::raw,
    ptr,
    sync::{Arc, Mutex, MutexGuard},
};

const EGL_CONTEXT_FLAGS_KHR: i32 = 0x30FC;
const EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR: i32 = 0x0001;
const EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT: i32 = 0x30BF;
const EGL_PLATFORM_WAYLAND_KHR: u32 = 0x31D8;
const EGL_PLATFORM_X11_KHR: u32 = 0x31D5;
const EGL_PLATFORM_ANGLE_ANGLE: u32 = 0x3202;
const EGL_PLATFORM_ANGLE_NATIVE_PLATFORM_TYPE_ANGLE: u32 = 0x348F;
const EGL_PLATFORM_ANGLE_DEBUG_LAYERS_ENABLED: u32 = 0x3451;
const EGL_PLATFORM_SURFACELESS_MESA: u32 = 0x31DD;
const EGL_GL_COLORSPACE_KHR: u32 = 0x309D;
const EGL_GL_COLORSPACE_SRGB_KHR: u32 = 0x3089;

const EGL_DEBUG_MSG_CRITICAL_KHR: u32 = 0x33B9;
const EGL_DEBUG_MSG_ERROR_KHR: u32 = 0x33BA;
const EGL_DEBUG_MSG_WARN_KHR: u32 = 0x33BB;
const EGL_DEBUG_MSG_INFO_KHR: u32 = 0x33BC;

type XOpenDisplayFun =
    unsafe extern "system" fn(display_name: *const raw::c_char) -> *mut raw::c_void;

type WlDisplayConnectFun =
    unsafe extern "system" fn(display_name: *const raw::c_char) -> *mut raw::c_void;

type WlDisplayDisconnectFun = unsafe extern "system" fn(display: *const raw::c_void);

type EglInstance = egl::DynamicInstance<egl::EGL1_4>;

type WlEglWindowCreateFun = unsafe extern "system" fn(
    surface: *const raw::c_void,
    width: raw::c_int,
    height: raw::c_int,
) -> *mut raw::c_void;

type WlEglWindowResizeFun = unsafe extern "system" fn(
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
    version: (i32, i32),
    display: egl::Display,
    raw: egl::Context,
    pbuffer: Option<egl::Surface>,
    srgb_kind: SrgbFrameBufferKind,
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

#[derive(Clone, Copy, Debug, PartialEq)]
enum WindowKind {
    Wayland,
    X11,
    AngleX11,
    Unknown,
}

#[derive(Clone, Debug)]
struct WindowSystemInterface {
    library: Option<Arc<libloading::Library>>,
    kind: WindowKind,
}

pub struct Context {
    wsi: WindowSystemInterface,
    inner: Mutex<(EglContext, glow::Context)>,
}

pub struct ContextLock<'a> {
    guard: MutexGuard<'a, (EglContext, glow::Context)>,
}
impl<'a> std::ops::Deref for ContextLock<'a> {
    type Target = glow::Context;
    fn deref(&self) -> &Self::Target {
        &self.guard.1
    }
}
impl<'a> Drop for ContextLock<'a> {
    fn drop(&mut self) {
        self.guard.0.unmake_current();
    }
}

impl Context {
    pub fn init(desc: crate::ContextDesc) -> Result<Self, crate::NotSupportedError> {
        let egl = unsafe {
            let egl_result = if cfg!(windows) {
                egl::DynamicInstance::<egl::EGL1_4>::load_required_from_filename("libEGL.dll")
            } else if cfg!(any(target_os = "macos", target_os = "ios")) {
                egl::DynamicInstance::<egl::EGL1_4>::load_required_from_filename("libEGL.dylib")
            } else {
                egl::DynamicInstance::<egl::EGL1_4>::load_required()
            };
            egl_result.map_err(|_| crate::NotSupportedError)?
        };

        let client_ext_str = match egl.query_string(None, egl::EXTENSIONS) {
            Ok(ext) => ext.to_string_lossy().into_owned(),
            Err(_) => String::new(),
        };
        log::debug!(
            "Client extensions: {:#?}",
            client_ext_str.split_whitespace().collect::<Vec<_>>()
        );

        let wayland_library = if client_ext_str.contains("EGL_EXT_platform_wayland") {
            unsafe { test_wayland_display() }
        } else {
            None
        };
        let x11_display_library = if client_ext_str.contains("EGL_EXT_platform_x11") {
            unsafe { open_x_display() }
        } else {
            None
        };
        let angle_x11_display_library = if client_ext_str.contains("EGL_ANGLE_platform_angle") {
            unsafe { open_x_display() }
        } else {
            None
        };

        let egl1_5 = egl.upcast::<egl::EGL1_5>();

        let (display, wsi_library, wsi_kind) = if let (Some(library), Some(egl)) =
            (wayland_library, egl1_5)
        {
            log::info!("Using Wayland platform");
            let display_attributes = [egl::ATTRIB_NONE];
            let display = egl
                .get_platform_display(
                    EGL_PLATFORM_WAYLAND_KHR,
                    egl::DEFAULT_DISPLAY,
                    &display_attributes,
                )
                .unwrap();
            (display, Some(Arc::new(library)), WindowKind::Wayland)
        } else if let (Some((display, library)), Some(egl)) = (x11_display_library, egl1_5) {
            log::info!("Using X11 platform");
            let display_attributes = [egl::ATTRIB_NONE];
            let display = egl
                .get_platform_display(EGL_PLATFORM_X11_KHR, display.as_ptr(), &display_attributes)
                .unwrap();
            (display, Some(Arc::new(library)), WindowKind::X11)
        } else if let (Some((display, library)), Some(egl)) = (angle_x11_display_library, egl1_5) {
            log::info!("Using Angle platform with X11");
            let display_attributes = [
                EGL_PLATFORM_ANGLE_NATIVE_PLATFORM_TYPE_ANGLE as egl::Attrib,
                EGL_PLATFORM_X11_KHR as egl::Attrib,
                EGL_PLATFORM_ANGLE_DEBUG_LAYERS_ENABLED as egl::Attrib,
                if desc.validation { 1 } else { 0 },
                egl::ATTRIB_NONE,
            ];
            let display = egl
                .get_platform_display(
                    EGL_PLATFORM_ANGLE_ANGLE,
                    display.as_ptr(),
                    &display_attributes,
                )
                .unwrap();
            (display, Some(Arc::new(library)), WindowKind::AngleX11)
        } else if client_ext_str.contains("EGL_MESA_platform_surfaceless") {
            log::info!("No windowing system present. Using surfaceless platform");
            let egl = egl1_5.expect("Failed to get EGL 1.5 for surfaceless");
            let display = egl
                .get_platform_display(
                    EGL_PLATFORM_SURFACELESS_MESA,
                    std::ptr::null_mut(),
                    &[egl::ATTRIB_NONE],
                )
                .unwrap();
            (display, None, WindowKind::Unknown)
        } else {
            log::info!("EGL_MESA_platform_surfaceless not available. Using default platform");
            let display = egl.get_display(egl::DEFAULT_DISPLAY).unwrap();
            (display, None, WindowKind::Unknown)
        };

        if desc.validation && client_ext_str.contains("EGL_KHR_debug") {
            log::info!("Enabling EGL debug output");
            let function: EglDebugMessageControlFun = {
                let addr = egl.get_proc_address("eglDebugMessageControlKHR").unwrap();
                unsafe { std::mem::transmute(addr) }
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
            unsafe { (function)(Some(egl_debug_proc), attributes.as_ptr()) };
        }

        let egl_context = EglContext::init(&desc, egl, display)?;
        egl_context.make_current();
        let gl = unsafe {
            use glow::HasContext as _;
            let gl = glow::Context::from_loader_function(|name| {
                egl_context
                    .instance
                    .get_proc_address(name)
                    .map_or(ptr::null(), |p| p as *const _)
            });
            if desc.validation && gl.supports_debug() {
                log::info!("Enabling GLES debug output");
                gl.enable(glow::DEBUG_OUTPUT);
                gl.debug_message_callback(gl_debug_message_callback);
            }
            gl
        };
        egl_context.unmake_current();

        Ok(Self {
            wsi: WindowSystemInterface {
                library: wsi_library,
                kind: wsi_kind,
            },
            inner: Mutex::new((egl_context, gl)),
        })
    }

    pub(super) fn lock(&self) -> ContextLock {
        let inner = self.inner.lock().unwrap();
        inner.0.make_current();
        ContextLock { guard: inner }
    }
}

unsafe fn open_x_display() -> Option<(ptr::NonNull<raw::c_void>, libloading::Library)> {
    log::info!("Loading X11 library to get the current display");
    let library = libloading::Library::new("libX11.so").ok()?;
    let func: libloading::Symbol<XOpenDisplayFun> = library.get(b"XOpenDisplay").unwrap();
    let result = func(ptr::null());
    ptr::NonNull::new(result).map(|ptr| (ptr, library))
}

unsafe fn find_library(paths: &[&str]) -> Option<libloading::Library> {
    paths
        .iter()
        .find_map(|&path| libloading::Library::new(path).ok())
}

fn test_wayland_display() -> Option<libloading::Library> {
    // We try to connect and disconnect here to simply ensure there
    // is an active wayland display available.
    log::info!("Loading Wayland library to get the current display");
    let library = unsafe {
        let client_library = find_library(&["libwayland-client.so.0", "libwayland-client.so"])?;
        let wl_display_connect: libloading::Symbol<WlDisplayConnectFun> =
            client_library.get(b"wl_display_connect").unwrap();
        let wl_display_disconnect: libloading::Symbol<WlDisplayDisconnectFun> =
            client_library.get(b"wl_display_disconnect").unwrap();
        let display = ptr::NonNull::new(wl_display_connect(ptr::null()))?;
        wl_display_disconnect(display.as_ptr());
        find_library(&["libwayland-egl.so.1", "libwayland-egl.so"])?
    };
    Some(library)
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

    let log_severity = match severity {
        glow::DEBUG_SEVERITY_HIGH => log::Level::Error,
        glow::DEBUG_SEVERITY_MEDIUM => log::Level::Warn,
        glow::DEBUG_SEVERITY_LOW => log::Level::Info,
        glow::DEBUG_SEVERITY_NOTIFICATION => log::Level::Trace,
        _ => unreachable!(),
    };

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
            .map_err(|_| crate::NotSupportedError)?;
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
                log::trace!("\tCONFORMANT=0x{:X}, RENDERABLE=0x{:X}, NATIVE_RENDERABLE=0x{:X}, SURFACE_TYPE=0x{:X}, ALPHA_SIZE={}",
                    egl.get_config_attrib(display, config, egl::CONFORMANT).unwrap(),
                    egl.get_config_attrib(display, config, egl::RENDERABLE_TYPE).unwrap(),
                    egl.get_config_attrib(display, config, egl::NATIVE_RENDERABLE).unwrap(),
                    egl.get_config_attrib(display, config, egl::SURFACE_TYPE).unwrap(),
                    egl.get_config_attrib(display, config, egl::ALPHA_SIZE).unwrap(),
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
                return Err(crate::NotSupportedError);
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
                        crate::NotSupportedError
                    })?
            };

        Ok(Self {
            instance: egl,
            version,
            display,
            raw: context,
            pbuffer,
            srgb_kind,
        })
    }
}

impl Drop for EglContext {
    fn drop(&mut self) {
        if let Err(e) = self.instance.destroy_context(self.display, self.raw) {
            log::warn!("Error in destroy_context: {:?}", e);
        }
        if let Err(e) = self.instance.terminate(self.display) {
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
        #[cfg(not(target_os = "android"))]
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

    Err(crate::NotSupportedError)
}
