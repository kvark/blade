#![cfg(target_os = "android")]

use std::{
    ffi::CString,
    fs,
    io::Read,
    path::Path,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use blade_graphics as gpu;
use log::info;
use openxr as xr;

#[derive(Default)]
enum AppState {
    #[default]
    Idle,
    Running {
        rendered_frames: u64,
    },
}

fn prepare_shader_dir() -> PathBuf {
    let shader_dir = ndk_glue::native_activity()
        .internal_data_path()
        .join("blade-render")
        .join("code");
    fs::create_dir_all(&shader_dir).unwrap();
    copy_assets_to_dir("", &shader_dir);

    shader_dir
}

fn copy_assets_to_dir(asset_dir_name: &str, output_dir: &Path) {
    let asset_manager = ndk_glue::native_activity().asset_manager();
    let dir_name = CString::new(asset_dir_name).unwrap();
    let mut asset_dir = asset_manager
        .open_dir(&dir_name)
        .unwrap_or_else(|| panic!("Unable to open asset dir '{asset_dir_name}'"));

    let mut copied = 0usize;
    for file_name in &mut asset_dir {
        let file_name = file_name.to_string_lossy();
        if !file_name.ends_with(".wgsl") {
            continue;
        }
        let asset_path = if asset_dir_name.is_empty() {
            file_name.to_string()
        } else {
            format!("{asset_dir_name}/{file_name}")
        };
        let asset_path_c = CString::new(asset_path.as_str()).unwrap();
        let mut asset = asset_manager
            .open(&asset_path_c)
            .unwrap_or_else(|| panic!("Unable to open asset '{asset_path}'"));
        let mut contents = Vec::with_capacity(asset.get_length());
        asset.read_to_end(&mut contents).unwrap();
        fs::write(output_dir.join(file_name.as_ref()), contents).unwrap();
        copied += 1;
    }
    assert!(
        copied > 0,
        "No WGSL files copied from asset dir '{asset_dir_name}'"
    );
}

fn spawn_android_event_pump(xr_debug: bool) -> Arc<AtomicBool> {
    let should_exit = Arc::new(AtomicBool::new(false));
    let should_exit_thread = Arc::clone(&should_exit);
    thread::spawn(move || {
        while let Some(event) = ndk_glue::poll_events() {
            if xr_debug {
                info!("Android event: {:?}", event);
            }
            if matches!(event, ndk_glue::Event::Destroy) {
                should_exit_thread.store(true, Ordering::Relaxed);
                break;
            }
        }
    });
    should_exit
}

#[ndk_glue::main]
pub fn main() {
    let xr_debug = std::env::var_os("BLADE_XR_DEBUG").is_some();

    macro_rules! mark {
        ($($arg:tt)*) => {{
            if xr_debug {
                let msg = format!($($arg)*);
                info!("{}", msg);
                eprintln!("{}", msg);
            }
        }};
    }

    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("blade-asteroids"),
    );
    mark!("XR mark: logger initialized");

    let entry = unsafe {
        xr::Entry::load()
            .expect("couldn't find the OpenXR loader; try enabling the \"static\" feature")
    };
    mark!("XR mark: OpenXR entry loaded");
    entry.initialize_android_loader().unwrap();
    mark!("XR mark: Android loader initialized");

    let available_extensions = entry.enumerate_extensions().unwrap();
    assert!(available_extensions.khr_vulkan_enable2);

    let mut enabled_extensions = xr::ExtensionSet::default();
    enabled_extensions.khr_vulkan_enable2 = true;
    enabled_extensions.khr_android_create_instance = true;

    let xr_instance = entry
        .create_instance(
            &xr::ApplicationInfo {
                application_name: "Blade Asteroids",
                application_version: 0,
                engine_name: "Blade",
                engine_version: 0,
                api_version: xr::Version::new(1, 0, 0),
            },
            &enabled_extensions,
            &[],
        )
        .unwrap();
    mark!("XR mark: OpenXR instance created");

    let system = xr_instance
        .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
        .unwrap();
    mark!("XR mark: OpenXR system acquired");
    let shader_dir = prepare_shader_dir();
    mark!("XR mark: shader dir prepared");
    let mut engine = blade_engine::Engine::new(
        blade_engine::Presentation::Xr(gpu::XrDesc {
            instance: xr_instance.clone(),
            system_id: system,
        }),
        &blade_engine::config::Engine {
            shader_path: shader_dir.to_string_lossy().into_owned(),
            data_path: String::new(),
            time_step: 0.01,
            render_backend: blade_engine::config::RenderBackend::Rasterizer,
            gui_enabled: false,
        },
    );
    mark!("XR mark: engine created");

    let mut state = AppState::Idle;
    let should_exit = spawn_android_event_pump(xr_debug);
    let mut event_storage = xr::EventDataBuffer::new();
    let mut last_heartbeat = Instant::now();
    mark!("XR mark: app initialized; entering main loop");

    'main_loop: loop {
        if should_exit.load(Ordering::Relaxed) {
            break 'main_loop;
        }

        if last_heartbeat.elapsed() >= Duration::from_secs(1) {
            if xr_debug {
                let (session_running, rendered_frames) = match &state {
                    AppState::Idle => (false, 0),
                    AppState::Running {
                        rendered_frames, ..
                    } => (true, *rendered_frames),
                };
                info!(
                    "XR heartbeat: session_running={}, rendered_frames={}",
                    session_running, rendered_frames
                );
            }
            last_heartbeat = Instant::now();
        }

        while let Some(event) = xr_instance.poll_event(&mut event_storage).unwrap() {
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => {
                    if xr_debug {
                        info!("XR session state changed: {:?}", e.state());
                    }
                    match e.state() {
                        xr::SessionState::READY => {
                            if xr_debug {
                                info!("XR state READY -> begin session");
                            }
                            state = AppState::Running { rendered_frames: 0 };
                            mark!("XR mark: calling engine.begin_xr");
                            engine.begin_xr();
                            mark!("XR mark: engine.begin_xr returned");
                        }
                        xr::SessionState::STOPPING => {
                            if xr_debug {
                                info!("XR state STOPPING -> end session");
                            }
                            engine.end_xr();
                            state = AppState::Idle;
                        }
                        xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                            if xr_debug {
                                info!("XR state {:?} -> exiting main loop", e.state());
                            }
                            break 'main_loop;
                        }
                        _ => {}
                    }
                }
                InstanceLossPending(_) => {
                    if xr_debug {
                        info!("XR instance loss pending -> exiting main loop");
                    }
                    break 'main_loop;
                }
                _ => {}
            }
        }

        match &mut state {
            AppState::Idle => std::thread::sleep(Duration::from_millis(100)),
            AppState::Running { rendered_frames } => {
                engine.update(0.016);
                if engine.render_xr() {
                    *rendered_frames += 1;
                    if xr_debug && (*rendered_frames <= 5 || *rendered_frames % 120 == 0) {
                        info!("XR frame submitted: {}", *rendered_frames);
                    }
                }
            }
        }
    }

    if matches!(state, AppState::Running { .. }) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.end_xr()));
    }
    engine.destroy();
}
