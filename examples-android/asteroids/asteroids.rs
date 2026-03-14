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

// --- Mesh generation utilities ---

fn pack4x8snorm(v: [f32; 4]) -> u32 {
    v.iter().rev().fold(0u32, |u, f| {
        (u << 8) | (f.clamp(-1.0, 1.0) * 127.0 + 0.5) as i8 as u8 as u32
    })
}

fn encode_normal(n: [f32; 3]) -> u32 {
    pack4x8snorm([n[0], n[1], n[2], 0.0])
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 1.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Simple hash-based noise for deterministic displacement.
fn hash_noise(seed: u32, x: f32, y: f32, z: f32) -> f32 {
    let ix = (x * 1000.0) as i32;
    let iy = (y * 1000.0) as i32;
    let iz = (z * 1000.0) as i32;
    let mut h = seed.wrapping_mul(374761393);
    h = h.wrapping_add((ix as u32).wrapping_mul(1103515245));
    h = h.wrapping_add((iy as u32).wrapping_mul(12345));
    h = h.wrapping_add((iz as u32).wrapping_mul(2654435761));
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    h ^= h >> 16;
    (h as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

/// Multi-octave noise for richer surface detail.
fn fbm_noise(seed: u32, x: f32, y: f32, z: f32, octaves: u32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut total_amplitude = 0.0;
    for i in 0..octaves {
        value += amplitude
            * hash_noise(
                seed.wrapping_add(i * 31),
                x * frequency,
                y * frequency,
                z * frequency,
            );
        total_amplitude += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    value / total_amplitude
}

/// Generate an icosphere with `subdivisions` levels and noise displacement.
/// `axis_scales` stretches the sphere along each axis before displacement for non-spherical shapes.
fn generate_asteroid_mesh(
    seed: u32,
    radius: f32,
    roughness: f32,
    subdivisions: u32,
    axis_scales: [f32; 3],
) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    // Base icosahedron vertices
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let base_positions: Vec<[f32; 3]> = vec![
        normalize([-1.0, phi, 0.0]),
        normalize([1.0, phi, 0.0]),
        normalize([-1.0, -phi, 0.0]),
        normalize([1.0, -phi, 0.0]),
        normalize([0.0, -1.0, phi]),
        normalize([0.0, 1.0, phi]),
        normalize([0.0, -1.0, -phi]),
        normalize([0.0, 1.0, -phi]),
        normalize([phi, 0.0, -1.0]),
        normalize([phi, 0.0, 1.0]),
        normalize([-phi, 0.0, -1.0]),
        normalize([-phi, 0.0, 1.0]),
    ];

    let base_indices: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    let mut positions = base_positions;
    let mut triangles = base_indices;

    // Subdivide
    use std::collections::HashMap;
    for _ in 0..subdivisions {
        let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut new_triangles = Vec::new();

        let get_midpoint = |positions: &mut Vec<[f32; 3]>,
                            cache: &mut HashMap<(u32, u32), u32>,
                            a: u32,
                            b: u32|
         -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&idx) = cache.get(&key) {
                return idx;
            }
            let pa = positions[a as usize];
            let pb = positions[b as usize];
            let mid = normalize([
                (pa[0] + pb[0]) * 0.5,
                (pa[1] + pb[1]) * 0.5,
                (pa[2] + pb[2]) * 0.5,
            ]);
            let idx = positions.len() as u32;
            positions.push(mid);
            cache.insert(key, idx);
            idx
        };

        for tri in triangles.iter() {
            let a = tri[0];
            let b = tri[1];
            let c = tri[2];
            let ab = get_midpoint(&mut positions, &mut midpoint_cache, a, b);
            let bc = get_midpoint(&mut positions, &mut midpoint_cache, b, c);
            let ca = get_midpoint(&mut positions, &mut midpoint_cache, c, a);
            new_triangles.push([a, ab, ca]);
            new_triangles.push([b, bc, ab]);
            new_triangles.push([c, ca, bc]);
            new_triangles.push([ab, bc, ca]);
        }
        triangles = new_triangles;
    }

    // Apply axis scaling + multi-octave noise displacement
    let displaced: Vec<[f32; 3]> = positions
        .iter()
        .map(|p| {
            let noise = fbm_noise(seed, p[0], p[1], p[2], 3);
            let r = radius * (1.0 + roughness * noise);
            [
                p[0] * r * axis_scales[0],
                p[1] * r * axis_scales[1],
                p[2] * r * axis_scales[2],
            ]
        })
        .collect();

    // Build vertices with per-face normals for hard-edge rocky look
    let mut vertices = Vec::with_capacity(triangles.len() * 3);
    let mut indices = Vec::with_capacity(triangles.len() * 3);

    for tri in triangles.iter() {
        let p0 = displaced[tri[0] as usize];
        let p1 = displaced[tri[1] as usize];
        let p2 = displaced[tri[2] as usize];

        let face_normal = normalize(cross(sub(p1, p0), sub(p2, p0)));
        let encoded_normal = encode_normal(face_normal);

        let base_idx = vertices.len() as u32;
        for &pos in &[p0, p1, p2] {
            vertices.push(blade_render::Vertex {
                position: pos,
                bitangent_sign: 1.0,
                tex_coords: [0.0, 0.0],
                normal: encoded_normal,
                tangent: encode_normal([1.0, 0.0, 0.0]),
            });
        }
        indices.push(base_idx);
        indices.push(base_idx + 1);
        indices.push(base_idx + 2);
    }

    (vertices, indices)
}

// --- Asteroid field parameters ---

const ASTEROID_COUNT: usize = 30;
const ASTEROID_MESH_VARIANTS: usize = 8;
const FIELD_RADIUS: f32 = 20.0;
const COMFORT_RADIUS: f32 = 3.0;
const DESPAWN_RADIUS: f32 = 40.0;

struct Asteroid {
    object_handle: blade_engine::ObjectHandle,
    velocity: [f32; 3],
    angular_velocity: [f32; 3],
}

struct AsteroidField {
    asteroids: Vec<Asteroid>,
    model_handles: Vec<blade_asset::Handle<blade_render::Model>>,
    next_seed: u32,
}

impl AsteroidField {
    fn new(engine: &mut blade_engine::Engine) -> Self {
        // Generate mesh variants with diverse shapes and roughness
        let variant_params: &[(f32, [f32; 3], [f32; 4])] = &[
            // (roughness, axis_scales, color)
            (0.35, [1.0, 1.0, 1.0], [0.45, 0.40, 0.35, 1.0]), // round rocky
            (0.50, [1.4, 0.7, 1.0], [0.50, 0.45, 0.38, 1.0]), // elongated
            (0.45, [1.0, 0.5, 1.2], [0.40, 0.38, 0.35, 1.0]), // flattened
            (0.60, [1.0, 1.0, 1.0], [0.35, 0.32, 0.30, 1.0]), // very rough dark
            (0.30, [0.8, 1.3, 0.9], [0.55, 0.50, 0.42, 1.0]), // tall lighter
            (0.55, [1.3, 0.8, 1.3], [0.42, 0.38, 0.32, 1.0]), // wide rough
            (0.40, [1.1, 1.1, 0.6], [0.48, 0.44, 0.38, 1.0]), // disc-like
            (0.65, [0.9, 0.9, 1.4], [0.38, 0.35, 0.30, 1.0]), // long rough dark
        ];
        let mut model_handles = Vec::new();
        for (i, &(roughness, axis_scales, color)) in variant_params.iter().enumerate() {
            let seed = (i * 7 + 42) as u32;
            let (vertices, indices) = generate_asteroid_mesh(seed, 1.0, roughness, 2, axis_scales);
            let handle = engine.create_model(
                &format!("asteroid_{i}"),
                vec![blade_render::ProceduralGeometry {
                    name: format!("asteroid_{i}"),
                    vertices,
                    indices,
                    base_color_factor: color,
                }],
            );
            model_handles.push(handle);
        }

        let mut field = AsteroidField {
            asteroids: Vec::with_capacity(ASTEROID_COUNT),
            model_handles,
            next_seed: 100,
        };

        // Spawn initial asteroids
        for _ in 0..ASTEROID_COUNT {
            field.spawn_asteroid(engine);
        }

        field
    }

    fn pick_spawn_position(&mut self) -> [f32; 3] {
        loop {
            let seed = self.next_seed;
            self.next_seed += 1;
            let x = hash_noise(seed, 0.1, 0.2, 0.3) * FIELD_RADIUS;
            let y = hash_noise(seed, 0.4, 0.5, 0.6) * FIELD_RADIUS * 0.4;
            let z = hash_noise(seed, 0.7, 0.8, 0.9) * FIELD_RADIUS;
            let dist = (x * x + y * y + z * z).sqrt();
            if dist > COMFORT_RADIUS {
                return [x, y, z];
            }
        }
    }

    fn spawn_asteroid(&mut self, engine: &mut blade_engine::Engine) {
        let pos = self.pick_spawn_position();
        let seed = self.next_seed;
        self.next_seed += 1;

        let variant = (seed as usize) % self.model_handles.len();
        let scale_factor = 0.3 + hash_noise(seed, 1.0, 2.0, 3.0).abs() * 0.7;
        let _ = scale_factor; // scale applied via similarity in add_object_with_model

        let velocity = [
            hash_noise(seed, 4.0, 5.0, 6.0) * 0.5,
            hash_noise(seed, 7.0, 8.0, 9.0) * 0.1,
            hash_noise(seed, 10.0, 11.0, 12.0) * 0.5,
        ];
        let angular_velocity = [
            hash_noise(seed, 13.0, 14.0, 15.0) * 0.3,
            hash_noise(seed, 16.0, 17.0, 18.0) * 0.3,
            hash_noise(seed, 19.0, 20.0, 21.0) * 0.3,
        ];

        let transform = blade_engine::Transform {
            position: mint::Vector3 {
                x: pos[0],
                y: pos[1],
                z: pos[2],
            },
            ..Default::default()
        };

        let handle = engine.add_object_with_model(
            "asteroid",
            self.model_handles[variant],
            transform,
            blade_engine::DynamicInput::SetVelocity,
        );

        // Set initial velocity on the rigid body
        engine.set_velocity(
            handle,
            mint::Vector3 {
                x: velocity[0],
                y: velocity[1],
                z: velocity[2],
            },
            mint::Vector3 {
                x: angular_velocity[0],
                y: angular_velocity[1],
                z: angular_velocity[2],
            },
        );

        self.asteroids.push(Asteroid {
            object_handle: handle,
            velocity,
            angular_velocity,
        });
    }

    fn update(&mut self, engine: &mut blade_engine::Engine) {
        // Recycle asteroids that have gone too far
        let mut i = 0;
        while i < self.asteroids.len() {
            let pos = engine.get_object_position(self.asteroids[i].object_handle);
            let dist = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();
            if dist > DESPAWN_RADIUS {
                let asteroid = self.asteroids.swap_remove(i);
                engine.remove_object(asteroid.object_handle);
                self.spawn_asteroid(engine);
                // Don't increment i - swap_remove moved the last element here
            } else {
                i += 1;
            }
        }
    }
}

// --- App state ---

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

    // Create asteroid field with procedural geometry
    let mut asteroid_field = AsteroidField::new(&mut engine);
    mark!(
        "XR mark: asteroid field created ({} asteroids)",
        asteroid_field.asteroids.len()
    );

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
                asteroid_field.update(&mut engine);
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
