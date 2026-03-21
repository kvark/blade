#![cfg(target_os = "android")]

use std::{
    ffi::CString,
    fs,
    io::Read,
    path::Path,
    path::PathBuf,
    time::{Duration, Instant},
};

use android_activity::{AndroidApp, InputStatus, MainEvent, PollEvent};
use blade_graphics as gpu;
use log::info;
use openxr as xr;

mod game;
mod mesh;

use game::{AsteroidField, GameState, LASER_DAMAGE, SIZE_RADII};

// --- XR Input ---

const LASER_LENGTH: f32 = 60.0;
const LASER_BEAM_RADIUS: f32 = 0.015;

struct HandState {
    trigger_pressed: bool,
    aim_pose: Option<xr::Posef>,
}

struct XrInput {
    action_set: xr::ActionSet,
    trigger_action: xr::Action<f32>,
    aim_action: xr::Action<xr::Posef>,
    left_aim_space: xr::Space,
    right_aim_space: xr::Space,
    hand_paths: [xr::Path; 2],
    hands: [HandState; 2],
    laser_model: blade_asset::Handle<blade_render::Model>,
    aim_model: blade_asset::Handle<blade_render::Model>,
    laser_objects: [Option<blade_engine::ObjectHandle>; 2],
    laser_firing: [bool; 2],
}

impl XrInput {
    fn new(
        instance: &xr::Instance,
        session: &xr::Session<xr::Vulkan>,
        engine: &mut blade_engine::Engine,
    ) -> Self {
        let action_set = instance
            .create_action_set("gameplay", "Gameplay", 0)
            .unwrap();

        let trigger_action = action_set
            .create_action::<f32>(
                "trigger",
                "Trigger",
                &[
                    instance.string_to_path("/user/hand/left").unwrap(),
                    instance.string_to_path("/user/hand/right").unwrap(),
                ],
            )
            .unwrap();

        let aim_action = action_set
            .create_action::<xr::Posef>(
                "aim_pose",
                "Aim Pose",
                &[
                    instance.string_to_path("/user/hand/left").unwrap(),
                    instance.string_to_path("/user/hand/right").unwrap(),
                ],
            )
            .unwrap();

        let left_trigger_path = instance
            .string_to_path("/user/hand/left/input/trigger/value")
            .unwrap();
        let right_trigger_path = instance
            .string_to_path("/user/hand/right/input/trigger/value")
            .unwrap();
        let left_aim_path = instance
            .string_to_path("/user/hand/left/input/aim/pose")
            .unwrap();
        let right_aim_path = instance
            .string_to_path("/user/hand/right/input/aim/pose")
            .unwrap();

        let trigger_aim_bindings = [
            xr::Binding::new(&trigger_action, left_trigger_path),
            xr::Binding::new(&trigger_action, right_trigger_path),
            xr::Binding::new(&aim_action, left_aim_path),
            xr::Binding::new(&aim_action, right_aim_path),
        ];

        let profiles = [
            "/interaction_profiles/oculus/touch_controller",
            "/interaction_profiles/meta/touch_controller_plus",
        ];
        for profile_path in profiles {
            if let Ok(profile) = instance.string_to_path(profile_path) {
                match instance.suggest_interaction_profile_bindings(profile, &trigger_aim_bindings)
                {
                    Ok(()) => info!("Bound input profile: {profile_path}"),
                    Err(e) => info!("Failed to bind {profile_path}: {e}"),
                }
            }
        }

        if let (Ok(profile), Ok(left_select), Ok(right_select)) = (
            instance.string_to_path("/interaction_profiles/khr/simple_controller"),
            instance.string_to_path("/user/hand/left/input/select/click"),
            instance.string_to_path("/user/hand/right/input/select/click"),
        ) {
            let simple_bindings = [
                xr::Binding::new(&trigger_action, left_select),
                xr::Binding::new(&trigger_action, right_select),
                xr::Binding::new(&aim_action, left_aim_path),
                xr::Binding::new(&aim_action, right_aim_path),
            ];
            match instance.suggest_interaction_profile_bindings(profile, &simple_bindings) {
                Ok(()) => info!("Bound KHR simple controller profile"),
                Err(e) => info!("Failed to bind KHR simple: {e}"),
            }
        }

        session.attach_action_sets(&[&action_set]).unwrap();

        let left_hand = instance.string_to_path("/user/hand/left").unwrap();
        let right_hand = instance.string_to_path("/user/hand/right").unwrap();

        let left_aim_space = aim_action
            .create_space(session.clone(), left_hand, xr::Posef::IDENTITY)
            .unwrap();
        let right_aim_space = aim_action
            .create_space(session.clone(), right_hand, xr::Posef::IDENTITY)
            .unwrap();

        let (laser_verts, laser_idxs) = mesh::generate_laser_mesh(LASER_LENGTH, LASER_BEAM_RADIUS);
        let laser_model = engine.create_model(
            "laser_beam",
            vec![blade_render::ProceduralGeometry {
                name: "laser_beam".to_string(),
                vertices: laser_verts,
                indices: laser_idxs,
                base_color_factor: [0.2, 1.0, 0.2, 1.0],
            }],
        );
        let (aim_verts, aim_idxs) = mesh::generate_laser_mesh(3.0, LASER_BEAM_RADIUS * 0.5);
        let aim_model = engine.create_model(
            "aim_line",
            vec![blade_render::ProceduralGeometry {
                name: "aim_line".to_string(),
                vertices: aim_verts,
                indices: aim_idxs,
                base_color_factor: [0.1, 0.3, 0.6, 1.0],
            }],
        );

        XrInput {
            action_set,
            trigger_action,
            aim_action,
            left_aim_space,
            right_aim_space,
            hand_paths: [left_hand, right_hand],
            hands: [
                HandState {
                    trigger_pressed: false,
                    aim_pose: None,
                },
                HandState {
                    trigger_pressed: false,
                    aim_pose: None,
                },
            ],
            laser_model,
            aim_model,
            laser_objects: [None, None],
            laser_firing: [false, false],
        }
    }

    fn sync(
        &mut self,
        engine: &mut blade_engine::Engine,
        session: &xr::Session<xr::Vulkan>,
        frame_count: u64,
    ) {
        if let Err(e) = session.sync_actions(&[xr::ActiveActionSet::new(&self.action_set)]) {
            if frame_count <= 5 || frame_count % 300 == 0 {
                info!("sync_actions failed: {e}");
            }
            return;
        }

        let aim_spaces = [&self.left_aim_space, &self.right_aim_space];
        for (i, aim_space) in aim_spaces.iter().enumerate() {
            let trigger_state = self.trigger_action.state(session, self.hand_paths[i]).ok();
            self.hands[i].trigger_pressed = trigger_state
                .map(|s| s.current_state > 0.5)
                .unwrap_or(false);

            self.hands[i].aim_pose = engine.xr_locate_space(aim_space);

            if frame_count <= 5 || frame_count % 300 == 0 {
                let trigger_val = trigger_state.map(|s| s.current_state).unwrap_or(-1.0);
                let has_pose = self.hands[i].aim_pose.is_some();
                info!("Hand {i}: trigger={trigger_val:.2}, pose={has_pose}",);
            }
        }
    }

    fn update(
        &mut self,
        engine: &mut blade_engine::Engine,
        asteroid_field: &mut AsteroidField,
        laser_hit_ps: usize,
    ) {
        for hand in &self.hands {
            if !hand.trigger_pressed {
                continue;
            }
            let pose = match hand.aim_pose {
                Some(p) => p,
                None => continue,
            };

            let origin = [pose.position.x, pose.position.y, pose.position.z];
            let q = pose.orientation;
            let dir = mesh::quat_rotate([q.x, q.y, q.z, q.w], [0.0, 0.0, -1.0]);

            let mut closest_t = LASER_LENGTH;
            let mut closest_idx = None;
            for (idx, asteroid) in asteroid_field.asteroids.iter().enumerate() {
                let pos = engine.get_object_position(asteroid.object_handle);
                let to_asteroid = [pos.x - origin[0], pos.y - origin[1], pos.z - origin[2]];
                let along =
                    to_asteroid[0] * dir[0] + to_asteroid[1] * dir[1] + to_asteroid[2] * dir[2];
                let hit_radius = SIZE_RADII[asteroid.size_class];
                let perp_sq = (to_asteroid[0] - dir[0] * along).powi(2)
                    + (to_asteroid[1] - dir[1] * along).powi(2)
                    + (to_asteroid[2] - dir[2] * along).powi(2);
                if perp_sq >= hit_radius * hit_radius {
                    continue;
                }
                // Ray enters the sphere at t = along - sqrt(r² - perp²)
                let t_enter = along - (hit_radius * hit_radius - perp_sq).sqrt();
                if t_enter < 0.0 || t_enter > closest_t {
                    continue;
                }
                closest_t = t_enter;
                closest_idx = Some(idx);
            }

            if let Some(idx) = closest_idx {
                asteroid_field.asteroids[idx].health -= LASER_DAMAGE;
                // Hit point is where the ray enters the sphere (near side)
                let hit_point = [
                    origin[0] + dir[0] * closest_t,
                    origin[1] + dir[1] * closest_t,
                    origin[2] + dir[2] * closest_t,
                ];
                engine.particle_burst(laser_hit_ps, 3, hit_point);
            }
        }
    }

    fn pose_to_transform(p: &xr::Posef) -> blade_engine::Transform {
        blade_engine::Transform {
            position: mint::Vector3 {
                x: p.position.x,
                y: p.position.y,
                z: p.position.z,
            },
            orientation: mint::Quaternion {
                s: p.orientation.w,
                v: mint::Vector3 {
                    x: p.orientation.x,
                    y: p.orientation.y,
                    z: p.orientation.z,
                },
            },
        }
    }

    fn update_laser_objects(&mut self, engine: &mut blade_engine::Engine) {
        for (i, hand) in self.hands.iter().enumerate() {
            let firing = hand.trigger_pressed;
            let has_pose = hand.aim_pose.is_some();

            if let Some(handle) = self.laser_objects[i] {
                if !has_pose || firing != self.laser_firing[i] {
                    engine.remove_object(handle);
                    self.laser_objects[i] = None;
                }
            }

            if let Some(pose) = hand.aim_pose {
                let transform = Self::pose_to_transform(&pose);
                if let Some(handle) = self.laser_objects[i] {
                    engine.teleport_object(handle, transform);
                } else {
                    let model = if firing {
                        self.laser_model
                    } else {
                        self.aim_model
                    };
                    let handle = engine.add_object_with_model(
                        "laser",
                        model,
                        transform,
                        blade_engine::DynamicInput::Full,
                    );
                    self.laser_objects[i] = Some(handle);
                    self.laser_firing[i] = firing;
                }
            }
        }
    }
}

// --- App state ---

#[derive(Clone, Copy, PartialEq, Eq)]
enum Visibility {
    Hidden,
    Visible,
    Focused,
}

enum AppState {
    Idle,
    Running {
        rendered_frames: u64,
        game: GameState,
        visibility: Visibility,
    },
}

fn prepare_shader_dir(app: &AndroidApp) -> PathBuf {
    let internal_path = app
        .internal_data_path()
        .expect("No internal data path available");
    let shader_dir = internal_path.join("blade-render").join("code");
    fs::create_dir_all(&shader_dir).unwrap();
    copy_assets_to_dir(app, "", &shader_dir);

    shader_dir
}

fn copy_assets_to_dir(app: &AndroidApp, asset_dir_name: &str, output_dir: &Path) {
    let asset_manager = app.asset_manager();
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
        let mut contents = Vec::with_capacity(asset.length());
        asset.read_to_end(&mut contents).unwrap();
        fs::write(output_dir.join(file_name.as_ref()), contents).unwrap();
        copied += 1;
    }
    assert!(
        copied > 0,
        "No WGSL files copied from asset dir '{asset_dir_name}'"
    );
}

#[no_mangle]
fn android_main(app: AndroidApp) {
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
    if available_extensions.meta_touch_controller_plus {
        enabled_extensions.meta_touch_controller_plus = true;
        info!("Enabling XR_META_touch_controller_plus");
    }

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
    let shader_dir = prepare_shader_dir(&app);
    mark!("XR mark: shader dir prepared");
    let internal_path = app
        .internal_data_path()
        .expect("No internal data path available");
    let cache_path = internal_path.join("asset-cache");
    let mut engine = blade_engine::Engine::new(
        blade_engine::Presentation::Xr(gpu::XrDesc {
            instance: xr_instance.clone(),
            system_id: system,
        }),
        &blade_engine::config::Engine {
            shader_path: shader_dir.to_string_lossy().into_owned(),
            data_path: String::new(),
            cache_path: cache_path.to_string_lossy().into_owned(),
            time_step: 0.01,
            render_backend: blade_engine::config::RenderBackend::Rasterizer,
            gui_enabled: false,
        },
    );
    mark!("XR mark: engine created");

    let mut state = AppState::Idle;
    let mut xr_input: Option<XrInput> = None;
    let mut resumed = false;
    let mut destroy_requested = false;
    let mut event_storage = xr::EventDataBuffer::new();
    let mut last_heartbeat = Instant::now();
    let mut frame_start;
    let mut frame_time_sum = 0.0f64;
    let mut frame_time_max = 0.0f64;
    let mut frame_count_in_period = 0u32;
    mark!("XR mark: app initialized; entering main loop");

    'main_loop: loop {
        let session_running = matches!(state, AppState::Running { .. });
        let timeout = if !resumed && !session_running {
            Some(Duration::from_millis(250))
        } else {
            Some(Duration::ZERO)
        };
        app.poll_events(timeout, |event| match event {
            PollEvent::Main(main_event) => {
                if xr_debug {
                    info!("Android event: {:?}", main_event);
                }
                match main_event {
                    MainEvent::Resume { .. } => resumed = true,
                    MainEvent::Pause => resumed = false,
                    MainEvent::Destroy => destroy_requested = true,
                    MainEvent::InputAvailable => {
                        if let Ok(mut iter) = app.input_events_iter() {
                            loop {
                                if !iter.next(|_event| InputStatus::Unhandled) {
                                    break;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        });
        if destroy_requested {
            break 'main_loop;
        }

        if last_heartbeat.elapsed() >= Duration::from_secs(1) {
            if xr_debug {
                let (session_running, rendered_frames, asteroid_count) = match state {
                    AppState::Idle => (false, 0, 0),
                    AppState::Running {
                        rendered_frames,
                        ref game,
                        ..
                    } => (true, rendered_frames, game.asteroid_field.asteroids.len()),
                };
                let avg_ms = if frame_count_in_period > 0 {
                    frame_time_sum / frame_count_in_period as f64 * 1000.0
                } else {
                    0.0
                };
                info!(
                    "XR heartbeat: running={}, frames={}, avg={:.1}ms, max={:.1}ms, asteroids={}",
                    session_running,
                    rendered_frames,
                    avg_ms,
                    frame_time_max * 1000.0,
                    asteroid_count,
                );
            }
            frame_time_sum = 0.0;
            frame_time_max = 0.0;
            frame_count_in_period = 0;
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
                            mark!("XR mark: setting up game");
                            let game = game::setup_game(&mut engine);
                            if xr_input.is_none() {
                                let session = engine.xr_session().expect("XR session required");
                                xr_input = Some(XrInput::new(&xr_instance, &session, &mut engine));
                            }
                            mark!("XR mark: game setup done");
                            state = AppState::Running {
                                rendered_frames: 0,
                                game,
                                visibility: Visibility::Hidden,
                            };
                            mark!("XR mark: calling engine.begin_xr");
                            engine.begin_xr();
                            mark!("XR mark: engine.begin_xr returned");
                        }
                        xr::SessionState::FOCUSED => {
                            if let AppState::Running {
                                ref mut visibility, ..
                            } = state
                            {
                                *visibility = Visibility::Focused;
                            }
                        }
                        xr::SessionState::VISIBLE => {
                            if let AppState::Running {
                                ref mut visibility, ..
                            } = state
                            {
                                *visibility = Visibility::Visible;
                            }
                        }
                        xr::SessionState::SYNCHRONIZED => {
                            if let AppState::Running {
                                ref mut visibility, ..
                            } = state
                            {
                                *visibility = Visibility::Hidden;
                            }
                        }
                        xr::SessionState::STOPPING => {
                            if xr_debug {
                                info!("XR state STOPPING -> end session");
                            }
                            if let Some(ref mut input) = xr_input {
                                for obj in input.laser_objects.iter_mut() {
                                    if let Some(handle) = obj.take() {
                                        engine.remove_object(handle);
                                    }
                                }
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

        match state {
            AppState::Idle => {}
            AppState::Running {
                ref mut rendered_frames,
                ref mut game,
                visibility,
            } => {
                if visibility != Visibility::Focused {
                    if !engine.render_xr() {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    continue;
                }
                frame_start = Instant::now();
                if let (Some(ref mut input), Some(session)) = (&mut xr_input, engine.xr_session()) {
                    input.sync(&mut engine, &session, *rendered_frames);
                    input.update(&mut engine, &mut game.asteroid_field, game.laser_hit_ps);
                    input.update_laser_objects(&mut engine);
                }
                game.asteroid_field
                    .update(&mut engine, game.explosion_ps, 0.016);
                game.comet_field.update(&mut engine);
                engine.update(0.016);
                // Process physics collision events — only between asteroids
                let contacts: Vec<_> = engine.drain_contacts().collect();
                for contact in contacts {
                    let is_asteroid_a = game
                        .asteroid_field
                        .asteroids
                        .iter()
                        .any(|a| a.object_handle == contact.object_a);
                    let is_asteroid_b = game
                        .asteroid_field
                        .asteroids
                        .iter()
                        .any(|a| a.object_handle == contact.object_b);
                    if is_asteroid_a && is_asteroid_b {
                        engine.particle_burst(
                            game.collision_ps,
                            15,
                            [contact.position.x, contact.position.y, contact.position.z],
                        );
                    }
                }
                if engine.render_xr() {
                    let dt = frame_start.elapsed().as_secs_f64();
                    frame_time_sum += dt;
                    if dt > frame_time_max {
                        frame_time_max = dt;
                    }
                    frame_count_in_period += 1;
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
