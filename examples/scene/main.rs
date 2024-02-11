#![allow(irrefutable_let_patterns)]
#![cfg(not(target_arch = "wasm32"))]

use blade_graphics as gpu;
use std::{
    collections::VecDeque,
    fmt, fs,
    path::{Path, PathBuf},
    sync::Arc,
    time,
};

const FRAME_TIME_HISTORY: usize = 30;
const RENDER_WHILE_LOADING: bool = true;
const MAX_DEPTH: f32 = 1e9;

#[derive(Clone, Copy, PartialEq, strum::EnumIter)]
enum DebugBlitInput {
    None,
    SelectedBaseColor,
    SelectedNormal,
    Environment,
    EnvironmentWeight,
}
impl fmt::Display for DebugBlitInput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let desc = match *self {
            Self::None => "-",
            Self::SelectedBaseColor => "selected base color",
            Self::SelectedNormal => "selected normal",
            Self::Environment => "environment map",
            Self::EnvironmentWeight => "environment weight",
        };
        desc.fmt(f)
    }
}

struct TransformComponents {
    scale: glam::Vec3,
    rotation: glam::Quat,
    translation: glam::Vec3,
}
impl From<gpu::Transform> for TransformComponents {
    fn from(bm: gpu::Transform) -> Self {
        let transposed = glam::Mat4 {
            x_axis: bm.x.into(),
            y_axis: bm.y.into(),
            z_axis: bm.z.into(),
            w_axis: glam::Vec4::W,
        };
        let (scale, rotation, translation) = transposed.transpose().to_scale_rotation_translation();
        Self {
            scale,
            rotation,
            translation,
        }
    }
}
impl TransformComponents {
    fn to_blade(&self) -> gpu::Transform {
        let m = glam::Mat4::from_scale_rotation_translation(
            self.scale,
            self.rotation,
            self.translation,
        )
        .transpose();
        gpu::Transform {
            x: m.x_axis.into(),
            y: m.y_axis.into(),
            z: m.z_axis.into(),
        }
    }
    fn is_inversible(&self) -> bool {
        self.scale
            .x
            .abs()
            .min(self.scale.y.abs())
            .min(self.scale.z.abs())
            > 0.01
    }
}

struct ObjectExtra {
    path: PathBuf,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct ConfigCamera {
    position: mint::Vector3<f32>,
    orientation: mint::Quaternion<f32>,
    fov_y: f32,
    speed: f32,
}

fn default_transform() -> mint::RowMatrix3x4<f32> {
    gpu::IDENTITY_TRANSFORM
}
fn default_luminocity() -> f32 {
    1.0
}

#[derive(serde::Deserialize, serde::Serialize)]
struct ConfigObject {
    path: String,
    #[serde(default = "default_transform")]
    transform: mint::RowMatrix3x4<f32>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct ConfigScene {
    camera: ConfigCamera,
    #[serde(default)]
    environment_map: String,
    #[serde(default = "default_luminocity")]
    average_luminocity: f32,
    objects: Vec<ConfigObject>,
}

struct Example {
    scene_path: PathBuf,
    scene_environment_map: String,
    pacer: blade_render::util::FramePacer,
    renderer: blade_render::Renderer,
    scene_load_task: Option<choir::RunningTask>,
    gui_painter: blade_egui::GuiPainter,
    asset_hub: blade_render::AssetHub,
    context: Arc<gpu::Context>,
    environment_map: Option<blade_asset::Handle<blade_render::Texture>>,
    objects: Vec<blade_render::Object>,
    object_extras: Vec<ObjectExtra>,
    selected_object_index: Option<usize>,
    need_picked_selection_frames: usize,
    gizmo_mode: egui_gizmo::GizmoMode,
    have_objects_changed: bool,
    scene_revision: usize,
    camera: blade_render::Camera,
    fly_speed: f32,
    debug: blade_render::DebugConfig,
    track_hot_reloads: bool,
    need_accumulation_reset: bool,
    is_point_selected: bool,
    is_file_hovered: bool,
    last_render_time: time::Instant,
    render_times: VecDeque<u32>,
    ray_config: blade_render::RayConfig,
    denoiser_enabled: bool,
    denoiser_config: blade_render::DenoiserConfig,
    post_proc_config: blade_render::PostProcConfig,
    debug_blit: Option<blade_render::DebugBlit>,
    debug_blit_input: DebugBlitInput,
    workers: Vec<choir::WorkerHandle>,
    choir: Arc<choir::Choir>,
}

impl Example {
    fn make_surface_config(physical_size: winit::dpi::PhysicalSize<u32>) -> gpu::SurfaceConfig {
        gpu::SurfaceConfig {
            size: gpu::Extent {
                width: physical_size.width,
                height: physical_size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            frame_count: 3,
            color_space: gpu::ColorSpace::Linear,
        }
    }

    #[profiling::function]
    fn new(window: &winit::window::Window) -> Self {
        log::info!("Initializing");

        let context = Arc::new(unsafe {
            gpu::Context::init_windowed(
                window,
                gpu::ContextDesc {
                    validation: cfg!(debug_assertions),
                    capture: false,
                },
            )
            .unwrap()
        });

        let surface_config = Self::make_surface_config(window.inner_size());
        let screen_size = surface_config.size;
        let surface_format = context.resize(surface_config);

        let num_workers = num_cpus::get_physical().max((num_cpus::get() * 3 + 2) / 4);
        log::info!("Initializing Choir with {} workers", num_workers);
        let choir = choir::Choir::new();
        let workers = (0..num_workers)
            .map(|i| choir.add_worker(&format!("Worker-{}", i)))
            .collect();

        let asset_hub = blade_render::AssetHub::new(Path::new("asset-cache"), &choir, &context);
        let (shaders, shader_task) =
            blade_render::Shaders::load("blade-render/code/".as_ref(), &asset_hub);

        log::info!("Spinning up the renderer");
        shader_task.join();
        let mut pacer = blade_render::util::FramePacer::new(&context);
        let (command_encoder, _) = pacer.begin_frame();
        let render_config = blade_render::RenderConfig {
            screen_size,
            surface_format,
            max_debug_lines: 1000,
        };
        let renderer = blade_render::Renderer::new(
            command_encoder,
            &context,
            shaders,
            &asset_hub.shaders,
            &render_config,
        );
        pacer.end_frame(&context);
        let gui_painter = blade_egui::GuiPainter::new(surface_format, &context);

        Self {
            scene_path: PathBuf::new(),
            scene_environment_map: String::new(),
            pacer,
            renderer,
            scene_load_task: None,
            gui_painter,
            asset_hub,
            context,
            environment_map: None,
            objects: Vec::new(),
            object_extras: Vec::new(),
            selected_object_index: None,
            need_picked_selection_frames: 0,
            gizmo_mode: egui_gizmo::GizmoMode::Translate,
            have_objects_changed: false,
            scene_revision: 0,
            camera: blade_render::Camera {
                pos: mint::Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                rot: mint::Quaternion {
                    v: mint::Vector3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    s: 1.0,
                },
                fov_y: 0.0,
                depth: 0.0,
            },
            fly_speed: 0.0,
            debug: blade_render::DebugConfig::default(),
            track_hot_reloads: false,
            need_accumulation_reset: true,
            is_point_selected: false,
            is_file_hovered: false,
            last_render_time: time::Instant::now(),
            render_times: VecDeque::with_capacity(FRAME_TIME_HISTORY),
            ray_config: blade_render::RayConfig {
                num_environment_samples: 1,
                environment_importance_sampling: false,
                temporal_history: 10,
                spatial_taps: 1,
                spatial_tap_history: 5,
                spatial_radius: 10,
            },
            denoiser_enabled: true,
            denoiser_config: blade_render::DenoiserConfig {
                num_passes: 3,
                temporal_weight: 0.1,
            },
            post_proc_config: blade_render::PostProcConfig {
                average_luminocity: 1.0,
                exposure_key_value: 1.0 / 9.6,
                white_level: 1.0,
            },
            debug_blit: None,
            debug_blit_input: DebugBlitInput::None,
            workers,
            choir,
        }
    }

    fn destroy(&mut self) {
        self.workers.clear();
        self.pacer.destroy(&self.context);
        self.gui_painter.destroy(&self.context);
        self.renderer.destroy(&self.context);
        self.asset_hub.destroy();
    }

    pub fn load_scene(&mut self, scene_path: &Path) {
        if self.scene_load_task.is_some() {
            log::error!("Unable to reload the scene while something is loading");
            return;
        }

        self.objects.clear();
        self.object_extras.clear();
        self.selected_object_index = None;
        self.have_objects_changed = true;

        log::info!("Loading scene from: {}", scene_path.display());
        let config_scene: ConfigScene =
            ron::de::from_bytes(&fs::read(scene_path).expect("Unable to open the scene file"))
                .expect("Unable to parse the scene file");

        self.camera = blade_render::Camera {
            pos: config_scene.camera.position,
            rot: glam::Quat::from(config_scene.camera.orientation)
                .normalize()
                .into(),
            fov_y: config_scene.camera.fov_y,
            depth: MAX_DEPTH,
        };
        self.fly_speed = config_scene.camera.speed;
        self.ray_config.environment_importance_sampling = !config_scene.environment_map.is_empty();
        self.post_proc_config.average_luminocity = config_scene.average_luminocity;

        self.environment_map = None;
        let parent = scene_path.parent().unwrap();
        let mut load_finish = self.choir.spawn("load finish").init_dummy();

        if !config_scene.environment_map.is_empty() {
            let meta = blade_render::texture::Meta {
                format: gpu::TextureFormat::Rgba32Float,
                generate_mips: false,
                y_flip: false,
            };
            let (texture, texture_task) = self
                .asset_hub
                .textures
                .load(parent.join(&config_scene.environment_map), meta);
            load_finish.depend_on(texture_task);
            self.environment_map = Some(texture);
        }
        for config_object in config_scene.objects {
            let (model, model_task) = self.asset_hub.models.load(
                parent.join(&config_object.path),
                blade_render::model::Meta {
                    generate_tangents: true,
                    ..Default::default()
                },
            );
            load_finish.depend_on(model_task);
            self.objects.push(blade_render::Object {
                model,
                transform: config_object.transform,
                prev_transform: config_object.transform,
            });
            self.object_extras.push(ObjectExtra {
                path: PathBuf::from(config_object.path),
            });
        }

        self.scene_load_task = Some(load_finish.run());
        self.scene_path = scene_path.to_owned();
        self.scene_environment_map = config_scene.environment_map;
    }

    pub fn save_scene(&self, scene_path: &Path) {
        let config_scene = ConfigScene {
            camera: ConfigCamera {
                position: self.camera.pos,
                orientation: self.camera.rot,
                fov_y: self.camera.fov_y,
                speed: self.fly_speed,
            },
            environment_map: self.scene_environment_map.clone(),
            average_luminocity: self.post_proc_config.average_luminocity,
            objects: self
                .objects
                .iter()
                .zip(self.object_extras.iter())
                .map(|(object, extra)| ConfigObject {
                    path: extra.path.to_string_lossy().into_owned(),
                    transform: object.transform,
                })
                .collect(),
        };

        let string = ron::ser::to_string_pretty(&config_scene, ron::ser::PrettyConfig::default())
            .expect("Unable to form the scene file");
        fs::write(scene_path, &string).expect("Unable to write the scene file");
        log::info!("Saving scene to: {}", scene_path.display());
    }

    fn reset_object_motion(&mut self) {
        for object in self.objects.iter_mut() {
            object.prev_transform = object.transform;
        }
    }

    #[profiling::function]
    fn render(
        &mut self,
        gui_primitives: &[egui::ClippedPrimitive],
        gui_textures: &egui::TexturesDelta,
        physical_size: winit::dpi::PhysicalSize<u32>,
        scale_factor: f32,
    ) {
        if self.track_hot_reloads {
            self.need_accumulation_reset |= self.renderer.hot_reload(
                &self.asset_hub,
                &self.context,
                self.pacer.last_sync_point().unwrap(),
            );
        }

        // Note: the resize is split in 2 parts because `wait_for_previous_frame`
        // wants to borrow `self` mutably, and `command_encoder` blocks that.
        let surface_config = Self::make_surface_config(physical_size);
        let new_render_size = surface_config.size;
        if new_render_size != self.renderer.get_screen_size() {
            log::info!("Resizing to {}", new_render_size);
            self.pacer.wait_for_previous_frame(&self.context);
            self.context.resize(surface_config);
        }

        let (command_encoder, temp) = self.pacer.begin_frame();
        if new_render_size != self.renderer.get_screen_size() {
            self.renderer
                .resize_screen(new_render_size, command_encoder, &self.context);
            self.need_accumulation_reset = true;
        }

        self.gui_painter
            .update_textures(command_encoder, gui_textures, &self.context);

        self.asset_hub.flush(command_encoder, &mut temp.buffers);

        if let Some(ref task) = self.scene_load_task {
            if task.is_done() {
                log::info!("Scene is loaded");
                self.scene_load_task = None;
                self.have_objects_changed = true;
            }
        }

        if self.scene_load_task.is_none() && self.have_objects_changed {
            assert_eq!(self.objects.len(), self.object_extras.len());
            self.renderer.build_scene(
                command_encoder,
                &self.objects,
                self.environment_map,
                &self.asset_hub,
                &self.context,
                temp,
            );
            self.have_objects_changed = false;
            self.scene_revision += 1;
        }

        // We should be able to update TLAS and render content
        // even while it's still being loaded.
        let do_render =
            self.scene_load_task.is_none() || (RENDER_WHILE_LOADING && self.scene_revision != 0);
        if do_render {
            self.renderer.prepare(
                command_encoder,
                &self.camera,
                self.is_point_selected || self.is_file_hovered,
                self.debug.mouse_pos.is_some(),
                self.need_accumulation_reset,
            );
            self.need_accumulation_reset = false;

            //TODO: figure out why the main RT pipeline
            // causes a GPU crash when there are no objects
            if !self.objects.is_empty() {
                self.renderer
                    .ray_trace(command_encoder, self.debug, self.ray_config);
                if self.denoiser_enabled {
                    self.renderer.denoise(command_encoder, self.denoiser_config);
                }
            }
        }

        let frame = self.context.acquire_frame();
        command_encoder.init_texture(frame.texture());

        if let mut pass = command_encoder.render(gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: frame.texture_view(),
                init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        }) {
            let screen_desc = blade_egui::ScreenDescriptor {
                physical_size: (physical_size.width, physical_size.height),
                scale_factor,
            };
            if do_render {
                let mut debug_blit_array = [blade_render::DebugBlit::default()];
                let debug_blits = match self.debug_blit {
                    Some(ref blit) => {
                        debug_blit_array[0] = *blit;
                        &debug_blit_array[..]
                    }
                    None => &[],
                };
                self.renderer.post_proc(
                    &mut pass,
                    self.debug,
                    self.post_proc_config,
                    &[],
                    debug_blits,
                );
            }
            self.gui_painter
                .paint(&mut pass, gui_primitives, &screen_desc, &self.context);
        }

        command_encoder.present(frame);
        let sync_point = self.pacer.end_frame(&self.context);
        self.gui_painter.after_submit(sync_point);

        self.reset_object_motion();
    }

    fn add_manipulation_gizmo(&mut self, obj_index: usize, ui: &mut egui::Ui) {
        let view_matrix =
            glam::Mat4::from_rotation_translation(self.camera.rot.into(), self.camera.pos.into())
                .inverse();
        let extent = self.renderer.get_screen_size();
        let aspect = extent.width as f32 / extent.height as f32;
        let projection_matrix =
            glam::Mat4::perspective_rh(self.camera.fov_y, aspect, 1.0, self.camera.depth);
        let object = &mut self.objects[obj_index];
        let model_matrix = mint::ColumnMatrix4::from(mint::RowMatrix4 {
            x: object.transform.x,
            y: object.transform.y,
            z: object.transform.z,
            w: [0.0, 0.0, 0.0, 1.0].into(),
        });
        let gizmo = egui_gizmo::Gizmo::new("Object")
            .view_matrix(mint::ColumnMatrix4::from(view_matrix))
            .projection_matrix(mint::ColumnMatrix4::from(projection_matrix))
            .model_matrix(model_matrix)
            .mode(self.gizmo_mode)
            .orientation(egui_gizmo::GizmoOrientation::Global)
            .snapping(true);
        if let Some(response) = gizmo.interact(ui) {
            let t1 = TransformComponents {
                scale: response.scale,
                rotation: response.rotation,
                translation: response.translation,
            }
            .to_blade();
            if object.transform != t1 {
                object.transform = t1;
                self.have_objects_changed = true;
            }
        }
    }

    #[profiling::function]
    fn populate_view(&mut self, ui: &mut egui::Ui) {
        use strum::IntoEnumIterator as _;

        let delta = self.last_render_time.elapsed();
        self.last_render_time += delta;
        while self.render_times.len() >= FRAME_TIME_HISTORY {
            self.render_times.pop_back();
        }
        self.render_times.push_front(delta.as_millis() as u32);

        if self.scene_load_task.is_some() {
            ui.horizontal(|ui| {
                ui.label("Loading...");
                ui.spinner();
            });
            //TODO: seeing GPU Device Lost issues without this
            for task in self.asset_hub.list_running_tasks() {
                ui.label(format!("{}", task.as_ref()));
            }
            return;
        }

        let mut selection = blade_render::SelectionInfo::default();
        if self.debug.mouse_pos.is_some() {
            selection = self.renderer.read_debug_selection_info();
            if self.need_picked_selection_frames > 0 {
                self.need_picked_selection_frames -= 1;
                self.selected_object_index = self.find_object(selection.custom_index);
            }
        }

        egui::CollapsingHeader::new("Camera").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Position:");
                ui.add(egui::DragValue::new(&mut self.camera.pos.x));
                ui.add(egui::DragValue::new(&mut self.camera.pos.y));
                ui.add(egui::DragValue::new(&mut self.camera.pos.z));
            });
            ui.horizontal(|ui| {
                ui.label("Rotation:");
                ui.add(egui::DragValue::new(&mut self.camera.rot.v.x));
                ui.add(egui::DragValue::new(&mut self.camera.rot.v.y));
                ui.add(egui::DragValue::new(&mut self.camera.rot.v.z));
                ui.add(egui::DragValue::new(&mut self.camera.rot.s));
            });
            ui.add(egui::Slider::new(&mut self.camera.fov_y, 0.5f32..=2.0f32).text("FOV"));
            ui.add(
                egui::Slider::new(&mut self.fly_speed, 1f32..=100000f32)
                    .text("Fly speed")
                    .logarithmic(true),
            );
        });

        egui::CollapsingHeader::new("Debug")
            .default_open(true)
            .show(ui, |ui| {
                ui.checkbox(&mut self.track_hot_reloads, "Hot reloading");
                // debug mode
                egui::ComboBox::from_label("View mode")
                    .selected_text(format!("{:?}", self.debug.view_mode))
                    .show_ui(ui, |ui| {
                        for value in blade_render::DebugMode::iter() {
                            ui.selectable_value(
                                &mut self.debug.view_mode,
                                value,
                                format!("{value:?}"),
                            );
                        }
                    });
                // debug flags
                ui.label("Draw debug:");
                for (name, bit) in blade_render::DebugDrawFlags::all().iter_names() {
                    let mut enabled = self.debug.draw_flags.contains(bit);
                    ui.checkbox(&mut enabled, name);
                    self.debug.draw_flags.set(bit, enabled);
                }
                ui.label("Ignore textures:");
                for (name, bit) in blade_render::DebugTextureFlags::all().iter_names() {
                    let mut enabled = self.debug.texture_flags.contains(bit);
                    ui.checkbox(&mut enabled, name);
                    self.debug.texture_flags.set(bit, enabled);
                }

                // selection info
                if let Some(screen_pos) = self.debug.mouse_pos {
                    let style = ui.style();
                    egui::Frame::group(style).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Pixel:");
                            ui.colored_label(
                                egui::Color32::WHITE,
                                format!("{}x{}", screen_pos[0], screen_pos[1]),
                            );
                            if ui.button("Unselect").clicked() {
                                self.debug.mouse_pos = None;
                            }
                        });
                        ui.horizontal(|ui| {
                            let sd = &selection.std_deviation;
                            ui.label("Std Deviation:");
                            ui.colored_label(
                                egui::Color32::WHITE,
                                format!("{:.2} {:.2} {:.2}", sd.x, sd.y, sd.z),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Samples:");
                            let power = selection
                                .std_deviation_history
                                .next_power_of_two()
                                .trailing_zeros();
                            ui.colored_label(egui::Color32::WHITE, format!("2^{}", power));
                            self.need_accumulation_reset |= ui.button("Reset").clicked();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Depth:");
                            ui.colored_label(
                                egui::Color32::WHITE,
                                format!("{:.2}", selection.depth),
                            );
                        });
                        ui.horizontal(|ui| {
                            let tc = &selection.tex_coords;
                            ui.label("Texture coords:");
                            ui.colored_label(
                                egui::Color32::WHITE,
                                format!("{:.2} {:.2}", tc.x, tc.y),
                            );
                        });
                        ui.horizontal(|ui| {
                            let wp = &selection.position;
                            ui.label("World pos:");
                            ui.colored_label(
                                egui::Color32::WHITE,
                                format!("{:.2} {:.2} {:.2}", wp.x, wp.y, wp.z),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Base color:");
                            if let Some(handle) = selection.base_color_texture {
                                let name = self
                                    .asset_hub
                                    .textures
                                    .get_main_source_path(handle)
                                    .map(|path| path.display().to_string())
                                    .unwrap_or_default();
                                ui.colored_label(egui::Color32::WHITE, name);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Normal:");
                            if let Some(handle) = selection.normal_texture {
                                let name = self
                                    .asset_hub
                                    .textures
                                    .get_main_source_path(handle)
                                    .map(|path| path.display().to_string())
                                    .unwrap_or_default();
                                ui.colored_label(egui::Color32::WHITE, name);
                            }
                        });
                    });
                }

                // blits
                ui.label("Debug blit:");
                egui::ComboBox::from_label("Input")
                    .selected_text(format!("{}", self.debug_blit_input))
                    .show_ui(ui, |ui| {
                        for value in DebugBlitInput::iter() {
                            ui.selectable_value(
                                &mut self.debug_blit_input,
                                value,
                                format!("{value}"),
                            );
                        }
                    });
                let blit_view = match self.debug_blit_input {
                    DebugBlitInput::None => None,
                    DebugBlitInput::SelectedBaseColor => selection
                        .base_color_texture
                        .map(|handle| self.asset_hub.textures[handle].view),
                    DebugBlitInput::SelectedNormal => selection
                        .normal_texture
                        .map(|handle| self.asset_hub.textures[handle].view),
                    DebugBlitInput::Environment => Some(self.renderer.view_environment_main()),
                    DebugBlitInput::EnvironmentWeight => {
                        Some(self.renderer.view_environment_weight())
                    }
                };
                let min_size = 64u32;
                self.debug_blit = if let Some(view) = blit_view {
                    let mut db = match self.debug_blit.take() {
                        Some(db) => db,
                        None => {
                            let mut db = blade_render::DebugBlit::default();
                            db.target_size = [min_size, min_size];
                            db
                        }
                    };
                    db.input = view;
                    let style = ui.style();
                    egui::Frame::group(style).show(ui, |ui| {
                        ui.add(egui::Slider::new(&mut db.mip_level, 0u32..=15u32).text("Mip"));
                        ui.add(
                            egui::Slider::new(&mut db.target_size[0], min_size..=1024u32)
                                .text("Target size"),
                        );
                        db.target_size[1] = db.target_size[0];
                    });
                    Some(db)
                } else {
                    None
                };
            });

        let old_ray_config = self.ray_config;
        egui::CollapsingHeader::new("Ray Trace")
            .default_open(true)
            .show(ui, |ui| {
                let rc = &mut self.ray_config;
                ui.add(
                    egui::Slider::new(&mut rc.num_environment_samples, 1..=100u32)
                        .text("Num env samples")
                        .logarithmic(true),
                );
                ui.checkbox(
                    &mut rc.environment_importance_sampling,
                    "Env importance sampling",
                );
                ui.add(
                    egui::widgets::Slider::new(&mut rc.temporal_history, 0..=50)
                        .text("Temporal history"),
                );
                ui.add(
                    egui::widgets::Slider::new(&mut rc.spatial_taps, 0..=10).text("Spatial taps"),
                );
                ui.add(
                    egui::widgets::Slider::new(&mut rc.spatial_tap_history, 0..=50)
                        .text("Spatial tap history"),
                );
                ui.add(
                    egui::widgets::Slider::new(&mut rc.spatial_radius, 1..=50)
                        .text("Spatial radius (px)"),
                );
            });
        self.need_accumulation_reset |= self.ray_config != old_ray_config;

        egui::CollapsingHeader::new("Denoise")
            .default_open(true)
            .show(ui, |ui| {
                ui.checkbox(&mut self.denoiser_enabled, "Enable");
                let dc = &mut self.denoiser_config;
                ui.add(
                    egui::Slider::new(&mut dc.temporal_weight, 0.0..=1.0f32)
                        .text("Temporal weight"),
                );
                ui.add(egui::Slider::new(&mut dc.num_passes, 0..=5u32).text("A-trous passes"));
            });

        egui::CollapsingHeader::new("Tone Map").show(ui, |ui| {
            ui.add(
                egui::Slider::new(
                    &mut self.post_proc_config.average_luminocity,
                    0.1f32..=1_000f32,
                )
                .text("Average luminocity")
                .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(
                    &mut self.post_proc_config.exposure_key_value,
                    0.01f32..=10f32,
                )
                .text("Key value")
                .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut self.post_proc_config.white_level, 0.1f32..=2f32)
                    .text("White level"),
            );
        });

        egui::CollapsingHeader::new("Performance").show(ui, |ui| {
            let times = self.render_times.as_slices();
            let fd_points = egui_plot::PlotPoints::from_iter(
                times
                    .0
                    .iter()
                    .chain(times.1.iter())
                    .enumerate()
                    .map(|(x, &y)| [x as f64, y as f64]),
            );
            let fd_line = egui_plot::Line::new(fd_points).name("last");
            egui_plot::Plot::new("Frame time")
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_drag(false)
                .show_x(false)
                .include_y(0.0)
                .show_axes([false, true])
                .show(ui, |plot_ui| {
                    plot_ui.line(fd_line);
                    plot_ui.hline(egui_plot::HLine::new(1000.0 / 60.0).name("smooth"));
                });
        });
    }

    #[profiling::function]
    fn populate_content(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.colored_label(egui::Color32::WHITE, self.scene_path.display().to_string());
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    self.save_scene(&self.scene_path);
                }
                if ui.button("Reload").clicked() {
                    let path = self.scene_path.clone();
                    self.load_scene(&path);
                }
            });
        });

        egui::CollapsingHeader::new("Objects")
            .default_open(true)
            .show(ui, |ui| {
                for (index, extra) in self.object_extras.iter().enumerate() {
                    let name = extra.path.file_name().unwrap().to_str().unwrap();
                    ui.selectable_value(&mut self.selected_object_index, Some(index), name);
                }
            });

        if let Some(index) = self.selected_object_index {
            self.add_manipulation_gizmo(index, ui);
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    if ui.button("Unselect").clicked() {
                        self.selected_object_index = None;
                    }
                    if ui.button("Delete!").clicked() {
                        self.selected_object_index = None;
                        self.objects.remove(index);
                        self.object_extras.remove(index);
                        self.have_objects_changed = true;
                    }
                })
            });
        }

        if let Some(index) = self.selected_object_index {
            egui::CollapsingHeader::new("Transform")
                .default_open(true)
                .show(ui, |ui| {
                    let object = self.objects.get_mut(index).unwrap();
                    let mut tc = TransformComponents::from(object.transform);
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut self.gizmo_mode,
                            egui_gizmo::GizmoMode::Translate,
                            "Translate",
                        );
                        ui.add(egui::DragValue::new(&mut tc.translation.x));
                        ui.add(egui::DragValue::new(&mut tc.translation.y));
                        ui.add(egui::DragValue::new(&mut tc.translation.z));
                    });
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut self.gizmo_mode,
                            egui_gizmo::GizmoMode::Rotate,
                            "Rotate",
                        );
                        ui.add(egui::DragValue::new(&mut tc.rotation.x));
                        ui.add(egui::DragValue::new(&mut tc.rotation.y));
                        ui.add(egui::DragValue::new(&mut tc.rotation.z));
                        ui.add(egui::DragValue::new(&mut tc.rotation.w));
                    });
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut self.gizmo_mode,
                            egui_gizmo::GizmoMode::Scale,
                            "Scale",
                        );
                        ui.add(egui::DragValue::new(&mut tc.scale.x));
                        ui.add(egui::DragValue::new(&mut tc.scale.y));
                        ui.add(egui::DragValue::new(&mut tc.scale.z));
                    });

                    let transform = tc.to_blade();
                    if object.transform != transform {
                        if tc.is_inversible() {
                            object.transform = transform;
                            self.have_objects_changed = true;
                        }
                    }
                });
        }
    }

    fn find_object(&self, geometry_index: u32) -> Option<usize> {
        let mut index = geometry_index as usize;
        for (obj_index, object) in self.objects.iter().enumerate() {
            let model = &self.asset_hub.models[object.model];
            match index.checked_sub(model.geometries.len()) {
                Some(i) => index = i,
                None => return Some(obj_index),
            }
        }
        None
    }

    fn add_object(&mut self, file_path: &Path) -> bool {
        if self.scene_load_task.is_some() {
            return false;
        }

        let transform = if self.debug.mouse_pos.is_some() {
            let selection = self.renderer.read_debug_selection_info();
            //Note: assuming the object is Y-up
            let rotation = glam::Quat::from_rotation_arc(glam::Vec3::Y, selection.normal.into());
            let m = glam::Mat4::from_rotation_translation(rotation, selection.position.into())
                .transpose();
            gpu::Transform {
                x: m.x_axis.into(),
                y: m.y_axis.into(),
                z: m.z_axis.into(),
            }
        } else {
            gpu::IDENTITY_TRANSFORM
        };

        let (model, model_task) = self.asset_hub.models.load(
            file_path,
            blade_render::model::Meta {
                generate_tangents: true,
                ..Default::default()
            },
        );
        self.scene_load_task = Some(model_task.clone());
        self.objects.push(blade_render::Object {
            model,
            transform,
            prev_transform: transform,
        });
        self.object_extras.push(ObjectExtra {
            path: file_path.to_owned(),
        });
        true
    }

    fn move_camera_by(&mut self, offset: glam::Vec3) {
        let dir = glam::Quat::from(self.camera.rot) * offset;
        self.camera.pos = (glam::Vec3::from(self.camera.pos) + dir).into();
        self.debug.mouse_pos = None;
    }
    fn rotate_camera_z_by(&mut self, angle: f32) -> glam::Quat {
        let quat = glam::Quat::from(self.camera.rot);
        let rotation = glam::Quat::from_rotation_z(angle);
        self.camera.rot = (quat * rotation).into();
        self.debug.mouse_pos = None;
        rotation
    }
}

fn main() {
    env_logger::init();
    //let _ = profiling::tracy_client::Client::start();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("blade-scene")
        .build(&event_loop)
        .unwrap();

    let egui_ctx = egui::Context::default();
    let mut egui_winit = egui_winit::State::new(&event_loop);

    let mut args = std::env::args();
    let path_to_scene = args
        .nth(1)
        .unwrap_or("examples/scene/data/scene.ron".to_string());

    let mut example = Example::new(&window);
    example.load_scene(Path::new(&path_to_scene));

    struct Drag {
        _screen_pos: glam::IVec2,
        rotation: glam::Quat,
    }
    let mut drag_start = None::<Drag>;
    let mut last_event = time::Instant::now();
    let mut last_mouse_pos = [0i32; 2];

    event_loop.run(move |event, _, control_flow| {
        example.choir.check_panic();
        *control_flow = winit::event_loop::ControlFlow::Poll;
        let delta = last_event.elapsed().as_secs_f32();
        last_event = time::Instant::now();
        let move_speed = example.fly_speed * delta;
        let rotate_speed = 0.01f32;
        let rotate_speed_z = 1000.0 * delta;

        match event {
            winit::event::Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            winit::event::Event::WindowEvent { event, .. } => {
                let response = egui_winit.on_event(&egui_ctx, &event);
                if response.consumed {
                    return;
                }
                if response.repaint {
                    window.request_redraw();
                }

                match event {
                    winit::event::WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                virtual_keycode: Some(key_code),
                                state: winit::event::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => match key_code {
                        winit::event::VirtualKeyCode::Escape => {
                            *control_flow = winit::event_loop::ControlFlow::Exit;
                        }
                        winit::event::VirtualKeyCode::W => {
                            example.move_camera_by(glam::Vec3::new(0.0, 0.0, -move_speed));
                        }
                        winit::event::VirtualKeyCode::S => {
                            example.move_camera_by(glam::Vec3::new(0.0, 0.0, move_speed));
                        }
                        winit::event::VirtualKeyCode::A => {
                            example.move_camera_by(glam::Vec3::new(-move_speed, 0.0, 0.0));
                        }
                        winit::event::VirtualKeyCode::D => {
                            example.move_camera_by(glam::Vec3::new(move_speed, 0.0, 0.0));
                        }
                        winit::event::VirtualKeyCode::Z => {
                            example.move_camera_by(glam::Vec3::new(0.0, -move_speed, 0.0));
                        }
                        winit::event::VirtualKeyCode::X => {
                            example.move_camera_by(glam::Vec3::new(0.0, move_speed, 0.0));
                        }
                        winit::event::VirtualKeyCode::Q => {
                            let rot = example.rotate_camera_z_by(rotate_speed_z);
                            if let Some(ref mut drag) = drag_start {
                                drag.rotation *= rot;
                            }
                        }
                        winit::event::VirtualKeyCode::E => {
                            let rot = example.rotate_camera_z_by(-rotate_speed_z);
                            if let Some(ref mut drag) = drag_start {
                                drag.rotation *= rot;
                            }
                        }
                        _ => {}
                    },
                    winit::event::WindowEvent::CloseRequested => {
                        *control_flow = winit::event_loop::ControlFlow::Exit;
                    }
                    winit::event::WindowEvent::MouseInput {
                        state,
                        button: winit::event::MouseButton::Left,
                        ..
                    } => {
                        drag_start = match state {
                            winit::event::ElementState::Pressed => Some(Drag {
                                _screen_pos: last_mouse_pos.into(),
                                rotation: example.camera.rot.into(),
                            }),
                            winit::event::ElementState::Released => None,
                        };
                    }
                    winit::event::WindowEvent::MouseInput {
                        state: winit::event::ElementState::Pressed,
                        button: winit::event::MouseButton::Right,
                        ..
                    } => {
                        example.is_point_selected = true;
                        example.need_picked_selection_frames = 3;
                    }
                    winit::event::WindowEvent::MouseInput {
                        state: winit::event::ElementState::Released,
                        button: winit::event::MouseButton::Right,
                        ..
                    } => {
                        example.is_point_selected = false;
                    }
                    winit::event::WindowEvent::CursorMoved { position, .. } => {
                        if let Some(_) = drag_start {
                            let prev = glam::Quat::from(example.camera.rot);
                            let rotation = glam::Quat::from_euler(
                                glam::EulerRot::ZYX,
                                0.0,
                                (last_mouse_pos[0] as f32 - position.x as f32) * rotate_speed,
                                (last_mouse_pos[1] as f32 - position.y as f32) * rotate_speed,
                            );
                            example.camera.rot = (prev * rotation).into();
                            example.debug.mouse_pos = None;
                        }
                        last_mouse_pos = [position.x as i32, position.y as i32];
                    }
                    winit::event::WindowEvent::HoveredFile(_) => {
                        example.is_file_hovered = true;
                        example
                            .debug
                            .draw_flags
                            .set(blade_render::DebugDrawFlags::SPACE, true);
                    }
                    winit::event::WindowEvent::HoveredFileCancelled => {
                        example.is_file_hovered = false;
                        example
                            .debug
                            .draw_flags
                            .set(blade_render::DebugDrawFlags::SPACE, false);
                    }
                    winit::event::WindowEvent::DroppedFile(ref file_path) => {
                        example.is_file_hovered = false;
                        example
                            .debug
                            .draw_flags
                            .set(blade_render::DebugDrawFlags::SPACE, false);
                        if !example.add_object(file_path) {
                            log::warn!(
                                "Unable to drop {}, loading in progress",
                                file_path.display()
                            );
                        }
                    }
                    _ => {}
                }

                if example.is_point_selected || example.is_file_hovered {
                    //TODO: unfortunately winit doesn't update cursor position during a drag
                    // https://github.com/rust-windowing/winit/issues/1550
                    example.debug.mouse_pos = Some(last_mouse_pos);
                }
            }
            winit::event::Event::RedrawRequested(_) => {
                let mut quit = false;
                let raw_input = egui_winit.take_egui_input(&window);
                let egui_output = egui_ctx.run(raw_input, |egui_ctx| {
                    let frame = {
                        let mut frame = egui::Frame::side_top_panel(&egui_ctx.style());
                        let mut fill = frame.fill.to_array();
                        for f in fill.iter_mut() {
                            *f = (*f as u32 * 7 / 8) as u8;
                        }
                        frame.fill = egui::Color32::from_rgba_premultiplied(
                            fill[0], fill[1], fill[2], fill[3],
                        );
                        frame
                    };
                    egui::SidePanel::right("view")
                        .frame(frame)
                        .show(egui_ctx, |ui| {
                            example.populate_view(ui);
                        });
                    egui::SidePanel::left("content")
                        .frame(frame)
                        .show(egui_ctx, |ui| {
                            example.populate_content(ui);
                            ui.separator();
                            if ui.button("Quit").clicked() {
                                quit = true;
                            }
                        });
                });

                //HACK: https://github.com/urholaukkarinen/egui-gizmo/issues/29
                if example.have_objects_changed && egui_ctx.wants_pointer_input() {
                    drag_start = None;
                }

                egui_winit.handle_platform_output(&window, &egui_ctx, egui_output.platform_output);

                let primitives = egui_ctx.tessellate(egui_output.shapes);

                *control_flow = if quit {
                    winit::event_loop::ControlFlow::Exit
                } else if let Some(repaint_after_instant) =
                    std::time::Instant::now().checked_add(egui_output.repaint_after)
                {
                    winit::event_loop::ControlFlow::WaitUntil(repaint_after_instant)
                } else {
                    winit::event_loop::ControlFlow::Wait
                };

                example.render(
                    &primitives,
                    &egui_output.textures_delta,
                    window.inner_size(),
                    egui_ctx.pixels_per_point(),
                );
                profiling::finish_frame!();
            }
            winit::event::Event::LoopDestroyed => {
                example.destroy();
            }
            _ => {}
        }
    })
}
