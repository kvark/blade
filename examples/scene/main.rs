#![allow(irrefutable_let_patterns)]
#![cfg(not(target_arch = "wasm32"))]

use blade_graphics as gpu;
use blade_helpers::ControlledCamera;
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
impl From<transform_gizmo_egui::math::Transform> for TransformComponents {
    fn from(t: transform_gizmo_egui::math::Transform) -> Self {
        Self {
            scale: glam::DVec3::from(t.scale).as_vec3(),
            rotation: glam::DQuat::from(t.rotation).as_quat(),
            translation: glam::DVec3::from(t.translation).as_vec3(),
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
    fn to_egui(&self) -> transform_gizmo_egui::math::Transform {
        transform_gizmo_egui::math::Transform {
            scale: self.scale.as_dvec3().into(),
            rotation: self.rotation.as_dquat().into(),
            translation: self.translation.as_dvec3().into(),
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
    have_objects_changed: bool,
    gizmo: transform_gizmo_egui::Gizmo,
    scene_revision: usize,
    camera: ControlledCamera,
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
            display_sync: gpu::DisplaySync::Block,
            ..Default::default()
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
                    timing: false,
                    capture: false,
                    overlay: false,
                },
            )
            .unwrap()
        });

        let surface_config = Self::make_surface_config(window.inner_size());
        let surface_size = surface_config.size;
        let surface_info = context.resize(surface_config);

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
            surface_size,
            surface_info,
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
        let gui_painter = blade_egui::GuiPainter::new(surface_info, &context);

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
            have_objects_changed: false,
            gizmo: Default::default(),
            scene_revision: 0,
            camera: ControlledCamera::default(),
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
                temporal_tap: true,
                temporal_history: 10,
                spatial_taps: 1,
                spatial_tap_history: 5,
                spatial_radius: 10,
                t_start: 0.1,
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

        self.camera.inner = blade_render::Camera {
            pos: config_scene.camera.position,
            rot: glam::Quat::from(config_scene.camera.orientation)
                .normalize()
                .into(),
            fov_y: config_scene.camera.fov_y,
            depth: MAX_DEPTH,
        };
        self.camera.fly_speed = config_scene.camera.speed;
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
                position: self.camera.inner.pos,
                orientation: self.camera.inner.rot,
                fov_y: self.camera.inner.fov_y,
                speed: self.camera.fly_speed,
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
        if new_render_size != self.renderer.get_surface_size() {
            log::info!("Resizing to {}", new_render_size);
            self.pacer.wait_for_previous_frame(&self.context);
            self.context.resize(surface_config);
        }

        let (command_encoder, temp) = self.pacer.begin_frame();
        if new_render_size != self.renderer.get_surface_size() {
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
                &self.camera.inner,
                blade_render::FrameConfig {
                    frozen: false,
                    debug_draw: self.is_point_selected || self.is_file_hovered,
                    reset_variance: self.debug.mouse_pos.is_none(),
                    reset_reservoirs: self.need_accumulation_reset,
                },
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
        use transform_gizmo_egui::GizmoExt as _;

        let view_matrix = self.camera.get_view_matrix();
        let extent = self.renderer.get_surface_size();
        let projection_matrix = self
            .camera
            .get_projection_matrix(extent.width as f32 / extent.height as f32);

        self.gizmo.update_config(transform_gizmo_egui::GizmoConfig {
            view_matrix: view_matrix.as_dmat4().into(),
            projection_matrix: projection_matrix.as_dmat4().into(),
            viewport: transform_gizmo_egui::Rect {
                min: transform_gizmo_egui::math::Pos2::ZERO,
                max: transform_gizmo_egui::math::Pos2 {
                    x: extent.width as f32,
                    y: extent.height as f32,
                },
            },
            orientation: transform_gizmo_egui::GizmoOrientation::Global,
            snapping: true,
            ..Default::default()
        });

        let object = &mut self.objects[obj_index];
        let tc = TransformComponents::from(object.transform);

        if let Some((_result, transforms)) = self.gizmo.interact(ui, &[tc.to_egui()]) {
            object.transform = TransformComponents::from(transforms[0]).to_blade();
            self.have_objects_changed = true;
        }
    }

    #[profiling::function]
    fn populate_view(&mut self, ui: &mut egui::Ui) {
        use blade_helpers::{populate_debug_selection, ExposeHud as _};
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
            self.camera.populate_hud(ui);
        });

        egui::CollapsingHeader::new("Debug")
            .default_open(true)
            .show(ui, |ui| {
                self.debug.populate_hud(ui);
                populate_debug_selection(
                    &mut self.debug.mouse_pos,
                    &selection,
                    &self.asset_hub,
                    ui,
                );

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
            .default_open(false)
            .show(ui, |ui| {
                self.ray_config.populate_hud(ui);
            });
        self.need_accumulation_reset |= self.ray_config != old_ray_config;

        egui::CollapsingHeader::new("Denoise")
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut self.denoiser_enabled, "Enable");
                self.denoiser_config.populate_hud(ui);
            });

        egui::CollapsingHeader::new("Tone Map").show(ui, |ui| {
            self.post_proc_config.populate_hud(ui);
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
                    use std::f32::consts::PI;
                    let object = self.objects.get_mut(index).unwrap();
                    let mut tc = TransformComponents::from(object.transform);
                    let (mut a1, mut a2, mut a3) = tc.rotation.to_euler(glam::EulerRot::default());
                    ui.horizontal(|ui| {
                        ui.label("Translate");
                        ui.add(egui::DragValue::new(&mut tc.translation.x));
                        ui.add(egui::DragValue::new(&mut tc.translation.y));
                        ui.add(egui::DragValue::new(&mut tc.translation.z));
                    });
                    ui.add(egui::Slider::new(&mut a1, -PI..=PI).text("Euler Y"));
                    ui.add(egui::Slider::new(&mut a2, -PI * 0.5..=PI * 0.5).text("Euler X"));
                    ui.add(egui::Slider::new(&mut a3, -PI..=PI).text("Euler Z"));
                    ui.horizontal(|ui| {
                        ui.label("Scale");
                        ui.add(egui::DragValue::new(&mut tc.scale.x));
                        ui.add(egui::DragValue::new(&mut tc.scale.y));
                        ui.add(egui::DragValue::new(&mut tc.scale.z));
                    });

                    tc.rotation = glam::Quat::from_euler(glam::EulerRot::default(), a1, a2, a3);
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
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_title("blade-scene")
        .build(&event_loop)
        .unwrap();

    let egui_ctx = egui::Context::default();
    let viewport_id = egui_ctx.viewport_id();
    let mut egui_winit = egui_winit::State::new(egui_ctx, viewport_id, &window, None, None);

    let mut args = std::env::args();
    let path_to_scene = args
        .nth(1)
        .unwrap_or("examples/scene/data/scene.ron".to_string());

    let mut example = Example::new(&window);
    example.load_scene(Path::new(&path_to_scene));

    struct Drag {
        _screen_pos: glam::IVec2,
        _rotation: glam::Quat,
    }
    let mut drag_start = None::<Drag>;
    let mut last_event = time::Instant::now();
    let mut last_mouse_pos = [0i32; 2];

    event_loop
        .run(|event, target| {
            example.choir.check_panic();
            target.set_control_flow(winit::event_loop::ControlFlow::Poll);

            let delta = last_event.elapsed().as_secs_f32();
            let drag_speed = 0.01f32;
            last_event = time::Instant::now();

            match event {
                winit::event::Event::AboutToWait => {
                    window.request_redraw();
                }
                winit::event::Event::WindowEvent { event, .. } => {
                    let response = egui_winit.on_window_event(&window, &event);
                    if response.repaint {
                        window.request_redraw();
                    }
                    if response.consumed {
                        return;
                    }

                    match event {
                        winit::event::WindowEvent::KeyboardInput {
                            event:
                                winit::event::KeyEvent {
                                    physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                                    state: winit::event::ElementState::Pressed,
                                    ..
                                },
                            ..
                        } => {
                            if key_code == winit::keyboard::KeyCode::Escape {
                                target.exit();
                            } else if drag_start.is_none() && example.camera.on_key(key_code, delta)
                            {
                                example.debug.mouse_pos = None;
                            }
                        }
                        winit::event::WindowEvent::CloseRequested => {
                            target.exit();
                        }
                        winit::event::WindowEvent::MouseInput {
                            state,
                            button: winit::event::MouseButton::Left,
                            ..
                        } => {
                            drag_start = match state {
                                winit::event::ElementState::Pressed => Some(Drag {
                                    _screen_pos: last_mouse_pos.into(),
                                    _rotation: example.camera.inner.rot.into(),
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
                                let prev = glam::Quat::from(example.camera.inner.rot);
                                let rotation_local = glam::Quat::from_rotation_x(
                                    (last_mouse_pos[1] as f32 - position.y as f32) * drag_speed,
                                );
                                let rotation_global = glam::Quat::from_rotation_y(
                                    (last_mouse_pos[0] as f32 - position.x as f32) * drag_speed,
                                );
                                example.camera.inner.rot =
                                    (rotation_global * prev * rotation_local).into();
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
                        winit::event::WindowEvent::RedrawRequested => {
                            let raw_input = egui_winit.take_egui_input(&window);
                            let egui_output = egui_winit.egui_ctx().run(raw_input, |egui_ctx| {
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
                                egui::SidePanel::left("content").frame(frame).show(
                                    egui_ctx,
                                    |ui| {
                                        example.populate_content(ui);
                                        ui.separator();
                                        if ui.button("Quit").clicked() {
                                            target.exit();
                                        }
                                    },
                                );
                            });

                            //HACK: https://github.com/urholaukkarinen/egui-gizmo/issues/29
                            if example.have_objects_changed
                                && egui_winit.egui_ctx().wants_pointer_input()
                            {
                                drag_start = None;
                            }

                            egui_winit.handle_platform_output(&window, egui_output.platform_output);
                            let repaint_delay =
                                egui_output.viewport_output[&viewport_id].repaint_delay;
                            let primitives = egui_winit
                                .egui_ctx()
                                .tessellate(egui_output.shapes, egui_output.pixels_per_point);

                            let control_flow = if let Some(repaint_after_instant) =
                                std::time::Instant::now().checked_add(repaint_delay)
                            {
                                winit::event_loop::ControlFlow::WaitUntil(repaint_after_instant)
                            } else {
                                winit::event_loop::ControlFlow::Wait
                            };
                            target.set_control_flow(control_flow);

                            example.render(
                                &primitives,
                                &egui_output.textures_delta,
                                window.inner_size(),
                                window.scale_factor() as f32,
                            );
                            profiling::finish_frame!();
                        }
                        _ => {}
                    }

                    if example.is_point_selected || example.is_file_hovered {
                        //TODO: unfortunately winit doesn't update cursor position during a drag
                        // https://github.com/rust-windowing/winit/issues/1550
                        example.debug.mouse_pos = Some(last_mouse_pos);
                    }
                }
                _ => {}
            }
        })
        .unwrap();

    example.destroy();
}
