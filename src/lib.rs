#![cfg(not(any(gles, target_arch = "wasm32")))]
#![allow(
    irrefutable_let_patterns,
    clippy::new_without_default,
    // Conflicts with `pattern_type_mismatch`
    clippy::needless_borrowed_reference,
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

use blade_graphics as gpu;
use std::{ops, path::Path, sync::Arc};

pub mod config;
mod trimesh;

const ZERO_V3: mint::Vector3<f32> = mint::Vector3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Transform {
    pub position: mint::Vector3<f32>,
    pub orientation: mint::Quaternion<f32>,
}
impl Default for Transform {
    fn default() -> Self {
        Self {
            position: ZERO_V3,
            orientation: mint::Quaternion { s: 1.0, v: ZERO_V3 },
        }
    }
}
impl Transform {
    fn from_isometry(isometry: nalgebra::Isometry3<f32>) -> Self {
        Self {
            position: isometry.translation.vector.into(),
            orientation: isometry.rotation.into(),
        }
    }
    fn into_isometry(self) -> nalgebra::Isometry3<f32> {
        nalgebra::Isometry3 {
            translation: nalgebra::Translation {
                vector: self.position.into(),
            },
            rotation: nalgebra::Unit::new_unchecked(self.orientation.into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoundingBox {
    pub center: mint::Vector3<f32>,
    pub half: mint::Vector3<f32>,
}

/// Type of prediction to be made about the object transformation.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum Prediction {
    /// Report the last known transform. It always makes sense,
    /// but it could be out of date.
    #[default]
    LastKnown,
    /// Integrate using velocity only.
    IntegrateVelocity,
    /// Integrate using velocity and forces affecting the object.
    IntegrateVelocityAndForces,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum DynamicInput {
    /// Object is not controllable, it's static.
    Empty,
    /// Object is controlled by user setting the position.
    SetPosition,
    /// Object is controlled by user setting the velocity.
    SetVelocity,
    /// Object is affected by all forces around.
    #[default]
    Full,
}
impl DynamicInput {
    fn into_rapier(self) -> rapier3d::dynamics::RigidBodyType {
        use rapier3d::dynamics::RigidBodyType as Rbt;
        match self {
            Self::Empty => Rbt::Fixed,
            Self::SetPosition => Rbt::KinematicPositionBased,
            Self::SetVelocity => Rbt::KinematicVelocityBased,
            Self::Full => Rbt::Dynamic,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum JointAxis {
    LinearX = 0,
    LinearY = 1,
    LinearZ = 2,
    AngularX = 3,
    AngularY = 4,
    AngularZ = 5,
}
impl JointAxis {
    fn into_rapier(self) -> rapier3d::dynamics::JointAxis {
        use rapier3d::dynamics::JointAxis as Ja;
        match self {
            Self::LinearX => Ja::LinX,
            Self::LinearY => Ja::LinY,
            Self::LinearZ => Ja::LinZ,
            Self::AngularX => Ja::AngX,
            Self::AngularY => Ja::AngY,
            Self::AngularZ => Ja::AngZ,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum JointHandle {
    Soft(#[doc(hidden)] rapier3d::dynamics::ImpulseJointHandle),
    Hard(#[doc(hidden)] rapier3d::dynamics::MultibodyJointHandle),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FreedomAxis {
    pub limits: Option<ops::Range<f32>>,
    pub motor: Option<config::Motor>,
}

impl FreedomAxis {
    pub const FREE: Self = Self {
        limits: None,
        motor: None,
    };
    pub const ALL_FREE: mint::Vector3<Option<Self>> = mint::Vector3 {
        x: Some(Self::FREE),
        y: Some(Self::FREE),
        z: Some(Self::FREE),
    };
}

#[derive(Clone, Debug, PartialEq)]
pub struct JointDesc {
    pub parent_anchor: Transform,
    pub child_anchor: Transform,
    pub linear: mint::Vector3<Option<FreedomAxis>>,
    pub angular: mint::Vector3<Option<FreedomAxis>>,
    /// Allow the contacts to happen between A and B
    pub allow_contacts: bool,
    /// Hard joints guarantee the releation between
    /// objects, while soft joints only try to get there.
    pub is_hard: bool,
}
impl Default for JointDesc {
    fn default() -> Self {
        Self {
            parent_anchor: Transform::default(),
            child_anchor: Transform::default(),
            linear: mint::Vector3 {
                x: None,
                y: None,
                z: None,
            },
            angular: mint::Vector3 {
                x: None,
                y: None,
                z: None,
            },
            allow_contacts: false,
            is_hard: false,
        }
    }
}

const MAX_DEPTH: f32 = 1e9;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ObjectHandle(usize);

fn make_quaternion(degrees: mint::Vector3<f32>) -> nalgebra::geometry::UnitQuaternion<f32> {
    nalgebra::geometry::UnitQuaternion::from_euler_angles(
        degrees.x.to_radians(),
        degrees.y.to_radians(),
        degrees.z.to_radians(),
    )
}

trait UiValue {
    fn value(&mut self, v: f32);
    fn value_vec3(&mut self, v3: &nalgebra::Vector3<f32>) {
        for &v in v3.as_slice() {
            self.value(v);
        }
    }
}
impl UiValue for egui::Ui {
    fn value(&mut self, v: f32) {
        self.colored_label(egui::Color32::WHITE, format!("{v:.1}"));
    }
}

#[derive(Default)]
struct DebugPhysicsRender {
    lines: Vec<blade_render::DebugLine>,
}
impl rapier3d::pipeline::DebugRenderBackend for DebugPhysicsRender {
    fn draw_line(
        &mut self,
        _object: rapier3d::pipeline::DebugRenderObject,
        a: nalgebra::Point3<f32>,
        b: nalgebra::Point3<f32>,
        color: [f32; 4],
    ) {
        // Looks like everybody encodes HSL(A) differently...
        let hsl = colorsys::Hsl::new(
            color[0] as f64,
            color[1] as f64 * 100.0,
            color[2] as f64 * 100.0,
            None,
        );
        let rgb = colorsys::Rgb::from(&hsl);
        let color = [
            rgb.red(),
            rgb.green(),
            rgb.blue(),
            color[3].clamp(0.0, 1.0) as f64 * 255.0,
        ]
        .iter()
        .rev()
        .fold(0u32, |u, &c| (u << 8) | c as u32);
        self.lines.push(blade_render::DebugLine {
            a: blade_render::DebugPoint {
                pos: a.into(),
                color,
            },
            b: blade_render::DebugPoint {
                pos: b.into(),
                color,
            },
        });
    }
}

#[derive(Default)]
struct Physics {
    rigid_bodies: rapier3d::dynamics::RigidBodySet,
    integration_params: rapier3d::dynamics::IntegrationParameters,
    island_manager: rapier3d::dynamics::IslandManager,
    impulse_joints: rapier3d::dynamics::ImpulseJointSet,
    multibody_joints: rapier3d::dynamics::MultibodyJointSet,
    solver: rapier3d::dynamics::CCDSolver,
    colliders: rapier3d::geometry::ColliderSet,
    broad_phase: rapier3d::geometry::DefaultBroadPhase,
    narrow_phase: rapier3d::geometry::NarrowPhase,
    gravity: rapier3d::math::Vector<f32>,
    pipeline: rapier3d::pipeline::PhysicsPipeline,
    debug_pipeline: rapier3d::pipeline::DebugRenderPipeline,
    last_time: f32,
}

impl Physics {
    fn step(&mut self) {
        let query_pipeline = None;
        let physics_hooks = ();
        let event_handler = ();
        self.pipeline.step(
            &self.gravity,
            &self.integration_params,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.solver,
            query_pipeline,
            &physics_hooks,
            &event_handler,
        );
        self.last_time += self.integration_params.dt;
    }
    fn render_debug(&mut self) -> Vec<blade_render::DebugLine> {
        let mut backend = DebugPhysicsRender::default();
        self.debug_pipeline.render(
            &mut backend,
            &self.rigid_bodies,
            &self.colliders,
            &self.impulse_joints,
            &self.multibody_joints,
            &self.narrow_phase,
        );
        backend.lines
    }
}

#[doc(hidden)]
impl ops::Index<JointHandle> for Physics {
    type Output = rapier3d::dynamics::GenericJoint;
    fn index(&self, handle: JointHandle) -> &Self::Output {
        match handle {
            JointHandle::Soft(h) => &self.impulse_joints.get(h).unwrap().data,
            JointHandle::Hard(h) => {
                let (multibody, link_index) = self.multibody_joints.get(h).unwrap();
                &multibody.link(link_index).unwrap().joint.data
            }
        }
    }
}
impl ops::IndexMut<JointHandle> for Physics {
    fn index_mut(&mut self, handle: JointHandle) -> &mut Self::Output {
        match handle {
            JointHandle::Soft(h) => &mut self.impulse_joints.get_mut(h).unwrap().data,
            JointHandle::Hard(h) => {
                let (multibody, link_index) = self.multibody_joints.get_mut(h).unwrap();
                &mut multibody.link_mut(link_index).unwrap().joint.data
            }
        }
    }
}

struct Visual {
    model: blade_asset::Handle<blade_render::Model>,
    similarity: nalgebra::geometry::Similarity3<f32>,
}

struct Object {
    name: String,
    rigid_body: rapier3d::dynamics::RigidBodyHandle,
    prev_isometry: nalgebra::Isometry3<f32>,
    colliders: Vec<rapier3d::geometry::ColliderHandle>,
    visuals: Vec<Visual>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrameCamera {
    pub transform: Transform,
    pub fov_y: f32,
}

impl From<blade_render::Camera> for FrameCamera {
    fn from(cam: blade_render::Camera) -> Self {
        Self {
            transform: Transform {
                position: cam.pos,
                orientation: cam.rot,
            },
            fov_y: cam.fov_y,
        }
    }
}

/// Blade Engine encapsulates all the context for applications,
/// such as the GPU context, Ray-tracing context, EGUI integration,
/// asset hub, physics context, task processing, and more.
pub struct Engine {
    pacer: blade_render::util::FramePacer,
    renderer: blade_render::Renderer,
    physics: Physics,
    load_tasks: Vec<choir::RunningTask>,
    gui_painter: blade_egui::GuiPainter,
    asset_hub: blade_render::AssetHub,
    gpu_context: Arc<gpu::Context>,
    environment_map: Option<blade_asset::Handle<blade_render::Texture>>,
    objects: slab::Slab<Object>,
    selected_object_handle: Option<ObjectHandle>,
    selected_collider: Option<rapier3d::geometry::ColliderHandle>,
    render_objects: Vec<blade_render::Object>,
    debug: blade_render::DebugConfig,
    pub frame_config: blade_render::FrameConfig,
    pub ray_config: blade_render::RayConfig,
    pub denoiser_config: blade_render::DenoiserConfig,
    pub post_proc_config: blade_render::PostProcConfig,
    track_hot_reloads: bool,
    workers: Vec<choir::WorkerHandle>,
    choir: Arc<choir::Choir>,
    data_path: String,
    time_ahead: f32,
}

impl Engine {
    fn make_surface_config(physical_size: winit::dpi::PhysicalSize<u32>) -> gpu::SurfaceConfig {
        gpu::SurfaceConfig {
            size: gpu::Extent {
                width: physical_size.width,
                height: physical_size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            //TODO: make it `Recent`
            display_sync: gpu::DisplaySync::Block,
            color_space: gpu::ColorSpace::Linear,
            transparent: false,
            allow_exclusive_full_screen: true,
        }
    }

    /// Create a new context based on a given window.
    #[profiling::function]
    pub fn new(window: &winit::window::Window, config: &config::Engine) -> Self {
        log::info!("Initializing the engine");

        let gpu_context = Arc::new(unsafe {
            gpu::Context::init_windowed(
                window,
                gpu::ContextDesc {
                    validation: cfg!(debug_assertions),
                    capture: false,
                    overlay: false,
                },
            )
            .unwrap()
        });

        let surface_config = Self::make_surface_config(window.inner_size());
        let surface_size = surface_config.size;
        let surface_info = gpu_context.resize(surface_config);

        let num_workers = num_cpus::get_physical().max((num_cpus::get() * 3 + 2) / 4);
        log::info!("Initializing Choir with {} workers", num_workers);
        let choir = choir::Choir::new();
        let workers = (0..num_workers)
            .map(|i| choir.add_worker(&format!("Worker-{}", i)))
            .collect();

        let asset_hub = blade_render::AssetHub::new(Path::new("asset-cache"), &choir, &gpu_context);
        let (shaders, shader_task) =
            blade_render::Shaders::load(config.shader_path.as_ref(), &asset_hub);

        log::info!("Spinning up the renderer");
        shader_task.join();
        let mut pacer = blade_render::util::FramePacer::new(&gpu_context);
        let (command_encoder, _) = pacer.begin_frame();

        let render_config = blade_render::RenderConfig {
            surface_size,
            surface_info,
            max_debug_lines: 1 << 14,
        };
        let renderer = blade_render::Renderer::new(
            command_encoder,
            &gpu_context,
            shaders,
            &asset_hub.shaders,
            &render_config,
        );

        pacer.end_frame(&gpu_context);

        let gui_painter = blade_egui::GuiPainter::new(surface_info, &gpu_context);
        let mut physics = Physics::default();
        physics.debug_pipeline.mode = rapier3d::pipeline::DebugRenderMode::empty();
        physics.integration_params.dt = config.time_step;

        Self {
            pacer,
            renderer,
            physics,
            load_tasks: Vec::new(),
            gui_painter,
            asset_hub,
            gpu_context,
            environment_map: None,
            objects: slab::Slab::new(),
            selected_object_handle: None,
            selected_collider: None,
            render_objects: Vec::new(),
            debug: blade_render::DebugConfig::default(),
            frame_config: blade_render::FrameConfig {
                frozen: false,
                debug_draw: true,
                reset_variance: false,
                reset_reservoirs: true,
            },
            ray_config: blade_render::RayConfig {
                num_environment_samples: 1,
                environment_importance_sampling: true,
                temporal_tap: true,
                temporal_confidence: 10.0,
                spatial_taps: 1,
                spatial_confidence: 5.0,
                spatial_min_distance: 4,
                group_mixer: 10,
                t_start: 0.01,
                pairwise_mis: true,
            },
            denoiser_config: blade_render::DenoiserConfig {
                enabled: true,
                num_passes: 4,
                temporal_weight: 0.1,
            },
            post_proc_config: blade_render::PostProcConfig {
                average_luminocity: 0.5,
                exposure_key_value: 1.0 / 9.6,
                white_level: 1.0,
            },
            track_hot_reloads: false,
            workers,
            choir,
            data_path: config.data_path.clone(),
            time_ahead: 0.0,
        }
    }

    pub fn destroy(&mut self) {
        self.workers.clear();
        self.pacer.destroy(&self.gpu_context);
        self.gui_painter.destroy(&self.gpu_context);
        self.renderer.destroy(&self.gpu_context);
        self.asset_hub.destroy();
    }

    #[profiling::function]
    pub fn update(&mut self, dt: f32) {
        self.choir.check_panic();
        self.time_ahead += dt;
        while self.time_ahead >= self.physics.integration_params.dt {
            self.physics.step();
            self.time_ahead -= self.physics.integration_params.dt;
        }
    }

    #[profiling::function]
    pub fn render(
        &mut self,
        camera: &FrameCamera,
        gui_primitives: &[egui::ClippedPrimitive],
        gui_textures: &egui::TexturesDelta,
        physical_size: winit::dpi::PhysicalSize<u32>,
        scale_factor: f32,
    ) {
        if self.track_hot_reloads {
            self.renderer.hot_reload(
                &self.asset_hub,
                &self.gpu_context,
                self.pacer.last_sync_point().unwrap(),
            );
        }

        // Note: the resize is split in 2 parts because `wait_for_previous_frame`
        // wants to borrow `self` mutably, and `command_encoder` blocks that.
        let surface_config = Self::make_surface_config(physical_size);
        let new_render_size = surface_config.size;
        if new_render_size != self.renderer.get_surface_size() {
            log::info!("Resizing to {}", new_render_size);
            self.pacer.wait_for_previous_frame(&self.gpu_context);
            self.gpu_context.resize(surface_config);
        }

        let (command_encoder, temp) = self.pacer.begin_frame();
        if new_render_size != self.renderer.get_surface_size() {
            self.renderer
                .resize_screen(new_render_size, command_encoder, &self.gpu_context);
            self.frame_config.reset_reservoirs = true;
        }
        self.frame_config.reset_variance = self.debug.mouse_pos.is_none();

        self.gui_painter
            .update_textures(command_encoder, gui_textures, &self.gpu_context);

        self.asset_hub.flush(command_encoder, &mut temp.buffers);

        self.load_tasks.retain(|task| !task.is_done());

        // We should be able to update TLAS and render content
        // even while it's still being loaded.
        let mut frame_key = blade_render::FrameKey::default();
        if self.load_tasks.is_empty() {
            self.render_objects.clear();
            for (_, object) in self.objects.iter_mut() {
                let isometry = self
                    .physics
                    .rigid_bodies
                    .get(object.rigid_body)
                    .unwrap()
                    .predict_position_using_velocity_and_forces(self.time_ahead);

                for visual in object.visuals.iter() {
                    let mc = (isometry * visual.similarity).to_homogeneous().transpose();
                    let mp = (object.prev_isometry * visual.similarity)
                        .to_homogeneous()
                        .transpose();
                    self.render_objects.push(blade_render::Object {
                        transform: gpu::Transform {
                            x: mc.column(0).into(),
                            y: mc.column(1).into(),
                            z: mc.column(2).into(),
                        },
                        prev_transform: gpu::Transform {
                            x: mp.column(0).into(),
                            y: mp.column(1).into(),
                            z: mp.column(2).into(),
                        },
                        model: visual.model,
                    });
                }
                object.prev_isometry = isometry;
            }

            // Rebuilding every frame
            self.renderer.build_scene(
                command_encoder,
                &self.render_objects,
                self.environment_map,
                &self.asset_hub,
                &self.gpu_context,
                temp,
            );

            self.renderer.prepare(
                command_encoder,
                &blade_render::Camera {
                    pos: camera.transform.position,
                    rot: camera.transform.orientation,
                    fov_y: camera.fov_y,
                    depth: MAX_DEPTH,
                },
                self.frame_config,
            );
            self.frame_config.reset_reservoirs = false;

            if !self.render_objects.is_empty() {
                frame_key = self.renderer.ray_trace(
                    command_encoder,
                    self.debug,
                    self.ray_config,
                    self.denoiser_config,
                );
            }
        }

        let mut debug_lines = self.physics.render_debug();
        if let Some(handle) = self.selected_object_handle {
            let object = &self.objects[handle.0];
            let rb = self.physics.rigid_bodies.get(object.rigid_body).unwrap();
            for (_, joint) in self.physics.impulse_joints.iter() {
                let local_frame = if joint.body1 == object.rigid_body {
                    joint.data.local_frame1
                } else if joint.body2 == object.rigid_body {
                    joint.data.local_frame2
                } else {
                    continue;
                };
                let position = rb.position() * local_frame;
                let length = 1.0;
                let base = blade_render::DebugPoint {
                    pos: position.translation.into(),
                    color: 0xFFFFFF,
                };
                debug_lines.push(blade_render::DebugLine {
                    a: base,
                    b: blade_render::DebugPoint {
                        pos: position
                            .transform_point(&nalgebra::Point3::new(length, 0.0, 0.0))
                            .into(),
                        color: 0x0000FF,
                    },
                });
                debug_lines.push(blade_render::DebugLine {
                    a: base,
                    b: blade_render::DebugPoint {
                        pos: position
                            .transform_point(&nalgebra::Point3::new(0.0, length, 0.0))
                            .into(),
                        color: 0x00FF00,
                    },
                });
                debug_lines.push(blade_render::DebugLine {
                    a: base,
                    b: blade_render::DebugPoint {
                        pos: position
                            .transform_point(&nalgebra::Point3::new(0.0, 0.0, length))
                            .into(),
                        color: 0xFF0000,
                    },
                });
            }
        }

        let frame = self.gpu_context.acquire_frame();
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
            if self.load_tasks.is_empty() {
                self.renderer.post_proc(
                    &mut pass,
                    frame_key,
                    self.debug,
                    self.post_proc_config,
                    &debug_lines,
                    &[],
                );
            }
            self.gui_painter
                .paint(&mut pass, gui_primitives, &screen_desc, &self.gpu_context);
        }

        command_encoder.present(frame);
        let sync_point = self.pacer.end_frame(&self.gpu_context);
        self.gui_painter.after_submit(sync_point);

        profiling::finish_frame!();
    }

    #[profiling::function]
    pub fn populate_hud(&mut self, ui: &mut egui::Ui) {
        use blade_helpers::ExposeHud as _;

        let mut selection = blade_render::SelectionInfo::default();
        if self.debug.mouse_pos.is_some() {
            selection = self.renderer.read_debug_selection_info();
            self.selected_object_handle = self.find_object(selection.custom_index);
        }

        ui.checkbox(&mut self.track_hot_reloads, "Hot reloading");

        egui::CollapsingHeader::new("Rendering")
            .default_open(false)
            .show(ui, |ui| {
                self.ray_config.populate_hud(ui);
                self.frame_config.populate_hud(ui);
                self.denoiser_config.populate_hud(ui);
                self.post_proc_config.populate_hud(ui);
            });
        egui::CollapsingHeader::new("Debug")
            .default_open(true)
            .show(ui, |ui| {
                self.debug.populate_hud(ui);
                blade_helpers::populate_debug_selection(
                    &mut self.debug.mouse_pos,
                    &selection,
                    &self.asset_hub,
                    ui,
                );
            });

        egui::CollapsingHeader::new("Visualize")
            .default_open(false)
            .show(ui, |ui| {
                let all_bits = rapier3d::pipeline::DebugRenderMode::all().bits();
                for bit_pos in 0..=all_bits.ilog2() {
                    let flag = match rapier3d::pipeline::DebugRenderMode::from_bits(1 << bit_pos) {
                        Some(flag) => flag,
                        None => continue,
                    };
                    let mut enabled = self.physics.debug_pipeline.mode.contains(flag);
                    ui.checkbox(&mut enabled, format!("{flag:?}"));
                    self.physics.debug_pipeline.mode.set(flag, enabled);
                }
            });

        egui::CollapsingHeader::new("Objects")
            .default_open(true)
            .show(ui, |ui| {
                for (handle, object) in self.objects.iter() {
                    ui.selectable_value(
                        &mut self.selected_object_handle,
                        Some(ObjectHandle(handle)),
                        &object.name,
                    );
                }

                if let Some(handle) = self.selected_object_handle {
                    let object = &self.objects[handle.0];
                    let rigid_body = &mut self.physics.rigid_bodies[object.rigid_body];
                    if ui.button("Unselect").clicked() {
                        self.selected_object_handle = None;
                        self.selected_collider = None;
                    }
                    egui::CollapsingHeader::new("Stats")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(format!("Position:"));
                                ui.value_vec3(&rigid_body.translation());
                            });
                            ui.horizontal(|ui| {
                                ui.label(format!("Linear velocity:"));
                                ui.value_vec3(&rigid_body.linvel());
                            });
                            ui.horizontal(|ui| {
                                ui.label(format!("Linear Damping:"));
                                ui.value(rigid_body.linear_damping());
                            });
                            ui.horizontal(|ui| {
                                ui.label(format!("Angular velocity:"));
                                ui.value_vec3(&rigid_body.angvel());
                            });
                            ui.horizontal(|ui| {
                                ui.label(format!("Angular Damping:"));
                                ui.value(rigid_body.angular_damping());
                            });
                            ui.horizontal(|ui| {
                                ui.label(format!("Kinematic energy:"));
                                ui.value(rigid_body.kinetic_energy());
                            });
                        });
                    ui.heading("Colliders");
                    for &collider_handle in rigid_body.colliders() {
                        let collider = &self.physics.colliders[collider_handle];
                        let name = format!("{:?}", collider.shape().shape_type());
                        ui.selectable_value(
                            &mut self.selected_collider,
                            Some(collider_handle),
                            name,
                        );
                    }
                }

                if let Some(collider_handle) = self.selected_collider {
                    ui.heading("Properties");
                    let collider = self.physics.colliders.get_mut(collider_handle).unwrap();
                    let mut density = collider.density();
                    if ui
                        .add(
                            egui::DragValue::new(&mut density)
                                .prefix("Density: ")
                                .range(0.1..=1e6),
                        )
                        .changed()
                    {
                        collider.set_density(density);
                    }
                    let mut friction = collider.friction();
                    if ui
                        .add(
                            egui::DragValue::new(&mut friction)
                                .prefix("Friction: ")
                                .range(0.0..=5.0)
                                .speed(0.01),
                        )
                        .changed()
                    {
                        collider.set_friction(friction);
                    }
                    let mut restitution = collider.restitution();
                    if ui
                        .add(
                            egui::DragValue::new(&mut restitution)
                                .prefix("Restituion: ")
                                .range(0.0..=1.0)
                                .speed(0.01),
                        )
                        .changed()
                    {
                        collider.set_restitution(restitution);
                    }
                }
            });
    }

    pub fn screen_aspect(&self) -> f32 {
        let size = self.renderer.get_surface_size();
        size.width as f32 / size.height.max(1) as f32
    }

    fn find_object(&self, geometry_index: u32) -> Option<ObjectHandle> {
        let mut index = geometry_index as usize;
        for (obj_handle, object) in self.objects.iter() {
            for visual in object.visuals.iter() {
                let model = &self.asset_hub.models[visual.model];
                match index.checked_sub(model.geometries.len()) {
                    Some(i) => index = i,
                    None => return Some(ObjectHandle(obj_handle)),
                }
            }
        }
        None
    }

    pub fn add_object(
        &mut self,
        config: &config::Object,
        transform: Transform,
        dynamic_input: DynamicInput,
    ) -> ObjectHandle {
        use rapier3d::{
            dynamics::MassProperties,
            geometry::{ColliderBuilder, TriMeshFlags},
        };

        let mut visuals = Vec::new();
        for visual in config.visuals.iter() {
            let (model, task) = self.asset_hub.models.load(
                format!("{}/{}", self.data_path, visual.model),
                blade_render::model::Meta {
                    generate_tangents: true,
                    front_face: match visual.front_face {
                        config::FrontFace::Cw => blade_render::model::FrontFace::Clockwise,
                        config::FrontFace::Ccw => blade_render::model::FrontFace::CounterClockwise,
                    },
                },
            );
            visuals.push(Visual {
                model,
                similarity: nalgebra::geometry::Similarity3::from_parts(
                    nalgebra::Vector3::from(visual.pos).into(),
                    make_quaternion(visual.rot),
                    visual.scale,
                ),
            });
            self.load_tasks.push(task.clone());
        }

        let add_mass_properties = match config.additional_mass {
            Some(ref am) => match am.shape {
                config::Shape::Ball { radius } => MassProperties::from_ball(am.density, radius),
                config::Shape::Cylinder {
                    half_height,
                    radius,
                } => MassProperties::from_cylinder(am.density, half_height, radius),
                config::Shape::Cuboid { half } => {
                    MassProperties::from_cuboid(am.density, half.into())
                }
                config::Shape::ConvexHull { .. } | config::Shape::TriMesh { .. } => {
                    unimplemented!()
                }
            },
            None => Default::default(),
        };

        let rigid_body = rapier3d::dynamics::RigidBodyBuilder::new(dynamic_input.into_rapier())
            .position(transform.into_isometry())
            .additional_mass_properties(add_mass_properties)
            .build();
        let rb_handle = self.physics.rigid_bodies.insert(rigid_body);

        let mut colliders = Vec::new();
        for cc in config.colliders.iter() {
            let isometry = nalgebra::geometry::Isometry3::from_parts(
                nalgebra::Vector3::from(cc.pos).into(),
                make_quaternion(cc.rot),
            );
            let builder = match cc.shape {
                config::Shape::Ball { radius } => ColliderBuilder::ball(radius),
                config::Shape::Cylinder {
                    half_height,
                    radius,
                } => ColliderBuilder::cylinder(half_height, radius),
                config::Shape::Cuboid { half } => ColliderBuilder::cuboid(half.x, half.y, half.z),
                config::Shape::ConvexHull {
                    ref points,
                    border_radius,
                } => {
                    let pv = points
                        .iter()
                        .map(|p| nalgebra::Vector3::from(*p).into())
                        .collect::<Vec<_>>();
                    let result = if border_radius != 0.0 {
                        ColliderBuilder::round_convex_hull(&pv, border_radius)
                    } else {
                        ColliderBuilder::convex_hull(&pv)
                    };
                    result.expect("Unable to build convex hull shape")
                }
                config::Shape::TriMesh {
                    ref model,
                    convex,
                    border_radius,
                } => {
                    let trimesh = trimesh::load(&format!("{}/{}", self.data_path, model));
                    if convex && border_radius != 0.0 {
                        ColliderBuilder::round_convex_mesh(
                            trimesh.points,
                            &trimesh.triangles,
                            border_radius,
                        )
                        .expect("Unable to build rounded convex mesh")
                    } else if convex {
                        ColliderBuilder::convex_mesh(trimesh.points, &trimesh.triangles)
                            .expect("Unable to build convex mesh")
                    } else {
                        assert_eq!(border_radius, 0.0);
                        let flags = TriMeshFlags::empty();
                        ColliderBuilder::trimesh_with_flags(
                            trimesh.points,
                            trimesh.triangles,
                            flags,
                        )
                    }
                }
            };
            let collider = builder
                .density(cc.density)
                .friction(cc.friction)
                .restitution(cc.restitution)
                .position(isometry)
                .build();
            let c_handle = self.physics.colliders.insert_with_parent(
                collider,
                rb_handle,
                &mut self.physics.rigid_bodies,
            );
            colliders.push(c_handle);
        }

        let raw_handle = self.objects.insert(Object {
            name: config.name.clone(),
            rigid_body: rb_handle,
            prev_isometry: nalgebra::Isometry3::default(),
            colliders,
            visuals,
        });
        ObjectHandle(raw_handle)
    }

    pub fn wake_up(&mut self, object: ObjectHandle) {
        let rb_handle = self.objects[object.0].rigid_body;
        let rb = self.physics.rigid_bodies.get_mut(rb_handle).unwrap();
        rb.wake_up(true);
    }

    pub fn add_joint(
        &mut self,
        parent: ObjectHandle,
        child: ObjectHandle,
        desc: JointDesc,
    ) -> JointHandle {
        let data = {
            let mut locked_axes = rapier3d::dynamics::JointAxesMask::empty();
            let freedoms = [
                (JointAxis::LinearX, &desc.linear.x),
                (JointAxis::LinearY, &desc.linear.y),
                (JointAxis::LinearZ, &desc.linear.z),
                (JointAxis::AngularX, &desc.angular.x),
                (JointAxis::AngularY, &desc.angular.y),
                (JointAxis::AngularZ, &desc.angular.z),
            ];
            let mut joint_builder =
                rapier3d::dynamics::GenericJointBuilder::new(Default::default())
                    .local_frame1(desc.parent_anchor.into_isometry())
                    .local_frame2(desc.child_anchor.into_isometry())
                    .contacts_enabled(desc.allow_contacts);
            for &(axis, ref maybe_freedom) in freedoms.iter() {
                let rapier_axis = axis.into_rapier();
                match *maybe_freedom {
                    Some(freedom) => {
                        if let Some(ref limits) = freedom.limits {
                            joint_builder =
                                joint_builder.limits(rapier_axis, [limits.start, limits.end]);
                        }
                        if let Some(ref motor) = freedom.motor {
                            joint_builder = joint_builder
                                .motor_position(rapier_axis, 0.0, motor.stiffness, motor.damping)
                                .motor_max_force(rapier_axis, motor.max_force);
                        }
                    }
                    None => {
                        locked_axes |= rapier3d::dynamics::JointAxesMask::from(rapier_axis);
                    }
                }
            }
            joint_builder.locked_axes(locked_axes).0
        };

        let body1 = self.objects[parent.0].rigid_body;
        let body2 = self.objects[child.0].rigid_body;
        if desc.is_hard {
            JointHandle::Hard(
                self.physics
                    .multibody_joints
                    .insert(body1, body2, data, true)
                    .unwrap(),
            )
        } else {
            JointHandle::Soft(self.physics.impulse_joints.insert(body1, body2, data, true))
        }
    }

    /// Get the current object transform.
    ///
    /// Since the simulation is done at fixed key frames, the position specifically
    /// at the current time needs to be predicted.
    pub fn get_object_transform(&self, handle: ObjectHandle, prediction: Prediction) -> Transform {
        let object = &self.objects[handle.0];
        let body = &self.physics.rigid_bodies[object.rigid_body];
        let isometry = match prediction {
            Prediction::LastKnown => *body.position(),
            Prediction::IntegrateVelocity => unimplemented!(),
            Prediction::IntegrateVelocityAndForces => {
                body.predict_position_using_velocity_and_forces(self.time_ahead)
            }
        };
        Transform::from_isometry(isometry)
    }

    pub fn get_object_bounds(&self, handle: ObjectHandle) -> BoundingBox {
        let object = &self.objects[handle.0];
        let mut aabb = rapier3d::geometry::Aabb::new_invalid();
        for &collider_handle in object.colliders.iter() {
            let collider = &self.physics.colliders[collider_handle];
            rapier3d::geometry::BoundingVolume::merge(&mut aabb, &collider.compute_aabb());
        }
        BoundingBox {
            //TODO: proper Point3 -> Mint conversion?
            center: (aabb.center() - nalgebra::Point3::default()).into(),
            half: aabb.half_extents().into(),
        }
    }

    pub fn apply_linear_impulse(&mut self, handle: ObjectHandle, impulse: mint::Vector3<f32>) {
        let object = &self.objects[handle.0];
        let body = &mut self.physics.rigid_bodies[object.rigid_body];
        body.apply_impulse(impulse.into(), false)
    }

    pub fn apply_angular_impulse(&mut self, handle: ObjectHandle, impulse: mint::Vector3<f32>) {
        let object = &self.objects[handle.0];
        let body = &mut self.physics.rigid_bodies[object.rigid_body];
        body.apply_torque_impulse(impulse.into(), false)
    }

    pub fn teleport_object(&mut self, handle: ObjectHandle, transform: Transform) {
        let object = &self.objects[handle.0];
        let body = &mut self.physics.rigid_bodies[object.rigid_body];
        body.set_linvel(Default::default(), false);
        body.set_angvel(Default::default(), false);
        body.set_position(transform.into_isometry(), true);
    }

    pub fn set_joint_motor(
        &mut self,
        handle: JointHandle,
        axis: JointAxis,
        target_pos: f32,
        target_vel: f32,
    ) {
        let joint = &mut self.physics[handle];
        let rapier_axis = axis.into_rapier();
        match joint.motor(rapier_axis) {
            Some(&rapier3d::dynamics::JointMotor {
                damping, stiffness, ..
            }) => {
                joint.set_motor(rapier_axis, target_pos, target_vel, stiffness, damping);
            }
            None => panic!("Axis {:?} of {:?} is not motorized", axis, handle),
        }
    }

    pub fn set_environment_map(&mut self, path: &str) {
        if path.is_empty() {
            self.environment_map = None;
        } else {
            let full = format!("{}/{}", self.data_path, path);
            let (handle, task) = self.asset_hub.textures.load(
                full,
                blade_render::texture::Meta {
                    format: gpu::TextureFormat::Rgba32Float,
                    generate_mips: false,
                    y_flip: false,
                },
            );
            self.environment_map = Some(handle);
            self.load_tasks.push(task.clone());
        }
    }

    pub fn set_gravity(&mut self, force: f32) {
        self.physics.gravity.y = -force;
    }

    pub fn set_average_luminosity(&mut self, avg_lum: f32) {
        self.post_proc_config.average_luminocity = avg_lum;
    }

    pub fn set_debug_pixel(&mut self, mouse_pos: Option<[i32; 2]>) {
        self.debug.mouse_pos = mouse_pos;
    }
}
