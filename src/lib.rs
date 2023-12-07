#![cfg(not(target_arch = "wasm32"))]
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
use std::{path::Path, sync::Arc};

pub mod config;

//TODO: hide Rapier3D as a private dependency
pub use rapier3d::dynamics::ImpulseJointHandle as JointHandle;
pub use rapier3d::dynamics::RigidBodyType as BodyType;

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
    broad_phase: rapier3d::geometry::BroadPhase,
    narrow_phase: rapier3d::geometry::NarrowPhase,
    gravity: rapier3d::math::Vector<f32>,
    pipeline: rapier3d::pipeline::PhysicsPipeline,
    debug_pipeline: rapier3d::pipeline::DebugRenderPipeline,
}

impl Physics {
    fn step(&mut self, dt: f32) {
        self.integration_params.dt = dt;
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
            None, // query pipeline
            &physics_hooks,
            &event_handler,
        );
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

struct Visual {
    model: blade_asset::Handle<blade_render::Model>,
    similarity: nalgebra::geometry::Similarity3<f32>,
}

struct Object {
    name: String,
    rigid_body: rapier3d::dynamics::RigidBodyHandle,
    prev_isometry: nalgebra::Isometry3<f32>,
    _colliders: Vec<rapier3d::geometry::ColliderHandle>,
    visuals: Vec<Visual>,
}

pub struct Camera {
    pub isometry: nalgebra::Isometry3<f32>,
    pub fov_y: f32,
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
    selected_object_index: Option<ObjectHandle>,
    render_objects: Vec<blade_render::Object>,
    debug: blade_render::DebugConfig,
    need_accumulation_reset: bool,
    is_debug_drawing: bool,
    ray_config: blade_render::RayConfig,
    denoiser_enabled: bool,
    denoiser_config: blade_render::DenoiserConfig,
    post_proc_config: blade_render::PostProcConfig,
    track_hot_reloads: bool,
    workers: Vec<choir::WorkerHandle>,
    choir: Arc<choir::Choir>,
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
            frame_count: 3,
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
                },
            )
            .unwrap()
        });

        let surface_config = Self::make_surface_config(window.inner_size());
        let screen_size = surface_config.size;
        let surface_format = gpu_context.resize(surface_config);

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
            screen_size,
            surface_format,
            max_debug_lines: 1000,
        };
        let renderer = blade_render::Renderer::new(
            command_encoder,
            &gpu_context,
            shaders,
            &asset_hub.shaders,
            &render_config,
        );

        pacer.end_frame(&gpu_context);

        let gui_painter = blade_egui::GuiPainter::new(surface_format, &gpu_context);
        let mut physics = Physics::default();
        physics.debug_pipeline.mode = rapier3d::pipeline::DebugRenderMode::empty();

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
            selected_object_index: None,
            render_objects: Vec::new(),
            debug: blade_render::DebugConfig::default(),
            need_accumulation_reset: true,
            is_debug_drawing: false,
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
                num_passes: 4,
                temporal_weight: 0.1,
            },
            post_proc_config: blade_render::PostProcConfig {
                average_luminocity: 1.0,
                exposure_key_value: 1.0 / 9.6,
                white_level: 1.0,
            },
            track_hot_reloads: false,
            workers,
            choir,
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
        self.physics.step(dt);
    }

    #[profiling::function]
    pub fn render(
        &mut self,
        camera: &Camera,
        gui_primitives: &[egui::ClippedPrimitive],
        gui_textures: &egui::TexturesDelta,
        physical_size: winit::dpi::PhysicalSize<u32>,
        scale_factor: f32,
    ) {
        if self.track_hot_reloads {
            self.need_accumulation_reset |= self.renderer.hot_reload(
                &self.asset_hub,
                &self.gpu_context,
                self.pacer.last_sync_point().unwrap(),
            );
        }

        // Note: the resize is split in 2 parts because `wait_for_previous_frame`
        // wants to borrow `self` mutably, and `command_encoder` blocks that.
        let surface_config = Self::make_surface_config(physical_size);
        let new_render_size = surface_config.size;
        if new_render_size != self.renderer.get_screen_size() {
            log::info!("Resizing to {}", new_render_size);
            self.pacer.wait_for_previous_frame(&self.gpu_context);
            self.gpu_context.resize(surface_config);
        }

        let (command_encoder, temp) = self.pacer.begin_frame();
        if new_render_size != self.renderer.get_screen_size() {
            self.renderer
                .resize_screen(new_render_size, command_encoder, &self.gpu_context);
            self.need_accumulation_reset = true;
        }

        self.gui_painter
            .update_textures(command_encoder, gui_textures, &self.gpu_context);

        self.asset_hub.flush(command_encoder, &mut temp.buffers);

        self.load_tasks.retain(|task| !task.is_done());

        // We should be able to update TLAS and render content
        // even while it's still being loaded.
        if self.load_tasks.is_empty() {
            self.render_objects.clear();
            for (_, object) in self.objects.iter_mut() {
                let isometry = self
                    .physics
                    .rigid_bodies
                    .get(object.rigid_body)
                    .unwrap()
                    .position();
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
                object.prev_isometry = *isometry;
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
                    pos: camera.isometry.translation.vector.into(),
                    rot: camera.isometry.rotation.into(),
                    fov_y: camera.fov_y,
                    depth: MAX_DEPTH,
                },
                self.is_debug_drawing,
                self.debug.mouse_pos.is_some(),
                self.need_accumulation_reset,
            );
            self.need_accumulation_reset = false;

            if !self.render_objects.is_empty() {
                self.renderer
                    .ray_trace(command_encoder, self.debug, self.ray_config);
                if self.denoiser_enabled {
                    self.renderer.denoise(command_encoder, self.denoiser_config);
                }
            }
        }

        let debug_lines = self.physics.render_debug();

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
        use strum::IntoEnumIterator as _;
        egui::CollapsingHeader::new("Rendering")
            .default_open(true)
            .show(ui, |ui| {
                ui.checkbox(&mut self.denoiser_enabled, "Enable Denoiser");
                egui::ComboBox::from_label("Debug mode")
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
            });
        egui::CollapsingHeader::new("Visualize")
            .default_open(true)
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
                        &mut self.selected_object_index,
                        Some(ObjectHandle(handle)),
                        &object.name,
                    );
                }
            });
        if let Some(handle) = self.selected_object_index {
            let object = &self.objects[handle.0];
            let rigid_body = &self.physics.rigid_bodies[object.rigid_body];
            ui.horizontal(|ui| {
                ui.label(format!("Position:"));
                ui.value_vec3(&rigid_body.translation());
            });
            ui.horizontal(|ui| {
                ui.label(format!("Linear velocity:"));
                ui.value_vec3(&rigid_body.linvel());
                ui.label(format!("Damping:"));
                ui.value(rigid_body.linear_damping());
            });
            ui.horizontal(|ui| {
                ui.label(format!("Angular velocity:"));
                ui.value_vec3(&rigid_body.angvel());
                ui.label(format!("Damping:"));
                ui.value(rigid_body.angular_damping());
            });
            ui.horizontal(|ui| {
                ui.label(format!("Kinematic energy:"));
                ui.value(rigid_body.kinetic_energy());
            });
        }
    }

    pub fn screen_aspect(&self) -> f32 {
        let size = self.renderer.get_screen_size();
        size.width as f32 / size.height.max(1) as f32
    }

    pub fn add_object(
        &mut self,
        config: &config::Object,
        isometry: nalgebra::Isometry3<f32>,
        body_type: BodyType,
    ) -> ObjectHandle {
        let mut visuals = Vec::new();
        for visual in config.visuals.iter() {
            let (model, task) = self.asset_hub.models.load(
                format!("data/{}", visual.model),
                blade_render::model::Meta {
                    generate_tangents: true,
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

        let rigid_body = rapier3d::dynamics::RigidBodyBuilder::new(body_type)
            .position(isometry)
            .build();
        let rb_handle = self.physics.rigid_bodies.insert(rigid_body);

        let mut colliders = Vec::new();
        for cc in config.colliders.iter() {
            let isometry = nalgebra::geometry::Isometry3::from_parts(
                nalgebra::Vector3::from(cc.pos).into(),
                make_quaternion(cc.rot),
            );
            let builder = match cc.shape {
                config::Shape::Ball { radius } => rapier3d::geometry::ColliderBuilder::ball(radius),
                config::Shape::Cylinder {
                    half_height,
                    radius,
                } => rapier3d::geometry::ColliderBuilder::cylinder(half_height, radius),
                config::Shape::Cuboid { half } => {
                    rapier3d::geometry::ColliderBuilder::cuboid(half.x, half.y, half.z)
                }
                config::Shape::ConvexHull { ref points } => {
                    let pv = points
                        .iter()
                        .map(|p| nalgebra::Vector3::from(*p).into())
                        .collect::<Vec<_>>();
                    rapier3d::geometry::ColliderBuilder::convex_hull(&pv)
                        .expect("Unable to build convex full")
                }
            };
            let collider = builder.mass(cc.mass).position(isometry).build();
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
            _colliders: colliders,
            visuals,
        });
        ObjectHandle(raw_handle)
    }

    pub fn add_joint(
        &mut self,
        a: ObjectHandle,
        b: ObjectHandle,
        data: impl Into<rapier3d::dynamics::GenericJoint>,
    ) -> JointHandle {
        self.physics.impulse_joints.insert(
            self.objects[a.0].rigid_body,
            self.objects[b.0].rigid_body,
            data,
            true,
        )
    }

    pub fn get_joint_mut(&mut self, handle: JointHandle) -> &mut rapier3d::dynamics::ImpulseJoint {
        self.physics.impulse_joints.get_mut(handle).unwrap()
    }

    pub fn get_joint_impulse(&self, handle: JointHandle) -> &rapier3d::math::SpacialVector<f32> {
        &self.physics.impulse_joints.get(handle).unwrap().impulses
    }

    pub fn get_object_isometry(&self, handle: ObjectHandle) -> &nalgebra::Isometry3<f32> {
        let object = &self.objects[handle.0];
        let body = &self.physics.rigid_bodies[object.rigid_body];
        body.position()
    }

    pub fn apply_impulse(&mut self, handle: ObjectHandle, impulse: nalgebra::Vector3<f32>) {
        let object = &self.objects[handle.0];
        let body = &mut self.physics.rigid_bodies[object.rigid_body];
        body.apply_impulse(impulse, false)
    }

    pub fn set_environment_map(&mut self, path: &str) {
        if path.is_empty() {
            self.environment_map = None;
        } else {
            let full = format!("data/{}", path);
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
}
