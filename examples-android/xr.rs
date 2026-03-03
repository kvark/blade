#![cfg(target_os = "android")]

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use log::info;
use openxr as xr;

const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;
const MAX_XR_EYES: usize = 2;
const NEAR_Z: f32 = 0.05;
const FAR_Z: f32 = 100.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    tint: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct Params {
    globals: gpu::BufferPiece,
}

fn model_matrix() -> Mat4 {
    Mat4::from_translation(Vec3::new(0.0, 0.0, -2.0))
}

fn view_matrix_from_pose(pose: gpu::XrPose) -> Mat4 {
    let orientation = Quat::from_xyzw(
        pose.orientation[0],
        pose.orientation[1],
        pose.orientation[2],
        pose.orientation[3],
    );
    let position = Vec3::from_array(pose.position);
    Mat4::from_rotation_translation(orientation, position).inverse()
}

fn projection_matrix_from_fov(fov: gpu::XrFov, near_z: f32, far_z: f32) -> Mat4 {
    let tan_left = fov.angle_left.tan();
    let tan_right = fov.angle_right.tan();
    let tan_down = fov.angle_down.tan();
    let tan_up = fov.angle_up.tan();

    let tan_width = tan_right - tan_left;
    let tan_height = tan_up - tan_down;
    let z_range = far_z - near_z;

    Mat4::from_cols_array_2d(&[
        [2.0 / tan_width, 0.0, 0.0, 0.0],
        [0.0, 2.0 / tan_height, 0.0, 0.0],
        [
            (tan_right + tan_left) / tan_width,
            (tan_up + tan_down) / tan_height,
            -far_z / z_range,
            -1.0,
        ],
        [0.0, 0.0, -(far_z * near_z) / z_range, 0.0],
    ])
}

struct Example {
    xr_surface: gpu::XrSurface,
    pipeline: gpu::RenderPipeline,
    command_encoder: gpu::CommandEncoder,
    params_buf: gpu::Buffer,
    depth_texture: gpu::Texture,
    depth_views: [gpu::TextureView; MAX_XR_EYES],
    start_time: Instant,
    rendered_frames: u64,
    xr_debug: bool,
}

impl Example {
    fn new(context: &gpu::Context, xr_debug: bool) -> Self {
        let xr_surface = context
            .create_xr_surface()
            .expect("Unable to create XR surface");
        let color_format = xr_surface.format();

        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("xr.wgsl"),
        });
        let data_layout = <Params as gpu::ShaderData>::layout();
        let pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "xr",
            data_layouts: &[&data_layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState {
                ..Default::default()
            },
            depth_stencil: Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: gpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState::from(color_format)],
            multisample_state: gpu::MultisampleState::default(),
        });
        let command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "xr",
            buffer_count: 1,
        });
        let params_buf = context.create_buffer(gpu::BufferDesc {
            name: "xr-params",
            size: (std::mem::size_of::<Uniforms>() * MAX_XR_EYES) as u64,
            memory: gpu::Memory::Shared,
        });
        let extent = xr_surface.extent();
        let view_count = xr_surface.view_count() as usize;
        let depth_texture = context.create_texture(gpu::TextureDesc {
            name: "xr-depth",
            format: gpu::TextureFormat::Depth32Float,
            size: extent,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: view_count as u32,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET,
            sample_count: 1,
            external: None,
        });
        let mut depth_views = [gpu::TextureView::default(); MAX_XR_EYES];
        for eye in 0..view_count.min(MAX_XR_EYES) {
            depth_views[eye] = context.create_texture_view(
                depth_texture,
                gpu::TextureViewDesc {
                    name: "xr-depth-eye",
                    format: gpu::TextureFormat::Depth32Float,
                    dimension: gpu::ViewDimension::D2,
                    subresources: &gpu::TextureSubresources {
                        base_mip_level: 0,
                        mip_level_count: std::num::NonZeroU32::new(1),
                        base_array_layer: eye as u32,
                        array_layer_count: std::num::NonZeroU32::new(1),
                    },
                },
            );
        }

        Self {
            xr_surface,
            pipeline,
            command_encoder,
            params_buf,
            depth_texture,
            depth_views,
            start_time: Instant::now(),
            rendered_frames: 0,
            xr_debug,
        }
    }

    fn render(&mut self, context: &gpu::Context) {
        let frame = if let Some(frame) = self.xr_surface.acquire_frame(context) {
            frame
        } else {
            if self.xr_debug {
                info!("XR frame: should_render=false (ended without layers)");
            }
            return;
        };

        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        let angle = self.start_time.elapsed().as_secs_f32();
        let model = model_matrix() * Mat4::from_rotation_y(angle);
        let view_count = frame.xr_view_count().min(MAX_XR_EYES as u32);
        for eye in 0..view_count {
            let xr_view = frame.xr_view(eye);
            let view = view_matrix_from_pose(xr_view.pose);
            let proj = projection_matrix_from_fov(xr_view.fov, NEAR_Z, FAR_Z);
            let mvp = proj * view * model;
            let uniforms = Uniforms {
                mvp: mvp.to_cols_array_2d(),
                tint: [1.0, 1.0, 1.0, 1.0],
            };
            let params_offset = (eye as usize * std::mem::size_of::<Uniforms>()) as isize;
            unsafe {
                *(self.params_buf.data().offset(params_offset) as *mut Uniforms) = uniforms;
            }
        }
        context.sync_buffer(self.params_buf);

        for eye in 0..view_count {
            let eye_view = frame.xr_texture_view(eye);
            let mut pass = self.command_encoder.render(
                "xr-eye",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: eye_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: Some(gpu::RenderTarget {
                        view: self.depth_views[eye as usize],
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Discard,
                    }),
                },
            );
            let mut rc = pass.with(&mut self.pipeline);
            let params_offset = (eye as usize * std::mem::size_of::<Uniforms>()) as u64;
            rc.bind(
                0,
                &Params {
                    globals: self.params_buf.at(params_offset),
                },
            );
            rc.draw(0, 12, 0, 1);
        }

        self.command_encoder.present(frame);
        let _sync_point = context.submit(&mut self.command_encoder);
        self.rendered_frames += 1;
        if self.xr_debug && (self.rendered_frames <= 5 || self.rendered_frames % 120 == 0) {
            info!("XR frame submitted: {}", self.rendered_frames);
        }
    }

    fn destroy(mut self, context: &gpu::Context) {
        for view in &mut self.depth_views {
            if *view != gpu::TextureView::default() {
                context.destroy_texture_view(*view);
                *view = gpu::TextureView::default();
            }
        }
        context.destroy_texture(self.depth_texture);
        context.destroy_buffer(self.params_buf);
        context.destroy_xr_surface(&mut self.xr_surface);
        context.destroy_command_encoder(&mut self.command_encoder);
        context.destroy_render_pipeline(&mut self.pipeline);
    }
}

#[derive(Default)]
enum AppState {
    #[default]
    Idle,
    Running(Example),
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
            .with_tag("blade-xr"),
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
                application_name: "Blade XR",
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
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            xr: Some(gpu::XrDesc {
                instance: xr_instance.clone(),
                system_id: system,
            }),
            ..Default::default()
        })
        .unwrap()
    };
    mark!("XR mark: gpu::Context initialized");

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
                    AppState::Running(example) => (true, example.rendered_frames),
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
                            mark!("XR mark: calling session.begin");
                            context.xr_session().unwrap().begin(VIEW_TYPE).unwrap();
                            mark!("XR mark: session.begin returned");

                            if matches!(state, AppState::Idle) {
                                mark!("XR mark: calling Example::new");
                                state = AppState::Running(Example::new(&context, xr_debug));
                                mark!("XR mark: Example created");
                            }
                        }
                        xr::SessionState::STOPPING => {
                            if xr_debug {
                                info!("XR state STOPPING -> end session");
                            }
                            context.xr_session().unwrap().end().unwrap();
                            if let AppState::Running(example) =
                                std::mem::replace(&mut state, AppState::Idle)
                            {
                                example.destroy(&context);
                            }
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
            AppState::Running(example) => example.render(&context),
        }
    }

    if let AppState::Running(example) = state {
        example.destroy(&context);
    }
}
