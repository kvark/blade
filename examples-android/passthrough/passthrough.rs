#![cfg(target_os = "android")]

//! Passthrough on Meta Quest, composed of three layers: the cameras, tinted
//! red with colored edges, at the back; the scene, drawn with alpha, over
//! them; and a HUD on a quad in front of the head, which shows the depth of
//! the environment when the headset provides it.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
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
const HUD_SIZE: gpu::Extent = gpu::Extent {
    width: 512,
    height: 256,
    depth: 1,
};

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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HudParams {
    time: f32,
    has_depth: u32,
    pad: [f32; 2],
}

#[derive(blade_macros::ShaderData)]
struct HudData {
    hud: HudParams,
    depth_tex: gpu::TextureView,
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
    scene_surface: gpu::XrSurface,
    hud_surface: gpu::XrSurface,
    passthrough: Option<gpu::XrPassthrough>,
    depth: Option<gpu::XrEnvironmentDepth>,
    // Bound in place of the depth of the environment, without it.
    no_depth: gpu::Texture,
    no_depth_view: gpu::TextureView,
    // The head, which the HUD stays in front of.
    view_space: xr::Space,
    scene_pipeline: gpu::RenderPipeline,
    hud_pipeline: gpu::RenderPipeline,
    command_encoder: gpu::CommandEncoder,
    params_buf: gpu::Buffer,
    depth_texture: gpu::Texture,
    depth_views: [gpu::TextureView; MAX_XR_EYES],
    start_time: Instant,
    last_report: Instant,
    rendered_frames: u64,
}

impl Example {
    fn new(context: &gpu::Context) -> Self {
        let scene_config = context
            .xr_recommended_surface_config(VIEW_TYPE)
            .expect("Unable to query the XR surface configuration");
        let scene_surface = context
            .create_xr_surface_configured(scene_config)
            .expect("Unable to create the XR scene swapchain");
        let hud_surface = context
            .create_xr_surface_configured(gpu::XrSurfaceConfig {
                size: HUD_SIZE,
                view_count: 1,
                ..scene_config
            })
            .expect("Unable to create the XR HUD swapchain");

        let style = gpu::XrPassthroughStyle {
            opacity: 1.0,
            edge_color: [0.2, 0.9, 1.0, 1.0],
            color_map: gpu::XrPassthroughColorMap::tint([1.0, 0.15, 0.1, 1.0]),
        };
        let passthrough = context.create_xr_passthrough(&style);
        if passthrough.is_none() {
            log::warn!("No passthrough: the scene is drawn over black");
        }
        let depth = context.create_xr_environment_depth();
        match depth {
            Some(ref depth) => {
                if let Err(e) = depth.start() {
                    log::warn!("Environment depth didn't start: {e}");
                }
            }
            None => log::warn!("No environment depth (is USE_SCENE granted?)"),
        }
        let no_depth = context.create_texture(gpu::TextureDesc {
            name: "no-depth",
            format: gpu::TextureFormat::Depth16Unorm,
            size: gpu::Extent {
                width: 1,
                height: 1,
                depth: 1,
            },
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 2,
            mip_level_count: 1,
            usage: gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        let no_depth_view = context.create_texture_view(
            no_depth,
            gpu::TextureViewDesc {
                name: "no-depth",
                format: gpu::TextureFormat::Depth16Unorm,
                dimension: gpu::ViewDimension::D2Array,
                subresources: &Default::default(),
            },
        );
        let view_space = context
            .xr_create_reference_space(xr::ReferenceSpaceType::VIEW)
            .expect("Unable to create the XR view space");

        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("passthrough.wgsl"),
            naga_module: None,
        });
        let scene_layout = <Params as gpu::ShaderData>::layout();
        let scene_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "passthrough-scene",
            data_layouts: &[&scene_layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState::default(),
            depth_stencil: Some(gpu::DepthStencilState {
                format: gpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: gpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState::from(scene_surface.format())],
            multisample_state: gpu::MultisampleState::default(),
        });
        let hud_layout = <HudData as gpu::ShaderData>::layout();
        let hud_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "passthrough-hud",
            data_layouts: &[&hud_layout],
            vertex: shader.at("hud_vs"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState::default(),
            depth_stencil: None,
            fragment: Some(shader.at("hud_fs")),
            color_targets: &[gpu::ColorTargetState::from(hud_surface.format())],
            multisample_state: gpu::MultisampleState::default(),
        });

        let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "passthrough",
            buffer_count: 1,
            manual_barriers: false,
        });
        let params_buf = context.create_buffer(gpu::BufferDesc {
            name: "passthrough-params",
            size: (std::mem::size_of::<Uniforms>() * MAX_XR_EYES) as u64,
            memory: gpu::Memory::Shared,
        });
        let extent = scene_surface.extent();
        let view_count = scene_surface.view_count();
        let depth_texture = context.create_texture(gpu::TextureDesc {
            name: "passthrough-depth",
            format: gpu::TextureFormat::Depth32Float,
            size: extent,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: view_count,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET,
            sample_count: 1,
            external: None,
        });
        let mut depth_views = [gpu::TextureView::default(); MAX_XR_EYES];
        for (eye, view) in depth_views.iter_mut().enumerate().take(view_count as usize) {
            *view = context.create_texture_view(
                depth_texture,
                gpu::TextureViewDesc {
                    name: "passthrough-depth-eye",
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

        // Blade's attachments and resources are in the general layout, which
        // this takes them to once.
        command_encoder.start();
        command_encoder.init_texture(no_depth);
        command_encoder.init_texture(depth_texture);
        context
            .wait_for(&context.submit(&mut command_encoder), !0)
            .unwrap();

        Self {
            scene_surface,
            hud_surface,
            passthrough,
            depth,
            no_depth,
            no_depth_view,
            view_space,
            scene_pipeline,
            hud_pipeline,
            command_encoder,
            params_buf,
            depth_texture,
            depth_views,
            start_time: Instant::now(),
            last_report: Instant::now(),
            rendered_frames: 0,
        }
    }

    fn render(&mut self, context: &gpu::Context) {
        let Some(frame) = context.xr_begin_frame() else {
            return;
        };
        let scene = self.scene_surface.acquire_image();
        let hud = self.hud_surface.acquire_image();
        let (Some(scene), Some(hud)) = (scene, hud) else {
            log::error!("Unable to acquire the XR swapchain images");
            // What was acquired goes back, and the frame ends empty.
            if scene.is_some() {
                self.scene_surface.release_image();
            }
            if hud.is_some() {
                self.hud_surface.release_image();
            }
            let _ = context.xr_end_frame(frame, gpu::XrBlendMode::Opaque, &[]);
            return;
        };
        let environment_depth = self
            .depth
            .as_ref()
            .and_then(|depth| depth.acquire(context, &frame));

        self.command_encoder.start();
        // Both are cleared, so what they held doesn't matter.
        self.command_encoder.init_texture(scene.texture());
        self.command_encoder.init_texture(hud.texture());
        if let Some(ref depth) = environment_depth {
            self.command_encoder.begin_xr_depth(depth);
        }

        let time = self.start_time.elapsed().as_secs_f32();
        let model = Mat4::from_translation(Vec3::new(0.0, 0.0, -1.5)) * Mat4::from_rotation_y(time);
        let views = frame.views();
        let view_count = views.len().min(MAX_XR_EYES);
        for (eye, xr_view) in views.iter().take(view_count).enumerate() {
            let view = view_matrix_from_pose(xr_view.pose);
            let proj = projection_matrix_from_fov(xr_view.fov, NEAR_Z, FAR_Z);
            let uniforms = Uniforms {
                mvp: (proj * view * model).to_cols_array_2d(),
                tint: [1.0; 4],
            };
            unsafe {
                *(self
                    .params_buf
                    .data()
                    .add(eye * std::mem::size_of::<Uniforms>())
                    as *mut Uniforms) = uniforms;
            }
        }
        context.sync_buffer(
            self.params_buf.into(),
            self.params_buf.size(),
            gpu::BufferTarget::Data,
        );

        for eye in 0..view_count {
            let mut pass = self.command_encoder.render(
                "passthrough-eye",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: scene.layer_view(eye as u32),
                        // Clear to nothing, where the passthrough shows.
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: Some(gpu::RenderTarget {
                        view: self.depth_views[eye],
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                        finish_op: gpu::FinishOp::Discard,
                    }),
                },
            );
            let mut rc = pass.with(&self.scene_pipeline);
            rc.bind(
                0,
                &Params {
                    globals: self
                        .params_buf
                        .at((eye * std::mem::size_of::<Uniforms>()) as u64),
                },
            );
            rc.draw(0, 12, 0, 1);
        }

        {
            let mut pass = self.command_encoder.render(
                "passthrough-hud",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: hud.layer_view(0),
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            );
            let mut rc = pass.with(&self.hud_pipeline);
            rc.bind(
                0,
                &HudData {
                    hud: HudParams {
                        time,
                        has_depth: environment_depth.is_some() as u32,
                        pad: [0.0; 2],
                    },
                    depth_tex: environment_depth.map_or(self.no_depth_view, |depth| depth.view),
                },
            );
            rc.draw(0, 3, 0, 1);
        }

        // Back in the layouts that the runtime takes them in.
        self.command_encoder.end_xr_image(&scene);
        self.command_encoder.end_xr_image(&hud);
        if let Some(ref depth) = environment_depth {
            self.command_encoder.end_xr_depth(depth);
        }
        let sync_point = context.submit(&mut self.command_encoder);
        // The images go back to the runtime once the work is done, like
        // `CommandEncoder::present` does.
        context.wait_for(&sync_point, !0).unwrap();
        self.scene_surface.release_image();
        self.hud_surface.release_image();

        let mut layers = Vec::new();
        if let Some(ref passthrough) = self.passthrough {
            layers.push(passthrough.layer());
        }
        layers.push(gpu::XrLayer::Projection(gpu::XrProjectionLayer {
            surface: &self.scene_surface,
            views: None,
            space: None,
            flags: gpu::XrLayerFlags::BLEND_SOURCE_ALPHA,
        }));
        layers.push(gpu::XrLayer::Quad(gpu::XrQuadLayer {
            surface: &self.hud_surface,
            array_index: 0,
            space: Some(&self.view_space),
            pose: gpu::XrPose {
                orientation: [0.0, 0.0, 0.0, 1.0],
                position: [0.0, -0.25, -1.0],
            },
            size: [0.5, 0.25],
            eye_visibility: gpu::XrEyeVisibility::Both,
            flags: gpu::XrLayerFlags::BLEND_SOURCE_ALPHA,
        }));
        if let Err(e) = context.xr_end_frame(frame, gpu::XrBlendMode::Opaque, &layers) {
            log::error!("XR frame end failed: {e}");
        }
        self.rendered_frames += 1;

        if self.last_report.elapsed() >= Duration::from_secs(1) {
            self.last_report = Instant::now();
            self.report_head_pose(context);
        }
    }

    /// Where the head was a moment ago, as for the time a camera took an
    /// image: Android camera timestamps are of `CLOCK_BOOTTIME` when their
    /// source is real time.
    // `timespec` is narrower on 32-bit targets.
    #[allow(clippy::useless_conversion)]
    fn report_head_pose(&self, context: &gpu::Context) {
        let mut now = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        if unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, &mut now) } != 0 {
            return;
        }
        let captured = i64::from(now.tv_sec) * 1_000_000_000 + i64::from(now.tv_nsec) - 30_000_000;
        match context.xr_time_from_clock(libc::CLOCK_BOOTTIME, captured) {
            Some(time) => info!(
                "Head 30 ms ago: {:?}, after {} frames",
                context.xr_locate(&self.view_space, time),
                self.rendered_frames
            ),
            None => info!("No XR_KHR_convert_timespec_time to locate the head in the past"),
        }
    }

    fn destroy(mut self, context: &gpu::Context) {
        if let Some(mut depth) = self.depth.take() {
            let _ = depth.stop();
            context.destroy_xr_environment_depth(&mut depth);
        }
        self.passthrough = None;
        for view in &mut self.depth_views {
            if *view != gpu::TextureView::default() {
                context.destroy_texture_view(*view);
            }
        }
        context.destroy_texture(self.depth_texture);
        context.destroy_texture_view(self.no_depth_view);
        context.destroy_texture(self.no_depth);
        context.destroy_buffer(self.params_buf);
        context.destroy_xr_surface(&mut self.scene_surface);
        context.destroy_xr_surface(&mut self.hud_surface);
        context.destroy_command_encoder(&mut self.command_encoder);
        context.destroy_render_pipeline(&mut self.scene_pipeline);
        context.destroy_render_pipeline(&mut self.hud_pipeline);
    }
}

#[derive(Default)]
enum AppState {
    #[default]
    Idle,
    Running(Box<Example>),
}

fn spawn_android_event_pump() -> Arc<AtomicBool> {
    let should_exit = Arc::new(AtomicBool::new(false));
    let should_exit_thread = Arc::clone(&should_exit);
    thread::spawn(move || {
        while let Some(event) = ndk_glue::poll_events() {
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
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("blade-passthrough"),
    );

    let entry = unsafe {
        xr::Entry::load()
            .expect("couldn't find the OpenXR loader; try enabling the \"static\" feature")
    };
    entry.initialize_android_loader().unwrap();

    let available = entry.enumerate_extensions().unwrap();
    assert!(available.khr_vulkan_enable2);
    let mut extensions = xr::ExtensionSet::default();
    extensions.khr_vulkan_enable2 = true;
    extensions.khr_android_create_instance = true;
    // The rest is optional: each helper is `None` without its extension.
    extensions.fb_passthrough = available.fb_passthrough;
    extensions.meta_environment_depth = available.meta_environment_depth;
    extensions.khr_convert_timespec_time = available.khr_convert_timespec_time;
    info!(
        "Passthrough: {}, environment depth: {}, timespec time: {}",
        available.fb_passthrough,
        available.meta_environment_depth,
        available.khr_convert_timespec_time
    );

    let xr_instance = entry
        .create_instance(
            &xr::ApplicationInfo {
                application_name: "Blade Passthrough",
                application_version: 0,
                engine_name: "Blade",
                engine_version: 0,
                api_version: xr::Version::new(1, 0, 0),
            },
            &extensions,
            &[],
        )
        .unwrap();
    let system = xr_instance
        .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
        .unwrap();
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
    info!("XR blend modes: {:?}", context.xr_blend_modes());

    let mut state = AppState::Idle;
    let should_exit = spawn_android_event_pump();
    let mut event_storage = xr::EventDataBuffer::new();

    'main_loop: loop {
        if should_exit.load(Ordering::Relaxed) {
            break 'main_loop;
        }
        while let Some(event) = xr_instance.poll_event(&mut event_storage).unwrap() {
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => match e.state() {
                    xr::SessionState::READY => {
                        // Passthrough and the depth of the environment need
                        // a running session.
                        context.xr_session().unwrap().begin(VIEW_TYPE).unwrap();
                        if matches!(state, AppState::Idle) {
                            state = AppState::Running(Box::new(Example::new(&context)));
                        }
                    }
                    xr::SessionState::STOPPING => {
                        context.xr_session().unwrap().end().unwrap();
                        if let AppState::Running(example) =
                            std::mem::replace(&mut state, AppState::Idle)
                        {
                            example.destroy(&context);
                        }
                    }
                    xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                        break 'main_loop;
                    }
                    _ => {}
                },
                InstanceLossPending(_) => break 'main_loop,
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
