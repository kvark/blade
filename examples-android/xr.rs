#![cfg(target_os = "android")]
#![allow(irrefutable_let_patterns)]

use std::{
    mem, ptr,
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

const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;
const TORUS_RADIUS: f32 = 1.5;
const FAR_Z: f32 = 100.0;
const TARGET_FORMAT: gpu::TextureFormat = gpu::TextureFormat::Rgba8Unorm;
const XR_WORLD_Z_OFFSET: f32 = 8.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Parameters {
    cam_position: [f32; 3],
    depth: f32,
    cam_orientation: [f32; 4],
    fov: [f32; 4], // left, right, down, up
    torus_radius: f32,
    rotation_angle: f32,
    pad: [f32; 2],
}

#[derive(blade_macros::ShaderData)]
struct TraceData {
    parameters: Parameters,
    acc_struct: gpu::AccelerationStructure,
    output: gpu::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct DrawData {
    input: gpu::TextureView,
}

struct Example {
    xr_surface: gpu::XrSurface,
    target: gpu::Texture,
    target_view: gpu::TextureView,
    blas: gpu::AccelerationStructure,
    tlas: gpu::AccelerationStructure,
    rt_pipeline: gpu::ComputePipeline,
    draw_pipeline: gpu::RenderPipeline,
    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,
    start_time: Instant,
    xr_debug: bool,
    rendered_frames: u64,
    anchor_head_pos: Option<[f32; 3]>,
}

impl Example {
    fn new(context: &gpu::Context, xr_debug: bool) -> Self {
        let capabilities = context.capabilities();
        assert!(capabilities
            .ray_query
            .contains(gpu::ShaderVisibility::COMPUTE));

        let xr_surface = context
            .create_xr_surface()
            .expect("Unable to create XR surface");
        let extent = xr_surface.extent();

        let target = context.create_texture(gpu::TextureDesc {
            name: "ray-query-target",
            format: TARGET_FORMAT,
            size: extent,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            usage: gpu::TextureUsage::RESOURCE | gpu::TextureUsage::STORAGE,
            external: None,
        });
        let target_view = context.create_texture_view(
            target,
            gpu::TextureViewDesc {
                name: "ray-query-target",
                format: TARGET_FORMAT,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );

        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("xr.wgsl"),
        });
        shader.check_struct_size::<Parameters>();

        let rt_layout = <TraceData as gpu::ShaderData>::layout();
        let draw_layout = <DrawData as gpu::ShaderData>::layout();
        let rt_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "ray-trace",
            data_layouts: &[&rt_layout],
            compute: shader.at("main"),
        });
        let draw_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "ray-query-draw",
            data_layouts: &[&draw_layout],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            vertex: shader.at("draw_vs"),
            vertex_fetches: &[],
            fragment: Some(shader.at("draw_fs")),
            color_targets: &[gpu::ColorTargetState::from(xr_surface.format())],
            depth_stencil: None,
            multisample_state: Default::default(),
        });

        let (indices, vertex_values) =
            del_msh_core::trimesh3_primitive::torus_yup::<u16, f32>(TORUS_RADIUS, 1.0, 100, 20);
        let vertex_buf = context.create_buffer(gpu::BufferDesc {
            name: "vertices",
            size: (vertex_values.len() * mem::size_of::<f32>()) as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                vertex_values.as_ptr(),
                vertex_buf.data() as *mut f32,
                vertex_values.len(),
            )
        };

        let index_buf = context.create_buffer(gpu::BufferDesc {
            name: "indices",
            size: (indices.len() * mem::size_of::<u16>()) as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(indices.as_ptr(), index_buf.data() as *mut u16, indices.len())
        };

        let meshes = [gpu::AccelerationStructureMesh {
            vertex_data: vertex_buf.at(0),
            vertex_format: gpu::VertexFormat::F32Vec3,
            vertex_stride: mem::size_of::<f32>() as u32 * 3,
            vertex_count: vertex_values.len() as u32 / 3,
            index_data: index_buf.at(0),
            index_type: Some(gpu::IndexType::U16),
            triangle_count: indices.len() as u32 / 3,
            transform_data: gpu::Buffer::default().at(0),
            is_opaque: true,
        }];

        let blas_sizes = context.get_bottom_level_acceleration_structure_sizes(&meshes);
        let blas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "torus-blas",
            ty: gpu::AccelerationStructureType::BottomLevel,
            size: blas_sizes.data,
        });

        let x_angle = 0.5f32;
        let instances = [
            gpu::AccelerationStructureInstance {
                acceleration_structure_index: 0,
                transform: [
                    [1.0, 0.0, 0.0, -1.5],
                    [0.0, x_angle.cos(), x_angle.sin(), 0.0],
                    [0.0, -x_angle.sin(), x_angle.cos(), 0.0],
                ]
                .into(),
                mask: 0xFF,
                custom_index: 0,
            },
            gpu::AccelerationStructureInstance {
                acceleration_structure_index: 0,
                transform: [
                    [1.0, 0.0, 0.0, 1.5],
                    [0.0, x_angle.sin(), -x_angle.cos(), 0.0],
                    [0.0, x_angle.cos(), x_angle.sin(), 0.0],
                ]
                .into(),
                mask: 0xFF,
                custom_index: 0,
            },
        ];
        let tlas_sizes = context.get_top_level_acceleration_structure_sizes(instances.len() as u32);
        let instance_buffer =
            context.create_acceleration_structure_instance_buffer(&instances, &[blas]);
        let tlas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "torus-tlas",
            ty: gpu::AccelerationStructureType::TopLevel,
            size: tlas_sizes.data,
        });
        let tlas_scratch_offset =
            (blas_sizes.scratch | (gpu::limits::ACCELERATION_STRUCTURE_SCRATCH_ALIGNMENT - 1)) + 1;
        let scratch_buffer = context.create_buffer(gpu::BufferDesc {
            name: "scratch",
            size: tlas_scratch_offset + tlas_sizes.scratch,
            memory: gpu::Memory::Device,
        });

        let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "xr-ray-query",
            buffer_count: 1,
        });
        command_encoder.start();
        command_encoder.init_texture(target);
        if let mut pass = command_encoder.acceleration_structure("build BLAS") {
            pass.build_bottom_level(blas, &meshes, scratch_buffer.at(0));
        }
        if let mut pass = command_encoder.acceleration_structure("build TLAS") {
            pass.build_top_level(
                tlas,
                &[blas],
                instances.len() as u32,
                instance_buffer.at(0),
                scratch_buffer.at(tlas_scratch_offset),
            );
        }
        let sync_point = context.submit(&mut command_encoder);
        assert!(context.wait_for(&sync_point, !0));

        context.destroy_buffer(vertex_buf);
        context.destroy_buffer(index_buf);
        context.destroy_buffer(instance_buffer);
        context.destroy_buffer(scratch_buffer);

        Self {
            xr_surface,
            target,
            target_view,
            blas,
            tlas,
            rt_pipeline,
            draw_pipeline,
            command_encoder,
            prev_sync_point: None,
            start_time: Instant::now(),
            xr_debug,
            rendered_frames: 0,
            anchor_head_pos: None,
        }
    }

    fn render(&mut self, context: &gpu::Context) {
        let frame = if let Some(frame) = self.xr_surface.acquire_frame(context) {
            frame
        } else {
            return;
        };

        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());
        self.command_encoder.init_texture(self.target);

        let target_extent = self.xr_surface.extent();
        let view_count = frame.xr_view_count();
        let rotation_angle = self.start_time.elapsed().as_secs_f32() * 0.4;

        // Anchor world-space to the initial headset position for stable tracking.
        if self.anchor_head_pos.is_none() && view_count >= 2 {
            let left = frame.xr_view(0).pose.position;
            let right = frame.xr_view(1).pose.position;
            self.anchor_head_pos = Some([
                0.5 * (left[0] + right[0]),
                0.5 * (left[1] + right[1]),
                0.5 * (left[2] + right[2]),
            ]);
        }
        let anchor = self.anchor_head_pos.unwrap_or([0.0; 3]);

        for eye in 0..view_count {
            let view = frame.xr_view(eye);
            let cam_position = [
                view.pose.position[0] - anchor[0],
                view.pose.position[1] - anchor[1],
                view.pose.position[2] - anchor[2] + XR_WORLD_Z_OFFSET,
            ];
            if let mut pass = self.command_encoder.compute("ray-trace") {
                let groups = self.rt_pipeline.get_dispatch_for(target_extent);
                if let mut pc = pass.with(&self.rt_pipeline) {
                    pc.bind(
                        0,
                        &TraceData {
                            parameters: Parameters {
                                cam_position,
                                depth: FAR_Z,
                                cam_orientation: view.pose.orientation,
                                fov: [
                                    view.fov.angle_left,
                                    view.fov.angle_right,
                                    view.fov.angle_down,
                                    view.fov.angle_up,
                                ],
                                torus_radius: TORUS_RADIUS,
                                rotation_angle,
                                pad: [0.0; 2],
                            },
                            acc_struct: self.tlas,
                            output: self.target_view,
                        },
                    );
                    pc.dispatch(groups);
                }
            }

            let mut pass = self.command_encoder.render(
                "draw",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: frame.xr_texture_view(eye),
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            );
            if let mut pc = pass.with(&self.draw_pipeline) {
                pc.bind(
                    0,
                    &DrawData {
                        input: self.target_view,
                    },
                );
                pc.draw(0, 3, 0, 1);
            }
        }

        self.command_encoder.present(frame);
        let sync_point = context.submit(&mut self.command_encoder);

        if let Some(sp) = self.prev_sync_point.take() {
            context.wait_for(&sp, !0);
        }
        self.prev_sync_point = Some(sync_point);
        self.rendered_frames += 1;
        if self.xr_debug && (self.rendered_frames <= 5 || self.rendered_frames % 120 == 0) {
            info!("XR frame submitted: {}", self.rendered_frames);
        }
    }

    fn destroy(mut self, context: &gpu::Context) {
        if let Some(sp) = self.prev_sync_point.take() {
            context.wait_for(&sp, !0);
        }
        context.destroy_texture_view(self.target_view);
        context.destroy_texture(self.target);
        context.destroy_acceleration_structure(self.blas);
        context.destroy_acceleration_structure(self.tlas);
        context.destroy_xr_surface(&mut self.xr_surface);
        context.destroy_command_encoder(&mut self.command_encoder);
        context.destroy_compute_pipeline(&mut self.rt_pipeline);
        context.destroy_render_pipeline(&mut self.draw_pipeline);
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
                            context.xr_session().unwrap().begin(VIEW_TYPE).unwrap();
                            if matches!(state, AppState::Idle) {
                                state = AppState::Running(Example::new(&context, xr_debug));
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
                    }
                }
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
