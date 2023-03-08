#![allow(irrefutable_let_patterns)]

use std::time;

const TARGET_FORMAT: blade::TextureFormat = blade::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Parameters {
    cam_position: [f32; 3],
    depth: f32,
    cam_orientation: [f32; 4],
    fov: [f32; 2],
    pad: [f32; 2],
}

#[derive(blade::ShaderData)]
struct ShaderData {
    parameters: Parameters,
    acc_struct: blade::AccelerationStructure,
    output: blade::TextureView,
}

#[derive(blade::ShaderData)]
struct DrawData {
    input: blade::TextureView,
}

struct Camera {
    pos: glam::Vec3,
    rot: glam::Quat,
    fov: f32,
}

struct Example {
    _start_time: time::Instant,
    prev_sync_point: Option<blade::SyncPoint>,
    target: blade::Texture,
    target_view: blade::TextureView,
    command_encoder: blade::CommandEncoder,
    rt_pipeline: blade::ComputePipeline,
    draw_pipeline: blade::RenderPipeline,
    scene: blade_render::Scene,
    context: blade::Context,
    screen_size: blade::Extent,
    camera: Camera,
}

impl Example {
    fn new(window: &winit::window::Window, gltf_path: &str, camera: Camera) -> Self {
        let window_size = window.inner_size();
        let context = unsafe {
            blade::Context::init_windowed(
                window,
                blade::ContextDesc {
                    validation: cfg!(debug_assertions),
                    capture: false,
                },
            )
            .unwrap()
        };
        let capabilities = context.capabilities();
        assert!(capabilities
            .ray_query
            .contains(blade::ShaderVisibility::COMPUTE));

        let screen_size = blade::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        };
        let target = context.create_texture(blade::TextureDesc {
            name: "main",
            format: TARGET_FORMAT,
            size: screen_size,
            dimension: blade::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: blade::TextureUsage::RESOURCE | blade::TextureUsage::STORAGE,
        });
        let target_view = context.create_texture_view(blade::TextureViewDesc {
            name: "main",
            texture: target,
            format: TARGET_FORMAT,
            dimension: blade::ViewDimension::D2,
            subresources: &blade::TextureSubresources::default(),
        });
        let surface_format = context.resize(blade::SurfaceConfig {
            size: screen_size,
            usage: blade::TextureUsage::TARGET,
            frame_count: 3,
        });

        let source = std::fs::read_to_string("examples/scene/shader.wgsl").unwrap();
        let shader = context.create_shader(blade::ShaderDesc { source: &source });
        let rt_layout = <ShaderData as blade::ShaderData>::layout();
        let rt_pipeline = context.create_compute_pipeline(blade::ComputePipelineDesc {
            name: "ray-trace",
            data_layouts: &[&rt_layout],
            compute: shader.at("main"),
        });
        let draw_layout = <DrawData as blade::ShaderData>::layout();
        let draw_pipeline = context.create_render_pipeline(blade::RenderPipelineDesc {
            name: "main",
            data_layouts: &[&draw_layout],
            primitive: blade::PrimitiveState {
                topology: blade::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            vertex: shader.at("draw_vs"),
            fragment: shader.at("draw_fs"),
            color_targets: &[surface_format.into()],
            depth_stencil: None,
        });

        let mut command_encoder = context.create_command_encoder(blade::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();
        command_encoder.init_texture(target);

        let (scene, temp_buffers) =
            blade_render::Scene::load_gltf(gltf_path.as_ref(), &mut command_encoder, &context);
        let sync_point = context.submit(&mut command_encoder);
        context.wait_for(&sync_point, !0);
        for buffer in temp_buffers {
            context.destroy_buffer(buffer);
        }

        Self {
            _start_time: time::Instant::now(),
            prev_sync_point: None,
            target,
            target_view,
            scene,
            rt_pipeline,
            draw_pipeline,
            command_encoder,
            context,
            screen_size,
            camera,
        }
    }

    fn destroy(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        self.context.destroy_texture_view(self.target_view);
        self.context.destroy_texture(self.target);
        self.scene.destroy(&self.context);
    }

    fn render(&mut self) {
        self.command_encoder.start();

        if let mut pass = self.command_encoder.compute() {
            if let mut pc = pass.with(&self.rt_pipeline) {
                let wg_size = self.rt_pipeline.get_workgroup_size();
                let group_count = [
                    (self.screen_size.width + wg_size[0] - 1) / wg_size[0],
                    (self.screen_size.height + wg_size[1] - 1) / wg_size[1],
                    1,
                ];
                let fov_y = self.camera.fov;
                let fov_x = fov_y * self.screen_size.width as f32 / self.screen_size.height as f32;

                pc.bind(
                    0,
                    &ShaderData {
                        parameters: Parameters {
                            cam_position: self.camera.pos.into(),
                            depth: 100.0,
                            cam_orientation: self.camera.rot.into(),
                            fov: [fov_x, fov_y],
                            pad: [0.0; 2],
                        },
                        acc_struct: self.scene.acceleration_structure,
                        output: self.target_view,
                    },
                );
                pc.dispatch(group_count);
            }
        }

        let frame = self.context.acquire_frame();
        self.command_encoder.init_texture(frame.texture());

        if let mut pass = self.command_encoder.render(blade::RenderTargetSet {
            colors: &[blade::RenderTarget {
                view: frame.texture_view(),
                init_op: blade::InitOp::Clear(blade::TextureColor::TransparentBlack),
                finish_op: blade::FinishOp::Store,
            }],
            depth_stencil: None,
        }) {
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
        let sync_point = self.context.submit(&mut self.command_encoder);

        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        self.prev_sync_point = Some(sync_point);
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("blade-scene")
        .build(&event_loop)
        .unwrap();

    let camera = Camera {
        pos: glam::Vec3::new(0.0, 1.0, 5.0),
        rot: glam::Quat::from_xyzw(0.0, 1.0, 0.0, 0.0),
        fov: 0.3,
    };
    let mut example = Example::new(&window, "examples/scene/data/cornellBox.gltf", camera);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
        match event {
            winit::event::Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            winit::event::Event::WindowEvent { event, .. } => match event {
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
                    _ => {}
                },
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                _ => {}
            },
            winit::event::Event::RedrawRequested(_) => {
                *control_flow = winit::event_loop::ControlFlow::Wait;
                example.render();
            }
            winit::event::Event::LoopDestroyed => {
                example.destroy();
            }
            _ => {}
        }
    })
}
