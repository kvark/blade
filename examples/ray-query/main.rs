#![allow(irrefutable_let_patterns)]

use blade_graphics as gpu;
use std::{mem, ptr, time};

const TORUS_RADIUS: f32 = 3.0;
const TARGET_FORMAT: gpu::TextureFormat = gpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Parameters {
    cam_position: [f32; 3],
    depth: f32,
    cam_orientation: [f32; 4],
    fov: [f32; 2],
    torus_radius: f32,
    rotation_angle: f32,
}

#[derive(blade_macros::ShaderData)]
struct ShaderData {
    parameters: Parameters,
    acc_struct: gpu::AccelerationStructure,
    output: gpu::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct DrawData {
    input: gpu::TextureView,
}

struct Example {
    start_time: time::Instant,
    target: gpu::Texture,
    target_view: gpu::TextureView,
    blas: gpu::AccelerationStructure,
    tlas: gpu::AccelerationStructure,
    rt_pipeline: gpu::ComputePipeline,
    draw_pipeline: gpu::RenderPipeline,
    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,
    screen_size: gpu::Extent,
    context: gpu::Context,
}

impl Example {
    fn new(window: &winit::window::Window) -> Self {
        let window_size = window.inner_size();
        let context = unsafe {
            gpu::Context::init_windowed(
                window,
                gpu::ContextDesc {
                    validation: cfg!(debug_assertions),
                    capture: false,
                    overlay: false,
                },
            )
            .unwrap()
        };
        let capabilities = context.capabilities();
        assert!(capabilities
            .ray_query
            .contains(gpu::ShaderVisibility::COMPUTE));

        let screen_size = gpu::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        };

        let target = context.create_texture(gpu::TextureDesc {
            name: "main",
            format: TARGET_FORMAT,
            size: screen_size,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::RESOURCE | gpu::TextureUsage::STORAGE,
        });
        let target_view = context.create_texture_view(gpu::TextureViewDesc {
            name: "main",
            texture: target,
            format: TARGET_FORMAT,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        });

        let surface_format = context.resize(gpu::SurfaceConfig {
            size: screen_size,
            usage: gpu::TextureUsage::TARGET,
            frame_count: 3,
            color_space: gpu::ColorSpace::Linear,
        });

        let source = std::fs::read_to_string("examples/ray-query/shader.wgsl").unwrap();
        let shader = context.create_shader(gpu::ShaderDesc { source: &source });
        let rt_layout = <ShaderData as gpu::ShaderData>::layout();
        let draw_layout = <DrawData as gpu::ShaderData>::layout();
        let rt_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "ray-trace",
            data_layouts: &[&rt_layout],
            compute: shader.at("main"),
        });
        let draw_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "main",
            data_layouts: &[&draw_layout],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            vertex: shader.at("draw_vs"),
            fragment: shader.at("draw_fs"),
            color_targets: &[surface_format.into()],
            depth_stencil: None,
        });

        let (indices_usize, vertex_values) =
            del_msh::trimesh3_primitive::torus_yup(TORUS_RADIUS, 1.0, 100, 20);
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

        let indices = indices_usize
            .into_iter()
            .map(|i| i as u16)
            .collect::<Vec<_>>();
        let index_buf = context.create_buffer(gpu::BufferDesc {
            name: "indices",
            size: (indices.len() * mem::size_of::<u16>()) as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                indices.as_ptr(),
                index_buf.data() as *mut u16,
                indices.len(),
            )
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
            name: "triangle",
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
            name: "TLAS",
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
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();
        command_encoder.init_texture(target);
        if let mut pass = command_encoder.acceleration_structure() {
            pass.build_bottom_level(blas, &meshes, scratch_buffer.at(0));
        }
        //Note: separate pass in order to enforce synchronization
        if let mut pass = command_encoder.acceleration_structure() {
            pass.build_top_level(
                tlas,
                &[blas],
                instances.len() as u32,
                instance_buffer.at(0),
                scratch_buffer.at(tlas_scratch_offset),
            );
        }
        let sync_point = context.submit(&mut command_encoder);

        context.wait_for(&sync_point, !0);
        context.destroy_buffer(vertex_buf);
        context.destroy_buffer(index_buf);
        context.destroy_buffer(scratch_buffer);
        context.destroy_buffer(instance_buffer);

        Self {
            start_time: time::Instant::now(),
            target,
            target_view,
            blas,
            tlas,
            rt_pipeline,
            draw_pipeline,
            command_encoder,
            prev_sync_point: None,
            screen_size,
            context,
        }
    }

    fn delete(mut self) {
        if let Some(sp) = self.prev_sync_point {
            self.context.wait_for(&sp, !0);
        }
        self.context
            .destroy_command_encoder(&mut self.command_encoder);
        self.context.destroy_texture_view(self.target_view);
        self.context.destroy_texture(self.target);
        self.context.destroy_acceleration_structure(self.blas);
        self.context.destroy_acceleration_structure(self.tlas);
    }

    fn render(&mut self) {
        self.command_encoder.start();

        if let mut pass = self.command_encoder.compute() {
            let groups = self.rt_pipeline.get_dispatch_for(self.screen_size);
            if let mut pc = pass.with(&self.rt_pipeline) {
                let fov_y = 0.3;
                let fov_x = fov_y * self.screen_size.width as f32 / self.screen_size.height as f32;
                let rotation_angle = self.start_time.elapsed().as_secs_f32() * 0.4;

                pc.bind(
                    0,
                    &ShaderData {
                        parameters: Parameters {
                            cam_position: [0.0, 0.0, -20.0],
                            depth: 100.0,
                            cam_orientation: [0.0, 0.0, 0.0, 1.0],
                            fov: [fov_x, fov_y],
                            torus_radius: TORUS_RADIUS,
                            rotation_angle,
                        },
                        acc_struct: self.tlas,
                        output: self.target_view,
                    },
                );
                pc.dispatch(groups);
            }
        }

        let frame = self.context.acquire_frame();
        self.command_encoder.init_texture(frame.texture());

        if let mut pass = self.command_encoder.render(gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: frame.texture_view(),
                init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                finish_op: gpu::FinishOp::Store,
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
        .with_title("blade-ray-query")
        .build(&event_loop)
        .unwrap();

    let mut example = Some(Example::new(&window));

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
                example.as_mut().unwrap().render();
            }
            winit::event::Event::LoopDestroyed => {
                example.take().unwrap().delete();
            }
            _ => {}
        }
    })
}
