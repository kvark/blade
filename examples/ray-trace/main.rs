#![allow(irrefutable_let_patterns)]

use std::{mem, ptr};

struct Example {
    //tlas: blade::AccelerationStructure,
    blas: blade::AccelerationStructure,
    blas_buffer: blade::Buffer,
    command_encoder: blade::CommandEncoder,
    prev_sync_point: Option<blade::SyncPoint>,
    context: blade::Context,
}

impl Example {
    fn new(window: &winit::window::Window) -> Self {
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

        let _surface_format = context.resize(blade::SurfaceConfig {
            size: blade::Extent {
                width: window_size.width,
                height: window_size.height,
                depth: 1,
            },
            usage: blade::TextureUsage::TARGET,
            frame_count: 3,
        });

        type Vertex = [f32; 3];
        let vertices = [
            [-0.5f32, -0.5, 0.0],
            [0.0f32, 0.5, 0.0],
            [0.5f32, -0.5, 0.0],
        ];
        let vertex_buf = context.create_buffer(blade::BufferDesc {
            name: "vertices",
            size: (vertices.len() * mem::size_of::<Vertex>()) as u64,
            memory: blade::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                vertices.as_ptr(),
                vertex_buf.data() as *mut Vertex,
                vertices.len(),
            )
        };

        let indices = [0u16, 1, 2];
        let index_buf = context.create_buffer(blade::BufferDesc {
            name: "indices",
            size: (indices.len() * mem::size_of::<u16>()) as u64,
            memory: blade::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                indices.as_ptr(),
                index_buf.data() as *mut u16,
                indices.len(),
            )
        };

        let meshes = [blade::AccelerationStructureMesh {
            vertex_data: vertex_buf.at(0),
            vertex_format: blade::VertexFormat::Rgb32Float,
            vertex_stride: mem::size_of::<Vertex>() as u32,
            vertex_count: vertices.len() as u32,
            index_data: index_buf.at(0),
            index_type: Some(blade::IndexType::U16),
            triangle_count: 1,
            transform: None,
            is_opaque: true,
        }];
        let blas_sizes = context.get_bottom_level_acceleration_structure_sizes(&meshes);
        let blas_buffer = context.create_buffer(blade::BufferDesc {
            name: "BLAS",
            size: blas_sizes.data,
            memory: blade::Memory::Device,
        });
        let scratch_buffer = context.create_buffer(blade::BufferDesc {
            name: "BLAS scratch",
            size: blas_sizes.scratch,
            memory: blade::Memory::Device,
        });

        let blas = context.create_acceleration_structure(blade::AccelerationStructureDesc {
            name: "triangle",
            ty: blade::AccelerationStructureType::BottomLevel,
            buffer: blas_buffer,
            offset: 0,
            size: blas_sizes.data,
        });

        let mut command_encoder = context.create_command_encoder(blade::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();
        if let mut pass = command_encoder.compute() {
            pass.build_bottom_level_acceleration_structure(blas, &meshes, scratch_buffer.at(0));
        }
        let sync_point = context.submit(&mut command_encoder);

        context.wait_for(&sync_point, !0);
        context.destroy_buffer(vertex_buf);
        context.destroy_buffer(index_buf);
        context.destroy_buffer(scratch_buffer);

        Self {
            blas,
            blas_buffer,
            command_encoder,
            prev_sync_point: None,
            context,
        }
    }

    fn delete(self) {
        if let Some(sp) = self.prev_sync_point {
            self.context.wait_for(&sp, !0);
        }
        self.context.destroy_acceleration_structure(self.blas);
        self.context.destroy_buffer(self.blas_buffer);
    }

    fn render(&mut self) {
        let frame = self.context.acquire_frame();

        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        if let _pass = self.command_encoder.render(blade::RenderTargetSet {
            colors: &[blade::RenderTarget {
                view: frame.texture_view(),
                init_op: blade::InitOp::Clear(blade::TextureColor::TransparentBlack),
                finish_op: blade::FinishOp::Store,
            }],
            depth_stencil: None,
        }) {
            //draw
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
        .with_title("blade-ray-trace")
        .build(&event_loop)
        .unwrap();

    let egui_ctx = egui::Context::default();
    let mut egui_winit = egui_winit::State::new(&event_loop);

    let mut example = Some(Example::new(&window));

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
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
                        _ => {}
                    },
                    winit::event::WindowEvent::CloseRequested => {
                        *control_flow = winit::event_loop::ControlFlow::Exit;
                    }
                    _ => {}
                }
            }
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
