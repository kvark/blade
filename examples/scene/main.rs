#![allow(irrefutable_let_patterns)]
#![cfg(not(target_arch = "wasm32"))]

use blade_render::{Camera, Renderer};
use std::time;

struct Example {
    _start_time: time::Instant,
    prev_temp_buffers: Vec<blade::Buffer>,
    prev_sync_point: Option<blade::SyncPoint>,
    renderer: Renderer,
    gui_painter: blade_egui::GuiPainter,
    command_encoder: blade::CommandEncoder,
    context: blade::Context,
    camera: blade_render::Camera,
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

        let screen_size = blade::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        };
        let surface_format = context.resize(blade::SurfaceConfig {
            size: screen_size,
            usage: blade::TextureUsage::TARGET,
            frame_count: 3,
        });
        let mut command_encoder = context.create_command_encoder(blade::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();

        let mut renderer =
            Renderer::new(&mut command_encoder, &context, screen_size, surface_format);

        let gui_painter = blade_egui::GuiPainter::new(&context, surface_format);

        let (scene, prev_temp_buffers) =
            blade_render::Scene::load_gltf(gltf_path.as_ref(), &mut command_encoder, &context);
        renderer.merge_scene(scene);
        let sync_point = context.submit(&mut command_encoder);

        Self {
            _start_time: time::Instant::now(),
            prev_temp_buffers,
            prev_sync_point: Some(sync_point),
            renderer,
            gui_painter,
            command_encoder,
            context,
            camera,
        }
    }

    fn destroy(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
        for buffer in self.prev_temp_buffers.drain(..) {
            self.context.destroy_buffer(buffer);
        }
        self.gui_painter.destroy(&self.context);
        self.renderer.destroy(&self.context);
    }

    fn render(
        &mut self,
        gui_primitives: &[egui::ClippedPrimitive],
        gui_textures: &egui::TexturesDelta,
        screen_desc: &blade_egui::ScreenDescriptor,
    ) {
        self.command_encoder.start();

        self.gui_painter
            .update_textures(&mut self.command_encoder, gui_textures, &self.context);

        let mut temp_buffers = Vec::new();
        self.renderer
            .prepare(&mut self.command_encoder, &self.context, &mut temp_buffers);
        self.renderer
            .ray_trace(&mut self.command_encoder, &self.camera);

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
            self.renderer.blit(&mut pass);
            self.gui_painter
                .paint(&mut pass, gui_primitives, screen_desc, &self.context);
        }

        self.command_encoder.present(frame);
        let sync_point = self.context.submit(&mut self.command_encoder);
        self.gui_painter.after_submit(sync_point.clone());

        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
            for buffer in self.prev_temp_buffers.drain(..) {
                self.context.destroy_buffer(buffer);
            }
        }
        self.prev_sync_point = Some(sync_point);
        self.prev_temp_buffers.extend(temp_buffers);
    }
}

fn main() {
    env_logger::init();

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
        .unwrap_or("examples/scene/data/CesiumMilkTruck.gltf".to_string());

    let camera = Camera {
        pos: [5.0, 2.0, 7.0].into(),
        rot: [0.0, 1.0, 0.0, 0.0].into(),
        fov_y: 0.8,
        depth: 1000.0,
    };
    let mut example = Example::new(&window, &path_to_scene, camera);

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
                let mut quit = false;
                let raw_input = egui_winit.take_egui_input(&window);
                let egui_output = egui_ctx.run(raw_input, |egui_ctx| {
                    egui::SidePanel::left("my_side_panel").show(egui_ctx, |ui| {
                        ui.heading("Eye");
                        if ui.button("Quit").clicked() {
                            quit = true;
                        }
                    });
                });

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

                //Note: this will probably look different with proper support for resizing
                let window_size = window.inner_size();
                let screen_desc = blade_egui::ScreenDescriptor {
                    physical_size: (window_size.width, window_size.height),
                    scale_factor: egui_ctx.pixels_per_point(),
                };

                example.render(&primitives, &egui_output.textures_delta, &screen_desc);
            }
            winit::event::Event::LoopDestroyed => {
                example.destroy();
            }
            _ => {}
        }
    })
}
