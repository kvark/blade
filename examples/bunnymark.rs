#![allow(irrefutable_let_patterns)]

use std::ptr;

const MAX_BUNNIES: usize = 1 << 20;
const BUNNY_SIZE: f32 = 0.15 * 256.0;
const GRAVITY: f32 = -9.8 * 100.0;
const MAX_VELOCITY: f32 = 750.0;

struct Globals {
    mvp_transform: [[f32; 4]; 4],
    sprite_size: [f32; 2],
    sprite_texture: lame::TextureView,
    sprite_sampler: lame::Sampler,
}

struct Locals {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
}

//TEMP
impl lame::ShaderData for Globals {
    fn layout() -> lame::ShaderDataLayout {
        lame::ShaderDataLayout {
            bindings: vec![
                (
                    "mvp_transform".to_string(),
                    lame::ShaderBinding::Plain {
                        ty: lame::PlainType::F32,
                        container: lame::PlainContainer::Matrix(
                            lame::VectorSize::Quad,
                            lame::VectorSize::Quad,
                        ),
                    },
                ),
                (
                    "sprite_size".to_string(),
                    lame::ShaderBinding::Plain {
                        ty: lame::PlainType::F32,
                        container: lame::PlainContainer::Vector(lame::VectorSize::Bi),
                    },
                ),
                (
                    "sprite_texture".to_string(),
                    lame::ShaderBinding::Texture {
                        dimension: lame::TextureViewDimension::D2,
                    },
                ),
                (
                    "sprite_sampler".to_string(),
                    lame::ShaderBinding::Sampler { comparison: false },
                ),
            ],
        }
    }
    fn fill<E: lame::ShaderDataEncoder>(&self, mut encoder: E) {
        encoder.set_plain(0, self.mvp_transform);
        encoder.set_plain(1, self.sprite_size);
        encoder.set_texture(2, self.sprite_texture);
        encoder.set_sampler(3, self.sprite_sampler);
    }
}

impl lame::ShaderData for Locals {
    fn layout() -> lame::ShaderDataLayout {
        lame::ShaderDataLayout {
            bindings: vec![
                (
                    "position".to_string(),
                    lame::ShaderBinding::Plain {
                        ty: lame::PlainType::F32,
                        container: lame::PlainContainer::Vector(lame::VectorSize::Bi),
                    },
                ),
                (
                    "velocity".to_string(),
                    lame::ShaderBinding::Plain {
                        ty: lame::PlainType::F32,
                        container: lame::PlainContainer::Vector(lame::VectorSize::Bi),
                    },
                ),
                (
                    "color".to_string(),
                    lame::ShaderBinding::Plain {
                        ty: lame::PlainType::U32,
                        container: lame::PlainContainer::Scalar,
                    },
                ),
            ],
        }
    }
    fn fill<E: lame::ShaderDataEncoder>(&self, mut encoder: E) {
        encoder.set_plain(0, self.position);
        encoder.set_plain(1, self.velocity);
        encoder.set_plain(2, self.color);
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("hal-bunnymark")
        .build(&event_loop)
        .unwrap();
    let window_size = window.inner_size();

    let context = unsafe {
        lame::Context::init_windowed(
            &window,
            lame::ContextDesc {
                validation: true,
                capture: true,
            },
        )
        .unwrap()
    };

    let surface_format = context.resize(lame::SurfaceConfig {
        size: lame::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        },
        usage: lame::TextureUsage::TARGET,
        frame_count: 2,
    });

    let global_layout = <Globals as lame::ShaderData>::layout();
    let local_layout = <Locals as lame::ShaderData>::layout();
    let shader_source = std::fs::read_to_string("examples/bunnymark.wgsl").unwrap();
    let shader = context.create_shader(lame::ShaderDesc {
        source: &shader_source,
        data_layouts: &[Some(&global_layout), Some(&local_layout)],
    });

    let pipeline = context.create_render_pipeline(lame::RenderPipelineDesc {
        name: "main",
        vertex: shader.at("vs_main"),
        primitive: lame::PrimitiveState {
            topology: lame::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        fragment: shader.at("fs_main"),
        color_targets: &[lame::ColorTargetState {
            format: surface_format,
            blend: Some(lame::BlendState::ALPHA_BLENDING),
            write_mask: lame::ColorWrites::default(),
        }],
    });

    let extent = lame::Extent {
        width: 1,
        height: 1,
        depth: 1,
    };
    let texture = context.create_texture(lame::TextureDesc {
        name: "texutre",
        format: lame::TextureFormat::Rgba8Unorm,
        size: extent,
        dimension: lame::TextureDimension::D2,
        array_layers: 1,
        mip_level_count: 1,
        usage: lame::TextureUsage::RESOURCE | lame::TextureUsage::COPY,
    });
    let view = context.create_texture_view(lame::TextureViewDesc {
        name: "view",
        texture,
        format: lame::TextureFormat::Rgba8Unorm,
        dimension: lame::TextureViewDimension::D2,
        subresources: &Default::default(),
    });

    let upload_buffer = context.create_buffer(lame::BufferDesc {
        name: "staging",
        size: (extent.width * extent.height) as u64 * 4,
        memory: lame::Memory::Upload,
    });
    let texture_data = vec![0xFFu8; 4];
    unsafe {
        ptr::copy_nonoverlapping(
            texture_data.as_ptr(),
            upload_buffer.data(),
            texture_data.len(),
        );
    }

    let sampler = context.create_sampler(lame::SamplerDesc {
        name: "main",
        ..Default::default()
    });

    let mut locals = Vec::new();
    locals.push(Locals {
        position: [0.0; 2],
        velocity: [0.0; 2],
        color: 0x00FF0000,
    });

    let mut command_encoder =
        context.create_command_encoder(lame::CommandEncoderDesc { name: "main" });
    let mut prev_sync_point = None;

    loop {
        let frame = context.acquire_frame();

        command_encoder.start();
        if let mut pass = command_encoder.with_render_targets(lame::RenderTargetSet {
            colors: &[lame::RenderTarget {
                view: frame.texture_view(),
                init_op: lame::InitOp::Clear(lame::TextureColor::TransparentBlack),
                finish_op: lame::FinishOp::Store,
            }],
            depth_stencil: None,
        }) {
            let mut rc = pass.with_pipeline(&pipeline);
            rc.bind_data(
                0,
                &Globals {
                    mvp_transform: [
                        [2.0 / window_size.width as f32, 0.0, 0.0, 0.0],
                        [0.0, 2.0 / window_size.height as f32, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [-1.0, -1.0, 0.0, 1.0],
                    ],
                    sprite_size: [BUNNY_SIZE; 2],
                    sprite_texture: view,
                    sprite_sampler: sampler,
                },
            );

            for local in locals.iter() {
                rc.bind_data(1, local);
                rc.draw(0, 4, 0, 1);
            }
        }

        let sync_point = context.submit(&mut command_encoder);
        if let Some(sp) = prev_sync_point.take() {
            context.wait_for(sp, !0);
        }
        context.present(frame);
        prev_sync_point = Some(sync_point);
    }
}
