use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};
use std::{mem, ptr};

const BUNNY_SIZE: f32 = 0.15 * 256.0;
const GRAVITY: f32 = -9.8 * 100.0;
const MAX_VELOCITY: i32 = 750;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    mvp_transform: [[f32; 4]; 4],
    sprite_size: [f32; 2],
    pad: [f32; 2],
}

#[derive(blade_macros::ShaderData)]
struct Params {
    globals: Globals,
    sprite_texture: gpu::TextureView,
    sprite_sampler: gpu::Sampler,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Locals {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
    pad: u32,
}

#[derive(blade_macros::ShaderData)]
struct SpriteData {
    locals: Locals,
}
#[derive(blade_macros::Vertex)]
struct SpriteVertex {
    pos: [f32; 2],
}
struct Sprite {
    data: SpriteData,
    vertex_buf: gpu::BufferPiece,
}

pub struct Example {
    pipeline: gpu::RenderPipeline,
    texture: gpu::Texture,
    view: gpu::TextureView,
    sampler: gpu::Sampler,
    vertex_buf: gpu::Buffer,
    screen_size: gpu::Extent,
    bunnies: Vec<Sprite>,
    rng: nanorand::WyRand,
}

impl Example {
    pub fn new(
        context: &gpu::Context,
        screen_size: gpu::Extent,
        surface_format: gpu::TextureFormat,
    ) -> Self {
        let global_layout = <Params as gpu::ShaderData>::layout();
        let local_layout = <SpriteData as gpu::ShaderData>::layout();
        #[cfg(target_arch = "wasm32")]
        let shader_source = include_str!("shader.wgsl");
        #[cfg(not(target_arch = "wasm32"))]
        let shader_source = std::fs::read_to_string("examples/bunnymark/shader.wgsl").unwrap();
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &shader_source,
        });

        let pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "main",
            data_layouts: &[&global_layout, &local_layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[gpu::VertexFetchState {
                layout: &<SpriteVertex as gpu::Vertex>::layout(),
                instanced: false,
            }],
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState {
                format: surface_format,
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::default(),
            }],
            multisample_state: gpu::MultisampleState::default(),
        });

        let extent = gpu::Extent {
            width: 1,
            height: 1,
            depth: 1,
        };
        let texture = context.create_texture(gpu::TextureDesc {
            name: "texutre",
            format: gpu::TextureFormat::Rgba8Unorm,
            size: extent,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::RESOURCE | gpu::TextureUsage::COPY,
            sample_count: 1,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "view",
                format: gpu::TextureFormat::Rgba8Unorm,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );

        let upload_buffer = context.create_buffer(gpu::BufferDesc {
            name: "staging",
            size: (extent.width * extent.height) as u64 * 4,
            memory: gpu::Memory::Upload,
        });
        let texture_data = [0xFFu8; 4];
        unsafe {
            ptr::copy_nonoverlapping(
                texture_data.as_ptr(),
                upload_buffer.data(),
                texture_data.len(),
            );
        }
        context.sync_buffer(upload_buffer);

        let sampler = context.create_sampler(gpu::SamplerDesc {
            name: "main",
            ..Default::default()
        });

        let vertex_data = [
            SpriteVertex { pos: [0.0, 0.0] },
            SpriteVertex { pos: [1.0, 0.0] },
            SpriteVertex { pos: [0.0, 1.0] },
            SpriteVertex { pos: [1.0, 1.0] },
        ];
        let vertex_buf = context.create_buffer(gpu::BufferDesc {
            name: "vertex",
            size: (vertex_data.len() * mem::size_of::<SpriteVertex>()) as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                vertex_data.as_ptr(),
                vertex_buf.data() as *mut SpriteVertex,
                vertex_data.len(),
            );
        }
        context.sync_buffer(vertex_buf);

        let mut bunnies = Vec::new();
        bunnies.push(Sprite {
            data: SpriteData {
                locals: Locals {
                    position: [-100.0, 100.0],
                    velocity: [10.0, 0.0],
                    color: 0xFFFFFFFF,
                    pad: 0,
                },
            },
            vertex_buf: vertex_buf.into(),
        });

        let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "init",
            buffer_count: 1,
        });
        command_encoder.start();
        command_encoder.init_texture(texture);
        if let mut transfer = command_encoder.transfer("init texture") {
            transfer.copy_buffer_to_texture(upload_buffer.into(), 4, texture.into(), extent);
        }
        let sync_point = context.submit(&mut command_encoder);
        context.wait_for(&sync_point, !0);

        context.destroy_command_encoder(&mut command_encoder);
        context.destroy_buffer(upload_buffer);

        Self {
            pipeline,
            texture,
            view,
            sampler,
            vertex_buf,
            screen_size,
            bunnies,
            rng: nanorand::WyRand::new_seed(73),
        }
    }

    pub fn screen_size(&self) -> gpu::Extent {
        self.screen_size
    }

    pub fn set_screen_size(&mut self, size: gpu::Extent) {
        self.screen_size = size;
    }

    pub fn increase(&mut self) {
        use nanorand::Rng as _;
        let spawn_count = 64 + self.bunnies.len() / 2;
        for _ in 0..spawn_count {
            let speed = self.rng.generate_range(-MAX_VELOCITY..=MAX_VELOCITY) as f32;
            self.bunnies.push(Sprite {
                data: SpriteData {
                    locals: Locals {
                        position: [0.0, 0.5 * (self.screen_size.height as f32)],
                        velocity: [speed, 0.0],
                        color: self.rng.generate::<u32>(),
                        pad: 0,
                    },
                },
                vertex_buf: self.vertex_buf.into(),
            });
        }
        println!("Population: {} bunnies", self.bunnies.len());
    }

    pub fn step(&mut self, delta: f32) {
        for bunny in self.bunnies.iter_mut() {
            let Sprite {
                data:
                    SpriteData {
                        locals:
                            Locals {
                                position: ref mut pos,
                                velocity: ref mut vel,
                                ..
                            },
                    },
                ..
            } = *bunny;

            pos[0] += vel[0] * delta;
            pos[1] += vel[1] * delta;
            vel[1] += GRAVITY * delta;
            if (vel[0] > 0.0 && pos[0] + 0.5 * BUNNY_SIZE > self.screen_size.width as f32)
                || (vel[0] < 0.0 && pos[0] - 0.5 * BUNNY_SIZE < 0.0)
            {
                vel[0] *= -1.0;
            }
            if vel[1] < 0.0 && pos[1] < 0.5 * BUNNY_SIZE {
                vel[1] *= -1.0;
            }
        }
    }

    pub fn render(&mut self, encoder: &mut gpu::CommandEncoder, target: gpu::TextureView) {
        if let mut pass = encoder.render(
            "main",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            let mut rc = pass.with(&self.pipeline);
            rc.bind(
                0,
                &Params {
                    globals: Globals {
                        mvp_transform: [
                            [2.0 / self.screen_size.width as f32, 0.0, 0.0, 0.0],
                            [0.0, 2.0 / self.screen_size.height as f32, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [-1.0, -1.0, 0.0, 1.0],
                        ],
                        sprite_size: [BUNNY_SIZE; 2],
                        pad: [0.0; 2],
                    },
                    sprite_texture: self.view,
                    sprite_sampler: self.sampler,
                },
            );

            for sprite in self.bunnies.iter() {
                rc.bind(1, &sprite.data);
                rc.bind_vertex(0, sprite.vertex_buf);
                rc.draw(0, 4, 0, 1);
            }
        }
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_texture_view(self.view);
        context.destroy_texture(self.texture);
        context.destroy_sampler(self.sampler);
        context.destroy_buffer(self.vertex_buf);
        context.destroy_render_pipeline(&mut self.pipeline);
    }
}
