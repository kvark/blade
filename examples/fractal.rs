use lame;

//#[derive(lame::ShaderData)]
struct Globals {
    some_uniform: f32,
    other_vec: [f32; 4],
    palette: lame::TextureView,
}

impl lame::ShaderData for Globals {
    fn layout() -> lame::ShaderDataLayout {
        lame::ShaderDataLayout {
            bindings: vec![
                (
                    "some_uniform".to_string(),
                    lame::ShaderBinding::Plain {
                        ty: lame::PlainType::F32,
                        container: lame::PlainContainer::Scalar,
                    },
                ),
                (
                    "other_vec".to_string(),
                    lame::ShaderBinding::Plain {
                        ty: lame::PlainType::F32,
                        container: lame::PlainContainer::Vector(lame::VectorSize::Quad),
                    },
                ),
                (
                    "palette".to_string(),
                    lame::ShaderBinding::Texture {
                        dimension: lame::TextureViewDimension::D1,
                    },
                ),
            ],
        }
    }
    fn fill<E: lame::ShaderDataEncoder>(&self, mut encoder: E) {
        encoder.set_plain(0, self.some_uniform);
        encoder.set_plain(1, self.other_vec);
        encoder.set_texture(2, self.palette);
    }
}

fn main() {
    env_logger::init();
    let context = unsafe {
        lame::Context::init(lame::ContextDesc {
            validation: true,
            capture: true,
        })
        .unwrap()
    };

    let global_layout = <Globals as lame::ShaderData>::layout();
    let shader_source = std::fs::read_to_string("examples/fractal.wgsl").unwrap();
    let shader = context.create_shader(lame::ShaderDesc {
        source: &shader_source,
        data_layouts: &[Some(&global_layout)],
    });

    let pipeline = context.create_render_pipeline(lame::RenderPipelineDesc {
        name: "main",
        vertex: shader.at("vs"),
        primitive: lame::PrimitiveState::default(),
        depth_stencil: None,
        fragment: shader.at("fs"),
        color_targets: &[&lame::TextureFormat::Rgba8Unorm.into()],
    });

    let res_texture = context.create_texture(lame::TextureDesc {
        name: "palette",
        format: lame::TextureFormat::Rgba8Unorm,
        size: lame::Extent {
            width: 256,
            height: 1,
            depth: 1,
        },
        dimension: lame::TextureDimension::D1,
        array_layers: 1,
        mip_level_count: 1,
        usage: lame::TextureUsage::RESOURCE | lame::TextureUsage::COPY,
    });
    let res_view = context.create_texture_view(lame::TextureViewDesc {
        name: "",
        texture: res_texture,
    });

    let target_texture = context.create_texture(lame::TextureDesc {
        name: "target",
        format: lame::TextureFormat::Rgba8Unorm,
        size: lame::Extent {
            width: 200,
            height: 200,
            depth: 1,
        },
        dimension: lame::TextureDimension::D2,
        array_layers: 1,
        mip_level_count: 1,
        usage: lame::TextureUsage::TARGET,
    });
    let target_view = context.create_texture_view(lame::TextureViewDesc {
        name: "target",
        texture: target_texture,
    });

    //let mut frame = window.acquire_frame(&context);

    let mut command_encoder =
        context.create_command_encoder(lame::CommandEncoderDesc { name: "main" });
    command_encoder.start();
    {
        let mut pass = command_encoder.with_render_targets(lame::RenderTargetSet {
            colors: &[lame::RenderTarget {
                view: target_view,
                init_op: lame::InitOp::Clear(lame::TextureColor::TransparentBlack),
                finish_op: lame::FinishOp::Store,
            }],
            depth_stencil: None,
        });
        let mut pc = pass.with_pipeline(&pipeline);
        pc.bind_data(
            0,
            &Globals {
                some_uniform: 0.0,
                other_vec: [0.0; 4],
                diffuse_tex: res_view,
            },
        );
        pc.draw(0, 3, 0, 1);
    }
    context.submit(&mut command_encoder);

    //frame.present();
}
