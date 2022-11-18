use lame;

struct Globals {
    modulator: [f32; 4],
    input: lame::TextureView,
    output: lame::BufferSlice,
}

impl lame::ShaderData for Globals {
    fn layout() -> lame::ShaderDataLayout {
        lame::ShaderDataLayout {
            bindings: vec![
                (
                    "modulator".to_string(),
                    lame::ShaderBinding::Plain {
                        ty: lame::PlainType::F32,
                        container: lame::PlainContainer::Vector(lame::VectorSize::Quad),
                    },
                ),
                (
                    "input".to_string(),
                    lame::ShaderBinding::Texture {
                        dimension: lame::TextureViewDimension::D2,
                    },
                ),
                (
                    "output".to_string(),
                    lame::ShaderBinding::Buffer {
                        type_name: "Output",
                        access: lame::StorageAccess::STORE,
                    },
                ),
            ],
        }
    }
    fn fill<E: lame::ShaderDataEncoder>(&self, mut encoder: E) {
        encoder.set_plain(0, self.modulator);
        encoder.set_texture(1, self.input);
        encoder.set_buffer(2, self.output);
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
    let shader_source = std::fs::read_to_string("examples/minimal.wgsl").unwrap();
    let shader = context.create_shader(lame::ShaderDesc {
        source: &shader_source,
        data_layouts: &[Some(&global_layout)],
    });

    let pipeline = context.create_compute_pipeline(lame::ComputePipelineDesc {
        name: "main",
        compute: shader.at("main"),
    });

    let res_texture = context.create_texture(lame::TextureDesc {
        name: "input",
        format: lame::TextureFormat::Rgba8Unorm,
        size: lame::Extent {
            width: 16,
            height: 16,
            depth: 1,
        },
        dimension: lame::TextureDimension::D2,
        array_layers: 1,
        mip_level_count: 1,
        usage: lame::TextureUsage::RESOURCE | lame::TextureUsage::COPY,
    });
    let res_view = context.create_texture_view(lame::TextureViewDesc {
        name: "",
        texture: res_texture,
        dimension: lame::TextureViewDimension::D2,
    });

    let buffer = context.create_buffer(lame::BufferDesc {
        name: "output",
        size: 4,
        memory: lame::Memory::Shared,
    });

    let mut command_encoder =
        context.create_command_encoder(lame::CommandEncoderDesc { name: "main" });
    command_encoder.start();
    {
        let mut pc = command_encoder.with_pipeline(&pipeline);
        pc.bind_data(
            0,
            &Globals {
                modulator: [0.2, 0.4, 0.3, 0.0],
                input: res_view,
                output: buffer.at(0),
            },
        );
        pc.dispatch([1, 1, 1]);
    }
    context.submit(&mut command_encoder);

    //frame.present();
}
