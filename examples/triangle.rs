use lame;

//#[derive(lame::ShaderData)]
struct Globals {
    some_uniform: f32,
    other_vec: [f32; 4],
    diffuse_tex: lame::TextureView,
}

impl lame::ShaderData for Globals {
    fn layout() -> lame::ShaderDataLayout {
        lame::ShaderDataLayout {
            plain_size: 32,
            bindings: vec![
                (
                    "some_uniform".to_string(),
                    lame::ShaderBinding::Plain {
                        ty: lame::PlainType::F32,
                        container: lame::PlainContainer::Scalar,
                        offset: 0,
                    },
                ),
                (
                    "other_vec".to_string(),
                    lame::ShaderBinding::Plain {
                        ty: lame::PlainType::F32,
                        container: lame::PlainContainer::Vector(lame::VectorSize::Quad),
                        offset: 16,
                    },
                ),
                (
                    "diffuse_tex".to_string(),
                    lame::ShaderBinding::Resource {
                        ty: lame::BindingType::Texture,
                    },
                ),
            ],
        }
    }
    fn fill(&self, collector: &mut lame::ShaderDataCollector) {
        /*use std::mem;
        unsafe {
            std::ptr::copy_nonoverlapping(
                &self.some_uniform as *const _ as *const u8,
                collector.plain_data.as_mut_ptr().add(0),
                mem::size_of::<f32>(),
            );
            std::ptr::copy_nonoverlapping(
                &self.other_vec as *const _ as *const u8,
                collector.plain_data.as_mut_ptr().add(16),
                mem::size_of::<[f32; 4]>(),
            );
        }
        collector.textures.push(lame::TextureBinding {
            view: &self.diffuse_tex.raw,
            usage: lame::TextureUses::RESOURCE,
        });*/
    }
}

fn main() {
    let context = unsafe { lame::Context::init(lame::ContextDesc { validation: true }).unwrap() };

    let global_layout = <Globals as lame::ShaderData>::layout();
    let shader_source = std::fs::read_to_string("examples/foo.wgsl").unwrap();
    let shader = context.create_shader(lame::ShaderDesc {
        source: &shader_source,
        data_layouts: &[&global_layout],
    });

    let pipeline = context.create_render_pipeline(lame::RenderPipelineDesc {
        name: "main",
        layouts: &[&global_layout],
        vertex: shader.at("vs"),
        fragment: shader.at("fs"),
    });

    let res_texture = context.create_texture(lame::TextureDesc {
        name: "",
        format: lame::TextureFormat::Rgba8Unorm,
    });
    let res_view = context.create_texture_view(lame::TextureViewDesc {
        name: "",
        texture: res_texture,
    });

    let target_texture = context.create_texture(lame::TextureDesc {
        name: "target",
        format: lame::TextureFormat::Rgba8Unorm,
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
