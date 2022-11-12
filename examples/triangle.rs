use lame;

use std::mem;

//#[derive(lame::ShaderData)]
struct Globals<'a> {
    some_uniform: f32,
    other_vec: [f32; 4],
    diffuse_tex: &'a lame::TextureView,
}

impl<'a> lame::ShaderData<'a> for Globals<'a> {
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
                        ty: lame::BindingType::Texture {
                            sample_type: lame::TextureSampleType::Float { filterable: true },
                            view_dimension: lame::TextureViewDimension::D2,
                            multisampled: false,
                        },
                    },
                ),
            ],
        }
    }
    fn fill(&self, collector: &mut lame::ShaderDataCollector<'a>) {
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
        });
    }
}

fn main() {
    let context = lame::Context::init(&lame::ContextDesc { validation: true }).unwrap();

    let global_layout = <Globals as lame::ShaderData>::layout();
    let shader_source = std::fs::read_to_string("examples/foo.wgsl").unwrap();
    let shader = context.create_shader(&lame::ShaderDesc {
        source: &shader_source,
        data_layouts: &[&global_layout],
    });

    let pipeline = context.create_render_pipeline(&lame::RenderPipelineDesc {
        layouts: &[&global_layout],
        vertex: lame::ShaderStage {
            shader: &shader,
            entry_point: "vs",
        },
        fragment: lame::ShaderStage {
            shader: &shader,
            entry_point: "fs",
        },
    });

    let texture = context.create_texture(&lame::TextureDesc {
        format: lame::TextureFormat::Rgba8Unorm,
    });
    let view = context.create_texture_view(&lame::TextureViewDesc { texture: &texture });

    //let mut frame = window.acquire_frame(&context);

    let mut command_encoder = context.create_command_encoder();
    command_encoder.begin();
    let mut pc = command_encoder.with_pipeline(&pipeline);
    pc.bind_data(
        0,
        &Globals {
            some_uniform: 0.0,
            other_vec: [0.0; 4],
            diffuse_tex: &view,
        },
    );
    pc.draw(0, 3, 0, 1);
    command_encoder.submit();

    //frame.present();
}
