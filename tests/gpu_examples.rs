#![allow(irrefutable_let_patterns)]

use blade_graphics as gpu;
use blade_graphics::ShaderData;
use std::slice;

#[path = "../examples/bunnymark/example.rs"]
mod bunnymark_example;
mod snapshot;

#[derive(Clone, Copy)]
struct DispatchGlobals {
    input: gpu::BufferPiece,
    output: gpu::BufferPiece,
}

impl gpu::ShaderData for DispatchGlobals {
    fn layout() -> gpu::ShaderDataLayout {
        gpu::ShaderDataLayout {
            bindings: vec![
                ("input", gpu::ShaderBinding::Buffer),
                ("output", gpu::ShaderBinding::Buffer),
            ],
        }
    }

    fn fill(&self, mut ctx: gpu::PipelineContext) {
        use gpu::ShaderBindable as _;
        self.input.bind_to(&mut ctx, 0);
        self.output.bind_to(&mut ctx, 1);
    }
}

#[derive(blade_macros::ShaderData)]
struct EnvSampleData {
    env_main: gpu::TextureView,
    env_weights: gpu::TextureView,
}

struct EnvMapSampler {
    sample_count: u32,
    accum_texture: gpu::Texture,
    accum_view: gpu::TextureView,
    init_pipeline: gpu::RenderPipeline,
    accum_pipeline: gpu::RenderPipeline,
}

impl EnvMapSampler {
    fn new(size: gpu::Extent, shader: &gpu::Shader, context: &gpu::Context) -> Self {
        let format = gpu::TextureFormat::Rgba16Float;
        let accum_texture = context.create_texture(gpu::TextureDesc {
            name: "env-test",
            format,
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
            sample_count: 1,
            external: None,
        });
        let accum_view = context.create_texture_view(
            accum_texture,
            gpu::TextureViewDesc {
                name: "env-test",
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );

        let layout = <EnvSampleData as gpu::ShaderData>::layout();
        let init_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "env-init",
            data_layouts: &[&layout],
            vertex: shader.at("vs_init"),
            vertex_fetches: &[],
            fragment: Some(shader.at("fs_init")),
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            color_targets: &[gpu::ColorTargetState {
                format,
                blend: None,
                write_mask: gpu::ColorWrites::ALL,
            }],
            multisample_state: gpu::MultisampleState::default(),
        });
        let accum_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "env-accum",
            data_layouts: &[&layout],
            vertex: shader.at("vs_accum"),
            vertex_fetches: &[],
            fragment: Some(shader.at("fs_accum")),
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            color_targets: &[gpu::ColorTargetState {
                format,
                blend: Some(gpu::BlendState::ADDITIVE),
                write_mask: gpu::ColorWrites::RED,
            }],
            multisample_state: gpu::MultisampleState::default(),
        });

        Self {
            sample_count: size.width * size.height * 2,
            accum_texture,
            accum_view,
            init_pipeline,
            accum_pipeline,
        }
    }

    fn accumulate(
        &self,
        command_encoder: &mut gpu::CommandEncoder,
        env_main: gpu::TextureView,
        env_weights: gpu::TextureView,
    ) {
        command_encoder.init_texture(self.accum_texture);
        let mut pass = command_encoder.render(
            "accumulate",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: self.accum_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        );
        if let mut encoder = pass.with(&self.init_pipeline) {
            encoder.bind(
                0,
                &EnvSampleData {
                    env_main,
                    env_weights,
                },
            );
            encoder.draw(0, 4, 0, 1);
        };
        if let mut encoder = pass.with(&self.accum_pipeline) {
            encoder.bind(
                0,
                &EnvSampleData {
                    env_main,
                    env_weights,
                },
            );
            encoder.draw(0, self.sample_count, 0, 1);
        };
    }

    fn destroy(mut self, context: &gpu::Context) {
        context.destroy_render_pipeline(&mut self.init_pipeline);
        context.destroy_render_pipeline(&mut self.accum_pipeline);
        context.destroy_texture_view(self.accum_view);
        context.destroy_texture(self.accum_texture);
    }
}

#[test]
#[ignore = "requires a working GPU context"]
fn dispatch_gpu_test() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };

    let input = context.create_buffer(gpu::BufferDesc {
        name: "dispatch-input",
        size: 16,
        memory: gpu::Memory::Shared,
    });
    let output = context.create_buffer(gpu::BufferDesc {
        name: "dispatch-output",
        size: 16,
        memory: gpu::Memory::Shared,
    });

    unsafe {
        let input_data = slice::from_raw_parts_mut(input.data() as *mut u32, 4);
        input_data.copy_from_slice(&[1, 2, 3, 4]);
    }
    context.sync_buffer(input);

    let shader = context.create_shader(gpu::ShaderDesc {
        source: include_str!("shaders/dispatch.wgsl"),
    });
    let global_layout = DispatchGlobals::layout();
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "dispatch-test",
        data_layouts: &[&global_layout],
        compute: shader.at("main"),
    });

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "dispatch-test",
        buffer_count: 1,
    });
    command_encoder.start();
    if let mut compute = command_encoder.compute("dispatch") {
        if let mut pass = compute.with(&pipeline) {
            pass.bind(
                0,
                &DispatchGlobals {
                    input: input.into(),
                    output: output.into(),
                },
            );
            pass.dispatch([1, 1, 1]);
        }
    }

    let sync_point = context.submit(&mut command_encoder);
    assert!(context.wait_for(&sync_point, 2000));

    let actual = unsafe { slice::from_raw_parts(output.data() as *const u32, 4) };
    let expected = [3, 5, 7, 9];
    assert_eq!(actual, expected);

    context.destroy_command_encoder(&mut command_encoder);
    context.destroy_compute_pipeline(&mut pipeline);
    context.destroy_buffer(output);
    context.destroy_buffer(input);
}

#[test]
#[ignore = "requires a working GPU context"]
fn env_map_gpu_test() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };

    let shader_prepare = context.create_shader(gpu::ShaderDesc {
        source: include_str!("../blade-render/code/env-prepare.wgsl"),
    });
    let shader_sample = context.create_shader(gpu::ShaderDesc {
        source: include_str!("shaders/env_map_sample.wgsl"),
    });

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "env-map-test",
        buffer_count: 1,
    });
    command_encoder.start();

    let mut dummy = blade_render::DummyResources::new(&mut command_encoder, &context);
    let mut env_map = blade_render::EnvironmentMap::new(&shader_prepare, &dummy, &context);
    env_map.assign(dummy.white_view, dummy.size, &mut command_encoder, &context);

    let env_sampler = EnvMapSampler::new(dummy.size, &shader_sample, &context);
    env_sampler.accumulate(&mut command_encoder, env_map.main_view, env_map.weight_view);

    let readback = context.create_buffer(gpu::BufferDesc {
        name: "env-map-readback",
        size: 8,
        memory: gpu::Memory::Shared,
    });
    if let mut transfer = command_encoder.transfer("readback-env-map") {
        transfer.copy_texture_to_buffer(
            env_sampler.accum_texture.into(),
            readback.into(),
            8,
            gpu::Extent {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
    }

    let sync_point = context.submit(&mut command_encoder);
    assert!(context.wait_for(&sync_point, 2000));

    let actual = unsafe { slice::from_raw_parts(readback.data(), 8) };
    assert!(
        actual.iter().any(|b| *b != 0),
        "environment map output is entirely zero"
    );

    context.destroy_buffer(readback);
    env_map.destroy(&context);
    dummy.destroy(&context);
    env_sampler.destroy(&context);
    context.destroy_command_encoder(&mut command_encoder);
}

#[test]
#[ignore = "requires a working GPU context"]
fn snapshot_bunnymark() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    let size = gpu::Extent {
        width: 400,
        height: 300,
        depth: 1,
    };
    let format = gpu::TextureFormat::Rgba8Unorm;

    let target = snapshot::OffscreenTarget::new(&context, size, format);
    let mut example = bunnymark_example::Example::new(&context, size, format);

    // Add bunnies and step the simulation for a deterministic scene
    example.increase();
    for _ in 0..10 {
        example.step(0.01);
    }

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "snapshot-bunnymark",
        buffer_count: 1,
    });
    command_encoder.start();
    command_encoder.init_texture(target.texture);
    example.render(&mut command_encoder, target.view);

    let pixels = target.read_pixels(&context, &mut command_encoder);

    let reference_path = std::path::Path::new("tests/reference/bunnymark.png");
    if std::env::var("BLADE_UPDATE_SNAPSHOTS").is_ok() {
        snapshot::save_image(reference_path, &pixels, size);
        println!("Updated reference image: {}", reference_path.display());
    } else {
        let (reference, ref_size) = snapshot::load_reference(reference_path);
        assert_eq!(
            ref_size, size,
            "Reference image size mismatch: expected {:?}, got {:?}",
            size, ref_size
        );
        if let Err(report) = snapshot::compare_images(&pixels, &reference, size, 2) {
            let actual_path = std::path::Path::new("tests/reference/bunnymark_actual.png");
            snapshot::save_image(actual_path, &pixels, size);
            panic!(
                "Bunnymark snapshot mismatch! {}\nActual output saved to: {}",
                report,
                actual_path.display()
            );
        }
    }

    example.deinit(&context);
    context.destroy_command_encoder(&mut command_encoder);
    target.destroy(&context);
}
