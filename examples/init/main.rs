#![allow(irrefutable_let_patterns)]

use std::{env, path::Path, ptr, sync::Arc};

use blade_graphics as gpu;

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
    fn new(size: gpu::Extent, shader: &blade_graphics::Shader, context: &gpu::Context) -> Self {
        let format = gpu::TextureFormat::Rgba16Float;
        let accum_texture = context.create_texture(gpu::TextureDesc {
            name: "env-test",
            format,
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET,
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

    fn destroy(self, context: &gpu::Context) {
        context.destroy_texture_view(self.accum_view);
        context.destroy_texture(self.accum_texture);
    }
}

const NUM_WORKERS: usize = 2;

fn main() {
    env_logger::init();
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    let mut rd = renderdoc::RenderDoc::<renderdoc::V120>::new();

    println!("Initializing");
    let context = Arc::new(unsafe {
        gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            ..Default::default()
        })
        .unwrap()
    });

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    if let Ok(ref mut rd) = rd {
        rd.start_frame_capture(ptr::null(), ptr::null());
    }

    let choir = Arc::new(choir::Choir::new());
    let _workers = (0..NUM_WORKERS)
        .map(|i| choir.add_worker(&format!("Worker-{}", i)))
        .collect::<Vec<_>>();

    let mut asset_hub = blade_render::AssetHub::new(Path::new("asset-cache"), &choir, &context);
    let mut environment_map = None;
    let mut _object = None;

    println!("Populating the scene");
    let mut load_finish = choir.spawn("load finish").init_dummy();
    let (shader_main_handle, shader_main_task) = asset_hub
        .shaders
        .load("examples/init/env-sample.wgsl", blade_render::shader::Meta);
    load_finish.depend_on(shader_main_task);
    let (shader_init_handle, shader_init_task) = asset_hub.shaders.load(
        "blade-render/code/env-prepare.wgsl",
        blade_render::shader::Meta,
    );
    load_finish.depend_on(shader_init_task);

    for arg in env::args().skip(1) {
        if arg.ends_with(".exr") {
            println!("\tenvironment map = {}", arg);
            let meta = blade_render::texture::Meta {
                format: blade_graphics::TextureFormat::Rgba32Float,
                generate_mips: false,
                y_flip: false,
            };
            let (texture, texture_task) = asset_hub.textures.load(arg, meta);
            load_finish.depend_on(texture_task);
            environment_map = Some(texture);
        } else if arg.ends_with(".gltf") {
            println!("\tmodels += {}", arg);
            let (model, model_task) = asset_hub
                .models
                .load(arg, blade_render::model::Meta::default());
            load_finish.depend_on(model_task);
            _object = Some(blade_render::Object::from(model));
        } else {
            print!("\tunrecognized: {}", arg);
        }
    }
    println!("Waiting for scene to load");
    let _ = load_finish.run().join();

    println!("Flushing GPU work");
    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "init",
        buffer_count: 1,
    });
    command_encoder.start();
    let mut dummy = blade_render::DummyResources::new(&mut command_encoder, &context);
    let mut temp_buffers = Vec::new();
    asset_hub.flush(&mut command_encoder, &mut temp_buffers);

    let mut env_map = blade_render::EnvironmentMap::new(
        asset_hub.shaders[shader_init_handle].raw.as_ref().unwrap(),
        &dummy,
        &context,
    );
    let env_size = match environment_map {
        Some(handle) => {
            let texture = &asset_hub.textures[handle];
            env_map.assign(texture.view, texture.extent, &mut command_encoder, &context);
            texture.extent
        }
        None => dummy.size,
    };
    let env_sampler = EnvMapSampler::new(
        env_size,
        asset_hub.shaders[shader_main_handle].raw.as_ref().unwrap(),
        &context,
    );
    env_sampler.accumulate(&mut command_encoder, env_map.main_view, env_map.weight_view);
    let sync_point = context.submit(&mut command_encoder);

    context.wait_for(&sync_point, !0);
    context.destroy_command_encoder(&mut command_encoder);
    for buffer in temp_buffers {
        context.destroy_buffer(buffer);
    }
    env_map.destroy(&context);
    env_sampler.destroy(&context);
    dummy.destroy(&context);
    asset_hub.destroy();

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    if let Ok(ref mut rd) = rd {
        rd.end_frame_capture(ptr::null(), ptr::null());
        // RenderDoc doesn't like when the app suddenly closes at the end.
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}
