#![allow(irrefutable_let_patterns)]

use std::{env, fs, path::Path, ptr, sync::Arc};

use blade_graphics as gpu;

#[derive(blade_macros::ShaderData)]
struct EnvSampleData {
    env_weights: gpu::TextureView,
    sampler_nearest: gpu::Sampler,
}

struct EnvMapSampler {
    sample_count: u32,
    accum_texture: gpu::Texture,
    accum_view: gpu::TextureView,
    sampler: gpu::Sampler,
    pipeline: gpu::RenderPipeline,
}

impl EnvMapSampler {
    fn new(size: gpu::Extent, context: &gpu::Context) -> Self {
        let accum_format = gpu::TextureFormat::Rg32Float;
        let accum_texture = context.create_texture(gpu::TextureDesc {
            name: "sample-accum",
            format: accum_format,
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET,
        });
        let accum_view = context.create_texture_view(gpu::TextureViewDesc {
            texture: accum_texture,
            name: "sample-accum",
            format: accum_format,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        });
        let sampler = context.create_sampler(gpu::SamplerDesc {
            name: "nearest",
            address_modes: [gpu::AddressMode::ClampToEdge; 3],
            min_filter: gpu::FilterMode::Nearest,
            mag_filter: gpu::FilterMode::Nearest,
            mipmap_filter: gpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: None,
            compare: None,
            anisotropy_clamp: 0,
            border_color: None,
        });

        let source = fs::read_to_string("examples/init/env-sample.wgsl").unwrap();
        let shader = context.create_shader(gpu::ShaderDesc { source: &source });
        let layout = <EnvSampleData as gpu::ShaderData>::layout();

        let pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "env-sample",
            data_layouts: &[&layout],
            vertex: shader.at("vs_accum"),
            fragment: shader.at("fs_accum"),
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            color_targets: &[gpu::ColorTargetState {
                format: accum_format,
                blend: Some(gpu::BlendState::ADDITIVE),
                write_mask: gpu::ColorWrites::all(),
            }],
        });

        Self {
            sample_count: 1_000_000,
            accum_texture,
            accum_view,
            sampler,
            pipeline,
        }
    }

    fn accumulate(&self, command_encoder: &mut gpu::CommandEncoder, env_weights: gpu::TextureView) {
        command_encoder.init_texture(self.accum_texture);
        let mut pass = command_encoder.render(gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: self.accum_view,
                init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        });
        let mut encoder = pass.with(&self.pipeline);
        encoder.bind(
            0,
            &EnvSampleData {
                env_weights,
                sampler_nearest: self.sampler,
            },
        );
        encoder.draw(0, self.sample_count, 0, 1);
    }

    fn destroy(self, context: &gpu::Context) {
        context.destroy_texture_view(self.accum_view);
        context.destroy_texture(self.accum_texture);
        context.destroy_sampler(self.sampler);
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
            validation: cfg!(debug_assertions),
            capture: true,
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

    let mut asset_hub =
        blade_render::AssetHub::new(".".as_ref(), Path::new("asset-cache"), &choir, &context);

    let mut scene = blade_render::Scene::default();
    let mut env_map = blade_render::EnvironmentMap {
        main_view: blade_graphics::TextureView::default(),
        size: blade_graphics::Extent::default(),
        weight_texture: blade_graphics::Texture::default(),
        weight_view: blade_graphics::TextureView::default(),
        weight_mips: Vec::new(),
        preproc_pipeline: blade_render::EnvironmentMap::init_pipeline(&context).unwrap(),
    };
    println!("Populating the scene");
    let mut load_finish = choir.spawn("load finish").init_dummy();
    for arg in env::args().skip(1) {
        if arg.ends_with(".exr") {
            println!("\tenvironment map = {}", arg);
            let meta = blade_render::texture::Meta {
                format: blade_graphics::TextureFormat::Rgba32Float,
            };
            let (texture, texture_task) = asset_hub.textures.load(arg.as_ref(), meta);
            load_finish.depend_on(texture_task);
            scene.environment_map = Some(texture);
        } else if arg.ends_with(".gltf") {
            println!("\tmodels += {}", arg);
            let (model, model_task) = asset_hub
                .models
                .load(arg.as_ref(), blade_render::model::Meta);
            load_finish.depend_on(model_task);
            scene.objects.push(model.into());
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
    let mut temp_buffers = Vec::new();
    asset_hub.flush(&mut command_encoder, &mut temp_buffers);

    let mut env_sampler = None;
    if let Some(handle) = scene.environment_map {
        let texture = &asset_hub.textures[handle];
        env_map.assign(texture.view, texture.extent, &mut command_encoder, &context);
        let es = EnvMapSampler::new(texture.extent, &context);
        es.accumulate(&mut command_encoder, env_map.weight_view);
        env_sampler = Some(es);
    }
    let sync_point = context.submit(&mut command_encoder);

    context.wait_for(&sync_point, !0);
    for buffer in temp_buffers {
        context.destroy_buffer(buffer);
    }
    if scene.environment_map.is_some() {
        env_map.destroy(&context);
    }
    if let Some(env_sampler) = env_sampler {
        env_sampler.destroy(&context);
    }
    asset_hub.destroy();

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    if let Ok(ref mut rd) = rd {
        rd.end_frame_capture(ptr::null(), ptr::null());
        // RenderDoc doesn't like when the app suddenly closes at the end.
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}
