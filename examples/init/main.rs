use std::{env, path::Path, sync::Arc};

use blade_graphics as gpu;

const NUM_WORKERS: usize = 2;

fn main() {
    println!("Initializing");
    let context = Arc::new(unsafe {
        gpu::Context::init(gpu::ContextDesc {
            validation: cfg!(debug_assertions),
            capture: true,
        })
        .unwrap()
    });

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
    if let Some(handle) = scene.environment_map {
        let texture = &asset_hub.textures[handle];
        env_map.assign(texture.view, texture.extent, &mut command_encoder, &context);
    }
    let sync_point = context.submit(&mut command_encoder);

    context.wait_for(&sync_point, !0);
    for buffer in temp_buffers {
        context.destroy_buffer(buffer);
    }
    if scene.environment_map.is_some() {
        env_map.destroy(&context);
    }
    asset_hub.destroy();
}
