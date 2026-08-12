#![allow(irrefutable_let_patterns)]

// nanorand uses RtlGenRandom but doesn't link advapi32 itself.
// On Vulkan builds, other deps pull it in transitively; on GLES builds we must link it explicitly.
#[cfg(all(gles, windows))]
#[link(name = "advapi32")]
unsafe extern "C" {}

use blade_graphics as gpu;
use blade_graphics::ShaderData;
use std::slice;

#[allow(dead_code)]
#[path = "../examples/bunnymark/example.rs"]
mod bunnymark_example;
#[cfg(not(gles))]
mod pbr_scene;
#[cfg(not(gles))]
#[path = "../examples/ray-query/example.rs"]
mod ray_query_example;
mod snapshot;

/// Directory with the renderer shaders, needed by the asset hub.
#[cfg(not(gles))]
const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/blade-render/code");

// --- Sky snapshot test structs ---

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct SkyFrameParams {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    ambient_color: [f32; 4],
    settings: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct SkyTestData {
    sky_params: SkyFrameParams,
    samp: gpu::Sampler,
    env_map: gpu::TextureView,
}

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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct EnvSampleParams {
    mip_count: u32,
}

#[derive(blade_macros::ShaderData)]
struct EnvSampleData {
    env_main: gpu::TextureView,
    env_weights: gpu::TextureView,
    params: EnvSampleParams,
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
        mip_count: u32,
    ) {
        let params = EnvSampleParams { mip_count };
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
                    params,
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
                    params,
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

fn run_dispatch_gpu_test(manual_barriers: bool) {
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
    context.sync_buffer(input, gpu::BufferTarget::Data);

    let shader = context.create_shader(gpu::ShaderDesc {
        source: include_str!("shaders/dispatch.wgsl"),
        naga_module: None,
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
        manual_barriers,
    });
    command_encoder.start();
    if let mut compute = command_encoder.compute("dispatch")
        && let mut pass = compute.with(&pipeline)
    {
        pass.bind(
            0,
            &DispatchGlobals {
                input: input.into(),
                output: output.into(),
            },
        );
        pass.dispatch([1, 1, 1]);
    }
    if manual_barriers {
        command_encoder.barrier();
        if let mut compute = command_encoder.compute("dispatch-again")
            && let mut pass = compute.with(&pipeline)
        {
            pass.bind(
                0,
                &DispatchGlobals {
                    input: output.into(),
                    output: input.into(),
                },
            );
            pass.dispatch([1, 1, 1]);
        }
    }

    let sync_point = context.submit(&mut command_encoder);
    assert!(context.wait_for(&sync_point, 2000).unwrap());

    let actual_buffer = if manual_barriers { input } else { output };
    let actual = unsafe { slice::from_raw_parts(actual_buffer.data() as *const u32, 4) };
    let expected = if manual_barriers {
        [7, 11, 15, 19]
    } else {
        [3, 5, 7, 9]
    };
    assert_eq!(actual, expected);

    context.destroy_command_encoder(&mut command_encoder);
    context.destroy_compute_pipeline(&mut pipeline);
    context.destroy_buffer(output);
    context.destroy_buffer(input);
}

#[test]
#[ignore = "requires a working GPU context"]
fn dispatch_gpu_test() {
    run_dispatch_gpu_test(false);
}

#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context"]
fn manual_barrier_gpu_test() {
    run_dispatch_gpu_test(true);
}

/// Big enough for the weight chain to have several levels, so a mistake in the
/// mip math has somewhere to show up.
const ENV_TEST_SIZE: gpu::Extent = gpu::Extent {
    width: 64,
    height: 32,
    depth: 1,
};

/// An equirectangular map with a bright spot in it.
///
/// Uniform content would make the importance sampling degenerate, and the
/// point of the weight chain is to concentrate on where the light actually is.
/// Returns the staging buffer as well, for the caller to destroy after submit.
fn make_environment(
    encoder: &mut gpu::CommandEncoder,
    context: &gpu::Context,
) -> (gpu::Texture, gpu::TextureView, gpu::Buffer) {
    let format = gpu::TextureFormat::Rgba8Unorm;
    let texel_count = (ENV_TEST_SIZE.width * ENV_TEST_SIZE.height) as usize;
    let texture = context.create_texture(gpu::TextureDesc {
        name: "env-test",
        format,
        size: ENV_TEST_SIZE,
        dimension: gpu::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
        sample_count: 1,
        external: None,
    });
    let view = context.create_texture_view(
        texture,
        gpu::TextureViewDesc {
            name: "env-test",
            format,
            dimension: gpu::ViewDimension::D2,
            subresources: &Default::default(),
        },
    );

    let mut texels = vec![[16u8, 16, 24, 255]; texel_count];
    for y in 8..12 {
        for x in 40..46 {
            texels[y * ENV_TEST_SIZE.width as usize + x] = [255, 250, 235, 255];
        }
    }
    let stage = context.create_buffer(gpu::BufferDesc {
        name: "env-test/stage",
        size: (texel_count * 4) as u64,
        memory: gpu::Memory::Upload,
    });
    unsafe {
        std::ptr::copy_nonoverlapping(texels.as_ptr() as *const u8, stage.data(), texel_count * 4);
    }

    encoder.init_texture(texture);
    if let mut transfer = encoder.transfer("env-test-upload") {
        transfer.copy_buffer_to_texture(
            stage.at(0),
            ENV_TEST_SIZE.width * 4,
            texture.into(),
            ENV_TEST_SIZE,
        );
    }
    (texture, view, stage)
}

#[test]
#[ignore = "requires a working GPU context"]
fn env_map_gpu_test() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };

    let shader_prepare = context.create_shader(gpu::ShaderDesc {
        source: include_str!("../blade-render/code/env-prepare.wgsl"),
        naga_module: None,
    });
    let shader_sample = context.create_shader(gpu::ShaderDesc {
        source: include_str!("shaders/env_map_sample.wgsl"),
        naga_module: None,
    });

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "env-map-test",
        buffer_count: 1,
        manual_barriers: false,
    });
    command_encoder.start();

    let mut dummy = blade_render::DummyResources::new(&mut command_encoder, &context);
    let mut env_map = blade_render::EnvironmentMap::new(&shader_prepare, &dummy, &context);

    // A map of its own rather than `dummy.white_view`: `EnvironmentMap::new`
    // starts out holding that view, so assigning it again returns immediately
    // and the weight chain this test goes on to sample is never built.
    let (env_texture, env_view, env_stage) = make_environment(&mut command_encoder, &context);
    env_map.assign(env_view, ENV_TEST_SIZE, &mut command_encoder, &context);
    assert!(
        !env_map.weight_mips.is_empty(),
        "assigning a map has to build the importance sampling chain"
    );

    let env_sampler = EnvMapSampler::new(ENV_TEST_SIZE, &shader_sample, &context);
    env_sampler.accumulate(
        &mut command_encoder,
        env_map.main_view,
        env_map.weight_view,
        env_map.weight_mips.len() as u32,
    );

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
    assert!(context.wait_for(&sync_point, 2000).unwrap());

    let actual = unsafe { slice::from_raw_parts(readback.data(), 8) };
    assert!(
        actual.iter().any(|b| *b != 0),
        "environment map output is entirely zero"
    );

    context.destroy_buffer(readback);
    context.destroy_buffer(env_stage);
    env_map.destroy(&context);
    context.destroy_texture_view(env_view);
    context.destroy_texture(env_texture);
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
        manual_barriers: false,
    });
    command_encoder.start();
    command_encoder.init_texture(target.texture);
    example.render(&mut command_encoder, target.view);

    let pixels = target.read_pixels(&context, &mut command_encoder);
    snapshot::check("bunnymark", &pixels, size);

    example.deinit(&context);
    context.destroy_command_encoder(&mut command_encoder);
    target.destroy(&context);
}

#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn snapshot_ray_query() {
    // Metal acceleration structure APIs can throw uncatchable ObjC exceptions
    // in CI environments, even when the device reports ray tracing support.
    if cfg!(target_os = "macos") {
        println!("Skipping: ray tracing snapshot not supported on macOS CI");
        return;
    }

    let context = unsafe {
        match gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            ..Default::default()
        }) {
            Ok(c) => c,
            Err(e) => {
                println!("Skipping: GPU context with ray tracing not available: {e:?}");
                return;
            }
        }
    };
    let capabilities = context.capabilities();
    if !capabilities
        .ray_query
        .contains(gpu::ShaderVisibility::COMPUTE)
    {
        println!("Skipping: ray_query compute not supported");
        return;
    }

    let size = gpu::Extent {
        width: 400,
        height: 300,
        depth: 1,
    };
    let format = gpu::TextureFormat::Rgba8Unorm;

    let target = snapshot::OffscreenTarget::new(&context, size, format);
    let mut example = ray_query_example::Example::new(&context, size, format);

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "snapshot-ray-query",
        buffer_count: 1,
        manual_barriers: false,
    });
    command_encoder.start();
    command_encoder.init_texture(target.texture);
    // Fixed rotation angle for deterministic output
    example.render(&mut command_encoder, target.view, 1.0);

    let pixels = target.read_pixels(&context, &mut command_encoder);
    snapshot::check("ray-query", &pixels, size);

    example.deinit(&context);
    context.destroy_command_encoder(&mut command_encoder);
    target.destroy(&context);
}

#[test]
#[ignore = "requires a working GPU context"]
fn snapshot_particle() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    let size = gpu::Extent {
        width: 400,
        height: 300,
        depth: 1,
    };
    let format = gpu::TextureFormat::Rgba8Unorm;

    let target = snapshot::OffscreenTarget::new(&context, size, format);
    let mut pipeline = blade_particle::ParticlePipeline::new(
        &context,
        blade_particle::PipelineDesc {
            name: "snapshot particle",
            draw_format: format,
            depth_format: None,
            sample_count: 1,
        },
    );
    let effect = blade_particle::ParticleEffect {
        capacity: 10_000,
        emitter: blade_particle::Emitter {
            rate: 6400.0,
            burst_count: 0,
            shape: blade_particle::EmitterShape::Point,
            cone_angle: std::f32::consts::PI,
        },
        particle: blade_particle::ParticleConfig {
            life: [1.0, 5.0],
            speed: [50.0, 250.0],
            scale: [1.0, 15.0],
            color: blade_particle::ColorConfig::Solid([255, 255, 255, 255]),
        },
    };
    let mut particle_system = pipeline.create_system(&context, "snapshot particle", &effect);

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "snapshot-particle",
        buffer_count: 1,
        manual_barriers: false,
    });
    command_encoder.start();
    // Run several update cycles to emit and move particles
    for _ in 0..20 {
        particle_system.update(&pipeline, &mut command_encoder, 0.01);
    }

    let camera = {
        let distance = 1000.0_f32;
        let fov_y = 2.0 * (500.0_f32 / distance).atan();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let near = 0.01_f32;
        let far = distance * 2.0;
        let pos = glam::Vec3::new(0.0, 0.0, distance);
        let view = glam::Mat4::look_at_rh(pos, glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(fov_y, aspect, near, far);
        let view_proj = proj * view;
        blade_particle::CameraParams {
            view_proj: view_proj.to_cols_array(),
            camera_right: [1.0, 0.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0, 0.0],
        }
    };

    command_encoder.init_texture(target.texture);
    if let mut pass = command_encoder.render(
        "draw particles",
        gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: target.view,
                init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        },
    ) {
        particle_system.draw(&pipeline, &mut pass, &camera);
    }

    let pixels = target.read_pixels(&context, &mut command_encoder);
    snapshot::check("particle", &pixels, size);

    particle_system.destroy(&context);
    pipeline.destroy(&context);
    context.destroy_command_encoder(&mut command_encoder);
    target.destroy(&context);
}

#[test]
#[ignore = "requires a working GPU context"]
fn snapshot_space_sky() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    let size = gpu::Extent {
        width: 400,
        height: 300,
        depth: 1,
    };
    // A plain format, like an XR swapchain: the shader has to encode
    let format = gpu::TextureFormat::Rgba8Unorm;

    // Create offscreen target
    let target = snapshot::OffscreenTarget::new(&context, size, format);

    // Create a dummy 1x1 black texture for the env_map binding
    let dummy_tex = context.create_texture(gpu::TextureDesc {
        name: "sky-test-dummy",
        format: gpu::TextureFormat::Rgba8Unorm,
        size: gpu::Extent {
            width: 1,
            height: 1,
            depth: 1,
        },
        array_layer_count: 1,
        mip_level_count: 1,
        dimension: gpu::TextureDimension::D2,
        usage: gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
        sample_count: 1,
        external: None,
    });
    let dummy_view = context.create_texture_view(
        dummy_tex,
        gpu::TextureViewDesc {
            name: "sky-test-dummy",
            format: gpu::TextureFormat::Rgba8Unorm,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        },
    );
    let sampler = context.create_sampler(gpu::SamplerDesc {
        name: "sky-test",
        address_modes: [gpu::AddressMode::Repeat; 3],
        mag_filter: gpu::FilterMode::Linear,
        min_filter: gpu::FilterMode::Linear,
        mipmap_filter: gpu::FilterMode::Linear,
        ..Default::default()
    });

    // Compile the raster shader and create sky pipeline (no depth attachment)
    let source = snapshot::shader_source("raster.wgsl");
    let shader = context.create_shader(gpu::ShaderDesc {
        source: &source,
        naga_module: None,
    });
    let sky_layout = <SkyTestData as gpu::ShaderData>::layout();
    let mut sky_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
        name: "sky-test",
        data_layouts: &[&sky_layout],
        vertex: shader.at("raster_sky_vs"),
        vertex_fetches: &[],
        primitive: gpu::PrimitiveState {
            topology: gpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        fragment: Some(shader.at("raster_sky_fs")),
        color_targets: &[format.into()],
        multisample_state: gpu::MultisampleState::default(),
    });

    // Build camera: look along +Z from origin
    let aspect = size.width as f32 / size.height as f32;
    let fov_y: f32 = 1.0; // ~57 degrees
    let near = 0.01f32;
    let far = 100.0f32;
    let proj = glam::Mat4::perspective_rh(fov_y, aspect, near, far);
    let view = glam::Mat4::IDENTITY; // camera at origin, looking along -Z in RH
    let view_proj = proj * view;
    let inv_view_proj = view_proj.inverse();

    let frame_params = SkyFrameParams {
        view_proj: view_proj.to_cols_array(),
        inv_view_proj: inv_view_proj.to_cols_array(),
        camera_pos: [0.0, 0.0, 0.0, 1.0],
        light_dir: [0.0, -1.0, 0.0, 0.0],
        light_color: [1.0, 1.0, 1.0, 0.0],
        ambient_color: [0.0, 0.0, 0.0, 1.0], // w=1.0 -> space_sky mode
        // x=0: no environment map, y=1: encode for a non-sRGB surface
        settings: [0.0, 1.0, 0.0, 0.0],
    };

    // Render
    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "sky-test",
        buffer_count: 1,
        manual_barriers: false,
    });
    command_encoder.start();
    command_encoder.init_texture(target.texture);
    command_encoder.init_texture(dummy_tex);

    if let mut pass = command_encoder.render(
        "sky-test",
        gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: target.view,
                init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        },
    ) && let mut pc = pass.with(&sky_pipeline)
    {
        pc.bind(
            0,
            &SkyTestData {
                sky_params: frame_params,
                samp: sampler,
                env_map: dummy_view,
            },
        );
        pc.draw(0, 3, 0, 1);
    }

    let pixels = target.read_pixels(&context, &mut command_encoder);

    // Check that we have non-black pixels (stars/dots should be visible)
    let non_black_count = pixels
        .chunks(4)
        .filter(|px| px[0] > 10 || px[1] > 10 || px[2] > 10)
        .count();
    let total_pixels = (size.width * size.height) as usize;
    println!(
        "space_sky: {non_black_count}/{total_pixels} non-black pixels ({:.1}%)",
        non_black_count as f64 / total_pixels as f64 * 100.0
    );

    // Save the image for visual inspection
    snapshot::check("space-sky", &pixels, size);

    // Cleanup
    context.destroy_render_pipeline(&mut sky_pipeline);
    context.destroy_sampler(sampler);
    context.destroy_texture_view(dummy_view);
    context.destroy_texture(dummy_tex);
    context.destroy_command_encoder(&mut command_encoder);
    target.destroy(&context);
}

/// Number of accumulated frames for the ray traced snapshot.
///
/// ReSTIR needs a bit of history to converge, but the cost is paid
/// by the software rasterizers used in CI.
#[cfg(not(gles))]
const RAY_TRACE_FRAMES: usize = 8;

/// Frames accumulated by the canonical renderer.
///
/// A uniform environment converges quickly, and the cost is paid
/// by the software rasterizers used in CI.
#[cfg(not(gles))]
const CANONICAL_FRAMES: usize = 32;
/// How far the real-time result may land from the canonical one,
/// as a mean absolute difference of the 8-bit channels.
#[cfg(not(gles))]
const CANONICAL_MAX_DIFFERENCE: f64 = 12.0;

#[cfg(not(gles))]
struct PbrHarness {
    context: std::sync::Arc<gpu::Context>,
    choir: std::sync::Arc<choir::Choir>,
    workers: Vec<choir::WorkerHandle>,
    asset_hub: blade_render::AssetHub,
    shaders: blade_render::Shaders,
}

#[cfg(not(gles))]
impl PbrHarness {
    /// Bring up the asset hub and cook the renderer shaders.
    fn new(context: gpu::Context, cache_name: &str, ray_tracing: bool) -> Self {
        let context = std::sync::Arc::new(context);
        let choir = choir::Choir::new();
        let workers = (0..2)
            .map(|i| choir.add_worker(&format!("{cache_name}-{i}")))
            .collect();
        let cache_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-assets")
            .join(cache_name);
        let asset_hub = blade_render::AssetHub::new(&cache_path, &choir, &context);
        let (shaders, shader_task) =
            blade_render::Shaders::load(SHADER_DIR.as_ref(), &asset_hub, ray_tracing);
        shader_task.join();
        Self {
            context,
            choir,
            workers,
            asset_hub,
            shaders,
        }
    }

    fn create_grid_model(
        &self,
        roughness_range: [f32; 2],
    ) -> blade_asset::Handle<blade_render::Model> {
        let geometries = pbr_scene::material_grid(roughness_range);
        let model = self
            .asset_hub
            .models
            .baker
            .create_model("pbr-material-grid", geometries);
        self.asset_hub.models.insert(model)
    }

    fn destroy(mut self) {
        self.asset_hub.destroy();
        // let the workers finish before the choir goes away
        self.workers.clear();
        drop(self.choir);
    }
}

/// Rasterize a grid of spheres covering the metallic-roughness space,
/// plus a row of emissive materials.
#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context"]
fn snapshot_pbr_raster() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    let size = gpu::Extent {
        width: 400,
        height: 300,
        depth: 1,
    };
    // The renderers write linear values, so let the hardware encode them
    let format = gpu::TextureFormat::Rgba8UnormSrgb;

    let harness = PbrHarness::new(context, "pbr-raster", false);
    let context = std::sync::Arc::clone(&harness.context);
    let target = snapshot::OffscreenTarget::new(&context, size, format);

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "snapshot-pbr-raster",
        buffer_count: 1,
        manual_barriers: false,
    });
    command_encoder.start();

    let mut rasterizer = blade_render::Rasterizer::new(
        &mut command_encoder,
        &context,
        harness.shaders.clone(),
        &harness.asset_hub.shaders,
        &blade_render::RenderConfig {
            surface_size: size,
            surface_info: gpu::SurfaceInfo {
                format,
                alpha: gpu::AlphaMode::Ignored,
            },
            // matching an sRGB surface: the hardware does the encoding
            color_space: gpu::ColorSpace::Linear,
            max_debug_lines: 16,
        },
    );

    let objects = vec![blade_render::Object::from(
        harness.create_grid_model([0.05, 1.0]),
    )];
    let mut temp_buffers = Vec::new();
    harness
        .asset_hub
        .flush(&mut command_encoder, &mut temp_buffers);

    command_encoder.init_texture(target.texture);
    command_encoder.init_texture(rasterizer.depth_texture());
    if let mut pass = command_encoder.render(
        "raster-pbr",
        gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: target.view,
                init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: Some(gpu::RenderTarget {
                view: rasterizer.depth_view(),
                init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                finish_op: gpu::FinishOp::Discard,
            }),
        },
    ) {
        rasterizer.render(
            &mut pass,
            &pbr_scene::camera(),
            &objects,
            &harness.asset_hub,
            None,
            blade_render::RasterConfig {
                light_dir: mint::Vector3 {
                    x: 0.4,
                    y: 0.5,
                    z: 1.0,
                },
                ..Default::default()
            },
        );
    }

    let pixels = target.read_pixels(&context, &mut command_encoder);
    snapshot::check("pbr-raster", &pixels, size);

    for buffer in temp_buffers {
        context.destroy_buffer(buffer);
    }
    rasterizer.destroy(&context);
    context.destroy_command_encoder(&mut command_encoder);
    target.destroy(&context);
    harness.destroy();
}

/// The material grid, lit by a uniform white environment.
///
/// This is a white furnace test: an energy conserving BRDF keeps the spheres
/// close to their base color, with the darkening coming from the roughness
/// and from the occlusion between the neighbors.
#[cfg(not(gles))]
const RAY_TRACE_SIZE: gpu::Extent = gpu::Extent {
    width: 256,
    height: 192,
    depth: 1,
};
/// The post processing writes linear values, so let the hardware encode them.
#[cfg(not(gles))]
const RAY_TRACE_FORMAT: gpu::TextureFormat = gpu::TextureFormat::Rgba8UnormSrgb;

#[cfg(not(gles))]
enum RayTraceMode {
    /// The real-time path: ReSTIR with a denoiser.
    Restir,
    /// The canonical path: accumulated brute force paths.
    Canonical,
}

/// What the post processing should hand back.
#[cfg(not(gles))]
#[derive(Clone, Copy, PartialEq)]
enum Capture {
    /// Tone mapped and encoded for a display, in `Rgba8UnormSrgb`.
    Display,
    /// The composed linear radiance, untouched, in `Rgba32Float`.
    Hdr,
}

#[cfg(not(gles))]
impl Capture {
    fn format(self) -> gpu::TextureFormat {
        match self {
            Self::Display => RAY_TRACE_FORMAT,
            Self::Hdr => gpu::TextureFormat::Rgba32Float,
        }
    }
}

/// Render the material grid with the ray tracer, if the GPU can do it.
#[cfg(not(gles))]
fn render_ray_traced_grid(cache_name: &str, mode: RayTraceMode) -> Option<Vec<u8>> {
    render_ray_traced_grid_as(cache_name, mode, Capture::Display, |_, _, _| {})
}

/// Render the grid, letting the caller look at the renderer's own state after
/// the frame is rendered and before it is torn down.
///
/// `inspect` is where a test reaches for something the post processing does not
/// expose, such as the G-buffer.
#[cfg(not(gles))]
fn render_ray_traced_grid_as(
    cache_name: &str,
    mode: RayTraceMode,
    capture: Capture,
    inspect: impl FnOnce(&blade_render::RayTracer, &gpu::Context, &mut gpu::CommandEncoder),
) -> Option<Vec<u8>> {
    // Metal acceleration structure APIs can throw uncatchable ObjC exceptions
    // in CI environments, even when the device reports ray tracing support.
    if cfg!(target_os = "macos") {
        println!("Skipping: ray tracing snapshot not supported on macOS CI");
        return None;
    }

    let context = unsafe {
        match gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            ..Default::default()
        }) {
            Ok(c) => c,
            Err(e) => {
                println!("Skipping: GPU context with ray tracing not available: {e:?}");
                return None;
            }
        }
    };
    if !context
        .capabilities()
        .ray_query
        .contains(gpu::ShaderVisibility::COMPUTE)
    {
        println!("Skipping: ray_query compute not supported");
        return None;
    }

    let size = RAY_TRACE_SIZE;
    let harness = PbrHarness::new(context, cache_name, true);
    let context = std::sync::Arc::clone(&harness.context);
    let target = snapshot::OffscreenTarget::new(&context, size, capture.format());

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "snapshot-ray-traced-grid",
        buffer_count: 1,
        manual_barriers: false,
    });
    command_encoder.start();

    let mut renderer = blade_render::RayTracer::new(
        &mut command_encoder,
        &context,
        harness.shaders.clone(),
        &harness.asset_hub.shaders,
        &blade_render::RenderConfig {
            surface_size: size,
            surface_info: gpu::SurfaceInfo {
                format: capture.format(),
                alpha: gpu::AlphaMode::Ignored,
            },
            color_space: gpu::ColorSpace::Linear,
            max_debug_lines: 16,
        },
    );

    // A narrow specular lobe is hard on the real-time estimator,
    // so the smoothest materials are left out of these.
    let objects = vec![blade_render::Object::from(
        harness.create_grid_model([0.3, 1.0]),
    )];
    let mut temp = blade_render::FrameResources::default();
    harness
        .asset_hub
        .flush(&mut command_encoder, &mut temp.buffers);

    let camera = pbr_scene::camera();
    let debug_config = blade_render::DebugConfig::default();
    let ray_config = blade_render::RayConfig {
        // The canonical mode takes these at every vertex of every path,
        // so it needs fewer of them to stay affordable.
        num_environment_samples: match mode {
            RayTraceMode::Restir => 4,
            RayTraceMode::Canonical => 1,
        },
        num_brdf_samples: 4,
        // the dummy environment map has no importance sampling data
        environment_importance_sampling: false,
        max_bounces: 3,
        max_accumulated_samples: 0,
        tap_count: 2,
        tap_radius: 16,
        tap_confidence_near: 8,
        tap_confidence_far: 4,
        t_start: 0.01,
        // Exercise the standard re-evaluated-target reuse path. Its previous
        // implementation could select a stale non-zero target that shaded to
        // black at the receiving surface, creating persistent dark patches.
        pairwise_mis: false,
        defensive_mis: 0.1,
    };
    let denoiser_config = blade_render::DenoiserConfig {
        num_passes: 3,
        temporal_weight: 0.1,
    };
    let frame_count = match mode {
        RayTraceMode::Restir => RAY_TRACE_FRAMES,
        RayTraceMode::Canonical => CANONICAL_FRAMES,
    };
    for frame_index in 0..frame_count {
        renderer.build_scene(
            &mut command_encoder,
            &objects,
            None,
            &harness.asset_hub,
            &context,
            &mut temp,
        );
        renderer.prepare(
            &mut command_encoder,
            &camera,
            blade_render::FrameConfig {
                frozen: false,
                debug_draw: false,
                reset_variance: frame_index == 0,
                reset_reservoirs: frame_index == 0,
                reset_accumulation: frame_index == 0,
            },
        );
        renderer.render(
            &mut command_encoder,
            match mode {
                RayTraceMode::Restir => blade_render::RenderMode::RealTime,
                RayTraceMode::Canonical => blade_render::RenderMode::Canonical,
            },
            debug_config,
            ray_config,
            Some(denoiser_config),
        );
    }

    inspect(&renderer, &context, &mut command_encoder);

    command_encoder.init_texture(target.texture);
    if let mut pass = command_encoder.render(
        "ray-traced-grid",
        gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: target.view,
                init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        },
    ) {
        renderer.post_proc(
            &mut pass,
            debug_config,
            blade_render::PostProcConfig {
                tone_map: capture == Capture::Display,
                ..Default::default()
            },
            &[],
            &[],
        );
    }

    let pixels = target.read_pixels(&context, &mut command_encoder);

    for buffer in temp.buffers {
        context.destroy_buffer(buffer);
    }
    for acceleration_structure in temp.acceleration_structures {
        context.destroy_acceleration_structure(acceleration_structure);
    }
    renderer.destroy(&context);
    context.destroy_command_encoder(&mut command_encoder);
    target.destroy(&context);
    harness.destroy();
    Some(pixels)
}

#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn snapshot_pbr_ray_trace() {
    if let Some(pixels) = render_ray_traced_grid("pbr-ray-trace", RayTraceMode::Restir) {
        snapshot::check_at("pbr-ray-trace", &pixels, RAY_TRACE_SIZE, 0.99);
    }
}

/// Render the same grid with the canonical renderer, and confirm that the
/// real-time one lands in the same place.
#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn snapshot_pbr_canonical() {
    let Some(pixels) = render_ray_traced_grid("pbr-canonical", RayTraceMode::Canonical) else {
        return;
    };
    snapshot::check("pbr-canonical", &pixels, RAY_TRACE_SIZE);

    // The real-time estimator is noisy and blurred, but it must not be
    // systematically brighter or darker than the ground truth.
    let (restir, restir_size) = snapshot::load("pbr-ray-trace");
    assert_eq!(restir_size, RAY_TRACE_SIZE);
    let difference = snapshot::mean_abs_diff(&pixels, &restir);
    println!("pbr-canonical: mean difference from ReSTIR = {difference:.2}/255");
    assert!(
        difference < CANONICAL_MAX_DIFFERENCE,
        "the real-time result is {difference:.2}/255 away from the canonical one"
    );
}

/// Clearing `PostProcConfig::tone_map` has to hand back the composed radiance
/// as it is, with nothing compressed into display range.
///
/// The grid's emissive row is far brighter than white, so a path that clamps
/// or tone maps cannot produce these values. This is what lets a capture be
/// used as data rather than only as a picture.
#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn hdr_capture_is_unclipped() {
    let Some(bytes) = render_ray_traced_grid_as(
        "pbr-hdr",
        RayTraceMode::Canonical,
        Capture::Hdr,
        |_, _, _| {},
    ) else {
        return;
    };
    let pixels: &[f32] = bytemuck::cast_slice(&bytes);
    assert_eq!(
        pixels.len(),
        (RAY_TRACE_SIZE.width * RAY_TRACE_SIZE.height * 4) as usize
    );

    let mut peak = 0.0f32;
    for texel in pixels.chunks_exact(4) {
        for &channel in &texel[..3] {
            assert!(channel.is_finite(), "non-finite radiance {channel}");
            assert!(channel >= 0.0, "negative radiance {channel}");
            peak = peak.max(channel);
        }
    }
    println!("hdr capture: peak radiance = {peak:.2}");
    assert!(
        peak > 1.0,
        "peak radiance is {peak:.3}, so the capture was clamped into display range"
    );

    // Stronger than a range check: applying the same curve on the CPU has to
    // reproduce the display capture. That pins the HDR path to being the same
    // signal one stage earlier, rather than some other buffer that merely
    // happens to be bright.
    let Some(display) = render_ray_traced_grid_as(
        "pbr-hdr-display",
        RayTraceMode::Canonical,
        Capture::Display,
        |_, _, _| {},
    ) else {
        return;
    };

    // The same curve the shader applies, at PostProcConfig::default(): unit
    // exposure and a white level of 1. Note that a white level of 1 makes the
    // curve an identity, so what separates the two captures here is the
    // transfer encoding and the 8-bit clamp.
    let tone_map = |l: f32| {
        let white = 1.0f32;
        l * (1.0 + l / (white * white)) / (1.0 + l)
    };
    let encode_srgb = |v: f32| {
        if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };

    let mut worst = 0.0f32;
    for (hdr, shown) in pixels.chunks_exact(4).zip(display.chunks_exact(4)) {
        for (&linear, &byte) in hdr[..3].iter().zip(shown[..3].iter()) {
            let expected = encode_srgb(tone_map(linear)).clamp(0.0, 1.0) * 255.0;
            worst = worst.max((expected - byte as f32).abs());
        }
    }
    println!("hdr capture: worst channel difference after tone mapping = {worst:.2}/255");
    // Both passes are independent accumulations of a stochastic estimator, so
    // they differ by sampling noise rather than being bit-identical.
    assert!(
        worst < 8.0,
        "the tone mapped HDR capture is {worst:.2}/255 away from the display one"
    );
}

#[cfg(not(gles))]
#[derive(blade_macros::ShaderData)]
struct GBufferProbeData {
    t_depth: gpu::TextureView,
    t_basis: gpu::TextureView,
    t_flat_normal: gpu::TextureView,
    t_diffuse_albedo: gpu::TextureView,
    t_specular_f0: gpu::TextureView,
    output: gpu::TextureView,
}

/// The G-buffer views have to be bindable from outside the renderer, and the
/// explicit fill used after canonical path tracing has to describe the frame
/// that was just rendered.
///
/// The material grid is what makes the second half checkable: its columns
/// sweep the roughness and its rows sweep the metalness, so a buffer that came
/// from the wrong place, or from before the scene was drawn, will not show that
/// structure.
#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn gbuffer_views_describe_the_rendered_frame() {
    let size = RAY_TRACE_SIZE;
    let mut probe = Vec::new();

    let rendered = render_ray_traced_grid_as(
        "pbr-gbuffer",
        RayTraceMode::Canonical,
        Capture::Display,
        |renderer, context, encoder| {
            renderer.fill_gbuffer(encoder, blade_render::DebugConfig::default());
            let format = gpu::TextureFormat::Rgba32Float;
            let target = snapshot::OffscreenTarget::new(context, size, format);
            // The probe writes through a storage binding rather than as a
            // render target.
            let storage = context.create_texture(gpu::TextureDesc {
                name: "gbuffer-probe",
                format,
                size,
                dimension: gpu::TextureDimension::D2,
                array_layer_count: 1,
                mip_level_count: 1,
                usage: gpu::TextureUsage::STORAGE | gpu::TextureUsage::COPY,
                sample_count: 1,
                external: None,
            });
            let storage_view = context.create_texture_view(
                storage,
                gpu::TextureViewDesc {
                    name: "gbuffer-probe",
                    format,
                    dimension: gpu::ViewDimension::D2,
                    subresources: &gpu::TextureSubresources::default(),
                },
            );
            encoder.init_texture(storage);

            let shader = context.create_shader(gpu::ShaderDesc {
                source: include_str!("shaders/gbuffer_probe.wgsl"),
                naga_module: None,
            });
            let layout = <GBufferProbeData as gpu::ShaderData>::layout();
            let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
                name: "gbuffer-probe",
                data_layouts: &[&layout],
                compute: shader.at("probe"),
            });

            let views = renderer.view_gbuffer();
            {
                let mut pass = encoder.compute("gbuffer-probe");
                let mut commands = pass.with(&pipeline);
                commands.bind(
                    0,
                    &GBufferProbeData {
                        t_depth: views.depth,
                        t_basis: views.basis,
                        t_flat_normal: views.flat_normal,
                        t_diffuse_albedo: views.diffuse_albedo,
                        t_specular_f0: views.specular_f0,
                        output: storage_view,
                    },
                );
                commands.dispatch([size.width.div_ceil(8), size.height.div_ceil(8), 1]);
            }

            {
                let mut transfer = encoder.transfer("gbuffer-probe-readback");
                transfer.copy_texture_to_buffer(
                    storage.into(),
                    target.readback.into(),
                    size.width * 16,
                    size,
                );
            }
            let sync_point = context.submit(encoder);
            // Same budget as a snapshot readback: this waits on the same
            // ray-traced frame, so it faces the same software rasterizer.
            assert!(
                context
                    .wait_for(&sync_point, snapshot::READBACK_TIMEOUT_MS)
                    .unwrap(),
                "GPU timed out reading back the G-buffer probe"
            );

            let count = (size.width * size.height * 4) as usize;
            probe = vec![0.0f32; count];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    target.readback.data() as *const f32,
                    probe.as_mut_ptr(),
                    count,
                );
            }

            // The encoder has to be live again for the caller's post processing.
            encoder.start();
            context.destroy_compute_pipeline(&mut pipeline);
            context.destroy_texture_view(storage_view);
            context.destroy_texture(storage);
            target.destroy(context);
        },
    );
    if rendered.is_none() {
        return;
    }
    assert!(!probe.is_empty(), "the inspect hook did not run");

    // Depth: positive wherever a ray hit something, and the grid fills enough
    // of the frame that plenty of rays did.
    let hits = probe.chunks_exact(4).filter(|t| t[0] > 0.0).count();
    let total = (size.width * size.height) as usize;
    println!("gbuffer: {hits}/{total} pixels have depth");
    assert!(
        hits > total / 10,
        "only {hits} of {total} pixels carry depth, the buffer looks empty"
    );

    // Roughness: the grid sweeps it across the columns, so a G-buffer read
    // from the right place shows a spread rather than one value.
    let mut min_roughness = f32::INFINITY;
    let mut max_roughness = f32::NEG_INFINITY;
    for texel in probe.chunks_exact(4).filter(|t| t[0] > 0.0) {
        min_roughness = min_roughness.min(texel[1]);
        max_roughness = max_roughness.max(texel[1]);
    }
    println!("gbuffer: roughness spans {min_roughness:.2}..{max_roughness:.2}");
    assert!(
        max_roughness - min_roughness > 0.3,
        "roughness barely varies ({min_roughness}..{max_roughness}), \
         the grid sweeps it so this should be a wide range"
    );

    // Albedo is a unorm target, so it cannot leave [0, 1].
    assert!(
        probe.chunks_exact(4).all(|t| (0.0..=1.0).contains(&t[2])),
        "albedo left the unit range"
    );

    // The spheres carry no normal map, so the shading normal should agree with
    // the geometric one almost everywhere a ray hit.
    let aligned = probe
        .chunks_exact(4)
        .filter(|t| t[0] > 0.0 && t[3] > 0.9)
        .count();
    println!("gbuffer: {aligned}/{hits} hits have basis aligned with the flat normal");
    assert!(
        aligned * 10 > hits * 9,
        "the basis quaternion does not decode to the geometric normal, \
         so it is not the tangent frame it is documented to be"
    );
}

/// Cook and serve a glTF model, checking that the PBR factors survive the trip.
#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context"]
fn gltf_material_test() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    let harness = PbrHarness::new(context, "gltf-material", false);
    let context = std::sync::Arc::clone(&harness.context);

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("scene")
        .join("data")
        .join("monkey.gltf");
    let (handle, task) = harness.asset_hub.models.load(
        &path,
        blade_render::model::Meta {
            generate_tangents: true,
            front_face: blade_render::model::FrontFace::CounterClockwise,
        },
    );
    task.clone().join();

    // The uploads have to be flushed before the buffers can be destroyed.
    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "gltf-material",
        buffer_count: 1,
        manual_barriers: false,
    });
    command_encoder.start();
    let mut temp_buffers = Vec::new();
    harness
        .asset_hub
        .flush(&mut command_encoder, &mut temp_buffers);
    let sync_point = context.submit(&mut command_encoder);
    assert!(context.wait_for(&sync_point, 5000).unwrap());

    let model = &harness.asset_hub.models[handle];
    assert!(!model.geometries.is_empty());
    assert_eq!(model.materials.len(), 2);
    for material in model.materials.iter() {
        // Matching "pbrMetallicRoughness" of the source
        assert_eq!(material.metalness, 0.0);
        assert_eq!(material.roughness, 0.5);
        assert_eq!(material.base_color_factor[3], 1.0);
        assert!((material.base_color_factor[0] - 0.8).abs() < 0.01);
        // The model has no textures and doesn't emit light
        assert!(material.metallic_roughness_texture.is_none());
        assert!(material.emissive_texture.is_none());
        assert_eq!(material.emissive_factor, [0.0; 3]);
    }

    for buffer in temp_buffers {
        context.destroy_buffer(buffer);
    }
    context.destroy_command_encoder(&mut command_encoder);
    harness.destroy();
}

// --- Render pipeline state tests ---

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct QuadParams {
    color: [f32; 4],
    depth: f32,
    pad: [f32; 3],
}

#[derive(blade_macros::ShaderData)]
struct QuadData {
    quad_params: QuadParams,
}

#[derive(blade_macros::ShaderData)]
struct SplitData {
    left_texture: gpu::TextureView,
    right_texture: gpu::TextureView,
    samp: gpu::Sampler,
}

const STATE_TEST_SIZE: gpu::Extent = gpu::Extent {
    width: 16,
    height: 16,
    depth: 1,
};

/// Upload a 1x1 texture of the given color, for a binding that only has to be
/// told apart from another one.
fn make_solid_texture(
    name: &'static str,
    color: [u8; 4],
    encoder: &mut gpu::CommandEncoder,
    context: &gpu::Context,
) -> (gpu::Texture, gpu::TextureView, gpu::Buffer) {
    let format = gpu::TextureFormat::Rgba8Unorm;
    let size = gpu::Extent {
        width: 1,
        height: 1,
        depth: 1,
    };
    let texture = context.create_texture(gpu::TextureDesc {
        name,
        format,
        size,
        dimension: gpu::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
        sample_count: 1,
        external: None,
    });
    let view = context.create_texture_view(
        texture,
        gpu::TextureViewDesc {
            name,
            format,
            dimension: gpu::ViewDimension::D2,
            subresources: &Default::default(),
        },
    );
    let stage = context.create_buffer(gpu::BufferDesc {
        name,
        size: 4,
        memory: gpu::Memory::Upload,
    });
    unsafe {
        std::ptr::copy_nonoverlapping(color.as_ptr(), stage.data(), 4);
    }

    encoder.init_texture(texture);
    if let mut transfer = encoder.transfer(name) {
        transfer.copy_buffer_to_texture(stage.at(0), 4, texture.into(), size);
    }
    (texture, view, stage)
}

/// Draw a near quad and then a far one, and check the near one survives.
///
/// Draw order alone would put the far quad on top, so a backend that drops
/// `RenderPipelineDesc::depth_stencil` renders this inside out.
#[test]
#[ignore = "requires a working GPU context"]
fn depth_state_gpu_test() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    let format = gpu::TextureFormat::Rgba8Unorm;
    let target = snapshot::OffscreenTarget::new(&context, STATE_TEST_SIZE, format);

    let depth_texture = context.create_texture(gpu::TextureDesc {
        name: "depth-state/depth",
        format: gpu::TextureFormat::Depth32Float,
        size: STATE_TEST_SIZE,
        dimension: gpu::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: gpu::TextureUsage::TARGET,
        sample_count: 1,
        external: None,
    });
    let depth_view = context.create_texture_view(
        depth_texture,
        gpu::TextureViewDesc {
            name: "depth-state/depth",
            format: gpu::TextureFormat::Depth32Float,
            dimension: gpu::ViewDimension::D2,
            subresources: &Default::default(),
        },
    );

    let shader = context.create_shader(gpu::ShaderDesc {
        source: include_str!("shaders/pipeline_state.wgsl"),
        naga_module: None,
    });
    let quad_layout = <QuadData as gpu::ShaderData>::layout();
    let mut pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
        name: "depth-state",
        data_layouts: &[&quad_layout],
        vertex: shader.at("quad_vs"),
        vertex_fetches: &[],
        primitive: gpu::PrimitiveState {
            topology: gpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(gpu::DepthStencilState {
            format: gpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: gpu::CompareFunction::Less,
            stencil: gpu::StencilState::default(),
            bias: gpu::DepthBiasState::default(),
        }),
        fragment: Some(shader.at("quad_fs")),
        color_targets: &[format.into()],
        multisample_state: gpu::MultisampleState::default(),
    });

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "depth-state",
        buffer_count: 1,
        manual_barriers: false,
    });
    command_encoder.start();
    command_encoder.init_texture(target.texture);
    command_encoder.init_texture(depth_texture);

    if let mut pass = command_encoder.render(
        "depth-state",
        gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: target.view,
                init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: Some(gpu::RenderTarget {
                view: depth_view,
                init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                finish_op: gpu::FinishOp::Discard,
            }),
        },
    ) && let mut pc = pass.with(&pipeline)
    {
        // near, and green
        pc.bind(
            0,
            &QuadData {
                quad_params: QuadParams {
                    color: [0.0, 1.0, 0.0, 1.0],
                    depth: 0.25,
                    pad: [0.0; 3],
                },
            },
        );
        pc.draw(0, 3, 0, 1);
        // far, and red: covers the same pixels, and has to lose all of them
        pc.bind(
            0,
            &QuadData {
                quad_params: QuadParams {
                    color: [1.0, 0.0, 0.0, 1.0],
                    depth: 0.75,
                    pad: [0.0; 3],
                },
            },
        );
        pc.draw(0, 3, 0, 1);
    }

    let pixels = target.read_pixels(&context, &mut command_encoder);
    for (i, texel) in pixels.chunks(4).enumerate() {
        assert!(
            texel[1] > 200 && texel[0] < 50,
            "texel {} is {:?}, so the far quad won the depth test",
            i,
            texel
        );
    }

    context.destroy_render_pipeline(&mut pipeline);
    context.destroy_texture_view(depth_view);
    context.destroy_texture(depth_texture);
    context.destroy_command_encoder(&mut command_encoder);
    target.destroy(&context);
}

/// Bind two textures to one pipeline and check they land in different slots.
///
/// A backend that leaves them sharing a slot shows whichever was bound last on
/// both halves of the screen, which no single-texture pipeline would reveal.
///
/// Which of the GLES binding paths this takes is the driver's call: the slots
/// come from the shader's explicit bindings where `GL_EXT_buffer_storage` is
/// available, and are assigned by hand where it isn't. A desktop driver has the
/// extension, so covering the hand-assigned path — the one WebGL2 always takes
/// — means running this on a context that lacks it.
#[test]
#[ignore = "requires a working GPU context"]
fn multi_texture_gpu_test() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    let format = gpu::TextureFormat::Rgba8Unorm;
    let target = snapshot::OffscreenTarget::new(&context, STATE_TEST_SIZE, format);

    let shader = context.create_shader(gpu::ShaderDesc {
        source: include_str!("shaders/pipeline_state.wgsl"),
        naga_module: None,
    });
    let split_layout = <SplitData as gpu::ShaderData>::layout();
    let mut pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
        name: "multi-texture",
        data_layouts: &[&split_layout],
        vertex: shader.at("split_vs"),
        vertex_fetches: &[],
        primitive: gpu::PrimitiveState {
            topology: gpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        fragment: Some(shader.at("split_fs")),
        color_targets: &[format.into()],
        multisample_state: gpu::MultisampleState::default(),
    });
    let sampler = context.create_sampler(gpu::SamplerDesc {
        name: "multi-texture",
        ..Default::default()
    });

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "multi-texture",
        buffer_count: 1,
        manual_barriers: false,
    });
    command_encoder.start();
    command_encoder.init_texture(target.texture);

    let (left_texture, left_view, left_stage) = make_solid_texture(
        "multi-texture/left",
        [0, 255, 0, 255],
        &mut command_encoder,
        &context,
    );
    let (right_texture, right_view, right_stage) = make_solid_texture(
        "multi-texture/right",
        [0, 0, 255, 255],
        &mut command_encoder,
        &context,
    );

    if let mut pass = command_encoder.render(
        "multi-texture",
        gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: target.view,
                init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        },
    ) && let mut pc = pass.with(&pipeline)
    {
        pc.bind(
            0,
            &SplitData {
                left_texture: left_view,
                right_texture: right_view,
                samp: sampler,
            },
        );
        pc.draw(0, 3, 0, 1);
    }

    let pixels = target.read_pixels(&context, &mut command_encoder);
    let width = STATE_TEST_SIZE.width as usize;
    for (i, texel) in pixels.chunks(4).enumerate() {
        let (channel, name) = if i % width < width / 2 {
            (1, "green")
        } else {
            (2, "blue")
        };
        assert!(
            texel[channel] > 200,
            "texel {} is {:?}, expected {}: the two textures share a slot",
            i,
            texel,
            name
        );
    }

    context.destroy_sampler(sampler);
    context.destroy_render_pipeline(&mut pipeline);
    context.destroy_buffer(left_stage);
    context.destroy_buffer(right_stage);
    context.destroy_texture_view(left_view);
    context.destroy_texture(left_texture);
    context.destroy_texture_view(right_view);
    context.destroy_texture(right_texture);
    context.destroy_command_encoder(&mut command_encoder);
    target.destroy(&context);
}
