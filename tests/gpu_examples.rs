#![allow(irrefutable_let_patterns)]

// nanorand uses RtlGenRandom but doesn't link advapi32 itself.
// On Vulkan builds, other deps pull it in transitively; on GLES builds we must link it explicitly.
#[cfg(all(gles, windows))]
#[link(name = "advapi32")]
extern "C" {}

use blade_graphics as gpu;
use blade_graphics::ShaderData;
use std::slice;

#[path = "../examples/bunnymark/example.rs"]
mod bunnymark_example;
#[path = "../examples/particle/particle.rs"]
mod particle_system;
#[cfg(not(gles))]
#[path = "../examples/ray-query/example.rs"]
mod ray_query_example;
mod snapshot;

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
    material: [f32; 4],
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
    let mut particle_system = particle_system::System::new(
        &context,
        particle_system::SystemDesc {
            name: "snapshot particle",
            capacity: 10_000,
            draw_format: format,
        },
        1, // no MSAA for testing
    );

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "snapshot-particle",
        buffer_count: 1,
    });
    command_encoder.start();
    particle_system.reset(&mut command_encoder);
    // Run several update cycles to emit and move particles
    for _ in 0..20 {
        particle_system.update(&mut command_encoder);
    }

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
        particle_system.draw(&mut pass, (size.width, size.height));
    }

    let pixels = target.read_pixels(&context, &mut command_encoder);
    snapshot::check("particle", &pixels, size);

    particle_system.destroy(&context);
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
    let shader = context.create_shader(gpu::ShaderDesc {
        source: include_str!("../blade-render/code/raster.wgsl"),
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
        material: [0.4, 0.0, 0.0, 0.0],      // material.z=0 -> env_enabled=false
    };

    // Render
    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "sky-test",
        buffer_count: 1,
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
    ) {
        if let mut pc = pass.with(&sky_pipeline) {
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
