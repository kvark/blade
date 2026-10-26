//! Small synthetic inputs isolate SVGF guide/variance errors from ray noise.
use blade_graphics as gpu;
use gpu::ShaderData as _;

#[test]
#[ignore = "requires a working compute GPU"]
fn normals_use_inverse_transpose_for_nonuniform_scale() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    let output = context.create_buffer(gpu::BufferDesc {
        name: "normal transform results",
        size: 48,
        memory: gpu::Memory::Shared,
    });
    let source = format!(
        "{}\n{}",
        super::snapshot::shader_source("vertex.inc.wgsl"),
        include_str!("../shaders/normal_transform.wgsl")
    );
    let shader = context.create_shader(gpu::ShaderDesc {
        source: &source,
        naga_module: None,
    });
    let layout = super::DispatchGlobals::layout();
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "normal transform",
        data_layouts: &[&layout],
        compute: shader.at("main"),
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "normal transform test",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    if let mut pass = encoder.compute("normal transform")
        && let mut pc = pass.with(&pipeline)
    {
        pc.bind(
            0,
            &super::DispatchGlobals {
                input: output.into(),
                output: output.into(),
            },
        );
        pc.dispatch([3, 1, 1]);
    }
    let sync = context.submit(&mut encoder);
    assert!(context.wait_for(&sync, 5000).unwrap());
    let actual = unsafe { std::slice::from_raw_parts(output.data().cast::<[f32; 4]>(), 3) };
    for (actual, expected) in actual.iter().zip([
        glam::vec3(0.5, 2.0 / 3.0, 0.75),
        glam::vec3(-0.5, 2.0 / 3.0, 0.75),
        glam::vec3(-2.0 / 3.0, 0.5, 0.75),
    ]) {
        assert!(
            glam::Vec3::from_slice(actual).abs_diff_eq(expected.normalize(), 1.0e-5),
            "{actual:?}"
        );
    }
    context.destroy_command_encoder(&mut encoder);
    context.destroy_compute_pipeline(&mut pipeline);
    context.destroy_buffer(output);
}

#[derive(blade_macros::ShaderData)]
struct FilterData {
    camera: [u32; 16],
    prev_camera: [u32; 16],
    params: [u32; 6],
    input: gpu::TextureView,
    input_specular: gpu::TextureView,
    current_diffuse: gpu::TextureView,
    current_specular: gpu::TextureView,
    output: gpu::TextureView,
    output_specular: gpu::TextureView,
    t_depth: gpu::TextureView,
    t_prev_depth: gpu::TextureView,
    t_flat_normal: gpu::TextureView,
    t_prev_flat_normal: gpu::TextureView,
    t_basis: gpu::TextureView,
    t_prev_basis: gpu::TextureView,
    t_specular_f0: gpu::TextureView,
    t_prev_specular_f0: gpu::TextureView,
    t_motion: gpu::TextureView,
}

#[test]
#[ignore = "requires a working compute GPU"]
fn svgf_preserves_edges_and_temporal_moments() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    let size = gpu::Extent {
        width: 16,
        height: 4,
        depth: 1,
    };
    let count = (size.width * size.height) as usize;
    let source = super::snapshot::shader_source("a-trous.wgsl");
    let shader = context.create_shader(gpu::ShaderDesc {
        source: &source,
        naga_module: None,
    });
    let layout = FilterData::layout();
    let mut spatial = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "SVGF edge test",
        data_layouts: &[&layout],
        compute: shader.at("atrous3x3"),
    });
    let mut temporal = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "SVGF moment test",
        data_layouts: &[&layout],
        compute: shader.at("temporal_accum"),
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "SVGF tests",
        buffer_count: 1,
        manual_barriers: false,
    });
    // CameraParams: identity orientation, narrow FOV, target size.
    let mut camera = [0u32; 16];
    camera[3] = 100.0f32.to_bits();
    camera[7] = 1.0f32.to_bits();
    camera[8] = 0.1f32.to_bits();
    camera[9] = 0.1f32.to_bits();
    camera[12] = size.width;
    camera[13] = size.height;

    for case in [
        "depth edge",
        "specular normal edge",
        "moving history",
        "disocclusion",
        "fresh noisy surface",
    ] {
        encoder.start();
        let mut textures = Vec::new();
        let mut buffers = Vec::new();
        let mut upload = |name: &str, pixels: Vec<[f32; 4]>| {
            let texture = context.create_texture(gpu::TextureDesc {
                name,
                format: gpu::TextureFormat::Rgba16Float,
                size,
                dimension: gpu::TextureDimension::D2,
                array_layer_count: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: gpu::TextureUsage::COPY
                    | gpu::TextureUsage::RESOURCE
                    | gpu::TextureUsage::STORAGE,
                external: None,
            });
            let view = context.create_texture_view(
                texture,
                gpu::TextureViewDesc {
                    name,
                    format: gpu::TextureFormat::Rgba16Float,
                    dimension: gpu::ViewDimension::D2,
                    subresources: &Default::default(),
                },
            );
            let half_pixels: Vec<half::f16> = pixels
                .iter()
                .flatten()
                .map(|&v| half::f16::from_f32(v))
                .collect();
            let stage = context.create_buffer(gpu::BufferDesc {
                name,
                size: (count * 8) as u64,
                memory: gpu::Memory::Upload,
            });
            unsafe {
                std::ptr::copy_nonoverlapping(
                    half_pixels.as_ptr().cast::<u8>(),
                    stage.data(),
                    count * 8,
                );
            }
            context.sync_buffer(stage.into(), stage.size(), gpu::BufferTarget::Data);
            encoder.init_texture(texture);
            if let mut pass = encoder.transfer(name) {
                pass.copy_buffer_to_texture(stage.into(), size.width * 8, texture.into(), size);
            }
            textures.push((texture, view));
            buffers.push(stage);
            view
        };
        let history_case = case == "moving history" || case == "disocclusion";
        let input = upload(
            "input",
            (0..count)
                .map(|i| {
                    if history_case {
                        [1.0, 1.0, 1.0, 0.25]
                    } else if i % 16 < 8 {
                        [1.0; 4]
                    } else {
                        [0.0, 0.0, 0.0, 1.0]
                    }
                })
                .collect(),
        );
        let output = upload("output", vec![[2.0, 2.0, 2.0, 0.0]; count]);
        let output_specular = upload(
            "output specular",
            vec![
                if history_case {
                    [4.0, 4.0, 4.0, 0.0]
                } else {
                    [2.0, 2.0, 2.0, 0.0]
                };
                count
            ],
        );
        let input_specular = if history_case {
            upload("specular history", vec![[3.0, 3.0, 3.0, 0.25]; count])
        } else {
            input
        };
        let fresh = case == "fresh noisy surface";
        let current_diffuse = upload(
            "current raw diffuse",
            (0..count)
                .map(|i| {
                    let light = if fresh {
                        2.0 * ((i + i / 16) % 2) as f32
                    } else {
                        2.0
                    };
                    [light, light, light, 0.0]
                })
                .collect(),
        );
        let current_specular = upload(
            "current raw specular",
            (0..count)
                .map(|i| {
                    let light = if fresh {
                        2.0 * ((i + i / 16) % 2) as f32
                    } else if history_case {
                        4.0
                    } else {
                        2.0
                    };
                    [light, light, light, 0.0]
                })
                .collect(),
        );
        let t_depth = upload(
            "depth",
            (0..count)
                .map(|i| {
                    [
                        if case == "depth edge" && i % 16 >= 8 {
                            1.1
                        } else {
                            1.0
                        },
                        0.0,
                        0.0,
                        0.0,
                    ]
                })
                .collect(),
        );
        let t_prev_depth = upload("previous depth", vec![[1.1, 0.0, 0.0, 0.0]; count]);
        let t_flat_normal = upload("flat normal", vec![[0.0, 0.0, 1.0, 0.0]; count]);
        let t_basis = upload(
            "shading basis",
            (0..count)
                .map(|i| {
                    if case == "specular normal edge" && i % 16 >= 8 {
                        let angle = std::f32::consts::FRAC_PI_8;
                        [0.0, angle.sin(), 0.0, angle.cos()]
                    } else {
                        [0.0, 0.0, 0.0, 1.0]
                    }
                })
                .collect(),
        );
        let t_specular_f0 = upload("material", vec![[0.04, 0.04, 0.04, 0.1]; count]);
        let t_motion = upload(
            "motion and previous distance",
            vec![
                [
                    0.0,
                    0.0,
                    if case == "disocclusion" { 2.0 } else { 1.1 },
                    0.0
                ];
                count
            ],
        );
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "SVGF readback",
            size: (count * 16) as u64,
            memory: gpu::Memory::Download,
        });
        let data = FilterData {
            camera,
            prev_camera: camera,
            params: [
                size.width,
                size.height,
                if fresh { 1.0f32 } else { 0.1f32 }.to_bits(),
                0,
                1,
                0,
            ],
            input,
            input_specular,
            current_diffuse,
            current_specular,
            output,
            output_specular,
            t_depth,
            t_prev_depth,
            t_flat_normal,
            t_prev_flat_normal: t_flat_normal,
            t_basis,
            t_prev_basis: t_basis,
            t_specular_f0,
            t_prev_specular_f0: t_specular_f0,
            t_motion,
        };
        if let mut pass = encoder.compute(case)
            && let mut pc = pass.with(if history_case || fresh {
                &temporal
            } else {
                &spatial
            })
        {
            pc.bind(0, &data);
            pc.dispatch([2, 1, 1]);
        }
        if let mut pass = encoder.transfer("read SVGF") {
            for lobe in 0..2 {
                pass.copy_texture_to_buffer(
                    textures[lobe + 1].0.into(),
                    readback.at((lobe * count * 8) as u64),
                    size.width * 8,
                    size,
                );
            }
        }
        let sync = context.submit(&mut encoder);
        assert!(context.wait_for(&sync, 5000).unwrap());
        let result =
            unsafe { std::slice::from_raw_parts(readback.data().cast::<half::f16>(), count * 8) };
        for (i, pixel) in result.as_chunks::<4>().0.iter().enumerate() {
            let lobe = i / count;
            if fresh {
                let index = i % count;
                let light = 2.0 * ((index + index / 16) % 2) as f32;
                assert!((pixel[0].to_f32() - light).abs() < 0.001);
                assert!(
                    pixel[3].to_f32() > 0.9,
                    "newly visible pixels need spatial variance: {pixel:?}"
                );
            } else if history_case {
                let (color, variance) = if case == "moving history" {
                    (1.1 + 2.0 * lobe as f32, 0.315)
                } else {
                    (2.0 + 2.0 * lobe as f32, 0.0)
                };
                assert!(
                    (pixel[0].to_f32() - color).abs() < 0.003,
                    "{case}: {pixel:?}"
                );
                assert!(
                    (pixel[3].to_f32() - variance).abs() < 0.003,
                    "{case}: {pixel:?}"
                );
            } else {
                if case == "specular normal edge" && lobe == 0 {
                    assert!((0.0..=1.0).contains(&pixel[0].to_f32()));
                    continue;
                }
                let expected = if i % 16 < 8 { 1.0 } else { 0.0 };
                assert!(
                    (pixel[0].to_f32() - expected).abs() < 0.001,
                    "{case}: pixel {i}: {pixel:?}"
                );
            }
        }
        println!("SVGF verified: {case}");
        context.destroy_buffer(readback);
        for buffer in buffers {
            context.destroy_buffer(buffer);
        }
        for (texture, view) in textures {
            context.destroy_texture_view(view);
            context.destroy_texture(texture);
        }
    }
    context.destroy_command_encoder(&mut encoder);
    context.destroy_compute_pipeline(&mut spatial);
    context.destroy_compute_pipeline(&mut temporal);
}
