#![allow(irrefutable_let_patterns)]

use blade_graphics as gpu;
use blade_graphics::ShaderData;
use std::slice;

#[allow(dead_code)]
#[path = "../examples/bunnymark/example.rs"]
mod bunnymark_example;
#[cfg(not(gles))]
mod denoiser;
mod pbr_scene;
#[cfg(not(gles))]
#[path = "../examples/ray-query/example.rs"]
mod ray_query_example;
mod snapshot;

/// Directory with the renderer shaders, needed by the asset hub.
const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/blade-render/code");

// --- Sky snapshot test structs ---

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct SkyFrameParams {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    light_view_proj: [f32; 16],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    ambient_color: [f32; 4],
    settings: [f32; 4],
    shadow_params: [f32; 4],
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

#[cfg(not(gles))]
#[derive(blade_macros::ShaderData)]
struct GeometryTransformData {
    acc_struct: gpu::AccelerationStructure,
    rays: gpu::BufferPiece,
    hits: gpu::BufferPiece,
}

/// Test geometry transforms independently of shading, including a private
/// transform buffer, a nonzero offset, TLAS composition, and a BLAS refit.
#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn geometry_transform_ray_query() {
    if cfg!(target_os = "macos") && std::env::var_os("CI").is_some() {
        println!("Skipping: acceleration structure tests are disabled on macOS CI");
        return;
    }
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            ..Default::default()
        })
        .unwrap()
    };
    if !context
        .capabilities()
        .ray_query
        .contains(gpu::ShaderVisibility::COMPUTE)
    {
        println!("Skipping: ray_query compute not supported");
        return;
    }
    let upload = |name, size| {
        context.create_buffer(gpu::BufferDesc {
            name,
            size,
            memory: gpu::Memory::Shared,
        })
    };
    let vertex_buffer = upload("triangle", 36);
    unsafe {
        let vertices = slice::from_raw_parts_mut(vertex_buffer.data().cast::<f32>(), 9);
        vertices.copy_from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }
    context.sync_buffer(vertex_buffer.into(), 36, gpu::BufferTarget::Data);
    let transform_size = std::mem::size_of::<gpu::AccelerationStructureTransform>() as u64;
    let transform_stage = upload("transform upload", transform_size * 2);
    let transform_buffer = context.create_buffer(gpu::BufferDesc {
        name: "private geometry transform",
        size: transform_size * 2,
        memory: gpu::Memory::Device,
    });
    let mesh = gpu::AccelerationStructureMesh {
        vertex_data: vertex_buffer.into(),
        vertex_format: gpu::VertexFormat::F32Vec3,
        vertex_stride: 12,
        vertex_count: 3,
        index_data: gpu::Buffer::default().into(),
        index_type: None,
        triangle_count: 1,
        transform_data: gpu::Buffer::default().into(),
        is_opaque: true,
    };
    let meshes = [
        mesh.clone(),
        gpu::AccelerationStructureMesh {
            transform_data: transform_buffer.at(transform_size),
            ..mesh
        },
    ];
    let blas_sizes = context.get_bottom_level_acceleration_structure_sizes(&meshes);
    let blas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
        name: "geometry transform BLAS",
        ty: gpu::AccelerationStructureType::BottomLevel,
        size: blas_sizes.data,
        updatable: true,
    });
    let to_transform = |matrix: glam::Mat4| -> gpu::Transform {
        let rows = matrix.transpose().to_cols_array_2d();
        [rows[0], rows[1], rows[2]].into()
    };
    let instance_matrix = glam::Mat4::from_scale_rotation_translation(
        glam::vec3(0.8, 1.4, 1.2),
        glam::Quat::from_rotation_x(0.4),
        glam::vec3(1.0, -2.0, 3.0),
    );
    let instance_buffer = context.create_acceleration_structure_instance_buffer(
        &[gpu::AccelerationStructureInstance {
            transform: to_transform(instance_matrix),
            ..Default::default()
        }],
        &[blas],
    );
    let tlas_sizes = context.get_top_level_acceleration_structure_sizes(1);
    let tlas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
        name: "geometry transform TLAS",
        ty: gpu::AccelerationStructureType::TopLevel,
        size: tlas_sizes.data,
        updatable: false,
    });
    let scratch = context.create_buffer(gpu::BufferDesc {
        name: "AS scratch",
        size: blas_sizes.scratch.max(tlas_sizes.scratch),
        memory: gpu::Memory::Device,
    });
    let rays = upload("rays", 6 * 32);
    let hits = upload("hits", 6 * 16);
    let shader = context.create_shader(gpu::ShaderDesc {
        source: include_str!("shaders/geometry_transform.wgsl"),
        naga_module: None,
    });
    let layout = GeometryTransformData::layout();
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "geometry transform query",
        data_layouts: &[&layout],
        compute: shader.at("main"),
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "geometry transform test",
        buffer_count: 1,
        manual_barriers: false,
    });
    for (phase, geometry_matrix) in [
        glam::Mat4::from_scale_rotation_translation(
            glam::vec3(1.7, 0.6, 2.3),
            glam::Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.7),
            glam::vec3(4.0, 1.0, 2.0),
        ),
        glam::Mat4::from_scale_rotation_translation(
            glam::vec3(0.7, 2.1, 0.8),
            glam::Quat::from_euler(glam::EulerRot::XYZ, 0.4, 0.2, -0.3),
            glam::vec3(7.0, -2.0, 1.0),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        // Poison the unused first slot to catch an ignored transform buffer offset.
        let packed = [
            bytemuck::Zeroable::zeroed(),
            gpu::AccelerationStructureTransform::from(to_transform(geometry_matrix)),
        ];
        unsafe {
            std::ptr::copy_nonoverlapping(
                packed.as_ptr(),
                transform_stage
                    .data()
                    .cast::<gpu::AccelerationStructureTransform>(),
                packed.len(),
            );
        }
        let mut ray_data = Vec::new();
        let mut expected = Vec::new();
        for (geometry_index, geometry) in [glam::Mat4::IDENTITY, geometry_matrix]
            .into_iter()
            .enumerate()
        {
            let world = instance_matrix * geometry;
            let normal = world
                .transform_vector3(glam::Vec3::X)
                .cross(world.transform_vector3(glam::Vec3::Y))
                .normalize();
            for (u, v) in [(0.2, 0.3), (0.8, 0.1), (1.2, 0.3)] {
                let target = world.transform_point3(glam::vec3(u, v, 0.0));
                ray_data.push((target + normal * 5.0).extend(0.0).to_array());
                ray_data.push((-normal).extend(0.0).to_array());
                expected.push(if u + v < 1.0 {
                    [5.0, u, v, geometry_index as f32]
                } else {
                    [-1.0; 4]
                });
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                ray_data.as_ptr(),
                rays.data().cast::<[f32; 4]>(),
                ray_data.len(),
            );
        }
        context.sync_buffer(
            transform_stage.into(),
            transform_stage.size(),
            gpu::BufferTarget::Data,
        );
        context.sync_buffer(rays.into(), rays.size(), gpu::BufferTarget::Data);
        encoder.start();
        if let mut pass = encoder.transfer("upload transform") {
            pass.copy_buffer_to_buffer(
                transform_stage.into(),
                transform_buffer.into(),
                transform_buffer.size(),
            );
        }
        if let mut pass = encoder.acceleration_structure("BLAS") {
            if phase == 0 {
                pass.build_bottom_level(blas, &meshes, scratch.into());
            } else {
                pass.update_bottom_level(blas, &meshes, scratch.into());
            }
        }
        if let mut pass = encoder.acceleration_structure("TLAS") {
            pass.build_top_level(tlas, &[blas], 1, instance_buffer.into(), scratch.into());
        }
        if let mut pass = encoder.compute("trace")
            && let mut pc = pass.with(&pipeline)
        {
            pc.bind(
                0,
                &GeometryTransformData {
                    acc_struct: tlas,
                    rays: rays.into(),
                    hits: hits.into(),
                },
            );
            pc.dispatch([6, 1, 1]);
        }
        let sync = context.submit(&mut encoder);
        assert!(context.wait_for(&sync, 5000).unwrap());
        let actual =
            unsafe { slice::from_raw_parts(hits.data().cast::<[f32; 4]>(), expected.len()) };
        for (ray, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, e)| (a - e).abs() < 0.0001),
                "phase {phase}, ray {ray}: {actual:?} != {expected:?}"
            );
        }
    }
    context.destroy_command_encoder(&mut encoder);
    context.destroy_compute_pipeline(&mut pipeline);
    context.destroy_acceleration_structure(tlas);
    context.destroy_acceleration_structure(blas);
    for buffer in [
        vertex_buffer,
        transform_stage,
        transform_buffer,
        instance_buffer,
        scratch,
        rays,
        hits,
    ] {
        context.destroy_buffer(buffer);
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
    context.sync_buffer(input.into(), input.size(), gpu::BufferTarget::Data);

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

#[test]
#[ignore = "requires a working GPU context with bfloat16"]
fn bfloat16_gpu_test() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    if !context.capabilities().shader_bfloat16 {
        println!("Skipping: shader bfloat16 is not supported");
        return;
    }

    let input = context.create_buffer(gpu::BufferDesc {
        name: "bfloat16-input",
        size: 16,
        memory: gpu::Memory::Shared,
    });
    let output = context.create_buffer(gpu::BufferDesc {
        name: "bfloat16-output",
        size: 16,
        memory: gpu::Memory::Shared,
    });
    unsafe {
        slice::from_raw_parts_mut(input.data() as *mut f32, 4)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    }
    context.sync_buffer(input.into(), input.size(), gpu::BufferTarget::Data);

    let shader = context.create_shader(gpu::ShaderDesc {
        source: r#"
enable wgpu_bfloat16;

var<storage, read> input: array<f32>;
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output[id.x] = f32(bf16(input[id.x]));
}
"#,
        naga_module: None,
    });
    let global_layout = DispatchGlobals::layout();
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "bfloat16-test",
        data_layouts: &[&global_layout],
        compute: shader.at("main"),
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "bfloat16-test",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    if let mut compute = encoder.compute("bfloat16")
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
    let sync_point = context.submit(&mut encoder);
    assert!(context.wait_for(&sync_point, 2000).unwrap());

    let actual = unsafe { slice::from_raw_parts(output.data() as *const f32, 4) };
    assert_eq!(actual, [1.0, 2.0, 3.0, 4.0]);

    context.destroy_command_encoder(&mut encoder);
    context.destroy_compute_pipeline(&mut pipeline);
    context.destroy_buffer(output);
    context.destroy_buffer(input);
}

#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context"]
fn manual_barrier_gpu_test() {
    run_dispatch_gpu_test(true);
}

/// `last_timing` reports the last submission after its sync point is waited on.
#[test]
#[ignore = "requires a working GPU context"]
fn last_timing_reports_the_last_submission() {
    let probe = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    if !probe.capabilities().timing {
        println!("Skipping: this device does not support GPU timestamps");
        return;
    }
    drop(probe);
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            timing: true,
            ..Default::default()
        })
        .unwrap()
    };

    let shader = context.create_shader(gpu::ShaderDesc {
        source: "@compute @workgroup_size(1) fn main() {}",
        naga_module: None,
    });
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "timing-test",
        data_layouts: &[],
        compute: shader.at("main"),
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "timing-test",
        buffer_count: 2,
        manual_barriers: false,
    });

    let submit = |encoder: &mut gpu::CommandEncoder, label: &str| {
        encoder.start();
        if let mut compute = encoder.compute(label)
            && let mut pass = compute.with(&pipeline)
        {
            pass.dispatch([1, 1, 1]);
        }
        let sync_point = context.submit(encoder);
        assert!(context.wait_for(&sync_point, 5000).unwrap());
        encoder
            .last_timing()
            .passes
            .iter()
            .map(|(name, _)| name.to_string())
            .collect::<Vec<_>>()
    };

    let first = submit(&mut encoder, "first");
    let first_again = encoder
        .last_timing()
        .passes
        .iter()
        .map(|(name, _)| name.to_string())
        .collect::<Vec<_>>();
    let second = submit(&mut encoder, "second");

    let empty_target = snapshot::OffscreenTarget::new(
        &context,
        gpu::Extent {
            width: 1,
            height: 1,
            depth: 1,
        },
        gpu::TextureFormat::Rgba8Unorm,
    );
    // Empty trailing passes must not expose unwritten stage-boundary samples.
    for kind in ["compute", "transfer", "render"] {
        encoder.start();
        if kind == "render" {
            encoder.init_texture(empty_target.texture);
        }
        {
            let mut pass = encoder.compute("work");
            pass.with(&pipeline).dispatch([1, 1, 1]);
        }
        match kind {
            "compute" => drop(encoder.compute("empty compute")),
            "transfer" => drop(encoder.transfer("empty transfer")),
            "render" => drop(encoder.render(
                "empty render",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: empty_target.view,
                        init_op: gpu::InitOp::DontCare,
                        finish_op: gpu::FinishOp::Discard,
                    }],
                    depth_stencil: None,
                },
            )),
            _ => unreachable!(),
        }
        let sync = context.submit(&mut encoder);
        assert!(context.wait_for(&sync, 5000).unwrap());
        let timing = encoder.last_timing();
        assert_eq!(timing.passes.len(), 2);
        assert!(timing.done >= timing.passes[1].1, "empty {kind}");
        println!(
            "Empty {kind} timing verified: {:?}",
            timing.pass_durations().collect::<Vec<_>>()
        );
    }
    empty_target.destroy(&context);

    // Empty submissions must not expose an unwritten/stale completion counter.
    encoder.start();
    let sync = context.submit(&mut encoder);
    assert!(context.wait_for(&sync, 5000).unwrap());
    assert!(encoder.last_timing().passes.is_empty());

    context.destroy_command_encoder(&mut encoder);
    context.destroy_compute_pipeline(&mut pipeline);

    assert_eq!(first, ["first"]);
    assert_eq!(first_again, first);
    assert_eq!(second, ["second"]);
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
    if cfg!(target_os = "macos") && std::env::var_os("CI").is_some() {
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
        let view = glam::camera::rh::view::look_at_mat4(pos, glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(fov_y, aspect, near, far);
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
    let proj = glam::camera::rh::proj::directx::perspective(fov_y, aspect, near, far);
    let view = glam::Mat4::IDENTITY; // camera at origin, looking along -Z in RH
    let view_proj = proj * view;
    let inv_view_proj = view_proj.inverse();

    let frame_params = SkyFrameParams {
        view_proj: view_proj.to_cols_array(),
        inv_view_proj: inv_view_proj.to_cols_array(),
        light_view_proj: glam::Mat4::IDENTITY.to_cols_array(),
        camera_pos: [0.0, 0.0, 0.0, 1.0],
        light_dir: [0.0, -1.0, 0.0, 0.0],
        light_color: [1.0, 1.0, 1.0, 0.0],
        ambient_color: [0.0, 0.0, 0.0, 1.0], // w=1.0 -> space_sky mode
        // x=0: no environment map, y=1: encode for a non-sRGB surface
        settings: [0.0, 1.0, 0.0, 0.0],
        shadow_params: [0.0; 4],
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

struct PbrHarness {
    context: std::sync::Arc<gpu::Context>,
    choir: std::sync::Arc<choir::Choir>,
    workers: Vec<choir::WorkerHandle>,
    asset_hub: blade_render::AssetHub,
    shaders: blade_render::Shaders,
}

impl PbrHarness {
    /// Bring up the asset hub and cook the renderer shaders.
    fn new(context: gpu::Context, cache_name: &str, ray_tracing: bool) -> Self {
        let context = std::sync::Arc::new(context);
        let choir = choir::Choir::new();
        #[cfg(not(any(gles, target_arch = "wasm32")))]
        let workers = (0..2)
            .map(|i| choir.add_worker(&format!("{cache_name}-{i}")))
            .collect::<Vec<_>>();
        #[cfg(any(gles, target_arch = "wasm32"))]
        let workers = Vec::new();
        let cache_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-assets")
            .join(cache_name);
        let asset_hub = blade_render::AssetHub::new(&cache_path, &choir, &context);
        let (shaders, shader_task) =
            blade_render::Shaders::load(SHADER_DIR.as_ref(), &asset_hub, ray_tracing);
        if workers.is_empty() {
            shader_task.join_active();
        } else {
            shader_task.join();
        }
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

    let mut objects = vec![blade_render::Object::from(
        harness.create_grid_model([0.05, 1.0]),
    )];
    let mut temp_buffers = Vec::new();
    let mut frame_resources = blade_render::FrameResources::default();
    harness
        .asset_hub
        .flush(&mut command_encoder, &mut temp_buffers);

    let raster_config = blade_render::RasterConfig {
        light_dir: mint::Vector3 {
            x: 0.4,
            y: 0.5,
            z: 1.0,
        },
        // Exercise the depth prepass and shadow resource bindings without changing
        // this material-reference snapshot's expected lighting.
        directional_shadows: Some(blade_render::DirectionalShadowConfig {
            resolution: 512,
            strength: 0.0,
            ..Default::default()
        }),
        ..Default::default()
    };
    rasterizer.prepare(
        &mut command_encoder,
        &mut objects,
        &harness.asset_hub,
        &context,
        &mut frame_resources,
    );
    rasterizer.render_directional_shadows(
        &mut command_encoder,
        &pbr_scene::camera(),
        &objects,
        &harness.asset_hub,
        &raster_config,
    );

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
            &[],
            &raster_config,
        );
    }

    let pixels = target.read_pixels(&context, &mut command_encoder);
    snapshot::check("pbr-raster", &pixels, size);

    for buffer in temp_buffers {
        context.destroy_buffer(buffer);
    }
    for buffer in frame_resources.buffers {
        context.destroy_buffer(buffer);
    }
    for object in objects.iter_mut() {
        object.destroy_gpu(&context);
    }
    rasterizer.destroy(&context);
    context.destroy_command_encoder(&mut command_encoder);
    target.destroy(&context);
    harness.destroy();
}

/// Load the repository's minimal animated GLB and render three exact clip
/// keyframes. This keeps the reference deterministic while covering glTF skin
/// cooking, joint palettes, non-uniform scale normals, and both skinned raster
/// pipelines (main and directional shadow).
#[test]
#[ignore = "requires a working GPU context"]
fn snapshot_animated_skin() {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    let size = gpu::Extent {
        width: 256,
        height: 256,
        depth: 1,
    };
    let format = gpu::TextureFormat::Rgba8UnormSrgb;
    let harness = PbrHarness::new(context, "animated-skin-raster", false);
    let context = std::sync::Arc::clone(&harness.context);
    let target = snapshot::OffscreenTarget::new(&context, size, format);

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("assets")
        .join("animated_skin.glb");
    let (model, task) = harness.asset_hub.models.load(
        path,
        blade_render::model::Meta {
            generate_tangents: false,
            front_face: blade_render::model::FrontFace::CounterClockwise,
        },
    );
    if harness.workers.is_empty() {
        task.join_active();
    } else {
        task.join();
    }

    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "snapshot-animated-skin",
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
            color_space: gpu::ColorSpace::Linear,
            max_debug_lines: 1,
        },
    );
    let mut temp_buffers = Vec::new();
    let mut frame_resources = blade_render::FrameResources::default();
    harness
        .asset_hub
        .flush(&mut command_encoder, &mut temp_buffers);

    let keyframes = [
        (0.0, glam::Quat::IDENTITY, glam::Vec3::ONE),
        (
            0.5,
            glam::Quat::from_rotation_z(45.0f32.to_radians()),
            glam::Vec3::new(1.6, 0.7, 0.55),
        ),
        (
            1.0,
            glam::Quat::from_rotation_z(-35.0f32.to_radians()),
            glam::Vec3::new(0.7, 1.4, 1.5),
        ),
    ];
    let mut objects: Vec<_> = keyframes
        .into_iter()
        .enumerate()
        .map(|(index, (_time, rotation, scale))| {
            let mut object = blade_render::Object::from(model);
            object.transform.x.w = index as f32 * 2.0 - 2.0;
            object.prev_transform = object.transform;
            object.pose = Some(blade_render::Pose::from_node_matrices(vec![
                glam::Mat4::IDENTITY,
                glam::Mat4::IDENTITY,
                glam::Mat4::from_scale_rotation_translation(scale, rotation, glam::Vec3::Y),
            ]));
            object
        })
        .collect();
    let camera = blade_render::Camera {
        pos: [0.0, 1.0, 6.0].into(),
        rot: mint::Quaternion {
            v: [0.0; 3].into(),
            s: 1.0,
        },
        fov_y: 0.8,
        depth: 100.0,
        fov: None,
    };
    let raster_config = blade_render::RasterConfig {
        clear_color: gpu::TextureColor::OpaqueBlack,
        light_dir: mint::Vector3 {
            x: 0.4,
            y: 0.3,
            z: 1.0,
        },
        light_color: mint::Vector3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        },
        ambient_color: mint::Vector3 {
            x: 0.08,
            y: 0.08,
            z: 0.08,
        },
        directional_shadows: Some(blade_render::DirectionalShadowConfig {
            strength: 0.5,
            ..Default::default()
        }),
        ..Default::default()
    };

    rasterizer.prepare(
        &mut command_encoder,
        &mut objects,
        &harness.asset_hub,
        &context,
        &mut frame_resources,
    );
    rasterizer.render_directional_shadows(
        &mut command_encoder,
        &camera,
        &objects,
        &harness.asset_hub,
        &raster_config,
    );
    command_encoder.init_texture(target.texture);
    command_encoder.init_texture(rasterizer.depth_texture());
    if let mut pass = command_encoder.render(
        "raster-animated-skin",
        gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: target.view,
                init_op: gpu::InitOp::Clear(raster_config.clear_color),
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
            &camera,
            &objects,
            &harness.asset_hub,
            None,
            &[],
            &raster_config,
        );
    }

    let pixels = target.read_pixels(&context, &mut command_encoder);
    snapshot::check("animated-skin", &pixels, size);

    for buffer in temp_buffers {
        context.destroy_buffer(buffer);
    }
    for buffer in frame_resources.buffers {
        context.destroy_buffer(buffer);
    }
    for object in objects.iter_mut() {
        object.destroy_gpu(&context);
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

#[cfg(not(gles))]
#[derive(Clone, Copy)]
struct RayTraceTestConfig {
    max_bounces: u32,
    pairwise_mis: bool,
    include_emissive_row: bool,
    emissive_scale: f32,
    black_environment: bool,
    tap_count: u32,
    denoise: bool,
    frozen_frames: usize,
}

#[cfg(not(gles))]
impl Default for RayTraceTestConfig {
    fn default() -> Self {
        Self {
            max_bounces: 3,
            pairwise_mis: true,
            include_emissive_row: true,
            emissive_scale: 1.0,
            black_environment: false,
            tap_count: 2,
            denoise: true,
            frozen_frames: 0,
        }
    }
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
    render_ray_traced_grid_configured(
        cache_name,
        mode,
        capture,
        RayTraceTestConfig::default(),
        inspect,
    )
}

#[cfg(not(gles))]
fn render_ray_traced_grid_configured(
    cache_name: &str,
    mode: RayTraceMode,
    capture: Capture,
    test_config: RayTraceTestConfig,
    inspect: impl FnOnce(&blade_render::RayTracer, &gpu::Context, &mut gpu::CommandEncoder),
) -> Option<Vec<u8>> {
    let profile_gpu = std::env::var_os("BLADE_PROFILE_GPU").is_some();
    // Metal acceleration structure APIs can throw uncatchable ObjC exceptions
    // in CI environments, even when the device reports ray tracing support.
    if cfg!(target_os = "macos") && std::env::var_os("CI").is_some() {
        println!("Skipping: ray tracing snapshot not supported on macOS CI");
        return None;
    }

    let context = unsafe {
        match gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            timing: profile_gpu,
            ..Default::default()
        }) {
            Ok(c) => c,
            Err(e) => {
                println!("Skipping: GPU context with ray tracing not available: {e:?}");
                return None;
            }
        }
    };
    let caps = context.capabilities();
    if !caps.ray_query.contains(gpu::ShaderVisibility::COMPUTE) || !caps.binding_array {
        println!("Skipping: ray_query compute or binding arrays not supported");
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
    let model = if test_config.include_emissive_row && test_config.emissive_scale == 1.0 {
        harness.create_grid_model([0.3, 1.0])
    } else {
        let mut geometries = pbr_scene::material_grid([0.3, 1.0]);
        if test_config.include_emissive_row {
            for geometry in geometries
                .iter_mut()
                .skip(pbr_scene::COLUMNS * (pbr_scene::ROWS - 1))
            {
                for channel in &mut geometry.emissive_factor {
                    *channel *= test_config.emissive_scale;
                }
            }
        } else {
            geometries.truncate(pbr_scene::COLUMNS * (pbr_scene::ROWS - 1));
        }
        let model = harness
            .asset_hub
            .models
            .baker
            .create_model("pbr-material-grid-no-emissive", geometries);
        harness.asset_hub.models.insert(model)
    };
    let mut objects = vec![blade_render::Object::from(model)];
    let environment = test_config.black_environment.then(|| {
        let texture = harness.asset_hub.textures.baker.create_texture(
            "pbr-black-environment",
            2,
            2,
            &[[0, 0, 0, 255]; 4],
        );
        harness.asset_hub.textures.insert(texture)
    });
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
        jitter_primary_rays: true,
        // the dummy environment map has no importance sampling data
        environment_importance_sampling: false,
        max_bounces: test_config.max_bounces,
        max_accumulated_samples: 0,
        tap_count: test_config.tap_count,
        tap_radius: 16,
        tap_confidence_near: 8,
        tap_confidence_far: 4,
        t_start: 0.01,
        // Snapshots use the energy-preserving pairwise path. Dedicated
        // regressions can disable reuse or isolate the cheaper approximation.
        pairwise_mis: test_config.pairwise_mis,
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
    for frame_index in 0..frame_count + test_config.frozen_frames {
        renderer.build_scene(
            &mut command_encoder,
            &mut objects,
            environment,
            &harness.asset_hub,
            &context,
            &mut temp,
        );
        renderer.prepare(
            &mut command_encoder,
            &camera,
            blade_render::FrameConfig {
                frozen: frame_index >= frame_count,
                debug_draw: false,
                // Match the engine: debug statistics reset each frame without
                // discarding the denoiser's temporal history.
                reset_variance: true,
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
            test_config.denoise.then_some(denoiser_config),
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

    if profile_gpu {
        let mut totals = std::collections::BTreeMap::<&str, Vec<f64>>::new();
        let timing = command_encoder.last_timing();
        for (name, duration) in timing.pass_durations() {
            totals
                .entry(name)
                .or_default()
                .push(duration.as_secs_f64() * 1000.0);
        }
        println!(
            "GPU profile: {}x{}, {frame_count} frames",
            size.width, size.height
        );
        for (name, mut millis) in totals {
            millis.sort_by(f64::total_cmp);
            println!(
                "  {name}: {:.3} ms/frame, median/pass {:.3}, range {:.3}–{:.3} ({} passes)",
                millis.iter().sum::<f64>() / frame_count as f64,
                millis[millis.len() / 2],
                millis[0],
                millis[millis.len() - 1],
                millis.len()
            );
        }
    }

    for buffer in temp.buffers {
        context.destroy_buffer(buffer);
    }
    for acceleration_structure in temp.acceleration_structures {
        context.destroy_acceleration_structure(acceleration_structure);
    }
    for object in objects.iter_mut() {
        object.destroy_gpu(&context);
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
        snapshot::check("pbr-ray-trace", &pixels, RAY_TRACE_SIZE);
    }
}

/// Pausing must preserve the warmed-up denoiser result, not start it over.
#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn frozen_frames_preserve_denoiser_history() {
    let render = |frozen_frames| {
        render_ray_traced_grid_configured(
            &format!("pbr-frozen-history-{frozen_frames}"),
            RayTraceMode::Restir,
            Capture::Display,
            RayTraceTestConfig {
                frozen_frames,
                ..Default::default()
            },
            |_, _, _| {},
        )
    };
    let (Some(warm), Some(frozen)) = (render(0), render(2)) else {
        return;
    };
    let difference = snapshot::mean_abs_diff(&warm, &frozen);
    println!("Frozen denoiser difference: {difference:.4}/255");
    // ReSTIR can update the estimate while paused, but dropping history causes
    // a much larger change (over 3/255 on the M3, versus about 0.5/255 retained).
    assert!(difference < 1.0, "pausing discarded denoiser history");
}

/// Exercise BLAS reuse and fenced retirement while poses and object count change.
#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn animated_blas_memory_stays_bounded() {
    let _ = env_logger::builder().is_test(true).try_init();
    let profile_vehicle = std::env::var_os("BLADE_PROFILE_VEHICLE").is_some();
    let profile_gpu = std::env::var_os("BLADE_PROFILE_GPU").is_some();
    let profile_updates = profile_vehicle || profile_gpu;
    if cfg!(target_os = "macos") && std::env::var_os("CI").is_some() {
        println!("Skipping: ray tracing stress test not supported on macOS CI");
        return;
    }
    let context = unsafe {
        match gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            timing: profile_gpu,
            validation: true,
            ..Default::default()
        }) {
            Ok(context) => context,
            Err(error) => {
                println!("Skipping: GPU context with ray tracing not available: {error:?}");
                return;
            }
        }
    };
    if !context
        .capabilities()
        .ray_query
        .contains(gpu::ShaderVisibility::COMPUTE)
    {
        println!("Skipping: ray_query compute not supported");
        return;
    }

    if !context.capabilities().binding_array {
        println!("Skipping: binding arrays not supported");
        return;
    }

    const OBJECT_COUNT: usize = 64;
    const FRAME_COUNT: usize = 32;
    const WARM_UP_FRAMES: usize = 8;
    const MAX_LATE_GROWTH: u64 = 64 << 20;

    println!(
        "animated BLAS: compiling renderer on {:?}",
        context.device_information()
    );
    let harness = PbrHarness::new(context, "animated-blas-stress", true);
    let context = std::sync::Arc::clone(&harness.context);
    let size = gpu::Extent {
        width: 16,
        height: 16,
        depth: 1,
    };
    let mut pacer = blade_render::util::FramePacer::new(&context);
    let (encoder, temp) = pacer.begin_frame();
    let mut renderer = blade_render::RayTracer::new(
        encoder,
        &context,
        harness.shaders.clone(),
        &harness.asset_hub.shaders,
        &blade_render::RenderConfig {
            surface_size: size,
            surface_info: gpu::SurfaceInfo {
                format: gpu::TextureFormat::Rgba8Unorm,
                alpha: gpu::AlphaMode::Ignored,
            },
            color_space: gpu::ColorSpace::Linear,
            max_debug_lines: 1,
        },
    );
    let make_geometry = || blade_render::ProceduralGeometry {
        name: "triangle".into(),
        vertices: vec![
            blade_render::Vertex {
                position: [-0.25, -0.25, 0.0],
                normal: 0x007f_0000,
                tangent: 0x0000_007f,
                bitangent_sign: 1.0,
                ..Default::default()
            },
            blade_render::Vertex {
                position: [0.25, -0.25, 0.0],
                normal: 0x007f_0000,
                tangent: 0x0000_007f,
                bitangent_sign: 1.0,
                ..Default::default()
            },
            blade_render::Vertex {
                position: [0.0, 0.25, 0.0],
                normal: 0x007f_0000,
                tangent: 0x0000_007f,
                bitangent_sign: 1.0,
                ..Default::default()
            },
        ],
        indices: vec![0, 1, 2],
        ..Default::default()
    };
    // Multiple material primitives on one node still share a rigid pose.
    let procedural_model =
        harness
            .asset_hub
            .models
            .insert(harness.asset_hub.models.baker.create_model(
                "animated-blas-triangle",
                vec![make_geometry(), make_geometry()],
            ));
    // Keep a skinned object on the vertex-skinning/BLAS-refit path as well.
    let (skinned_model, task) = harness.asset_hub.models.load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/animated_skin.glb"),
        blade_render::model::Meta {
            generate_tangents: false,
            front_face: blade_render::model::FrontFace::CounterClockwise,
        },
    );
    if harness.workers.is_empty() {
        task.join_active();
    } else {
        task.join();
    }
    let rigid_model = if profile_vehicle {
        let (model, task) = harness.asset_hub.models.load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("examples/vehicle/data/wheelRacing.glb"),
            blade_render::model::Meta {
                generate_tangents: false,
                front_face: blade_render::model::FrontFace::CounterClockwise,
            },
        );
        if harness.workers.is_empty() {
            task.join_active();
        } else {
            task.join();
        }
        model
    } else {
        procedural_model
    };
    let models = [rigid_model, skinned_model];
    harness.asset_hub.flush(encoder, &mut temp.buffers);
    pacer.end_frame(&context);

    let make_object = |index: usize| {
        let mut object = blade_render::Object::from(
            models[usize::from(index % OBJECT_COUNT == OBJECT_COUNT - 1)],
        );
        let slot = index % OBJECT_COUNT;
        let x = (slot % 8) as f32 - 3.5;
        let y = (slot / 8) as f32 - 3.5;
        object.transform.x.w = x;
        object.transform.y.w = y;
        object.prev_transform = object.transform;
        object
    };
    let mut objects: Vec<_> = (0..OBJECT_COUNT).map(make_object).collect();
    let mut next_object_index = OBJECT_COUNT;
    let camera = blade_render::Camera {
        pos: [0.0, 0.0, 12.0].into(),
        rot: mint::Quaternion {
            v: [0.0; 3].into(),
            s: 1.0,
        },
        fov_y: 0.8,
        depth: 100.0,
        fov: None,
    };
    let ray_config = blade_render::RayConfig {
        num_environment_samples: 1,
        num_brdf_samples: 1,
        jitter_primary_rays: false,
        environment_importance_sampling: false,
        max_bounces: 0,
        t_start: 0.01,
        max_accumulated_samples: 0,
        tap_count: 2,
        tap_radius: 4,
        tap_confidence_near: 4,
        tap_confidence_far: 2,
        pairwise_mis: true,
        defensive_mis: 0.1,
    };
    let mut late_memory_usage = Vec::new();
    let mut profile = std::collections::BTreeMap::<String, f64>::new();
    let frame_loop_start = std::time::Instant::now();

    for frame_index in 0..FRAME_COUNT {
        if frame_index % 8 == 0 {
            println!("animated BLAS: frame {frame_index}");
        }
        let (encoder, temp) = pacer.begin_frame();
        if frame_index != 0 && frame_index % 8 == 0 {
            for mut object in objects.drain(..16) {
                object.retire_gpu(temp);
            }
        } else if frame_index % 8 == 4 {
            while objects.len() < OBJECT_COUNT {
                objects.push(make_object(next_object_index));
                next_object_index += 1;
            }
        }

        let clip = (frame_index / 4) & 1;
        for (index, object) in objects.iter_mut().enumerate() {
            let phase = frame_index as f32 * 0.17 + index as f32 * 0.03;
            let offset = if clip == 0 {
                phase.sin() * 0.1
            } else {
                phase.cos() * 0.1 + 0.25
            };
            object.pose = (frame_index % 8 != 2).then(|| {
                let mut translation =
                    glam::Mat4::from_translation(glam::Vec3::new(0.0, offset, 0.0));
                if profile_vehicle && object.model == rigid_model {
                    translation *= glam::Mat4::from_rotation_x(phase);
                }
                blade_render::Pose::from_node_matrices(if object.model == skinned_model {
                    vec![
                        glam::Mat4::IDENTITY,
                        glam::Mat4::IDENTITY,
                        translation * glam::Mat4::from_translation(glam::Vec3::Y),
                    ]
                } else {
                    vec![translation]
                })
            });
        }

        renderer.build_scene(
            encoder,
            &mut objects,
            None,
            &harness.asset_hub,
            &context,
            temp,
        );
        assert!(temp.acceleration_structures.len() >= usize::from(frame_index != 0));
        renderer.prepare(
            encoder,
            &camera,
            blade_render::FrameConfig {
                reset_variance: frame_index == 0,
                reset_reservoirs: frame_index == 0,
                reset_accumulation: frame_index == 0,
                ..Default::default()
            },
        );
        renderer.render(
            encoder,
            blade_render::RenderMode::RealTime,
            blade_render::DebugConfig::default(),
            ray_config,
            None,
        );
        pacer.end_frame(&context);
        if profile_updates {
            assert!(
                context
                    .wait_for(pacer.last_sync_point().unwrap(), !0)
                    .unwrap()
            );
        }
        if profile_gpu {
            for (name, duration) in pacer.last_timing().pass_durations() {
                *profile.entry(name.to_owned()).or_default() += duration.as_secs_f64() * 1000.0;
            }
        }
        for object in &mut objects {
            object.flip();
        }

        let usage = context.memory_stats().usage;
        if usage != 0 && frame_index >= WARM_UP_FRAMES {
            late_memory_usage.push(usage);
        }
    }

    if profile_updates {
        println!(
            "Animated profile CPU+GPU frame loop: {:.3} ms/frame, vehicle={profile_vehicle}",
            frame_loop_start.elapsed().as_secs_f64() * 1000.0 / FRAME_COUNT as f64
        );
    }
    for (name, millis) in profile {
        println!(
            "Animated profile {name}: {:.3} ms/frame",
            millis / FRAME_COUNT as f64
        );
    }
    if let (Some(min), Some(max)) = (
        late_memory_usage.iter().min(),
        late_memory_usage.iter().max(),
    ) {
        println!(
            "animated BLAS late-frame device memory: {:.1}–{:.1} MiB",
            *min as f64 / (1 << 20) as f64,
            *max as f64 / (1 << 20) as f64,
        );
        assert!(
            max - min <= MAX_LATE_GROWTH,
            "animated BLAS memory kept growing after warm-up: {} MiB",
            (max - min) >> 20,
        );
    }

    pacer.destroy(&context);
    for object in objects.iter_mut() {
        object.destroy_gpu(&context);
    }
    renderer.destroy(&context);
    harness.destroy();
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

/// ReSTIR and a canonical path trace must agree on direct reflected energy in
/// linear HDR space.  A display-space snapshot can hide a dark estimator
/// behind tone mapping, and a multi-bounce reference legitimately contains
/// indirect transport the real-time pass does not attempt.
#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn restir_direct_energy_matches_canonical() {
    let base_config = RayTraceTestConfig {
        // No secondary surface shading: this is exactly the environment
        // transport ReSTIR estimates, without indirect fill.
        max_bounces: 0,
        pairwise_mis: false,
        // Blade does not yet explicitly sample emissive geometry. Removing
        // that row isolates the common environment-light estimator here; a
        // separate scene regression exercises first-hit emission.
        include_emissive_row: false,
        emissive_scale: 1.0,
        black_environment: false,
        tap_count: 0,
        denoise: true,
        frozen_frames: 0,
    };
    let Some(reference_bytes) = render_ray_traced_grid_configured(
        "pbr-canonical-direct-energy",
        RayTraceMode::Canonical,
        Capture::Hdr,
        base_config,
        |_, _, _| {},
    ) else {
        return;
    };
    let reference: &[f32] = bytemuck::cast_slice(&reference_bytes);
    let luminance = |rgba: &[f32]| 0.3 * rgba[0] + 0.4 * rgba[1] + 0.3 * rgba[2];

    // Pin both the canonical reservoir math and the production pairwise reuse
    // path. The cheaper non-pairwise spatial approximation is intentionally
    // excluded: different neighbor target distributions require MIS weights
    // and otherwise lose roughly ten percent of the energy in this scene.
    for (label, pairwise_mis, tap_count) in [("no reuse", false, 0), ("pairwise reuse", true, 2)] {
        let config = RayTraceTestConfig {
            pairwise_mis,
            tap_count,
            ..base_config
        };
        let Some(restir_bytes) = render_ray_traced_grid_configured(
            &format!("pbr-restir-direct-energy-{}", label.replace(' ', "-")),
            RayTraceMode::Restir,
            Capture::Hdr,
            config,
            |_, _, _| {},
        ) else {
            return;
        };
        let restir: &[f32] = bytemuck::cast_slice(&restir_bytes);
        assert_eq!(restir.len(), reference.len());

        let mut restir_sum = 0.0f64;
        let mut reference_sum = 0.0f64;
        let mut compared = 0usize;
        let mut unexpectedly_black = 0usize;
        for (actual, expected) in restir
            .as_chunks::<4>()
            .0
            .iter()
            .zip(reference.as_chunks::<4>().0.iter())
        {
            let expected_luma = luminance(expected);
            // The dummy environment is exactly white. Select the ordinary
            // shaded material range, excluding that background.
            if !(0.05..0.95).contains(&expected_luma) {
                continue;
            }
            let actual_luma = luminance(actual);
            restir_sum += actual_luma as f64;
            reference_sum += expected_luma as f64;
            compared += 1;
            unexpectedly_black += (expected_luma > 0.15 && actual_luma < 0.02) as usize;
        }
        assert!(compared > 1_000, "only {compared} shaded pixels selected");
        let energy_ratio = restir_sum / reference_sum;
        let black_fraction = unexpectedly_black as f64 / compared as f64;
        println!(
            "ReSTIR {label}: energy ratio {energy_ratio:.4}, black pixels {unexpectedly_black}/{compared} ({:.2}%)",
            100.0 * black_fraction,
        );
        assert!(
            (0.95..1.05).contains(&energy_ratio),
            "ReSTIR {label} carries {energy_ratio:.3}x the matched canonical direct energy"
        );
        assert!(
            black_fraction < 0.02,
            "{:.2}% of canonically lit pixels became black with {label}",
            100.0 * black_fraction,
        );
    }
}

/// An emissive mesh hit by a ReSTIR candidate is a light source, not a black
/// occluder. This scene has no environment radiance, so any light on the
/// ordinary material rows must have arrived from the emissive row.
#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn restir_receives_emissive_geometry() {
    let config = RayTraceTestConfig {
        max_bounces: 1,
        pairwise_mis: true,
        include_emissive_row: true,
        emissive_scale: 20.0,
        black_environment: true,
        tap_count: 2,
        denoise: true,
        frozen_frames: 0,
    };
    let Some(bytes) = render_ray_traced_grid_configured(
        "pbr-restir-emissive-energy",
        RayTraceMode::Restir,
        Capture::Hdr,
        config,
        |_, _, _| {},
    ) else {
        return;
    };
    let pixels: &[f32] = bytemuck::cast_slice(&bytes);
    let mut energy = 0.0f64;
    let mut peak = 0.0f32;
    let mut peak_pixel = [0, 0];
    // The emissive row reaches to roughly y=129. Stop at y=120 so neither
    // those pixels nor the seven-pixel radius of the three SVGF passes can be
    // mistaken for light received by the ordinary rows.
    let ordinary_rows_end = RAY_TRACE_SIZE.height * 5 / 8;
    for y in 0..ordinary_rows_end {
        for x in 0..RAY_TRACE_SIZE.width {
            let offset = ((y * RAY_TRACE_SIZE.width + x) * 4) as usize;
            let texel = &pixels[offset..offset + 4];
            let luma = 0.3 * texel[0] + 0.4 * texel[1] + 0.3 * texel[2];
            energy += luma as f64;
            if luma > peak {
                peak = luma;
                peak_pixel = [x, y];
            }
        }
    }
    let mean = energy / (ordinary_rows_end * RAY_TRACE_SIZE.width) as f64;
    println!("ReSTIR emissive geometry: mean {mean:.6}, peak {peak:.3} at {peak_pixel:?}");
    assert!(
        mean > 1.0e-4 && peak > 0.01,
        "ordinary surfaces received no measurable emissive-mesh light"
    );
}

/// The spatial filter must redistribute radiance, not destroy it. Its old
/// centre-normalized bilateral weights were asymmetric: a noisy bright pixel
/// accepted a dark neighbour that rejected it in return. Three passes lost
/// visible energy even when the ReSTIR estimate entering SVGF was correct.
#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn svgf_preserves_restir_energy() {
    let base = RayTraceTestConfig {
        max_bounces: 0,
        pairwise_mis: true,
        include_emissive_row: false,
        emissive_scale: 1.0,
        black_environment: false,
        tap_count: 2,
        denoise: false,
        frozen_frames: 0,
    };
    let Some(reference_bytes) = render_ray_traced_grid_configured(
        "pbr-restir-svgf-energy-reference",
        RayTraceMode::Canonical,
        Capture::Hdr,
        RayTraceTestConfig {
            denoise: true,
            ..base
        },
        |_, _, _| {},
    ) else {
        return;
    };
    let Some(raw_bytes) = render_ray_traced_grid_configured(
        "pbr-restir-svgf-energy-raw",
        RayTraceMode::Restir,
        Capture::Hdr,
        base,
        |_, _, _| {},
    ) else {
        return;
    };
    let Some(filtered_bytes) = render_ray_traced_grid_configured(
        "pbr-restir-svgf-energy-filtered",
        RayTraceMode::Restir,
        Capture::Hdr,
        RayTraceTestConfig {
            denoise: true,
            ..base
        },
        |_, _, _| {},
    ) else {
        return;
    };
    let raw: &[f32] = bytemuck::cast_slice(&raw_bytes);
    let filtered: &[f32] = bytemuck::cast_slice(&filtered_bytes);
    assert_eq!(raw.len(), filtered.len());
    let reference: &[f32] = bytemuck::cast_slice(&reference_bytes);
    assert_eq!(raw.len(), reference.len());

    // The white background is exactly one. Select canonically shaded sphere
    // interiors, excluding both background and near-black silhouette pixels;
    // the same mask is applied to raw and filtered ReSTIR.
    let luminance = |rgba: &[f32]| 0.3 * rgba[0] + 0.4 * rgba[1] + 0.3 * rgba[2];
    let energy = |pixels: &[f32]| -> (f64, usize) {
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .zip(reference.as_chunks::<4>().0.iter())
            .filter_map(|(actual, expected)| {
                (0.05..0.95)
                    .contains(&luminance(expected))
                    .then_some(luminance(actual) as f64)
            })
            .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1))
    };
    let (raw_energy, raw_values) = energy(raw);
    let (filtered_energy, filtered_values) = energy(filtered);
    assert_eq!(raw_values, filtered_values);
    assert!(
        raw_values > 1_000,
        "only {raw_values} shaded pixels selected"
    );
    assert!(raw_energy > 0.0, "the regression scene carried no light");
    let ratio = filtered_energy / raw_energy;
    println!("SVGF / raw ReSTIR received energy: {ratio:.4}");
    assert!(
        (0.98..1.02).contains(&ratio),
        "SVGF changed received radiance energy by {:.1}%",
        (ratio - 1.0) * 100.0,
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
    for texel in pixels.as_chunks::<4>().0.iter() {
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
    for (hdr, shown) in pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(display.as_chunks::<4>().0.iter())
    {
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
    let hits = probe
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|t| t[0] > 0.0)
        .count();
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
    for texel in probe.as_chunks::<4>().0.iter().filter(|t| t[0] > 0.0) {
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
        probe
            .as_chunks::<4>()
            .0
            .iter()
            .all(|t| (0.0..=1.0).contains(&t[2])),
        "albedo left the unit range"
    );

    // The spheres carry no normal map, so the shading normal should agree with
    // the geometric one almost everywhere a ray hit.
    let aligned = probe
        .as_chunks::<4>()
        .0
        .iter()
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
    if harness.workers.is_empty() {
        task.join_active();
    } else {
        task.join();
    }

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

#[cfg(not(gles))]
#[test]
#[ignore = "requires a working GPU context with binding arrays"]
fn binding_arrays_gpu_test() {
    #[derive(blade_macros::ShaderData)]
    struct ArrayData<'a> {
        textures: &'a gpu::TextureArray<8>,
        buffers: &'a gpu::BufferArray<8>,
        output: gpu::BufferPiece,
        samp: gpu::Sampler,
    }

    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    if !context.capabilities().binding_array {
        println!("Skipping: binding arrays are unsupported");
        return;
    }
    let shader = context.create_shader(gpu::ShaderDesc {
        source: include_str!("shaders/binding_arrays.wgsl"),
        naga_module: None,
    });
    // A device-specific layout can be smaller than the Rust allocation ceiling.
    let mut layout = ArrayData::layout();
    for (_, binding) in &mut layout.bindings {
        match binding {
            gpu::ShaderBinding::TextureArray { count }
            | gpu::ShaderBinding::BufferArray { count } => *count = 4,
            _ => {}
        }
    }
    let mut compute = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "binding-arrays",
        data_layouts: &[&layout],
        compute: shader.at("cs_main"),
    });
    let format = gpu::TextureFormat::Rgba8Unorm;
    let target = snapshot::OffscreenTarget::new(
        &context,
        gpu::Extent {
            width: 4,
            height: 1,
            depth: 1,
        },
        format,
    );
    let mut render = context.create_render_pipeline(gpu::RenderPipelineDesc {
        name: "binding-arrays",
        data_layouts: &[&layout],
        vertex: shader.at("vs_main"),
        vertex_fetches: &[],
        fragment: Some(shader.at("fs_main")),
        primitive: gpu::PrimitiveState::default(),
        depth_stencil: None,
        color_targets: &[format.into()],
        multisample_state: gpu::MultisampleState::default(),
    });
    let sampler = context.create_sampler(gpu::SamplerDesc::default());
    let input = context.create_buffer(gpu::BufferDesc {
        name: "binding-array-input",
        size: 64,
        memory: gpu::Memory::Shared,
    });
    let output = context.create_buffer(gpu::BufferDesc {
        name: "binding-array-output",
        size: 128,
        memory: gpu::Memory::Shared,
    });
    unsafe {
        let values = slice::from_raw_parts_mut(input.data().cast::<f32>(), 16);
        values.fill(-100.0);
        values[4] = 3.0;
        values[8] = 7.0;
    }
    context.sync_buffer(input.into(), input.size(), gpu::BufferTarget::Data);
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "binding-arrays",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    encoder.init_texture(target.texture);
    let (green, green_view, green_stage) =
        make_solid_texture("array-green", [0, 255, 0, 255], &mut encoder, &context);
    let (blue, blue_view, blue_stage) =
        make_solid_texture("array-blue", [0, 0, 255, 255], &mut encoder, &context);
    let mut textures = gpu::TextureArray::new();
    textures.alloc(green_view);
    textures.alloc(blue_view);
    let mut reversed = gpu::TextureArray::new();
    reversed.alloc(blue_view);
    reversed.alloc(green_view);
    let mut buffers = gpu::BufferArray::new();
    buffers.alloc(input.at(16));
    buffers.alloc(input.at(32));
    {
        let mut pass = encoder.compute("binding-arrays");
        let mut pc = pass.with(&compute);
        for (index, array) in [&textures, &reversed].into_iter().enumerate() {
            pc.bind(
                0,
                &ArrayData {
                    textures: array,
                    buffers: &buffers,
                    output: output.at(index as u64 * 64),
                    samp: sampler,
                },
            );
            pc.dispatch([1, 1, 1]);
        }
    }
    {
        let mut pass = encoder.render(
            "binding-arrays",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target.view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        );
        let mut pc = pass.with(&render);
        pc.bind(
            0,
            &ArrayData {
                textures: &textures,
                buffers: &buffers,
                output: output.into(),
                samp: sampler,
            },
        );
        pc.draw(0, 3, 0, 1);
    }
    let pixels = target.read_pixels(&context, &mut encoder);
    let actual = unsafe { slice::from_raw_parts(output.data().cast::<[f32; 4]>(), 8) };
    assert_eq!(
        actual,
        &[
            [3.0, 4.0, 3.0, 4.0],
            [7.0, 7.0, 8.0, 8.0],
            [3.0, 4.0, 3.0, 4.0],
            [7.0, 7.0, 8.0, 8.0],
            [3.0, 3.0, 4.0, 4.0],
            [7.0, 8.0, 7.0, 8.0],
            [3.0, 3.0, 4.0, 4.0],
            [7.0, 8.0, 7.0, 8.0],
        ]
    );
    for (index, pixel) in pixels.as_chunks::<4>().0.iter().enumerate() {
        let expected = if index % 2 == 0 {
            [0, 255, 64, 255]
        } else {
            [0, 0, 255, 255]
        };
        for (&actual, expected) in pixel.iter().zip(expected) {
            assert!(
                (actual as i32 - expected).abs() <= 1,
                "pixel {index}: {pixel:?}"
            );
        }
    }

    context.destroy_compute_pipeline(&mut compute);
    context.destroy_render_pipeline(&mut render);
    context.destroy_command_encoder(&mut encoder);
    context.destroy_sampler(sampler);
    for buffer in [input, output, green_stage, blue_stage] {
        context.destroy_buffer(buffer);
    }
    for view in [green_view, blue_view] {
        context.destroy_texture_view(view);
    }
    for texture in [green, blue] {
        context.destroy_texture(texture);
    }
    target.destroy(&context);
}
