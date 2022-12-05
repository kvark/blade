#![allow(irrefutable_let_patterns)]

use std::num::NonZeroU32;

#[derive(blade::ShaderData)]
struct ShaderData {
    i1: i32,
    i4: [i32; 4],
    u2: [u32; 2],
    f1: f32,
    f4: [f32; 4],
    #[struct_name("StorageBuffer")]
    buffer: blade::BufferPiece,
    tex_1d: blade::TextureView,
    tex_1d_array: blade::TextureView,
    tex_2d: blade::TextureView,
    tex_2d_array: blade::TextureView,
    tex_cube: blade::TextureView,
    tex_cube_array: blade::TextureView,
    tex_3d: blade::TextureView,
    tex_depth: blade::TextureView,
    sam: blade::Sampler,
    sam_comparison: blade::Sampler,
}

const DIMENSIONS: &[(
    blade::TextureDimension,
    blade::Extent,
    u32,
    &[(blade::TextureViewDimension, u32)],
)] = &[
    (
        blade::TextureDimension::D1,
        blade::Extent {
            width: 4,
            height: 1,
            depth: 1,
        },
        3,
        &[
            (blade::TextureViewDimension::D1, 1),
            (blade::TextureViewDimension::D1Array, 1),
        ],
    ),
    (
        blade::TextureDimension::D2,
        blade::Extent {
            width: 4,
            height: 4,
            depth: 1,
        },
        12,
        &[
            (blade::TextureViewDimension::D2, 1),
            (blade::TextureViewDimension::D2Array, 1),
            (blade::TextureViewDimension::Cube, 6),
            (blade::TextureViewDimension::CubeArray, 6),
        ],
    ),
    (
        blade::TextureDimension::D3,
        blade::Extent {
            width: 4,
            height: 4,
            depth: 4,
        },
        1,
        &[(blade::TextureViewDimension::D3, 1)],
    ),
];

#[test]
fn main() {
    env_logger::init();
    let context = match unsafe {
        blade::Context::init(blade::ContextDesc {
            validation: true,
            capture: true,
        })
    } {
        Ok(context) => context,
        Err(e) => {
            println!("GPU not supported: {:?}", e);
            return;
        }
    };

    let data_layout = <ShaderData as blade::ShaderData>::layout();
    let shader_source = std::fs::read_to_string("tests/bindings/shader.wgsl").unwrap();
    let shader = context.create_shader(blade::ShaderDesc {
        source: &shader_source,
        data_layouts: &[&data_layout],
    });

    let pipeline = context.create_compute_pipeline(blade::ComputePipelineDesc {
        name: "main",
        compute: shader.at("main"),
    });
    let wg_size = pipeline.get_workgroup_size();
    assert_eq!(wg_size, [1, 2, 3]);

    let mut command_encoder = context.create_command_encoder(blade::CommandEncoderDesc {
        name: "main",
        buffer_count: 1,
    });
    command_encoder.start();

    let mut textures = Vec::new();
    let mut texture_views = Vec::new();
    for &(dimension, extent, array_layer_count, view_descs) in DIMENSIONS {
        let format = blade::TextureFormat::Rgba8Unorm;
        let texture = context.create_texture(blade::TextureDesc {
            name: "",
            format,
            size: extent,
            dimension,
            array_layer_count,
            mip_level_count: 1,
            usage: blade::TextureUsage::RESOURCE,
        });
        command_encoder.init_texture(texture);
        textures.push(texture);

        for &(view_dimension, view_layers) in view_descs {
            let view = context.create_texture_view(blade::TextureViewDesc {
                name: "",
                texture,
                format,
                dimension: view_dimension,
                subresources: &blade::TextureSubresources {
                    array_layer_count: NonZeroU32::new(view_layers),
                    ..Default::default()
                },
            });
            texture_views.push(view);
        }
    }

    let depth_format = blade::TextureFormat::Depth32Float;
    let depth_texture = context.create_texture(blade::TextureDesc {
        name: "depth",
        format: depth_format,
        size: blade::Extent {
            width: 4,
            height: 4,
            depth: 1,
        },
        dimension: blade::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: blade::TextureUsage::RESOURCE,
    });
    let depth_texture_view = context.create_texture_view(blade::TextureViewDesc {
        name: "depth view",
        texture: depth_texture,
        format: depth_format,
        dimension: blade::TextureViewDimension::D2,
        subresources: &blade::TextureSubresources::default(),
    });

    let sampler = context.create_sampler(blade::SamplerDesc {
        ..Default::default()
    });
    let sampler_comparison = context.create_sampler(blade::SamplerDesc {
        compare: Some(blade::CompareFunction::Never),
        ..Default::default()
    });

    let buffer = context.create_buffer(blade::BufferDesc {
        name: "",
        size: 4,
        memory: blade::Memory::Device,
    });

    if let mut compute = command_encoder.compute() {
        if let mut pc = compute.with(&pipeline) {
            pc.bind(
                0,
                &ShaderData {
                    i1: 0,
                    i4: [1; 4],
                    u2: [2; 2],
                    f1: 0.3,
                    f4: [0.4; 4],
                    buffer: buffer.into(),
                    tex_1d: texture_views[0],
                    tex_1d_array: texture_views[1],
                    tex_2d: texture_views[2],
                    tex_2d_array: texture_views[3],
                    tex_cube: texture_views[4],
                    tex_cube_array: texture_views[5],
                    tex_3d: texture_views[6],
                    tex_depth: depth_texture_view,
                    sam: sampler,
                    sam_comparison: sampler_comparison,
                },
            );
            pc.dispatch([1; 3]);
        }
    }
    let sync_point = context.submit(&mut command_encoder);

    let ok = context.wait_for(sync_point, 1000);
    assert!(ok);

    context.destroy_command_encoder(command_encoder);
    context.destroy_buffer(buffer);
    for view in texture_views {
        context.destroy_texture_view(view);
    }
    context.destroy_texture_view(depth_texture_view);
    for texture in textures {
        context.destroy_texture(texture);
    }
    context.destroy_texture(depth_texture);
}
