use blade_graphics as gpu;
use gpu::ShaderData as _;

#[derive(blade_macros::ShaderData)]
struct Data {
    a: gpu::BufferPiece,
    b: gpu::BufferPiece,
    c: gpu::BufferPiece,
}

fn run(context: &gpu::Context, m: usize, n: usize, k: usize, layout: u32, fp16: bool) {
    let col = [layout & 1 != 0, layout & 2 != 0, layout & 4 != 0];
    let stride = [
        if col[0] { m } else { k },
        if col[1] { k } else { n },
        if col[2] { m } else { n },
    ]
    .map(|v| v + 16);
    let outer = [
        if col[0] { k } else { m },
        if col[1] { n } else { k },
        if col[2] { n } else { m },
    ];
    let index = |which: usize, row: usize, column: usize| {
        16 + if col[which] {
            column * stride[which] + row
        } else {
            row * stride[which] + column
        }
    };
    let mut arrays: [Vec<f32>; 3] = std::array::from_fn(|i| vec![123.0; 32 + stride[i] * outer[i]]);
    for row in 0..m {
        for p in 0..k {
            arrays[0][index(0, row, p)] = ((row * 17 + p * 13 + 3) % 31) as f32 * 0.125 - 1.875;
        }
    }
    for p in 0..k {
        for column in 0..n {
            arrays[1][index(1, p, column)] = ((p * 7 + column * 19 + 5) % 29) as f32 * 0.125 - 1.75;
        }
    }
    for row in 0..m {
        for column in 0..n {
            arrays[2][index(2, row, column)] = ((row * 3 + column * 11) % 17) as f32 * 0.25 - 2.0;
        }
    }
    let mut expected = arrays[2].clone();
    for row in 0..m {
        for column in 0..n {
            let mut product = 0.0;
            for p in 0..k {
                product += arrays[0][index(0, row, p)] * arrays[1][index(1, p, column)];
            }
            expected[index(2, row, column)] += 2.0 * product;
        }
    }
    let bytes: [Vec<u8>; 3] = std::array::from_fn(|i| {
        if fp16 && i < 2 {
            let half: Vec<_> = arrays[i].iter().map(|&v| half::f16::from_f32(v)).collect();
            bytemuck::cast_slice(&half).to_vec()
        } else {
            bytemuck::cast_slice(&arrays[i]).to_vec()
        }
    });
    let make = |i: usize, memory| {
        context.create_buffer(gpu::BufferDesc {
            name: "matrix",
            size: bytes[i].len() as u64,
            memory,
        })
    };
    let upload: [_; 3] = std::array::from_fn(|i| make(i, gpu::Memory::Upload));
    let device: [_; 3] = std::array::from_fn(|i| make(i, gpu::Memory::Device));
    let download = make(2, gpu::Memory::Download);
    for i in 0..3 {
        unsafe {
            std::ptr::copy_nonoverlapping(bytes[i].as_ptr(), upload[i].data(), bytes[i].len())
        };
    }
    let scalar = if fp16 { "f16" } else { "f32" };
    let call = |i: usize, store: bool| {
        format!(
            "coop{}{}",
            if store { "Store" } else { "Load" },
            if col[i] { "" } else { "T" }
        )
    };
    let address = |i: usize, row: &str, column: &str| {
        format!(
            "16u + ({}) * {}u + ({})",
            if col[i] { column } else { row },
            stride[i],
            if col[i] { row } else { column }
        )
    };
    let source = format!(
        r#"
enable wgpu_cooperative_matrix;
{}
var<storage, read> a: array<{scalar}>;
var<storage, read> b: array<{scalar}>;
var<storage, read_write> c: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {{
    let row = wg.x * 16u;
    let column = wg.y * 16u;
    var acc = {}<coop_mat16x16<f32, C>>(&c[{}], {}u);
    for (var p = 0u; p < {k}u; p += 16u) {{
        let x = {}<coop_mat16x16<{scalar}, A>>(&a[{}], {}u);
        let y = {}<coop_mat16x16<{scalar}, B>>(&b[{}], {}u);
        acc = coopMultiplyAdd(x, y, acc);
    }}
    {}(acc, &c[{}], {}u);
}}
"#,
        if fp16 { "enable f16;" } else { "" },
        call(2, false),
        address(2, "row", "column"),
        stride[2],
        call(0, false),
        address(0, "row", "p"),
        stride[0],
        call(1, false),
        address(1, "p", "column"),
        stride[1],
        call(2, true),
        address(2, "row", "column"),
        stride[2]
    );
    let shader = context.create_shader(gpu::ShaderDesc {
        source: &source,
        naga_module: None,
    });
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "cooperative-smoke",
        data_layouts: &[&Data::layout()],
        compute: shader.at("main"),
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "cooperative-smoke",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    {
        let mut pass = encoder.transfer("upload");
        for i in 0..3 {
            pass.copy_buffer_to_buffer(upload[i].into(), device[i].into(), bytes[i].len() as u64);
        }
    }
    for _ in 0..2 {
        let mut pass = encoder.compute("matrix");
        let mut command = pass.with(&pipeline);
        command.bind(
            0,
            &Data {
                a: device[0].into(),
                b: device[1].into(),
                c: device[2].into(),
            },
        );
        command.dispatch([m as u32 / 16, n as u32 / 16, 1]);
    }
    encoder.transfer("readback").copy_buffer_to_buffer(
        device[2].into(),
        download.into(),
        bytes[2].len() as u64,
    );
    let sync = context.submit(&mut encoder);
    assert!(context.wait_for(&sync, 10_000).unwrap());
    let result =
        unsafe { std::slice::from_raw_parts(download.data().cast::<f32>(), expected.len()) };
    let mismatch = result
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(i, (&a, &b))| (i, a, b));
    context.destroy_command_encoder(&mut encoder);
    context.destroy_compute_pipeline(&mut pipeline);
    for buffer in upload.into_iter().chain(device).chain([download]) {
        context.destroy_buffer(buffer);
    }
    assert_eq!(
        mismatch, None,
        "{scalar} {m}x{n}x{k} layout={layout}: index, actual, expected"
    );
    println!(
        "PASS: {scalar} {m}x{n}x{k} layout={layout}, nonzero C, two dispatches, padding intact"
    );
}

fn main() {
    env_logger::init();
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            device_id: Some(0x74b5),
            validation: true,
            ..Default::default()
        })
    }
    .unwrap();
    assert!(!context.device_information().is_software_emulated);
    let caps = context.capabilities();
    assert!(
        caps.cooperative_matrix
            .f16_f32_shapes
            .contains(&[16, 16, 16])
    );
    assert!(caps.cooperative_matrix.f32_shapes.contains(&[16, 16, 16]));
    println!("Device: {}", context.device_information().device_name);
    let repetitions: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(1);
    for _ in 0..repetitions {
        for fp16 in [true, false] {
            for (m, n, k) in [(16, 16, 16), (32, 48, 64), (64, 64, 128)] {
                for layout in 0..8 {
                    run(&context, m, n, k, layout, fp16);
                }
            }
        }
    }
}
