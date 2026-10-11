#![cfg(all(target_os = "linux", not(gles)))]

use blade_graphics as gpu;
use gpu::{ShaderBindable, ShaderData};

const CASES: usize = 4;
const K: usize = 64;
const MATRIX_INPUTS: usize = 2 * 16 * K;
const SOURCE: &str = r#"
enable wgpu_bfloat16;
enable wgpu_cooperative_matrix;
var<storage> src: array<f32>;
var<storage, read_write> packed: array<bf16>;
var<storage, read_write> decoded: array<f32>;
var<storage, read_write> product: array<f32>;

@compute @workgroup_size(32)
fn convert(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&src) {
        let value = bf16(src[id.x]);
        packed[id.x] = value;
        decoded[id.x] = f32(value);
    }
}

@compute @workgroup_size(32)
fn multiply(@builtin(workgroup_id) group: vec3<u32>) {
    let base = group.x * 2048u;
    var acc = coop_mat16x16<f32, C>();
    for (var k = 0u; k < 64u; k += 16u) {
        let a = coopLoadT<coop_mat16x16<bf16, A>>(&packed[base + k], 64u);
        let b = coopLoadT<coop_mat16x16<bf16, B>>(&packed[base + 1024u + k * 16u], 16u);
        acc = coopMultiplyAdd(a, b, acc);
    }
    coopStoreT(acc, &product[group.x * 256u], 16u);
}
"#;

struct Data([gpu::BufferPiece; 4]);

impl ShaderData for Data {
    fn layout() -> gpu::ShaderDataLayout {
        gpu::ShaderDataLayout {
            bindings: ["src", "packed", "decoded", "product"]
                .map(|name| (name, gpu::ShaderBinding::Buffer))
                .to_vec(),
        }
    }

    fn fill(&self, mut context: gpu::PipelineContext) {
        for (index, buffer) in self.0.iter().enumerate() {
            buffer.bind_to(&mut context, index as u32);
        }
    }
}

fn nearest_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[test]
#[ignore = "requires native BF16 cooperative matrices and a guarded GPU allocation"]
fn bfloat16_storage_and_cooperative_products() {
    let expected_device = std::env::var("BLADE_TEST_DEVICE_ID")
        .ok()
        .map(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).unwrap());
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            device_id: expected_device,
            validation: true,
            ..Default::default()
        })
    }
    .unwrap();
    let device = context.device_information();
    assert!(!device.is_software_emulated);
    if let Some(id) = expected_device {
        assert!(context.enumerate_devices().iter().any(|report| {
            report.device_id == id
                && matches!(
                    report.status,
                    gpu::DeviceReportStatus::Available {
                        is_default: true,
                        ..
                    }
                )
        }));
    }
    let caps = context.capabilities();
    assert!(caps.shader_bfloat16);
    assert!(
        caps.cooperative_matrix
            .bf16_f32_shapes
            .contains(&[16, 16, 16])
    );
    println!("device={device:?} capabilities={caps:?}");
    let memory = context.memory_stats();
    assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    println!("initial_memory={memory:?}");

    let mut input = vec![0.0f32; CASES * MATRIX_INPUTS];
    for case in 0..CASES {
        let (a_scale, b_scale) = match case {
            1 => (1e20, 1e-20),
            2 => (1e-20, 1e20),
            _ => (1.0, 1.0),
        };
        for i in 0..16 * K {
            input[case * MATRIX_INPUTS + i] = (((i * 37 + 11) % 53) as f32 - 26.0) / 17.0 * a_scale;
            input[case * MATRIX_INPUTS + 16 * K + i] =
                (((i * 23 + 7) % 47) as f32 - 23.0) / 13.0 * b_scale;
        }
    }
    for k in 0..K {
        for row in 0..16 {
            input[3 * MATRIX_INPUTS + row * K + k] = if k % 2 == 0 { 1.0 } else { -1.0 };
        }
        for col in 0..16 {
            input[3 * MATRIX_INPUTS + 16 * K + k * 16 + col] = 1.0 + col as f32 / 128.0;
        }
    }
    input.extend([
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.0 + 1.0 / 256.0,
        1.0 + 3.0 / 256.0,
        -1.0 - 1.0 / 256.0,
        -1.0 - 3.0 / 256.0,
        1e-30,
        -1e-30,
        1e30,
        -1e30,
        3e38,
        -3e38,
        f32::MIN_POSITIVE,
    ]);
    let sizes = [
        input.len() * 4,
        input.len() * 2,
        input.len() * 4,
        CASES * 256 * 4,
    ];
    let buffers = sizes.map(|size| {
        context.create_buffer(gpu::BufferDesc {
            name: "bf16-device",
            size: size as u64,
            memory: gpu::Memory::Device,
        })
    });
    let upload = context.create_buffer(gpu::BufferDesc {
        name: "bf16-upload",
        size: sizes[0] as u64,
        memory: gpu::Memory::Upload,
    });
    let downloads = [sizes[1], sizes[2], sizes[3]].map(|size| {
        context.create_buffer(gpu::BufferDesc {
            name: "bf16-download",
            size: size as u64,
            memory: gpu::Memory::Download,
        })
    });
    unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), upload.data().cast(), input.len()) };
    let shader = context.create_shader(gpu::ShaderDesc {
        source: SOURCE,
        naga_module: None,
    });
    let layout = Data::layout();
    let mut pipelines = ["convert", "multiply"].map(|entry| {
        context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: entry,
            data_layouts: &[&layout],
            compute: shader.at(entry),
        })
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "bf16",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    encoder.transfer("upload").copy_buffer_to_buffer(
        upload.at(0),
        buffers[0].at(0),
        sizes[0] as u64,
    );
    {
        let mut compute = encoder.compute("convert");
        let mut pass = compute.with(&pipelines[0]);
        pass.bind(0, &Data(buffers.map(|buffer| buffer.at(0))));
        pass.dispatch([input.len().div_ceil(32) as u32, 1, 1]);
    }
    for _ in 0..8 {
        let mut compute = encoder.compute("multiply");
        let mut pass = compute.with(&pipelines[1]);
        pass.bind(0, &Data(buffers.map(|buffer| buffer.at(0))));
        pass.dispatch([CASES as u32, 1, 1]);
    }
    for (index, download) in downloads.iter().enumerate() {
        encoder.transfer("download").copy_buffer_to_buffer(
            buffers[index + 1].at(0),
            download.at(0),
            sizes[index + 1] as u64,
        );
    }
    let sync = context.submit(&mut encoder);
    assert!(context.wait_for(&sync, 10_000).unwrap());
    let packed =
        unsafe { std::slice::from_raw_parts(downloads[0].data().cast::<u16>(), input.len()) };
    let decoded =
        unsafe { std::slice::from_raw_parts(downloads[1].data().cast::<f32>(), input.len()) };
    let products =
        unsafe { std::slice::from_raw_parts(downloads[2].data().cast::<f32>(), CASES * 256) };
    let rte = input
        .iter()
        .zip(packed)
        .all(|(&src, &dst)| nearest_bf16(src) == dst);
    let rtz = input
        .iter()
        .zip(packed)
        .all(|(&src, &dst)| (src.to_bits() >> 16) as u16 == dst);
    assert!(
        rte || rtz,
        "conversion must follow a consistent supported rounding mode"
    );
    for (&bits, value) in packed.iter().zip(decoded) {
        assert_eq!(u32::from(bits) << 16, value.to_bits());
    }
    let mut error_squared = 0.0;
    let mut reference_squared = 0.0;
    let mut maximum_error = 0.0f64;
    for case in 0..CASES {
        for row in 0..16 {
            for col in 0..16 {
                let terms: Vec<_> = (0..K)
                    .map(|k| {
                        f64::from(decoded[case * MATRIX_INPUTS + row * K + k])
                            * f64::from(decoded[case * MATRIX_INPUTS + 16 * K + k * 16 + col])
                    })
                    .collect();
                let expected: f64 = terms.iter().sum();
                let scale: f64 = terms.iter().map(|value| value.abs()).sum();
                let actual = f64::from(products[case * 256 + row * 16 + col]);
                let error = (actual - expected).abs();
                assert!(
                    actual.is_finite() && error <= 1e-6 + 2e-5 * scale,
                    "case={case} row={row} col={col}: {actual} != {expected}"
                );
                maximum_error = maximum_error.max(error);
                error_squared += error * error;
                reference_squared += expected * expected;
            }
        }
    }
    let relative_l2 = (error_squared / reference_squared).sqrt();
    assert!(relative_l2 < 5e-6);
    let memory = context.memory_stats();
    assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    println!(
        "bf16_verified conversions={} products={} repetitions=8 rounding={} relative_l2={relative_l2:.9e} max_error={maximum_error:.9e} final_memory={memory:?}",
        input.len(),
        CASES * 256,
        if rte { "nearest_even" } else { "toward_zero" }
    );
    context.destroy_command_encoder(&mut encoder);
    for pipeline in &mut pipelines {
        context.destroy_compute_pipeline(pipeline);
    }
    for buffer in buffers.into_iter().chain(downloads).chain([upload]) {
        context.destroy_buffer(buffer);
    }
}
