//! Headless compute correctness checks, including upload and download through device memory.
//! Run `scripts/mi300-vulkan cargo run --example compute-smoke` on the MI300 bring-up machine.
use blade_graphics as gpu;
use gpu::ShaderData as _;

const COUNT: usize = 65_536;
const WIDTH: u32 = 128;

#[derive(blade_macros::ShaderData)]
struct Data {
    input: gpu::BufferPiece,
    output: gpu::BufferPiece,
    params: [u32; 4],
}

fn run(context: &gpu::Context, name: &str, body: &str, expected: &[u32], indirect: bool) {
    let dimensions = if name == "3d" || name == "3d-indirect" {
        [16, 4, 8]
    } else {
        [COUNT as u32 / WIDTH, 1, 1]
    };
    let size = (COUNT * 4) as u64;
    let allocate = |name, memory| context.create_buffer(gpu::BufferDesc { name, size, memory });
    let upload = allocate("upload", gpu::Memory::Upload);
    let input = allocate("input", gpu::Memory::Device);
    let output = allocate("output", gpu::Memory::Device);
    let download = allocate("download", gpu::Memory::Download);
    let indirect_args = context.create_buffer(gpu::BufferDesc {
        name: "indirect",
        size: 12,
        memory: gpu::Memory::Shared,
    });
    unsafe {
        let words = std::slice::from_raw_parts_mut(upload.data().cast::<u32>(), COUNT);
        for (i, word) in words.iter_mut().enumerate() {
            *word = if name == "f16-storage" {
                u32::from(half::f16::from_f32((i & 31) as f32 * 0.25).to_bits())
                    | (u32::from(half::f16::from_f32(((i >> 5) & 31) as f32 * 0.5).to_bits()) << 16)
            } else {
                i as u32
            };
        }
        std::ptr::copy_nonoverlapping(dimensions.as_ptr(), indirect_args.data().cast(), 3);
    }
    let source = if name == "atomic" {
        format!(
            "var<uniform> params: vec4<u32>; var<storage, read> input: array<u32>; var<storage, read_write> output: array<atomic<u32>>; @compute @workgroup_size({WIDTH}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{ atomicAdd(&output[0], input[id.x] & 7u); }}"
        )
    } else {
        format!(
            "{} var<uniform> params: vec4<u32>; var<storage, read> input: array<u32>; var<storage, read_write> output: array<u32>; var<workgroup> tile: array<u32, {WIDTH}>; @compute @workgroup_size({WIDTH}) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {{ let i = id.x; {body} }}",
            if name.starts_with("f16") {
                "enable f16;"
            } else {
                ""
            }
        )
    };
    let source = if name == "3d" || name == "3d-indirect" {
        source
            .replace("workgroup_size(128)", "workgroup_size(4, 8, 4)")
            .replace("let i = id.x;", "let i = id.x + 64u * (id.y + 32u * id.z);")
    } else {
        source
    };
    let source = if name == "f16-storage" {
        source.replace("input: array<u32>", "input: array<vec2<f16>>")
    } else {
        source
    };
    let shader = context.create_shader(gpu::ShaderDesc {
        source: &source,
        naga_module: None,
    });
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name,
        data_layouts: &[&Data::layout()],
        compute: shader.at("main"),
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name,
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    {
        let mut pass = encoder.transfer("upload-and-clear");
        pass.copy_buffer_to_buffer(upload.into(), input.into(), size);
        pass.fill_buffer(output.into(), size, 0);
    }
    {
        let mut pass = encoder.compute(name);
        let mut commands = pass.with(&pipeline);
        commands.bind(
            0,
            &Data {
                input: input.into(),
                output: output.into(),
                params: [23, 17, 19, 65536],
            },
        );
        if indirect {
            commands.dispatch_indirect(indirect_args.into());
        } else {
            commands.dispatch(dimensions);
        }
    }
    {
        let mut pass = encoder.transfer("download");
        pass.copy_buffer_to_buffer(output.into(), download.into(), size);
    }
    let sync = context.submit(&mut encoder);
    assert!(
        context.wait_for(&sync, 10_000).unwrap(),
        "{name}: GPU timeout"
    );
    let actual =
        unsafe { std::slice::from_raw_parts(download.data().cast::<u32>(), expected.len()) };
    let mismatch = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (a, b))| {
            if name == "sine" || name == "cosine" || name == "sincos" {
                let error = (f32::from_bits(**a) - f32::from_bits(**b)).abs();
                !error.is_finite() || error > 2e-6
            } else {
                a != b
            }
        })
        .map(|(i, (&a, &b))| (i, a, b));
    context.destroy_command_encoder(&mut encoder);
    context.destroy_compute_pipeline(&mut pipeline);
    for buffer in [upload, input, output, download, indirect_args] {
        context.destroy_buffer(buffer);
    }
    assert_eq!(mismatch, None, "{name}: (index, actual, expected)");
    println!("PASS: {name} ({} values)", expected.len());
}

fn main() {
    env_logger::init();
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            device_id: Some(0x74b5),
            validation: true,
            ..Default::default()
        })
        .expect("MI300X Vulkan context")
    };
    let info = context.device_information();
    assert!(!info.is_software_emulated, "hardware required");
    println!("Device: {} / {}", info.device_name, info.driver_info);
    let only = std::env::args().nth(1);
    if let Some(name) = &only {
        assert!(
            [
                "integer",
                "uniform",
                "compare",
                "uniform-pair",
                "division",
                "dot4",
                "dot4-chain",
                "pack-i8",
                "mix-half",
                "sine",
                "cosine",
                "sincos",
                "wrap",
                "indirect",
                "float",
                "float-trans",
                "packed-half",
                "shared",
                "private-array",
                "scratch-array",
                "scratch-grow",
                "scratch-reuse",
                "3d",
                "3d-indirect",
                "f16",
                "f16-storage",
                "atomic"
            ]
            .contains(&name.as_str()),
            "unknown test: {name}"
        );
        assert!(
            !name.starts_with("f16") || context.capabilities().shader_float16,
            "f16 is unavailable"
        );
    }
    let check = |name: &str, body: &str, reference: &dyn Fn(u32) -> u32, indirect| {
        if only.as_deref().is_none_or(|requested| requested == name) {
            let expected: Vec<_> = (0..COUNT as u32).map(reference).collect();
            run(&context, name, body, &expected, indirect);
        }
    };
    check(
        "pack-i8",
        "let x = input[i]; var word = 0u; for (var b = 0u; b < 4u; b++) { let q = clamp(i32(round(f32((x >> (b * 4u)) & 255u) - 127.0)), -127, 127); word |= (bitcast<u32>(q) & 255u) << (b * 8u); } output[i] = word;",
        &|i| {
            (0..4).fold(0, |word, b| {
                word | (((((i >> (b * 4)) & 255) as i32 - 127).clamp(-127, 127) as u32 & 255)
                    << (b * 8))
            })
        },
        false,
    );
    check(
        "mix-half",
        "let x = input[i]; let h = unpack2x16float(0x40003c00u + ((x & 1023u) << 16u) + (x & 1023u)); output[i] = bitcast<u32>(h.x * f32(x & 255u) + h.y);",
        &|i| {
            let x = half::f16::from_bits(0x3c00 + (i & 1023) as u16).to_f32();
            let y = half::f16::from_bits(0x4000 + (i & 1023) as u16).to_f32();
            (x * (i & 255) as f32 + y).to_bits()
        },
        false,
    );
    for (name, repeats) in [("dot4", 1u32), ("dot4-chain", 8u32)] {
        check(
            name,
            &format!(
                "let a = input[i] * 1664525u + 1013904223u; let b = input[i] * 22695477u + 1u; var result = 0i; for (var j = 0u; j < {repeats}u; j++) {{ result += dot4I8Packed(a + j, b); }} output[i] = bitcast<u32>(result);"
            ),
            &|i| {
                let a = i.wrapping_mul(1664525).wrapping_add(1013904223);
                let b = i.wrapping_mul(22695477).wrapping_add(1);
                let mut result = 0i32;
                for j in 0..repeats {
                    for k in 0..4 {
                        result += ((a.wrapping_add(j) >> (k * 8)) as i8 as i32)
                            * ((b >> (k * 8)) as i8 as i32);
                    }
                }
                result as u32
            },
            false,
        );
    }
    check(
        "cosine",
        "output[i] = bitcast<u32>(cos(f32(input[i] & 255u) * 0.0625));",
        &|i| ((i & 255) as f32 * 0.0625).cos().to_bits(),
        false,
    );
    check(
        "sincos",
        "let x = f32(input[i] & 255u) * 0.0625; output[i] = bitcast<u32>(sin(x) + cos(x));",
        &|i| {
            let x = (i & 255) as f32 * 0.0625;
            (x.sin() + x.cos()).to_bits()
        },
        false,
    );
    check(
        "sine",
        "output[i] = bitcast<u32>(sin(f32(input[i] & 255u) * 0.0625));",
        &|i| ((i & 255) as f32 * 0.0625).sin().to_bits(),
        false,
    );
    check(
        "division",
        "output[i] = input[i] / params.y;",
        &|i| i / 17,
        false,
    );
    check(
        "uniform-pair",
        "output[i] = input[i] * params.y + params.x;",
        &|i| i * 17 + 23,
        false,
    );
    check(
        "compare",
        "if input[i] < params.y { output[i] = input[i] + 7u; } else { output[i] = 42u; }",
        &|i| if i < 17 { i + 7 } else { 42 },
        false,
    );
    check(
        "uniform",
        "output[i] = input[i] * params.y + params.x + params.z + params.w;",
        &|i| i * 17 + 23 + 19 + 65536,
        false,
    );
    check(
        "integer",
        "output[i] = input[i] * 1664525u + 1013904223u;",
        &|i| i.wrapping_mul(1664525).wrapping_add(1013904223),
        false,
    );
    check(
        "wrap",
        "output[i] = (input[i] + 128u) & 65535u;",
        &|i| (i + 128) & 65535,
        false,
    );
    check(
        "indirect",
        "output[i] = input[i] * 3u + 7u;",
        &|i| i * 3 + 7,
        true,
    );
    for name in ["3d", "3d-indirect"] {
        check(
            name,
            "output[i] = input[i] * 3u + 7u + lane;",
            &|i| i * 3 + 7 + (i % 4) + 4 * ((i / 64) % 8) + 32 * ((i / 2048) % 4),
            name == "3d-indirect",
        );
    }
    check(
        "float",
        "let x = f32(input[i] & 1023u) * 0.25; output[i] = bitcast<u32>(fma(x, 1.5, 0.125));",
        &|i| ((i & 1023) as f32 * 0.375 + 0.125).to_bits(),
        false,
    );
    check(
        "float-trans",
        "let x = f32(input[i] & 7u); output[i] = bitcast<u32>(exp2(x) + 1.0);",
        &|i| ((1u32 << (i & 7)) as f32 + 1.0).to_bits(),
        false,
    );
    check(
        "packed-half",
        "let x = f32(input[i] & 31u) * 0.25; let h = pack2x16float(vec2<f32>(x, x + 0.5)); let v = unpack2x16float(h); output[i] = bitcast<u32>(v.x + v.y);",
        &|i| ((i & 31) as f32 * 0.5 + 0.5).to_bits(),
        false,
    );
    check(
        "shared",
        "tile[lane] = input[i] & 255u; workgroupBarrier(); var stride = 64u; loop { if lane < stride { tile[lane] += tile[lane + stride]; } workgroupBarrier(); if stride == 1u { break; } stride /= 2u; } output[i] = tile[0];",
        &|i| if (i / WIDTH) & 1 == 0 { 8128 } else { 24512 },
        false,
    );
    check(
        "private-array",
        "var values: array<u32, 64>; for (var j = 0u; j < 64u; j++) { values[j] = input[i] + j; } var total = 0u; for (var j = 0u; j < 64u; j++) { total += values[(input[i] + j) & 63u]; } output[i] = total;",
        &|i| i * 64 + 2016,
        false,
    );
    check(
        "scratch-array",
        "var values: array<u32, 1024>; for (var j = 0u; j < 1024u; j++) { values[j] = input[i] ^ (j * 17u); } var total = 0u; for (var j = 0u; j < 1024u; j++) { total += values[(input[i] + j) & 1023u]; } output[i] = total;",
        &|i| (0..1024u32).map(|j| i ^ (j * 17)).sum(),
        false,
    );
    for (name, length) in [("scratch-grow", 2048u32), ("scratch-reuse", 1024)] {
        let body = format!(
            "var values: array<u32, {length}>; for (var j = 0u; j < {length}u; j++) {{ values[j] = input[i] ^ (j * 17u); }} var total = 0u; for (var j = 0u; j < {length}u; j++) {{ total += values[(input[i] + j) & {}u]; }} output[i] = total;",
            length - 1
        );
        check(
            name,
            &body,
            &|i| (0..length).map(|j| i ^ (j * 17)).sum(),
            false,
        );
    }
    if context.capabilities().shader_float16 {
        check(
            "f16-storage",
            "let a = input[i].x; let b = input[(i + 128u) % 65536u].y; output[i] = bitcast<u32>(f32(a) + f32(b));",
            &|i| ((i & 31) as f32 * 0.25 + (((i + 128) >> 5) & 31) as f32 * 0.5).to_bits(),
            false,
        );
        check(
            "f16",
            "let x = f16(input[i] & 31u) * f16(0.25); output[i] = bitcast<u32>(f32(x * f16(1.5) + f16(0.125)));",
            &|i| ((i & 31) as f32 * 0.375 + 0.125).to_bits(),
            false,
        );
    }
    if only
        .as_deref()
        .is_none_or(|requested| requested == "atomic")
    {
        run(&context, "atomic", "", &[COUNT as u32 / 8 * 28], false);
    }
}
