//! Measures verified FP32 compute across independent contexts on the same GPU.
use blade_graphics as gpu;
use gpu::ShaderData as _;
use std::{
    sync::{Arc, Barrier},
    time::Instant,
};

const ELEMENTS: u32 = 262_144;
const DEFAULT_ITERATIONS: u32 = 16_384;
const PASSES: usize = 8;

#[derive(blade_macros::ShaderData)]
struct Data {
    output: gpu::BufferPiece,
}

fn worker(index: usize, barrier: Arc<Barrier>, iterations: u32) -> (f64, f64) {
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            device_id: Some(0x74b5),
            timing: true,
            ..Default::default()
        })
        .unwrap()
    };
    assert!(!context.device_information().is_software_emulated);
    let size = ELEMENTS as u64 * 8 * 4;
    let output = context.create_buffer(gpu::BufferDesc {
        name: "throughput",
        size,
        memory: gpu::Memory::Device,
    });
    let download = context.create_buffer(gpu::BufferDesc {
        name: "results",
        size,
        memory: gpu::Memory::Download,
    });
    let source = format!(
        r#"
var<storage, read_write> output: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let base = f32(gid.x & 15u) * 0.125;
    var x = vec4<f32>(base, base + 0.125, base + 0.25, base + 0.375);
    var y = x + vec4<f32>(0.5);
    for (var j = 0u; j < {iterations}u; j++) {{
        x = fma(x, vec4<f32>(0.9999), vec4<f32>(0.01));
        y = fma(y, vec4<f32>(0.9998), vec4<f32>(0.02));
    }}
    output[gid.x * 2u] = x;
    output[gid.x * 2u + 1u] = y;
}}
"#
    );
    let shader = context.create_shader(gpu::ShaderDesc {
        source: &source,
        naga_module: None,
    });
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "throughput",
        data_layouts: &[&Data::layout()],
        compute: shader.at("main"),
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "throughput",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut reference = [[0.0f32; 8]; 16];
    let mut unfused = reference;
    for (id, values) in reference.iter_mut().enumerate() {
        for (c, v) in values.iter_mut().enumerate() {
            *v = (id + c) as f32 * 0.125;
            unfused[id][c] = *v;
            for _ in 0..iterations {
                unfused[id][c] = unfused[id][c] * if c < 4 { 0.9999 } else { 0.9998 }
                    + if c < 4 { 0.01 } else { 0.02 };
                *v = v.mul_add(
                    if c < 4 { 0.9999 } else { 0.9998 },
                    if c < 4 { 0.01 } else { 0.02 },
                );
            }
        }
    }
    encoder.start();
    {
        let mut pass = encoder.compute("warmup");
        let mut commands = pass.with(&pipeline);
        commands.bind(
            0,
            &Data {
                output: output.into(),
            },
        );
        commands.dispatch([1, 1, 1]);
    }
    let warmup = context.submit(&mut encoder);
    assert!(context.wait_for(&warmup, 10_000).unwrap());
    barrier.wait();
    let start = Instant::now();
    let mut gpu_seconds = 0.0;
    for pass_index in 0..PASSES {
        encoder.start();
        {
            let mut pass = encoder.compute("fma");
            let mut commands = pass.with(&pipeline);
            commands.bind(
                0,
                &Data {
                    output: output.into(),
                },
            );
            commands.dispatch([ELEMENTS / 256, 1, 1]);
        }
        if pass_index == PASSES - 1 {
            encoder.transfer("download").copy_buffer_to_buffer(
                output.into(),
                download.into(),
                size,
            );
        }
        let sync = context.submit(&mut encoder);
        assert!(context.wait_for(&sync, 10_000).unwrap());
        for (label, duration) in encoder.last_timing().pass_durations() {
            if label == "fma" {
                gpu_seconds += duration.as_secs_f64();
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let words =
        unsafe { std::slice::from_raw_parts(download.data().cast::<f32>(), (size / 4) as usize) };
    let mut max_error = 0.0f32;
    for (id, actual) in words.chunks_exact(8).enumerate() {
        for (c, &a) in actual.iter().enumerate() {
            let error = (a - reference[id & 15][c])
                .abs()
                .min((a - unfused[id & 15][c]).abs());
            assert!(
                a.is_finite() && error < 0.0001,
                "worker {index}, element {id}, component {c}: {a} != {}",
                reference[id & 15][c]
            );
            max_error = max_error.max(error);
        }
    }
    let flops = ELEMENTS as f64 * iterations as f64 * 16.0 * PASSES as f64;
    println!(
        "worker={index} verified={} gpu_ms={:.3} wall_ms={:.3} kernel_tflops={:.3} max_error={max_error}",
        words.len(),
        gpu_seconds * 1000.0,
        elapsed * 1000.0,
        flops / gpu_seconds / 1e12
    );
    context.destroy_command_encoder(&mut encoder);
    context.destroy_compute_pipeline(&mut pipeline);
    context.destroy_buffer(output);
    context.destroy_buffer(download);
    (elapsed, flops)
}

fn main() {
    let workers: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(1);
    let iterations = std::env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(DEFAULT_ITERATIONS);
    assert!((1..=1_048_576).contains(&iterations));
    assert!((1..=16).contains(&workers));
    let barrier = Arc::new(Barrier::new(workers));
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|i| {
                let b = barrier.clone();
                scope.spawn(move || worker(i, b, iterations))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let wall = results.iter().map(|r| r.0).fold(0.0f64, f64::max);
    let flops: f64 = results.iter().map(|r| r.1).sum();
    println!(
        "PASS: workers={workers} aggregate_wall_tflops={:.3}",
        flops / wall / 1e12
    );
}
