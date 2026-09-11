//! Measure whether a low-priority compute queue yields to interactive work.
//!
//! The benchmark keeps one logical device busy with a dependent integer
//! workload while a separate normal-priority device submits tiny probes. Run
//! it on a desktop GPU with `VK_KHR_global_priority`; lower foreground p95
//! latency in the `Low` case means the driver is honoring the request.

use blade_graphics as gpu;
use gpu::ShaderData as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

const HEAVY_WORDS: u32 = 65_536;
const PROBES: usize = 40;

const SHADER: &str = r#"
var<storage, read_write> values: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn heavy(@builtin(global_invocation_id) id: vec3<u32>) {
    var value = values[id.x];
    for (var round = 0u; round < 4096u; round += 1u) {
        value = value * 1664525u + 1013904223u;
        value = value ^ (value >> 13u);
    }
    values[id.x] = value;
}

@compute @workgroup_size(1, 1, 1)
fn probe() {
    values[0] = values[0] + 1u;
}
"#;

#[derive(blade_macros::ShaderData)]
struct WorkData {
    values: gpu::BufferPiece,
}

#[derive(Debug)]
struct CaseResult {
    background_priority: gpu::QueuePriority,
    idle_median_us: u128,
    busy_median_us: u128,
    busy_p95_us: u128,
    busy_max_us: u128,
    background_dispatches: usize,
}

fn context(priority: gpu::QueuePriority) -> gpu::Context {
    unsafe {
        gpu::Context::init(gpu::ContextDesc {
            validation: cfg!(debug_assertions),
            queue_priority: priority,
            ..Default::default()
        })
        .expect("failed to initialize GPU context")
    }
}

fn pipeline(
    context: &gpu::Context,
    entry: &'static str,
    name: &'static str,
) -> gpu::ComputePipeline {
    let shader = context.create_shader(gpu::ShaderDesc {
        source: SHADER,
        naga_module: None,
    });
    context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name,
        data_layouts: &[&WorkData::layout()],
        compute: shader.at(entry),
    })
}

fn buffer(context: &gpu::Context, words: u32, name: &'static str) -> gpu::Buffer {
    context.create_buffer(gpu::BufferDesc {
        name,
        size: u64::from(words) * 4,
        memory: gpu::Memory::Shared,
    })
}

fn submit_probe(
    context: &gpu::Context,
    encoder: &mut gpu::CommandEncoder,
    pipeline: &gpu::ComputePipeline,
    values: gpu::Buffer,
) -> Duration {
    encoder.start();
    {
        let mut pass = encoder.compute("foreground-probe");
        let mut command = pass.with(pipeline);
        command.bind(
            0,
            &WorkData {
                values: values.into(),
            },
        );
        command.dispatch([1, 1, 1]);
    }
    let started = Instant::now();
    let sync = context.submit(encoder);
    context
        .wait_for(&sync, !0)
        .expect("foreground probe wait failed");
    started.elapsed()
}

fn percentile_us(samples: &[Duration], numerator: usize, denominator: usize) -> u128 {
    let mut micros: Vec<_> = samples.iter().map(Duration::as_micros).collect();
    micros.sort_unstable();
    let index = ((micros.len() - 1) * numerator).div_ceil(denominator);
    micros[index]
}

fn run_case(background_priority: gpu::QueuePriority) -> CaseResult {
    let foreground = context(gpu::QueuePriority::Normal);
    let mut foreground_pipeline = pipeline(&foreground, "probe", "foreground-probe");
    let foreground_buffer = buffer(&foreground, 1, "foreground-value");
    let mut foreground_encoder = foreground.create_command_encoder(gpu::CommandEncoderDesc {
        name: "foreground-probe",
        buffer_count: 2,
        manual_barriers: false,
    });
    let idle: Vec<_> = (0..10)
        .map(|_| {
            submit_probe(
                &foreground,
                &mut foreground_encoder,
                &foreground_pipeline,
                foreground_buffer,
            )
        })
        .collect();

    let running = Arc::new(AtomicBool::new(true));
    let worker_running = running.clone();
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let worker = std::thread::spawn(move || {
        let background = context(background_priority);
        let mut background_pipeline = pipeline(&background, "heavy", "background-heavy");
        let background_buffer = buffer(&background, HEAVY_WORDS, "background-values");
        let mut encoder = background.create_command_encoder(gpu::CommandEncoderDesc {
            name: "background-heavy",
            buffer_count: 2,
            manual_barriers: false,
        });
        let mut dispatches = 0usize;
        while worker_running.load(Ordering::Relaxed) {
            encoder.start();
            {
                let mut pass = encoder.compute("background-heavy");
                let mut command = pass.with(&background_pipeline);
                command.bind(
                    0,
                    &WorkData {
                        values: background_buffer.into(),
                    },
                );
                command.dispatch([HEAVY_WORDS / 64, 1, 1]);
            }
            let sync = background.submit(&mut encoder);
            if dispatches == 0 {
                let _ = ready_tx.send(());
            }
            background
                .wait_for(&sync, !0)
                .expect("background wait failed");
            dispatches += 1;
        }
        background.destroy_buffer(background_buffer);
        background.destroy_compute_pipeline(&mut background_pipeline);
        background.destroy_command_encoder(&mut encoder);
        dispatches
    });
    ready_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("background queue did not start");

    let busy: Vec<_> = (0..PROBES)
        .map(|_| {
            let latency = submit_probe(
                &foreground,
                &mut foreground_encoder,
                &foreground_pipeline,
                foreground_buffer,
            );
            std::thread::sleep(Duration::from_millis(2));
            latency
        })
        .collect();
    running.store(false, Ordering::Relaxed);
    let background_dispatches = worker.join().expect("background worker panicked");

    foreground.destroy_buffer(foreground_buffer);
    foreground.destroy_compute_pipeline(&mut foreground_pipeline);
    foreground.destroy_command_encoder(&mut foreground_encoder);

    CaseResult {
        background_priority,
        idle_median_us: percentile_us(&idle, 1, 2),
        busy_median_us: percentile_us(&busy, 1, 2),
        busy_p95_us: percentile_us(&busy, 95, 100),
        busy_max_us: percentile_us(&busy, 1, 1),
        background_dispatches,
    }
}

fn main() {
    env_logger::init();
    println!("Normal-priority foreground probe under sustained compute load");
    let normal = run_case(gpu::QueuePriority::Normal);
    let low = run_case(gpu::QueuePriority::Low);
    for result in [&normal, &low] {
        println!(
            "background={:?}: idle={}us busy median={}us p95={}us max={}us ({} heavy dispatches)",
            result.background_priority,
            result.idle_median_us,
            result.busy_median_us,
            result.busy_p95_us,
            result.busy_max_us,
            result.background_dispatches,
        );
    }
    if low.busy_p95_us < normal.busy_p95_us {
        let improvement = normal.busy_p95_us as f64 / low.busy_p95_us.max(1) as f64;
        println!("low-priority foreground p95 improvement: {improvement:.2}x");
    } else {
        println!("low priority did not improve p95 in this run");
    }
}
