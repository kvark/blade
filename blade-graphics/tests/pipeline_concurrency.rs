#![cfg(not(any(gles, target_arch = "wasm32")))]

use blade_graphics as gpu;
use std::{sync, thread};

/// Exercise both compiler entry points, including calls on different devices.
/// This is a driver stress test, not a deterministic reproducer of the Quest crash.
#[test]
#[ignore = "requires a GPU; repeat in fresh processes to exercise compiler initialization"]
fn concurrent_compute_and_render_pipelines() {
    const THREADS: usize = 8;
    const ROUNDS: usize = 16;
    let contexts: Vec<_> = (0..4)
        .map(|_| unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() })
        .collect();
    let barrier = sync::Barrier::new(THREADS);
    thread::scope(|scope| {
        for worker in 0..THREADS {
            let context = &contexts[worker / 2];
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                for round in 0..ROUNDS {
                    let salt = std::process::id().wrapping_mul(65537)
                        .wrapping_add((worker * ROUNDS + round) as u32);
                    if (worker + round) % 2 == 0 {
                        let source = format!(
                            "var<storage, read_write> values: array<u32>;
                             @compute @workgroup_size(64)
                             fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                                 var value = values[id.x] ^ {salt}u;
                                 for (var i = 0u; i < 16u; i++) {{
                                     value = (value ^ (value >> 13u)) * 1664525u + i;
                                 }}
                                 values[id.x] = value;
                             }}"
                        );
                        let shader = context.create_shader(gpu::ShaderDesc {
                            source: &source,
                            naga_module: None,
                        });
                        let layout = gpu::ShaderDataLayout {
                            bindings: vec![("values", gpu::ShaderBinding::Buffer)],
                        };
                        let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
                            name: "concurrent-compute",
                            data_layouts: &[&layout],
                            compute: shader.at("main"),
                        });
                        context.destroy_compute_pipeline(&mut pipeline);
                    } else {
                        let source = format!(
                            "@vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {{
                                 return vec4<f32>(f32(index & 1u), f32(index >> 1u), 0.0, 1.0);
                             }}
                             @fragment fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {{
                                 let color = f32((u32(position.x) ^ {salt}u) & 255u) / 255.0;
                                 return vec4<f32>(color, 0.0, 0.0, 1.0);
                             }}"
                        );
                        let shader = context.create_shader(gpu::ShaderDesc {
                            source: &source,
                            naga_module: None,
                        });
                        let mut pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
                            name: "concurrent-render",
                            data_layouts: &[],
                            vertex: shader.at("vs"),
                            vertex_fetches: &[],
                            primitive: gpu::PrimitiveState::default(),
                            depth_stencil: None,
                            fragment: Some(shader.at("fs")),
                            color_targets: &[gpu::ColorTargetState {
                                format: gpu::TextureFormat::Rgba8Unorm,
                                blend: None,
                                write_mask: gpu::ColorWrites::ALL,
                            }],
                            multisample_state: gpu::MultisampleState::default(),
                        });
                        context.destroy_render_pipeline(&mut pipeline);
                    }
                }
            });
        }
    });
}
