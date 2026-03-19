# blade-particle

GPU-driven particle system for the [blade](https://github.com/kvark/blade) engine.

## Overview

Particles are simulated on the GPU using compute shaders (reset, emit, update)
and rendered as camera-facing billboard quads. The crate is split into:

- **`ParticlePipeline`** – shared GPU pipelines, created once per surface format.
- **`ParticleSystem`** – per-instance buffers and emitter state. Many systems can
  share one pipeline.

## Data format

Particle effects are defined as `ParticleEffect` structs, serializable with
[RON](https://github.com/ron-rs/ron):

```ron
(
    capacity: 2000,
    emitter: (
        rate: 0.0,
        burst_count: 0,
        shape: Sphere(radius: 0.3),
    ),
    particle: (
        life: [0.3, 1.0],
        speed: [3.0, 12.0],
        scale: [0.03, 0.1],
        color: Palette([
            [255, 200, 50, 255],
            [255, 120, 20, 255],
            [200, 60, 10, 255],
        ]),
    ),
)
```

## Rendering backend

Particle drawing is currently integrated with the **rasterizer** backend only.
The compute simulation runs regardless of backend, but the draw call
(alpha-blended billboard quads) is issued during the rasterizer render pass.
Ray-tracing integration is not yet implemented.

## Usage

```rust
// Create pipeline (once)
let pipeline = ParticlePipeline::new(&context, PipelineDesc {
    name: "particles",
    draw_format: surface_format,
    sample_count: 1,
});

// Create system (per effect instance)
let mut system = pipeline.create_system(&context, "explosion", &effect);

// Trigger a burst
system.burst(100, [x, y, z]);

// Each frame: simulate then draw
system.update(&pipeline, &mut encoder, dt);
system.draw(&pipeline, &mut render_pass, &camera);
```
