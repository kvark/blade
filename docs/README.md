# Blade

[![Matrix](https://img.shields.io/static/v1?label=dev&message=%23blade&color=blueviolet&logo=matrix)](https://matrix.to/#/#blade-dev:matrix.org)
[![Build Status](https://github.com/kvark/blade/workflows/check/badge.svg)](https://github.com/kvark/blade/actions)
[![Docs](https://docs.rs/blade/badge.svg)](https://docs.rs/blade)
[![Crates.io](https://img.shields.io/crates/v/blade.svg?label=blade)](https://crates.io/crates/blade)
[![Crates.io](https://img.shields.io/crates/v/blade-graphics.svg?label=blade-graphics)](https://crates.io/crates/blade-graphics)
[![Crates.io](https://img.shields.io/crates/v/blade-render.svg?label=blade-render)](https://crates.io/crates/blade-render)

![](logo.png)

Blade is an innovative rendering solution for Rust. It starts with a lean low-level GPU abstraction focused at ergonomics and fun. It then grows into a high-level rendering library that utilizes hardware ray-tracing. It's accompanied by a [task-parallel asset pipeline](https://youtu.be/1DiA3OYqvqU) together with [egui](https://www.egui.rs/) support, turning into a minimal rendering engine. Finally, the top-level Blade engine combines all of this with Rapier3D-based physics and hides them behind a concise API. Talks:
- [In GPU we Rust](https://youtu.be/92mwRCXvMVk) (Rust AI meetup, 2024)
- [Blade - lean and mean graphics library](https://youtu.be/63dnzjw4azI?t=623) (Rust Graphics meetup, 2023)
- [Blade asset pipeline](https://youtu.be/1DiA3OYqvqU) (Rust Gamedev meetup, 2023)
- [Blade scene editor](https://www.youtube.com/watch?v=Q5IUOvuXoC8) (Rust Gamedev meetup, 2023)

![architecture](https://raw.githubusercontent.com/kvark/blade/main/docs/architecture2.png)

## Examples

![scene editor](../blade-egui/etc/scene-editor.jpg)
![particle example](../blade-graphics/etc/particles.png)
![vehicle example](vehicle-colliders.jpg)
![sponza scene](../blade-render/etc/sponza.jpg)

## Instructions

Just the usual :crab: workflow. E.g. to run the bunny-mark benchmark run:

```bash
cargo run --release --example bunnymark
```

CPU tests run with the default command:

```bash
cargo test
```

GPU integration tests are marked `#[ignore]` and need to be requested explicitly:

```bash
cargo test --test gpu_examples -- --ignored --nocapture
```

## Platforms

Blade-graphics can run on:
- Linux (Vulkan and GLES)
- Windows (Vulkan)
- iOS/MacOS/tvOS (Metal)
- Android (OpenXR/Vulkan)
- Web (WebGL2)

The full-stack Blade Engine can only run on Vulkan with hardware Ray Tracing support.

## Citing Blade

If Blade appears in a paper, a technical report, or a thesis, cite the software
itself and pin the revision you measured or built against — the synchronization
model this library is built on has changed within single releases, so a bare
project name is not enough for anyone to reproduce a number.

```bibtex
@software{blade,
  author    = {Dzmitry Malyshau and contributors},
  title     = {Blade: a sharp and simple graphics library},
  year      = {2026},
  url       = {https://github.com/kvark/blade},
  version   = {0.3.0},
  note      = {Revision <commit>}
}
```

Replace `<commit>` with the short hash of what you used (`git rev-parse --short
HEAD`). If you measured performance, say which backend and which driver
version, because both change results more than the library version does.

There is also a `CITATION.cff` in the repository root, so GitHub's "Cite this
repository" button and tools that read it produce the same metadata.
