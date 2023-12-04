# Blade EGUI

[![Docs](https://docs.rs/blade-egui/badge.svg)](https://docs.rs/blade-egui)
[![Crates.io](https://img.shields.io/crates/v/blade-egui.svg?maxAge=2592000)](https://crates.io/crates/blade-egui)

[EGUI](https://www.egui.rs/) support for [Blade-graphics](https://crates.io/crates/blade-graphics).

![scene editor](etc/scene-editor.jpg)

## Instructions

Just the usual :crab: workflow. E.g. to run the bunny-mark benchmark run:
```bash
cargo run --release --example bunnymark
```

## Platforms

The full-stack Blade Engine can only run on Vulkan with hardware Ray Tracing support.
However, on secondary platforms, such as Metal and GLES/WebGL2, one can still use Blde-Graphics and Blade-Egui.
