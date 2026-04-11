Changelog for *Blade* project

## blade-graphics-0.8.2 (TBD)

- metal: enable fast math, skip debug groups in production
- add `ComputeCommandEncoder::barrier()` for inline compute-to-compute synchronization within a pass

## blade-graphics-0.8.1 (28 Mar 2026)

- new API for enumerating device availability
- bumped the max pass count to 1000 for better ML compatibility

## blade-graphics-0.8, blade-util-0.3, blade-egui-0.7, blade-particle-0.1, blade-asset-0.2.1, blade-engine-0.1 (26 Mar 2026)

- examples:
  - moved some of the old example code into GPU tests
  - new "info" example to show supported GPUs
  - new Asteroids XR example 
- graphics:
  - OpenXR / Android support (tested on Quest 3S)
  - option to disable ray tracing initialization
  - separate `Capabilities` flag for binding arrays, including TLAS arrays
  - cooperative matrix operations support (auto-detected via `Capabilities`)
  - `wait_for` now returns `Result<bool, DeviceError>` instead of `bool`,
    distinguishing timeout from device-lost errors
  - `memory_stats()` API for querying VRAM budget/usage (via `VK_EXT_memory_budget`)
  - `Buffer::size()` accessor on all backends
    - debug bounds check on `BufferPiece::data()`
  - `PlatformError` is now a unified opaque type across all backends
  - `ComputePipelineBase` trait exposes `get_workgroup_size()` for generic code
  - `NotSupportedError`, `DeviceError`, and `PlatformError` implement `Display` + `Error`
  - vk: set `MUTABLE_FORMAT` on depth+stencil textures for flexible view creation
  - vk: graceful handling of surface acquire errors instead of panicking
  - vk: reject GPUs that cannot present in Intel+NVIDIA PRIME configurations
  - egl: use DMA-BUF sharing with different displays for presentation
  - vk: uniform buffer fallback for buggy Qualcomm devices
  - metal: fix lifetimes of acceleration structures
- particle:
  - new crate forged from the original particle example
- asset:
  - support procedural assets
- engine:
  - moved the engine from "blade" itself, reserving it for future use
  - choice between ray-tracing and rasterization rendering pipelines
  - first-class XR support

## blade-graphics-0.7.1 (22 Feb 2026)

- vk: make us compatible with Mesa's LavaPipe

## blade-egui-0.7 (21 Feb 2026)

- update to egui-0.33 and blade-graphics-0.7

## blade-graphics-0.7 (27 Sep 2025)

- graphics
  - pipeline constants API
  - allow buffer bindings for uniform data
  - supported MSAA samples are now returned in context `Capabilities`
  - Vulkan:
    - improve correctness of present synchronization

## blade-graphics-0.6, blade-util-0.2, blade-egui-0.6, blade-render-0.4, blade-0.3 (21 Dec 2024)

- graphics:
  - API for surface creation
    - allows multiple windows used by the same context
  - multi-sampling support
  - API for destruction of pipelines
  - return detailed initialization errors
  - every pass now takes a label
  - automatic GPU pass markers
  - ability to capture pass GPU timings
  - ability to force the use of a specific GPU
  - ability to set viewport
  - fragment shader is optional
  - support more texture formats
  - Metal:
    - migrate to "objc2"
    - support for workgroup memory
    - concurrent compute dispatches
  - Egl:
    - destroy old surface on resize
  - Vulkan:
    - support unused bind groups
- egui:
  - fix blending color space

## blade-egui-0.5 (09 Nov 2024)

- update egui to 0.29

## blade-graphics-0.5, blade-macros-0.3, blade-egui-0.4, blade-util-0.1 (27 Aug 2024)

- crate: `blade-util` for helper utilities
- graphics:
  - vertex buffers support
  - surface configuration:
    - transparency support
    - option to disable exclusive fullscreen
    - VK: using linear sRGB color space if available
  - exposed initialization errors
  - exposed device information
  - Vk:
    - fixed initial RAM consumption
    - worked around Intel descriptor memory allocation bug
    - fixed coherent memory requirements
    - rudimentary cleanup on destruction
  - GLES:
    - support for storage buffer and compute
    - scissor rects, able to run "particle" example
    - blending and draw masks
    - fixed texture uploads
- examples: "move"
- window API switched to raw-window-handle-0.6

## blade-graphics-0.4, blade-render-0.3, blade-0.2 (22 Mar 2024)

- crate: `blade` for high-level engine
  - built-in physics via Rapier3D
- examples: "vehicle"
- render:
  - support object motion
  - support clockwise mesh winding
  - fixed mipmap generation
- update to egui-0.26 and winit-0.29
- graphics:
  - display sync configuration
  - color space configuration
  - work around Intel+Nvidia presentation bug
  - overlay support

## blade-graphics-0.3, blade-render-0.2 (17 Nov 2023)

- tangent space generation
- spatio-temporal resampling
- SVGF de-noising
- environment map importance sampling
- shaders as assets
  - with includes, enums, and bitflags
  - with hot reloading
- load textures: `exr`, `hdr`
- utility: `FramePacer`
- examples: scene editing in "scene"
  - using egui-gizmo for manipulation

## blade-graphics-0.2, blade-render-0.1 (31 May 2023)

- ray tracing support
- examples: "ray-query", "scene"
- crate: `blade-egui` for egui integration
- crate: `blade-asset` for asset pipeline
- crate: `blade-render` for ray-traced renderer
  - load models: `gltf`
  - load textures: `png`, `jpg`

## blade-graphics-0.1 (25 Jan 2023)

- backends: Vulkan, Metal, OpenGL ES + WebGL2
- examples: "mini", "bunnymark", "particle"
- crate: `blade-graphics` for GPU abstracting GPU operations
- crate: `blade-macros` for `ShaderData` derivation
