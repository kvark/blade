Changelog for Blade

## blade-graphics-0.6 (TBD)

- graphics:
  - API for destruction of pipelines
  - every pass now takes a label
  - automatic GPU pass markers
  - ability to capture pass GPU timings
  - Metal:
    - support for workgroup memory

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
