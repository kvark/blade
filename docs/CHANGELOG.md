Changelog for *Blade* project

## blade-graphics-0.7.1 (22 Feb 2025)
- vk: make us compatible with Mesa's LavaPipe

## blade-egui-0.7 (21 Feb 2025)
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
