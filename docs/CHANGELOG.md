Changelog for *Blade* project

## (TBD)

- gles/web: pick the offscreen surface format from `ColorSpace` the same
  way native GLES does (`Linear` → `Rgba8UnormSrgb`). Shaders still write
  linear; the GPU encodes into the texture that present blits to the
  canvas.
- gles/egui: keep WebGL2 buffer bind classes honest. The upload belt records
  a `BufferTarget` and packs 4-byte-aligned subranges, then `sync_buffer`s
  each alloc so copies and draws see GPU data immediately. The GUI painter
  uses separate data and index belts; mixing those in one WebGL buffer made
  `texSubImage` see 0 bytes and `drawElements` bind an element-array target
  to a generic buffer.
  - breaking: `sync_buffer` takes a `BufferPiece` range and size, plus the
    `BufferTarget` bind class.
- gles: skip 0×0 `texStorage` / renderbuffer allocations (WebGL rejects
  them) and reset `UNPACK_ROW_LENGTH` after buffer-to-texture copies.
- render/engine: skeletal animation from glTF skins and TRS clips with step,
  linear, and cubic-spline interpolation. `Engine::set_animation` drives
  `AnimationPlayer` playback, which evaluates `AnimationModel` into a `Pose`;
  `Object` carries the pose and `Object::flip` advances motion-vector history.
  Skinning runs in a compute pass on native backends (feeding per-instance BLAS
  build/refit for the ray tracer), and in the vertex stage on GLES. Covered by
  a minimal animated GLB fixture with raster snapshots.
  - breaking: skinning data (packed 8-bit joints and unorm8 weights) moved to
    a separate `SkinVertex` buffer, keeping the base `Vertex` at 32 bytes.
    Render `Object` gained `pose`/`prev_pose` and `flip`. Raster skinning
    parameters are bound to a second slot for skinned pipelines only.
    `AccelerationStructureDesc` gained `updatable`, and the encoder gained
    `update_bottom_level` for refitting animated BLASes.
  - skinning assumes uniform scale: assets with non-uniform rest or animated
    scale log a warning at load, and their normals may be slightly skewed.
- engine/render: support the raster rendering path on WebGL2. Raster vertex
  data now uses ordinary vertex attributes, model and texture uploads observe
  WebGL's buffer binding classes, and the renderer validates its shaders by
  exporting them as WebGL2 GLSL ES 3.00.
- gfx/engine/render: expose compute and indirect-draw support as runtime device
  capabilities and use them for optional engine features. WebGL-only context
  affinity and upload mechanics remain private target-specific implementation
  details.
  - breaking: `Capabilities` gained `compute` and `indirect_draw` fields.
- render/gles: keep WebGL contexts thread-affine. The engine runs GPU asset
  tasks inline on the context's owning thread.
- gles: `sync_buffer` takes a `BufferTarget` argument naming the buffer's
  binding class (`Data` or `Index`). WebGL2 permanently assigns a buffer to
  the element-array class or the general data class on its first bind, so the
  backend now defers all binding to the first sync, where the caller-provided
  target makes the choice. This makes index buffers work on WebGL2.
  - breaking: existing callers pass `BufferTarget::Data`.
- gfx/vk: derive each global pass barrier's pipeline stages and access masks
  from a lightweight encoder-wide summary of the pass kinds recorded since the
  previous barrier. This adds no per-resource state and does not change the
  public API or automatic barrier placement. Repeated explicit barriers are
  skipped when no intervening pass has produced writes.
- docs: add the cross-vendor study of Blade's global pass barriers and an
  engine-facing synchronization guide.
- gfx: `Memory::Download` asks the allocator for a host-cached mapping, for
  buffers the CPU reads after a transfer. `Shared` can land on write-combined
  or even device-local host-visible memory, where a scan of the range runs
  three orders of magnitude slower and copying it out first does not help.
- render: `RayTracer::view_gbuffer` hands out the geometry and material buffer
  of the frame that was last prepared, as `GBufferViews`. A post process that
  knows what the renderer knows can take silhouettes from the depth and the
  normals, separate texture detail from lighting through the albedo, and read
  the width of a specular highlight from the roughness, none of which is
  recoverable from the color alone. The views belong to the renderer and stay
  valid until the next `resize_screen`.
- render: `RayTracer::view_radiance` hands an external denoiser the current
  demodulated diffuse and specular lighting views. `post_proc_external` accepts
  the denoiser's composed linear-radiance result and runs it through Blade's
  normal tone mapping and surface encoding, completing the round trip without
  a readback or a second graphics context.
- render: `PostProcConfig::tone_map` can be cleared to leave the composed
  radiance alone, so a frame can be captured as high dynamic range data rather
  than only as a picture. The exposure controls are unused when it is off, and
  the display transfer function is skipped along with the curve, since it is
  only defined over the display range. Rendering into a floating point target
  is what makes this observable — a fixed point one still clamps.
  - breaking: `PostProcConfig` gained a field, so it can no longer be built
    without `..Default::default()`
- render: physically based materials, authored as glTF metallic-roughness, shaded in the specular workflow
  - `Material` carries metallic-roughness and emissive, both as factors and textures, cooked from glTF (including `KHR_materials_emissive_strength`)
  - internally, a material is a diffuse albedo, a specular reflectance at normal incidence, and a roughness; the conversion happens in `material_from_metallic_roughness` when the textures are sampled, and is the only place aware of the metalness
  - shared BRDF in `brdf.inc.wgsl`: GGX distribution, height-correlated Smith visibility, Schlick Fresnel, used by the ray tracer as well as the rasterizer
  - shared lobe sampling in `sampling.inc.wgsl` and environment sampling in `env-light.inc.wgsl`
  - ray tracing: material G-buffer, separate diffuse and specular lighting, light samples drawn from the BRDF as well as the environment with MIS between them
  - `ProceduralGeometry` gained the PBR factors, and now builds an acceleration structure, so it can be ray traced
  - breaking: `RasterConfig` no longer overrides the roughness and metalness of all the materials
  - breaking: the cooked model format has changed, the asset caches need to be cleared
  - new debug views for roughness, specular reflectance, and emissive
- render: `RenderMode` selects what the ray tracer does with the scene
  - `RenderMode::Canonical` traces full paths with BSDF sampling and next event estimation on the environment, combined by MIS, accumulating the result over the frames with no reuse and no denoising, so it converges to the ground truth
  - accumulation is reset by `FrameConfig::reset_accumulation` or by moving the camera, and can be capped by `RayConfig::max_accumulated_samples`
  - breaking: `RayTracer::ray_trace` and `denoise` are replaced by `RayTracer::render`, which takes the mode
  - breaking: `RayConfig` describes the sampling of both of the modes: `num_environment_samples` and `num_brdf_samples` are the light and material samples taken at a shading point, combined by multi-sample MIS, while `max_bounces` limits the path length
  - the light found at the last vertex of a path is no longer partly thrown
    away: next event estimation there was weighted against a BSDF sample that
    the path never goes on to take, so the balance heuristic held back a share
    of the contribution and nothing ever supplied it. The weight is the whole
    of it whenever the path ends. Longer paths hide the loss in the throughput
    they have left, so the material grid at three bounces moves by SSIM 0.9998,
    while `max_bounces` of zero — direct lighting and nothing else — was
    missing enough that a white furnace sphere rendered visibly darker than
    the environment it has to disappear into, and now matches it exactly.
- both of the render paths now produce the color space that the surface was
  configured with, taken as `RenderConfig::color_space`, instead of the
  rasterizer always encoding gamma and the ray tracer never doing so
- vk: an XR swapchain honors the requested color space through its format,
  since it has no way to declare one: `Linear` picks an sRGB format for the
  runtime to convert, `Srgb` picks a plain one that is passed through. The
  recommended configuration asks for `Srgb`, which is what the plain format
  the runtimes prefer actually needs.
- vk: `xr_recommended_surface_config` and `create_xr_surface_configured` are
  public, so that an application knows the configuration of its XR surface
- fix `fill-gbuf.wgsl` missing the `wgpu_binding_array` enable directive
- tests: validate the renderer shaders, snapshot the PBR material grid in both of the render paths
- vk: support `VK_EXT_external_memory_host` — enable the extension, query memory-type compatibility via `vkGetMemoryHostPointerPropertiesEXT`, and round allocation size to `minImportedHostPointerAlignment` so `Memory::External(HostAllocation)` imports succeed on drivers that expose the extension
- gles: assign texture units to sampler uniforms where GLSL ES 3.00 can't carry explicit bindings, so multi-texture pipelines don't collide on unit 0 in WebGL2
- gles: apply `RenderPipelineDesc::depth_stencil`, which the backend previously ignored, leaving draw order to decide visibility

## blade-egui-0.8.2, blade-util-0.4.1 (25 Apr 2026)

- fix leaking textures and non-reusable buffers in egui workflow

## blade-graphics-0.8.4 (17 Apr 2026)

- vk: use driver API version for instance creation
- vk: drop unnecessary `UNIFORM_READ` from intra-pass compute barrier
- vk: log leaked GPU memory blocks by name on context teardown

## blade-graphics-0.8.3 (17 Apr 2026)

- vk: fix descriptor over-allocation for uniform data

## blade-graphics-0.8.2 (14 Apr 2026)

- add `ComputeCommandEncoder::barrier()` for inline compute-to-compute synchronization within a pass
- enable `naga::valid::Capabilities::SUBGROUP` for shader validation
- vulkan: fall back to UBO for larger uniform blocks on all systems
- metal: enable fast math, skip debug groups in production

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
