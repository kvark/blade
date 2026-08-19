# blade-engine

High-level engine crate for Blade. It provides the scene/object API, integrates physics, and hooks objects into rendering backends.

## Rendering backends

- `RayTracer`: ray-query based rendering.
- `Rasterizer`: forward+ rasterization with PBR-style shading.

Select a backend via `blade_engine::config::RenderBackend` when creating the engine.
If the selected device does not expose ray-query functionality, including on
WebGL2, the engine falls back to `Rasterizer`.
