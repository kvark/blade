Changelog for *Blade* project

## (TBD)

- render: treat wasm32 as the GLES raster profile. WebGL2 cannot link a
  vertex-only program, so directional-shadow pipelines always attach the
  empty `raster_shadow_fs` stage. Compute skinning stays native-only
  (`noop.wgsl` on wasm32/GLES) without requiring `RUSTFLAGS=--cfg gles`.
- gles/web: pick the offscreen surface format from `ColorSpace` the same
  way native GLES does (`Linear` → `Rgba8UnormSrgb`). Shaders still write
  linear; the GPU encodes into the texture that present blits to the
  canvas.
