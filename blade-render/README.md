# Blade Render

[![Docs](https://docs.rs/blade-render/badge.svg)](https://docs.rs/blade-render)
[![Crates.io](https://img.shields.io/crates/v/blade-render.svg?maxAge=2592000)](https://crates.io/crates/blade-render)

Rasterized and ray-traced rendering based on [blade-graphics](https://crates.io/crates/blade-graphics) and [blade-asset](https://crates.io/crates/blade-asset).

![sponza scene](etc/sponza.jpg)

## Platforms

The rasterizer supports the portable graphics profile, including WebGL2. The
ray tracer requires a backend and device with ray-query support.

## Skeletal animation

The renderer's glTF model contains only the geometry-side skin bindings. Clip
data, hierarchy evaluation, playback, and interpolation belong to a separate
animation model in `blade-engine`. The renderer receives only the evaluated
result:

```rust
let mut object = blade_render::Object::from(model_handle);
object.pose = Some(pose_evaluated_by_the_animation_system);

// After the frame is submitted, preserve what was rendered before the next pose.
object.flip();
object.pose = Some(next_pose_evaluated_by_the_animation_system);
```

`pose` and `prev_pose` are public, just like `transform` and `prev_transform`.
`Object::flip` copies both current values into the previous-frame slots so the
ray tracer can generate motion vectors from the state that produced the previous
pixels. Applications using `blade-engine::Engine` can instead call
`Engine::set_animation`; the engine loads a separate animation component,
advances and evaluates its player, and fills both poses when it submits render
objects.

Joint indices are compacted per mesh primitive and stored as 8-bit values with
unorm8 weights (8 bytes total). A primitive may use up to 64 distinct joints,
matching the portable WebGL2 uniform-block limit. Pose node transforms and the
raster joint palette are affine 3x4 matrices (`blade_graphics::Transform` /
uniform `mat3x4`), 48 bytes each.

The 3.4 KiB fixture at `tests/assets/animated_skin.glb` contains two joints and
one rotation-and-scale clip. Its fixed-keyframe raster reference can be run on
any supported GPU backend:

```bash
cargo test --test gpu_examples snapshot_animated_skin -- --ignored --nocapture
```

Set `BLADE_UPDATE_SNAPSHOTS=1` to regenerate the reference image after an
intentional rendering change.
