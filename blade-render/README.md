# Blade Render

[![Docs](https://docs.rs/blade-render/badge.svg)](https://docs.rs/blade-render)
[![Crates.io](https://img.shields.io/crates/v/blade-render.svg?maxAge=2592000)](https://crates.io/crates/blade-render)

Rasterized and ray-traced rendering based on [blade-graphics](https://crates.io/crates/blade-graphics) and [blade-asset](https://crates.io/crates/blade-asset).

![sponza scene](etc/sponza.jpg)

## Platforms

The rasterizer supports the portable graphics profile, including WebGL2. The
ray tracer requires a backend and device with ray-query support.
