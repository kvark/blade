//! Bunnymark's shader, checked as Rust. The build script serializes it as a Naga module.
// Resources are named after the fields of the `ShaderData` that binds them.
#![allow(non_upper_case_globals)]

#[path = "sprite.rs"]
pub mod sprite;

// `rustc`'s layout of `Globals` and `Locals`, asserted to be the shader's.
synaga_shader::check_layout!("bunnymark_shaders_layout.rs");
