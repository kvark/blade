//! egui's shader, checked as Rust. The build script serializes it as a Naga module.
// Resources are named after the fields of the `ShaderData` that binds them.
#![allow(non_upper_case_globals)]

pub mod egui;

// `rustc`'s layout of `Uniforms` and `Vertex`, asserted to be the shader's.
synaga_shader::check_layout!();
