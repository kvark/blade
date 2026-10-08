//! egui's shader, checked as Rust. The build script serializes it as a Naga module.
// Resources are named after the fields of the `ShaderData` that binds them.
#![allow(non_upper_case_globals)]

pub mod egui;

// `rustc`'s layout of `Uniforms`, asserted to be the shader's. `Vertex` is
// not asserted here: the shader never sees it as a uniform, so synaga models
// no layout for it, and the `Vertex` derive takes the buffer's layout from
// this same struct. `src/lib.rs` asserts it matches epaint's.
synaga_shader::check_layout!();
