//! Particle shaders, checked as Rust. The build script serializes them as a Naga module.
// Resources are named after the fields of the `ShaderData` that binds them.
#![allow(non_upper_case_globals)]

pub mod particle;

// `rustc`'s layout of the particle structs, asserted to be the shaders'.
synaga_shader::check_layout!();
