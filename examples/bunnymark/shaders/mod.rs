//! Bunnymark's shader, checked as Rust. The build script serializes it as a Naga module.
// Resources are named after the fields of the `ShaderData` that binds them.
#![allow(non_upper_case_globals)]

#[path = "sprite.rs"]
pub mod sprite;
