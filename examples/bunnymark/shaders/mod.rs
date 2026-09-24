//! Bunnymark's shader, checked as Rust. The build script serializes it as a Naga module.
#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    unused_imports,
    unused_variables,
    unused_mut,
    clippy::all
)]

#[path = "sprite.rs"]
pub mod sprite;
