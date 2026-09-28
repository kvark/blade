//! Shader modules.
//!
//! `rustc` type-checks these against `synaga-shader`. `build.rs` reads the
//! same files and serializes a Naga module for each one.
// Resources are named after the fields of the `ShaderData` that binds them.
#![allow(non_upper_case_globals)]

pub mod a_trous;
pub mod brdf;
pub mod camera;
pub mod color;
pub mod config;
pub mod debug;
pub mod debug_blit;
pub mod debug_draw;
pub mod debug_param;
pub mod env_importance;
pub mod env_light;
pub mod env_prepare;
pub mod fill_gbuf;
pub mod gbuf;
pub mod hit;
pub mod noop;
pub mod path_trace;
pub mod post_proc;
pub mod quaternion;
pub mod random;
pub mod raster;
pub mod ray_trace;
pub mod sampling;
pub mod skin;
pub mod skin_inc;
pub mod surface;
pub mod vertex;
