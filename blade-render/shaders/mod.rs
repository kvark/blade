//! Shader modules.
//!
//! `rustc` type-checks these against `synaga-shader`. `build.rs` reads the
//! same files and writes WGSL into `code/`.
#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    unused_assignments,
    unused_imports,
    unused_variables,
    unused_mut,
    static_mut_refs,
    clippy::all
)]

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

const _: () = {
    assert!(crate::DebugMode::Final as u32 == config::DebugMode_Final);
    assert!(crate::DebugMode::Variance as u32 == config::DebugMode_Variance);
    assert!(crate::DebugDrawFlags::SPACE.bits() == config::DebugDrawFlags_SPACE);
    assert!(crate::DebugDrawFlags::RESTIR.bits() == config::DebugDrawFlags_RESTIR);
    assert!(crate::DebugTextureFlags::ALBEDO.bits() == config::DebugTextureFlags_ALBEDO);
    assert!(crate::DebugTextureFlags::EMISSIVE.bits() == config::DebugTextureFlags_EMISSIVE);
    assert!(crate::MAX_LOCAL_LIGHTS as u32 == config::MAX_LOCAL_LIGHTS);
    assert!(crate::MAX_JOINTS_PER_DRAW as u32 == config::MAX_JOINTS_PER_DRAW);
};
