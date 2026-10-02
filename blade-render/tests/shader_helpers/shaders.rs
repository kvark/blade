//! The helpers under test, as the modules they are in `shaders/`, and the
//! probe beside them, so that `super::brdf` names the same module for `rustc`
//! here as for synaga on the GPU.
//!
//! `HELPERS` is the same list, for the GPU's copy.

// What the probe does not reach is still compiled; resources are named after
// the fields of the `ShaderData` that binds them.
#![allow(dead_code, non_upper_case_globals)]

#[path = "../../shaders/brdf.rs"]
pub mod brdf;
#[path = "../../shaders/camera.rs"]
pub mod camera;
#[path = "../../shaders/color.rs"]
pub mod color;
#[path = "../../shaders/config.rs"]
pub mod config;
#[path = "../../shaders/quaternion.rs"]
pub mod quaternion;
#[path = "../../shaders/random.rs"]
pub mod random;
#[path = "../../shaders/sampling.rs"]
pub mod sampling;
#[path = "../../shaders/skin_inc.rs"]
pub mod skin_inc;
#[path = "../../shaders/surface.rs"]
pub mod surface;
#[path = "../../shaders/vertex.rs"]
pub mod vertex;

#[path = "probe.rs"]
pub mod probe;

/// Each module above, by the name the others reach it by, and its source.
pub const HELPERS: [(&str, &str); 10] = [
    ("brdf", include_str!("../../shaders/brdf.rs")),
    ("camera", include_str!("../../shaders/camera.rs")),
    ("color", include_str!("../../shaders/color.rs")),
    ("config", include_str!("../../shaders/config.rs")),
    ("quaternion", include_str!("../../shaders/quaternion.rs")),
    ("random", include_str!("../../shaders/random.rs")),
    ("sampling", include_str!("../../shaders/sampling.rs")),
    ("skin_inc", include_str!("../../shaders/skin_inc.rs")),
    ("surface", include_str!("../../shaders/surface.rs")),
    ("vertex", include_str!("../../shaders/vertex.rs")),
];

pub const PROBE: &str = include_str!("probe.rs");
