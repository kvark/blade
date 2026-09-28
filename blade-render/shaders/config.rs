//! Constants and discriminants shared by the shader modules.
//!
//! `DEBUG_MODE` is `cfg!(debug_assertions)`. The build script reads the same
//! predicate from Cargo, so a debug build and a release build disagree here
//! the way `rustc` does.
//!
//! The host reads these too: `blade_render::DebugMode` is the enum here, the
//! host's flag sets take each bit from the flag enums, and its limits are the
//! ones here. A shader compares or masks with `Variant as u32` because the
//! uniform itself is a plain integer.

pub const DEBUG_MODE: bool = cfg!(debug_assertions);
pub const MAX_LOCAL_LIGHTS: usize = 8;
pub const MAX_JOINTS_PER_DRAW: usize = 64;

/// What the renderer shows: the final image, or one of its inputs.
#[repr(u32)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    blade_macros::AsPrimitive,
    strum::EnumIter,
)]
pub enum DebugMode {
    #[default]
    Final = 0,
    Depth = 1,
    DiffuseAlbedoTexture = 2,
    DiffuseAlbedoFactor = 3,
    NormalTexture = 4,
    NormalScale = 5,
    GeometryNormal = 6,
    ShadingNormal = 7,
    Motion = 8,
    HitConsistency = 9,
    SampleReuse = 10,
    Roughness = 11,
    SpecularF0 = 12,
    Emissive = 13,
    Variance = 15,
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DebugDrawFlags {
    Space = 1,
    Geometry = 2,
    Restir = 4,
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DebugTextureFlags {
    Albedo = 1,
    Normal = 2,
    MetallicRoughness = 4,
    Emissive = 8,
}
