//! Constants and discriminants shared by the shader modules.
//!
//! `DEBUG_MODE` is `cfg!(debug_assertions)`. The build script reads the same
//! predicate from Cargo, so a debug build and a release build disagree here
//! the way `rustc` does.
//!
//! The enums match the host types in `blade-render`. A shader compares or
//! masks with `Variant as u32` because the uniform itself is a plain integer.

pub const DEBUG_MODE: bool = cfg!(debug_assertions);

pub const MAX_LOCAL_LIGHTS: u32 = 8;
pub const MAX_LOCAL_LIGHTS_LEN: usize = 8;
pub const MAX_JOINTS_PER_DRAW: u32 = 64;
pub const MAX_JOINTS_PER_DRAW_LEN: usize = 64;

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DebugMode {
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
