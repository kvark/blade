//! Constants, the debug view's mode and its flags, shared by the shader
//! modules and the host.
//!
//! `DEBUG_MODE` is `cfg!(debug_assertions)`. The build script reads the same
//! predicate from Cargo, so a debug build and a release build disagree here
//! the way `rustc` does.
//!
//! The host uses these as they are: `blade_render::DebugMode`,
//! `DebugDrawFlags` and `DebugTextureFlags` are the types here, and its limits
//! are the ones here.

use synaga_shader::Shared;

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
    bytemuck::NoUninit,
    bytemuck::Zeroable,
    strum::EnumIter,
)]
pub enum DebugMode {
    #[default]
    Final,
    Depth,
    DiffuseAlbedoTexture,
    DiffuseAlbedoFactor,
    NormalTexture,
    NormalScale,
    GeometryNormal,
    ShadingNormal,
    Motion,
    HitConsistency,
    SampleReuse,
    Roughness,
    SpecularF0,
    Emissive,
    Variance,
}

/// What the debug view draws over the frame.
#[repr(transparent)]
#[derive(Hash, PartialEq, Eq, PartialOrd, Shared)]
pub struct DebugDrawFlags(u32);

bitflags::bitflags! {
    impl DebugDrawFlags: u32 {
        const SPACE = 1;
        const GEOMETRY = 1 << 1;
        const RESTIR = 1 << 2;
    }
}

/// The material textures the debug view leaves out.
#[repr(transparent)]
#[derive(Hash, PartialEq, Eq, PartialOrd, Shared)]
pub struct DebugTextureFlags(u32);

bitflags::bitflags! {
    impl DebugTextureFlags: u32 {
        const ALBEDO = 1;
        const NORMAL = 1 << 1;
        const METALLIC_ROUGHNESS = 1 << 2;
        const EMISSIVE = 1 << 3;
    }
}
