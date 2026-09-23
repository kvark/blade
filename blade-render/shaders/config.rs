//! Constants the WGSL preprocessor used to paste in with `#use`.
//!
//! `DEBUG_MODE` is `cfg!(debug_assertions)` when rustc checks this file. The
//! build script replaces that `cfg` with a literal before synaga parses it.

pub const DEBUG_MODE: bool = cfg!(debug_assertions);

pub const MAX_LOCAL_LIGHTS: u32 = 8;
pub const MAX_LOCAL_LIGHTS_LEN: usize = 8;
pub const MAX_JOINTS_PER_DRAW: u32 = 64;
pub const MAX_JOINTS_PER_DRAW_LEN: usize = 64;

pub const DebugMode_Final: u32 = 0;
pub const DebugMode_Depth: u32 = 1;
pub const DebugMode_DiffuseAlbedoTexture: u32 = 2;
pub const DebugMode_DiffuseAlbedoFactor: u32 = 3;
pub const DebugMode_NormalTexture: u32 = 4;
pub const DebugMode_NormalScale: u32 = 5;
pub const DebugMode_GeometryNormal: u32 = 6;
pub const DebugMode_ShadingNormal: u32 = 7;
pub const DebugMode_Motion: u32 = 8;
pub const DebugMode_HitConsistency: u32 = 9;
pub const DebugMode_SampleReuse: u32 = 10;
pub const DebugMode_Roughness: u32 = 11;
pub const DebugMode_SpecularF0: u32 = 12;
pub const DebugMode_Emissive: u32 = 13;
pub const DebugMode_Variance: u32 = 15;

pub const DebugDrawFlags_SPACE: u32 = 1;
pub const DebugDrawFlags_GEOMETRY: u32 = 2;
pub const DebugDrawFlags_RESTIR: u32 = 4;

pub const DebugTextureFlags_ALBEDO: u32 = 1;
pub const DebugTextureFlags_NORMAL: u32 = 2;
pub const DebugTextureFlags_METALLIC_ROUGHNESS: u32 = 4;
pub const DebugTextureFlags_EMISSIVE: u32 = 8;
