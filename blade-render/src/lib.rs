#![allow(
    irrefutable_let_patterns,
    clippy::new_without_default,
    clippy::needless_borrowed_reference
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    //TODO: re-enable. Currently doesn't like "mem::size_of" on newer Rust
    //unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

mod dummy;
mod env_map;
pub use dummy::DummyResources;
pub use env_map::EnvironmentMap;

#[cfg(not(any(gles, target_arch = "wasm32")))]
mod asset_hub;
#[cfg(not(any(gles, target_arch = "wasm32")))]
pub mod model;
#[cfg(not(any(gles, target_arch = "wasm32")))]
pub mod raster;
#[cfg(not(any(gles, target_arch = "wasm32")))]
mod render;
#[cfg(not(any(gles, target_arch = "wasm32")))]
pub mod shader;
#[cfg(not(any(gles, target_arch = "wasm32")))]
pub mod texture;
#[cfg(not(any(gles, target_arch = "wasm32")))]
pub mod util;

#[cfg(not(any(gles, target_arch = "wasm32")))]
pub use asset_hub::*;
#[cfg(not(any(gles, target_arch = "wasm32")))]
pub use model::{Model, ProceduralGeometry};
#[cfg(not(any(gles, target_arch = "wasm32")))]
pub use raster::{RasterConfig, Rasterizer};
#[cfg(not(any(gles, target_arch = "wasm32")))]
pub use render::*;
#[cfg(not(any(gles, target_arch = "wasm32")))]
pub use shader::Shader;
#[cfg(not(any(gles, target_arch = "wasm32")))]
pub use texture::Texture;

// Has to match the `Vertex` in shaders
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    pub bitangent_sign: f32,
    pub tex_coords: [f32; 2],
    pub normal: u32,
    pub tangent: u32,
}

/// Asymmetric field-of-view angles (in radians).
/// All angles are positive: left/down are measured from center toward left/down.
#[derive(Clone, Copy, Debug)]
pub struct Fov {
    pub left: f32,
    pub right: f32,
    pub up: f32,
    pub down: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub pos: mint::Vector3<f32>,
    pub rot: mint::Quaternion<f32>,
    pub fov_y: f32,
    pub depth: f32,
    /// Per-eye asymmetric FOV. When set, overrides `fov_y` for projection.
    pub fov: Option<Fov>,
}

#[cfg(not(any(gles, target_arch = "wasm32")))]
pub struct Object {
    pub model: blade_asset::Handle<Model>,
    pub transform: blade_graphics::Transform,
    pub prev_transform: blade_graphics::Transform,
    /// Per-object color tint multiplied with the material's base_color_factor.
    /// Default: [1.0, 1.0, 1.0, 1.0] (no tint).
    pub color_tint: [f32; 4],
}

#[cfg(not(any(gles, target_arch = "wasm32")))]
impl From<blade_asset::Handle<Model>> for Object {
    fn from(model: blade_asset::Handle<Model>) -> Self {
        Self {
            model,
            transform: blade_graphics::IDENTITY_TRANSFORM,
            prev_transform: blade_graphics::IDENTITY_TRANSFORM,
            color_tint: [1.0; 4],
        }
    }
}

#[cfg(not(any(gles, target_arch = "wasm32")))]
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
struct CameraParams {
    position: [f32; 3],
    depth: f32,
    orientation: [f32; 4],
    fov: [f32; 2],
    target_size: [u32; 2],
}
