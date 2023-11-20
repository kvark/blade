#![cfg(not(target_arch = "wasm32"))]
#![allow(irrefutable_let_patterns, clippy::new_without_default)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

mod asset_hub;
pub mod model;
mod render;
pub mod shader;
pub mod texture;
pub mod util;

pub use asset_hub::*;
pub use model::Model;
pub use render::*;
pub use shader::Shader;
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

pub struct Camera {
    pub pos: mint::Vector3<f32>,
    pub rot: mint::Quaternion<f32>,
    pub fov_y: f32,
    pub depth: f32,
}

pub struct Object {
    pub model: blade_asset::Handle<Model>,
    pub transform: blade_graphics::Transform,
    pub prev_transform: blade_graphics::Transform,
}

impl From<blade_asset::Handle<Model>> for Object {
    fn from(model: blade_asset::Handle<Model>) -> Self {
        Self {
            model,
            transform: blade_graphics::IDENTITY_TRANSFORM,
            prev_transform: blade_graphics::IDENTITY_TRANSFORM,
        }
    }
}
