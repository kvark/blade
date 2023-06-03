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
mod env_map;
pub mod model;
mod renderer;
mod scene;
pub mod texture;

pub use asset_hub::*;
pub use env_map::EnvironmentMap;
pub use model::Model;
pub use renderer::*;
pub use texture::Texture;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    // XY of the normal encoded as signed-normalized
    pub normal: [i16; 2],
    pub tex_coords: [f32; 2],
    pub pad: [f32; 2],
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
}

impl From<blade_asset::Handle<Model>> for Object {
    fn from(model: blade_asset::Handle<Model>) -> Self {
        Self {
            model,
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
            .into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PostProcessing {
    //TODO: remove this, compute automatically
    pub average_luminocity: f32,
    pub exposure_key_value: f32,
    pub white_level: f32,
}
impl Default for PostProcessing {
    fn default() -> Self {
        Self {
            average_luminocity: 1.0,
            exposure_key_value: 1.0,
            white_level: 1.0,
        }
    }
}

#[derive(Default)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub environment_map: Option<blade_asset::Handle<Texture>>,
    pub post_processing: PostProcessing,
}
