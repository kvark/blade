#![cfg(not(target_arch = "wasm32"))]
#![allow(irrefutable_let_patterns)]

mod asset_hub;
mod model;
mod renderer;
mod scene;
mod texture;

pub use asset_hub::*;
pub use model::{Geometry, Material, Model};
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
    pub transform: blade::Transform,
}

#[derive(Default)]
pub struct Scene {
    pub objects: Vec<Object>,
}
