#![allow(irrefutable_let_patterns)]

#[cfg(not(target_arch = "wasm32"))]
mod gltf_loader;
#[cfg(not(target_arch = "wasm32"))]
mod renderer;

#[cfg(not(target_arch = "wasm32"))]
pub use renderer::*;

#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],
    // XY of the normal encoded as signed-normalized
    pub normal: [i16; 2],
    pub tex_coords: [f32; 2],
    pub pad: [f32; 2],
}

pub struct Geometry {
    pub vertex_buf: blade::Buffer,
    pub vertex_count: u32,
    pub index_buf: blade::Buffer,
    pub index_type: Option<blade::IndexType>,
    pub triangle_count: u32,
    pub material_index: usize,
}

pub struct Object {
    pub name: String,
    pub geometries: Vec<Geometry>,
    pub transform: blade::Transform,
    pub acceleration_structure: blade::AccelerationStructure,
}

pub struct Texture {
    texture: blade::Texture,
    view: blade::TextureView,
}

pub struct Material {
    base_color_texture_index: usize,
    base_color_factor: [f32; 4],
}

#[derive(Default)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
}

pub struct Camera {
    pub pos: mint::Vector3<f32>,
    pub rot: mint::Quaternion<f32>,
    pub fov_y: f32,
    pub depth: f32,
}
