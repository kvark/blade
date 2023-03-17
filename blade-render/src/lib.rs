#[cfg(not(target_arch = "wasm32"))]
mod gltf_loader;
#[cfg(not(target_arch = "wasm32"))]
mod renderer;

const MAX_DATA_BUFFERS: u32 = 1000;

#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],
    pub dummy: f32,
}

pub struct Geometry {
    pub vertex_buf: blade::Buffer,
    pub vertex_count: u32,
    pub index_buf: blade::Buffer,
    pub index_type: Option<blade::IndexType>,
    pub triangle_count: u32,
}

pub struct Object {
    pub name: String,
    pub geometries: Vec<Geometry>,
    pub transform: blade::Transform,
    pub acceleration_structure: blade::AccelerationStructure,
}

#[derive(Default)]
pub struct Scene {
    pub objects: Vec<Object>,
}

pub struct Renderer {
    target: blade::Texture,
    target_view: blade::TextureView,
    rt_pipeline: blade::ComputePipeline,
    draw_pipeline: blade::RenderPipeline,
    scene: Scene,
    acceleration_structure: blade::AccelerationStructure,
    hit_buffer: blade::Buffer,
    vertex_buffers: blade::BufferArray<MAX_DATA_BUFFERS>,
    index_buffers: blade::BufferArray<MAX_DATA_BUFFERS>,
    is_tlas_dirty: bool,
    screen_size: blade::Extent,
}

pub struct Camera {
    pub pos: mint::Vector3<f32>,
    pub rot: mint::Quaternion<f32>,
    pub fov_y: f32,
    pub depth: f32,
}
