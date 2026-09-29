use super::brdf::{Material, material_from_metallic_roughness};
use super::config::DebugTextureFlags;
use super::vertex::{Vertex, tangent_basis};
use synaga_shader::*;

pub struct VertexBuffer {
    pub data: [Vertex],
}

struct IndexBuffer {
    pub data: [u32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct HitEntry {
    pub index_buf: u32,
    pub vertex_buf: u32,
    pub prev_vertex_buf: u32,
    pub flags: u32,
    pub geometry_to_object: Mat4x3,
    pub prev_geometry_to_object: Mat4x3,
    pub prev_object_to_world: Mat4x3,
    pub base_color_texture: u32,
    // packed color factor
    pub base_color_factor: u32,
    pub normal_texture: u32,
    pub normal_scale: f32,
    // green channel is roughness, blue channel is metalness
    pub metallic_roughness_texture: u32,
    pub metalness: f32,
    pub roughness: f32,
    pub emissive_texture: u32,
    pub emissive_factor: Vec4,
}

pub static vertex_buffers: Storage<BindingArray<VertexBuffer>> = binding();
static index_buffers: Storage<BindingArray<IndexBuffer>> = binding();
pub static hit_entries: Storage<[HitEntry]> = binding();
static textures: BindingArray<Texture2D<f32>> = binding();
pub static sampler_linear: Sampler = binding();

fn affine_linear(transform: Mat4x3) -> Mat3 {
    mat3(transform[0].xyz(), transform[1].xyz(), transform[2].xyz())
}

pub fn hit_winding(entry: HitEntry) -> f32 {
    select(1.0, -1.0, (entry.flags & 1) != 0)
}

pub fn fetch_triangle_indices(entry: HitEntry, primitive_index: u32) -> Vec3<u32> {
    let mut indices = primitive_index * 3 + vec3(0, 1, 2);
    if entry.index_buf != u32::MAX {
        indices = vec3(
            index_buffers[entry.index_buf as usize].data[indices.x as usize],
            index_buffers[entry.index_buf as usize].data[indices.y as usize],
            index_buffers[entry.index_buf as usize].data[indices.z as usize],
        );
    }
    indices
}

pub fn make_barycentrics(uv: Vec2) -> Vec3 {
    let w = 1.0 - uv.x - uv.y;
    vec3(w, uv.x, uv.y)
}

pub fn sample_hit_material(
    entry: HitEntry,
    tex_coords: Vec2,
    lod: f32,
    ignore_textures: DebugTextureFlags,
) -> Material {
    let mut base_color = unpack4x8unorm(entry.base_color_factor).xyz();
    if !ignore_textures.contains(DebugTextureFlags::ALBEDO) {
        base_color *= textures[entry.base_color_texture as usize]
            .sample_level(&sampler_linear, tex_coords, lod)
            .xyz();
    }

    let mut metalness = entry.metalness;
    let mut roughness = entry.roughness;
    if !ignore_textures.contains(DebugTextureFlags::METALLIC_ROUGHNESS) {
        let mr = textures[entry.metallic_roughness_texture as usize].sample_level(
            &sampler_linear,
            tex_coords,
            lod,
        );
        roughness *= mr.y;
        metalness *= mr.z;
    }

    material_from_metallic_roughness(base_color, metalness, roughness)
}

pub fn sample_hit_emissive(
    entry: HitEntry,
    tex_coords: Vec2,
    lod: f32,
    ignore_textures: DebugTextureFlags,
) -> Vec3 {
    let mut emissive = entry.emissive_factor.xyz();
    if !ignore_textures.contains(DebugTextureFlags::EMISSIVE) {
        emissive *= textures[entry.emissive_texture as usize]
            .sample_level(&sampler_linear, tex_coords, lod)
            .xyz();
    }
    emissive
}

pub fn sample_hit_normal_map(
    entry: HitEntry,
    tex_coords: Vec2,
    lod: f32,
    ignore_textures: DebugTextureFlags,
) -> Vec3 {
    if ignore_textures.contains(DebugTextureFlags::NORMAL) {
        return vec3(0.0, 0.0, 1.0);
    }
    let raw_unorm = textures[entry.normal_texture as usize]
        .sample_level(&sampler_linear, tex_coords, lod)
        .xy();
    let n_xy = entry.normal_scale * (2.0 * raw_unorm - 1.0);
    n_xy.extend((1.0 - n_xy.dot(n_xy)).max(0.0).sqrt())
}

pub fn hit_normal(entry: HitEntry, object_to_world: Mat4x3, normal: Vec3) -> Vec3 {
    // Skinning assumes uniform scale, so the linear part acts on normals
    // like a rotation after normalization.
    let linear = affine_linear(object_to_world) * affine_linear(entry.geometry_to_object);
    (linear * normal).normalize()
}

pub fn hit_tangent_space(
    entry: HitEntry,
    object_to_world: Mat4x3,
    normal: Vec3,
    tangent: Vec3,
    bitangent_sign: f32,
) -> Mat3 {
    let linear = affine_linear(object_to_world) * affine_linear(entry.geometry_to_object);
    let n = hit_normal(entry, object_to_world, normal);
    tangent_basis(
        n,
        linear * tangent,
        bitangent_sign,
        sign(determinant(linear)),
    )
}
