use super::brdf::*;
use super::config::*;
use super::vertex::*;
use synaga_shader::*;

pub struct VertexBuffer {
    pub data: [Vertex],
}

pub struct IndexBuffer {
    pub data: [u32],
}

#[derive(Clone, Copy, Default)]
pub struct HitEntry {
    pub index_buf: u32,
    pub vertex_buf: u32,
    pub prev_vertex_buf: u32,
    pub flags: u32,
    pub geometry_to_object: mat4x3,
    pub prev_geometry_to_object: mat4x3,
    pub prev_object_to_world: mat4x3,
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
    pub emissive_factor: vec4,
}

pub static vertex_buffers: Storage<binding_array<VertexBuffer>> = binding();

pub static index_buffers: Storage<binding_array<IndexBuffer>> = binding();

pub static hit_entries: Storage<[HitEntry]> = binding();

pub static textures: binding_array<texture_2d<f32>> = binding();

pub static sampler_linear: sampler = binding();

#[shader]
pub fn affine_linear(transform: mat4x3) -> mat3x3 {
    return mat3x3(transform[0].xyz(), transform[1].xyz(), transform[2].xyz());
}

#[shader]
pub fn hit_winding(entry: HitEntry) -> f32 {
    return select(1.0, -1.0, (entry.flags & 1u32) != 0u32);
}

#[shader]
pub fn fetch_triangle_indices(entry: HitEntry, primitive_index: u32) -> vec3u {
    let mut indices = primitive_index * 3u32 + vec3u(0u32, 1u32, 2u32);
    if (entry.index_buf != !0u32) {
        indices = vec3u(
            (index_buffers[(entry.index_buf) as usize].data)[(indices.x) as usize],
            (index_buffers[(entry.index_buf) as usize].data)[(indices.y) as usize],
            (index_buffers[(entry.index_buf) as usize].data)[(indices.z) as usize],
        );
    }
    return indices;
}

#[shader]
pub fn make_barycentrics(uv: vec2) -> vec3 {
    let w = 1.0 - uv.x - uv.y;
    return vec3(w, uv.x, uv.y);
}

#[shader]
pub fn sample_hit_material(
    entry: HitEntry,
    tex_coords: vec2,
    lod: f32,
    ignore_textures: u32,
) -> Material {
    let mut base_color = unpack4x8unorm(entry.base_color_factor).xyz();
    if ((ignore_textures & DebugTextureFlags_ALBEDO) == 0u32) {
        base_color *= textureSampleLevel(
            &textures[(entry.base_color_texture) as usize],
            &sampler_linear,
            tex_coords,
            lod,
        )
        .xyz();
    }

    let mut metalness = entry.metalness;
    let mut roughness = entry.roughness;
    if ((ignore_textures & DebugTextureFlags_METALLIC_ROUGHNESS) == 0u32) {
        let mr = textureSampleLevel(
            &textures[(entry.metallic_roughness_texture) as usize],
            &sampler_linear,
            tex_coords,
            lod,
        );
        roughness *= mr.y;
        metalness *= mr.z;
    }

    return material_from_metallic_roughness(base_color, metalness, roughness);
}

#[shader]
pub fn sample_hit_emissive(
    entry: HitEntry,
    tex_coords: vec2,
    lod: f32,
    ignore_textures: u32,
) -> vec3 {
    let mut emissive = entry.emissive_factor.xyz();
    if ((ignore_textures & DebugTextureFlags_EMISSIVE) == 0u32) {
        emissive *= textureSampleLevel(
            &textures[(entry.emissive_texture) as usize],
            &sampler_linear,
            tex_coords,
            lod,
        )
        .xyz();
    }
    return emissive;
}

#[shader]
pub fn sample_hit_normal_map(
    entry: HitEntry,
    tex_coords: vec2,
    lod: f32,
    ignore_textures: u32,
) -> vec3 {
    if ((ignore_textures & DebugTextureFlags_NORMAL) != 0u32) {
        return vec3(0.0, 0.0, 1.0);
    }
    let raw_unorm = textureSampleLevel(
        &textures[(entry.normal_texture) as usize],
        &sampler_linear,
        tex_coords,
        lod,
    )
    .xy();
    let n_xy = entry.normal_scale * (2.0 * raw_unorm - 1.0);
    return (n_xy).extend(sqrt(max(0.0, 1.0 - dot(n_xy, n_xy))));
}

#[shader]
pub fn hit_normal(entry: HitEntry, object_to_world: mat4x3, normal: vec3) -> vec3 {
    // Skinning assumes uniform scale, so the linear part acts on normals
    // like a rotation after normalization.
    let linear = affine_linear(object_to_world) * affine_linear(entry.geometry_to_object);
    return normalize(linear * normal);
}

#[shader]
pub fn hit_tangent_space(
    entry: HitEntry,
    object_to_world: mat4x3,
    normal: vec3,
    tangent: vec3,
    bitangent_sign: f32,
) -> mat3x3 {
    let linear = affine_linear(object_to_world) * affine_linear(entry.geometry_to_object);
    let n = hit_normal(entry, object_to_world, normal);
    return tangent_basis(
        n,
        linear * tangent,
        bitangent_sign,
        sign(determinant(linear)),
    );
}
