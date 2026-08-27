// Geometry and material data of the scene, as seen by the ray tracing passes.
//
// Requires `brdf.inc.wgsl` for the material model,
// and `debug-param.inc.wgsl` for the texture flags.
#include "vertex.inc.wgsl"

struct VertexBuffer {
    data: array<Vertex>,
}
struct IndexBuffer {
    data: array<u32>,
}

// Has to match the host!
struct HitEntry {
    index_buf: u32,
    vertex_buf: u32,
    prev_vertex_buf: u32,
    flags: u32,
    geometry_to_object: mat4x3<f32>,
    prev_geometry_to_object: mat4x3<f32>,
    prev_object_to_world: mat4x3<f32>,
    base_color_texture: u32,
    // packed color factor
    base_color_factor: u32,
    normal_texture: u32,
    normal_scale: f32,
    // green channel is roughness, blue channel is metalness
    metallic_roughness_texture: u32,
    metalness: f32,
    roughness: f32,
    emissive_texture: u32,
    emissive_factor: vec4<f32>,
}

var<storage, read> vertex_buffers: binding_array<VertexBuffer>;
var<storage, read> index_buffers: binding_array<IndexBuffer>;
var<storage, read> hit_entries: array<HitEntry>;
var textures: binding_array<texture_2d<f32>>;
var sampler_linear: sampler;

fn affine_linear(transform: mat4x3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(transform[0], transform[1], transform[2]);
}

fn hit_winding(entry: HitEntry) -> f32 {
    return select(1.0, -1.0, (entry.flags & 1u) != 0u);
}

fn hit_normal(entry: HitEntry, object_to_world: mat4x3<f32>, normal: vec3<f32>) -> vec3<f32> {
    // Skinning assumes uniform scale, so the linear part acts on normals
    // like a rotation after normalization.
    let linear = affine_linear(object_to_world) * affine_linear(entry.geometry_to_object);
    return normalize(linear * normal);
}

fn hit_tangent_space(
    entry: HitEntry,
    object_to_world: mat4x3<f32>,
    normal: vec3<f32>,
    tangent: vec3<f32>,
    bitangent_sign: f32,
) -> mat3x3<f32> {
    let linear = affine_linear(object_to_world) * affine_linear(entry.geometry_to_object);
    let n = hit_normal(entry, object_to_world, normal);
    return tangent_basis(
        n,
        linear * tangent,
        bitangent_sign,
        sign(determinant(linear)),
    );
}

fn fetch_triangle_indices(entry: HitEntry, primitive_index: u32) -> vec3<u32> {
    var indices = primitive_index * 3u + vec3<u32>(0u, 1u, 2u);
    if (entry.index_buf != ~0u) {
        let iptr = &index_buffers[entry.index_buf].data;
        indices = vec3<u32>((*iptr)[indices.x], (*iptr)[indices.y], (*iptr)[indices.z]);
    }
    return indices;
}

fn make_barycentrics(uv: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(1.0 - uv.x - uv.y, uv);
}

// Read the material of a hit, converting it into the specular workflow.
//
// `ignore_textures` carries `DebugTextureFlags`, which the debug views
// use to look at the factors in isolation.
fn sample_hit_material(entry: HitEntry, tex_coords: vec2<f32>, lod: f32, ignore_textures: u32) -> Material {
    var base_color = unpack4x8unorm(entry.base_color_factor).xyz;
    if ((ignore_textures & DebugTextureFlags_ALBEDO) == 0u) {
        base_color *= textureSampleLevel(textures[entry.base_color_texture], sampler_linear, tex_coords, lod).xyz;
    }

    var metalness = entry.metalness;
    var roughness = entry.roughness;
    if ((ignore_textures & DebugTextureFlags_METALLIC_ROUGHNESS) == 0u) {
        let mr = textureSampleLevel(textures[entry.metallic_roughness_texture], sampler_linear, tex_coords, lod);
        roughness *= mr.y;
        metalness *= mr.z;
    }

    return material_from_metallic_roughness(base_color, metalness, roughness);
}

fn sample_hit_emissive(entry: HitEntry, tex_coords: vec2<f32>, lod: f32, ignore_textures: u32) -> vec3<f32> {
    var emissive = entry.emissive_factor.xyz;
    if ((ignore_textures & DebugTextureFlags_EMISSIVE) == 0u) {
        emissive *= textureSampleLevel(textures[entry.emissive_texture], sampler_linear, tex_coords, lod).xyz;
    }
    return emissive;
}

// Direction of the normal map, in tangent space.
fn sample_hit_normal_map(entry: HitEntry, tex_coords: vec2<f32>, lod: f32, ignore_textures: u32) -> vec3<f32> {
    if ((ignore_textures & DebugTextureFlags_NORMAL) != 0u) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let raw_unorm = textureSampleLevel(textures[entry.normal_texture], sampler_linear, tex_coords, lod).xy;
    let n_xy = entry.normal_scale * (2.0 * raw_unorm - 1.0);
    return vec3<f32>(n_xy, sqrt(max(0.0, 1.0 - dot(n_xy, n_xy))));
}
