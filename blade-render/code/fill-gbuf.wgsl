#include "quaternion.inc.wgsl"
#include "camera.inc.wgsl"
#include "debug.inc.wgsl"
#include "debug-param.inc.wgsl"

// Has to match the host!
struct Vertex {
    pos: vec3<f32>,
    normal: u32,
    tex_coords: vec2<f32>,
    pad: vec2<f32>,
}
struct VertexBuffer {
    data: array<Vertex>,
}
struct IndexBuffer {
    data: array<u32>,
}
var<storage, read> vertex_buffers: binding_array<VertexBuffer>;
var<storage, read> index_buffers: binding_array<IndexBuffer>;
var textures: binding_array<texture_2d<f32>>;
var sampler_linear: sampler;
var sampler_nearest: sampler;

struct HitEntry {
    index_buf: u32,
    vertex_buf: u32,
    // packed quaternion
    geometry_to_world_rotation: u32,
    pad: u32,
    geometry_to_object: mat4x3<f32>,
    base_color_texture: u32,
    // packed color factor
    base_color_factor: u32,
}
var<storage, read> hit_entries: array<HitEntry>;

var<uniform> camera: CameraParams;
var acc_struct: acceleration_structure;

var out_depth: texture_storage_2d<r32float, write>;
var out_basis: texture_storage_2d<rgba8snorm, write>;
var out_albedo: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id.xy > camera.target_size)) {
        return;
    }

    var rq: ray_query;
    let ray_dir = get_ray_direction(camera, global_id.xy);
    rayQueryInitialize(&rq, acc_struct, RayDesc(0x90u, 0xFFu, 0.0, camera.depth, camera.position, ray_dir));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);

    var depth = 0.0;
    var basis = vec4<f32>(0.0);
    var albedo = vec3<f32>(0.0);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        let enable_debug = (debug.flags & DEBUG_FLAGS_GEOMETRY) != 0u && all(global_id.xy == debug.mouse_pos);
        let entry = hit_entries[intersection.instance_custom_index + intersection.geometry_index];
        depth = intersection.t;

        var indices = intersection.primitive_index * 3u + vec3<u32>(0u, 1u, 2u);
        if (entry.index_buf != ~0u) {
            let iptr = &index_buffers[entry.index_buf].data;
            indices = vec3<u32>((*iptr)[indices.x], (*iptr)[indices.y], (*iptr)[indices.z]);
        }

        let vptr = &vertex_buffers[entry.vertex_buf].data;
        let vertices = array<Vertex, 3>(
            (*vptr)[indices.x],
            (*vptr)[indices.y],
            (*vptr)[indices.z],
        );

        let barycentrics = vec3<f32>(1.0 - intersection.barycentrics.x - intersection.barycentrics.y, intersection.barycentrics);
        let tex_coords = mat3x2(vertices[0].tex_coords, vertices[1].tex_coords, vertices[2].tex_coords) * barycentrics;
        let normal_rough = mat3x2(unpack2x16snorm(vertices[0].normal), unpack2x16snorm(vertices[1].normal), unpack2x16snorm(vertices[2].normal)) * barycentrics;
        let normal_geo = vec3<f32>(normal_rough, sqrt(max(0.0, 1.0 - dot(normal_rough, normal_rough))));
        let geo_to_world_rot = normalize(unpack4x8snorm(entry.geometry_to_world_rotation));
        let normal_world = qrot(geo_to_world_rot, normal_geo);

        if (enable_debug) {
            let debug_len = intersection.t * 0.2;
            let pos_object = entry.geometry_to_object * mat3x4(
                vec4<f32>(vertices[0].pos, 1.0), vec4<f32>(vertices[1].pos, 1.0), vec4<f32>(vertices[2].pos, 1.0)
            );
            let positions = intersection.object_to_world * mat3x4(
                vec4<f32>(pos_object[0], 1.0), vec4<f32>(pos_object[1], 1.0), vec4<f32>(pos_object[2], 1.0)
            );
            debug_line(positions[0].xyz, positions[1].xyz, 0x00FF00u);
            debug_line(positions[1].xyz, positions[2].xyz, 0x00FF00u);
            debug_line(positions[2].xyz, positions[0].xyz, 0x00FF00u);
            let poly_normal = normalize(cross(positions[1].xyz - positions[0].xyz, positions[2].xyz - positions[0].xyz));
            let poly_center = (positions[0].xyz + positions[1].xyz + positions[2].xyz) / 3.0;
            debug_line(poly_center, poly_center + debug_len * poly_normal, 0x0000FFu);
            let pos_world = camera.position + intersection.t * ray_dir;
            debug_line(pos_world, pos_world + debug_len * normal_world, 0xFF0000u);
        }

        let pre_tangent = select(
            vec3<f32>(0.0, 0.0, 1.0),
            vec3<f32>(0.0, 1.0, 0.0),
            abs(dot(normal_world, vec3<f32>(0.0, 1.0, 0.0))) < abs(dot(normal_world, vec3<f32>(0.0, 0.0, 1.0))));
        let bitangent = normalize(cross(normal_world, pre_tangent));
        let tangent = normalize(cross(bitangent, normal_world));
        basis = make_quat(mat3x3(tangent, bitangent, normal_world));

        let base_color_factor = unpack4x8unorm(entry.base_color_factor);
        let lod = 0.0; //TODO: this is actually complicated
        let base_color_sample = textureSampleLevel(textures[entry.base_color_texture], sampler_linear, tex_coords, lod);
        albedo = (base_color_factor * base_color_sample).xyz;
    }
    textureStore(out_depth, global_id.xy, vec4<f32>(depth, 0.0, 0.0, 0.0));
    textureStore(out_basis, global_id.xy, basis);
    textureStore(out_albedo, global_id.xy, vec4<f32>(albedo, 0.0));
}
