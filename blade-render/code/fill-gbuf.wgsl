enable wgpu_ray_query;
enable wgpu_binding_array;
#include "quaternion.inc.wgsl"
#include "camera.inc.wgsl"
#include "debug.inc.wgsl"
#include "debug-param.inc.wgsl"
#include "brdf.inc.wgsl"
#include "hit.inc.wgsl"
#include "gbuf.inc.wgsl"

var<uniform> camera: CameraParams;
var<uniform> prev_camera: CameraParams;
var<uniform> debug: DebugParams;
var acc_struct: acceleration_structure;

var out_depth: texture_storage_2d<r32float, write>;
var out_flat_normal: texture_storage_2d<rgba8snorm, write>;
var out_basis: texture_storage_2d<rgba8snorm, write>;
var out_diffuse_albedo: texture_storage_2d<rgba8unorm, write>;
// RGB is the specular reflectance at normal incidence, alpha is the roughness
var out_specular_f0: texture_storage_2d<rgba8unorm, write>;
var out_emissive: texture_storage_2d<rgba16float, write>;
var out_motion: texture_storage_2d<rg16float, write>;
var out_debug: texture_storage_2d<rgba8unorm, write>;

fn debug_raw_normal(
    pos: vec3<f32>, normal_raw: u32, entry: HitEntry, object_to_world: mat4x3<f32>, debug_len: f32, color: u32,
) {
    let nw = hit_normal(entry, object_to_world, decode_normal(normal_raw));
    debug_line(pos, pos + debug_len * nw, color);
}

@compute @workgroup_size(8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id.xy >= camera.target_size)) {
        return;
    }
    if (WRITE_DEBUG_IMAGE && debug.view_mode != DebugMode_Final) {
        textureStore(out_debug, global_id.xy, vec4<f32>(0.0));
    }

    var rq: ray_query;
    let ray_dir = get_ray_direction(camera, vec2<i32>(global_id.xy));
    rayQueryInitialize(&rq, acc_struct, RayDesc(RAY_FLAG_CULL_NO_OPAQUE, 0xFFu, 0.0, camera.depth, camera.position, ray_dir));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);

    var depth = 0.0;
    var basis = vec4<f32>(0.0);
    var flat_normal = vec3<f32>(0.0);
    // Note: the sky is fully diffuse and white, so that the environment
    // survives the modulation in the post-processing.
    var material = Material(vec3<f32>(1.0), vec3<f32>(0.0), 0.0);
    var emissive = vec3<f32>(0.0);
    var motion = vec2<f32>(0.0);
    let enable_debug = all(global_id.xy == debug.mouse_pos);

    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        let entry = hit_entries[intersection.instance_custom_data + intersection.geometry_index];
        depth = intersection.t;

        let indices = fetch_triangle_indices(entry, intersection.primitive_index);
        let vptr = &vertex_buffers[entry.vertex_buf].data;
        let vertices = array<Vertex, 3>(
            (*vptr)[indices.x],
            (*vptr)[indices.y],
            (*vptr)[indices.z],
        );
        let prev_vptr = &vertex_buffers[entry.prev_vertex_buf].data;
        let prev_vertices = array<Vertex, 3>(
            (*prev_vptr)[indices.x],
            (*prev_vptr)[indices.y],
            (*prev_vptr)[indices.z],
        );

        let positions_object = entry.geometry_to_object * mat3x4(
            vec4<f32>(vertices[0].position, 1.0), vec4<f32>(vertices[1].position, 1.0), vec4<f32>(vertices[2].position, 1.0)
        );
        let prev_positions_object = entry.prev_geometry_to_object * mat3x4(
            vec4<f32>(prev_vertices[0].position, 1.0), vec4<f32>(prev_vertices[1].position, 1.0), vec4<f32>(prev_vertices[2].position, 1.0)
        );
        let positions = intersection.object_to_world * mat3x4(
            vec4<f32>(positions_object[0], 1.0), vec4<f32>(positions_object[1], 1.0), vec4<f32>(positions_object[2], 1.0)
        );
        flat_normal = hit_winding(entry) * normalize(cross(positions[1].xyz - positions[0].xyz, positions[2].xyz - positions[0].xyz));

        let barycentrics = make_barycentrics(intersection.barycentrics);
        let position_object = vec4<f32>(positions_object * barycentrics, 1.0);
        let tex_coords = mat3x2(vertices[0].tex_coords, vertices[1].tex_coords, vertices[2].tex_coords) * barycentrics;
        let normal_geo = normalize(mat3x3(decode_normal(vertices[0].normal), decode_normal(vertices[1].normal), decode_normal(vertices[2].normal)) * barycentrics);
        let tangent_geo = normalize(mat3x3(decode_normal(vertices[0].tangent), decode_normal(vertices[1].tangent), decode_normal(vertices[2].tangent)) * barycentrics);
        let lod = 0.0; //TODO: this is actually complicated

        let tangent_space_world = hit_tangent_space(
            entry, intersection.object_to_world, normal_geo, tangent_geo, vertices[0].bitangent_sign,
        );
        let normal_local = sample_hit_normal_map(entry, tex_coords, lod, debug.texture_flags);
        var normal = tangent_space_world * normal_local;
        basis = shortest_arc_quat(vec3<f32>(0.0, 0.0, 1.0), normalize(normal));

        let hit_position = camera.position + intersection.t * ray_dir;
        if (enable_debug) {
            debug_buf.entry.custom_index = intersection.instance_custom_data;
            debug_buf.entry.depth = intersection.t;
            debug_buf.entry.tex_coords = tex_coords;
            debug_buf.entry.base_color_texture = entry.base_color_texture;
            debug_buf.entry.normal_texture = entry.normal_texture;
            debug_buf.entry.position = hit_position;
            debug_buf.entry.flat_normal = flat_normal;
        }
        if (enable_debug && (debug.draw_flags & DebugDrawFlags_SPACE) != 0u) {
            let normal_w = 0.15 * intersection.t * tangent_space_world[2];
            let tangent_w = 0.05 * intersection.t * tangent_space_world[0];
            let bitangent_w = 0.05 * intersection.t * tangent_space_world[1];
            debug_line(hit_position, hit_position + normal_w, 0xFF8000u);
            debug_line(hit_position - 0.5 * tangent_w, hit_position + tangent_w, 0x8080FFu);
            debug_line(hit_position - 0.5 * bitangent_w, hit_position + bitangent_w, 0x80FF80u);
        }
        if (enable_debug && (debug.draw_flags & DebugDrawFlags_GEOMETRY) != 0u) {
            let debug_len = intersection.t * 0.2;
            debug_line(positions[0].xyz, positions[1].xyz, 0x00FFFFu);
            debug_line(positions[1].xyz, positions[2].xyz, 0x00FFFFu);
            debug_line(positions[2].xyz, positions[0].xyz, 0x00FFFFu);
            let poly_center = (positions[0].xyz + positions[1].xyz + positions[2].xyz) / 3.0;
            debug_line(poly_center, poly_center + 0.2 * debug_len * flat_normal, 0xFF00FFu);
            // note: dynamic indexing into positions isn't allowed by WGSL yet
            debug_raw_normal(positions[0].xyz, vertices[0].normal, entry, intersection.object_to_world, 0.5*debug_len, 0xFFFF00u);
            debug_raw_normal(positions[1].xyz, vertices[1].normal, entry, intersection.object_to_world, 0.5*debug_len, 0xFFFF00u);
            debug_raw_normal(positions[2].xyz, vertices[2].normal, entry, intersection.object_to_world, 0.5*debug_len, 0xFFFF00u);
            // draw tangent space
            debug_line(hit_position, hit_position + debug_len * qrot(basis, vec3<f32>(1.0, 0.0, 0.0)), 0x0000FFu);
            debug_line(hit_position, hit_position + debug_len * qrot(basis, vec3<f32>(0.0, 1.0, 0.0)), 0x00FF00u);
            debug_line(hit_position, hit_position + debug_len * qrot(basis, vec3<f32>(0.0, 0.0, 1.0)), 0xFF0000u);
        }

        material = sample_hit_material(entry, tex_coords, lod, debug.texture_flags);
        emissive = sample_hit_emissive(entry, tex_coords, lod, debug.texture_flags);

        if (WRITE_DEBUG_IMAGE) {
            if (debug.view_mode == DebugMode_DiffuseAlbedoTexture) {
                textureStore(out_debug, global_id.xy, vec4<f32>(material.diffuse_albedo, 0.0));
            }
            if (debug.view_mode == DebugMode_DiffuseAlbedoFactor) {
                textureStore(out_debug, global_id.xy, unpack4x8unorm(entry.base_color_factor));
            }
            if (debug.view_mode == DebugMode_NormalTexture) {
                textureStore(out_debug, global_id.xy, vec4<f32>(normal_local, 0.0));
            }
            if (debug.view_mode == DebugMode_NormalScale) {
                textureStore(out_debug, global_id.xy, vec4<f32>(entry.normal_scale));
            }
            if (debug.view_mode == DebugMode_Roughness) {
                textureStore(out_debug, global_id.xy, vec4<f32>(material.roughness));
            }
            if (debug.view_mode == DebugMode_SpecularF0) {
                textureStore(out_debug, global_id.xy, vec4<f32>(material.specular_f0, 0.0));
            }
            if (debug.view_mode == DebugMode_Emissive) {
                textureStore(out_debug, global_id.xy, vec4<f32>(emissive, 0.0));
            }
            if (debug.view_mode == DebugMode_GeometryNormal) {
                textureStore(out_debug, global_id.xy, vec4<f32>(normal_geo, 0.0));
            }
            if (debug.view_mode == DebugMode_ShadingNormal) {
                textureStore(out_debug, global_id.xy, vec4<f32>(normal, 0.0));
            }
            if (debug.view_mode == DebugMode_HitConsistency) {
                let reprojected = get_projected_pixel(camera, hit_position);
                let barycentrics_pos_diff = (intersection.object_to_world * position_object).xyz - hit_position;
                let camera_projection_diff = vec2<f32>(global_id.xy) - vec2<f32>(reprojected);
                let consistency = vec4<f32>(length(barycentrics_pos_diff), length(camera_projection_diff), 0.0, 0.0);
                textureStore(out_debug, global_id.xy, consistency);
            }
        }

        let prev_position_object = vec4<f32>(prev_positions_object * barycentrics, 1.0);
        let prev_position = (entry.prev_object_to_world * prev_position_object).xyz;
        let prev_screen = get_projected_pixel_float(prev_camera, prev_position);
        //TODO: consider just storing integers here?
        //TODO: technically this "0.5" is just a waste compute on both packing and unpacking
        motion = prev_screen - vec2<f32>(global_id.xy) - 0.5;
        if (WRITE_DEBUG_IMAGE && debug.view_mode == DebugMode_Motion) {
            textureStore(out_debug, global_id.xy, vec4<f32>(motion * MOTION_SCALE + vec2<f32>(0.5), 0.0, 1.0));
        }
    } else {
        if (enable_debug) {
            debug_buf.entry = DebugEntry();
        }
    }

    // TODO: option to avoid writing data for the sky
    textureStore(out_depth, global_id.xy, vec4<f32>(depth, 0.0, 0.0, 0.0));
    textureStore(out_basis, global_id.xy, basis);
    textureStore(out_flat_normal, global_id.xy, vec4<f32>(flat_normal, 0.0));
    textureStore(out_diffuse_albedo, global_id.xy, vec4<f32>(material.diffuse_albedo, 0.0));
    textureStore(out_specular_f0, global_id.xy, vec4<f32>(material.specular_f0, material.roughness));
    textureStore(out_emissive, global_id.xy, vec4<f32>(emissive, 0.0));
    textureStore(out_motion, global_id.xy, vec4<f32>(motion * MOTION_SCALE, 0.0, 0.0));
}
