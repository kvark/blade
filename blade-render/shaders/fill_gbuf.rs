use super::brdf::*;
use super::camera::*;
use super::config::*;
use super::debug::*;
use super::debug_param::*;
use super::gbuf::*;
use super::hit::*;
use super::quaternion::*;
use super::vertex::*;
use synaga_shader::*;

pub static camera: Uniform<CameraParams> = binding();

pub static prev_camera: Uniform<CameraParams> = binding();

pub static debug: Uniform<DebugParams> = binding();

pub static acc_struct: acceleration_structure = binding();

pub static out_depth: texture_storage_2d<R32Float, Write> = binding();

pub static out_flat_normal: texture_storage_2d<Rgba8Snorm, Write> = binding();

pub static out_basis: texture_storage_2d<Rgba8Snorm, Write> = binding();

pub static out_diffuse_albedo: texture_storage_2d<Rgba8Unorm, Write> = binding();

pub static out_specular_f0: texture_storage_2d<Rgba8Unorm, Write> = binding();

pub static out_emissive: texture_storage_2d<Rgba16Float, Write> = binding();

pub static out_motion: texture_storage_2d<Rg16Float, Write> = binding();

pub static out_debug: texture_storage_2d<Rgba8Unorm, Write> = binding();

#[shader]
pub fn debug_raw_normal(
    pos: vec3,
    normal_raw: u32,
    entry: HitEntry,
    object_to_world: mat4x3,
    debug_len: f32,
    color: u32,
) {
    let nw = hit_normal(entry, object_to_world, decode_normal(normal_raw));
    debug_line(pos, pos + debug_len * nw, color);
}

#[compute]
#[workgroup_size(8, 4)]
pub fn main(#[builtin(global_invocation_id)] global_id: vec3u) {
    if (any(global_id.xy().cmpge(camera.target_size))) {
        return;
    }
    if (WRITE_DEBUG_IMAGE && debug.view_mode != DebugMode_Final) {
        textureStore(&out_debug, global_id.xy(), vec4::splat(0.0));
    }

    let mut rq = ray_query::default();
    let ray_dir = get_ray_direction(*camera, vec2i::from(global_id.xy()));
    rayQueryInitialize(
        rq,
        &acc_struct,
        RayDesc {
            flags: RAY_FLAG_CULL_NO_OPAQUE,
            cull_mask: 0xFFu32,
            tmin: 0.0,
            tmax: camera.depth,
            origin: camera.position,
            dir: ray_dir,
        },
    );
    rayQueryProceed(rq);
    let intersection = rayQueryGetCommittedIntersection(rq);

    let mut depth = 0.0;
    let mut basis = vec4::splat(0.0);
    let mut flat_normal = vec3::splat(0.0);
    // Note: the sky is fully diffuse and white, so that the environment
    // survives the modulation in the post-processing.
    let mut material = Material {
        diffuse_albedo: vec3::splat(1.0),
        specular_f0: vec3::splat(0.0),
        roughness: 0.0,
    };
    let mut emissive = vec3::splat(0.0);
    let mut motion = vec2::splat(0.0);
    let enable_debug = all(global_id.xy().cmpeq(debug.mouse_pos));

    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        let entry =
            hit_entries[(intersection.instance_custom_data + intersection.geometry_index) as usize];
        depth = intersection.t;

        let indices = fetch_triangle_indices(entry, intersection.primitive_index);

        let vertices = [
            (vertex_buffers[(entry.vertex_buf) as usize].data)[(indices.x) as usize],
            (vertex_buffers[(entry.vertex_buf) as usize].data)[(indices.y) as usize],
            (vertex_buffers[(entry.vertex_buf) as usize].data)[(indices.z) as usize],
        ];

        let prev_vertices = [
            (vertex_buffers[(entry.prev_vertex_buf) as usize].data)[(indices.x) as usize],
            (vertex_buffers[(entry.prev_vertex_buf) as usize].data)[(indices.y) as usize],
            (vertex_buffers[(entry.prev_vertex_buf) as usize].data)[(indices.z) as usize],
        ];

        let positions_object = entry.geometry_to_object
            * mat3x4(
                (vertices[0].position).extend(1.0),
                (vertices[1].position).extend(1.0),
                (vertices[2].position).extend(1.0),
            );
        let prev_positions_object = entry.prev_geometry_to_object
            * mat3x4(
                (prev_vertices[0].position).extend(1.0),
                (prev_vertices[1].position).extend(1.0),
                (prev_vertices[2].position).extend(1.0),
            );
        let positions = intersection.object_to_world
            * mat3x4(
                (positions_object[0]).extend(1.0),
                (positions_object[1]).extend(1.0),
                (positions_object[2]).extend(1.0),
            );
        flat_normal = hit_winding(entry)
            * normalize(cross(
                positions[1].xyz() - positions[0].xyz(),
                positions[2].xyz() - positions[0].xyz(),
            ));

        let barycentrics = make_barycentrics(intersection.barycentrics);
        let position_object = (positions_object * barycentrics).extend(1.0);
        let tex_coords = mat3x2(
            vertices[0].tex_coords,
            vertices[1].tex_coords,
            vertices[2].tex_coords,
        ) * barycentrics;
        let normal_geo = normalize(
            mat3x3(
                decode_normal(vertices[0].normal),
                decode_normal(vertices[1].normal),
                decode_normal(vertices[2].normal),
            ) * barycentrics,
        );
        let tangent_geo = normalize(
            mat3x3(
                decode_normal(vertices[0].tangent),
                decode_normal(vertices[1].tangent),
                decode_normal(vertices[2].tangent),
            ) * barycentrics,
        );
        let lod = 0.0; //TODO: this is actually complicated

        let tangent_space_world = hit_tangent_space(
            entry,
            intersection.object_to_world,
            normal_geo,
            tangent_geo,
            vertices[0].bitangent_sign,
        );
        let normal_local = sample_hit_normal_map(entry, tex_coords, lod, debug.texture_flags);
        let mut normal = tangent_space_world * normal_local;
        basis = shortest_arc_quat(vec3(0.0, 0.0, 1.0), normalize(normal));

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
        if (enable_debug && (debug.draw_flags & DebugDrawFlags_SPACE) != 0u32) {
            let normal_w = 0.15 * intersection.t * tangent_space_world[2];
            let tangent_w = 0.05 * intersection.t * tangent_space_world[0];
            let bitangent_w = 0.05 * intersection.t * tangent_space_world[1];
            debug_line(hit_position, hit_position + normal_w, 0xFF8000u32);
            debug_line(
                hit_position - 0.5 * tangent_w,
                hit_position + tangent_w,
                0x8080FFu32,
            );
            debug_line(
                hit_position - 0.5 * bitangent_w,
                hit_position + bitangent_w,
                0x80FF80u32,
            );
        }
        if (enable_debug && (debug.draw_flags & DebugDrawFlags_GEOMETRY) != 0u32) {
            let debug_len = intersection.t * 0.2;
            debug_line(positions[0].xyz(), positions[1].xyz(), 0x00FFFFu32);
            debug_line(positions[1].xyz(), positions[2].xyz(), 0x00FFFFu32);
            debug_line(positions[2].xyz(), positions[0].xyz(), 0x00FFFFu32);
            let poly_center = (positions[0].xyz() + positions[1].xyz() + positions[2].xyz()) / 3.0;
            debug_line(
                poly_center,
                poly_center + 0.2 * debug_len * flat_normal,
                0xFF00FFu32,
            );
            // note: dynamic indexing into positions isn't allowed by WGSL yet
            debug_raw_normal(
                positions[0].xyz(),
                vertices[0].normal,
                entry,
                intersection.object_to_world,
                0.5 * debug_len,
                0xFFFF00u32,
            );
            debug_raw_normal(
                positions[1].xyz(),
                vertices[1].normal,
                entry,
                intersection.object_to_world,
                0.5 * debug_len,
                0xFFFF00u32,
            );
            debug_raw_normal(
                positions[2].xyz(),
                vertices[2].normal,
                entry,
                intersection.object_to_world,
                0.5 * debug_len,
                0xFFFF00u32,
            );
            // draw tangent space
            debug_line(
                hit_position,
                hit_position + debug_len * qrot(basis, vec3(1.0, 0.0, 0.0)),
                0x0000FFu32,
            );
            debug_line(
                hit_position,
                hit_position + debug_len * qrot(basis, vec3(0.0, 1.0, 0.0)),
                0x00FF00u32,
            );
            debug_line(
                hit_position,
                hit_position + debug_len * qrot(basis, vec3(0.0, 0.0, 1.0)),
                0xFF0000u32,
            );
        }

        material = sample_hit_material(entry, tex_coords, lod, debug.texture_flags);
        emissive = sample_hit_emissive(entry, tex_coords, lod, debug.texture_flags);

        if (WRITE_DEBUG_IMAGE) {
            if (debug.view_mode == DebugMode_DiffuseAlbedoTexture) {
                textureStore(
                    &out_debug,
                    global_id.xy(),
                    (material.diffuse_albedo).extend(0.0),
                );
            }
            if (debug.view_mode == DebugMode_DiffuseAlbedoFactor) {
                textureStore(
                    &out_debug,
                    global_id.xy(),
                    unpack4x8unorm(entry.base_color_factor),
                );
            }
            if (debug.view_mode == DebugMode_NormalTexture) {
                textureStore(&out_debug, global_id.xy(), (normal_local).extend(0.0));
            }
            if (debug.view_mode == DebugMode_NormalScale) {
                textureStore(&out_debug, global_id.xy(), vec4::splat(entry.normal_scale));
            }
            if (debug.view_mode == DebugMode_Roughness) {
                textureStore(&out_debug, global_id.xy(), vec4::splat(material.roughness));
            }
            if (debug.view_mode == DebugMode_SpecularF0) {
                textureStore(
                    &out_debug,
                    global_id.xy(),
                    (material.specular_f0).extend(0.0),
                );
            }
            if (debug.view_mode == DebugMode_Emissive) {
                textureStore(&out_debug, global_id.xy(), (emissive).extend(0.0));
            }
            if (debug.view_mode == DebugMode_GeometryNormal) {
                textureStore(&out_debug, global_id.xy(), (normal_geo).extend(0.0));
            }
            if (debug.view_mode == DebugMode_ShadingNormal) {
                textureStore(&out_debug, global_id.xy(), (normal).extend(0.0));
            }
            if (debug.view_mode == DebugMode_HitConsistency) {
                let reprojected = get_projected_pixel(*camera, hit_position);
                let barycentrics_pos_diff =
                    (intersection.object_to_world * position_object).xyz() - hit_position;
                let camera_projection_diff = vec2::from(global_id.xy()) - vec2::from(reprojected);
                let consistency = vec4(
                    length(barycentrics_pos_diff),
                    length(camera_projection_diff),
                    0.0,
                    0.0,
                );
                textureStore(&out_debug, global_id.xy(), consistency);
            }
        }

        let prev_position_object = (prev_positions_object * barycentrics).extend(1.0);
        let prev_position = (entry.prev_object_to_world * prev_position_object).xyz();
        let prev_screen = get_projected_pixel_float(*prev_camera, prev_position);
        //TODO: consider just storing integers here?
        //TODO: technically this "0.5" is just a waste compute on both packing and unpacking
        motion = prev_screen - vec2::from(global_id.xy()) - 0.5;
        if (WRITE_DEBUG_IMAGE && debug.view_mode == DebugMode_Motion) {
            textureStore(
                &out_debug,
                global_id.xy(),
                ((motion * MOTION_SCALE + vec2::splat(0.5)).extend(0.0)).extend(1.0),
            );
        }
    } else {
        if (enable_debug) {
            debug_buf.entry = DebugEntry::default();
        }
    }

    // TODO: option to avoid writing data for the sky
    textureStore(&out_depth, global_id.xy(), vec4(depth, 0.0, 0.0, 0.0));
    textureStore(&out_basis, global_id.xy(), basis);
    textureStore(&out_flat_normal, global_id.xy(), (flat_normal).extend(0.0));
    textureStore(
        &out_diffuse_albedo,
        global_id.xy(),
        (material.diffuse_albedo).extend(0.0),
    );
    textureStore(
        &out_specular_f0,
        global_id.xy(),
        (material.specular_f0).extend(material.roughness),
    );
    textureStore(&out_emissive, global_id.xy(), (emissive).extend(0.0));
    textureStore(
        &out_motion,
        global_id.xy(),
        ((motion * MOTION_SCALE).extend(0.0)).extend(0.0),
    );
}
