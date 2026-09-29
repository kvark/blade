use super::brdf::Material;
use super::camera::{
    CameraParams, get_projected_pixel, get_projected_pixel_float, get_ray_direction,
};
use super::config::{DebugDrawFlags, DebugMode};
use super::debug::{DebugEntry, debug_buf, debug_line};
use super::debug_param::DebugParams;
use super::gbuf::{MOTION_SCALE, WRITE_DEBUG_IMAGE};
use super::hit::{
    HitEntry, fetch_triangle_indices, hit_entries, hit_normal, hit_tangent_space, hit_winding,
    make_barycentrics, sample_hit_emissive, sample_hit_material, sample_hit_normal_map,
    vertex_buffers,
};
use super::quaternion::{qrot, shortest_arc_quat};
use super::vertex::decode_normal;
use synaga_shader::*;

static camera: Uniform<CameraParams> = binding();
static prev_camera: Uniform<CameraParams> = binding();
static debug: Uniform<DebugParams> = binding();
static acc_struct: AccelerationStructure = binding();
static out_depth: TextureStorage2D<R32Float, Write> = binding();
static out_flat_normal: TextureStorage2D<Rgba8Snorm, Write> = binding();
static out_basis: TextureStorage2D<Rgba8Snorm, Write> = binding();
static out_diffuse_albedo: TextureStorage2D<Rgba8Unorm, Write> = binding();
static out_specular_f0: TextureStorage2D<Rgba8Unorm, Write> = binding();
static out_emissive: TextureStorage2D<Rgba16Float, Write> = binding();
static out_motion: TextureStorage2D<Rg16Float, Write> = binding();
static out_debug: TextureStorage2D<Rgba8Unorm, Write> = binding();

fn debug_raw_normal(
    pos: Vec3,
    normal_raw: u32,
    entry: HitEntry,
    object_to_world: Mat4x3,
    debug_len: f32,
    color: u32,
) {
    let nw = hit_normal(entry, object_to_world, decode_normal(normal_raw));
    debug_line(pos, pos + debug_len * nw, color);
}

#[entry_point(compute, threads(8, 4))]
fn main(global_invocation_id: Vec3<u32>) {
    if global_invocation_id.xy().cmpge(camera.target_size).any() {
        return;
    }
    if WRITE_DEBUG_IMAGE && debug.view_mode != DebugMode::Final {
        out_debug.store(global_invocation_id.xy(), Vec4::ZERO);
    }

    let mut rq = RayQuery::default();
    let ray_dir = get_ray_direction(*camera, global_invocation_id.xy().cast::<i32>());
    rq.initialize(
        &acc_struct,
        RayDesc {
            flags: RAY_FLAG_CULL_NO_OPAQUE,
            cull_mask: 0xFF,
            tmin: 0.0,
            tmax: camera.depth,
            origin: camera.position,
            dir: ray_dir,
        },
    );
    rq.proceed();
    let intersection = rq.committed_intersection();

    let mut depth = 0.0;
    let mut basis = Vec4::ZERO;
    let mut flat_normal = Vec3::ZERO;
    // Note: the sky is fully diffuse and white, so that the environment
    // survives the modulation in the post-processing.
    let mut material = Material {
        diffuse_albedo: Vec3::ONE,
        specular_f0: Vec3::ZERO,
        roughness: 0.0,
    };
    let mut emissive = Vec3::ZERO;
    let mut motion = Vec2::ZERO;
    let enable_debug = global_invocation_id.xy().cmpeq(debug.mouse_pos).all();

    if intersection.kind != RAY_QUERY_INTERSECTION_NONE {
        let entry =
            hit_entries[(intersection.instance_custom_data + intersection.geometry_index) as usize];
        depth = intersection.t;

        let indices = fetch_triangle_indices(entry, intersection.primitive_index);

        let vertices = [
            vertex_buffers[entry.vertex_buf as usize].data[indices.x as usize],
            vertex_buffers[entry.vertex_buf as usize].data[indices.y as usize],
            vertex_buffers[entry.vertex_buf as usize].data[indices.z as usize],
        ];

        let prev_vertices = [
            vertex_buffers[entry.prev_vertex_buf as usize].data[indices.x as usize],
            vertex_buffers[entry.prev_vertex_buf as usize].data[indices.y as usize],
            vertex_buffers[entry.prev_vertex_buf as usize].data[indices.z as usize],
        ];

        let positions_object = entry.geometry_to_object
            * mat3x4(
                vertices[0].position.extend(1.0),
                vertices[1].position.extend(1.0),
                vertices[2].position.extend(1.0),
            );
        let prev_positions_object = entry.prev_geometry_to_object
            * mat3x4(
                prev_vertices[0].position.extend(1.0),
                prev_vertices[1].position.extend(1.0),
                prev_vertices[2].position.extend(1.0),
            );
        let positions = intersection.object_to_world
            * mat3x4(
                positions_object[0].extend(1.0),
                positions_object[1].extend(1.0),
                positions_object[2].extend(1.0),
            );
        flat_normal = hit_winding(entry)
            * (positions[1].xyz() - positions[0].xyz())
                .cross(positions[2].xyz() - positions[0].xyz())
                .normalize();

        let barycentrics = make_barycentrics(intersection.barycentrics);
        let position_object = (positions_object * barycentrics).extend(1.0);
        let tex_coords = mat3x2(
            vertices[0].tex_coords,
            vertices[1].tex_coords,
            vertices[2].tex_coords,
        ) * barycentrics;
        let normal_geo = (mat3(
            decode_normal(vertices[0].normal),
            decode_normal(vertices[1].normal),
            decode_normal(vertices[2].normal),
        ) * barycentrics)
            .normalize();
        let tangent_geo = (mat3(
            decode_normal(vertices[0].tangent),
            decode_normal(vertices[1].tangent),
            decode_normal(vertices[2].tangent),
        ) * barycentrics)
            .normalize();
        let lod = 0.0; //TODO: this is actually complicated

        let tangent_space_world = hit_tangent_space(
            entry,
            intersection.object_to_world,
            normal_geo,
            tangent_geo,
            vertices[0].bitangent_sign,
        );
        let normal_local = sample_hit_normal_map(entry, tex_coords, lod, debug.texture_flags);
        let normal = tangent_space_world * normal_local;
        basis = shortest_arc_quat(vec3(0.0, 0.0, 1.0), normal.normalize());

        let hit_position = camera.position + intersection.t * ray_dir;
        if enable_debug {
            debug_buf.get_mut().entry.custom_index = intersection.instance_custom_data;
            debug_buf.get_mut().entry.depth = intersection.t;
            debug_buf.get_mut().entry.tex_coords = tex_coords;
            debug_buf.get_mut().entry.base_color_texture = entry.base_color_texture;
            debug_buf.get_mut().entry.normal_texture = entry.normal_texture;
            debug_buf.get_mut().entry.position = hit_position;
            debug_buf.get_mut().entry.flat_normal = flat_normal;
        }
        if enable_debug && debug.draw_flags.contains(DebugDrawFlags::SPACE) {
            let normal_w = 0.15 * intersection.t * tangent_space_world[2];
            let tangent_w = 0.05 * intersection.t * tangent_space_world[0];
            let bitangent_w = 0.05 * intersection.t * tangent_space_world[1];
            debug_line(hit_position, hit_position + normal_w, 0xFF8000);
            debug_line(
                hit_position - 0.5 * tangent_w,
                hit_position + tangent_w,
                0x8080FF,
            );
            debug_line(
                hit_position - 0.5 * bitangent_w,
                hit_position + bitangent_w,
                0x80FF80,
            );
        }
        if enable_debug && debug.draw_flags.contains(DebugDrawFlags::GEOMETRY) {
            let debug_len = intersection.t * 0.2;
            debug_line(positions[0].xyz(), positions[1].xyz(), 0x00FFFF);
            debug_line(positions[1].xyz(), positions[2].xyz(), 0x00FFFF);
            debug_line(positions[2].xyz(), positions[0].xyz(), 0x00FFFF);
            let poly_center = (positions[0].xyz() + positions[1].xyz() + positions[2].xyz()) / 3.0;
            debug_line(
                poly_center,
                poly_center + 0.2 * debug_len * flat_normal,
                0xFF00FF,
            );
            // note: dynamic indexing into positions isn't allowed by WGSL yet
            debug_raw_normal(
                positions[0].xyz(),
                vertices[0].normal,
                entry,
                intersection.object_to_world,
                0.5 * debug_len,
                0xFFFF00,
            );
            debug_raw_normal(
                positions[1].xyz(),
                vertices[1].normal,
                entry,
                intersection.object_to_world,
                0.5 * debug_len,
                0xFFFF00,
            );
            debug_raw_normal(
                positions[2].xyz(),
                vertices[2].normal,
                entry,
                intersection.object_to_world,
                0.5 * debug_len,
                0xFFFF00,
            );
            // draw tangent space
            debug_line(
                hit_position,
                hit_position + debug_len * qrot(basis, vec3(1.0, 0.0, 0.0)),
                0x0000FF,
            );
            debug_line(
                hit_position,
                hit_position + debug_len * qrot(basis, vec3(0.0, 1.0, 0.0)),
                0x00FF00,
            );
            debug_line(
                hit_position,
                hit_position + debug_len * qrot(basis, vec3(0.0, 0.0, 1.0)),
                0xFF0000,
            );
        }

        material = sample_hit_material(entry, tex_coords, lod, debug.texture_flags);
        emissive = sample_hit_emissive(entry, tex_coords, lod, debug.texture_flags);

        if WRITE_DEBUG_IMAGE {
            if debug.view_mode == DebugMode::DiffuseAlbedoTexture {
                out_debug.store(
                    global_invocation_id.xy(),
                    material.diffuse_albedo.extend(0.0),
                );
            }
            if debug.view_mode == DebugMode::DiffuseAlbedoFactor {
                out_debug.store(
                    global_invocation_id.xy(),
                    unpack4x8unorm(entry.base_color_factor),
                );
            }
            if debug.view_mode == DebugMode::NormalTexture {
                out_debug.store(global_invocation_id.xy(), normal_local.extend(0.0));
            }
            if debug.view_mode == DebugMode::NormalScale {
                out_debug.store(global_invocation_id.xy(), Vec4::splat(entry.normal_scale));
            }
            if debug.view_mode == DebugMode::Roughness {
                out_debug.store(global_invocation_id.xy(), Vec4::splat(material.roughness));
            }
            if debug.view_mode == DebugMode::SpecularF0 {
                out_debug.store(global_invocation_id.xy(), material.specular_f0.extend(0.0));
            }
            if debug.view_mode == DebugMode::Emissive {
                out_debug.store(global_invocation_id.xy(), emissive.extend(0.0));
            }
            if debug.view_mode == DebugMode::GeometryNormal {
                out_debug.store(global_invocation_id.xy(), normal_geo.extend(0.0));
            }
            if debug.view_mode == DebugMode::ShadingNormal {
                out_debug.store(global_invocation_id.xy(), normal.extend(0.0));
            }
            if debug.view_mode == DebugMode::HitConsistency {
                let reprojected = get_projected_pixel(*camera, hit_position);
                let barycentrics_pos_diff =
                    (intersection.object_to_world * position_object).xyz() - hit_position;
                let camera_projection_diff =
                    global_invocation_id.xy().cast::<f32>() - reprojected.cast::<f32>();
                let consistency = vec4(
                    barycentrics_pos_diff.length(),
                    camera_projection_diff.length(),
                    0.0,
                    0.0,
                );
                out_debug.store(global_invocation_id.xy(), consistency);
            }
        }

        let prev_position_object = (prev_positions_object * barycentrics).extend(1.0);
        let prev_position = (entry.prev_object_to_world * prev_position_object).xyz();
        let prev_screen = get_projected_pixel_float(*prev_camera, prev_position);
        //TODO: consider just storing integers here?
        //TODO: technically this "0.5" is just a waste compute on both packing and unpacking
        motion = prev_screen - Vec2::from(global_invocation_id.xy()) - 0.5;
        if WRITE_DEBUG_IMAGE && debug.view_mode == DebugMode::Motion {
            out_debug.store(
                global_invocation_id.xy(),
                (motion * MOTION_SCALE + 0.5).extend(0.0).extend(1.0),
            );
        }
    } else {
        if enable_debug {
            debug_buf.get_mut().entry = DebugEntry::default();
        }
    }

    // TODO: option to avoid writing data for the sky
    out_depth.store(global_invocation_id.xy(), vec4(depth, 0.0, 0.0, 0.0));
    out_basis.store(global_invocation_id.xy(), basis);
    out_flat_normal.store(global_invocation_id.xy(), flat_normal.extend(0.0));
    out_diffuse_albedo.store(
        global_invocation_id.xy(),
        material.diffuse_albedo.extend(0.0),
    );
    out_specular_f0.store(
        global_invocation_id.xy(),
        material.specular_f0.extend(material.roughness),
    );
    out_emissive.store(global_invocation_id.xy(), emissive.extend(0.0));
    out_motion.store(
        global_invocation_id.xy(),
        (motion * MOTION_SCALE).extend(0.0).extend(0.0),
    );
}
