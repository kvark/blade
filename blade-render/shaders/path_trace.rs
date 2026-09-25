use super::brdf::*;
use super::camera::*;
use super::config::*;
use super::debug_param::*;
use super::env_importance::*;
use super::env_light::*;
use super::hit::*;
use super::quaternion::*;
use super::random::*;
use super::sampling::*;
use super::vertex::*;
use synaga_shader::*;

pub const ROULETTE_START: u32 = 4u32;

pub const MAX_RADIANCE: f32 = 1.0e6;

#[derive(Clone, Copy, Default)]
pub struct PathTraceParams {
    pub frame_index: u32,
    // light samples taken at every vertex of a path
    pub num_environment_samples: u32,
    // material samples taken per pixel, i.e. the number of paths
    pub num_brdf_samples: u32,
    pub max_bounces: u32,
    // stop accumulating at this many samples, 0 for no limit
    pub max_accumulated_samples: u32,
    pub t_start: f32,
    pub environment_importance_sampling: u32,
    // when set, the previous accumulation is discarded
    pub reset_accumulation: u32,
    pub jitter_primary_rays: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[derive(Clone, Copy, Default)]
pub struct PathVertex {
    pub position: vec3,
    // Normal of the triangle, pointing outwards.
    pub flat_normal: vec3,
    // Interpolated normal with the normal map applied.
    pub normal: vec3,
    pub material: Material,
    pub emissive: vec3,
}

#[derive(Clone, Copy, Default)]
pub struct PathRadiance {
    pub total: vec3,
    pub diffuse: vec3,
    pub specular: vec3,
    pub emissive: vec3,
}

pub static camera: Uniform<CameraParams> = binding();

pub static parameters: Uniform<PathTraceParams> = binding();

pub static acc_struct: acceleration_structure = binding();

pub static accumulator: texture_storage_2d<Rgba32Float, ReadWrite> = binding();

pub static accumulator_diffuse: texture_storage_2d<Rgba32Float, ReadWrite> = binding();

pub static accumulator_specular: texture_storage_2d<Rgba32Float, ReadWrite> = binding();

pub static accumulator_emissive: texture_storage_2d<Rgba32Float, ReadWrite> = binding();

#[shader]
pub fn trace_ray(position: vec3, direction: vec3, t_min: f32) -> RayIntersection {
    let mut rq = ray_query::default();
    rayQueryInitialize(
        rq,
        &acc_struct,
        RayDesc {
            flags: RAY_FLAG_CULL_NO_OPAQUE,
            cull_mask: 0xFFu32,
            tmin: t_min,
            tmax: camera.depth,
            origin: position,
            dir: direction,
        },
    );
    rayQueryProceed(rq);
    return rayQueryGetCommittedIntersection(rq);
}

#[shader]
pub fn is_occluded(position: vec3, direction: vec3) -> bool {
    let mut rq = ray_query::default();
    let flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_CULL_NO_OPAQUE;
    rayQueryInitialize(
        rq,
        &acc_struct,
        RayDesc {
            flags: flags,
            cull_mask: 0xFFu32,
            tmin: parameters.t_start,
            tmax: camera.depth,
            origin: position,
            dir: direction,
        },
    );
    rayQueryProceed(rq);
    return rayQueryGetCommittedIntersection(rq).kind != RAY_QUERY_INTERSECTION_NONE;
}

#[shader]
pub fn resolve_hit(intersection: RayIntersection) -> PathVertex {
    let entry =
        hit_entries[(intersection.instance_custom_data + intersection.geometry_index) as usize];
    let indices = fetch_triangle_indices(entry, intersection.primitive_index);

    let vertices = [
        (vertex_buffers[(entry.vertex_buf) as usize].data)[(indices.x) as usize],
        (vertex_buffers[(entry.vertex_buf) as usize].data)[(indices.y) as usize],
        (vertex_buffers[(entry.vertex_buf) as usize].data)[(indices.z) as usize],
    ];

    let positions_object = entry.geometry_to_object
        * mat3x4(
            (vertices[0].position).extend(1.0),
            (vertices[1].position).extend(1.0),
            (vertices[2].position).extend(1.0),
        );
    let positions = intersection.object_to_world
        * mat3x4(
            (positions_object[0]).extend(1.0),
            (positions_object[1]).extend(1.0),
            (positions_object[2]).extend(1.0),
        );

    let barycentrics = make_barycentrics(intersection.barycentrics);
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
    let tangent_space_world = hit_tangent_space(
        entry,
        intersection.object_to_world,
        normal_geo,
        tangent_geo,
        vertices[0].bitangent_sign,
    );

    let lod = 0.0; //TODO: ray differentials
    let mut vertex = PathVertex::default();
    vertex.position = positions * barycentrics;
    vertex.flat_normal = hit_winding(entry)
        * normalize(cross(
            positions[1].xyz() - positions[0].xyz(),
            positions[2].xyz() - positions[0].xyz(),
        ));
    let normal_local = sample_hit_normal_map(entry, tex_coords, lod, 0u32);
    vertex.normal = normalize(tangent_space_world * normal_local);
    vertex.material = sample_hit_material(entry, tex_coords, lod, 0u32);
    vertex.emissive = sample_hit_emissive(entry, tex_coords, lod, 0u32);
    return vertex;
}

#[shader]
pub fn mis_weight(count: f32, pdf: f32, other_count: f32, other_pdf: f32) -> f32 {
    let total = count * pdf + other_count * other_pdf;
    // Same reason as `divide_if_positive` in the ReSTIR shader: the unselected
    // arm of a select is still evaluated, and a zero total is 0/0.
    if (total > 0.0) {
        return count * pdf / total;
    } else {
        return 0.0;
    }
}

#[shader]
pub fn zero_path_radiance() -> PathRadiance {
    return PathRadiance {
        total: vec3::splat(0.0),
        diffuse: vec3::splat(0.0),
        specular: vec3::splat(0.0),
        emissive: vec3::splat(0.0),
    };
}

#[shader]
pub fn trace_path(start_dir: vec3, rng: &mut RandomState) -> PathRadiance {
    let importance = parameters.environment_importance_sampling != 0u32;
    let num_light = (parameters.num_environment_samples) as f32;
    let mut radiance = zero_path_radiance();
    let mut primary_albedo = vec3::splat(1.0);
    // Throughput after the primary response, kept as two paths so everything
    // found at later vertices can still be attributed to that first lobe.
    let mut diffuse_throughput = vec3::splat(0.0);
    let mut specular_throughput = vec3::splat(0.0);
    let mut position = camera.position;
    let mut direction = start_dir;
    // Density of the sample that generated the current ray, which is
    // needed to weight the environment it may run into. Negative for
    // the camera ray, since there is no other way to generate it.
    let mut bsdf_pdf = -1.0;
    let mut t_min = 0.0;

    for bounce in (0u32)..=(parameters.max_bounces) {
        let intersection = trace_ray(position, direction, t_min);
        if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
            if (bsdf_pdf < 0.0) {
                // The G-buffer represents the sky as a white diffuse surface.
                radiance.diffuse += evaluate_environment_background(direction);
            } else {
                // The light sampling at the previous vertex could have found
                // this direction as well, so the two have to be weighted.
                let light_pdf = compute_light_pdf(map_equirect_dir_to_uv(direction), importance);
                let weight = mis_weight(1.0, bsdf_pdf, num_light, light_pdf);
                let incoming = evaluate_environment(direction) * weight;
                radiance.diffuse += diffuse_throughput * incoming;
                radiance.specular += specular_throughput * incoming;
            }
            break;
        }

        let vertex = resolve_hit(intersection);
        let view_dir = -direction;
        if (bounce == 0u32) {
            primary_albedo = vertex.material.diffuse_albedo;
            radiance.emissive += vertex.emissive;
        } else {
            radiance.diffuse += diffuse_throughput * vertex.emissive;
            radiance.specular += specular_throughput * vertex.emissive;
        }
        position = vertex.position;
        t_min = parameters.t_start;

        // Whether the path will be extended by a BSDF sampled ray. When it
        // will not, next event estimation is the only strategy that can find
        // the light at this vertex, so it has to carry the whole contribution
        // instead of the share the balance heuristic would leave it.
        let will_extend = bounce < parameters.max_bounces && parameters.num_brdf_samples != 0u32;
        let bsdf_count = select(0.0, 1.0, will_extend);

        // Next event estimation: connect to the environment light.
        for i in (0u32)..(parameters.num_environment_samples) {
            let ls = sample_light(importance, rng);
            if (ls.pdf <= 0.0) {
                continue;
            }
            let light_dir = map_equirect_uv_to_dir(ls.uv);
            let lobes = evaluate_brdf(vertex.material, vertex.normal, view_dir, light_dir);
            if (dot(light_dir, vertex.flat_normal) <= 0.0
                || is_brdf_black(lobes)
                || is_occluded(position, light_dir))
            {
                continue;
            }
            let other_pdf = compute_bsdf_pdf(vertex.material, vertex.normal, view_dir, light_dir);
            let weight =
                mis_weight(num_light, ls.pdf, bsdf_count, other_pdf) / (num_light * ls.pdf);
            let incoming = ls.radiance * weight;
            if (bounce == 0u32) {
                radiance.diffuse += lobes.diffuse * incoming;
                radiance.specular += lobes.specular * incoming;
            } else {
                let bsdf = vertex.material.diffuse_albedo * lobes.diffuse + lobes.specular;
                radiance.diffuse += diffuse_throughput * bsdf * incoming;
                radiance.specular += specular_throughput * bsdf * incoming;
            }
        }

        if (!will_extend) {
            // The next event estimation above was the last thing to do here.
            break;
        }

        // Extend the path along a direction drawn from the material.
        let bs = sample_bsdf(vertex.material, vertex.normal, view_dir, rng);
        if (bs.pdf <= 0.0 || dot(bs.dir, vertex.flat_normal) <= 0.0) {
            break;
        }
        let lobes = evaluate_brdf(vertex.material, vertex.normal, view_dir, bs.dir);
        if (bounce == 0u32) {
            diffuse_throughput = vec3::splat(lobes.diffuse / bs.pdf);
            specular_throughput = lobes.specular / bs.pdf;
        } else {
            let bsdf = vertex.material.diffuse_albedo * lobes.diffuse + lobes.specular;
            diffuse_throughput *= bsdf / bs.pdf;
            specular_throughput *= bsdf / bs.pdf;
        }
        bsdf_pdf = bs.pdf;
        direction = bs.dir;

        // Russian roulette on the remaining energy.
        if (bounce >= ROULETTE_START) {
            let throughput = primary_albedo * diffuse_throughput + specular_throughput;
            let probability = clamp(compute_luminocity(throughput), 0.05, 1.0);
            if (random_gen(rng) >= probability) {
                break;
            }
            diffuse_throughput /= probability;
            specular_throughput /= probability;
        }
        let throughput = primary_albedo * diffuse_throughput + specular_throughput;
        if (all(throughput.cmple(vec3::splat(0.0)))) {
            break;
        }
    }

    // A single bad path would poison the accumulator forever.
    radiance.total = primary_albedo * radiance.diffuse + radiance.specular + radiance.emissive;
    let is_finite = all(radiance.total.cmpeq(radiance.total))
        && all(radiance.diffuse.cmpeq(radiance.diffuse))
        && all(radiance.specular.cmpeq(radiance.specular))
        && all(radiance.emissive.cmpeq(radiance.emissive));
    if (!is_finite) {
        return zero_path_radiance();
    }
    // Scale the split and total together, preserving exact reconstruction.
    let scale = min(
        vec3::splat(1.0),
        vec3::splat(MAX_RADIANCE) / max(radiance.total, vec3::splat(1.0e-20)),
    );
    radiance.total *= scale;
    radiance.diffuse *= scale;
    radiance.specular *= scale;
    radiance.emissive *= scale;
    return radiance;
}

#[compute]
#[workgroup_size(8, 4)]
pub fn main(#[builtin(global_invocation_id)] global_id: vec3u) {
    if (any(global_id.xy().cmpge(camera.target_size))) {
        return;
    }

    let mut total = vec4::splat(0.0);
    let mut total_diffuse = vec4::splat(0.0);
    let mut total_specular = vec4::splat(0.0);
    let mut total_emissive = vec4::splat(0.0);
    if (parameters.reset_accumulation == 0u32) {
        total = textureLoadStorage(&accumulator, global_id.xy());
        if (parameters.max_accumulated_samples != 0u32
            && total.w >= (parameters.max_accumulated_samples) as f32)
        {
            // Converged enough, leave the accumulator alone.
            return;
        }
        total_diffuse = textureLoadStorage(&accumulator_diffuse, global_id.xy());
        total_specular = textureLoadStorage(&accumulator_specular, global_id.xy());
        total_emissive = textureLoadStorage(&accumulator_emissive, global_id.xy());
    }

    let global_index = global_id.y * camera.target_size.x + global_id.x;
    let mut rng = random_init(global_index, parameters.frame_index);

    // Each of the material samples at the primary hit starts a path of its own.
    let num_paths = max(parameters.num_brdf_samples, 1u32);
    let mut sum = zero_path_radiance();
    for i in (0u32)..(num_paths) {
        // Sparse captures may need the radiance ray to agree with a separately
        // rasterized center-sampled G-buffer. References retain stochastic
        // subpixel coverage for antialiasing.
        let jitter = select(
            vec2::splat(0.5),
            vec2(random_gen(&mut rng), random_gen(&mut rng)),
            parameters.jitter_primary_rays != 0u32,
        );
        let ray_dir = get_ray_direction_at(*camera, vec2::from(global_id.xy()) + jitter);
        let sample = trace_path(ray_dir, &mut rng);
        sum.total += sample.total;
        sum.diffuse += sample.diffuse;
        sum.specular += sample.specular;
        sum.emissive += sample.emissive;
    }

    let count = (num_paths) as f32;
    textureStore(
        &accumulator,
        global_id.xy(),
        total + (sum.total).extend(count),
    );
    textureStore(
        &accumulator_diffuse,
        global_id.xy(),
        total_diffuse + (sum.diffuse).extend(count),
    );
    textureStore(
        &accumulator_specular,
        global_id.xy(),
        total_specular + (sum.specular).extend(count),
    );
    textureStore(
        &accumulator_emissive,
        global_id.xy(),
        total_emissive + (sum.emissive).extend(count),
    );
}
