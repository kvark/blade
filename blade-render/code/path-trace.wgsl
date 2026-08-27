// The canonical renderer: a brute force path tracer.
//
// There is no reuse and no denoising here, just many paths accumulated over
// the frames, so the result converges to the ground truth. It's meant as a
// reference to compare the real-time path against, not for interactive use.
enable wgpu_ray_query;
enable wgpu_binding_array;
#include "quaternion.inc.wgsl"
#include "random.inc.wgsl"
#include "camera.inc.wgsl"
#include "debug-param.inc.wgsl"
#include "brdf.inc.wgsl"
#include "sampling.inc.wgsl"
#include "env-importance.inc.wgsl"
#include "env-light.inc.wgsl"
#include "hit.inc.wgsl"

// Paths longer than this may get terminated by Russian roulette.
// Note: short paths are cheap, and rouletting them only adds noise.
const ROULETTE_START: u32 = 4u;
// Only meant to catch the infinities: a reference renderer shouldn't
// clamp the actual radiance, however bright the environment is.
const MAX_RADIANCE: f32 = 1.0e6;

struct PathTraceParams {
    frame_index: u32,
    // light samples taken at every vertex of a path
    num_environment_samples: u32,
    // material samples taken per pixel, i.e. the number of paths
    num_brdf_samples: u32,
    max_bounces: u32,
    // stop accumulating at this many samples, 0 for no limit
    max_accumulated_samples: u32,
    t_start: f32,
    environment_importance_sampling: u32,
    // when set, the previous accumulation is discarded
    reset_accumulation: u32,
    jitter_primary_rays: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<uniform> camera: CameraParams;
var<uniform> parameters: PathTraceParams;
var acc_struct: acceleration_structure;
var env_map: texture_2d<f32>;
var sampler_nearest: sampler;
// RGB is the sum of the radiance, alpha is the number of samples in it.
// The total remains separate because stochastic primary coverage cannot be
// reconstructed by multiplying the accumulated diffuse illumination by one
// centre-sampled albedo after the fact.
var accumulator: texture_storage_2d<rgba32float, read_write>;
var accumulator_diffuse: texture_storage_2d<rgba32float, read_write>;
var accumulator_specular: texture_storage_2d<rgba32float, read_write>;
var accumulator_emissive: texture_storage_2d<rgba32float, read_write>;

struct PathVertex {
    position: vec3<f32>,
    // Normal of the triangle, pointing outwards.
    flat_normal: vec3<f32>,
    // Interpolated normal with the normal map applied.
    normal: vec3<f32>,
    material: Material,
    emissive: vec3<f32>,
}

fn trace_ray(position: vec3<f32>, direction: vec3<f32>, t_min: f32) -> RayIntersection {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct,
        RayDesc(RAY_FLAG_CULL_NO_OPAQUE, 0xFFu, t_min, camera.depth, position, direction)
    );
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq);
}

fn is_occluded(position: vec3<f32>, direction: vec3<f32>) -> bool {
    var rq: ray_query;
    let flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_CULL_NO_OPAQUE;
    rayQueryInitialize(&rq, acc_struct,
        RayDesc(flags, 0xFFu, parameters.t_start, camera.depth, position, direction)
    );
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq).kind != RAY_QUERY_INTERSECTION_NONE;
}

// Resolve the geometry and the material of a hit.
//Note: this is the leaner sibling of the body of "fill-gbuf.wgsl",
// which additionally needs the tangent space and the motion vectors.
fn resolve_hit(intersection: RayIntersection) -> PathVertex {
    let entry = hit_entries[intersection.instance_custom_data + intersection.geometry_index];
    let indices = fetch_triangle_indices(entry, intersection.primitive_index);
    let vptr = &vertex_buffers[entry.vertex_buf].data;
    let vertices = array<Vertex, 3>(
        (*vptr)[indices.x],
        (*vptr)[indices.y],
        (*vptr)[indices.z],
    );

    let positions_object = entry.geometry_to_object * mat3x4(
        vec4<f32>(vertices[0].position, 1.0), vec4<f32>(vertices[1].position, 1.0), vec4<f32>(vertices[2].position, 1.0)
    );
    let positions = intersection.object_to_world * mat3x4(
        vec4<f32>(positions_object[0], 1.0), vec4<f32>(positions_object[1], 1.0), vec4<f32>(positions_object[2], 1.0)
    );

    let barycentrics = make_barycentrics(intersection.barycentrics);
    let tex_coords = mat3x2(vertices[0].tex_coords, vertices[1].tex_coords, vertices[2].tex_coords) * barycentrics;
    let normal_geo = normalize(mat3x3(decode_normal(vertices[0].normal), decode_normal(vertices[1].normal), decode_normal(vertices[2].normal)) * barycentrics);
    let tangent_geo = normalize(mat3x3(decode_normal(vertices[0].tangent), decode_normal(vertices[1].tangent), decode_normal(vertices[2].tangent)) * barycentrics);
    let tangent_space_world = hit_tangent_space(
        entry, intersection.object_to_world, normal_geo, tangent_geo, vertices[0].bitangent_sign,
    );

    let lod = 0.0; //TODO: ray differentials
    var vertex: PathVertex;
    vertex.position = positions * barycentrics;
    vertex.flat_normal = hit_winding(entry) * normalize(cross(positions[1].xyz - positions[0].xyz, positions[2].xyz - positions[0].xyz));
    let normal_local = sample_hit_normal_map(entry, tex_coords, lod, 0u);
    vertex.normal = normalize(tangent_space_world * normal_local);
    vertex.material = sample_hit_material(entry, tex_coords, lod, 0u);
    vertex.emissive = sample_hit_emissive(entry, tex_coords, lod, 0u);
    return vertex;
}

// Balance heuristic weight for a strategy taking `count` samples at density
// `pdf`, against another one taking `other_count` samples at `other_pdf`.
fn mis_weight(count: f32, pdf: f32, other_count: f32, other_pdf: f32) -> f32 {
    let total = count * pdf + other_count * other_pdf;
    return select(0.0, count * pdf / total, total > 0.0);
}

// Radiance split by the primary surface's response. `diffuse` has its primary
// albedo divided out; `specular` retains Fresnel tint. Their sum is not used to
// reconstruct `total`, since primary-ray jitter may visit several materials.
struct PathRadiance {
    total: vec3<f32>,
    diffuse: vec3<f32>,
    specular: vec3<f32>,
    emissive: vec3<f32>,
}

fn zero_path_radiance() -> PathRadiance {
    return PathRadiance(
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
    );
}

// Estimate the light arriving at the camera through a single path while
// retaining the lobe selected at its first surface.
fn trace_path(start_dir: vec3<f32>, rng: ptr<function, RandomState>) -> PathRadiance {
    let importance = parameters.environment_importance_sampling != 0u;
    let num_light = f32(parameters.num_environment_samples);
    var radiance = zero_path_radiance();
    var primary_albedo = vec3<f32>(1.0);
    // Throughput after the primary response, kept as two paths so everything
    // found at later vertices can still be attributed to that first lobe.
    var diffuse_throughput = vec3<f32>(0.0);
    var specular_throughput = vec3<f32>(0.0);
    var position = camera.position;
    var direction = start_dir;
    // Density of the sample that generated the current ray, which is
    // needed to weight the environment it may run into. Negative for
    // the camera ray, since there is no other way to generate it.
    var bsdf_pdf = -1.0;
    var t_min = 0.0;

    for (var bounce = 0u; bounce <= parameters.max_bounces; bounce += 1u) {
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
        if (bounce == 0u) {
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
        let will_extend = bounce < parameters.max_bounces && parameters.num_brdf_samples != 0u;
        let bsdf_count = select(0.0, 1.0, will_extend);

        // Next event estimation: connect to the environment light.
        for (var i = 0u; i < parameters.num_environment_samples; i += 1u) {
            let ls = sample_light(importance, rng);
            if (ls.pdf <= 0.0) {
                continue;
            }
            let light_dir = map_equirect_uv_to_dir(ls.uv);
            let lobes = evaluate_brdf(vertex.material, vertex.normal, view_dir, light_dir);
            if (dot(light_dir, vertex.flat_normal) <= 0.0 || is_brdf_black(lobes)
                || is_occluded(position, light_dir)) {
                continue;
            }
            let other_pdf = compute_bsdf_pdf(vertex.material, vertex.normal, view_dir, light_dir);
            let weight = mis_weight(num_light, ls.pdf, bsdf_count, other_pdf) / (num_light * ls.pdf);
            let incoming = ls.radiance * weight;
            if (bounce == 0u) {
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
        if (bounce == 0u) {
            diffuse_throughput = vec3<f32>(lobes.diffuse / bs.pdf);
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
        if (all(throughput <= vec3<f32>(0.0))) {
            break;
        }
    }

    // A single bad path would poison the accumulator forever.
    radiance.total = primary_albedo * radiance.diffuse + radiance.specular + radiance.emissive;
    let is_finite = all(radiance.total == radiance.total)
        && all(radiance.diffuse == radiance.diffuse)
        && all(radiance.specular == radiance.specular)
        && all(radiance.emissive == radiance.emissive);
    if (!is_finite) {
        return zero_path_radiance();
    }
    // Scale the split and total together, preserving exact reconstruction.
    let scale = min(vec3<f32>(1.0), vec3<f32>(MAX_RADIANCE) / max(radiance.total, vec3<f32>(1.0e-20)));
    radiance.total *= scale;
    radiance.diffuse *= scale;
    radiance.specular *= scale;
    radiance.emissive *= scale;
    return radiance;
}

@compute @workgroup_size(8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id.xy >= camera.target_size)) {
        return;
    }

    var total = vec4<f32>(0.0);
    var total_diffuse = vec4<f32>(0.0);
    var total_specular = vec4<f32>(0.0);
    var total_emissive = vec4<f32>(0.0);
    if (parameters.reset_accumulation == 0u) {
        total = textureLoad(accumulator, global_id.xy);
        if (parameters.max_accumulated_samples != 0u
            && total.w >= f32(parameters.max_accumulated_samples)) {
            // Converged enough, leave the accumulator alone.
            return;
        }
        total_diffuse = textureLoad(accumulator_diffuse, global_id.xy);
        total_specular = textureLoad(accumulator_specular, global_id.xy);
        total_emissive = textureLoad(accumulator_emissive, global_id.xy);
    }

    let global_index = global_id.y * camera.target_size.x + global_id.x;
    var rng = random_init(global_index, parameters.frame_index);

    // Each of the material samples at the primary hit starts a path of its own.
    let num_paths = max(parameters.num_brdf_samples, 1u);
    var sum = zero_path_radiance();
    for (var i = 0u; i < num_paths; i += 1u) {
        // Sparse captures may need the radiance ray to agree with a separately
        // rasterized center-sampled G-buffer. References retain stochastic
        // subpixel coverage for antialiasing.
        let jitter = select(
            vec2<f32>(0.5),
            vec2<f32>(random_gen(&rng), random_gen(&rng)),
            parameters.jitter_primary_rays != 0u,
        );
        let ray_dir = get_ray_direction_at(camera, vec2<f32>(global_id.xy) + jitter);
        let sample = trace_path(ray_dir, &rng);
        sum.total += sample.total;
        sum.diffuse += sample.diffuse;
        sum.specular += sample.specular;
        sum.emissive += sample.emissive;
    }

    let count = f32(num_paths);
    textureStore(accumulator, global_id.xy, total + vec4<f32>(sum.total, count));
    textureStore(accumulator_diffuse, global_id.xy, total_diffuse + vec4<f32>(sum.diffuse, count));
    textureStore(accumulator_specular, global_id.xy, total_specular + vec4<f32>(sum.specular, count));
    textureStore(accumulator_emissive, global_id.xy, total_emissive + vec4<f32>(sum.emissive, count));
}
