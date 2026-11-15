// Differentiable path tracer.
//
// A sibling of "path-trace.wgsl" whose material factors, environment
// radiance, and optionally material textures come from storage buffers,
// so that an optimizer can change them, and which propagates a loss
// gradient back to them:
//   - `primal` renders the image with the current parameters,
//   - `adjoint` takes dLoss/dImage and accumulates dLoss/dParameters.
//
// The adjoint traces each path twice with the same random numbers. The first
// walk records what every vertex found and how much of it each vertex passed
// on, so that the radiance arriving at every vertex is known exactly. The
// second walk differentiates each vertex with that radiance in hand. This is
// Path Replay Backpropagation (Vicini et al. 2021), with the recording taking
// the place of its division by the throughput, which fails for black albedo.
//
// Only the integrand is differentiated. Sampling decisions, densities, MIS
// weights and roulette are held fixed, which keeps the gradient unbiased as
// long as they remain valid.
//
// Moving the geometry and the camera also moves what the camera sees, which
// `edges` and `scatter_hit_geometry_grad` differentiate: the first moves
// the discontinuities of the image along triangle edges, and the second the
// interpolated normal of every hit, as the hit slides along its ray, which
// `scatter_hit_motion` carries back along the path. The directions of the
// path are held fixed, like the rest of the sampling. Shadows and reflected
// silhouettes move with the edges that `sample_secondary_edges` draws.
enable wgpu_ray_query;
enable wgpu_binding_array;
#include "quaternion.inc.wgsl"
#include "random.inc.wgsl"
#include "camera.inc.wgsl"
#include "debug-param.inc.wgsl"
#include "brdf.inc.wgsl"
#include "brdf-grad.inc.wgsl"
#include "sampling.inc.wgsl"
#include "env-importance.inc.wgsl"
#include "env-light.inc.wgsl"
#include "hit.inc.wgsl"

// Longer paths are cut, so that the recording has a fixed size.
const MAX_DIFF_BOUNCES: u32 = 8u;
const PATH_LENGTH: u32 = MAX_DIFF_BOUNCES + 1u;
const ROULETTE_START: u32 = 4u;
const MAX_RADIANCE: f32 = 1.0e6;
// Floats per `MaterialParams`, also the stride of `material_grads`.
const MATERIAL_PARAM_COUNT: u32 = 12u;
// Floats per environment texel in `env_params` and `env_grads`.
const ENV_PARAM_COUNT: u32 = 4u;
// Floats per material texture texel in `texture_params` and `texture_grads`.
const TEXTURE_PARAM_COUNT: u32 = 4u;
const NO_TEXTURE_REGION: u32 = ~0u;

// Has to match the host!
struct DiffParams {
    frame_index: u32,
    num_environment_samples: u32,
    samples_per_pixel: u32,
    max_bounces: u32,
    t_start: f32,
    environment_importance_sampling: u32,
    // Frames already averaged into `image`, which `primal` adds to.
    accumulated_frames: u32,
    jitter_primary_rays: u32,
    // Differentiate translations of the instances and the camera.
    geometry: u32,
    instance_count: u32,
    triangle_count: u32,
    edge_samples: u32,
    // Edges sampled at every hit of the first `secondary_edge_depth` ones
    // of a path, for its incident light.
    secondary_edge_samples: u32,
    secondary_edge_depth: u32,
    // Draw the secondary edges by `edge_guide`, and record what they bring
    // in `edge_hits`.
    guided_edges: u32,
    // Pair every other path of the adjoint with the one before, see
    // `sample_bsdf_turned`.
    antithetic: u32,
    // Points drawn on the emitting triangles at every vertex of a path.
    emitter_samples: u32,
    // Sample the edges that the camera sees by projecting the hits of the
    // paths onto them, see `sample_projective_edge`, instead of `edges`.
    projective_edges: u32,
    // Lights in `light_params`.
    light_count: u32,
    _pad2: u32,
}

// Has to match the host!
// Triangles of one hit entry, for sampling their edges.
struct EdgeEntry {
    hit_entry: u32,
    instance: u32,
    first_triangle: u32,
    triangle_count: u32,
    // Where the vertices of the hit entry start in `vertex_slots`, if the
    // positions of vertices are parameters, or `~0u`.
    vertex_base: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Has to match the host!
// Where the texels of a texture of the binding array are in `texture_params`.
struct TextureRegion {
    // In texels, or `NO_TEXTURE_REGION` for a texture that is sampled as is.
    offset: u32,
    width: u32,
    height: u32,
    _pad: u32,
}

// Has to match the host!
struct TextureInit {
    texture_index: u32,
    offset: u32,
    width: u32,
    height: u32,
}

// Has to match the host!
// The glTF factors of a hit entry, multiplied with its textures.
struct MaterialParams {
    base_color: vec3<f32>,
    metalness: f32,
    emissive: vec3<f32>,
    roughness: f32,
    // The part of the diffuse albedo that goes through the surface instead,
    // see `evaluate_surface_gradient`.
    transmission: f32,
    // Index of refraction of a solid, through which the transmission is
    // refracted, or zero for a thin wall.
    ior: f32,
    _pad1: f32,
    _pad2: f32,
}

var<uniform> camera: CameraParams;
var<uniform> parameters: DiffParams;
var acc_struct: acceleration_structure;
var env_map: texture_2d<f32>;
var sampler_nearest: sampler;
// One per hit entry.
var<storage, read> material_params: array<MaterialParams>;
// Radiance of the environment texels, row by row.
var<storage, read> env_params: array<vec4<f32>>;
// Mean radiance of each pixel.
var<storage, read_write> image: array<vec4<f32>>;
// Derivative of the loss by `image`.
var<storage, read> image_adjoint: array<vec4<f32>>;
var<storage, read_write> material_grads: array<atomic<f32>>;
var<storage, read_write> env_grads: array<atomic<f32>>;
// One per texture of the binding array, or fewer.
var<storage, read> texture_regions: array<TextureRegion>;
var<storage, read> texture_params: array<vec4<f32>>;
var<storage, read_write> texture_grads: array<atomic<f32>>;
// The same buffers as the parameters, for the initialization.
var<storage, read_write> init_material_params: array<MaterialParams>;
var<storage, read_write> init_env_params: array<vec4<f32>>;
var<storage, read_write> init_texture_params: array<vec4<f32>>;
var<uniform> texture_init: TextureInit;
// Sorted by `first_triangle`.
var<storage, read> edge_entries: array<EdgeEntry>;
var<storage, read> instance_transforms: array<mat4x3<f32>>;
// By the translation of each instance, and of the camera, in XYZ of 4 floats.
var<storage, read_write> translation_grads: array<atomic<f32>>;
// By the rotation of each instance about its origin, as an axis times the
// angle, in XYZ of 4 floats.
var<storage, read_write> rotation_grads: array<atomic<f32>>;
var<storage, read> rotation_params: array<vec4<f32>>;
// Open addressing hash table of the edges, by their instance and the
// positions of their ends, with the number of triangles along each.
// The length is a power of two, and empty keys are zero.
var<storage, read_write> edge_keys: array<atomic<u32>>;
// The parameter of every vertex of the models, which copies of a vertex
// along a seam share, and its gradient: the offset of the position in the
// space of the model, in XYZ of 4 floats.
var<storage, read> vertex_slots: array<u32>;
var<storage, read_write> vertex_grads: array<atomic<f32>>;
// By the normal of every slot, in the space of its model, which a pass
// before every step carries to the positions around the slot.
var<storage, read_write> normal_grads: array<atomic<f32>>;
var<storage, read_write> edge_counts: array<atomic<u32>>;
// The normals of the first two triangles along every edge, in the space of
// their instance, by the slot of the edge.
var<storage, read_write> edge_normals: array<vec4<f32>>;
// The first two triangles along every edge, by the slot of the edge.
var<storage, read_write> edge_triangles: array<u32>;
// What the secondary edges drawn from every edge brought, since the last
// `guide_edges`, and the importance and the distribution that it makes.
var<storage, read_write> edge_hits: array<atomic<f32>>;
var<storage, read_write> edge_importance: array<f32>;
var<storage, read_write> edge_guide: array<f32>;
// The number of the triangles whose materials emit, and then their indices,
// which `collect_emitters` fills in for every pass.
var<storage, read_write> emitters: array<atomic<u32>>;

// Has to match the host!
// A point light, or a spot light, see `light_falloff`.
struct DiffLight {
    position: vec3<f32>,
    cos_outer: f32,
    intensity: vec3<f32>,
    cos_inner: f32,
    direction: vec3<f32>,
    _pad: f32,
}
const LIGHT_PARAM_COUNT: u32 = 12u;
var<storage, read> light_params: array<DiffLight>;
// In the layout of `DiffLight`.
var<storage, read_write> light_grads: array<atomic<f32>>;
var<storage, read_write> camera_grads: array<atomic<f32>>;

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

fn mis_weight(count: f32, pdf: f32, other_count: f32, other_pdf: f32) -> f32 {
    let total = count * pdf + other_count * other_pdf;
    return select(0.0, count * pdf / total, total > 0.0);
}

// The environment is constant over each texel, everywhere, including where
// the reference path tracer filters it for the background.
fn env_texel_at(dir: vec3<f32>) -> vec2<i32> {
    let dim = vec2<i32>(textureDimensions(env_map, 0));
    let uv = map_equirect_dir_to_uv(dir);
    return clamp(vec2<i32>(uv * vec2<f32>(dim)), vec2<i32>(0), dim - vec2<i32>(1));
}

fn env_texel_index(texel: vec2<i32>) -> u32 {
    return u32(texel.y) * textureDimensions(env_map, 0).x + u32(texel.x);
}

fn env_radiance(texel: vec2<i32>) -> vec3<f32> {
    return env_params[env_texel_index(texel)].xyz;
}

fn scatter_env_grad(texel: vec2<i32>, grad: vec3<f32>) {
    let base = env_texel_index(texel) * ENV_PARAM_COUNT;
    for (var c = 0u; c < 3u; c += 1u) {
        if (grad[c] != 0.0) {
            atomicAdd(&env_grads[base + c], grad[c]);
        }
    }
}

struct EnvSample {
    dir: vec3<f32>,
    // Solid angle density of drawing this sample.
    pdf: f32,
    texel: vec2<i32>,
}

// `sample_light`, keeping the texel instead of reading the texture.
fn sample_environment(importance: bool, rng: ptr<function, RandomState>) -> EnvSample {
    if (importance) {
        let dim = textureDimensions(env_map, 0);
        let es = generate_environment_sample(rng, dim);
        let u = (f32(es.pixel.x) + random_gen(rng)) / f32(dim.x);
        let bounds = compute_latitude_area_bounds(es.pixel.y, dim.y);
        let v = acos(mix(bounds.x, bounds.y, random_gen(rng))) / PI;
        return EnvSample(map_equirect_uv_to_dir(vec2<f32>(u, v)), es.pdf, es.pixel);
    }
    let a = random_gen(rng);
    let h = 1.0 - 2.0 * random_gen(rng);
    let tangential = sqrt(max(0.0, 1.0 - h * h)) * sample_circle_uniform(a);
    let dir = vec3<f32>(tangential.x, h, tangential.y);
    return EnvSample(dir, 1.0 / (4.0 * PI), env_texel_at(dir));
}

fn texture_region(texture_index: u32) -> TextureRegion {
    if (texture_index >= arrayLength(&texture_regions)) {
        return TextureRegion(NO_TEXTURE_REGION, 0u, 0u, 0u);
    }
    return texture_regions[texture_index];
}

// Texels of a bilinear tap, with their weights, for `sampler_linear`:
// linear filtering with repeating coordinates, at the top mip level.
struct BilinearTaps {
    indices: vec4<u32>,
    weights: vec4<f32>,
}

fn bilinear_taps(region: TextureRegion, tex_coords: vec2<f32>) -> BilinearTaps {
    let size = vec2<i32>(i32(region.width), i32(region.height));
    let pos = tex_coords * vec2<f32>(size) - 0.5;
    let corner = floor(pos);
    let f = pos - corner;
    let x0 = vec2<i32>(corner) % size;
    // Repeat, for negative coordinates as well.
    let lo = select(x0, x0 + size, x0 < vec2<i32>(0));
    let hi = select(lo + 1, vec2<i32>(0), lo + 1 >= size);
    let row_lo = region.offset + u32(lo.y) * region.width;
    let row_hi = region.offset + u32(hi.y) * region.width;
    var taps: BilinearTaps;
    taps.indices = vec4<u32>(row_lo + u32(lo.x), row_lo + u32(hi.x), row_hi + u32(lo.x), row_hi + u32(hi.x));
    taps.weights = vec4<f32>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    return taps;
}

// Sample a material texture, from its parameters when it has them.
fn sample_material_texture(texture_index: u32, tex_coords: vec2<f32>, lod: f32) -> vec4<f32> {
    let region = texture_region(texture_index);
    if (region.offset == NO_TEXTURE_REGION) {
        return textureSampleLevel(textures[texture_index], sampler_linear, tex_coords, lod);
    }
    let taps = bilinear_taps(region, tex_coords);
    var value = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i += 1u) {
        value += taps.weights[i] * texture_params[taps.indices[i]];
    }
    return value;
}

// Add the derivative of the loss by a sample of a material texture
// to the gradients of the texels it was filtered from.
fn scatter_texture_grad(texture_index: u32, tex_coords: vec2<f32>, grad: vec4<f32>) {
    let region = texture_region(texture_index);
    if (region.offset == NO_TEXTURE_REGION || all(grad == vec4<f32>(0.0))) {
        return;
    }
    let taps = bilinear_taps(region, tex_coords);
    for (var i = 0u; i < 4u; i += 1u) {
        let base = taps.indices[i] * TEXTURE_PARAM_COUNT;
        for (var c = 0u; c < 4u; c += 1u) {
            let value = taps.weights[i] * grad[c];
            if (value != 0.0) {
                atomicAdd(&texture_grads[base + c], value);
            }
        }
    }
}

struct DiffVertex {
    // With the normal from the parameters of the normal map, if it has them.
    surface: HitSurface,
    // The normal map, in tangent space.
    normal_local: vec3<f32>,
    material: Material,
    transmission: f32,
    // The relative index of refraction, see `evaluate_surface_gradient`,
    // which depends on the side that the vertex is seen from.
    eta: f32,
    emissive: vec3<f32>,
    // Texture values that multiply the factors.
    base_color_texel: vec3<f32>,
    metalness_texel: f32,
    roughness_texel: f32,
    emissive_texel: vec3<f32>,
}

fn resolve_vertex(intersection: RayIntersection) -> DiffVertex {
    let lod = 0.0; //TODO: ray differentials
    var v: DiffVertex;
    v.surface = resolve_hit_surface(intersection, lod);
    let entry = v.surface.entry;
    let tc = v.surface.tex_coords;
    let params = material_params[v.surface.entry_index];
    v.base_color_texel = sample_material_texture(entry.base_color_texture, tc, lod).xyz;
    let mr = sample_material_texture(entry.metallic_roughness_texture, tc, lod);
    v.roughness_texel = mr.y;
    v.metalness_texel = mr.z;
    v.emissive_texel = sample_material_texture(entry.emissive_texture, tc, lod).xyz;
    // `sample_hit_normal_map`
    let raw_unorm = sample_material_texture(entry.normal_texture, tc, lod).xy;
    let n_xy = entry.normal_scale * (2.0 * raw_unorm - 1.0);
    v.normal_local = vec3<f32>(n_xy, sqrt(max(0.0, 1.0 - dot(n_xy, n_xy))));
    v.surface.normal = normalize(v.surface.tangent_space * v.normal_local);
    v.material = material_from_metallic_roughness(
        params.base_color * v.base_color_texel,
        params.metalness * v.metalness_texel,
        params.roughness * v.roughness_texel,
    );
    v.emissive = params.emissive * v.emissive_texel;
    v.transmission = params.transmission;
    // As seen from outside, which `walk_path` turns around from inside.
    v.eta = select(0.0, max(params.ior, 1.0001), params.ior > 0.0);
    return v;
}

// Derivative of the loss by the material of one vertex.
struct MaterialGrad {
    diffuse_albedo: vec3<f32>,
    specular_f0: vec3<f32>,
    roughness: f32,
    emissive: vec3<f32>,
    transmission: f32,
    // By the relative index of refraction of the vertex.
    eta: f32,
    // By the shading normal, in world space, before its normalization.
    normal: vec3<f32>,
    // By the normal of the triangle, through `horizon_weight`.
    flat_normal: vec3<f32>,
}

// Add the BSDF's share, given the derivative of the loss by its value.
fn add_bsdf_grad(grad: ptr<function, MaterialGrad>, bsdf: BsdfGradient, mat: Material, d_value: vec3<f32>) {
    (*grad).diffuse_albedo += d_value * bsdf.d_diffuse_albedo;
    (*grad).specular_f0 += d_value * bsdf.d_specular_f0;
    (*grad).roughness += dot(d_value, bsdf.d_roughness);
    (*grad).transmission += dot(d_value, bsdf.d_transmission);
    (*grad).eta += dot(d_value, bsdf.d_eta);
    (*grad).normal += bsdf_normal_grad(bsdf, d_value);
}

fn add_material_grad(index: u32, component: u32, value: f32) {
    if (value != 0.0) {
        atomicAdd(&material_grads[index * MATERIAL_PARAM_COUNT + component], value);
    }
}

// Carry a vertex's gradient through `resolve_vertex` to the factors
// and the textures.
fn scatter_material_grad(v: DiffVertex, grad: MaterialGrad) {
    let params = material_params[v.surface.entry_index];
    let base_color = params.base_color * v.base_color_texel;
    let metalness = params.metalness * v.metalness_texel;
    // `material_from_metallic_roughness`
    let d_base_color = grad.diffuse_albedo * (1.0 - metalness) + grad.specular_f0 * metalness;
    let d_metalness = dot(grad.specular_f0, base_color - vec3<f32>(DIELECTRIC_F0))
        - dot(grad.diffuse_albedo, base_color);

    let index = v.surface.entry_index;
    let d_base_factor = d_base_color * v.base_color_texel;
    let d_emissive_factor = grad.emissive * v.emissive_texel;
    add_material_grad(index, 0u, d_base_factor.x);
    add_material_grad(index, 1u, d_base_factor.y);
    add_material_grad(index, 2u, d_base_factor.z);
    add_material_grad(index, 3u, d_metalness * v.metalness_texel);
    add_material_grad(index, 4u, d_emissive_factor.x);
    add_material_grad(index, 5u, d_emissive_factor.y);
    add_material_grad(index, 6u, d_emissive_factor.z);
    add_material_grad(index, 7u, grad.roughness * v.roughness_texel);
    add_material_grad(index, 8u, grad.transmission);
    // `eta` is the index, or its inverse from inside.
    if (v.eta > 0.0) {
        add_material_grad(index, 9u, grad.eta * select(-v.eta * v.eta, 1.0, v.eta >= 1.0));
    }

    let entry = v.surface.entry;
    let tc = v.surface.tex_coords;
    scatter_texture_grad(entry.base_color_texture, tc, vec4<f32>(d_base_color * params.base_color, 0.0));
    scatter_texture_grad(entry.metallic_roughness_texture, tc, vec4<f32>(
        0.0, grad.roughness * params.roughness, d_metalness * params.metalness, 0.0));
    scatter_texture_grad(entry.emissive_texture, tc, vec4<f32>(grad.emissive * params.emissive, 0.0));

    if (texture_region(entry.normal_texture).offset != NO_TEXTURE_REGION) {
        // Back through `normalize(tangent_space * normal_local)`.
        let unnormalized = v.surface.tangent_space * v.normal_local;
        let magnitude = length(unnormalized);
        let normal = unnormalized / magnitude;
        let d_unnormalized = (grad.normal - normal * dot(normal, grad.normal)) / magnitude;
        let d_local = transpose(v.surface.tangent_space) * d_unnormalized;
        // Z is the reconstruction of a unit normal from X and Y.
        var d_xy = d_local.xy;
        if (v.normal_local.z > 0.0) {
            d_xy -= d_local.z * v.normal_local.xy / v.normal_local.z;
        }
        let d_raw = 2.0 * entry.normal_scale * d_xy;
        scatter_texture_grad(entry.normal_texture, tc, vec4<f32>(d_raw, 0.0, 0.0));
    }
}

// `grad` is by turning the instance a little more, from where its rotation
// `r` turns it, which is `exp(d) * exp(r)` for a small `d`. The parameter is
// `r` itself, so `exp(r + dr) = exp(J * dr) * exp(r)`, where `J` is the
// left Jacobian of the exponential map, and the gradient is `J^T * grad`.
fn add_rotation_grad(instance: u32, grad: vec3<f32>) {
    let r = rotation_params[instance].xyz;
    let angle = length(r);
    var param_grad = grad;
    if (angle > 1.0e-4) {
        let a = (1.0 - cos(angle)) / (angle * angle);
        let b = (angle - sin(angle)) / (angle * angle * angle);
        // `J^T = I - a [r]x + b [r]x^2`, since `[r]x` is antisymmetric.
        let rg = cross(r, grad);
        param_grad = grad - a * rg + b * cross(r, rg);
    }
    for (var c = 0u; c < 3u; c += 1u) {
        if (param_grad[c] != 0.0) {
            atomicAdd(&rotation_grads[instance * 4u + c], param_grad[c]);
        }
    }
}

// Add the derivative of the loss by moving `point` of `instance` to the
// translation of the instance, and to its rotation: turning by a small
// `angle` about its origin moves the point by `cross(angle, point - origin)`.
fn vertex_slots_of(vertex_base: u32, indices: vec3<u32>) -> vec3<u32> {
    if (vertex_base == ~0u) {
        return vec3<u32>(~0u);
    }
    return vec3<u32>(
        vertex_slots[vertex_base + indices.x],
        vertex_slots[vertex_base + indices.y],
        vertex_slots[vertex_base + indices.z],
    );
}

// Add the derivative of the loss by moving a vertex in world space, `grad`,
// to the parameter of its position, through `linear`.
fn add_vertex_grad(slot: u32, linear: mat3x3<f32>, grad: vec3<f32>) {
    if (slot == ~0u) {
        return;
    }
    let local = transpose(linear) * grad;
    for (var c = 0u; c < 3u; c += 1u) {
        if (local[c] != 0.0) {
            atomicAdd(&vertex_grads[slot * 4u + c], local[c]);
        }
    }
}

fn add_normal_grad(slot: u32, grad: vec3<f32>) {
    if (slot == ~0u) {
        return;
    }
    for (var c = 0u; c < 3u; c += 1u) {
        if (grad[c] != 0.0) {
            atomicAdd(&normal_grads[slot * 4u + c], grad[c]);
        }
    }
}

// Derivative of the loss by the corners of a triangle, from the one by
// the cross product of its edges from the first corner.
fn cross_grad(e1: vec3<f32>, e2: vec3<f32>, d_cross: vec3<f32>) -> mat3x3<f32> {
    let d1 = cross(e2, d_cross);
    let d2 = cross(d_cross, e1);
    return mat3x3<f32>(-d1 - d2, d1, d2);
}

fn add_instance_grad(instance: u32, point: vec3<f32>, grad: vec3<f32>) {
    for (var c = 0u; c < 3u; c += 1u) {
        if (grad[c] != 0.0) {
            atomicAdd(&translation_grads[instance * 4u + c], grad[c]);
        }
    }
    let origin = instance_transforms[instance][3];
    add_rotation_grad(instance, cross(point - origin, grad));
}

fn add_camera_grad(grad: vec3<f32>) {
    for (var c = 0u; c < 3u; c += 1u) {
        if (grad[c] != 0.0) {
            atomicAdd(&camera_grads[c], grad[c]);
        }
    }
}

fn make_path_vertex(
    intersection: RayIntersection,
    direction: vec3<f32>,
    v: DiffVertex,
    normal: vec3<f32>,
    flat_normal: vec3<f32>,
    d_found: vec3<f32>,
) -> PathVertex {
    var pv: PathVertex;
    let entry = v.surface.entry;
    let indices = fetch_triangle_indices(entry, intersection.primitive_index);
    let vptr = &vertex_buffers[entry.vertex_buf].data;
    let to_world = intersection.object_to_world;
    let p0 = to_world * vec4<f32>(entry.geometry_to_object * vec4<f32>((*vptr)[indices.x].position, 1.0), 1.0);
    let p1 = to_world * vec4<f32>(entry.geometry_to_object * vec4<f32>((*vptr)[indices.y].position, 1.0), 1.0);
    let p2 = to_world * vec4<f32>(entry.geometry_to_object * vec4<f32>((*vptr)[indices.z].position, 1.0), 1.0);
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let det = dot(e1, cross(e2, -direction));
    pv.valid = abs(det) > 1.0e-12;
    pv.instance = intersection.instance_index;
    pv.position = v.surface.position;
    pv.normal = normal;
    pv.flat_normal = flat_normal;
    pv.view_dir = -direction;
    pv.material = v.material;
    pv.transmission = v.transmission;
    pv.eta = v.eta;
    pv.depth_row = cross(e1, e2) / det;
    pv.ray_dir = direction;
    pv.d_found = d_found;
    let entry_index = intersection.instance_custom_data + intersection.geometry_index;
    pv.slots = vertex_slots_of(edge_entries[entry_index].vertex_base, indices);
    pv.triangle = hit_triangle(intersection);
    pv.barycentrics = make_barycentrics(intersection.barycentrics);
    let linear = affine_linear(to_world) * affine_linear(entry.geometry_to_object);
    pv.local_depth_row = transpose(linear) * pv.depth_row;
    return pv;
}

// The interpolated normal of a primary hit moves as the camera or the hit
// instance move, since another point of the triangle comes under the pixel,
// and so does that of any hit, as the origin of its ray moves. Carry the
// derivative of the loss by the shading normal, `d_normal`, to the
// translations through it, and to the rotation, which also turns the normal,
// and return the part of it that is by moving the origin of the ray.
//
// The tangent frame is taken to turn with the normal, which is exact without
// a normal map, and the texture coordinates are taken to stay.
fn scatter_hit_geometry_grad(
    intersection: RayIntersection,
    direction: vec3<f32>,
    v: DiffVertex,
    d_normal: vec3<f32>,
    d_flat_normal: vec3<f32>,
) -> vec3<f32> {
    if (all(d_normal == vec3<f32>(0.0)) && all(d_flat_normal == vec3<f32>(0.0))) {
        return vec3<f32>(0.0);
    }
    let entry = v.surface.entry;
    let indices = fetch_triangle_indices(entry, intersection.primitive_index);
    let vptr = &vertex_buffers[entry.vertex_buf].data;
    let vertices = array<Vertex, 3>((*vptr)[indices.x], (*vptr)[indices.y], (*vptr)[indices.z]);
    let to_world = intersection.object_to_world;
    let p0 = to_world * vec4<f32>(entry.geometry_to_object * vec4<f32>(vertices[0].position, 1.0), 1.0);
    let p1 = to_world * vec4<f32>(entry.geometry_to_object * vec4<f32>(vertices[1].position, 1.0), 1.0);
    let p2 = to_world * vec4<f32>(entry.geometry_to_object * vec4<f32>(vertices[2].position, 1.0), 1.0);

    // The hit satisfies `p0 + b1 * e1 + b2 * e2 = origin + t * direction`,
    // so moving the origin by `d` moves `(b1, b2, t)` by the inverse of
    // `[e1, e2, -direction]` applied to `d`, whose rows are these.
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let row1 = cross(e2, -direction);
    let row2 = cross(-direction, e1);
    let det = dot(e1, row1);
    if (abs(det) < 1.0e-12) {
        return vec3<f32>(0.0);
    }

    // Back through `normalize(tangent_space * normal_local)`, to the normal
    // of the tangent space, which is `transform_normal` of the interpolated
    // vertex normals: `normalize(cofactor * sum)`.
    let shading = v.surface.tangent_space * v.normal_local;
    let shading_length = length(shading);
    let shading_normal = shading / shading_length;
    let d_shading = (d_normal - shading_normal * dot(shading_normal, d_normal)) / shading_length;
    let d_geometric = v.normal_local.z * d_shading;
    let linear = affine_linear(to_world) * affine_linear(entry.geometry_to_object);
    let cofactor = mat3x3<f32>(
        cross(linear[1], linear[2]), cross(linear[2], linear[0]), cross(linear[0], linear[1]),
    ) * sign(dot(linear[0], cross(linear[1], linear[2])));
    let n0 = decode_normal(vertices[0].normal);
    let n1 = decode_normal(vertices[1].normal);
    let n2 = decode_normal(vertices[2].normal);
    let b = make_barycentrics(intersection.barycentrics);
    let transformed = cofactor * (b.x * n0 + b.y * n1 + b.z * n2);
    let transformed_length = length(transformed);
    let geometric = transformed / transformed_length;
    let d_transformed = (d_geometric - geometric * dot(geometric, d_geometric)) / transformed_length;
    let d_sum = transpose(cofactor) * d_transformed;
    let d_b1 = dot(d_sum, n1 - n0);
    let d_b2 = dot(d_sum, n2 - n0);

    let d_camera = (d_b1 * row1 + d_b2 * row2) / det;
    // Moving the triangle is moving the camera the other way. Turning it
    // moves the hit, like translating it there would, and turns the normal.
    add_instance_grad(intersection.instance_index, v.surface.position, -d_camera);
    // Moving its vertices moves the hit by their mix. The normals are taken
    // to stay, though they follow the positions when those are applied.
    let entry_index = intersection.instance_custom_data + intersection.geometry_index;
    let slots = vertex_slots_of(edge_entries[entry_index].vertex_base, indices);
    add_vertex_grad(slots.x, linear, -b.x * d_camera);
    add_vertex_grad(slots.y, linear, -b.y * d_camera);
    add_vertex_grad(slots.z, linear, -b.z * d_camera);
    if (slots.x != ~0u) {
        // The vertex normals follow the positions around the vertices.
        add_normal_grad(slots.x, b.x * d_sum);
        add_normal_grad(slots.y, b.y * d_sum);
        add_normal_grad(slots.z, b.z * d_sum);
        // The normal of the triangle, for its horizon, follows its corners.
        let face = cross(e1, e2);
        let face_length = length(face);
        if (face_length > 0.0) {
            let unit = face / face_length;
            let d_face = hit_winding(entry) * (d_flat_normal - unit * dot(unit, d_flat_normal)) / face_length;
            let d_corners = cross_grad(e1, e2, d_face);
            add_vertex_grad(slots.x, linear, d_corners[0]);
            add_vertex_grad(slots.y, linear, d_corners[1]);
            add_vertex_grad(slots.z, linear, d_corners[2]);
        }
    }
    add_rotation_grad(intersection.instance_index, cross(shading_normal, d_normal));
    return d_camera;
}

// Interpolated and mapped normals can face away from the view where the
// surface doesn't, and the BSDF is black there, which makes the radiance
// jump across the curve where they turn. Nothing samples that curve, so
// instead the normal is bent towards the view, continuously, before it
// gets there. This is the only place where this tracer differs from
// "path-trace.wgsl", which shows those parts black.
const MIN_SHADING_COSINE: f32 = 0.02;

fn bend_normal(normal: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let cosine = dot(normal, view_dir);
    if (cosine >= MIN_SHADING_COSINE) {
        return normal;
    }
    return normalize(normal + (MIN_SHADING_COSINE - cosine) * view_dir);
}

// Derivative of the loss by the normal given to `bend_normal`, from the one
// by what it returned.
fn bend_normal_grad(normal: vec3<f32>, view_dir: vec3<f32>, d_bent: vec3<f32>) -> vec3<f32> {
    let cosine = dot(normal, view_dir);
    if (cosine >= MIN_SHADING_COSINE) {
        return d_bent;
    }
    let unnormalized = normal + (MIN_SHADING_COSINE - cosine) * view_dir;
    let magnitude = length(unnormalized);
    let bent = unnormalized / magnitude;
    let d_unnormalized = (d_bent - bent * dot(bent, d_bent)) / magnitude;
    return d_unnormalized - view_dir * dot(view_dir, d_unnormalized);
}

// Light is only taken from above the triangle, so that it doesn't leak
// through the surface where the shading normal leans past it. Cutting it off
// at the horizon would be a discontinuity that moves as the triangle turns,
// with smooth shading, so it fades in over a band above the horizon instead.
const HORIZON_BAND: f32 = 0.05;

fn horizon_weight(cosine: f32) -> f32 {
    let t = clamp(cosine / HORIZON_BAND, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// The cosine of `dir` with the normal of the triangle, on the side of the
// shading normal that `dir` is on, for `horizon_weight`: light through a
// transmissive surface has to come from below the triangle too.
fn side_cosine(dir: vec3<f32>, normal: vec3<f32>, flat_normal: vec3<f32>) -> f32 {
    return select(-1.0, 1.0, dot(dir, normal) >= 0.0) * dot(dir, flat_normal);
}

// Derivative of `side_cosine` by the normal of the triangle.
fn side_cosine_grad(dir: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return select(-1.0, 1.0, dot(dir, normal) >= 0.0) * dir;
}

fn horizon_weight_grad(cosine: f32) -> f32 {
    let t = cosine / HORIZON_BAND;
    return select(0.0, 6.0 * t * (1.0 - t) / HORIZON_BAND, t > 0.0 && t < 1.0);
}

// What the replay keeps of every hit of a path, for `sample_secondary_edges`
// and for moving the hit with the geometry, see `scatter_hit_motion`.
struct PathVertex {
    valid: bool,
    instance: u32,
    position: vec3<f32>,
    // The normals as the BSDF and the horizon take them.
    normal: vec3<f32>,
    flat_normal: vec3<f32>,
    view_dir: vec3<f32>,
    material: Material,
    transmission: f32,
    eta: f32,
    // Moving the origin of the camera ray by `d` moves the hit along the
    // ray by `dot(depth_row, d)` times its direction.
    depth_row: vec3<f32>,
    ray_dir: vec3<f32>,
    d_found: vec3<f32>,
    // The index of the triangle, as `triangle_edges` takes it.
    triangle: u32,
    // The vertex slots of the corners, and the weights of the hit.
    slots: vec3<u32>,
    barycentrics: vec3<f32>,
    // `depth_row` in the space of the vertex positions.
    local_depth_row: vec3<f32>,
}

// What the first walk of a path learns about each of its vertices.
struct PathRecord {
    // Radiance found at the vertex: its emission, the lights it connects to,
    // or the environment for a ray that escaped. Excludes the throughput
    // of the path up to the vertex.
    found: array<vec3<f32>, PATH_LENGTH>,
    // Multiplier of the throughput from the vertex to the next one.
    weight: array<vec3<f32>, PATH_LENGTH>,
    length: u32,
    // The hits of the replay, which are the first `hit_count` of the
    // vertices, along with the derivative of the loss by their positions.
    hits: array<PathVertex, PATH_LENGTH>,
    hit_count: u32,
    d_hits: array<vec3<f32>, PATH_LENGTH>,
}

// Radiance arriving at each vertex of a recorded path through the direction
// it was extended in, again without the throughput up to the vertex.
fn compute_incident_radiance(record: ptr<function, PathRecord>) -> array<vec3<f32>, PATH_LENGTH> {
    var incident: array<vec3<f32>, PATH_LENGTH>;
    var radiance = vec3<f32>(0.0);
    for (var i = (*record).length; i > 0u; i -= 1u) {
        incident[i - 1u] = radiance;
        radiance = (*record).found[i - 1u] + (*record).weight[i - 1u] * radiance;
    }
    return incident;
}

// List the triangles whose materials emit, by their factors, which are
// what the light is drawn from.
@compute @workgroup_size(64)
fn collect_emitters(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let triangle = global_id.y * num_workgroups.x * 64u + global_id.x;
    if (triangle >= parameters.triangle_count) {
        return;
    }
    let edges = triangle_edges(triangle);
    if (edges.degenerate || !is_emitter_entry(edges.entry)) {
        return;
    }
    let index = atomicAdd(&emitters[0], 1u);
    atomicStore(&emitters[1u + index], triangle);
}

fn is_emitter_entry(entry: u32) -> bool {
    return any(material_params[entry].emissive > vec3<f32>(0.0));
}

fn emitter_count() -> u32 {
    if (parameters.emitter_samples == 0u) {
        return 0u;
    }
    return atomicLoad(&emitters[0]);
}

// The global index of the triangle of a hit, as `triangle_edges` takes it.
fn hit_triangle(intersection: RayIntersection) -> u32 {
    let entry_index = intersection.instance_custom_data + intersection.geometry_index;
    return edge_entries[entry_index].first_triangle + intersection.primitive_index;
}

fn world_triangle(edges: TriangleEdges) -> mat3x3<f32> {
    let to_world = instance_transforms[edges.instance];
    return mat3x3<f32>(
        to_world * vec4<f32>(edges.local[0], 1.0),
        to_world * vec4<f32>(edges.local[1], 1.0),
        to_world * vec4<f32>(edges.local[2], 1.0),
    );
}

// Solid angle density of drawing `direction` from a point, which sees the
// emitting `triangle` at `distance` that way, by drawing an emitter
// uniformly and then a point on it uniformly.
fn emitter_pdf(triangle: u32, direction: vec3<f32>, distance: f32, count: u32) -> f32 {
    let corners = world_triangle(triangle_edges(triangle));
    let normal = cross(corners[1] - corners[0], corners[2] - corners[0]);
    let double_area = length(normal);
    let cosine = abs(dot(normal, direction)) / max(double_area, 1.0e-20);
    if (cosine < 1.0e-6 || double_area <= 0.0) {
        return 0.0;
    }
    return distance * distance / (cosine * 0.5 * double_area * f32(count));
}

// How much of its intensity a light sends towards `-light_dir`: all of it
// for a point light, and for a spot light, a smooth step from nothing at
// `cos_outer` off its direction to all at `cos_inner`, which keeps the edge
// of its cone from being a discontinuity. `y` is the derivative by the
// cosine off the direction.
fn light_falloff(light: DiffLight, light_dir: vec3<f32>) -> vec2<f32> {
    if (light.cos_outer < -1.0) {
        return vec2<f32>(1.0, 0.0);
    }
    let cosine = -dot(light_dir, normalize(light.direction));
    let width = max(light.cos_inner - light.cos_outer, 1.0e-6);
    let t = clamp((cosine - light.cos_outer) / width, 0.0, 1.0);
    let d_t = select(0.0, 1.0 / width, t > 0.0 && t < 1.0);
    return vec2<f32>(t * t * (3.0 - 2.0 * t), 6.0 * t * (1.0 - t) * d_t);
}

fn add_light_grad(index: u32, component: u32, grad: vec3<f32>) {
    for (var k = 0u; k < 3u; k += 1u) {
        if (grad[k] != 0.0) {
            atomicAdd(&light_grads[index * LIGHT_PARAM_COUNT + component + k], grad[k]);
        }
    }
}

// Whether anything is between `position` and the point at `distance` along
// `direction`.
fn is_occluded_within(position: vec3<f32>, direction: vec3<f32>, distance: f32) -> bool {
    var rq: ray_query;
    let flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_CULL_NO_OPAQUE;
    rayQueryInitialize(&rq, acc_struct,
        RayDesc(flags, 0xFFu, parameters.t_start, distance * (1.0 - 1.0e-4), position, direction)
    );
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq).kind != RAY_QUERY_INTERSECTION_NONE;
}

// Whether the walk turns the first direction that it samples from a BSDF,
// see `sample_bsdf_turned`.
var<private> turn_first_bsdf_sample: bool;

// `sample_bsdf`, turned by half a turn about the normal if `turned`, which
// is just as likely, as the lobes are symmetric about it. The derivative of
// the BSDF by its normal tilts the lobe, which is mostly odd about the
// normal, so a path and its turned twin have gradients that mostly cancel
// where each alone has a large one, which is antithetic sampling, after
// Zhang et al. "Antithetic Sampling for Monte Carlo Differentiable
// Rendering" (2021).
fn sample_bsdf_turned(mat: Material, transmission: f32, eta: f32, normal: vec3<f32>, view_dir: vec3<f32>, turned: bool, rng: ptr<function, RandomState>) -> BsdfSample {
    return sample_surface(mat, transmission, eta, normal, view_dir, turned, rng);
}

// Walk one path and return its radiance.
//
// Without `adjoint`, the walk fills the `record`. With it, the walk is a
// replay of the recorded one, given the same random state, and it scatters
// the gradient of the loss, of which `adjoint` is the derivative by the
// radiance, using the radiance arriving at every vertex from `incident`.
// `bounce_budget` is the number of bounces, after the first hit, that the
// path may take: `parameters.max_bounces` for a camera path, which lights
// the first hit, while a negative budget only takes the emission there.
fn walk_path(
    origin: vec3<f32>,
    start_t_min: f32,
    bounce_budget: i32,
    start_dir: vec3<f32>,
    rng: ptr<function, RandomState>,
    record: ptr<function, PathRecord>,
    replay: bool,
    adjoint: vec3<f32>,
    incident: ptr<function, array<vec3<f32>, PATH_LENGTH>>,
) -> vec3<f32> {
    let importance = parameters.environment_importance_sampling != 0u;
    let num_light = f32(parameters.num_environment_samples);
    let max_bounces = u32(clamp(bounce_budget, 0, i32(MAX_DIFF_BOUNCES)));
    let light_hits = bounce_budget >= 0;
    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var position = origin;
    var direction = start_dir;
    // Negative for the first ray, like for the camera in "path-trace.wgsl",
    // since it takes in all the light.
    var bsdf_pdf = -1.0;
    var t_min = start_t_min;
    (*record).length = 0u;
    if (replay) {
        (*record).hit_count = 0u;
    }

    for (var bounce = 0u; bounce <= max_bounces; bounce += 1u) {
        let intersection = trace_ray(position, direction, t_min);
        if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
            let texel = env_texel_at(direction);
            var weight = 1.0;
            if (bsdf_pdf >= 0.0) {
                let light_pdf = compute_light_pdf(map_equirect_dir_to_uv(direction), importance);
                weight = mis_weight(1.0, bsdf_pdf, num_light, light_pdf);
            }
            let found = env_radiance(texel) * weight;
            radiance += throughput * found;
            if (replay) {
                scatter_env_grad(texel, adjoint * throughput * weight);
            } else {
                (*record).found[bounce] = found;
                (*record).weight[bounce] = vec3<f32>(0.0);
                (*record).length = bounce + 1u;
            }
            break;
        }

        var v = resolve_vertex(intersection);
        let view_dir = -direction;
        // A transmissive surface is a thin wall, seen the same from both
        // sides, so its normals face the ray.
        let facing = select(1.0, -1.0, v.transmission > 0.0 && dot(v.surface.flat_normal, view_dir) < 0.0);
        let unbent_normal = facing * v.surface.normal;
        v.surface.normal = bend_normal(unbent_normal, view_dir);
        let normal = v.surface.normal;
        let flat_normal = facing * v.surface.flat_normal;
        if (facing < 0.0 && v.eta > 0.0) {
            v.eta = 1.0 / v.eta;
        }
        // Derivative of the loss by everything found at this vertex.
        let d_found = adjoint * throughput;
        // The emission is also found by drawing points on the emitters from
        // the previous vertex, if this one was found by its BSDF.
        var emission_weight = 1.0;
        if (bsdf_pdf >= 0.0 && any(v.emissive > vec3<f32>(0.0)) && is_emitter_entry(v.surface.entry_index)) {
            let count = emitter_count();
            if (count != 0u) {
                let light_pdf = emitter_pdf(hit_triangle(intersection), direction, intersection.t, count);
                emission_weight = mis_weight(1.0, bsdf_pdf, f32(parameters.emitter_samples), light_pdf);
            }
        }
        var grad = MaterialGrad(vec3<f32>(0.0), vec3<f32>(0.0), 0.0, d_found * emission_weight, 0.0, 0.0, vec3<f32>(0.0), vec3<f32>(0.0));
        var found = v.emissive * emission_weight;
        position = v.surface.position;
        t_min = parameters.t_start;

        // Each sample is a separate path here, so the path always goes on.
        let will_extend = bounce < max_bounces && light_hits;
        let bsdf_count = select(0.0, 1.0, will_extend);

        let light_count = select(0u, parameters.num_environment_samples, light_hits);
        for (var i = 0u; i < light_count; i += 1u) {
            var ls = sample_environment(importance, rng);
            if (bounce == 0u && turn_first_bsdf_sample && !importance) {
                // Uniform directions are just as likely turned about the
                // normal, which makes the light samples antithetic too.
                ls.dir = 2.0 * dot(ls.dir, normal) * normal - ls.dir;
                ls.texel = env_texel_at(ls.dir);
            }
            if (ls.pdf <= 0.0) {
                continue;
            }
            let bsdf = evaluate_surface_gradient(v.material, v.transmission, v.eta, normal, view_dir, ls.dir);
            let horizon_cosine = side_cosine(ls.dir, normal, flat_normal);
            let horizon = horizon_weight(horizon_cosine);
            if (horizon <= 0.0 || all(bsdf.value <= vec3<f32>(0.0))
                || is_occluded(position, ls.dir)) {
                continue;
            }
            let other_pdf = surface_pdf(v.material, v.transmission, v.eta, normal, view_dir, ls.dir);
            let unfaded = mis_weight(num_light, ls.pdf, bsdf_count, other_pdf) / (num_light * ls.pdf);
            let weight = unfaded * horizon;
            let light = env_radiance(ls.texel);
            found += bsdf.value * light * weight;
            if (replay) {
                add_bsdf_grad(&grad, bsdf, v.material, d_found * light * weight);
                scatter_env_grad(ls.texel, d_found * bsdf.value * weight);
                grad.flat_normal += dot(d_found, bsdf.value * light) * unfaded
                    * horizon_weight_grad(horizon_cosine) * side_cosine_grad(ls.dir, normal);
            }
        }
        // Points on the emitting triangles. Only their directions are drawn
        // from them: the light comes from what the ray that way hits, if it's
        // the triangle, which keeps the gradient in the same terms as for
        // the light found by the BSDF, with the directions held fixed.
        let emitters_drawn = select(0u, emitter_count(), light_hits);
        for (var i = 0u; i < select(0u, parameters.emitter_samples, emitters_drawn != 0u); i += 1u) {
            let pick = min(u32(random_gen(rng) * f32(emitters_drawn)), emitters_drawn - 1u);
            let r1 = sqrt(random_gen(rng));
            let r2 = random_gen(rng);
            let triangle = atomicLoad(&emitters[1u + pick]);
            let corners = world_triangle(triangle_edges(triangle));
            let point = (1.0 - r1) * corners[0] + r1 * (1.0 - r2) * corners[1] + r1 * r2 * corners[2];
            let offset = point - position;
            let distance = length(offset);
            if (distance < 1.0e-4) {
                continue;
            }
            let dir = offset / distance;
            let light_pdf = emitter_pdf(triangle, dir, distance, emitters_drawn);
            let horizon_cosine = side_cosine(dir, normal, flat_normal);
            let horizon = horizon_weight(horizon_cosine);
            if (light_pdf <= 0.0 || horizon <= 0.0) {
                continue;
            }
            let bsdf = evaluate_surface_gradient(v.material, v.transmission, v.eta, normal, view_dir, dir);
            if (all(bsdf.value <= vec3<f32>(0.0))) {
                continue;
            }
            let hit = trace_ray(position, dir, t_min);
            if (hit.kind == RAY_QUERY_INTERSECTION_NONE || hit_triangle(hit) != triangle) {
                continue;
            }
            let lamp = resolve_vertex(hit);
            let other_pdf = surface_pdf(v.material, v.transmission, v.eta, normal, view_dir, dir);
            let samples = f32(parameters.emitter_samples);
            let unfaded = mis_weight(samples, light_pdf, bsdf_count, other_pdf) / (samples * light_pdf);
            let weight = unfaded * horizon;
            found += bsdf.value * lamp.emissive * weight;
            if (replay) {
                add_bsdf_grad(&grad, bsdf, v.material, d_found * lamp.emissive * weight);
                grad.flat_normal += dot(d_found, bsdf.value * lamp.emissive) * unfaded
                    * horizon_weight_grad(horizon_cosine) * side_cosine_grad(dir, normal);
                var lamp_grad = MaterialGrad(vec3<f32>(0.0), vec3<f32>(0.0), 0.0, d_found * bsdf.value * weight, 0.0, 0.0, vec3<f32>(0.0), vec3<f32>(0.0));
                scatter_material_grad(lamp, lamp_grad);
            }
        }
        // The point and spot lights, whose directions aren't drawn, so the
        // BSDF moves with them, and with the hit, which the derivative by
        // its position takes, see `scatter_hit_motion`.
        var d_position = vec3<f32>(0.0);
        for (var i = 0u; i < select(0u, parameters.light_count, light_hits); i += 1u) {
            let light = light_params[i];
            let offset = light.position - position;
            let distance = length(offset);
            if (distance < 1.0e-4) {
                continue;
            }
            let dir = offset / distance;
            let falloff = light_falloff(light, dir);
            let horizon_cosine = side_cosine(dir, normal, flat_normal);
            let horizon = horizon_weight(horizon_cosine);
            if (horizon <= 0.0 || falloff.x <= 0.0) {
                continue;
            }
            let bsdf = evaluate_surface_gradient(v.material, v.transmission, v.eta, normal, view_dir, dir);
            if (all(bsdf.value <= vec3<f32>(0.0)) || is_occluded_within(position, dir, distance)) {
                continue;
            }
            let arriving = light.intensity / (distance * distance);
            found += bsdf.value * arriving * falloff.x * horizon;
            if (replay) {
                let d_value = d_found * arriving * falloff.x * horizon;
                add_bsdf_grad(&grad, bsdf, v.material, d_value);
                let d_horizon = dot(d_found, bsdf.value * arriving) * falloff.x;
                grad.flat_normal += d_horizon * horizon_weight_grad(horizon_cosine) * side_cosine_grad(dir, normal);
                add_light_grad(i, 4u, d_found * bsdf.value * falloff.x * horizon / (distance * distance));
                // By the direction to the light, through the BSDF, the
                // horizon, and the cone, and by the distance to it.
                let d_cone = dot(d_found, bsdf.value * arriving) * horizon * falloff.y;
                let spot_dir = normalize(light.direction);
                // `side_cosine` by the direction is the normal of the
                // triangle, on the side of the shading normal.
                let side = select(-1.0, 1.0, dot(dir, normal) >= 0.0);
                let d_dir = bsdf_light_grad(bsdf, d_value)
                    + d_horizon * horizon_weight_grad(horizon_cosine) * side * flat_normal
                    - d_cone * spot_dir;
                let d_distance = -2.0 * dot(d_found, bsdf.value * arriving * falloff.x * horizon) / distance;
                let d_light = (d_dir - dir * dot(dir, d_dir)) / distance + d_distance * dir;
                add_light_grad(i, 0u, d_light);
                if (light.cos_outer >= -1.0) {
                    add_light_grad(i, 8u, -d_cone * (dir - spot_dir * dot(spot_dir, dir)) / length(light.direction));
                }
                d_position -= d_light;
            }
        }
        radiance += throughput * found;
        if (!replay) {
            (*record).found[bounce] = found;
            (*record).weight[bounce] = vec3<f32>(0.0);
            (*record).length = bounce + 1u;
        }

        var weight = vec3<f32>(0.0);
        if (will_extend) {
            let bs = sample_bsdf_turned(v.material, v.transmission, v.eta, normal, view_dir, bounce == 0u && turn_first_bsdf_sample, rng);
            let horizon_cosine = side_cosine(bs.dir, normal, flat_normal);
            let horizon = horizon_weight(horizon_cosine);
            if (bs.pdf > 0.0 && horizon > 0.0) {
                let bsdf = evaluate_surface_gradient(v.material, v.transmission, v.eta, normal, view_dir, bs.dir);
                var scale = horizon / bs.pdf;
                var survives = true;
                if (bounce >= ROULETTE_START) {
                    let probability = clamp(compute_luminocity(throughput * bsdf.value * scale), 0.05, 1.0);
                    survives = random_gen(rng) < probability;
                    scale /= probability;
                }
                if (survives) {
                    weight = bsdf.value * scale;
                    if (replay) {
                        let d_value = d_found * (*incident)[bounce] * scale;
                        add_bsdf_grad(&grad, bsdf, v.material, d_value);
                        grad.flat_normal += dot(d_value, bsdf.value) / horizon
                            * horizon_weight_grad(horizon_cosine) * side_cosine_grad(bs.dir, normal);
                    }
                    bsdf_pdf = bs.pdf;
                    direction = bs.dir;
                }
            }
        }
        if (replay) {
            // By the normals as they are, not as they face the ray.
            grad.normal = facing * bend_normal_grad(unbent_normal, view_dir, grad.normal);
            grad.flat_normal *= facing;
            v.surface.normal = facing * unbent_normal;
            scatter_material_grad(v, grad);
            if (parameters.geometry != 0u) {
                // Only a path from the camera, whose origin is the previous hit.
                let ray_dir = select(start_dir, -view_dir, bounce > 0u);
                let d_origin = scatter_hit_geometry_grad(intersection, ray_dir, v, grad.normal, grad.flat_normal);
                if (bounce == 0u) {
                    add_camera_grad(d_origin);
                } else {
                    (*record).d_hits[bounce - 1u] += d_origin;
                }
                (*record).hits[bounce] = make_path_vertex(intersection, ray_dir, v, normal, flat_normal, d_found);
                (*record).d_hits[bounce] = d_position;
                (*record).hit_count = bounce + 1u;
                // The triangle turns with the instance.
                add_rotation_grad(intersection.instance_index, cross(v.surface.flat_normal, grad.flat_normal));
            }
        } else {
            (*record).weight[bounce] = weight;
        }
        throughput *= weight;
        if (all(throughput <= vec3<f32>(0.0))) {
            break;
        }
    }
    return radiance;
}

// Offset from the center of a pixel, with the density of the tent filter.
fn sample_tent(random: f32) -> f32 {
    if (random < 0.5) {
        return sqrt(2.0 * random) - 1.0;
    }
    return 1.0 - sqrt(2.0 - 2.0 * random);
}

// Pixels are tent filtered, over the area of the four pixels around their
// centers. With a box filter, an edge that moves across a pixel changes it
// with a kink every time that it passes a corner, which leaves the loss with
// kinks all over, and a tent filter smooths these out.
fn primary_point(pixel: vec2<u32>, rng: ptr<function, RandomState>) -> vec2<f32> {
    var jitter = vec2<f32>(0.5);
    if (parameters.jitter_primary_rays != 0u) {
        jitter += vec2<f32>(sample_tent(random_gen(rng)), sample_tent(random_gen(rng)));
    }
    return vec2<f32>(pixel) + jitter;
}

fn primary_direction(pixel: vec2<u32>, rng: ptr<function, RandomState>) -> vec3<f32> {
    return get_ray_direction_at(camera, primary_point(pixel, rng));
}

// Derivative of the loss by the radiance at a point of the film, through
// the tent filter of the pixels around it.
fn film_adjoint(p: vec2<f32>) -> vec3<f32> {
    let corner = floor(p - vec2<f32>(0.5));
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i += 1u) {
        let pixel = corner + vec2<f32>(f32(i & 1u), f32(i >> 1u));
        if (any(pixel < vec2<f32>(0.0)) || any(pixel >= vec2<f32>(camera.target_size))) {
            continue;
        }
        let weight = max(vec2<f32>(0.0), vec2<f32>(1.0) - abs(p - pixel - vec2<f32>(0.5)));
        let index = u32(pixel.y) * camera.target_size.x + u32(pixel.x);
        sum += weight.x * weight.y * image_adjoint[index].xyz;
    }
    return sum;
}

// How much of a path's radiance is kept, like in "path-trace.wgsl":
// nothing if it isn't finite, and only up to `MAX_RADIANCE`.
fn radiance_scale(radiance: vec3<f32>) -> vec3<f32> {
    if (!all(radiance == radiance)) {
        return vec3<f32>(0.0);
    }
    return min(vec3<f32>(1.0), vec3<f32>(MAX_RADIANCE) / max(radiance, vec3<f32>(1.0e-20)));
}

@compute @workgroup_size(8, 4)
fn primal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id.xy >= camera.target_size)) {
        return;
    }
    let pixel_index = global_id.y * camera.target_size.x + global_id.x;
    var rng = random_init(pixel_index, parameters.frame_index);
    var record: PathRecord;
    var incident: array<vec3<f32>, PATH_LENGTH>;
    var sum = vec3<f32>(0.0);
    let num_paths = max(parameters.samples_per_pixel, 1u);
    for (var i = 0u; i < num_paths; i += 1u) {
        let dir = primary_direction(global_id.xy, &rng);
        let radiance = walk_path(camera.position, 0.0, i32(parameters.max_bounces), dir, &rng, &record, false, vec3<f32>(0.0), &incident);
        sum += select(vec3<f32>(0.0), radiance * radiance_scale(radiance), all(radiance == radiance));
    }
    let mean = sum / f32(num_paths);
    let count = f32(parameters.accumulated_frames);
    var total = mean;
    if (parameters.accumulated_frames != 0u) {
        total = (image[pixel_index].xyz * count + mean) / (count + 1.0);
    }
    image[pixel_index] = vec4<f32>(total, 1.0);
}

@compute @workgroup_size(8, 4)
fn adjoint(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id.xy >= camera.target_size)) {
        return;
    }
    let pixel_index = global_id.y * camera.target_size.x + global_id.x;
    let num_paths = max(parameters.samples_per_pixel, 1u);
    // The pixel is the mean of its paths.
    let d_pixel = image_adjoint[pixel_index].xyz / f32(num_paths);
    let projective = parameters.projective_edges != 0u && parameters.geometry != 0u;
    if (all(d_pixel == vec3<f32>(0.0)) && !projective) {
        return;
    }
    var rng = random_init(pixel_index, parameters.frame_index);
    var record: PathRecord;
    var incident: array<vec3<f32>, PATH_LENGTH>;
    var pair_rng = rng;
    for (var i = 0u; i < num_paths; i += 1u) {
        // The twin of a path starts from the same random numbers, which
        // also make the same point of the pixel.
        let turned = parameters.antithetic != 0u && (i & 1u) == 1u;
        turn_first_bsdf_sample = false;
        if (turned) {
            rng = pair_rng;
        } else {
            pair_rng = rng;
        }
        let q = primary_point(global_id.xy, &rng);
        let dir = get_ray_direction_at(camera, q);
        // Other random numbers than the paths of the pixel.
        var edge_rng = random_init(pixel_index ^ 0x5bd1e995u, parameters.frame_index + i);
        if (projective) {
            // The pixels cover the screen with as many samples as each has.
            let hit = trace_ray(camera.position, dir, 0.0);
            if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
                var pv: PathVertex;
                pv.triangle = hit_triangle(hit);
                sample_projective_edge(pv, q, f32(num_paths), &edge_rng);
            }
        }
        if (all(d_pixel == vec3<f32>(0.0))) {
            continue;
        }
        var replay_rng = rng;
        turn_first_bsdf_sample = turned;
        let radiance = walk_path(camera.position, 0.0, i32(parameters.max_bounces), dir, &rng, &record, false, vec3<f32>(0.0), &incident);
        if (!all(radiance == radiance)) {
            turn_first_bsdf_sample = false;
            continue;
        }
        incident = compute_incident_radiance(&record);
        let adjoint = d_pixel * radiance_scale(radiance);
        walk_path(camera.position, 0.0, i32(parameters.max_bounces), dir, &replay_rng, &record, true, adjoint, &incident);
        turn_first_bsdf_sample = false;
        if (parameters.geometry == 0u) {
            continue;
        }
        if (projective) {
            let depth = min(record.hit_count, parameters.secondary_edge_depth);
            for (var k = 0u; k < depth; k += 1u) {
                if (record.hits[k].valid) {
                    record.d_hits[k] += sample_projective_directions(record.hits[k], k, &edge_rng);
                }
            }
        } else if (parameters.secondary_edge_samples != 0u) {
            let depth = min(record.hit_count, parameters.secondary_edge_depth);
            for (var k = 0u; k < depth; k += 1u) {
                if (record.hits[k].valid) {
                    record.d_hits[k] += sample_secondary_edges(record.hits[k], k, &edge_rng);
                }
            }
        }
        if (parameters.secondary_edge_samples != 0u && parameters.light_count != 0u) {
            let depth = min(record.hit_count, parameters.secondary_edge_depth);
            for (var k = 0u; k < depth; k += 1u) {
                if (record.hits[k].valid) {
                    record.d_hits[k] += sample_light_shadows(record.hits[k], k, &edge_rng);
                }
            }
        }
        scatter_hit_motion(&record);
    }
}

@compute @workgroup_size(64)
fn init_materials(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&init_material_params) || index >= arrayLength(&hit_entries)) {
        return;
    }
    let entry = hit_entries[index];
    init_material_params[index] = MaterialParams(
        unpack4x8unorm(entry.base_color_factor).xyz,
        entry.metalness,
        entry.emissive_factor.xyz,
        entry.roughness,
        0.0, 0.0, 0.0, 0.0,
    );
}

@compute @workgroup_size(8, 8)
fn init_environment(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = textureDimensions(env_map, 0);
    if (any(global_id.xy >= dim)) {
        return;
    }
    init_env_params[global_id.y * dim.x + global_id.x] = textureLoad(env_map, global_id.xy, 0);
}

// Copy the top mip level of `texture_init.texture_index` into the parameters.
@compute @workgroup_size(8, 8)
fn init_texture(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= texture_init.width || global_id.y >= texture_init.height) {
        return;
    }
    let index = texture_init.offset + global_id.y * texture_init.width + global_id.x;
    init_texture_params[index] = textureLoad(textures[texture_init.texture_index], global_id.xy, 0);
}

// The camera position moves the pixel of a point the other way, so this is
// minus the derivative of `get_projected_pixel_float` by the camera position.
// Columns are by world axis.
fn projection_jacobian(cp: CameraParams, point: vec3<f32>) -> mat3x2<f32> {
    let to_local = qinv(cp.orientation);
    let local = qrot(to_local, point - cp.position);
    let depth = -local.z;
    let scale = 0.5 * vec2<f32>(cp.target_size) * VFLIP / tan(0.5 * cp.fov);
    var jacobian: mat3x2<f32>;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        var world = vec3<f32>(0.0);
        world[axis] = 1.0;
        let d_local = qrot(to_local, world);
        jacobian[axis] = scale * (d_local.xy / depth + local.xy * d_local.z / (depth * depth));
    }
    return jacobian;
}

fn is_in_front(cp: CameraParams, point: vec3<f32>) -> bool {
    return qrot(qinv(cp.orientation), point - cp.position).z < 0.0;
}

// Lexicographic order of positions, for the triangles on both sides of an
// edge to agree on which of them samples it.
fn precedes(a: vec3<f32>, b: vec3<f32>) -> bool {
    if (a.x != b.x) {
        return a.x < b.x;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    return a.z < b.z;
}

// Sample the boundary term along the segment from `a` to `b` of `instance`.
//
// Wherever the segment is a discontinuity of the image, moving it by `dx`
// pixels across turns `dx` of the pixel on the side it moves to into what
// the other side shows. The radiance is estimated on both sides with the
// same random numbers, so that it cancels out where the segment is not
// a discontinuity, such as between two triangles of a smooth surface.
// The directions from `eye` to just either side of the edge from `a` to `b`
// of a triangle, whose third corner is `c`, at `point` on it: `below` on
// the side opposite to `across`, which is perpendicular to the direction to
// the point, and `above` on the side of `across`.
//
// The points are on the plane of the triangle, by a small fraction of its
// size, which keeps them on either side of this edge, and of no other edge
// of the mesh around it, however much the triangle leans away from the eye.
// An offset of the direction by a fixed angle instead crosses the outline
// of a mesh from many of the edges near it, where the mesh leans away the
// most, and those add to the jump across the outline itself.
struct EdgeSides {
    below: vec3<f32>,
    above: vec3<f32>,
    valid: bool,
}

fn edge_sides(eye: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, point: vec3<f32>, across: vec3<f32>) -> EdgeSides {
    var sides: EdgeSides;
    sides.valid = false;
    let along = b - a;
    let length_squared = dot(along, along);
    var inward = c - point;
    inward -= along * (dot(inward, along) / max(length_squared, 1.0e-20));
    let height = length(inward);
    if (height < 1.0e-8 || length_squared < 1.0e-16) {
        return sides;
    }
    let step = 1.0e-3 * min(sqrt(length_squared), height) / height * inward;
    let inside = normalize(point + step - eye);
    let outside = normalize(point - step - eye);
    let side = dot(inside - outside, across);
    if (side == 0.0) {
        return sides;
    }
    sides.valid = true;
    sides.below = select(inside, outside, side > 0.0);
    sides.above = select(outside, inside, side > 0.0);
    return sides;
}

fn sample_edge(
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    instance: u32,
    slots: vec2<u32>,
    linear: mat3x3<f32>,
    rng: ptr<function, RandomState>,
) {
    if (!is_in_front(camera, a) || !is_in_front(camera, b)) {
        return;
    }
    let pa = get_projected_pixel_float(camera, a);
    let pb = get_projected_pixel_float(camera, b);
    let segment_length = length(pb - pa);
    if (segment_length < 1.0e-6) {
        return;
    }
    let samples = max(parameters.edge_samples, 1u);
    for (var i = 0u; i < samples; i += 1u) {
        let t = (f32(i) + random_gen(rng)) / f32(samples);
        sample_edge_at(a, b, c, instance, slots, linear, pa, pb, mix(pa, pb, t), segment_length / f32(samples), rng);
    }
}

// The boundary term of the image at `p`, on the edge from `a` to `b`, which
// the camera sees from `pa` to `pb`, of a triangle with the third corner
// `c`, drawn with the density `1 / inverse_density` along the edge on the
// screen.
fn sample_edge_at(
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    instance: u32,
    slots: vec2<u32>,
    linear: mat3x3<f32>,
    pa: vec2<f32>,
    pb: vec2<f32>,
    p: vec2<f32>,
    inverse_density: f32,
    rng: ptr<function, RandomState>,
) {
    let segment = pb - pa;
    let segment_length = length(segment);
    let across = vec2<f32>(-segment.y, segment.x) / segment_length;
    let size = vec2<f32>(camera.target_size);
    if (any(p < vec2<f32>(0.0)) || any(p >= size)) {
        return;
    }
    let d_pixel = film_adjoint(p);
    if (all(d_pixel == vec3<f32>(0.0))) {
        return;
    }
    // The point of the segment that is seen through `p`.
    let dir = get_ray_direction_at(camera, p);
    let along = b - a;
    let to_a = a - camera.position;
    let aa = dot(along, along);
    let ad = dot(along, dir);
    let denom = aa - ad * ad;
    let s = clamp(select(0.0, (ad * dot(dir, to_a) - dot(along, to_a)) / denom, denom > 0.0), 0.0, 1.0);
    let point = a + s * along;
    // `across` on the screen, as a direction in the world.
    let ahead = get_ray_direction_at(camera, p + 1.0e-3 * across) - get_ray_direction_at(camera, p - 1.0e-3 * across);
    let sides = edge_sides(camera.position, a, b, c, point, ahead);
    if (!sides.valid) {
        return;
    }
    var record: PathRecord;
    var incident: array<vec3<f32>, PATH_LENGTH>;
    var side_rng = *rng;
    let below = walk_path(camera.position, 0.0, i32(parameters.max_bounces), sides.below, &side_rng, &record, false, vec3<f32>(0.0), &incident);
    side_rng = *rng;
    let above = walk_path(camera.position, 0.0, i32(parameters.max_bounces), sides.above, &side_rng, &record, false, vec3<f32>(0.0), &incident);
    *rng = side_rng;
    let jump = select(vec3<f32>(0.0), below * radiance_scale(below), all(below == below))
        - select(vec3<f32>(0.0), above * radiance_scale(above), all(above == above));
    let weight = dot(d_pixel, jump) * inverse_density;
    if (weight == 0.0) {
        return;
    }

    let jacobian = projection_jacobian(camera, point);
    let velocity = vec3<f32>(dot(across, jacobian[0]), dot(across, jacobian[1]), dot(across, jacobian[2]));
    add_instance_grad(instance, point, weight * velocity);
    add_camera_grad(-weight * velocity);
    add_vertex_grad(slots.x, linear, (1.0 - s) * weight * velocity);
    add_vertex_grad(slots.y, linear, s * weight * velocity);
}

// The shadows of the point and spot lights: where the edge of a silhouette,
// as the light sees it, casts the boundary of its shadow onto a receiver,
// along which the light that the receiver sends is discontinuous. The
// point `e` of the edge casts the point `y = light + k (e - light)` of the
// boundary, on the plane of the receiver with the normal `n`, which moves
// with the edge by `J = k (I - d n^T / dot(n, d))`, where `d = e - light`,
// with the light by `(1 - k) / k` of that, and with the plane by
// `d n^T / dot(n, d)`.
struct ShadowPoint {
    valid: bool,
    point: vec3<f32>,
    edge_point: vec3<f32>,
    k: f32,
    d: vec3<f32>,
    receiver_normal: vec3<f32>,
    denom: f32,
    // A step across the boundary, into the shadow.
    step: vec3<f32>,
    triangle: u32,
    barycentrics: vec3<f32>,
}

fn shadow_jacobian(sp: ShadowPoint, u: vec3<f32>) -> vec3<f32> {
    return sp.k * (u - sp.d * dot(sp.receiver_normal, u) / sp.denom);
}

fn shadow_jacobian_transposed(sp: ShadowPoint, g: vec3<f32>) -> vec3<f32> {
    return sp.k * (g - sp.receiver_normal * dot(sp.d, g) / sp.denom);
}

// The boundary that the point `t` along the edge from `corner` of the
// triangle with the world `corners` casts from the light at `light`, if the
// edge is a silhouette there: the light passes it on the outside of the
// triangle, which `neighbors`, the triangles along the edge, don't cover.
fn find_shadow_point(light: vec3<f32>, corners: mat3x3<f32>, corner: u32, t: f32, neighbors: vec2<u32>) -> ShadowPoint {
    var sp: ShadowPoint;
    sp.valid = false;
    let a = corners[corner];
    let b = corners[(corner + 1u) % 3u];
    let c = corners[(corner + 2u) % 3u];
    let e = mix(a, b, t);
    let along = b - a;
    let length_squared = dot(along, along);
    var inward = c - e;
    inward -= along * (dot(inward, along) / max(length_squared, 1.0e-20));
    let height = length(inward);
    if (height < 1.0e-8 || length_squared < 1.0e-16) {
        return sp;
    }
    let step = 1.0e-3 * min(sqrt(length_squared), height) / height * inward;
    let d = e - light;
    let distance = length(d);
    if (distance < 1.0e-4) {
        return sp;
    }
    let outside = normalize(e - step - light);
    let hit = trace_ray(light, outside, 0.0);
    // Hidden from the light, or not a silhouette.
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE || hit.t <= distance * (1.0 + 1.0e-4)) {
        return sp;
    }
    let triangle = hit_triangle(hit);
    if (triangle == neighbors.x || triangle == neighbors.y) {
        return sp;
    }
    let receiver = resolve_vertex(hit);
    let n = receiver.surface.flat_normal;
    let denom = dot(n, d);
    if (abs(denom) < 1.0e-6 * distance) {
        return sp;
    }
    sp.k = dot(n, light + hit.t * outside - light) / denom;
    sp.point = light + sp.k * d;
    sp.edge_point = e;
    sp.d = d;
    sp.receiver_normal = n;
    sp.denom = denom;
    sp.step = shadow_jacobian(sp, step);
    sp.triangle = triangle;
    sp.barycentrics = vec3<f32>(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics);
    sp.valid = true;
    return sp;
}

// Whether the point `sp` of a receiver is what the ray from `eye` towards it
// hits first.
fn sees_shadow_point(eye: vec3<f32>, t_min: f32, sp: ShadowPoint) -> bool {
    let offset = sp.point - eye;
    let distance = length(offset);
    let seen = trace_ray(eye, offset / distance, t_min);
    return seen.kind != RAY_QUERY_INTERSECTION_NONE && hit_triangle(seen) == sp.triangle
        && abs(seen.t - distance) <= 1.0e-3 * distance;
}

// Scatter the derivative `g` of the loss by moving the point of the boundary
// to the edge, the light, and the receiver that make it.
fn scatter_shadow_grad(sp: ShadowPoint, edges: TriangleEdges, corner: u32, t: f32, light_index: u32, g: vec3<f32>) {
    let g_edge = shadow_jacobian_transposed(sp, g);
    add_instance_grad(edges.instance, sp.edge_point, g_edge);
    add_vertex_grad(edges.slots[corner], edges.linear, (1.0 - t) * g_edge);
    add_vertex_grad(edges.slots[(corner + 1u) % 3u], edges.linear, t * g_edge);
    add_light_grad(light_index, 0u, (1.0 - sp.k) / sp.k * g_edge);
    let receiver = triangle_edges(sp.triangle);
    let g_plane = sp.receiver_normal * (dot(sp.d, g) / sp.denom);
    add_instance_grad(receiver.instance, sp.point, g_plane);
    for (var j = 0u; j < 3u; j += 1u) {
        add_vertex_grad(receiver.slots[j], receiver.linear, sp.barycentrics[j] * g_plane);
    }
}

// The triangles along the edge in `slot` of the table, or `~0u`.
fn edge_neighbors(slot: u32) -> vec2<u32> {
    if (slot == ~0u) {
        return vec2<u32>(~0u);
    }
    let count = atomicLoad(&edge_counts[slot]);
    return vec2<u32>(
        select(~0u, edge_triangles[2u * slot], count >= 1u),
        select(~0u, edge_triangles[2u * slot + 1u], count >= 2u),
    );
}

// The boundary term of the image at the shadow that the point `t` along the
// edge from `corner` of the triangle of `edges` casts from the light
// `light_index`, drawn with the density `1 / inverse_density` along `t`.
fn sample_light_shadow(
    light_index: u32,
    edges: TriangleEdges,
    corner: u32,
    t: f32,
    inverse_density: f32,
    neighbors: vec2<u32>,
    rng: ptr<function, RandomState>,
) {
    let corners = world_triangle(edges);
    let sp = find_shadow_point(light_params[light_index].position, corners, corner, t, neighbors);
    if (!sp.valid || !is_in_front(camera, sp.point)) {
        return;
    }
    let p = get_projected_pixel_float(camera, sp.point);
    if (any(p < vec2<f32>(0.0)) || any(p >= vec2<f32>(camera.target_size))) {
        return;
    }
    let d_pixel = film_adjoint(p);
    if (all(d_pixel == vec3<f32>(0.0)) || !sees_shadow_point(camera.position, 0.0, sp)) {
        return;
    }
    let jacobian = projection_jacobian(camera, sp.point);
    let along = jacobian * shadow_jacobian(sp, corners[(corner + 1u) % 3u] - corners[corner]);
    let speed = length(along);
    if (speed < 1.0e-8) {
        return;
    }
    let across = vec2<f32>(-along.y, along.x) / speed;
    let ahead = get_ray_direction_at(camera, p + 1.0e-3 * across) - get_ray_direction_at(camera, p - 1.0e-3 * across);
    let shadowed = normalize(sp.point + sp.step - camera.position);
    let lit = normalize(sp.point - sp.step - camera.position);
    let side = dot(shadowed - lit, ahead);
    if (side == 0.0) {
        return;
    }
    var record: PathRecord;
    var incident: array<vec3<f32>, PATH_LENGTH>;
    var side_rng = *rng;
    let below = walk_path(camera.position, 0.0, i32(parameters.max_bounces), select(shadowed, lit, side > 0.0), &side_rng, &record, false, vec3<f32>(0.0), &incident);
    side_rng = *rng;
    let above = walk_path(camera.position, 0.0, i32(parameters.max_bounces), select(lit, shadowed, side > 0.0), &side_rng, &record, false, vec3<f32>(0.0), &incident);
    *rng = side_rng;
    let jump = select(vec3<f32>(0.0), below * radiance_scale(below), all(below == below))
        - select(vec3<f32>(0.0), above * radiance_scale(above), all(above == above));
    let weight = dot(d_pixel, jump) * speed * inverse_density;
    if (weight == 0.0) {
        return;
    }
    let velocity = vec3<f32>(dot(across, jacobian[0]), dot(across, jacobian[1]), dot(across, jacobian[2]));
    scatter_shadow_grad(sp, edges, corner, t, light_index, weight * velocity);
    add_camera_grad(-weight * velocity);
}

// The width of the band along every edge of a triangle, as a part of the
// way to the opposite corner, whose points are projected onto the edge by
// `sample_projective_edge`. Narrower keeps the band clear of whatever
// hides the inside of the triangle, and wider uses more of the samples.
const PROJECTIVE_BAND: f32 = 0.4;

// Projective sampling of the edges that the camera sees, after Zhang et al.
// "Projective Sampling for Differentiable Rendering of Geometry" (2023):
// a camera ray that hits a triangle at the point `q` of the screen is
// carried from the opposite corner of an edge onto it. The screen is
// covered by the samples of the pixels evenly, with `density` of them per
// pixel, and carrying the triangle from a corner onto the opposite edge
// spreads the points `s` of the way to the edge over it by `s * height`,
// where `height` is that of the triangle on the screen, so the point on the
// edge has the density `density * s * height` for every part of the way.
// Only the points in a band along the edge are carried, which are rarely
// hidden where the edge isn't.
//
// Every edge is carried onto from one of the triangles along it: the one
// that faces the camera, if only one does, and otherwise the one that goes
// along it from the point that `precedes` the other.
fn sample_projective_edge(pv: PathVertex, q: vec2<f32>, density: f32, rng: ptr<function, RandomState>) {
    let edges = triangle_edges(pv.triangle);
    if (edges.degenerate) {
        return;
    }
    let corners = world_triangle(edges);
    var screen: array<vec2<f32>, 3>;
    for (var k = 0u; k < 3u; k += 1u) {
        if (!is_in_front(camera, corners[k])) {
            return;
        }
        screen[k] = get_projected_pixel_float(camera, corners[k]);
    }
    let linear = affine_linear(instance_transforms[edges.instance]);
    let local_normal = cross(edges.local[1] - edges.local[0], edges.local[2] - edges.local[0]);
    let faces = dot(linear * local_normal, camera.position - corners[0]) > 0.0;
    let e0 = screen[1] - screen[0];
    let e1 = screen[2] - screen[0];
    let area = e0.x * e1.y - e0.y * e1.x;
    if (abs(area) < 1.0e-8) {
        return;
    }
    let r = q - screen[0];
    let w1 = (r.x * e1.y - r.y * e1.x) / area;
    let w2 = (e0.x * r.y - e0.y * r.x) / area;
    let weights = vec3<f32>(1.0 - w1 - w2, w1, w2);
    // Every edge whose band `q` is in, which the bands of the others overlap.
    for (var opposite = 0u; opposite < 3u; opposite += 1u) {
        let s = 1.0 - weights[opposite];
        if (s < 1.0 - PROJECTIVE_BAND || s <= 0.0) {
            continue;
        }
        let ia = (opposite + 1u) % 3u;
        let ib = (opposite + 2u) % 3u;
        let la = edges.local[ia];
        let lb = edges.local[ib];
        let slot = edge_slot(edge_key(edges.instance, la, lb));
        if (slot == ~0u) {
            continue;
        }
        let count = atomicLoad(&edge_counts[slot]);
        if (count >= 2u) {
            // The other triangle along the edge, by its normal.
            var other = edge_normals[2u * slot].xyz;
            if (all(other == local_normal)) {
                other = edge_normals[2u * slot + 1u].xyz;
            }
            let other_faces = dot(linear * other, camera.position - corners[ia]) > 0.0;
            if (faces != other_faces) {
                if (!faces) {
                    continue;
                }
            } else if (!precedes(la, lb)) {
                continue;
            }
        }
        let c = screen[opposite];
        let y = c + (q - c) / s;
        let height = abs(area) / length(screen[ib] - screen[ia]);
        let slots = vec2<u32>(edges.slots[ia], edges.slots[ib]);
        sample_edge_at(corners[ia], corners[ib], corners[opposite], edges.instance, slots, edges.linear,
            screen[ia], screen[ib], y, 1.0 / (density * s * height * PROJECTIVE_BAND), rng);
    }
}

// The boundary term of the translations, sampled along the edges of every
// triangle in the scene. Requires jittered primary rays, i.e. filtered
// pixels, for the image to change continuously as edges move.
@compute @workgroup_size(64)
fn edges(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let triangle = global_id.y * num_workgroups.x * 64u + global_id.x;
    if (triangle >= parameters.triangle_count) {
        return;
    }
    let edges = triangle_edges(triangle);
    if (edges.degenerate) {
        return;
    }
    let local = edges.local;
    let to_world = instance_transforms[edges.instance];
    var rng = random_init(triangle, parameters.frame_index);
    for (var i = 0u; i < 3u; i += 1u) {
        let a = local[i];
        let b = local[(i + 1u) % 3u];
        // An edge in the middle of a mesh is shared with a triangle that
        // goes along it the other way, and only one of them takes it.
        let count = edge_triangle_count(edge_key(edges.instance, a, b));
        if (count == 0u || (count >= 2u && !precedes(a, b))) {
            continue;
        }
        if (parameters.projective_edges == 0u) {
            let slots = vec2<u32>(edges.slots[i], edges.slots[(i + 1u) % 3u]);
            let c = to_world * vec4<f32>(local[(i + 2u) % 3u], 1.0);
            sample_edge(to_world * vec4<f32>(a, 1.0), to_world * vec4<f32>(b, 1.0), c, edges.instance, slots, edges.linear, &rng);
        }
        if (parameters.light_count != 0u) {
            let neighbors = edge_neighbors(edge_slot(edge_key(edges.instance, a, b)));
            let samples = max(parameters.edge_samples, 1u);
            for (var light = 0u; light < parameters.light_count; light += 1u) {
                for (var k = 0u; k < samples; k += 1u) {
                    let t = (f32(k) + random_gen(&rng)) / f32(samples);
                    sample_light_shadow(light, edges, i, t, 1.0 / f32(samples), neighbors, &rng);
                }
            }
        }
    }
}

// Positions on a grid fine enough to tell vertices apart, and coarse enough
// for copies of a vertex along a seam to fall on the same point.
fn quantize_position(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(round(clamp(p, vec3<f32>(-30000.0), vec3<f32>(30000.0)) * 65536.0));
}

fn hash_combine(hash: u32, value: u32) -> u32 {
    return hash_jenkins(hash ^ (value + 0x9e3779b9u + (hash << 6u) + (hash >> 2u)));
}

// Key of the edge between `a` and `b` of `instance`, the same both ways.
fn edge_key(instance: u32, a: vec3<f32>, b: vec3<f32>) -> u32 {
    var qa = quantize_position(a);
    var qb = quantize_position(b);
    let swap = qb.x < qa.x || (qb.x == qa.x && (qb.y < qa.y || (qb.y == qa.y && qb.z < qa.z)));
    if (swap) {
        let q = qa;
        qa = qb;
        qb = q;
    }
    var hash = hash_jenkins(instance);
    hash = hash_combine(hash, bitcast<u32>(qa.x));
    hash = hash_combine(hash, bitcast<u32>(qa.y));
    hash = hash_combine(hash, bitcast<u32>(qa.z));
    hash = hash_combine(hash, bitcast<u32>(qb.x));
    hash = hash_combine(hash, bitcast<u32>(qb.y));
    hash = hash_combine(hash, bitcast<u32>(qb.z));
    return max(hash, 1u);
}

fn edge_slot_mask() -> u32 {
    return arrayLength(&edge_keys) - 1u;
}

// Number of triangles along an edge, which `count_edges` has found.
// The slot of the edge with `key`, or `~0u`.
fn edge_slot(key: u32) -> u32 {
    let mask = edge_slot_mask();
    for (var probe = 0u; probe <= mask; probe += 1u) {
        let slot = (key + probe) & mask;
        let found = atomicLoad(&edge_keys[slot]);
        if (found == key) {
            return slot;
        }
        if (found == 0u) {
            break;
        }
    }
    return ~0u;
}

fn edge_triangle_count(key: u32) -> u32 {
    let slot = edge_slot(key);
    if (slot == ~0u) {
        return 0u;
    }
    return atomicLoad(&edge_counts[slot]);
}

struct TriangleEdges {
    instance: u32,
    // The hit entry, which is the material.
    entry: u32,
    // The corners in the space of the instance.
    local: mat3x3<f32>,
    // Parameters of the corners, or `~0u`.
    slots: vec3<u32>,
    // Takes offsets of the vertices to world space.
    linear: mat3x3<f32>,
    // Two of the corners are at the same point, like at the poles of a UV
    // sphere. Such a triangle has no area, so it casts no silhouette, and
    // its edges are left to the triangles around it, which would otherwise
    // be counted twice, as both of its edges go along the same one.
    degenerate: bool,
}

fn triangle_edges(triangle: u32) -> TriangleEdges {
    var lo = 0u;
    var hi = arrayLength(&edge_entries);
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (edge_entries[mid].first_triangle <= triangle) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let edge_entry = edge_entries[lo];
    let entry = hit_entries[edge_entry.hit_entry];
    let indices = fetch_triangle_indices(entry, triangle - edge_entry.first_triangle);
    let vptr = &vertex_buffers[entry.vertex_buf].data;
    var edges: TriangleEdges;
    edges.instance = edge_entry.instance;
    edges.entry = edge_entry.hit_entry;
    edges.slots = vertex_slots_of(edge_entry.vertex_base, indices);
    edges.linear = affine_linear(instance_transforms[edge_entry.instance]) * affine_linear(entry.geometry_to_object);
    edges.local = entry.geometry_to_object * mat3x4<f32>(
        vec4<f32>((*vptr)[indices.x].position, 1.0),
        vec4<f32>((*vptr)[indices.y].position, 1.0),
        vec4<f32>((*vptr)[indices.z].position, 1.0),
    );
    let q0 = quantize_position(edges.local[0]);
    let q1 = quantize_position(edges.local[1]);
    let q2 = quantize_position(edges.local[2]);
    edges.degenerate = all(q0 == q1) || all(q1 == q2) || all(q2 == q0);
    return edges;
}

// Count the triangles along every edge, to tell the edges in the middle of a
// mesh from those on its boundary.
@compute @workgroup_size(64)
fn count_edges(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let triangle = global_id.y * num_workgroups.x * 64u + global_id.x;
    if (triangle >= parameters.triangle_count) {
        return;
    }
    let edges = triangle_edges(triangle);
    if (edges.degenerate) {
        return;
    }
    let face_normal = cross(edges.local[1] - edges.local[0], edges.local[2] - edges.local[0]);
    let mask = edge_slot_mask();
    for (var i = 0u; i < 3u; i += 1u) {
        let a = edges.local[i];
        let b = edges.local[(i + 1u) % 3u];
        let key = edge_key(edges.instance, a, b);
        var probe = 0u;
        loop {
            if (probe > mask) {
                break;
            }
            let slot = (key + probe) & mask;
            let result = atomicCompareExchangeWeak(&edge_keys[slot], 0u, key);
            if (result.exchanged || result.old_value == key) {
                let index = atomicAdd(&edge_counts[slot], 1u);
                if (index < 2u) {
                    edge_normals[2u * slot + index] = vec4<f32>(face_normal, 0.0);
                    edge_triangles[2u * slot + index] = triangle;
                }
                break;
            }
            // A weak exchange can fail spuriously, and then the slot is
            // still empty, so try it again.
            if (result.old_value != 0u) {
                probe += 1u;
            }
        }
    }
}

// The boundary term of the light arriving at a primary hit, from the edges
// that occlude part of it: moving such an edge, or the hit, moves a shadow
// or a reflected silhouette.
//
// The edges are drawn uniformly from all the triangle edges of the scene,
// and a point uniformly along each, and the light on both sides of where it
// is seen from the hit is traced with the same random numbers, so that it
// cancels out for the many edges that aren't silhouettes from there.
//
// Returns the derivative of the loss by moving the hit, which
// `scatter_hit_motion` takes on from there.
fn sample_secondary_edges(pv: PathVertex, depth: u32, rng: ptr<function, RandomState>) -> vec3<f32> {
    let triangle_count = parameters.triangle_count;
    let samples = parameters.secondary_edge_samples;
    var d_hit = vec3<f32>(0.0);
    if (triangle_count == 0u) {
        return d_hit;
    }
    for (var k = 0u; k < samples; k += 1u) {
        let drawn = draw_edge(random_gen(rng));
        let triangle = drawn.edge / 3u;
        let corner = drawn.edge % 3u;
        let t = random_gen(rng);
        let edges = triangle_edges(triangle);
        if (edges.degenerate) {
            continue;
        }
        let la = edges.local[corner];
        let lb = edges.local[(corner + 1u) % 3u];
        let count = edge_triangle_count(edge_key(edges.instance, la, lb));
        if (count == 0u || (count >= 2u && !precedes(la, lb))) {
            continue;
        }
        d_hit += secondary_edge_at(pv, depth, edges, corner, t, 1.0 / (f32(samples) * drawn.probability), drawn.edge, rng);
    }
    return d_hit;
}

// The boundary term of the light arriving at `pv` from the point `t` along
// the edge from `corner` of the triangle of `edges`, drawn with the density
// `1 / inverse_density` along `t`, recording what it brings for the guide
// of `guide_edge`, unless it's `~0u`. Returns the derivative of the loss by
// moving the hit.
fn secondary_edge_at(
    pv: PathVertex,
    depth: u32,
    edges: TriangleEdges,
    corner: u32,
    t: f32,
    inverse_density: f32,
    guide_edge: u32,
    rng: ptr<function, RandomState>,
) -> vec3<f32> {
    let corners = world_triangle(edges);
    let a = corners[corner];
    let b = corners[(corner + 1u) % 3u];
    let c = corners[(corner + 2u) % 3u];
    let point = mix(a, b, t);
    let offset = point - pv.position;
    let distance = length(offset);
    if (distance < 1.0e-4) {
        return vec3<f32>(0.0);
    }
    let dir = offset / distance;
    let horizon = horizon_weight(side_cosine(dir, pv.normal, pv.flat_normal));
    if (horizon <= 0.0) {
        return vec3<f32>(0.0);
    }
    let bsdf = evaluate_surface_gradient(pv.material, pv.transmission, pv.eta, pv.normal, pv.view_dir, dir).value;
    if (all(bsdf <= vec3<f32>(0.0))) {
        return vec3<f32>(0.0);
    }
    // How the direction of the point turns along the edge, and across it.
    let along = b - a;
    let turn = (along - dir * dot(dir, along)) / distance;
    let speed = length(turn);
    if (speed < 1.0e-8) {
        return vec3<f32>(0.0);
    }
    let across = normalize(cross(dir, turn));
    let sides = edge_sides(pv.position, a, b, c, point, across);
    if (!sides.valid) {
        return vec3<f32>(0.0);
    }
    var record: PathRecord;
    var incident: array<vec3<f32>, PATH_LENGTH>;
    var side_rng = *rng;
    // The light arriving at the hit, with the bounces left after it.
    let budget = i32(parameters.max_bounces) - i32(depth) - 1;
    let below = walk_path(pv.position, parameters.t_start, budget, sides.below, &side_rng, &record, false, vec3<f32>(0.0), &incident);
    side_rng = *rng;
    let above = walk_path(pv.position, parameters.t_start, budget, sides.above, &side_rng, &record, false, vec3<f32>(0.0), &incident);
    *rng = side_rng;
    let jump = select(vec3<f32>(0.0), below * radiance_scale(below), all(below == below))
        - select(vec3<f32>(0.0), above * radiance_scale(above), all(above == above));
    let weight = dot(pv.d_found, bsdf * horizon * jump) * speed * inverse_density;
    if (weight == 0.0) {
        return vec3<f32>(0.0);
    }
    if (guide_edge != ~0u && parameters.guided_edges != 0u) {
        atomicAdd(&edge_hits[guide_edge], abs(weight) / distance);
    }
    // Moving the point by `u` turns its direction across the edge by
    // `dot(across, u) / distance`, and moving the hit does the opposite.
    let grad = weight * across / distance;
    add_instance_grad(edges.instance, point, grad);
    add_vertex_grad(edges.slots[corner], edges.linear, (1.0 - t) * grad);
    add_vertex_grad(edges.slots[(corner + 1u) % 3u], edges.linear, t * grad);
    return -grad;
}

// The shadows of the lights that the hit `pv` sees, from
// `secondary_edge_samples` edges drawn like by `sample_secondary_edges`,
// each with a light drawn uniformly. Returns the derivative of the loss by
// moving the hit.
fn sample_light_shadows(pv: PathVertex, depth: u32, rng: ptr<function, RandomState>) -> vec3<f32> {
    let light_count = parameters.light_count;
    let samples = parameters.secondary_edge_samples;
    var d_hit = vec3<f32>(0.0);
    if (parameters.triangle_count == 0u || light_count == 0u) {
        return d_hit;
    }
    for (var k = 0u; k < samples; k += 1u) {
        let drawn = draw_edge(random_gen(rng));
        let triangle = drawn.edge / 3u;
        let corner = drawn.edge % 3u;
        let t = random_gen(rng);
        let light_index = min(u32(random_gen(rng) * f32(light_count)), light_count - 1u);
        let edges = triangle_edges(triangle);
        if (edges.degenerate) {
            continue;
        }
        let la = edges.local[corner];
        let lb = edges.local[(corner + 1u) % 3u];
        let slot = edge_slot(edge_key(edges.instance, la, lb));
        let neighbors = edge_neighbors(slot);
        let count = select(0u, atomicLoad(&edge_counts[slot]), slot != ~0u);
        if (count == 0u || (count >= 2u && !precedes(la, lb))) {
            continue;
        }
        let inverse_density = f32(light_count) / (f32(samples) * drawn.probability);
        d_hit += light_shadow_at(pv, depth, light_index, edges, corner, t, inverse_density, neighbors, drawn.edge, rng);
    }
    return d_hit;
}

// Like `secondary_edge_at`, for the shadow that the point `t` along the edge
// casts from the light `light_index`, see `find_shadow_point`.
fn light_shadow_at(
    pv: PathVertex,
    depth: u32,
    light_index: u32,
    edges: TriangleEdges,
    corner: u32,
    t: f32,
    inverse_density: f32,
    neighbors: vec2<u32>,
    guide_edge: u32,
    rng: ptr<function, RandomState>,
) -> vec3<f32> {
    let corners = world_triangle(edges);
    let sp = find_shadow_point(light_params[light_index].position, corners, corner, t, neighbors);
    if (!sp.valid) {
        return vec3<f32>(0.0);
    }
    let offset = sp.point - pv.position;
    let distance = length(offset);
    if (distance < 1.0e-4) {
        return vec3<f32>(0.0);
    }
    let dir = offset / distance;
    let horizon = horizon_weight(side_cosine(dir, pv.normal, pv.flat_normal));
    if (horizon <= 0.0) {
        return vec3<f32>(0.0);
    }
    let bsdf = evaluate_surface_gradient(pv.material, pv.transmission, pv.eta, pv.normal, pv.view_dir, dir).value;
    if (all(bsdf <= vec3<f32>(0.0)) || !sees_shadow_point(pv.position, parameters.t_start, sp)) {
        return vec3<f32>(0.0);
    }
    let along = shadow_jacobian(sp, corners[(corner + 1u) % 3u] - corners[corner]);
    let turn = (along - dir * dot(dir, along)) / distance;
    let speed = length(turn);
    if (speed < 1.0e-8) {
        return vec3<f32>(0.0);
    }
    let across = normalize(cross(dir, turn));
    let shadowed = normalize(sp.point + sp.step - pv.position);
    let lit = normalize(sp.point - sp.step - pv.position);
    let side = dot(shadowed - lit, across);
    if (side == 0.0) {
        return vec3<f32>(0.0);
    }
    var record: PathRecord;
    var incident: array<vec3<f32>, PATH_LENGTH>;
    var side_rng = *rng;
    let budget = i32(parameters.max_bounces) - i32(depth) - 1;
    let below = walk_path(pv.position, parameters.t_start, budget, select(shadowed, lit, side > 0.0), &side_rng, &record, false, vec3<f32>(0.0), &incident);
    side_rng = *rng;
    let above = walk_path(pv.position, parameters.t_start, budget, select(lit, shadowed, side > 0.0), &side_rng, &record, false, vec3<f32>(0.0), &incident);
    *rng = side_rng;
    let jump = select(vec3<f32>(0.0), below * radiance_scale(below), all(below == below))
        - select(vec3<f32>(0.0), above * radiance_scale(above), all(above == above));
    let weight = dot(pv.d_found, bsdf * horizon * jump) * speed * inverse_density;
    if (weight == 0.0) {
        return vec3<f32>(0.0);
    }
    if (parameters.guided_edges != 0u) {
        atomicAdd(&edge_hits[guide_edge], abs(weight) / distance);
    }
    let grad = weight * across / distance;
    scatter_shadow_grad(sp, edges, corner, t, light_index, grad);
    return -grad;
}

// `sample_projective_secondary` from the hits of `secondary_edge_samples`
// directions drawn by the BSDF at `pv`, of which the average is taken.
fn sample_projective_directions(pv: PathVertex, depth: u32, rng: ptr<function, RandomState>) -> vec3<f32> {
    var d_hit = vec3<f32>(0.0);
    let samples = max(parameters.secondary_edge_samples, 1u);
    for (var j = 0u; j < samples; j += 1u) {
        let bs = sample_bsdf_turned(pv.material, pv.transmission, pv.eta, pv.normal, pv.view_dir, false, rng);
        if (bs.pdf <= 0.0 || horizon_weight(side_cosine(bs.dir, pv.normal, pv.flat_normal)) <= 0.0) {
            continue;
        }
        let hit = trace_ray(pv.position, bs.dir, parameters.t_start);
        if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
            continue;
        }
        var next: PathVertex;
        next.triangle = hit_triangle(hit);
        next.position = pv.position + hit.t * bs.dir;
        next.barycentrics = make_barycentrics(hit.barycentrics);
        d_hit += sample_projective_secondary(pv, next, bs.pdf * f32(samples), depth, rng);
    }
    return d_hit;
}

// How far the hits are carried towards the edges by
// `sample_projective_secondary`, as the tangent of the angle.
const PROJECTIVE_REACH: f32 = 0.05;
// The most triangles that a hit is carried across.
const PROJECTIVE_STEPS: u32 = 16u;

// Coordinates of `point` on the plane at a unit distance from `eye` along
// the axis of `frame`, by the central projection from `eye`, which keeps
// lines as lines, or nothing behind.
fn gnomonic(eye: vec3<f32>, frame: mat3x3<f32>, point: vec3<f32>) -> vec3<f32> {
    let local = transpose(frame) * (point - eye);
    if (local.z <= 1.0e-6) {
        return vec3<f32>(0.0, 0.0, -1.0);
    }
    return vec3<f32>(local.xy / local.z, 1.0);
}

// Whether the triangle faces `eye`, with its normal in the instance.
fn triangle_faces(edges: TriangleEdges, corners: mat3x3<f32>, eye: vec3<f32>) -> bool {
    let linear = affine_linear(instance_transforms[edges.instance]);
    let local_normal = cross(edges.local[1] - edges.local[0], edges.local[2] - edges.local[0]);
    return dot(linear * local_normal, eye - corners[0]) > 0.0;
}

// Projective sampling of the edges that `pv` sees, after Zhang et al.
// "Projective Sampling for Differentiable Rendering of Geometry" (2023):
// the hit `next` of a direction drawn by the BSDF at `pv` with `pdf` is
// carried in a random way over the surface, from triangle to triangle, up
// to `PROJECTIVE_REACH`, and every edge that it crosses is sampled there,
// until it reaches the outline of what `pv` sees. So the edges are drawn
// where the BSDF sends the light, from any hit on the surface, without
// drawing them from the scene.
//
// This is on the plane at a unit distance along the normal, onto which the
// directions are projected from `pv`, where the hit has the density of the
// direction times the cube of its cosine with the normal. The points that
// are carried onto a point of an edge are those up to the reach before it,
// in as much as the edge goes across the way, which would make the edges
// along the way very heavy. So the way is taken as drawn along with the
// hit, and the points are weighted by how much the edge goes across them,
// over the ways that come to it: all of them for an edge between triangles
// that `pv` sees, and the half from the inside for the outline.
fn sample_projective_secondary(pv: PathVertex, next: PathVertex, pdf: f32, depth: u32, rng: ptr<function, RandomState>) -> vec3<f32> {
    var d_hit = vec3<f32>(0.0);
    let offset = next.position - pv.position;
    let distance = length(offset);
    if (pdf <= 0.0 || distance <= 0.0) {
        return d_hit;
    }
    let dir = offset / distance;
    // The plane on the side of the surface that the direction is on.
    let side = select(-1.0, 1.0, dot(dir, pv.normal) >= 0.0);
    let frame = make_tangent_frame(side * pv.normal);
    let cosine = dot(dir, frame[2]);
    if (cosine <= 1.0e-3) {
        return d_hit;
    }
    let density = pdf * cosine * cosine * cosine;
    let angle = 2.0 * PI * random_gen(rng);
    let way = vec2<f32>(cos(angle), sin(angle));
    var start = gnomonic(pv.position, frame, next.position).xy;
    var triangle = next.triangle;
    // The edge that the walk came in through, as the slot of the edge.
    var entry_slot = ~0u;
    var travelled = 0.0;
    for (var step = 0u; step < PROJECTIVE_STEPS; step += 1u) {
        let edges = triangle_edges(triangle);
        if (edges.degenerate) {
            break;
        }
        let corners = world_triangle(edges);
        var flat: array<vec2<f32>, 3>;
        var behind = false;
        for (var k = 0u; k < 3u; k += 1u) {
            let g = gnomonic(pv.position, frame, corners[k]);
            behind = behind || g.z < 0.0;
            flat[k] = g.xy;
        }
        if (behind) {
            break;
        }
        // The edge that the way leaves the triangle through, from `start`.
        var exit_corner = 3u;
        var exit_distance = 1.0e30;
        var exit_slot = ~0u;
        for (var k = 0u; k < 3u; k += 1u) {
            let ea = flat[k];
            let eb = flat[(k + 1u) % 3u];
            let edge = eb - ea;
            let denom = way.x * edge.y - way.y * edge.x;
            if (abs(denom) < 1.0e-12) {
                continue;
            }
            let rel = ea - start;
            let along = (rel.x * edge.y - rel.y * edge.x) / denom;
            let at = (rel.x * way.y - rel.y * way.x) / denom;
            if (along < 0.0 || at < 0.0 || at > 1.0) {
                continue;
            }
            let slot = edge_slot(edge_key(edges.instance, edges.local[k], edges.local[(k + 1u) % 3u]));
            if (slot == entry_slot) {
                continue;
            }
            if (along < exit_distance) {
                exit_distance = along;
                exit_corner = k;
                exit_slot = slot;
            }
        }
        if (exit_corner == 3u || exit_slot == ~0u || travelled + exit_distance > PROJECTIVE_REACH) {
            break;
        }
        travelled += exit_distance;
        let y = start + exit_distance * way;
        let ia = exit_corner;
        let ib = (exit_corner + 1u) % 3u;
        // The point of the edge in the world that is seen at `y`.
        let ray = normalize(frame * vec3<f32>(y, 1.0));
        let a = corners[ia];
        let b = corners[ib];
        let along3 = b - a;
        let to_a = a - pv.position;
        let aa = dot(along3, along3);
        let ad = dot(along3, ray);
        let denom3 = aa - ad * ad;
        let t = clamp(select(0.0, (ad * dot(ray, to_a) - dot(along3, to_a)) / denom3, denom3 > 0.0), 0.0, 1.0);
        // How the point moves on the plane along `t`, across the way.
        let point = a + t * along3;
        let depth_along = dot(point - pv.position, frame[2]);
        let moved = transpose(frame) * along3;
        let d_flat = (moved.xy - y * moved.z) / depth_along;
        let speed = length(d_flat);
        let count = atomicLoad(&edge_counts[exit_slot]);
        var other = edge_triangles[2u * exit_slot];
        if (other == triangle) {
            other = edge_triangles[2u * exit_slot + 1u];
        }
        let other_edges = triangle_edges(other);
        let outline = count < 2u || triangle_faces(other_edges, world_triangle(other_edges), pv.position) != triangle_faces(edges, corners, pv.position);
        if (speed > 1.0e-12) {
            // Only the ways out of the outline come to it, half of them,
            // and both halves to an edge between triangles that are seen.
            let half = select(1.0, 2.0, outline);
            let inverse_density = half * 0.5 * PI / (density * PROJECTIVE_REACH * speed);
            d_hit += secondary_edge_at(pv, depth, edges, ia, t, inverse_density, ~0u, rng);
        }
        // Over the outline, the way leaves what `pv` sees.
        if (outline) {
            break;
        }
        triangle = other;
        entry_slot = exit_slot;
        start = y;
    }
    return d_hit;
}

// Of the edges that secondary edges are drawn from, which are the corners
// of the triangles, this fraction is drawn uniformly, and the rest by
// `edge_guide`, so that every edge keeps being drawn.
const UNIFORM_EDGE_FRACTION: f32 = 0.25;

struct DrawnEdge {
    // `3 * triangle + corner`, for the edge from the corner to the next.
    edge: u32,
    probability: f32,
}

fn draw_edge(random: f32) -> DrawnEdge {
    let count = 3u * parameters.triangle_count;
    let uniform = 1.0 / f32(count);
    if (parameters.guided_edges == 0u) {
        return DrawnEdge(min(u32(random * f32(count)), count - 1u), uniform);
    }
    var edge = 0u;
    if (random < UNIFORM_EDGE_FRACTION) {
        edge = min(u32(random / UNIFORM_EDGE_FRACTION * f32(count)), count - 1u);
    } else {
        // The first edge whose cumulative probability exceeds the number.
        let target_value = (random - UNIFORM_EDGE_FRACTION) / (1.0 - UNIFORM_EDGE_FRACTION);
        var low = 0u;
        var high = count - 1u;
        while (low < high) {
            let middle = (low + high) / 2u;
            if (edge_guide[middle] > target_value) {
                high = middle;
            } else {
                low = middle + 1u;
            }
        }
        edge = low;
    }
    let below = select(0.0, edge_guide[max(edge, 1u) - 1u], edge > 0u);
    let guided = edge_guide[edge] - below;
    return DrawnEdge(edge, UNIFORM_EDGE_FRACTION * uniform + (1.0 - UNIFORM_EDGE_FRACTION) * guided);
}

var<workgroup> guide_sums: array<f32, 256>;

// Take what the edges brought since the last time into their importance,
// which halves every time, and make `edge_guide` the cumulative distribution
// of it. One workgroup does all, in chunks of consecutive edges.
@compute @workgroup_size(256)
fn guide_edges(@builtin(local_invocation_index) local_index: u32) {
    let count = 3u * parameters.triangle_count;
    let chunk = (count + 255u) / 256u;
    let start = min(local_index * chunk, count);
    let end = min(start + chunk, count);
    var sum = 0.0;
    for (var i = start; i < end; i += 1u) {
        let hits = atomicLoad(&edge_hits[i]);
        atomicStore(&edge_hits[i], 0.0);
        let importance = 0.5 * edge_importance[i] + select(0.0, hits, hits == hits && hits < 1.0e30);
        edge_importance[i] = importance;
        sum += importance;
    }
    guide_sums[local_index] = sum;
    workgroupBarrier();
    // Inclusive prefix sums of the chunks.
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var other = 0.0;
        if (local_index >= stride) {
            other = guide_sums[local_index - stride];
        }
        workgroupBarrier();
        guide_sums[local_index] += other;
        workgroupBarrier();
    }
    let total = guide_sums[255];
    var running = guide_sums[local_index] - sum;
    for (var i = start; i < end; i += 1u) {
        running += edge_importance[i];
        if (total > 0.0) {
            edge_guide[i] = running / total;
        } else {
            edge_guide[i] = f32(i + 1u) / f32(count);
        }
    }
    if (end == count && end > start) {
        // Free of rounding.
        edge_guide[count - 1u] = 1.0;
    }
}

// A hit slides along its ray as the origin of the ray moves, which is the
// previous hit or the camera, and as the triangle moves: moving the origin
// by `d` moves the hit by `d + dot(depth_row, d) * ray_dir`, and moving the
// triangle by `d` at the hit moves it by `-dot(depth_row, d) * ray_dir`.
// Carry the derivatives of the loss by the hits of a replayed path back
// along it, to the geometry and the camera.
fn scatter_hit_motion(record: ptr<function, PathRecord>) {
    var d_next = vec3<f32>(0.0);
    for (var i = (*record).hit_count; i > 0u; i -= 1u) {
        let pv = (*record).hits[i - 1u];
        let d_hit = (*record).d_hits[i - 1u] + d_next;
        d_next = vec3<f32>(0.0);
        if (!pv.valid || all(d_hit == vec3<f32>(0.0))) {
            continue;
        }
        let slide = dot(d_hit, pv.ray_dir);
        add_instance_grad(pv.instance, pv.position, -slide * pv.depth_row);
        for (var c = 0u; c < 3u; c += 1u) {
            let slot = pv.slots[c];
            if (slot == ~0u) {
                continue;
            }
            let local = -slide * pv.barycentrics[c] * pv.local_depth_row;
            for (var k = 0u; k < 3u; k += 1u) {
                if (local[k] != 0.0) {
                    atomicAdd(&vertex_grads[slot * 4u + k], local[k]);
                }
            }
        }
        let d_origin = d_hit + slide * pv.depth_row;
        if (i == 1u) {
            add_camera_grad(d_origin);
        } else {
            d_next = d_origin;
        }
    }
}

// Passes over the vertices and the triangles of a model whose positions are
// parameters, which `vertex_pass` describes.

// Has to match the host!
struct VertexPass {
    // Where the vertices of the model start in `pass_vertex_slots`.
    first_vertex: u32,
    vertex_count: u32,
    model: u32,
    // A geometry of the model, for the passes over triangles: its first
    // index in `mesh_indices`, which counts from its first vertex.
    first_index: u32,
    geometry_vertex_start: u32,
    triangle_count: u32,
    smoothness: f32,
    slot_count: u32,
}

var<uniform> vertex_pass: VertexPass;
// The vertex buffer of the model, which the passes write the positions to.
var<storage, read_write> mesh_vertices: array<Vertex>;
// The vertices of the model as it was loaded.
var<storage, read> base_vertices: array<Vertex>;
var<storage, read> mesh_indices: array<u32>;
var<storage, read_write> weld_keys: array<atomic<u32>>;
// Where the vertex of every slot was, as loaded.
var<storage, read_write> slot_positions: array<vec4<f32>>;
var<storage, read_write> pass_vertex_slots: array<u32>;
var<storage, read> vertex_offsets: array<vec4<f32>>;
var<storage, read_write> normal_sums: array<atomic<f32>>;
// The sum of the offsets of the neighbors of every slot, and their count.
var<storage, read_write> neighbor_sums: array<atomic<f32>>;
var<storage, read_write> smooth_grads: array<f32>;
var<storage, read_write> vertex_pass_grads: array<atomic<f32>>;
var<storage, read_write> pass_normal_grads: array<f32>;

fn vertex_pass_index(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return global_id.y * num_workgroups.x * 64u + global_id.x;
}

// Give every vertex the slot of its position, which its copies share.
@compute @workgroup_size(64)
fn weld_vertices(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let index = vertex_pass_index(global_id, num_workgroups);
    if (index >= vertex_pass.vertex_count) {
        return;
    }
    let position = base_vertices[index].position;
    let q = quantize_position(position);
    var key = hash_jenkins(vertex_pass.model);
    key = hash_combine(key, bitcast<u32>(q.x));
    key = hash_combine(key, bitcast<u32>(q.y));
    key = hash_combine(key, bitcast<u32>(q.z));
    key = max(key, 1u);
    let mask = arrayLength(&weld_keys) - 1u;
    var slot = ~0u;
    var probe = 0u;
    loop {
        if (probe > mask) {
            break;
        }
        let candidate = (key + probe) & mask;
        let result = atomicCompareExchangeWeak(&weld_keys[candidate], 0u, key);
        if (result.exchanged || result.old_value == key) {
            slot = candidate;
            break;
        }
        if (result.old_value != 0u) {
            probe += 1u;
        }
    }
    pass_vertex_slots[vertex_pass.first_vertex + index] = slot;
    if (slot != ~0u) {
        slot_positions[slot] = vec4<f32>(position, 1.0);
    }
}

fn offset_position(slot: u32) -> vec3<f32> {
    return slot_positions[slot].xyz + vertex_offsets[slot].xyz;
}

// Write the positions with their offsets into the vertex buffer.
@compute @workgroup_size(64)
fn apply_vertices(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let index = vertex_pass_index(global_id, num_workgroups);
    if (index >= vertex_pass.vertex_count) {
        return;
    }
    let slot = pass_vertex_slots[vertex_pass.first_vertex + index];
    if (slot != ~0u) {
        mesh_vertices[index].position = base_vertices[index].position + vertex_offsets[slot].xyz;
    }
}

fn triangle_slots(triangle: u32) -> vec3<u32> {
    let base = vertex_pass.first_index + 3u * triangle;
    let start = vertex_pass.first_vertex + vertex_pass.geometry_vertex_start;
    return vec3<u32>(
        pass_vertex_slots[start + mesh_indices[base]],
        pass_vertex_slots[start + mesh_indices[base + 1u]],
        pass_vertex_slots[start + mesh_indices[base + 2u]],
    );
}

// Add the normals of the triangles to the slots of their corners, by area.
@compute @workgroup_size(64)
fn sum_normals(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let triangle = vertex_pass_index(global_id, num_workgroups);
    if (triangle >= vertex_pass.triangle_count) {
        return;
    }
    let slots = triangle_slots(triangle);
    if (any(slots == vec3<u32>(~0u))) {
        return;
    }
    let p0 = offset_position(slots.x);
    let normal = cross(offset_position(slots.y) - p0, offset_position(slots.z) - p0);
    for (var i = 0u; i < 3u; i += 1u) {
        for (var c = 0u; c < 3u; c += 1u) {
            atomicAdd(&normal_sums[slots[i] * 4u + c], normal[c]);
        }
    }
}

@compute @workgroup_size(64)
fn write_normals(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let index = vertex_pass_index(global_id, num_workgroups);
    if (index >= vertex_pass.vertex_count) {
        return;
    }
    let slot = pass_vertex_slots[vertex_pass.first_vertex + index];
    if (slot == ~0u) {
        return;
    }
    let sum = vec3<f32>(
        atomicLoad(&normal_sums[slot * 4u]),
        atomicLoad(&normal_sums[slot * 4u + 1u]),
        atomicLoad(&normal_sums[slot * 4u + 2u]),
    );
    if (dot(sum, sum) > 0.0) {
        mesh_vertices[index].normal = pack4x8snorm(vec4<f32>(normalize(sum), 0.0));
    }
}

// Add the offsets of the neighbors of every slot along the triangles.
@compute @workgroup_size(64)
fn sum_neighbors(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let triangle = vertex_pass_index(global_id, num_workgroups);
    if (triangle >= vertex_pass.triangle_count) {
        return;
    }
    let slots = triangle_slots(triangle);
    if (any(slots == vec3<u32>(~0u))) {
        return;
    }
    for (var i = 0u; i < 3u; i += 1u) {
        for (var j = 1u; j < 3u; j += 1u) {
            let target_slot = slots[i];
            let neighbor = slots[(i + j) % 3u];
            let offset = vertex_offsets[neighbor].xyz;
            for (var c = 0u; c < 3u; c += 1u) {
                atomicAdd(&neighbor_sums[target_slot * 4u + c], offset[c]);
            }
            atomicAdd(&neighbor_sums[target_slot * 4u + 3u], 1.0);
        }
    }
}

// Pull the offset of every vertex towards the mean of those of its
// neighbors, by adding the gradient of `smoothness / 2` times the squared
// distance to it. Offsets are smoothed rather than positions, which would
// shrink every curved surface, so it changes smoothly, from whatever shape.
@compute @workgroup_size(64)
fn add_smoothness(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let slot = vertex_pass_index(global_id, num_workgroups);
    if (slot >= vertex_pass.slot_count) {
        return;
    }
    let count = atomicLoad(&neighbor_sums[slot * 4u + 3u]);
    if (count <= 0.0) {
        return;
    }
    let mean = vec3<f32>(
        atomicLoad(&neighbor_sums[slot * 4u]),
        atomicLoad(&neighbor_sums[slot * 4u + 1u]),
        atomicLoad(&neighbor_sums[slot * 4u + 2u]),
    ) / count;
    let pull = vertex_pass.smoothness * (vertex_offsets[slot].xyz - mean);
    for (var c = 0u; c < 3u; c += 1u) {
        smooth_grads[slot * 4u + c] += pull[c];
    }
}

// Carry the gradients by the normals of the slots, which are normalized sums
// of the cross products of the edges of the triangles around them, to the
// corners of those triangles.
@compute @workgroup_size(64)
fn normals_to_positions(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let triangle = vertex_pass_index(global_id, num_workgroups);
    if (triangle >= vertex_pass.triangle_count) {
        return;
    }
    let slots = triangle_slots(triangle);
    if (any(slots == vec3<u32>(~0u))) {
        return;
    }
    var d_face = vec3<f32>(0.0);
    for (var i = 0u; i < 3u; i += 1u) {
        let slot = slots[i];
        let sum = vec3<f32>(
            atomicLoad(&normal_sums[slot * 4u]),
            atomicLoad(&normal_sums[slot * 4u + 1u]),
            atomicLoad(&normal_sums[slot * 4u + 2u]),
        );
        let sum_length = length(sum);
        if (sum_length <= 0.0) {
            continue;
        }
        let normal = sum / sum_length;
        let d_normal = vec3<f32>(
            pass_normal_grads[slot * 4u],
            pass_normal_grads[slot * 4u + 1u],
            pass_normal_grads[slot * 4u + 2u],
        );
        d_face += (d_normal - normal * dot(normal, d_normal)) / sum_length;
    }
    let p0 = offset_position(slots.x);
    let d_corners = cross_grad(offset_position(slots.y) - p0, offset_position(slots.z) - p0, d_face);
    for (var i = 0u; i < 3u; i += 1u) {
        for (var c = 0u; c < 3u; c += 1u) {
            atomicAdd(&vertex_pass_grads[slots[i] * 4u + c], d_corners[i][c]);
        }
    }
}
