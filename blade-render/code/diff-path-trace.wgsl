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
// Translating objects and the camera also moves what the camera sees, which
// `edges` and `scatter_primary_geometry_grad` differentiate: the first moves
// the discontinuities of the image along triangle edges, and the second the
// interpolated normal of each primary hit. Everything past the primary hit
// is held fixed, which leaves out how shadows, reflections, and the lighting
// at the primary hit move with the geometry.
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
const MATERIAL_PARAM_COUNT: u32 = 8u;
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
}

// Has to match the host!
// Triangles of one hit entry, for sampling their edges.
struct EdgeEntry {
    hit_entry: u32,
    instance: u32,
    first_triangle: u32,
    triangle_count: u32,
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
var<storage, read_write> edge_counts: array<atomic<u32>>;
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
    return v;
}

// Derivative of the loss by the material of one vertex.
struct MaterialGrad {
    diffuse_albedo: vec3<f32>,
    specular_f0: vec3<f32>,
    roughness: f32,
    emissive: vec3<f32>,
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
    (*grad).normal += bsdf_normal_grad(bsdf, mat, d_value);
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

// The interpolated normal of a primary hit moves as the camera or the hit
// instance move, since another point of the triangle comes under the pixel.
// Carry the derivative of the loss by the shading normal, `d_normal`, to the
// translations through it, and to the rotation, which also turns the normal.
//
// The tangent frame is taken to turn with the normal, which is exact without
// a normal map, and the texture coordinates are taken to stay.
fn scatter_primary_geometry_grad(intersection: RayIntersection, direction: vec3<f32>, v: DiffVertex, d_normal: vec3<f32>) {
    if (all(d_normal == vec3<f32>(0.0))) {
        return;
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
        return;
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
    add_camera_grad(d_camera);
    // Moving the triangle is moving the camera the other way. Turning it
    // moves the hit, like translating it there would, and turns the normal.
    add_instance_grad(intersection.instance_index, v.surface.position, -d_camera);
    add_rotation_grad(intersection.instance_index, cross(shading_normal, d_normal));
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

fn horizon_weight_grad(cosine: f32) -> f32 {
    let t = cosine / HORIZON_BAND;
    return select(0.0, 6.0 * t * (1.0 - t) / HORIZON_BAND, t > 0.0 && t < 1.0);
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

// Walk one path and return its radiance.
//
// Without `adjoint`, the walk fills the `record`. With it, the walk is a
// replay of the recorded one, given the same random state, and it scatters
// the gradient of the loss, of which `adjoint` is the derivative by the
// radiance, using the radiance arriving at every vertex from `incident`.
fn walk_path(
    start_dir: vec3<f32>,
    rng: ptr<function, RandomState>,
    record: ptr<function, PathRecord>,
    replay: bool,
    adjoint: vec3<f32>,
    incident: ptr<function, array<vec3<f32>, PATH_LENGTH>>,
) -> vec3<f32> {
    let importance = parameters.environment_importance_sampling != 0u;
    let num_light = f32(parameters.num_environment_samples);
    let max_bounces = min(parameters.max_bounces, MAX_DIFF_BOUNCES);
    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var position = camera.position;
    var direction = start_dir;
    // Negative for the camera ray, like in "path-trace.wgsl".
    var bsdf_pdf = -1.0;
    var t_min = 0.0;
    (*record).length = 0u;

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
        let unbent_normal = v.surface.normal;
        v.surface.normal = bend_normal(unbent_normal, view_dir);
        let normal = v.surface.normal;
        let flat_normal = v.surface.flat_normal;
        // Derivative of the loss by everything found at this vertex.
        let d_found = adjoint * throughput;
        var grad = MaterialGrad(vec3<f32>(0.0), vec3<f32>(0.0), 0.0, d_found, vec3<f32>(0.0), vec3<f32>(0.0));
        var found = v.emissive;
        position = v.surface.position;
        t_min = parameters.t_start;

        // Each sample is a separate path here, so the path always goes on.
        let will_extend = bounce < max_bounces;
        let bsdf_count = select(0.0, 1.0, will_extend);

        for (var i = 0u; i < parameters.num_environment_samples; i += 1u) {
            let ls = sample_environment(importance, rng);
            if (ls.pdf <= 0.0) {
                continue;
            }
            let bsdf = evaluate_bsdf_gradient(v.material, normal, view_dir, ls.dir);
            let horizon_cosine = dot(ls.dir, flat_normal);
            let horizon = horizon_weight(horizon_cosine);
            if (horizon <= 0.0 || all(bsdf.value <= vec3<f32>(0.0))
                || is_occluded(position, ls.dir)) {
                continue;
            }
            let other_pdf = compute_bsdf_pdf(v.material, normal, view_dir, ls.dir);
            let unfaded = mis_weight(num_light, ls.pdf, bsdf_count, other_pdf) / (num_light * ls.pdf);
            let weight = unfaded * horizon;
            let light = env_radiance(ls.texel);
            found += bsdf.value * light * weight;
            if (replay) {
                add_bsdf_grad(&grad, bsdf, v.material, d_found * light * weight);
                scatter_env_grad(ls.texel, d_found * bsdf.value * weight);
                grad.flat_normal += dot(d_found, bsdf.value * light) * unfaded
                    * horizon_weight_grad(horizon_cosine) * ls.dir;
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
            let bs = sample_bsdf(v.material, normal, view_dir, rng);
            let horizon_cosine = dot(bs.dir, flat_normal);
            let horizon = horizon_weight(horizon_cosine);
            if (bs.pdf > 0.0 && horizon > 0.0) {
                let bsdf = evaluate_bsdf_gradient(v.material, normal, view_dir, bs.dir);
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
                            * horizon_weight_grad(horizon_cosine) * bs.dir;
                    }
                    bsdf_pdf = bs.pdf;
                    direction = bs.dir;
                }
            }
        }
        if (replay) {
            grad.normal = bend_normal_grad(unbent_normal, view_dir, grad.normal);
            v.surface.normal = unbent_normal;
            scatter_material_grad(v, grad);
            if (bounce == 0u && parameters.geometry != 0u) {
                scatter_primary_geometry_grad(intersection, start_dir, v, grad.normal);
                // The triangle turns with the instance.
                add_rotation_grad(intersection.instance_index, cross(flat_normal, grad.flat_normal));
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

fn primary_direction(pixel: vec2<u32>, rng: ptr<function, RandomState>) -> vec3<f32> {
    let jitter = select(
        vec2<f32>(0.5),
        vec2<f32>(random_gen(rng), random_gen(rng)),
        parameters.jitter_primary_rays != 0u,
    );
    return get_ray_direction_at(camera, vec2<f32>(pixel) + jitter);
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
        let radiance = walk_path(dir, &rng, &record, false, vec3<f32>(0.0), &incident);
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
    if (all(d_pixel == vec3<f32>(0.0))) {
        return;
    }
    var rng = random_init(pixel_index, parameters.frame_index);
    var record: PathRecord;
    var incident: array<vec3<f32>, PATH_LENGTH>;
    for (var i = 0u; i < num_paths; i += 1u) {
        let dir = primary_direction(global_id.xy, &rng);
        var replay_rng = rng;
        let radiance = walk_path(dir, &rng, &record, false, vec3<f32>(0.0), &incident);
        if (!all(radiance == radiance)) {
            continue;
        }
        incident = compute_incident_radiance(&record);
        let adjoint = d_pixel * radiance_scale(radiance);
        walk_path(dir, &replay_rng, &record, true, adjoint, &incident);
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
fn sample_edge(a: vec3<f32>, b: vec3<f32>, instance: u32, rng: ptr<function, RandomState>) {
    if (!is_in_front(camera, a) || !is_in_front(camera, b)) {
        return;
    }
    let pa = get_projected_pixel_float(camera, a);
    let pb = get_projected_pixel_float(camera, b);
    let segment = pb - pa;
    let segment_length = length(segment);
    if (segment_length < 1.0e-6) {
        return;
    }
    let across = vec2<f32>(-segment.y, segment.x) / segment_length;
    let size = vec2<f32>(camera.target_size);
    let samples = max(parameters.edge_samples, 1u);
    var record: PathRecord;
    var incident: array<vec3<f32>, PATH_LENGTH>;
    for (var i = 0u; i < samples; i += 1u) {
        let t = (f32(i) + random_gen(rng)) / f32(samples);
        let p = mix(pa, pb, t);
        if (any(p < vec2<f32>(0.0)) || any(p >= size)) {
            continue;
        }
        let pixel = vec2<u32>(p);
        let d_pixel = image_adjoint[pixel.y * camera.target_size.x + pixel.x].xyz;
        if (all(d_pixel == vec3<f32>(0.0))) {
            continue;
        }
        let offset = 1.0e-3 * across;
        var side_rng = *rng;
        let below = walk_path(get_ray_direction_at(camera, p - offset), &side_rng, &record, false, vec3<f32>(0.0), &incident);
        side_rng = *rng;
        let above = walk_path(get_ray_direction_at(camera, p + offset), &side_rng, &record, false, vec3<f32>(0.0), &incident);
        *rng = side_rng;
        let jump = select(vec3<f32>(0.0), below * radiance_scale(below), all(below == below))
            - select(vec3<f32>(0.0), above * radiance_scale(above), all(above == above));
        let weight = dot(d_pixel, jump) * segment_length / f32(samples);
        if (weight == 0.0) {
            continue;
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
        let jacobian = projection_jacobian(camera, point);
        let velocity = vec3<f32>(dot(across, jacobian[0]), dot(across, jacobian[1]), dot(across, jacobian[2]));
        add_instance_grad(instance, point, weight * velocity);
        add_camera_grad(-weight * velocity);
    }
}

// The boundary term of the translations, sampled along the edges of every
// triangle in the scene. Requires jittered primary rays, i.e. a box filter
// over each pixel, for the image to change continuously as edges move.
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
        sample_edge(to_world * vec4<f32>(a, 1.0), to_world * vec4<f32>(b, 1.0), edges.instance, &rng);
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
fn edge_triangle_count(key: u32) -> u32 {
    let mask = edge_slot_mask();
    for (var probe = 0u; probe <= mask; probe += 1u) {
        let slot = (key + probe) & mask;
        let found = atomicLoad(&edge_keys[slot]);
        if (found == key) {
            return atomicLoad(&edge_counts[slot]);
        }
        if (found == 0u) {
            break;
        }
    }
    return 0u;
}

struct TriangleEdges {
    instance: u32,
    // The corners in the space of the instance.
    local: mat3x3<f32>,
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
    edges.local = entry.geometry_to_object * mat3x4<f32>(
        vec4<f32>((*vptr)[indices.x].position, 1.0),
        vec4<f32>((*vptr)[indices.y].position, 1.0),
        vec4<f32>((*vptr)[indices.z].position, 1.0),
    );
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
    let mask = edge_slot_mask();
    for (var i = 0u; i < 3u; i += 1u) {
        let a = edges.local[i];
        let b = edges.local[(i + 1u) % 3u];
        if (all(quantize_position(a) == quantize_position(b))) {
            continue;
        }
        let key = edge_key(edges.instance, a, b);
        var probe = 0u;
        loop {
            if (probe > mask) {
                break;
            }
            let slot = (key + probe) & mask;
            let result = atomicCompareExchangeWeak(&edge_keys[slot], 0u, key);
            if (result.exchanged || result.old_value == key) {
                atomicAdd(&edge_counts[slot], 1u);
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
