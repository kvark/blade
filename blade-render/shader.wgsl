const MAX_BOUNCES: i32 = 3;
const PI: f32 = 3.1415926;

struct CameraParams {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>,
    fov: vec2<f32>,
    mouse_pos: vec2<u32>,
};
struct MainParams {
    frame_index: u32,
    debug_mode: u32,
    num_environment_samples: u32,
    temporal_history: u32,
};

//Must match host side `DebugMode`
const DEBUG_MODE_NONE: u32 = 0u;
const DEBUG_MODE_DEPTH: u32 = 1u;
const DEBUG_MODE_NORMAL: u32 = 2u;

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
var<storage, read> vertex_buffers: binding_array<VertexBuffer, 1>;
var<storage, read> index_buffers: binding_array<IndexBuffer, 1>;
var textures: binding_array<texture_2d<f32>, 1>;
var sampler_linear: sampler;

struct HitEntry {
    index_buf: u32,
    vertex_buf: u32,
    // packed object->world rotation quaternion
    rotation: u32,
    //geometry_to_object_rm: mat3x4<f32>,
    base_color_texture: u32,
    // packed color factor
    base_color_factor: u32,
}
var<storage, read> hit_entries: array<HitEntry>;

var<uniform> camera: CameraParams;
var<uniform> parameters: MainParams;
var acc_struct: acceleration_structure;

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}
fn qinv(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz,q.w);
}
fn make_quat(m: mat3x3<f32>) -> vec4<f32> {
    let w = sqrt(1.0 + m[0].x + m[1].y + m[2].z) / 2.0;
    let v = vec3<f32>(m[1].z - m[2].y, m[2].x - m[0].z, m[0].y - m[1].x);
    return vec4<f32>(v / (4.0 * w), w);
}

struct DebugPoint {
    pos: vec3<f32>,
    color: u32,
}
struct DebugLine {
    a: DebugPoint,
    b: DebugPoint,
}
struct DebugVariance {
    color_sum: vec3<f32>,
    color2_sum: vec3<f32>,
    count: u32,
}
struct DebugBuffer {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
    capacity: u32,
    open: u32,
    variance: DebugVariance,
    lines: array<DebugLine>,
}
var<storage, read_write> debug_buf: DebugBuffer;

fn debug_line(a: vec3<f32>, b: vec3<f32>, color: u32) {
    if (debug_buf.open != 0u) {
        let index = atomicAdd(&debug_buf.instance_count, 1u);
        if (index < debug_buf.capacity) {
            debug_buf.lines[index] = DebugLine(DebugPoint(a, color), DebugPoint(b, color));
        } else {
            // ensure the final value is never above the capacity
            atomicSub(&debug_buf.instance_count, 1u);
        }
    }
}
struct DebugVarying {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}
@vertex
fn debug_vs(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> DebugVarying {
    let line = debug_buf.lines[instance_id];
    var point = line.a;
    if (vertex_id != 0u) {
        point = line.b;
    }

    let world_dir = point.pos - camera.position;
    let local_dir = qrot(qinv(camera.orientation), world_dir);
    let ndc = local_dir.xy / tan(camera.fov);

    var out: DebugVarying;
    out.pos = vec4<f32>(ndc, 0.0, local_dir.z);
    out.color = unpack4x8unorm(point.color);
    return out;
}
@fragment
fn debug_fs(in: DebugVarying) -> @location(0) vec4<f32> {
    return in.color;
}

var out_depth: texture_storage_2d<r32float, write>;
var out_basis: texture_storage_2d<rgba8snorm, write>;
var out_albedo: texture_storage_2d<rgba8unorm, write>;

fn get_ray_direction(global_id: vec2<u32>, target_size: vec2<u32>) -> vec3<f32> {
    let half_size = vec2<f32>(target_size >> vec2<u32>(1u));
    let ndc = (vec2<f32>(global_id) - half_size) / half_size;
    let local_dir = vec3<f32>(ndc * tan(camera.fov), 1.0);
    return normalize(qrot(camera.orientation, local_dir));
}

@compute @workgroup_size(8, 8)
fn fill_gbuf(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_size = textureDimensions(out_depth);
    if (any(global_id.xy > target_size)) {
        return;
    }

    var rq: ray_query;
    let ray_dir = get_ray_direction(global_id.xy, target_size);
    rayQueryInitialize(&rq, acc_struct, RayDesc(0x10u, 0xFFu, 0.0, camera.depth, camera.position, ray_dir));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);

    var depth = 0.0;
    var basis = vec4<f32>(0.0);
    var albedo = vec3<f32>(0.0);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        let enable_debug = all(global_id.xy == camera.mouse_pos);
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
        let normal_object = vec3<f32>(normal_rough, sqrt(max(0.0, 1.0 - dot(normal_rough, normal_rough))));
        let object_to_world_rot = normalize(unpack4x8snorm(entry.rotation));
        let normal_world = qrot(object_to_world_rot, normal_object);

        if (enable_debug) {
            let debug_len = intersection.t * 0.2;
            let positions = intersection.object_to_world * mat3x4(
                vec4<f32>(vertices[0].pos, 1.0), vec4<f32>(vertices[1].pos, 1.0), vec4<f32>(vertices[2].pos, 1.0)
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

struct StoredReservoir {
    light_dir: vec3<f32>,
    target_score: f32,
    contribution_weight: f32,
    confidence: f32,
}
var<storage, read_write> reservoirs: array<StoredReservoir>;

struct LightSample {
    dir: vec3<f32>,
    radiance: vec3<f32>,
    pdf: f32,
}

struct LiveReservoir {
    selected_dir: vec3<f32>,
    selected_target_score: f32,
    weight_sum: f32,
    count: u32,
}

fn compute_target_score(radiance: vec3<f32>) -> f32 {
    return dot(radiance, vec3<f32>(0.3, 0.4, 0.3));
}

fn bump_reservoir(r: ptr<function, LiveReservoir>) {
    (*r).count += 1u;
}
fn make_reservoir(ls: LightSample) -> LiveReservoir {
    var r: LiveReservoir;
    r.selected_dir = ls.dir;
    r.selected_target_score = compute_target_score(ls.radiance);
    r.weight_sum = r.selected_target_score / ls.pdf;
    r.count = 1u;
    return r;
}
fn merge_reservoir(r: ptr<function, LiveReservoir>, other: LiveReservoir, random: f32) -> bool {
    (*r).weight_sum += other.weight_sum;
    (*r).count += other.count;
    if ((*r).weight_sum * random < other.weight_sum) {
        (*r).selected_dir = other.selected_dir;
        (*r).selected_target_score = other.selected_target_score;
        return true;
    } else {
        return false;
    }
}
fn update_reservoir(r: ptr<function, LiveReservoir>, ls: LightSample, random: f32) -> bool {
    let other = make_reservoir(ls);
    return merge_reservoir(r, other, random);
}
fn unpack_reservoir(f: StoredReservoir, max_count: u32) -> LiveReservoir {
    var r: LiveReservoir;
    r.selected_dir = f.light_dir;
    r.selected_target_score = f.target_score;
    let count = min(u32(f.confidence), max_count);
    r.weight_sum = f.contribution_weight * f.target_score * f32(count);
    r.count = count;
    return r;
}
fn pack_reservoir(r: LiveReservoir) -> StoredReservoir {
    var f: StoredReservoir;
    f.light_dir = r.selected_dir;
    f.target_score = r.selected_target_score;
    f.confidence = f32(r.count);
    let denom = f.target_score * f.confidence;
    f.contribution_weight = select(0.0, r.weight_sum / denom, denom > 0.0);
    return f;
}

var in_depth: texture_2d<f32>;
var in_basis: texture_2d<f32>;
var in_albedo: texture_2d<f32>;
var output: texture_storage_2d<rgba16float, write>;

struct RandomState {
    seed: u32,
    index: u32,
}

// 32 bit Jenkins hash
fn hash_jenkins(value: u32) -> u32 {
    var a = value;
    // http://burtleburtle.net/bob/hash/integer.html
    a = (a + 0x7ed55d16u) + (a << 12u);
    a = (a ^ 0xc761c23cu) ^ (a >> 19u);
    a = (a + 0x165667b1u) + (a << 5u);
    a = (a + 0xd3a2646cu) ^ (a << 9u);
    a = (a + 0xfd7046c5u) + (a << 3u);
    a = (a ^ 0xb55a4f09u) ^ (a >> 16u);
    return a;
}

fn random_init(pixel_index: u32, frame_index: u32) -> RandomState {
    var rs: RandomState;
    rs.seed = hash_jenkins(pixel_index) + frame_index;
    rs.index = 0u;
    return rs;
}

fn rot32(x: u32, bits: u32) -> u32 {
    return (x << bits) | (x >> (32u - bits));
}

// https://en.wikipedia.org/wiki/MurmurHash
fn murmur3(rng: ptr<function, RandomState>) -> u32 {
    let c1 = 0xcc9e2d51u;
    let c2 = 0x1b873593u;
    let r1 = 15u;
    let r2 = 13u;
    let m = 5u;
    let n = 0xe6546b64u;

    var hash = (*rng).seed;
    (*rng).index += 1u;
    var k = (*rng).index;
    k *= c1;
    k = rot32(k, r1);
    k *= c2;

    hash ^= k;
    hash = rot32(hash, r2) * m + n;

    hash ^= 4u;
    hash ^= (hash >> 16u);
    hash *= 0x85ebca6bu;
    hash ^= (hash >> 13u);
    hash *= 0xc2b2ae35u;
    hash ^= (hash >> 16u);

    return hash;
}

fn random_gen(rng: ptr<function, RandomState>) -> f32 {
    let v = murmur3(rng);
    let one = bitcast<u32>(1.0);
    let mask = (1u << 23u) - 1u;
    return bitcast<f32>((mask & v) | one) - 1.0;
}

fn sample_disk(rand: vec2<f32>) -> vec2<f32> {
    let angle = 2.0 * PI * rand.x;
    return vec2<f32>(cos(angle), sin(angle)) * sqrt(rand.y);
}

fn square(v: f32) -> f32 {
    return v * v;
}

fn sample_uniform_hemisphere(rng: ptr<function, RandomState>) -> vec3<f32> {
    // See (6-8) in https://mathworld.wolfram.com/SpherePointPicking.html
    let r = random_gen(rng);
    let h = random_gen(rng);
    let tangential = sample_disk(vec2<f32>(r, 1.0 - square(h)));
    return vec3<f32>(tangential.xy, h);
}

fn evaluate_environment(dir: vec3<f32>) -> vec3<f32> {
    //Note: Y axis is up
    return vec3<f32>(max(2.0 * dir.y, 0.0));
}

struct Surface {
    basis: vec4<f32>,
    albedo: vec3<f32>,
}

fn evaluate_color(surface: Surface, dir: vec3<f32>) -> vec3<f32> {
    let lambert_brdf = 1.0 / PI;
    let lambert_term = qrot(qinv(surface.basis), dir).z;
    if (lambert_term <= 0.0) {
        return vec3<f32>(0.0);
    }
    return surface.albedo * lambert_brdf;
}

fn compute_restir(ray_dir: vec3<f32>, depth: f32, surface: Surface, pixel_index: u32, rng: ptr<function, RandomState>) -> vec3<f32> {
    if (parameters.debug_mode == DEBUG_MODE_DEPTH) {
        return vec3<f32>(depth / camera.depth);
    }
    if (depth == 0.0) {
        return evaluate_environment(ray_dir);
    }

    let position = camera.position + depth * ray_dir;
    let normal = qrot(surface.basis, vec3<f32>(0.0, 0.0, 1.0));
    if (parameters.debug_mode == DEBUG_MODE_NORMAL) {
        return normal;
    }

    let start_t = 0.5; // some offset required to avoid self-shadowing

    var reservoir = LiveReservoir();
    var radiance = vec3<f32>(0.0);
    var rq: ray_query;
    for (var i = 0u; i < parameters.num_environment_samples; i += 1u) {
        var ls = LightSample();
        ls.pdf = 1.0 / (2.0 * PI);
        let light_dir_tbn = sample_uniform_hemisphere(rng);
        ls.dir = qrot(surface.basis, light_dir_tbn);
        ls.radiance = evaluate_environment(ls.dir);
        let target_score = compute_target_score(ls.radiance);
        if (target_score < 0.01) {
            bump_reservoir(&reservoir);
            continue;
        }

        rayQueryInitialize(&rq, acc_struct, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xFFu,
            start_t, camera.depth, position, ls.dir));
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
        if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
            bump_reservoir(&reservoir);
            continue;
        }

        if (update_reservoir(&reservoir, ls, random_gen(rng))) {
            radiance = ls.radiance;
        }
    }

    if (parameters.temporal_history != 0u) {
        let prev = unpack_reservoir(reservoirs[pixel_index], parameters.temporal_history);
        if (merge_reservoir(&reservoir, prev, random_gen(rng))) {
            radiance = evaluate_environment(prev.selected_dir);
        }
    }
    let stored = pack_reservoir(reservoir);
    reservoirs[pixel_index] = stored;
    return stored.contribution_weight * radiance * evaluate_color(surface, stored.light_dir);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_size = textureDimensions(output);
    if (any(global_id.xy > target_size)) {
        return;
    }

    let global_index = global_id.y * target_size.x + global_id.x;
    var rng = random_init(global_index, parameters.frame_index);

    var surface: Surface;
    let ray_dir = get_ray_direction(global_id.xy, target_size);
    let depth = textureLoad(in_depth, global_id.xy, 0).x;
    surface.basis = normalize(textureLoad(in_basis, global_id.xy, 0));
    surface.albedo = textureLoad(in_albedo, global_id.xy, 0).xyz;
    let color = compute_restir(ray_dir, depth, surface, global_index, &rng);
    if (all(global_id.xy == camera.mouse_pos)) {
        debug_buf.variance.color_sum += color;
        debug_buf.variance.color2_sum += color * color;
        debug_buf.variance.count += 1u;
    }
    textureStore(output, global_id.xy, vec4<f32>(color, 1.0));
}

var input: texture_2d<f32>;
struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) input_size: vec2<u32>,
}

@vertex
fn blit_vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo: VertexOutput;
    vo.clip_pos = vec4<f32>(f32(vi & 1u) * 4.0 - 1.0, f32(vi & 2u) * 2.0 - 1.0, 0.0, 1.0);
    vo.input_size = textureDimensions(input, 0);
    return vo;
}

@fragment
fn blit_fs(vo: VertexOutput) -> @location(0) vec4<f32> {
    let tc = vec2<i32>(i32(vo.clip_pos.x), i32(vo.input_size.y) - i32(vo.clip_pos.y));
    return textureLoad(input, tc, 0);
}
