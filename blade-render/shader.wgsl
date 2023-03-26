const MAX_BOUNCES: i32 = 3;
const PI: f32 = 3.1415926;

struct Parameters {
    cam_position: vec3<f32>,
    depth: f32,
    cam_orientation: vec4<f32>,
    fov: vec2<f32>,
    frame_index: u32,
    debug_mode: u32,
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

var<uniform> parameters: Parameters;
var acc_struct: acceleration_structure;


var output: texture_storage_2d<rgba8unorm, write>;

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}

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

fn get_environment_radiance(dir: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(0.0), vec3<f32>(1.0), dir.y > 0.0);
}

fn compute_hit_color(ri: RayIntersection, ray_dir: vec3<f32>, hit_world: vec3<f32>, rng: ptr<function, RandomState>) -> vec3<f32> {
    let entry = hit_entries[ri.instance_custom_index + ri.geometry_index];
    if (parameters.debug_mode == DEBUG_MODE_DEPTH) {
        return vec3<f32>(ri.t / parameters.depth);
    }

    var indices = ri.primitive_index * 3u + vec3<u32>(0u, 1u, 2u);
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

    let barycentrics = vec3<f32>(1.0 - ri.barycentrics.x - ri.barycentrics.y, ri.barycentrics);
    //let pos_object = mat3x3(vertices[0].pos, vertices[1].pos, vertices[2].pos) * barycentrics;
    //let pos_world = ri.object_to_world * vec4<f32>(pos_object, 1.0);
    let tex_coords = mat3x2(vertices[0].tex_coords, vertices[1].tex_coords, vertices[2].tex_coords) * barycentrics;
    let normal_rough = mat3x2(unpack2x16snorm(vertices[0].normal), unpack2x16snorm(vertices[1].normal), unpack2x16snorm(vertices[2].normal)) * barycentrics;
    let normal_object = vec3<f32>(normal_rough, sqrt(max(0.0, 1.0 - dot(normal_rough, normal_rough))));
    let object_to_world_rot = normalize(unpack4x8snorm(entry.rotation));
    let normal_world = qrot(object_to_world_rot, normal_object);

    // Note: this line allows to check the correctness of data passed in
    //return abs(pos_world - hit_world) * 1000000.0;
    if (parameters.debug_mode == DEBUG_MODE_NORMAL) {
        return normal_world;
    }

    let pre_tangent = select(
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0),
        abs(dot(normal_world, vec3<f32>(0.0, 1.0, 0.0))) < abs(dot(normal_world, vec3<f32>(0.0, 0.0, 1.0))));
    let bitangent = normalize(cross(normal_world, pre_tangent));
    let tangent = normalize(cross(bitangent, normal_world));

    let steradians_in_hemisphere = 2.0 * PI;
    let lambert_brdf = 1.0 / PI;
    let num_diffuse_samples = 10;
    var color = vec3<f32>(0.0);
    var rq: ray_query;
    for (var i = 0i; i < num_diffuse_samples; i += 1) {
        let light_dir_tbn = sample_uniform_hemisphere(rng);
        let light_dir = light_dir_tbn.x * tangent + light_dir_tbn.y * bitangent + light_dir_tbn.z * normal_world;
        let lambert_term = light_dir_tbn.z;
        let estimate_color = get_environment_radiance(light_dir) * lambert_term;
        if (dot(estimate_color, estimate_color) < 0.01) {
            continue;
        }

        rayQueryInitialize(&rq, acc_struct, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xFFu, 0.5, parameters.depth, hit_world, light_dir));
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
        if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
            color += estimate_color * steradians_in_hemisphere * lambert_brdf;
        }
    }
    color /= f32(num_diffuse_samples);

    let base_color_factor = unpack4x8unorm(entry.base_color_factor);
    let lod = 0.0; //TODO: this is actually complicated
    let base_color_sample = textureSampleLevel(textures[entry.base_color_texture], sampler_linear, tex_coords, lod);

    //return (base_color_factor * base_color_sample).xyz * color;
    return color;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_size = textureDimensions(output);
    let half_size = vec2<f32>(target_size >> vec2<u32>(1u));
    let ndc = (vec2<f32>(global_id.xy) - half_size) / half_size;
    if (any(global_id.xy > target_size)) {
        return;
    }

    let global_index = global_id.y * target_size.x + global_id.x;
    var rng = random_init(global_index, parameters.frame_index);
    let local_dir = vec3<f32>(ndc * tan(parameters.fov), 1.0);
    let world_dir = normalize(qrot(parameters.cam_orientation, local_dir));

    var rq: ray_query;
    var ray_pos = parameters.cam_position;
    var ray_dir = world_dir;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0x10u, 0xFFu, 0.0, parameters.depth, ray_pos, ray_dir));
    var iterations = 0u;
    while (rayQueryProceed(&rq)) {iterations += 1u;}
    let intersection = rayQueryGetCommittedIntersection(&rq);

    var color = vec3<f32>(0.0);
    if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
        color = get_environment_radiance(ray_dir);
    } else {
        let expected = ray_pos + intersection.t * ray_dir;
        color = compute_hit_color(intersection, ray_dir, expected, &rng);
    }
    textureStore(output, global_id.xy, vec4<f32>(color, 1.0));
}

var input: texture_2d<f32>;
struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) input_size: vec2<u32>,
}

@vertex
fn draw_vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo: VertexOutput;
    vo.clip_pos = vec4<f32>(f32(vi & 1u) * 4.0 - 1.0, f32(vi & 2u) * 2.0 - 1.0, 0.0, 1.0);
    vo.input_size = textureDimensions(input, 0);
    return vo;
}

@fragment
fn draw_fs(vo: VertexOutput) -> @location(0) vec4<f32> {
    let tc = vec2<i32>(i32(vo.clip_pos.x), i32(vo.input_size.y) - i32(vo.clip_pos.y));
    return textureLoad(input, tc, 0);
}
