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

struct DebugEntry {
    custom_index: u32,
    depth: f32,
    tex_coords: vec2<f32>,
    base_color_texture: u32,
    normal_texture: u32,
    pad: vec2<u32>,
    position: vec3<f32>,
    flat_normal: vec3<f32>,
}

struct DebugBuffer {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
    capacity: u32,
    open: u32,
    variance: DebugVariance,
    entry: DebugEntry,
    lines: array<DebugLine>,
}

struct CameraParams {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>,
    fov: vec2<f32>,
    film_offset: vec2<f32>,
    target_size: vec2<u32>,
}

struct DebugVarying {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) dir: vec3<f32>,
}

const DEBUG_MODE: bool = true;
const MAX_LOCAL_LIGHTS: u32 = 8u;
const MAX_LOCAL_LIGHTS_LEN: u32 = 8u;
const MAX_JOINTS_PER_DRAW: u32 = 64u;
const MAX_JOINTS_PER_DRAW_LEN: u32 = 64u;
const DebugMode_Final: u32 = 0u;
const DebugMode_Depth: u32 = 1u;
const DebugMode_DiffuseAlbedoTexture: u32 = 2u;
const DebugMode_DiffuseAlbedoFactor: u32 = 3u;
const DebugMode_NormalTexture: u32 = 4u;
const DebugMode_NormalScale: u32 = 5u;
const DebugMode_GeometryNormal: u32 = 6u;
const DebugMode_ShadingNormal: u32 = 7u;
const DebugMode_Motion: u32 = 8u;
const DebugMode_HitConsistency: u32 = 9u;
const DebugMode_SampleReuse: u32 = 10u;
const DebugMode_Roughness: u32 = 11u;
const DebugMode_SpecularF0: u32 = 12u;
const DebugMode_Emissive: u32 = 13u;
const DebugMode_Variance: u32 = 15u;
const DebugDrawFlags_SPACE: u32 = 1u;
const DebugDrawFlags_GEOMETRY: u32 = 2u;
const DebugDrawFlags_RESTIR: u32 = 4u;
const DebugTextureFlags_ALBEDO: u32 = 1u;
const DebugTextureFlags_NORMAL: u32 = 2u;
const DebugTextureFlags_METALLIC_ROUGHNESS: u32 = 4u;
const DebugTextureFlags_EMISSIVE: u32 = 8u;
const VFLIP: vec2<f32> = vec2<f32>(1f, -1f);

var<storage, read_write> debug_buf: DebugBuffer;
var<uniform> camera: CameraParams;
var<storage> debug_lines: array<DebugLine>;
var depth: texture_2d<f32>;

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return (v + (vec3(2f) * cross(q.xyz, (cross(q.xyz, v) + (vec3(q.w) * v)))));
}

fn qinv(q_1: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-(q_1.xyz), q_1.w);
}

fn make_quat(m: mat3x3<f32>) -> vec4<f32> {
    var q_2: vec4<f32>;
    var dif10: f32;
    var omm22: f32;
    var sum10: f32;
    var opm22: f32;

    q_2 = vec4<f32>();
    if (m[2].z < 0f) {
        dif10 = (m[1].y - m[0].x);
        omm22 = (1f - m[2].z);
        let _e18 = dif10;
        if (_e18 < 0f) {
            let _e21 = omm22;
            let _e22 = dif10;
            q_2 = vec4<f32>((_e21 - _e22), (m[0].y + m[1].x), (m[0].z + m[2].x), (m[1].z - m[2].y));
        } else {
            let _e45 = omm22;
            let _e46 = dif10;
            q_2 = vec4<f32>((m[0].y + m[1].x), (_e45 + _e46), (m[1].z + m[2].y), (m[2].x - m[0].z));
        }
    } else {
        sum10 = (m[1].y + m[0].x);
        opm22 = (1f + m[2].z);
        let _e70 = sum10;
        if (_e70 < 0f) {
            let _e83 = opm22;
            let _e84 = sum10;
            q_2 = vec4<f32>((m[0].z + m[2].x), (m[1].z + m[2].y), (_e83 - _e84), (m[0].y - m[1].x));
        } else {
            let _e107 = opm22;
            let _e108 = sum10;
            q_2 = vec4<f32>((m[1].z - m[2].y), (m[2].x - m[0].z), (m[0].y - m[1].x), (_e107 + _e108));
        }
    }
    let _e111 = q_2;
    return normalize(_e111);
}

fn shortest_arc_quat(a: vec3<f32>, b: vec3<f32>) -> vec4<f32> {
    if (dot(a, b) < -(0.99999f)) {
        return select(vec4<f32>(1f, 0f, 0f, 0f), vec4<f32>(0f, 1f, 0f, 0f), (abs(a.x) > abs(a.y)));
    } else {
        return normalize(vec4<f32>(cross(a, b), (1f + dot(a, b))));
    }
}

fn debug_line(a_1: vec3<f32>, b_1: vec3<f32>, color: u32) {
    var index: u32;

    let _e5 = debug_buf.open;
    if (_e5 != 0u) {
        let _e10 = atomicAdd((&debug_buf.instance_count), 1u);
        index = _e10;
        let _e12 = index;
        let _e14 = debug_buf.capacity;
        if (_e12 < _e14) {
            let _e17 = index;
            debug_buf.lines[_e17] = DebugLine(DebugPoint(a_1, color), DebugPoint(b_1, color));
        } else {
            let _e24 = atomicSub((&debug_buf.instance_count), 1u);
        }
    }
}

fn get_ray_direction_at(cp: CameraParams, film_pos: vec2<f32>) -> vec3<f32> {
    var half_size: vec2<f32>;
    var ndc_1: vec2<f32>;
    var local_dir_1: vec3<f32>;

    half_size = (vec2(0.5f) * vec2<f32>(cp.target_size));
    let _e9 = half_size;
    let _e11 = half_size;
    ndc_1 = ((film_pos - _e9) / _e11);
    let _e16 = ndc_1;
    local_dir_1 = vec3<f32>((cp.film_offset + ((VFLIP * _e16) * tan((vec2(0.5f) * cp.fov)))), -(1f));
    let _e30 = local_dir_1;
    let _e31 = qrot(cp.orientation, _e30);
    return normalize(_e31);
}

fn get_projected_pixel_float(cp_1: CameraParams, point_1: vec3<f32>) -> vec2<f32> {
    var local_dir_2: vec3<f32>;
    var slope: vec2<f32>;
    var ndc_2: vec2<f32>;
    var half_size_1: vec2<f32>;

    let _e4 = qinv(cp_1.orientation);
    let _e7 = qrot(_e4, (point_1 - cp_1.position));
    local_dir_2 = _e7;
    let _e10 = local_dir_2.z;
    if (_e10 >= 0f) {
        return vec2(-(1f));
    }
    let _e16 = local_dir_2;
    let _e19 = local_dir_2.z;
    slope = (_e16.xy / vec2(-(_e19)));
    let _e25 = slope;
    ndc_2 = ((VFLIP * (_e25 - cp_1.film_offset)) / tan((vec2(0.5f) * cp_1.fov)));
    half_size_1 = (vec2(0.5f) * vec2<f32>(cp_1.target_size));
    let _e42 = ndc_2;
    let _e46 = half_size_1;
    return ((_e42 + vec2(1f)) * _e46);
}

fn get_ray_direction(cp_2: CameraParams, pixel: vec2<i32>) -> vec3<f32> {
    let _e7 = get_ray_direction_at(cp_2, (vec2<f32>(pixel) + vec2(0.5f)));
    return _e7;
}

fn get_projected_pixel(cp_3: CameraParams, point_2: vec3<f32>) -> vec2<i32> {
    let _e3 = get_projected_pixel_float(cp_3, point_2);
    return vec2<i32>(_e3);
}

@vertex 
fn debug_vs(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> DebugVarying {
    var line: DebugLine;
    var point: DebugPoint;
    var world_dir: vec3<f32>;
    var local_dir: vec3<f32>;
    var ndc: vec2<f32>;
    var out: DebugVarying;

    let _e7 = debug_lines[instance_id];
    line = _e7;
    let _e10 = line.a;
    point = _e10;
    if (vertex_id != 0u) {
        let _e15 = line.b;
        point = _e15;
    }
    let _e17 = point.pos;
    let _e19 = camera.position;
    world_dir = (_e17 - _e19);
    let _e23 = camera.orientation;
    let _e24 = qinv(_e23);
    let _e25 = world_dir;
    let _e26 = qrot(_e24, _e25);
    local_dir = _e26;
    let _e28 = local_dir;
    let _e32 = camera.fov;
    ndc = (_e28.xy / tan((vec2(0.5f) * _e32)));
    out = DebugVarying();
    let _e41 = ndc;
    let _e45 = local_dir.z;
    out.pos = vec4<f32>(vec3<f32>(_e41, 0f), -(_e45));
    let _e50 = point.color;
    out.color = unpack4x8unorm(_e50);
    let _e53 = world_dir;
    out.dir = _e53;
    let _e54 = out;
    return _e54;
}

@fragment 
fn debug_fs(input: DebugVarying) -> @location(0) vec4<f32> {
    var geo_dim: vec2<u32>;
    var depth_itc: vec2<i32>;
    var stored: f32;
    var alpha: f32;

    let _e5 = textureDimensions(depth);
    geo_dim = _e5;
    let _e11 = geo_dim.y;
    depth_itc = vec2<i32>(i32(input.pos.x), (i32(_e11) - i32(input.pos.y)));
    let _e19 = depth_itc;
    let _e21 = textureLoad(depth, _e19, 0i);
    stored = _e21.x;
    let _e26 = stored;
    let _e32 = stored;
    let _e33 = stored;
    alpha = select(0.8f, 0.2f, ((_e26 != 0f) && (dot(input.dir, input.dir) > (_e32 * _e33))));
    let _e41 = alpha;
    return vec4<f32>(input.color.xyz, _e41);
}
