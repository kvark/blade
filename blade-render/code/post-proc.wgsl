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

struct DebugParams {
    view_mode: u32,
    draw_flags: u32,
    texture_flags: u32,
    pad: u32,
    mouse_pos: vec2<u32>,
}

struct PostProcParams {
    tone_map_enabled: u32,
    average_lum: f32,
    key_value: f32,
    white_level: f32,
    accumulated: u32,
    encode_srgb: u32,
    external_input: u32,
    _pad: u32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) input_size: vec2<u32>,
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

var<storage, read_write> debug_buf: DebugBuffer;
var t_diffuse_albedo: texture_2d<f32>;
var t_emissive: texture_2d<f32>;
var light_diffuse: texture_2d<f32>;
var light_specular: texture_2d<f32>;
var t_accumulation: texture_2d<f32>;
var t_debug: texture_2d<f32>;
var t_external: texture_2d<f32>;
var<uniform> post_proc_params: PostProcParams;
var<uniform> debug_params: DebugParams;

fn debug_line(a: vec3<f32>, b: vec3<f32>, color_1: u32) {
    var index: u32;

    let _e5 = debug_buf.open;
    if (_e5 != 0u) {
        let _e10 = atomicAdd((&debug_buf.instance_count), 1u);
        index = _e10;
        let _e12 = index;
        let _e14 = debug_buf.capacity;
        if (_e12 < _e14) {
            let _e17 = index;
            debug_buf.lines[_e17] = DebugLine(DebugPoint(a, color_1), DebugPoint(b, color_1));
        } else {
            let _e24 = atomicSub((&debug_buf.instance_count), 1u);
        }
    }
}

fn encode_srgb(linear: vec3<f32>) -> vec3<f32> {
    var low: vec3<f32>;
    var high: vec3<f32>;

    low = (vec3(12.92f) * linear);
    high = ((vec3(1.055f) * pow(max(linear, vec3(0f)), vec3((1f / 2.4f)))) - vec3(0.055f));
    let _e21 = high;
    let _e22 = low;
    return select(_e21, _e22, (linear <= vec3(0.0031308f)));
}

fn encode_surface_color(color_2: vec3<f32>, needs_encoding: bool) -> vec3<f32> {
    let _e3 = encode_srgb(color_2);
    return select(color_2, _e3, needs_encoding);
}

@vertex 
fn postfx_vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo: VertexOutput;

    vo = VertexOutput();
    vo.clip_pos = vec4<f32>(((f32((vi & 1u)) * 4f) - 1f), ((f32((vi & 2u)) * 2f) - 1f), 0f, 1f);
    let _e33 = textureDimensions(light_diffuse, 0i);
    vo.input_size = _e33;
    let _e34 = vo;
    return _e34;
}

@fragment 
fn postfx_fs(vo_1: VertexOutput) -> @location(0) vec4<f32> {
    var tc: vec2<i32>;
    var illumination: vec4<f32>;
    var color: vec3<f32>;
    var total: vec4<f32>;
    var diffuse_albedo: vec3<f32>;
    var specular: vec3<f32>;
    var emissive: vec3<f32>;
    var l_adjusted: vec3<f32>;
    var l_white: f32;
    var mapped: vec3<f32>;
    var encode: bool;

    tc = vec2<i32>(i32(vo_1.clip_pos.x), i32(vo_1.clip_pos.y));
    let _e19 = tc;
    let _e21 = textureLoad(light_diffuse, _e19, 0i);
    illumination = _e21;
    let _e24 = debug_params.view_mode;
    if (_e24 == DebugMode_Final) {
        color = vec3<f32>();
        let _e30 = post_proc_params.external_input;
        if (_e30 != 0u) {
            let _e33 = tc;
            let _e35 = textureLoad(t_external, _e33, 0i);
            color = _e35.xyz;
        } else {
            let _e38 = post_proc_params.accumulated;
            if (_e38 != 0u) {
                let _e41 = tc;
                let _e43 = textureLoad(t_accumulation, _e41, 0i);
                total = _e43;
                let _e45 = total;
                let _e48 = total.w;
                color = (_e45.xyz / vec3(max(_e48, 1f)));
            } else {
                let _e53 = tc;
                let _e55 = textureLoad(t_diffuse_albedo, _e53, 0i);
                diffuse_albedo = _e55.xyz;
                let _e58 = tc;
                let _e60 = textureLoad(light_specular, _e58, 0i);
                specular = _e60.xyz;
                let _e63 = tc;
                let _e65 = textureLoad(t_emissive, _e63, 0i);
                emissive = _e65.xyz;
                let _e68 = diffuse_albedo;
                let _e69 = illumination;
                let _e72 = specular;
                let _e74 = emissive;
                color = (((_e68 * _e69.xyz) + _e72) + _e74);
            }
        }
        let _e77 = post_proc_params.tone_map_enabled;
        if (_e77 == 0u) {
            let _e80 = color;
            return vec4<f32>(_e80, 1f);
        }
        let _e84 = post_proc_params.key_value;
        let _e86 = post_proc_params.average_lum;
        let _e88 = color;
        l_adjusted = (vec3((_e84 / _e86)) * _e88);
        let _e93 = post_proc_params.white_level;
        l_white = _e93;
        let _e95 = l_adjusted;
        let _e97 = l_adjusted;
        let _e98 = l_white;
        let _e99 = l_white;
        let _e107 = l_adjusted;
        mapped = ((_e95 * (vec3(1f) + (_e97 / vec3((_e98 * _e99))))) / (vec3(1f) + _e107));
        let _e113 = post_proc_params.encode_srgb;
        encode = (_e113 != 0u);
        let _e117 = mapped;
        let _e118 = encode;
        let _e119 = encode_surface_color(_e117, _e118);
        return vec4<f32>(_e119, 1f);
    } else {
        let _e123 = debug_params.view_mode;
        if (_e123 == DebugMode_Variance) {
            let _e127 = illumination.w;
            return vec4(_e127);
        } else {
            let _e129 = tc;
            let _e131 = textureLoad(t_debug, _e129, 0i);
            return _e131;
        }
    }
}
