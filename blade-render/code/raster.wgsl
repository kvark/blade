struct RasterFrameParams {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    ambient_color: vec4<f32>,
    material: vec4<f32>,
}

const PI: f32 = 3.1415926;

struct RasterDrawParams {
    model: mat4x4<f32>,
    normal_quat: vec4<f32>,
    base_color_factor: vec4<f32>,
    material: vec4<f32>,
}

struct Vertex {
    position: vec3<f32>,
    bitangent_sign: f32,
    tex_coords: vec2<f32>,
    normal: u32,
    tangent: u32,
}

struct VertexBuffer {
    data: array<Vertex>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) uv: vec2<f32>,
}

var<uniform> frame_params: RasterFrameParams;
var<uniform> draw_params: RasterDrawParams;
var<storage, read> vertices: VertexBuffer;
var samp: sampler;
var base_color_tex: texture_2d<f32>;
var normal_tex: texture_2d<f32>;

fn decode_normal(raw: u32) -> vec3<f32> {
    return unpack4x8snorm(raw).xyz;
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

@vertex
fn raster_vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let input = vertices.data[vertex_index];
    var out: VertexOutput;
    let pos_world = draw_params.model * vec4<f32>(input.position, 1.0);
    out.clip_pos = frame_params.view_proj * pos_world;
    out.world_pos = pos_world.xyz;
    let n = normalize(quat_rotate(draw_params.normal_quat, decode_normal(input.normal)));
    let t = normalize(quat_rotate(draw_params.normal_quat, decode_normal(input.tangent)));
    let b = normalize(cross(n, t)) * input.bitangent_sign;
    out.normal = n;
    out.tangent = t;
    out.bitangent = b;
    out.uv = input.tex_coords;
    return out;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

fn map_equirect_dir_to_uv(dir: vec3<f32>) -> vec2<f32> {
    let yaw = atan2(dir.x, dir.z);
    let pitch = asin(clamp(dir.y, -1.0, 1.0));
    return vec2<f32>((yaw / PI + 1.0) * 0.5, pitch / PI + 0.5);
}

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let ggx1 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx2 = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx1 * ggx2;
}

@fragment
fn raster_fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = textureSample(base_color_tex, samp, input.uv).rgb * draw_params.base_color_factor.rgb;

    var n = normalize(input.normal);
    let normal_scale = draw_params.material.x;
    if (normal_scale > 0.0) {
        let raw_unorm = textureSample(normal_tex, samp, input.uv).xy;
        let n_xy = normal_scale * (2.0 * raw_unorm - 1.0);
        let n_z = sqrt(max(0.0, 1.0 - dot(n_xy, n_xy)));
        let n_tangent = normalize(vec3<f32>(n_xy, n_z));
        let tbn = mat3x3<f32>(normalize(input.tangent), normalize(input.bitangent), n);
        n = normalize(tbn * n_tangent);
    }

    let v = normalize(frame_params.camera_pos.xyz - input.world_pos);
    let l = normalize(frame_params.light_dir.xyz);
    let h = normalize(v + l);

    let roughness = clamp(frame_params.material.x, 0.04, 1.0);
    let metallic = clamp(frame_params.material.y, 0.0, 1.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);
    let d = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let numerator = d * g * f;
    let denominator = max(4.0 * n_dot_v * n_dot_l, 0.001);
    let specular = numerator / denominator;

    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);
    let diffuse = k_d * albedo / PI;

    let light = (diffuse + specular) * frame_params.light_color.xyz * n_dot_l;
    let ambient = albedo * frame_params.ambient_color.xyz;
    let color = ambient + light;

    let mapped = color / (color + vec3<f32>(1.0));
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, 1.0);
}

struct SkyOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

var<uniform> sky_params: RasterFrameParams;
var env_map: texture_2d<f32>;

@vertex
fn raster_sky_vs(@builtin(vertex_index) vertex_id: u32) -> SkyOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let pos = positions[vertex_id];
    var out: SkyOutput;
    out.clip_pos = vec4<f32>(pos, 1.0, 1.0);
    out.ndc = pos;
    return out;
}

@fragment
fn raster_sky_fs(input: SkyOutput) -> @location(0) vec4<f32> {
    let ndc = vec4<f32>(input.ndc, 1.0, 1.0);
    let world = sky_params.inv_view_proj * ndc;
    let world_pos = world.xyz / world.w;
    let dir = normalize(world_pos - sky_params.camera_pos.xyz);
    let env_enabled = sky_params.material.z > 0.5;
    var color = vec3<f32>(0.0);
    if (env_enabled) {
        let uv = map_equirect_dir_to_uv(dir);
        color = textureSampleLevel(env_map, samp, uv, 0.0).xyz;
    } else {
        // Use ambient_color.w as a flag: values > 0.5 mean "space mode" (black sky)
        let space_mode = sky_params.ambient_color.w > 0.5;
        if (space_mode) {
            color = vec3<f32>(0.0);
        } else {
            let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
            let horizon = vec3<f32>(0.6, 0.7, 0.9);
            let zenith = vec3<f32>(0.2, 0.35, 0.6);
            color = mix(horizon, zenith, t);
        }
    }
    let mapped = color / (color + vec3<f32>(1.0));
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, 1.0);
}
