use super::brdf::{Material, evaluate_ambient, evaluate_brdf, material_from_metallic_roughness};
use super::color::encode_surface_color;
use super::config::MAX_LOCAL_LIGHTS;
use super::skin_inc::{SkinVertex, apply_affine, skin_blend, skin_linear};
use super::vertex::{Vertex, decode_normal};
use core::f32::consts::PI;
use synaga_shader::*;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct LocalLight {
    pub position_range: Vec4,
    pub intensity: Vec4,
    pub direction: Vec4,
    // x: inner cosine, y: outer cosine, z: falloff exponent, w: spot flag
    pub spot: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct LocalLightParams {
    // x: submitted light count, y: stochastic seed
    pub count_seed: Vec4,
    pub lights: [LocalLight; MAX_LOCAL_LIGHTS],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct RasterFrameParams {
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub light_view_proj: Mat4,
    pub camera_pos: Vec4,
    // direction towards the light
    pub light_dir: Vec4,
    pub light_color: Vec4,
    // w component is a flag for the procedural space sky
    pub ambient_color: Vec4,
    // x: environment map enabled, y: the surface needs sRGB encoding
    pub settings: Vec4,
    // x: enabled, y: strength, z: receiver normal bias, w: light-dir bias
    pub shadow_params: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct RasterDrawParams {
    pub model: Mat4,
    // Rotation of the object/geometry transform. Skinning assumes uniform
    // scale, so a quaternion is sufficient for normals.
    pub normal_quat: Vec4,
    pub base_color_factor: Vec4,
    pub emissive_factor: Vec4,
    // x: normal scale, y: metalness, z: roughness
    pub material: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct ShadowFrameParams {
    pub light_view_proj: Mat4,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct ShadowDrawParams {
    pub model: Mat4,
}

#[derive(Clone, Copy, Debug, Default, Io)]
struct VertexOutput {
    #[builtin(position)]
    clip_pos: Vec4,
    #[location(0)]
    world_pos: Vec3,
    #[location(1)]
    normal: Vec3,
    #[location(2)]
    tangent: Vec3,
    #[location(3)]
    bitangent: Vec3,
    #[location(4)]
    uv: Vec2,
}

#[derive(Clone, Copy, Debug, Default, Io)]
struct SkyOutput {
    #[builtin(position)]
    clip_pos: Vec4,
    #[location(0)]
    ndc: Vec2,
}

static frame_params: Uniform<RasterFrameParams> = binding();
static light_params: Uniform<LocalLightParams> = binding();
static draw_params: Uniform<RasterDrawParams> = binding();
static samp: Sampler = binding();
static base_color_tex: Texture2D<f32> = binding();
static normal_tex: Texture2D<f32> = binding();
static metallic_roughness_tex: Texture2D<f32> = binding();
static emissive_tex: Texture2D<f32> = binding();
static shadow_samp: SamplerComparison = binding();
static shadow_tex: TextureDepth2D = binding();
static shadow_frame_params: Uniform<ShadowFrameParams> = binding();
static shadow_draw_params: Uniform<ShadowDrawParams> = binding();
static sky_params: Uniform<RasterFrameParams> = binding();
static env_map: Texture2D<f32> = binding();

#[entry_point(vertex)]
fn raster_shadow_vs(input: Vertex) -> Vec4 {
    let world = shadow_draw_params.model * input.position.extend(1.0);
    shadow_frame_params.light_view_proj * world
}

#[entry_point(vertex)]
fn raster_shadow_skinned_vs(input: Vertex, skin_input: SkinVertex) -> Vec4 {
    let skinned = apply_affine(skin_blend(skin_input), input.position);
    let world = shadow_draw_params.model * skinned.extend(1.0);
    shadow_frame_params.light_view_proj * world
}

#[entry_point(fragment)]
fn raster_shadow_fs() {}

fn quat_rotate(q: Vec4, v: Vec3) -> Vec3 {
    v + 2.0 * q.xyz().cross(q.xyz().cross(v) + q.w * v)
}

fn map_equirect_dir_to_uv(dir: Vec3) -> Vec2 {
    let yaw = dir.x.atan2(dir.z);
    let pitch = dir.y.clamp(-1.0, 1.0).asin();
    vec2((yaw / PI + 1.0) * 0.5, pitch / PI + 0.5)
}

fn directional_shadow(world_pos: Vec3, n: Vec3) -> f32 {
    if frame_params.shadow_params.x < 0.5 {
        return 1.0;
    }
    let light_dir = frame_params.light_dir.xyz().normalize();
    let ndotl = n.dot(light_dir).max(0.0);
    // Mild slope scale: dense skinned panels need extra bias at grazing angles,
    // but a hard floor near 0.35 lifts ground receivers out of contact shadows
    // under a low dusk key.
    let normal_bias = frame_params.shadow_params.z * (1.0 + 0.75 * (1.0 - ndotl));
    let depth_bias = frame_params.shadow_params.w;
    let receiver = world_pos + n * normal_bias + light_dir * depth_bias;
    let clip = frame_params.light_view_proj * receiver.extend(1.0);
    let ndc = clip.xyz() / clip.w;
    let uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if ndc.z <= 0.0
        || ndc.z >= 1.0
        || uv.cmplt(Vec2::splat(0.0)).any()
        || uv.cmpgt(Vec2::splat(1.0)).any()
    {
        return 1.0;
    }

    // Four bilinear comparison samples give a compact 4x4 percentage-closer filter.
    let texel = 1.0 / shadow_tex.dimensions().x as f32;
    let reference = ndc.z;
    let mut visibility = 0.0;
    visibility +=
        shadow_tex.sample_compare(&shadow_samp, uv + vec2(-0.75, -0.75) * texel, reference);
    visibility +=
        shadow_tex.sample_compare(&shadow_samp, uv + vec2(0.75, -0.75) * texel, reference);
    visibility +=
        shadow_tex.sample_compare(&shadow_samp, uv + vec2(-0.75, 0.75) * texel, reference);
    visibility += shadow_tex.sample_compare(&shadow_samp, uv + vec2(0.75, 0.75) * texel, reference);
    visibility *= 0.25;
    mix(1.0, visibility, frame_params.shadow_params.y)
}

fn hash31(p: Vec3) -> f32 {
    fract(p.dot(vec3(127.1, 311.7, 74.7)).sin() * 43_758.547)
}

fn angular_attenuation(light: LocalLight, direction_to_light: Vec3) -> f32 {
    if light.spot.w < 0.5 {
        return 1.0;
    }
    let cosine = light.direction.xyz().dot(-direction_to_light);
    let width = light.spot.x - light.spot.y;
    if width <= 0.00001 {
        return select(0.0, 1.0, cosine >= light.spot.y);
    }
    let blend = ((cosine - light.spot.y) / width).clamp(0.0, 1.0);
    blend.powf(light.spot.z)
}

#[entry_point(vertex)]
fn raster_sky_vs(vertex_index: u32) -> SkyOutput {
    let positions = [vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0)];
    let pos = positions[vertex_index as usize];
    SkyOutput {
        clip_pos: pos.extend(1.0).extend(1.0),
        ndc: pos,
    }
}

fn raster_vertex(
    input: Vertex,
    position: Vec3,
    normal: Vec3,
    tangent: Vec3,
    bitangent_sign: f32,
) -> VertexOutput {
    let mut out = VertexOutput::default();
    let pos_world = draw_params.model * position.extend(1.0);
    out.clip_pos = frame_params.view_proj * pos_world;
    out.world_pos = pos_world.xyz();
    // GLES 3.00 requires matching uniform blocks in vs+fs. The multiply is
    // zero, so lighting does not leak into the vertex stage.
    out.world_pos.x += light_params.count_seed.x * 0.0;
    let n = quat_rotate(draw_params.normal_quat, normal).normalize();
    let t = quat_rotate(draw_params.normal_quat, tangent).normalize();
    let b = n.cross(t).normalize() * bitangent_sign;
    out.normal = n;
    out.tangent = t;
    out.bitangent = b;
    out.uv = input.tex_coords;
    out
}

#[entry_point(fragment)]
fn raster_sky_fs(input: SkyOutput) -> Vec4 {
    // Use z=0 (near plane) instead of z=1 (far plane) to avoid precision
    // issues: with far=1e9, inv_view_proj produces w≈1e-9 at z=1, causing
    // inf/NaN after perspective divide on mobile GPUs.
    let ndc = input.ndc.extend(0.0).extend(1.0);
    let world = sky_params.inv_view_proj * ndc;
    let world_pos = world.xyz() / world.w;
    let dir = (world_pos - sky_params.camera_pos.xyz()).normalize();
    let env_enabled = sky_params.settings.x > 0.5;
    let mut color = Vec3::splat(0.0);
    if env_enabled {
        let uv = map_equirect_dir_to_uv(dir);
        color = env_map.sample_level(&samp, uv, 0.0).xyz();
    } else {
        // Use ambient_color.w as a flag: values > 0.5 mean "space mode" (black sky)
        let space_mode = sky_params.ambient_color.w > 0.5;
        if space_mode {
            // Equal-area sky coordinates: (theta, dir.y) avoids polar bunching.
            let theta = dir.z.atan2(dir.x) + 10.0;
            let v = dir.y + 10.0;
            // Layer 1: bright stars (sparse, colored)
            {
                let uv = vec2(theta, v) * 50.0;
                let cell = uv.floor();
                let local = fract(uv) - Vec2::splat(0.5);
                let mut p3 = fract(vec3(cell.x, cell.y, cell.x) * vec3(0.1031, 0.1030, 0.0973));
                p3 = p3 + Vec3::splat(p3.dot(vec3(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33)));
                let h = fract((p3.x + p3.y) * p3.z);
                let h2 = fract((p3.y + p3.z) * p3.x);
                let h3 = fract((p3.z + p3.x) * p3.y);
                let star_pos = vec2(h - 0.5, h2 - 0.5) * 0.8;
                let d = (local - star_pos).length();
                let falloff = (1.0 - d / 0.08).clamp(0.0, 1.0);
                let b = falloff * falloff * 0.8 * step(0.92, h3);
                // Star color: cool blue, warm white, or reddish based on hash
                // Tints are saturated so they survive Reinhard tonemapping.
                let temp = h * 3.0;
                let mut tint = vec3(0.4, 0.55, 1.0); // blue
                if temp > 2.0 {
                    tint = vec3(1.0, 0.4, 0.15); // orange-red
                } else if temp > 1.0 {
                    tint = vec3(1.0, 0.9, 0.7); // warm yellow-white
                }
                color += tint * b;
            }
            // Layer 2: dim stars (dense, point-like)
            {
                let uv2 = vec2(theta, v) * 150.0;
                let cell2 = uv2.floor();
                let local2 = fract(uv2) - Vec2::splat(0.5);
                let mut q3 = fract(vec3(cell2.x, cell2.y, cell2.x) * vec3(0.1031, 0.1030, 0.0973));
                q3 = q3 + Vec3::splat(q3.dot(vec3(q3.y + 33.33, q3.z + 33.33, q3.x + 33.33)));
                let g = fract((q3.x + q3.y) * q3.z);
                let g2 = fract((q3.y + q3.z) * q3.x);
                let g3 = fract((q3.z + q3.x) * q3.y);
                let star_pos2 = vec2(g - 0.5, g2 - 0.5) * 0.8;
                let d2 = (local2 - star_pos2).length();
                let falloff2 = (1.0 - d2 / 0.06).clamp(0.0, 1.0);
                let b2 = falloff2 * falloff2 * 0.3 * step(0.94, g3);
                // Subtle color for dim stars too
                let tint2 = mix(vec3(0.5, 0.65, 1.0), vec3(1.0, 0.7, 0.5), g);
                color += tint2 * b2;
            }
        } else {
            let t = (dir.y * 0.5 + 0.5).clamp(0.0, 1.0);
            let horizon = vec3(0.6, 0.7, 0.9);
            let zenith = vec3(0.2, 0.35, 0.6);
            color = mix(horizon, zenith, t);
        }
    }
    let mapped = color / (color + Vec3::splat(1.0));
    encode_surface_color(mapped, sky_params.settings.y > 0.5).extend(1.0)
}

fn local_light_score(light: LocalLight, world_pos: Vec3, n: Vec3) -> f32 {
    let delta = light.position_range.xyz() - world_pos;
    let dist2 = delta.dot(delta).max(0.04);
    let dist = dist2.sqrt();
    let range = light.position_range.w.max(0.01);
    let falloff = (1.0 - dist / range).max(0.0);
    let ldir = delta / dist;
    let ndotl = n.dot(ldir).max(0.0);
    let intensity = light
        .intensity
        .x
        .max(light.intensity.y.max(light.intensity.z));
    intensity * angular_attenuation(light, ldir) * falloff * falloff * (0.2 + 0.8 * ndotl)
}

#[entry_point(vertex)]
fn raster_vs(input: Vertex) -> VertexOutput {
    raster_vertex(
        input,
        input.position,
        decode_normal(input.normal),
        decode_normal(input.tangent),
        input.bitangent_sign,
    )
}

#[entry_point(vertex)]
fn raster_skinned_vs(input: Vertex, skin_input: SkinVertex) -> VertexOutput {
    let skin = skin_blend(skin_input);
    let linear = skin_linear(skin);
    raster_vertex(
        input,
        apply_affine(skin, input.position),
        linear * decode_normal(input.normal),
        linear * decode_normal(input.tangent),
        input.bitangent_sign * sign(determinant(linear)),
    )
}

fn shade_local_light(mat: Material, n: Vec3, v: Vec3, world_pos: Vec3) -> Vec3 {
    let count = (light_params.count_seed.x as u32).min(MAX_LOCAL_LIGHTS as u32);
    if count == 0 {
        return Vec3::splat(0.0);
    }

    // Weighted reservoir over the submitted lights. Each fragment independently
    // samples one light with probability proportional to its local score.
    // TODO: spatial acceleration once scenes carry more local lights than this cap.
    let mut chosen = 0u32;
    let mut chosen_score = 0.0;
    let mut weight_sum = 0.0;
    for i in 0..MAX_LOCAL_LIGHTS as u32 {
        if i >= count {
            break;
        }
        let score = local_light_score(light_params.lights[i as usize], world_pos, n);
        if score <= 0.0 {
            continue;
        }
        weight_sum += score;
        let u = hash31(world_pos + vec3(i as f32, light_params.count_seed.y, score));
        if u * weight_sum < score {
            chosen = i;
            chosen_score = score;
        }
    }
    if weight_sum <= 0.0 {
        return Vec3::splat(0.0);
    }

    let light = light_params.lights[chosen as usize];
    let delta = light.position_range.xyz() - world_pos;
    let dist2 = delta.dot(delta).max(0.04);
    let dist = dist2.sqrt();
    let range = light.position_range.w.max(0.01);
    let falloff = (1.0 - dist / range).max(0.0);
    let ldir = delta / dist;
    let brdf = evaluate_brdf(mat, n, v, ldir);
    let atten = angular_attenuation(light, ldir) * falloff * falloff / dist2;
    // Divide by the reservoir selection probability so the result estimates
    // the sum of all local lights rather than their weighted average.
    let inverse_probability = weight_sum / chosen_score.max(0.000001);
    (mat.diffuse_albedo * brdf.diffuse + brdf.specular)
        * light.intensity.xyz()
        * atten
        * inverse_probability
}

#[entry_point(fragment)]
fn raster_fs(input: VertexOutput) -> Vec4 {
    let mr_sample = metallic_roughness_tex.sample(&samp, input.uv);
    let base_color =
        base_color_tex.sample(&samp, input.uv).rgb() * draw_params.base_color_factor.rgb();
    let mat = material_from_metallic_roughness(
        base_color,
        (draw_params.material.y * mr_sample.z).clamp(0.0, 1.0),
        (draw_params.material.z * mr_sample.y).clamp(0.0, 1.0),
    );

    let mut n = input.normal.normalize();
    let normal_scale = draw_params.material.x;
    if normal_scale > 0.0 {
        let raw_unorm = normal_tex.sample(&samp, input.uv).xy();
        let n_xy = normal_scale * (2.0 * raw_unorm - 1.0);
        let n_z = (1.0 - n_xy.dot(n_xy)).max(0.0).sqrt();
        let n_tangent = n_xy.extend(n_z).normalize();
        let tbn = mat3(input.tangent.normalize(), input.bitangent.normalize(), n);
        n = (tbn * n_tangent).normalize();
    }

    let v = (frame_params.camera_pos.xyz() - input.world_pos).normalize();
    let l = frame_params.light_dir.xyz().normalize();

    let brdf = evaluate_brdf(mat, n, v, l);
    let visibility = directional_shadow(input.world_pos, n);
    let light = (mat.diffuse_albedo * brdf.diffuse + brdf.specular)
        * frame_params.light_color.xyz()
        * visibility;
    let ambient = evaluate_ambient(mat) * frame_params.ambient_color.xyz();
    let emissive = draw_params.emissive_factor.rgb() * emissive_tex.sample(&samp, input.uv).rgb();
    let local = shade_local_light(mat, n, v, input.world_pos);
    let color = ambient + light + local + emissive;

    let mapped = color / (color + Vec3::splat(1.0));
    encode_surface_color(mapped, frame_params.settings.y > 0.5).extend(1.0)
}
