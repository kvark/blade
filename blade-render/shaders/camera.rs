use super::quaternion::*;
use synaga_shader::*;

pub const VFLIP: vec2 = vec2(1.0, -1.0);

#[derive(Clone, Copy, Default)]
pub struct CameraParams {
    pub position: vec3,
    pub depth: f32,
    pub orientation: vec4,
    pub fov: vec2,
    pub film_offset: vec2,
    pub target_size: vec2u,
}

#[shader]
pub fn get_ray_direction_at(cp: CameraParams, film_pos: vec2) -> vec3 {
    let half_size = 0.5 * vec2::from(cp.target_size);
    let ndc = (film_pos - half_size) / half_size;
    // Right-handed coordinate system with X=right, Y=up, and Z=towards the camera
    let local_dir = (cp.film_offset + VFLIP * ndc * tan(0.5 * cp.fov)).extend(-1.0);
    return normalize(qrot(cp.orientation, local_dir));
}

#[shader]
pub fn get_projected_pixel_float(cp: CameraParams, point: vec3) -> vec2 {
    let local_dir = qrot(qinv(cp.orientation), point - cp.position);
    if local_dir.z >= 0.0 {
        return vec2::splat(-1.0);
    }
    let slope = local_dir.xy() / -local_dir.z;
    let ndc = VFLIP * (slope - cp.film_offset) / tan(0.5 * cp.fov);
    let half_size = 0.5 * vec2::from(cp.target_size);
    return (ndc + vec2::splat(1.0)) * half_size;
}

#[shader]
pub fn get_ray_direction(cp: CameraParams, pixel: vec2i) -> vec3 {
    return get_ray_direction_at(cp, vec2::from(pixel) + vec2::splat(0.5));
}

#[shader]
pub fn get_projected_pixel(cp: CameraParams, point: vec3) -> vec2i {
    return vec2i::from(get_projected_pixel_float(cp, point));
}
