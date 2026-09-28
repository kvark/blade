use super::quaternion::{qinv, qrot};
use synaga_shader::*;

const VFLIP: Vec2 = vec2(1.0, -1.0);

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CameraParams {
    pub position: Vec3,
    pub depth: f32,
    pub orientation: Vec4,
    pub fov: Vec2,
    pub film_offset: Vec2,
    pub target_size: Vec2<u32>,
    pub _pad: Vec2<u32>,
}

pub fn get_ray_direction_at(cp: CameraParams, film_pos: Vec2) -> Vec3 {
    let half_size = 0.5 * Vec2::from(cp.target_size);
    let ndc = (film_pos - half_size) / half_size;
    // Right-handed coordinate system with X=right, Y=up, and Z=towards the camera
    let local_dir = (cp.film_offset + VFLIP * ndc * (0.5 * cp.fov).tan()).extend(-1.0);
    qrot(cp.orientation, local_dir).normalize()
}

pub fn get_projected_pixel_float(cp: CameraParams, point: Vec3) -> Vec2 {
    let local_dir = qrot(qinv(cp.orientation), point - cp.position);
    if local_dir.z >= 0.0 {
        return Vec2::splat(-1.0);
    }
    let slope = local_dir.xy() / -local_dir.z;
    let ndc = VFLIP * (slope - cp.film_offset) / (0.5 * cp.fov).tan();
    let half_size = 0.5 * Vec2::from(cp.target_size);
    (ndc + Vec2::splat(1.0)) * half_size
}

pub fn get_ray_direction(cp: CameraParams, pixel: Vec2<i32>) -> Vec3 {
    get_ray_direction_at(cp, Vec2::from(pixel) + Vec2::splat(0.5))
}

pub fn get_projected_pixel(cp: CameraParams, point: Vec3) -> Vec2<i32> {
    get_projected_pixel_float(cp, point).cast::<i32>()
}
