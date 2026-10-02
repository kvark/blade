//! The G-buffer the passes after `fill_gbuf` read, and the frame before it.

use super::camera::{CameraParams, get_projected_pixel_float};
use super::config::DEBUG_MODE;
use synaga_shader::*;

pub const MOTION_SCALE: f32 = 0.02;
pub const USE_MOTION_VECTORS: bool = true;
pub const WRITE_DEBUG_IMAGE: bool = DEBUG_MODE;

pub static prev_camera: Uniform<CameraParams> = binding();
pub static t_motion: Texture2D<f32> = binding();

/// Where `pos_world` was on screen a frame ago: moved by the motion vector
/// `fill_gbuf` wrote for `pixel`, or else projected with the camera then.
pub fn get_prev_pixel(pixel: Vec2<i32>, pos_world: Vec3, use_motion_vectors: bool) -> Vec2 {
    if USE_MOTION_VECTORS && use_motion_vectors {
        let motion = t_motion.load(pixel, 0).xy() / MOTION_SCALE;
        Vec2::from(pixel) + 0.5 + motion
    } else {
        get_projected_pixel_float(*prev_camera, pos_world)
    }
}
