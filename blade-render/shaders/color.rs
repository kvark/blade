use synaga_shader::*;

pub fn encode_srgb(linear: Vec3) -> Vec3 {
    let low = 12.92 * linear;
    let high = 1.055 * linear.max(Vec3::splat(0.0)).powf(1.0 / 2.4) - 0.055;
    select(high, low, linear.cmple(Vec3::splat(0.0031308)))
}

pub fn encode_surface_color(color: Vec3, needs_encoding: bool) -> Vec3 {
    select(color, encode_srgb(color), needs_encoding)
}
