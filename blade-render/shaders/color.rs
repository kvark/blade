use synaga_shader::*;

#[shader]
pub fn encode_srgb(linear: vec3) -> vec3 {
    let low = 12.92 * linear;
    let high = 1.055 * pow(max(linear, vec3::splat(0.0)), vec3::splat(1.0 / 2.4)) - 0.055;
    return select(high, low, linear.cmple(vec3::splat(0.0031308)));
}

#[shader]
pub fn encode_surface_color(color: vec3, needs_encoding: bool) -> vec3 {
    return select(color, encode_srgb(color), needs_encoding);
}
