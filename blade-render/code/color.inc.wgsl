// Encode a linear value with the sRGB transfer function.
//
// Only needed when the surface passes our values straight to the display
// instead of converting them, see `SurfaceInfo::color_space`.
fn encode_srgb(linear: vec3<f32>) -> vec3<f32> {
    let low = 12.92 * linear;
    let high = 1.055 * pow(max(linear, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, linear <= vec3<f32>(0.0031308));
}

fn encode_surface_color(color: vec3<f32>, needs_encoding: bool) -> vec3<f32> {
    return select(color, encode_srgb(color), needs_encoding);
}
