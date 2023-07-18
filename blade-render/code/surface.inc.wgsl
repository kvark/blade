struct Surface {
    basis: vec4<f32>,
    flat_normal: vec3<f32>,
    depth: f32,
}

// Return the compatibility rating, where
// 1.0 means fully compatible, and
// 0.0 means totally incompatible.
fn compare_surfaces(a: Surface, b: Surface) -> f32 {
    let r_normal = smoothstep(0.4, 0.9, dot(a.flat_normal, b.flat_normal));
    let r_depth = 1.0 - smoothstep(0.0, 100.0, abs(a.depth - b.depth));
    return r_normal * r_depth;
}
