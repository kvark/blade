struct Surface {
    basis: vec4<f32>,
    flat_normal: vec3<f32>,
    depth: f32,
}

const SIGMA_N: f32 = 4.0;

fn compare_flat_normals(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return pow(max(0.0, dot(a, b)), SIGMA_N);
}

fn compare_depths(a: f32, b: f32) -> f32 {
    return 1.0 - smoothstep(0.0, 100.0, abs(a - b));
}

// Return the compatibility rating, where
// 1.0 means fully compatible, and
// 0.0 means totally incompatible.
fn compare_surfaces(a: Surface, b: Surface) -> f32 {
    let r_normal = compare_flat_normals(a.flat_normal, b.flat_normal);
    let r_depth = compare_depths(a.depth, b.depth);
    return r_normal * r_depth;
}
