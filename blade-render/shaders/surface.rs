use synaga_shader::*;

const SIGMA_N: f32 = 4.0;

#[derive(Clone, Copy, Default)]
pub struct Surface {
    pub basis: Vec4,
    pub flat_normal: Vec3,
    pub depth: f32,
    // Direction towards the viewer, unit length.
    // Only filled in by the passes that do shading.
    pub view_dir: Vec3,
    // Material properties, only filled in by the passes that do shading.
    // Note: matching the fields of `Material` in `brdf.rs`, which
    // isn't available to all the users of this file.
    pub diffuse_albedo: Vec3,
    pub specular_f0: Vec3,
    pub roughness: f32,
}

pub fn compare_flat_normals(a: Vec3, b: Vec3) -> f32 {
    return pow(max(0.0, dot(a, b)), SIGMA_N);
}

pub fn compare_depths(a: f32, b: f32) -> f32 {
    return 1.0 - smoothstep(0.0, 100.0, abs(a - b));
}

pub fn compare_surfaces(a: Surface, b: Surface) -> f32 {
    let r_normal = compare_flat_normals(a.flat_normal, b.flat_normal);
    let r_depth = compare_depths(a.depth, b.depth);
    return r_normal * r_depth;
}
