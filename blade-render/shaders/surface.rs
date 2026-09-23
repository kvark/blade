use synaga_shader::*;

pub const SIGMA_N: f32 = 4.0;

#[derive(Clone, Copy, Default)]
pub struct Surface {
    pub basis: vec4,
    pub flat_normal: vec3,
    pub depth: f32,
    // Direction towards the viewer, unit length.
    // Only filled in by the passes that do shading.
    pub view_dir: vec3,
    // Material properties, only filled in by the passes that do shading.
    // Note: matching the fields of `Material` in `brdf.rs`, which
    // isn't available to all the users of this file.
    pub diffuse_albedo: vec3,
    pub specular_f0: vec3,
    pub roughness: f32,
}

#[shader]
pub fn compare_flat_normals(a: vec3, b: vec3) -> f32 {
    return pow(max(0.0, dot(a, b)), SIGMA_N);
}

#[shader]
pub fn compare_depths(a: f32, b: f32) -> f32 {
    return 1.0 - smoothstep(0.0, 100.0, abs(a - b));
}

#[shader]
pub fn compare_surfaces(a: Surface, b: Surface) -> f32 {
    let r_normal = compare_flat_normals(a.flat_normal, b.flat_normal);
    let r_depth = compare_depths(a.depth, b.depth);
    return r_normal * r_depth;
}
