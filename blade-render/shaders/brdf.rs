use synaga_shader::*;

pub const PI: f32 = 3.1415926;

pub const DIELECTRIC_F0: f32 = 0.04;

pub const MIN_ROUGHNESS: f32 = 0.05;

pub const LUMINOCITY_WEIGHTS: vec3 = vec3(0.3, 0.4, 0.3);

#[derive(Clone, Copy, Default)]
pub struct Material {
    // Fraction of the light that gets diffused, i.e. the base color
    // with the specularly reflected part already taken out.
    pub diffuse_albedo: vec3,
    // Specular reflectance at normal incidence.
    pub specular_f0: vec3,
    pub roughness: f32,
}

#[derive(Clone, Copy, Default)]
pub struct BrdfLobes {
    pub diffuse: f32,
    pub specular: vec3,
}

#[shader]
pub fn compute_luminocity(color: vec3) -> f32 {
    return dot(color, LUMINOCITY_WEIGHTS);
}

#[shader]
pub fn material_from_metallic_roughness(
    base_color: vec3,
    metalness: f32,
    roughness: f32,
) -> Material {
    let mut mat = Material::default();
    mat.diffuse_albedo = base_color * (1.0 - metalness);
    mat.specular_f0 = mix(vec3::splat(DIELECTRIC_F0), base_color, metalness);
    mat.roughness = roughness;
    return mat;
}

#[shader]
pub fn material_alpha(mat: Material) -> f32 {
    let r = clamp(mat.roughness, MIN_ROUGHNESS, 1.0);
    return r * r;
}

#[shader]
pub fn fresnel_schlick(cos_theta: f32, f0: vec3) -> vec3 {
    return f0 + (vec3::splat(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

#[shader]
pub fn fresnel_schlick_scalar(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

#[shader]
pub fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-7);
}

#[shader]
pub fn visibility_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let lambda_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let lambda_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(lambda_v + lambda_l, 1e-7);
}

#[shader]
pub fn zero_brdf() -> BrdfLobes {
    return BrdfLobes {
        diffuse: 0.0,
        specular: vec3::splat(0.0),
    };
}

#[shader]
pub fn is_brdf_black(lobes: BrdfLobes) -> bool {
    return lobes.diffuse <= 0.0 && all(lobes.specular.cmple(vec3::splat(0.0)));
}

#[shader]
pub fn evaluate_ambient(mat: Material) -> vec3 {
    return mat.diffuse_albedo * (vec3::splat(1.0) - mat.specular_f0) + mat.specular_f0;
}

#[shader]
pub fn specular_sampling_ratio(mat: Material) -> f32 {
    let diffuse = compute_luminocity(mat.diffuse_albedo);
    let specular = compute_luminocity(mat.specular_f0);
    return clamp(specular / max(diffuse + specular, 1.0e-5), 0.1, 0.9);
}

#[shader]
pub fn evaluate_brdf(mat: Material, normal: vec3, view_dir: vec3, light_dir: vec3) -> BrdfLobes {
    let n_dot_l = dot(normal, light_dir);
    let n_dot_v = dot(normal, view_dir);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return zero_brdf();
    }

    let half_dir = normalize(view_dir + light_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 0.0);
    let alpha = material_alpha(mat);

    let fresnel = fresnel_schlick(v_dot_h, mat.specular_f0);
    let specular =
        distribution_ggx(n_dot_h, alpha) * visibility_smith(n_dot_v, n_dot_l, alpha) * fresnel;

    // Whatever isn't reflected by the specular lobe is available to the diffuse one.
    let k_diffuse = 1.0 - fresnel_schlick_scalar(v_dot_h, DIELECTRIC_F0);

    return BrdfLobes {
        diffuse: k_diffuse * n_dot_l / PI,
        specular: specular * n_dot_l,
    };
}
