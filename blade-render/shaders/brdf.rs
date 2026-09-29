use core::f32::consts::PI;
use synaga_shader::*;

const DIELECTRIC_F0: f32 = 0.04;
const MIN_ROUGHNESS: f32 = 0.05;
const LUMINOCITY_WEIGHTS: Vec3 = vec3(0.3, 0.4, 0.3);

#[derive(Clone, Copy, Default)]
pub struct Material {
    // Fraction of the light that gets diffused, i.e. the base color
    // with the specularly reflected part already taken out.
    pub diffuse_albedo: Vec3,
    // Specular reflectance at normal incidence.
    pub specular_f0: Vec3,
    pub roughness: f32,
}

#[derive(Clone, Copy, Default)]
pub struct BrdfLobes {
    pub diffuse: f32,
    pub specular: Vec3,
}

pub fn compute_luminocity(color: Vec3) -> f32 {
    color.dot(LUMINOCITY_WEIGHTS)
}

pub fn material_from_metallic_roughness(
    base_color: Vec3,
    metalness: f32,
    roughness: f32,
) -> Material {
    Material {
        diffuse_albedo: base_color * (1.0 - metalness),
        specular_f0: mix(Vec3::splat(DIELECTRIC_F0), base_color, metalness),
        roughness,
    }
}

pub fn material_alpha(mat: Material) -> f32 {
    let r = mat.roughness.clamp(MIN_ROUGHNESS, 1.0);
    r * r
}

fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    f0 + (1.0 - f0) * (1.0 - cos_theta).powf(5.0)
}

fn fresnel_schlick_scalar(cos_theta: f32, f0: f32) -> f32 {
    f0 + (1.0 - f0) * (1.0 - cos_theta).powf(5.0)
}

pub fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom).max(1e-7)
}

fn visibility_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let lambda_v = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
    let lambda_l = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
    0.5 / (lambda_v + lambda_l).max(1e-7)
}

pub fn zero_brdf() -> BrdfLobes {
    BrdfLobes {
        diffuse: 0.0,
        specular: Vec3::ZERO,
    }
}

pub fn is_brdf_black(lobes: BrdfLobes) -> bool {
    lobes.diffuse <= 0.0 && lobes.specular.cmple(Vec3::ZERO).all()
}

pub fn evaluate_ambient(mat: Material) -> Vec3 {
    mat.diffuse_albedo * (1.0 - mat.specular_f0) + mat.specular_f0
}

pub fn specular_sampling_ratio(mat: Material) -> f32 {
    let diffuse = compute_luminocity(mat.diffuse_albedo);
    let specular = compute_luminocity(mat.specular_f0);
    (specular / (diffuse + specular).max(1.0e-5)).clamp(0.1, 0.9)
}

pub fn evaluate_brdf(mat: Material, normal: Vec3, view_dir: Vec3, light_dir: Vec3) -> BrdfLobes {
    let n_dot_l = normal.dot(light_dir);
    let n_dot_v = normal.dot(view_dir);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return zero_brdf();
    }

    let half_dir = (view_dir + light_dir).normalize();
    let n_dot_h = normal.dot(half_dir).max(0.0);
    let v_dot_h = view_dir.dot(half_dir).max(0.0);
    let alpha = material_alpha(mat);

    let fresnel = fresnel_schlick(v_dot_h, mat.specular_f0);
    let specular =
        distribution_ggx(n_dot_h, alpha) * visibility_smith(n_dot_v, n_dot_l, alpha) * fresnel;

    // Whatever isn't reflected by the specular lobe is available to the diffuse one.
    let k_diffuse = 1.0 - fresnel_schlick_scalar(v_dot_h, DIELECTRIC_F0);

    BrdfLobes {
        diffuse: k_diffuse * n_dot_l / PI,
        specular: specular * n_dot_l,
    }
}
