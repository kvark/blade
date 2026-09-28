use super::brdf::{
    Material, PI, distribution_ggx, evaluate_brdf, material_alpha, specular_sampling_ratio,
};
use super::random::{RandomState, random_gen};
use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct BsdfSample {
    // Direction towards the light, unit length.
    pub dir: Vec3,
    // Solid angle density of drawing this direction.
    pub pdf: f32,
}

fn make_tangent_frame(normal: Vec3) -> Mat3 {
    let s = select(-1.0, 1.0, normal.z >= 0.0);
    let a = -1.0 / (s + normal.z);
    let b = normal.x * normal.y * a;
    return mat3(
        vec3(1.0 + s * normal.x * normal.x * a, s * b, -s * normal.x),
        vec3(b, s + normal.y * normal.y * a, -normal.y),
        normal,
    );
}

pub fn sample_circle_uniform(random: f32) -> Vec2 {
    let angle = 2.0 * PI * random;
    return vec2(cos(angle), sin(angle));
}

pub fn compute_bsdf_pdf(mat: Material, normal: Vec3, view_dir: Vec3, light_dir: Vec3) -> f32 {
    let n_dot_l = dot(normal, light_dir);
    if n_dot_l <= 0.0 || dot(normal, view_dir) <= 0.0 {
        return 0.0;
    }
    let half_dir = normalize(view_dir + light_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 1.0e-5);
    let specular_pdf = distribution_ggx(n_dot_h, material_alpha(mat)) * n_dot_h / (4.0 * v_dot_h);
    let diffuse_pdf = n_dot_l / PI;
    return mix(diffuse_pdf, specular_pdf, specular_sampling_ratio(mat));
}

fn evaluate_bsdf(mat: Material, normal: Vec3, view_dir: Vec3, light_dir: Vec3) -> Vec3 {
    let brdf = evaluate_brdf(mat, normal, view_dir, light_dir);
    return mat.diffuse_albedo * brdf.diffuse + brdf.specular;
}

fn sample_hemisphere_cosine(rng: &mut RandomState) -> Vec3 {
    let r = random_gen(rng);
    let tangential = sqrt(r) * sample_circle_uniform(random_gen(rng));
    return (tangential).extend(sqrt(max(0.0, 1.0 - r)));
}

fn sample_ggx_half_dir(alpha: f32, rng: &mut RandomState) -> Vec3 {
    let a2 = alpha * alpha;
    let r = random_gen(rng);
    let cos_theta = sqrt((1.0 - r) / (1.0 + (a2 - 1.0) * r));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return (sin_theta * sample_circle_uniform(random_gen(rng))).extend(cos_theta);
}

pub fn sample_bsdf(
    mat: Material,
    normal: Vec3,
    view_dir: Vec3,
    rng: &mut RandomState,
) -> BsdfSample {
    let frame = make_tangent_frame(normal);
    let mut dir = Vec3::default();
    if random_gen(rng) < specular_sampling_ratio(mat) {
        let half_dir = frame * sample_ggx_half_dir(material_alpha(mat), rng);
        dir = 2.0 * dot(view_dir, half_dir) * half_dir - view_dir;
    } else {
        dir = frame * sample_hemisphere_cosine(rng);
    }
    dir = normalize(dir);
    return BsdfSample {
        dir: dir,
        pdf: compute_bsdf_pdf(mat, normal, view_dir, dir),
    };
}
