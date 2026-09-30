use super::brdf::{Material, distribution_ggx};
use super::random::{RandomState, random_gen};
use core::f32::consts::PI;
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
    mat3(
        vec3(1.0 + s * normal.x * normal.x * a, s * b, -s * normal.x),
        vec3(b, s + normal.y * normal.y * a, -normal.y),
        normal,
    )
}

pub fn sample_circle_uniform(random: f32) -> Vec2 {
    let angle = 2.0 * PI * random;
    vec2(angle.cos(), angle.sin())
}

pub fn compute_bsdf_pdf(mat: Material, normal: Vec3, view_dir: Vec3, light_dir: Vec3) -> f32 {
    let n_dot_l = normal.dot(light_dir);
    if n_dot_l <= 0.0 || normal.dot(view_dir) <= 0.0 {
        return 0.0;
    }
    let half_dir = (view_dir + light_dir).normalize();
    let n_dot_h = normal.dot(half_dir).max(0.0);
    let v_dot_h = view_dir.dot(half_dir).max(1.0e-5);
    let specular_pdf = distribution_ggx(n_dot_h, mat.alpha()) * n_dot_h / (4.0 * v_dot_h);
    let diffuse_pdf = n_dot_l / PI;
    mix(diffuse_pdf, specular_pdf, mat.specular_sampling_ratio())
}

fn sample_hemisphere_cosine(rng: &mut RandomState) -> Vec3 {
    let r = random_gen(rng);
    let tangential = r.sqrt() * sample_circle_uniform(random_gen(rng));
    tangential.extend((1.0 - r).max(0.0).sqrt())
}

fn sample_ggx_half_dir(alpha: f32, rng: &mut RandomState) -> Vec3 {
    let a2 = alpha * alpha;
    let r = random_gen(rng);
    let cos_theta = ((1.0 - r) / (1.0 + (a2 - 1.0) * r)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    (sin_theta * sample_circle_uniform(random_gen(rng))).extend(cos_theta)
}

pub fn sample_bsdf(
    mat: Material,
    normal: Vec3,
    view_dir: Vec3,
    rng: &mut RandomState,
) -> BsdfSample {
    let frame = make_tangent_frame(normal);
    let dir = if random_gen(rng) < mat.specular_sampling_ratio() {
        let half_dir = frame * sample_ggx_half_dir(mat.alpha(), rng);
        2.0 * view_dir.dot(half_dir) * half_dir - view_dir
    } else {
        frame * sample_hemisphere_cosine(rng)
    };
    let dir = dir.normalize();
    BsdfSample {
        dir,
        pdf: compute_bsdf_pdf(mat, normal, view_dir, dir),
    }
}
