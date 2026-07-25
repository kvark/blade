// Importance sampling of the material lobes.
//
// Requires "brdf.inc.wgsl" and "random.inc.wgsl".

struct BsdfSample {
    // Direction towards the light, unit length.
    dir: vec3<f32>,
    // Solid angle density of drawing this direction.
    pdf: f32,
}

// Orthonormal frame with Z pointing along the normal.
// Based on "Building an Orthonormal Basis, Revisited" by Duff et al.
fn make_tangent_frame(normal: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, normal.z >= 0.0);
    let a = -1.0 / (s + normal.z);
    let b = normal.x * normal.y * a;
    return mat3x3<f32>(
        vec3<f32>(1.0 + s * normal.x * normal.x * a, s * b, -s * normal.x),
        vec3<f32>(b, s + normal.y * normal.y * a, -normal.y),
        normal,
    );
}

fn sample_circle_uniform(random: f32) -> vec2<f32> {
    let angle = 2.0 * PI * random;
    return vec2<f32>(cos(angle), sin(angle));
}

// Cosine weighted direction in the upper hemisphere of the tangent frame.
fn sample_hemisphere_cosine(rng: ptr<function, RandomState>) -> vec3<f32> {
    let r = random_gen(rng);
    let tangential = sqrt(r) * sample_circle_uniform(random_gen(rng));
    return vec3<f32>(tangential, sqrt(max(0.0, 1.0 - r)));
}

// Half-way vector from the GGX distribution, in the tangent frame.
fn sample_ggx_half_dir(alpha: f32, rng: ptr<function, RandomState>) -> vec3<f32> {
    let a2 = alpha * alpha;
    let r = random_gen(rng);
    let cos_theta = sqrt((1.0 - r) / (1.0 + (a2 - 1.0) * r));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return vec3<f32>(sin_theta * sample_circle_uniform(random_gen(rng)), cos_theta);
}

// Solid angle density of `sample_bsdf` for a given direction.
fn compute_bsdf_pdf(mat: Material, normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let n_dot_l = dot(normal, light_dir);
    if (n_dot_l <= 0.0 || dot(normal, view_dir) <= 0.0) {
        return 0.0;
    }
    let half_dir = normalize(view_dir + light_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 1.0e-5);
    let specular_pdf = distribution_ggx(n_dot_h, material_alpha(mat)) * n_dot_h / (4.0 * v_dot_h);
    let diffuse_pdf = n_dot_l / PI;
    return mix(diffuse_pdf, specular_pdf, specular_sampling_ratio(mat));
}

// Draw a direction from one of the material lobes.
//
// The returned density is that of the mixture of both lobes, so a caller
// doesn't need to know which one the direction came from.
fn sample_bsdf(mat: Material, normal: vec3<f32>, view_dir: vec3<f32>, rng: ptr<function, RandomState>) -> BsdfSample {
    let frame = make_tangent_frame(normal);
    var dir: vec3<f32>;
    if (random_gen(rng) < specular_sampling_ratio(mat)) {
        let half_dir = frame * sample_ggx_half_dir(material_alpha(mat), rng);
        dir = 2.0 * dot(view_dir, half_dir) * half_dir - view_dir;
    } else {
        dir = frame * sample_hemisphere_cosine(rng);
    }
    dir = normalize(dir);
    return BsdfSample(dir, compute_bsdf_pdf(mat, normal, view_dir, dir));
}

// Total reflected light for a given direction, ready to be weighted by the
// density of the sample. Both of the lobes are modulated here.
fn evaluate_bsdf(mat: Material, normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> vec3<f32> {
    let brdf = evaluate_brdf(mat, normal, view_dir, light_dir);
    return mat.diffuse_albedo * brdf.diffuse + brdf.specular;
}
