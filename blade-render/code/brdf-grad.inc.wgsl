// Derivatives of the BSDF with respect to the material, for the
// differentiable path tracer.
//
// Requires "brdf.inc.wgsl". Mirrors `evaluate_brdf` term by term, so any
// change there has to be reflected here, and the finite difference test
// in `tests/gpu_examples.rs` keeps the two in agreement.

// The modulated BSDF value `diffuse_albedo * diffuse + specular`, with its
// partial derivatives. Each color channel of the value only depends on the
// same channel of the albedo and F0, so those derivatives are diagonal.
//
// The value is `diffuse_albedo * diffuse + fresnel * microfacet`, where only
// the scalars `diffuse` and `microfacet` depend on the normal, so the
// derivative by the normal is kept as theirs. The normal is taken as is,
// without its normalization.
struct BsdfGradient {
    value: vec3<f32>,
    d_diffuse_albedo: vec3<f32>,
    d_specular_f0: vec3<f32>,
    d_roughness: vec3<f32>,
    fresnel: vec3<f32>,
    d_normal_diffuse: vec3<f32>,
    d_normal_microfacet: vec3<f32>,
}

fn zero_bsdf_gradient() -> BsdfGradient {
    return BsdfGradient(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0),
        vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
}

// Derivative of the loss by the normal, given the one by the value.
fn bsdf_normal_grad(bsdf: BsdfGradient, mat: Material, d_value: vec3<f32>) -> vec3<f32> {
    return dot(d_value, mat.diffuse_albedo) * bsdf.d_normal_diffuse
        + dot(d_value, bsdf.fresnel) * bsdf.d_normal_microfacet;
}

fn evaluate_bsdf_gradient(mat: Material, normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> BsdfGradient {
    let n_dot_l = dot(normal, light_dir);
    let n_dot_v = dot(normal, view_dir);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return zero_bsdf_gradient();
    }

    let half_dir = normalize(view_dir + light_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 0.0);

    // `material_alpha`, which is flat outside of the clamped range.
    let r = clamp(mat.roughness, MIN_ROUGHNESS, 1.0);
    let alpha = r * r;
    let d_alpha = select(0.0, 2.0 * r, mat.roughness > MIN_ROUGHNESS && mat.roughness < 1.0);
    let a2 = alpha * alpha;

    // `distribution_ggx` and its derivative by `a2`.
    let n2 = n_dot_h * n_dot_h;
    let denom = n2 * (a2 - 1.0) + 1.0;
    let d_scale = PI * denom * denom;
    var distribution = a2 / 1e-7;
    var d_distribution = 1.0 / 1e-7;
    // By `n_dot_h`.
    var d_distribution_h = 0.0;
    if (d_scale > 1e-7) {
        distribution = a2 / d_scale;
        d_distribution = (1.0 - 2.0 * a2 * n2 / denom) / d_scale;
        d_distribution_h = -4.0 * a2 * n_dot_h * (a2 - 1.0) / (d_scale * denom);
    }

    // `visibility_smith` and its derivative by `a2`.
    let root_v = sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let root_l = sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    let lambda = n_dot_l * root_v + n_dot_v * root_l;
    var visibility = 0.5 / 1e-7;
    var d_visibility = 0.0;
    // By `n_dot_v` and `n_dot_l`.
    var d_visibility_v = 0.0;
    var d_visibility_l = 0.0;
    if (lambda > 1e-7) {
        visibility = 0.5 / lambda;
        let d_lambda = 0.5 * (n_dot_l * (1.0 - n_dot_v * n_dot_v) / root_v
            + n_dot_v * (1.0 - n_dot_l * n_dot_l) / root_l);
        d_visibility = -0.5 * d_lambda / (lambda * lambda);
        let d_lambda_v = n_dot_l * n_dot_v * (1.0 - a2) / root_v + root_l;
        let d_lambda_l = root_v + n_dot_v * n_dot_l * (1.0 - a2) / root_l;
        d_visibility_v = -0.5 * d_lambda_v / (lambda * lambda);
        d_visibility_l = -0.5 * d_lambda_l / (lambda * lambda);
    }

    let schlick = pow(1.0 - v_dot_h, 5.0);
    let fresnel = mat.specular_f0 + (vec3<f32>(1.0) - mat.specular_f0) * schlick;
    let k_diffuse = 1.0 - fresnel_schlick_scalar(v_dot_h, DIELECTRIC_F0);
    let diffuse = k_diffuse * n_dot_l / PI;
    let microfacet = distribution * visibility * n_dot_l;
    let d_microfacet = (d_distribution * visibility + distribution * d_visibility)
        * 2.0 * alpha * d_alpha * n_dot_l;

    var grad: BsdfGradient;
    grad.value = mat.diffuse_albedo * diffuse + microfacet * fresnel;
    grad.d_diffuse_albedo = vec3<f32>(diffuse);
    grad.d_specular_f0 = vec3<f32>(microfacet * (1.0 - schlick));
    grad.d_roughness = d_microfacet * fresnel;
    grad.fresnel = fresnel;
    grad.d_normal_diffuse = k_diffuse / PI * light_dir;
    // `max(n_dot_h, 0)` is flat below zero, where the BSDF is black anyway.
    let d_n_dot_h = select(0.0, d_distribution_h, dot(normal, half_dir) > 0.0);
    grad.d_normal_microfacet = d_n_dot_h * visibility * n_dot_l * half_dir
        + distribution * d_visibility_v * n_dot_l * view_dir
        + distribution * (d_visibility_l * n_dot_l + visibility) * light_dir;
    return grad;
}
