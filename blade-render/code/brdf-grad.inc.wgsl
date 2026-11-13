// Derivatives of the BSDF with respect to the material, for the
// differentiable path tracer.
//
// Requires "brdf.inc.wgsl". Mirrors `evaluate_brdf` term by term, so any
// change there has to be reflected here, and the finite difference test
// in `tests/gpu_examples.rs` keeps the two in agreement.

// The modulated BSDF value `diffuse_albedo * diffuse + specular`, with its
// partial derivatives. Each color channel of the value only depends on the
// same channel of the albedo and F0, so those derivatives are diagonal.
struct BsdfGradient {
    value: vec3<f32>,
    d_diffuse_albedo: vec3<f32>,
    d_specular_f0: vec3<f32>,
    d_roughness: vec3<f32>,
}

fn zero_bsdf_gradient() -> BsdfGradient {
    return BsdfGradient(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
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
    if (d_scale > 1e-7) {
        distribution = a2 / d_scale;
        d_distribution = (1.0 - 2.0 * a2 * n2 / denom) / d_scale;
    }

    // `visibility_smith` and its derivative by `a2`.
    let root_v = sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let root_l = sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    let lambda = n_dot_l * root_v + n_dot_v * root_l;
    var visibility = 0.5 / 1e-7;
    var d_visibility = 0.0;
    if (lambda > 1e-7) {
        visibility = 0.5 / lambda;
        let d_lambda = 0.5 * (n_dot_l * (1.0 - n_dot_v * n_dot_v) / root_v
            + n_dot_v * (1.0 - n_dot_l * n_dot_l) / root_l);
        d_visibility = -0.5 * d_lambda / (lambda * lambda);
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
    return grad;
}
