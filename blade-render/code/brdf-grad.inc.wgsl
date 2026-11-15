// Derivatives of the BSDF with respect to the material, for the
// differentiable path tracer.
//
// Requires "brdf.inc.wgsl". Mirrors `evaluate_brdf` term by term, so any
// change there has to be reflected here, and the finite difference test
// in `tests/gpu_examples.rs` keeps the two in agreement.

// The modulated BSDF value, with its partial derivatives. Each color
// channel of the value only depends on the same channel of the albedo and
// F0, so those derivatives are diagonal.
//
// The derivative by the normal is the sum of `dot(d_value, normal_weights[k])
// * normal_dirs[k]`, see `bsdf_normal_grad`. The normal is taken as is,
// without its normalization.
struct BsdfGradient {
    value: vec3<f32>,
    d_diffuse_albedo: vec3<f32>,
    d_specular_f0: vec3<f32>,
    d_roughness: vec3<f32>,
    d_transmission: vec3<f32>,
    normal_weights: array<vec3<f32>, 2>,
    normal_dirs: array<vec3<f32>, 2>,
}

fn zero_bsdf_gradient() -> BsdfGradient {
    var grad: BsdfGradient;
    grad.value = vec3<f32>(0.0);
    grad.d_diffuse_albedo = vec3<f32>(0.0);
    grad.d_specular_f0 = vec3<f32>(0.0);
    grad.d_roughness = vec3<f32>(0.0);
    grad.d_transmission = vec3<f32>(0.0);
    grad.normal_weights = array<vec3<f32>, 2>(vec3<f32>(0.0), vec3<f32>(0.0));
    grad.normal_dirs = array<vec3<f32>, 2>(vec3<f32>(0.0), vec3<f32>(0.0));
    return grad;
}

// Derivative of the loss by the normal, given the one by the value.
fn bsdf_normal_grad(bsdf: BsdfGradient, d_value: vec3<f32>) -> vec3<f32> {
    return dot(d_value, bsdf.normal_weights[0]) * bsdf.normal_dirs[0]
        + dot(d_value, bsdf.normal_weights[1]) * bsdf.normal_dirs[1];
}

// `distribution_ggx * visibility_smith * n_dot_l`, with its derivatives.
struct MicrofacetGradient {
    value: f32,
    d_roughness: f32,
    d_n_dot_h: f32,
    d_n_dot_v: f32,
    d_n_dot_l: f32,
}

fn microfacet_gradient(roughness: f32, n_dot_h: f32, n_dot_v: f32, n_dot_l: f32) -> MicrofacetGradient {
    // `material_alpha`, which is flat outside of the clamped range.
    let r = clamp(roughness, MIN_ROUGHNESS, 1.0);
    let alpha = r * r;
    let d_alpha = select(0.0, 2.0 * r, roughness > MIN_ROUGHNESS && roughness < 1.0);
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

    var mf: MicrofacetGradient;
    mf.value = distribution * visibility * n_dot_l;
    mf.d_roughness = (d_distribution * visibility + distribution * d_visibility)
        * 2.0 * alpha * d_alpha * n_dot_l;
    mf.d_n_dot_h = d_distribution_h * visibility * n_dot_l;
    mf.d_n_dot_v = distribution * d_visibility_v * n_dot_l;
    mf.d_n_dot_l = distribution * (d_visibility_l * n_dot_l + visibility);
    return mf;
}

fn evaluate_bsdf_gradient(mat: Material, normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> BsdfGradient {
    return evaluate_surface_gradient(mat, 0.0, normal, view_dir, light_dir);
}

// Like `evaluate_bsdf_gradient`, with a part `transmission` of the diffuse
// albedo going through the surface instead, as a thin wall: the specular
// lobe mirrored to the other side, tinted by the albedo, of the light that
// isn't reflected, as in `KHR_materials_transmission`.
fn evaluate_surface_gradient(mat: Material, transmission: f32, normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> BsdfGradient {
    let n_dot_l = dot(normal, light_dir);
    let n_dot_v = dot(normal, view_dir);
    if (n_dot_v <= 0.0 || n_dot_l == 0.0 || (n_dot_l < 0.0 && transmission <= 0.0)) {
        return zero_bsdf_gradient();
    }
    if (n_dot_l < 0.0) {
        return evaluate_transmission_gradient(mat, transmission, normal, view_dir, light_dir);
    }

    let half_dir = normalize(view_dir + light_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 0.0);
    let mf = microfacet_gradient(mat.roughness, n_dot_h, n_dot_v, n_dot_l);

    let schlick = pow(1.0 - v_dot_h, 5.0);
    let fresnel = mat.specular_f0 + (vec3<f32>(1.0) - mat.specular_f0) * schlick;
    let k_diffuse = 1.0 - fresnel_schlick_scalar(v_dot_h, DIELECTRIC_F0);
    let diffuse = k_diffuse * n_dot_l / PI;
    let kept = 1.0 - transmission;

    var grad: BsdfGradient;
    grad.value = mat.diffuse_albedo * kept * diffuse + mf.value * fresnel;
    grad.d_diffuse_albedo = vec3<f32>(kept * diffuse);
    grad.d_specular_f0 = vec3<f32>(mf.value * (1.0 - schlick));
    grad.d_roughness = mf.d_roughness * fresnel;
    grad.d_transmission = -mat.diffuse_albedo * diffuse;
    grad.normal_weights = array<vec3<f32>, 2>(mat.diffuse_albedo * kept, fresnel);
    // `max(n_dot_h, 0)` is flat below zero, where the BSDF is black anyway.
    let d_n_dot_h = select(0.0, mf.d_n_dot_h, dot(normal, half_dir) > 0.0);
    grad.normal_dirs = array<vec3<f32>, 2>(
        k_diffuse / PI * light_dir,
        d_n_dot_h * half_dir + mf.d_n_dot_v * view_dir + mf.d_n_dot_l * light_dir,
    );
    return grad;
}

// The light through the surface, from below it: the specular lobe at the
// light direction mirrored about the surface, `light_dir - 2 n_dot_l
// normal`, which depends on the normal as well.
fn evaluate_transmission_gradient(mat: Material, transmission: f32, normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> BsdfGradient {
    let n_dot_l = dot(normal, light_dir);
    let n_dot_v = dot(normal, view_dir);
    let mirrored = light_dir - 2.0 * n_dot_l * normal;
    let sum = view_dir + mirrored;
    let sum_length = length(sum);
    if (sum_length < 1.0e-6) {
        return zero_bsdf_gradient();
    }
    let half_dir = sum / sum_length;
    let n_dot_h_raw = dot(normal, half_dir);
    let n_dot_h = max(n_dot_h_raw, 0.0);
    let v_dot_h_raw = dot(view_dir, half_dir);
    let v_dot_h = max(v_dot_h_raw, 0.0);
    let n_dot_m = dot(normal, mirrored);
    let mf = microfacet_gradient(mat.roughness, n_dot_h, n_dot_v, n_dot_m);

    let schlick = pow(1.0 - v_dot_h, 5.0);
    let fresnel = mat.specular_f0 + (vec3<f32>(1.0) - mat.specular_f0) * schlick;
    // By `v_dot_h`.
    let d_fresnel = (vec3<f32>(1.0) - mat.specular_f0) * (-5.0 * pow(1.0 - v_dot_h, 4.0));
    let passed = vec3<f32>(1.0) - fresnel;

    var grad: BsdfGradient;
    grad.value = transmission * mat.diffuse_albedo * passed * mf.value;
    grad.d_diffuse_albedo = transmission * passed * mf.value;
    grad.d_specular_f0 = -transmission * mat.diffuse_albedo * (1.0 - schlick) * mf.value;
    grad.d_roughness = transmission * mat.diffuse_albedo * passed * mf.d_roughness;
    grad.d_transmission = mat.diffuse_albedo * passed * mf.value;

    // How the dot products move with the normal, through the mirrored
    // direction: `d mirrored = -2 (dot(light_dir, dn) normal + n_dot_l dn)`,
    // and the half direction moves by its part across itself.
    let across = 1.0 / sum_length;
    let pn = normal - half_dir * n_dot_h_raw;
    let pv = view_dir - half_dir * v_dot_h_raw;
    let d_n_dot_h = half_dir - 2.0 * across * (dot(normal, pn) * light_dir + n_dot_l * pn);
    let d_v_dot_h = -2.0 * across * (dot(normal, pv) * light_dir + n_dot_l * pv);
    let d_n_dot_m = light_dir - 2.0 * dot(normal, normal) * light_dir - 4.0 * n_dot_l * normal;
    let h_part = select(0.0, mf.d_n_dot_h, n_dot_h_raw > 0.0);
    let v_part = select(vec3<f32>(0.0), d_fresnel, v_dot_h_raw > 0.0);
    grad.normal_weights = array<vec3<f32>, 2>(
        transmission * mat.diffuse_albedo * passed,
        -transmission * mat.diffuse_albedo * v_part * mf.value,
    );
    grad.normal_dirs = array<vec3<f32>, 2>(
        h_part * d_n_dot_h + mf.d_n_dot_v * view_dir + mf.d_n_dot_l * d_n_dot_m,
        d_v_dot_h,
    );
    return grad;
}

// The density of the lobes of `evaluate_surface_gradient`, drawn by
// `sample_surface` in the differentiable tracer.
fn specular_pdf(mat: Material, normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let half_dir = normalize(view_dir + light_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 1.0e-5);
    return distribution_ggx(n_dot_h, material_alpha(mat)) * n_dot_h / (4.0 * v_dot_h);
}

fn surface_pdf(mat: Material, transmission: f32, normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let n_dot_l = dot(normal, light_dir);
    if (dot(normal, view_dir) <= 0.0 || n_dot_l == 0.0) {
        return 0.0;
    }
    let ratio = specular_sampling_ratio(mat);
    if (n_dot_l > 0.0) {
        return ratio * specular_pdf(mat, normal, view_dir, light_dir)
            + (1.0 - ratio) * (1.0 - transmission) * n_dot_l / PI;
    }
    let mirrored = light_dir - 2.0 * n_dot_l * normal;
    return (1.0 - ratio) * transmission * specular_pdf(mat, normal, view_dir, mirrored);
}
