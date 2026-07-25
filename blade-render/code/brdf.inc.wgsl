// Physically based shading in the specular workflow: the diffuse and the
// specular responses of a surface are described independently, which is what
// the BRDF, the light sampling, and the demodulation all want to work with.
//
// Assets author materials in the glTF metallic-roughness form, see
// `material_from_metallic_roughness` for the conversion:
// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation

const PI: f32 = 3.1415926;
// Specular reflectance of a dielectric surface at normal incidence.
const DIELECTRIC_F0: f32 = 0.04;
// A perfectly smooth surface has a singular specular lobe, which
// neither our light sampling nor the denoiser can deal with.
const MIN_ROUGHNESS: f32 = 0.05;

const LUMINOCITY_WEIGHTS: vec3<f32> = vec3<f32>(0.3, 0.4, 0.3);

fn compute_luminocity(color: vec3<f32>) -> f32 {
    return dot(color, LUMINOCITY_WEIGHTS);
}

// Material properties of a shaded point, as stored in the G-buffer.
struct Material {
    // Fraction of the light that gets diffused, i.e. the base color
    // with the specularly reflected part already taken out.
    diffuse_albedo: vec3<f32>,
    // Specular reflectance at normal incidence.
    specular_f0: vec3<f32>,
    roughness: f32,
}

// Convert the glTF metallic-roughness parameters into the specular workflow.
//
// Note: this is the only place that knows about "metallic", everything
// downstream of the G-buffer works with the diffuse/specular split.
fn material_from_metallic_roughness(base_color: vec3<f32>, metallic: f32, roughness: f32) -> Material {
    var mat: Material;
    mat.diffuse_albedo = base_color * (1.0 - metallic);
    mat.specular_f0 = mix(vec3<f32>(DIELECTRIC_F0), base_color, metallic);
    mat.roughness = roughness;
    return mat;
}

// GGX width parameter.
fn material_alpha(mat: Material) -> f32 {
    let r = clamp(mat.roughness, MIN_ROUGHNESS, 1.0);
    return r * r;
}

// Probability of sampling the specular lobe instead of the diffuse one,
// proportional to how much light each of them is expected to reflect.
fn specular_sampling_ratio(mat: Material) -> f32 {
    let diffuse = compute_luminocity(mat.diffuse_albedo);
    let specular = compute_luminocity(mat.specular_f0);
    return clamp(specular / max(diffuse + specular, 1.0e-5), 0.1, 0.9);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}
fn fresnel_schlick_scalar(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// Trowbridge-Reitz (GGX) normal distribution.
fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-7);
}

// Height-correlated Smith visibility, which already includes
// the "1 / (4 * NdotV * NdotL)" of the microfacet specular term.
fn visibility_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let lambda_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let lambda_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(lambda_v + lambda_l, 1e-7);
}

// Reflected fraction of the light, split into the lobes that are
// estimated and denoised separately. Both of the terms include the
// cosine factor of the light direction.
//
// Note: `diffuse` is demodulated, it needs to be multiplied
// by the diffuse albedo of the material to get the reflected light.
struct BrdfLobes {
    diffuse: f32,
    specular: vec3<f32>,
}

fn zero_brdf() -> BrdfLobes {
    return BrdfLobes(0.0, vec3<f32>(0.0));
}

fn is_brdf_black(lobes: BrdfLobes) -> bool {
    return lobes.diffuse <= 0.0 && all(lobes.specular <= vec3<f32>(0.0));
}

// Evaluate the BRDF for the light arriving from `light_dir`.
// All the directions are unit length and point away from the surface.
fn evaluate_brdf(mat: Material, normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> BrdfLobes {
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
    let specular = distribution_ggx(n_dot_h, alpha) * visibility_smith(n_dot_v, n_dot_l, alpha) * fresnel;

    // Whatever isn't reflected by the specular lobe is available to the diffuse one.
    let k_diffuse = 1.0 - fresnel_schlick_scalar(v_dot_h, DIELECTRIC_F0);

    return BrdfLobes(k_diffuse * n_dot_l / PI, specular * n_dot_l);
}

// Crude approximation of the response to uniform ambient light:
// the diffuse albedo that survives the specular reflection, plus the mirror one.
fn evaluate_ambient(mat: Material) -> vec3<f32> {
    return mat.diffuse_albedo * (vec3<f32>(1.0) - mat.specular_f0) + mat.specular_f0;
}
