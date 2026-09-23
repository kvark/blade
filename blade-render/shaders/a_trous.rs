use super::camera::*;
use super::config::*;
use super::gbuf::*;
use super::quaternion::*;
use super::surface::*;
use synaga_shader::*;

pub const LUMA: vec3 = vec3(0.2126, 0.7152, 0.0722);

pub const MIN_WEIGHT: f32 = 0.01;

pub const GAUSSIAN_WEIGHTS: vec2 = vec2(0.44198, 0.27901);

pub const SIGMA_L: f32 = 4.0;

pub const EPSILON: f32 = 0.001;

#[derive(Clone, Copy, Default)]
pub struct Params {
    pub extent: vec2i,
    pub temporal_weight: f32,
    pub iteration: u32,
    pub use_motion_vectors: u32,
}

pub static camera: Uniform<CameraParams> = binding();

pub static prev_camera: Uniform<CameraParams> = binding();

pub static params: Uniform<Params> = binding();

pub static t_depth: texture_2d<f32> = binding();

pub static t_prev_depth: texture_2d<f32> = binding();

pub static t_flat_normal: texture_2d<f32> = binding();

pub static t_prev_flat_normal: texture_2d<f32> = binding();

pub static t_motion: texture_2d<f32> = binding();

pub static input: texture_2d<f32> = binding();

pub static output: texture_storage_2d<Rgba16Float, ReadWrite> = binding();

#[shader]
pub fn read_surface(pixel: vec2i) -> Surface {
    let mut surface = Surface::default();
    surface.flat_normal = normalize(textureLoad(&t_flat_normal, pixel, 0).xyz());
    surface.depth = textureLoad(&t_depth, pixel, 0).x;
    return surface;
}

#[shader]
pub fn read_prev_surface(pixel: vec2i) -> Surface {
    let mut surface = Surface::default();
    surface.flat_normal = normalize(textureLoad(&t_prev_flat_normal, pixel, 0).xyz());
    surface.depth = textureLoad(&t_prev_depth, pixel, 0).x;
    return surface;
}

#[shader]
pub fn get_prev_pixel(pixel: vec2i, pos_world: vec3) -> vec2 {
    if (USE_MOTION_VECTORS && params.use_motion_vectors != 0u32) {
        let motion = textureLoad(&t_motion, pixel, 0).xy() / MOTION_SCALE;
        return vec2::from(pixel) + 0.5 + motion;
    } else {
        return get_projected_pixel_float(*prev_camera, pos_world);
    }
}

#[shader]
pub fn compare_luminance(a_lum: f32, b_lum: f32, variance: f32) -> f32 {
    return exp(-abs(a_lum - b_lum) / (SIGMA_L * variance + EPSILON));
}

#[shader]
pub fn w4(w: f32) -> vec4 {
    return (vec3::splat(w)).extend(w * w);
}

#[compute]
#[workgroup_size(8, 8)]
pub fn temporal_accum(#[builtin(global_invocation_id)] global_id: vec3u) {
    let pixel = vec2i::from(global_id.xy());
    if (any(pixel.cmpge(params.extent))) {
        return;
    }

    let surface = read_surface(pixel);
    let pos_world = camera.position + surface.depth * get_ray_direction(*camera, pixel);
    // considering all samples in 2x2 quad, to help with edges
    let mut center_pixel = get_prev_pixel(pixel, pos_world);
    let mut prev_pixels = [
        vec2i::from(vec2(center_pixel.x - 0.5, center_pixel.y - 0.5)),
        vec2i::from(vec2(center_pixel.x + 0.5, center_pixel.y - 0.5)),
        vec2i::from(vec2(center_pixel.x + 0.5, center_pixel.y + 0.5)),
        vec2i::from(vec2(center_pixel.x - 0.5, center_pixel.y + 0.5)),
    ];
    //Note: careful about the pixel center when there is a perfect match
    let w_bot_right = fract(center_pixel + vec2::splat(0.5));
    let mut prev_weights = vec4(
        (1.0 - w_bot_right.x) * (1.0 - w_bot_right.y),
        w_bot_right.x * (1.0 - w_bot_right.y),
        w_bot_right.x * w_bot_right.y,
        (1.0 - w_bot_right.x) * w_bot_right.y,
    );

    let mut sum_weight = 0.0;
    let mut sum_ilm = vec4::splat(0.0);
    if (params.temporal_weight != 1.0) {
        //TODO: optimize depth load with a gather operation
        for i in (0)..(4) {
            let prev_pixel = prev_pixels[(i) as usize];
            if (all(prev_pixel.cmpge(vec2i::splat(0))) && all(prev_pixel.cmplt(params.extent))) {
                let prev_surface = read_prev_surface(prev_pixel);
                if (compare_flat_normals(surface.flat_normal, prev_surface.flat_normal) < 0.5) {
                    continue;
                }
                let projected_distance = length(pos_world - prev_camera.position);
                if (compare_depths(prev_surface.depth, projected_distance) < 0.5) {
                    continue;
                }
                let w = prev_weights[(i) as usize];
                sum_weight += w;
                let illumination = w * textureLoad(&input, prev_pixel, 0).xyz();
                let luminocity = dot(illumination, LUMA);
                sum_ilm += (illumination).extend(luminocity * luminocity);
            }
        }
    }

    let cur_illumination = textureLoadStorage(&output, pixel).xyz();
    let cur_luminocity = dot(cur_illumination, LUMA);
    let mut mixed_ilm = (cur_illumination).extend(cur_luminocity * cur_luminocity);
    if (sum_weight > MIN_WEIGHT) {
        let prev_ilm =
            sum_ilm / (vec3::splat(sum_weight)).extend(max(0.001, sum_weight * sum_weight));
        mixed_ilm = mix(
            mixed_ilm,
            prev_ilm,
            sum_weight * (1.0 - params.temporal_weight),
        );
    }
    //Note: could also use HW blending for this
    textureStore(&output, pixel, mixed_ilm);
}

#[compute]
#[workgroup_size(8, 8)]
pub fn atrous_filter(#[builtin(global_invocation_id)] global_id: vec3u) {
    let center = vec2i::from(global_id.xy());
    if (any(center.cmpge(params.extent))) {
        return;
    }

    let center_ilm = textureLoad(&input, center, 0);
    let center_luma = dot(center_ilm.xyz(), LUMA);
    let center_suf = read_surface(center);
    let mut filtered_ilm = center_ilm;

    for yy in -1i32..=1i32 {
        for xx in -1i32..=1i32 {
            let p = center + vec2i(xx, yy) * (1i32 << params.iteration);
            if (all(p.cmpeq(center))
                || any(p.cmplt(vec2i::splat(0)))
                || any(p.cmpge(params.extent)))
            {
                continue;
            }

            //TODO: store in group-shared memory
            let surface = read_surface(p);
            let mut weight =
                GAUSSIAN_WEIGHTS[(abs(xx)) as usize] * GAUSSIAN_WEIGHTS[(abs(yy)) as usize];
            //TODO: make it stricter on higher iterations
            weight *= compare_flat_normals(surface.flat_normal, center_suf.flat_normal);
            //Note: should we use a projected depth instead of the surface one?
            weight *= compare_depths(surface.depth, center_suf.depth);
            let other_ilm = textureLoad(&input, p, 0);
            // The luminance gate must be symmetric. Using only the centre's
            // variance lets a noisy bright pixel accept a dark neighbour while
            // the dark pixel rejects the bright one, which systematically
            // moves radiance out of highlights and shadowed geometry.
            let variance = sqrt(max(center_ilm.w, other_ilm.w));
            weight *= compare_luminance(center_luma, dot(other_ilm.xyz(), LUMA), variance);

            // Rejected neighbour weight stays on the centre instead of
            // renormalising the surviving neighbours. Every pair then applies
            // equal and opposite RGB deltas, so an A-trous pass conserves
            // linear radiance over the frame. The Gaussian neighbour weights
            // sum to less than one, keeping this a convex update.
            filtered_ilm += w4(weight) * (other_ilm - center_ilm);
        }
    }

    textureStore(&output, global_id.xy(), filtered_ilm);
}
