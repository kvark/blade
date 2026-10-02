use super::camera::{CameraParams, get_ray_direction};
use super::gbuf::{get_prev_pixel, prev_camera};
use super::surface::{Surface, compare_depths, compare_flat_normals};
use synaga_shader::*;

const LUMA: Vec3 = vec3(0.2126, 0.7152, 0.0722);
const MIN_WEIGHT: f32 = 0.01;
const GAUSSIAN_WEIGHTS: Vec2 = vec2(0.44198, 0.27901);
const SIGMA_L: f32 = 4.0;
const EPSILON: f32 = 0.001;

#[repr(C)]
#[derive(Shared)]
pub struct BlurParams {
    pub extent: Vec2<i32>,
    pub temporal_weight: f32,
    pub iteration: u32,
    pub use_motion_vectors: u32,
    pub _pad: u32,
}

static camera: Uniform<CameraParams> = binding();
static params: Uniform<BlurParams> = binding();
static t_depth: Texture2D<f32> = binding();
static t_prev_depth: Texture2D<f32> = binding();
static t_flat_normal: Texture2D<f32> = binding();
static t_prev_flat_normal: Texture2D<f32> = binding();
static input: Texture2D<f32> = binding();
static output: TextureStorage2D<Rgba16Float, ReadWrite> = binding();

fn read_surface(pixel: Vec2<i32>) -> Surface {
    Surface {
        flat_normal: t_flat_normal.load(pixel, 0).xyz().normalize(),
        depth: t_depth.load(pixel, 0).x,
        ..Default::default()
    }
}

fn read_prev_surface(pixel: Vec2<i32>) -> Surface {
    Surface {
        flat_normal: t_prev_flat_normal.load(pixel, 0).xyz().normalize(),
        depth: t_prev_depth.load(pixel, 0).x,
        ..Default::default()
    }
}

fn compare_luminance(a_lum: f32, b_lum: f32, variance: f32) -> f32 {
    (-(a_lum - b_lum).abs() / (SIGMA_L * variance + EPSILON)).exp()
}

fn w4(w: f32) -> Vec4 {
    Vec3::splat(w).extend(w * w)
}

#[entry_point(compute, threads(8, 8))]
fn temporal_accum(global_invocation_id: Vec3<u32>) {
    let pixel = global_invocation_id.xy().cast::<i32>();
    if pixel.cmpge(params.extent).any() {
        return;
    }

    let surface = read_surface(pixel);
    let pos_world = camera.position + surface.depth * get_ray_direction(*camera, pixel);
    // considering all samples in 2x2 quad, to help with edges
    let center_pixel = get_prev_pixel(pixel, pos_world, params.use_motion_vectors != 0);
    let prev_pixels = [
        vec2(center_pixel.x - 0.5, center_pixel.y - 0.5).cast::<i32>(),
        vec2(center_pixel.x + 0.5, center_pixel.y - 0.5).cast::<i32>(),
        vec2(center_pixel.x + 0.5, center_pixel.y + 0.5).cast::<i32>(),
        vec2(center_pixel.x - 0.5, center_pixel.y + 0.5).cast::<i32>(),
    ];
    //Note: careful about the pixel center when there is a perfect match
    let w_bot_right = fract(center_pixel + 0.5);
    let prev_weights = vec4(
        (1.0 - w_bot_right.x) * (1.0 - w_bot_right.y),
        w_bot_right.x * (1.0 - w_bot_right.y),
        w_bot_right.x * w_bot_right.y,
        (1.0 - w_bot_right.x) * w_bot_right.y,
    );

    let mut sum_weight = 0.0;
    let mut sum_ilm = Vec4::ZERO;
    if params.temporal_weight != 1.0 {
        //TODO: optimize depth load with a gather operation
        for i in 0..4 {
            let prev_pixel = prev_pixels[i as usize];
            if prev_pixel >= Vec2::ZERO && prev_pixel < params.extent {
                let prev_surface = read_prev_surface(prev_pixel);
                if compare_flat_normals(surface.flat_normal, prev_surface.flat_normal) < 0.5 {
                    continue;
                }
                let projected_distance = (pos_world - prev_camera.position).length();
                if compare_depths(prev_surface.depth, projected_distance) < 0.5 {
                    continue;
                }
                let w = prev_weights[i as usize];
                sum_weight += w;
                let illumination = w * input.load(prev_pixel, 0).xyz();
                let luminocity = illumination.dot(LUMA);
                sum_ilm += illumination.extend(luminocity * luminocity);
            }
        }
    }

    let cur_illumination = output.load(pixel).xyz();
    let cur_luminocity = cur_illumination.dot(LUMA);
    let mut mixed_ilm = cur_illumination.extend(cur_luminocity * cur_luminocity);
    if sum_weight > MIN_WEIGHT {
        let prev_ilm =
            sum_ilm / Vec3::splat(sum_weight).extend((sum_weight * sum_weight).max(0.001));
        mixed_ilm = mix(
            mixed_ilm,
            prev_ilm,
            sum_weight * (1.0 - params.temporal_weight),
        );
    }
    //Note: could also use HW blending for this
    output.store(pixel, mixed_ilm);
}

#[entry_point(compute, threads(8, 8))]
fn atrous_filter(global_invocation_id: Vec3<u32>) {
    let center = global_invocation_id.xy().cast::<i32>();
    if center.cmpge(params.extent).any() {
        return;
    }

    let center_ilm = input.load(center, 0);
    let center_luma = center_ilm.xyz().dot(LUMA);
    let center_suf = read_surface(center);
    let mut filtered_ilm = center_ilm;

    for yy in -1i32..=1 {
        for xx in -1i32..=1 {
            let p = center + vec2(xx, yy) * (1 << params.iteration);
            let inside = p >= Vec2::ZERO && p < params.extent;
            if p == center || !inside {
                continue;
            }

            //TODO: store in group-shared memory
            let surface = read_surface(p);
            let mut weight = GAUSSIAN_WEIGHTS[xx.unsigned_abs() as usize]
                * GAUSSIAN_WEIGHTS[yy.unsigned_abs() as usize];
            //TODO: make it stricter on higher iterations
            weight *= compare_flat_normals(surface.flat_normal, center_suf.flat_normal);
            //Note: should we use a projected depth instead of the surface one?
            weight *= compare_depths(surface.depth, center_suf.depth);
            let other_ilm = input.load(p, 0);
            // The luminance gate must be symmetric. Using only the centre's
            // variance lets a noisy bright pixel accept a dark neighbour while
            // the dark pixel rejects the bright one, which systematically
            // moves radiance out of highlights and shadowed geometry.
            // The variance channel is a second moment, but the convex update
            // below can drive it negative. sqrt of a negative is undefined and
            // lavapipe LLVM 22 returns NaN, which then blanks the frame.
            let variance = center_ilm.w.max(other_ilm.w).max(0.0).sqrt();
            weight *= compare_luminance(center_luma, other_ilm.xyz().dot(LUMA), variance);

            // Rejected neighbour weight stays on the centre instead of
            // renormalising the surviving neighbours. Every pair then applies
            // equal and opposite RGB deltas, so an A-trous pass conserves
            // linear radiance over the frame. The Gaussian neighbour weights
            // sum to less than one, keeping this a convex update.
            // 0 * NaN is NaN. A non-positive weight adds nothing and must
            // not be multiplied through, or one neighbour blanks later passes.
            if weight > 0.0 {
                filtered_ilm += w4(weight) * (other_ilm - center_ilm);
            }
        }
    }

    output.store(global_invocation_id.xy(), filtered_ilm);
}
