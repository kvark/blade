//! Runs Blade's shader helpers on one input. The test calls `probe` itself on
//! the CPU, and `main` runs it on the GPU, from the module synaga makes of
//! this file and the helpers beside it.

use super::brdf::Material;
use super::camera::{CameraParams, get_ray_direction};
use super::color::encode_srgb;
use super::quaternion::{Quaternion, shortest_arc_quat};
use super::random::{random_gen, random_init};
use super::sampling::{compute_bsdf_pdf, sample_bsdf};
use super::skin_inc::{apply_affine, skin_linear};
use super::surface::compare_flat_normals;
use super::vertex::{decode_normal, tangent_basis};
use synaga_shader::*;

/// Three unit vectors in `xyz`, a number in `[0, 1)` in each `w`, and bits.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Input {
    pub a: Vec4,
    pub b: Vec4,
    pub c: Vec4,
    pub bits: Vec4<u32>,
}

/// What `probe` found.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Output {
    /// `random_init`'s seed, and the bits of the first three draws.
    pub random: Vec4<u32>,
    /// The specular F0 of a material, and how often it samples specular.
    pub material: Vec4,
    /// `Material::evaluate_brdf`: the specular lobe, then the diffuse one.
    pub brdf: Vec4,
    /// `sample_bsdf`'s direction and density, for a material at least 0.3
    /// rough. A sharper lobe draws half vectors so near its peak that an ulp
    /// in one moves the density by percents, which is the lobe and not the
    /// compiler: at 0.05, one ulp is four percent.
    pub sample: Vec4,
    /// `compute_bsdf_pdf` for the light, and `compare_flat_normals`.
    pub densities: Vec4,
    /// The rotation from `a` to `b`, and `c` rotated by it and back.
    pub arc: Vec4,
    pub rotated: Vec4,
    pub restored: Vec4,
    /// `encode_srgb` of a color.
    pub srgb: Vec4,
    /// `a`, through signed bytes and back.
    pub normal: Vec4,
    /// `tangent_basis`'s tangent and bitangent.
    pub tangent: Vec4,
    pub bitangent: Vec4,
    /// A point through an affine transform, and the transform's linear part.
    pub affine: Vec4,
    pub linear: Vec4,
    /// A camera ray through a pixel.
    pub ray: Vec4,
    /// Lanes cast from floats to integers.
    pub cast: Vec4<i32>,
    /// Control flow, and packing: see `lazy_calls` and `binding_is_a_copy`;
    /// then `a` packed as signed bytes.
    pub control: Vec4<u32>,
    /// Inclusive ranges: see `count_to_max`, `count_to_signed_max`,
    /// `count_empty` and `skip_the_last`.
    pub ranges: Vec4<u32>,
    /// `match`, vector comparisons and methods: see `first_arm`,
    /// `break_from_a_match`, `every_lane` and `count_up`.
    pub features: Vec4<u32>,
}

pub static inputs: Storage<[Input]> = binding();
pub static outputs: StorageMut<[Output]> = binding();

#[entry_point(compute, threads(64))]
fn main(global_invocation_id: Vec3<u32>) {
    let index = global_invocation_id.x as usize;
    if index < inputs.len() {
        outputs.get_mut()[index] = probe(inputs[index]);
    }
}

pub fn probe(input: Input) -> Output {
    let a = input.a.xyz();
    let b = input.b.xyz();
    let c = input.c.xyz();

    let mut rng = random_init(input.bits.x, input.bits.y);
    let seed = rng.seed;
    let first = random_gen(&mut rng).to_bits();
    let second = random_gen(&mut rng).to_bits();
    let third = random_gen(&mut rng).to_bits();

    // Seen from above the surface, as a BRDF is.
    let view = select(-b, b, b.dot(a) > 0.0);
    let light = select(-c, c, c.dot(a) > 0.0);
    let mat = Material::from_metallic_roughness(c.abs(), input.a.w, input.b.w);
    let lobes = mat.evaluate_brdf(a, view, light);
    let rough = Material::from_metallic_roughness(c.abs(), input.a.w, 0.3 + 0.7 * input.b.w);
    let sample = sample_bsdf(rough, a, view, &mut rng);
    let pdf = compute_bsdf_pdf(mat, a, view, light);

    let arc = shortest_arc_quat(a, b);
    let rotated = arc.rotate(c);
    let restored = arc.inv().rotate(rotated);

    let packed = pack4x8snorm(input.a);
    let basis = tangent_basis(a, b, 1.0, -1.0);
    let transform = mat3x4(input.a, input.b, input.c);
    let linear = skin_linear(transform);

    let camera = CameraParams {
        position: c,
        depth: 0.0,
        orientation: arc,
        fov: vec2(1.0, 0.75),
        film_offset: vec2(0.1, -0.05),
        target_size: vec2(640, 480),
        _pad: Vec2::ZERO,
    };
    let pixel = vec2((input.bits.z % 640) as i32, (input.bits.w % 480) as i32);

    Output {
        random: vec4(seed, first, second, third),
        material: mat.specular_f0.extend(mat.specular_sampling_ratio()),
        brdf: lobes.specular.extend(lobes.diffuse),
        sample: sample.dir.extend(sample.pdf),
        densities: vec4(pdf, compare_flat_normals(a, b), 0.0, 0.0),
        arc,
        rotated: rotated.extend(0.0),
        restored: restored.extend(0.0),
        srgb: encode_srgb(c.abs()).extend(1.0),
        normal: decode_normal(packed).extend(0.0),
        tangent: basis[0].extend(0.0),
        bitangent: basis[1].extend(0.0),
        affine: apply_affine(transform, c).extend(0.0),
        linear: linear[2].extend(0.0),
        ray: get_ray_direction(camera, pixel).extend(0.0),
        cast: (input.c * 1000.0).cast::<i32>(),
        control: vec4(
            lazy_calls((input.bits.x & 1) == 1),
            binding_is_a_copy(input.bits.z & 7),
            packed,
            0,
        ),
        ranges: vec4(
            count_to_max(input.bits.y & 3),
            count_to_signed_max((input.bits.y >> 2) & 3),
            count_empty(input.bits.w & 7),
            skip_the_last((input.bits.w >> 3) & 3),
        ),
        features: vec4(
            first_arm(input.bits.x % 5),
            break_from_a_match(input.bits.y & 15),
            every_lane(
                small_pair(input.bits.x, input.bits.y),
                small_pair(input.bits.z, input.bits.w),
            ),
            count_up(input.bits.z & 7),
        ),
    }
}

/// Two lanes in `-1..=2`, from the low bits of each.
pub fn small_pair(x: u32, y: u32) -> Vec2<i32> {
    vec2((x & 3) as i32 - 1, (y & 3) as i32 - 1)
}

/// `match` takes the first arm that matches, which a `switch` would not: 10
/// for 0 and 1, 20 for 2, 30 for 3, and a hundred times anything else.
// The second arm's `1` is the first's, which is the point.
#[allow(unreachable_patterns)]
fn first_arm(n: u32) -> u32 {
    match n {
        0 | 1 => 10,
        1 | 2 => 20,
        3 => 30,
        other => other * 100,
    }
}

/// The first multiple of 5 from `start` on. The `break` leaves the loop, as
/// in Rust, not the `match` it is in, as a `switch`'s would.
fn break_from_a_match(start: u32) -> u32 {
    let mut i = start;
    loop {
        match i % 5 {
            0 => break,
            _ => i += 1,
        }
    }
    i
}

/// 1 if `a < b` in every lane, 2 if `a >= b` in every lane, 4 if some lane
/// of `a` is not less, and 8 if they are equal.
fn every_lane(a: Vec2<i32>, b: Vec2<i32>) -> u32 {
    let mut found = 0u32;
    if a < b {
        found |= 1;
    }
    if a >= b {
        found |= 2;
    }
    if a.cmpge(b).any() {
        found |= 4;
    }
    if a == b {
        found |= 8;
    }
    found
}

#[derive(Clone, Copy, Default)]
struct Tally {
    total: u32,
    steps: u32,
}

impl Tally {
    fn add(&mut self, n: u32) {
        self.total += n;
        self.steps += 1;
    }

    fn mean(self) -> u32 {
        self.total / self.steps.max(1)
    }
}

/// A thousand times the mean of `0, 3, 6, ..` up to `n` of them, plus `n`:
/// a method that takes `&mut self` changes the local it is called on.
fn count_up(n: u32) -> u32 {
    let mut tally = Tally::default();
    for i in 0..n {
        tally.add(i * 3);
    }
    tally.mean() * 1000 + tally.steps
}

/// `bump` runs for `&&` only when `flag` holds, and for `||` only when it
/// does not, so this is 111 or 101: each operator's right side runs at most
/// once, and only when it decides.
fn lazy_calls(flag: bool) -> u32 {
    let mut calls = 0u32;
    if flag && bump(&mut calls) {
        calls += 10;
    }
    if flag || bump(&mut calls) {
        calls += 100;
    }
    calls
}

fn bump(calls: &mut u32) -> bool {
    *calls += 1;
    true
}

/// `below + 1`: how many values a range holds that ends at the largest
/// `u32`, which it cannot step past.
fn count_to_max(below: u32) -> u32 {
    let mut count = 0u32;
    for _ in (u32::MAX - below)..=u32::MAX {
        count += 1;
    }
    count
}

/// `below + 1`, for the largest `i32`.
fn count_to_signed_max(below: u32) -> u32 {
    let mut count = 0u32;
    for _ in (i32::MAX - below as i32)..=i32::MAX {
        count += 1;
    }
    count
}

/// Zero: a range whose end is before its start holds nothing.
fn count_empty(start: u32) -> u32 {
    let mut count = 0u32;
    for _ in (start + 1)..=start {
        count += 1;
    }
    count
}

/// `below`: the last iteration is skipped with `continue`, which still has
/// to end the loop.
fn skip_the_last(below: u32) -> u32 {
    let mut count = 0u32;
    for i in (u32::MAX - below)..=u32::MAX {
        if i == u32::MAX {
            continue;
        }
        count += 1;
    }
    count
}

/// Ten times the sum of `0..n`: assigning to the binding leaves the
/// iteration alone.
fn binding_is_a_copy(n: u32) -> u32 {
    let mut total = 0u32;
    for mut i in 0..n {
        i *= 10;
        total += i;
    }
    total
}
