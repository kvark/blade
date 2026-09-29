//! Runs Blade's shader helpers on one input. The test calls `probe` itself on
//! the CPU, and `main` runs it on the GPU, from the module synaga makes of
//! this file and the helpers beside it.

use super::brdf::{evaluate_brdf, material_from_metallic_roughness, specular_sampling_ratio};
use super::camera::{CameraParams, get_ray_direction};
use super::color::encode_srgb;
use super::quaternion::{qinv, qrot, shortest_arc_quat};
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
    /// `evaluate_brdf`: the specular lobe, then the diffuse one.
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
    /// Control flow, and packing: see `lazy_calls`, `count_to_max` and
    /// `binding_is_a_copy`; then `a` packed as signed bytes.
    pub control: Vec4<u32>,
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
    let mat = material_from_metallic_roughness(c.abs(), input.a.w, input.b.w);
    let lobes = evaluate_brdf(mat, a, view, light);
    let rough = material_from_metallic_roughness(c.abs(), input.a.w, 0.3 + 0.7 * input.b.w);
    let sample = sample_bsdf(rough, a, view, &mut rng);
    let pdf = compute_bsdf_pdf(mat, a, view, light);

    let arc = shortest_arc_quat(a, b);
    let rotated = qrot(arc, c);
    let restored = qrot(qinv(arc), rotated);

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
        material: mat.specular_f0.extend(specular_sampling_ratio(mat)),
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
            count_to_max(input.bits.y & 3),
            binding_is_a_copy(input.bits.z & 7),
            packed,
        ),
    }
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
