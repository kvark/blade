#![allow(irrefutable_let_patterns)]

//! Blade's shader helpers, run twice: on the CPU, as the Rust they are, and on
//! the GPU, as the module synaga makes of the same files. The two have to
//! agree, integers exactly and floats to within what the GPU's precision
//! allows.
//!
//! synaga's own tests check what each construct lowers to. That cannot catch
//! a mistake the lowering and its test share; running both can. The probe
//! also runs what synaga once got wrong: `&&` and `||` that evaluated their
//! right side regardless, a range to `u32::MAX` that never ended, and a
//! `for mut` binding that was the loop's counter. And what a `switch` would
//! get wrong about a `match`: which arm takes a value two arms name, and
//! where a `break` in an arm goes.

use std::mem::offset_of;
use std::path::PathBuf;
use std::sync::OnceLock;

use blade_graphics as gpu;
use synaga_shader::*;

#[path = "shader_helpers/shaders.rs"]
mod shaders;

use shaders::probe::{Input, Output, probe};

const COUNT: usize = 256;

/// A fixed stream of numbers, so a failure comes back the same.
struct Xorshift(u32);

impl Xorshift {
    fn bits(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// In `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.bits() >> 8) as f32 / 16_777_216.0
    }

    fn direction(&mut self) -> Vec3 {
        loop {
            let v = vec3(self.unit(), self.unit(), self.unit()) * 2.0 - 1.0;
            let length = v.length();
            if (0.1..=1.0).contains(&length) {
                return v / length;
            }
        }
    }
}

fn inputs() -> Vec<Input> {
    let mut rng = Xorshift(0x2545_f491);
    (0..COUNT)
        .map(|_| Input {
            a: rng.direction().extend(rng.unit()),
            b: rng.direction().extend(rng.unit()),
            c: rng.direction().extend(rng.unit()),
            bits: vec4(rng.bits(), rng.bits(), rng.bits(), rng.bits()),
        })
        .collect()
}

#[test]
fn the_helpers_run_on_the_cpu() {
    for input in inputs() {
        let out = probe(input);
        // Worked out from what each probe says it computes, not from the
        // shader that computes it.
        let flag = input.bits.x & 1 == 1;
        assert_eq!(out.control.x, if flag { 111 } else { 101 });
        assert_eq!(out.control.y, 10 * (0..input.bits.z & 7).sum::<u32>());
        assert_eq!(out.ranges.x, (input.bits.y & 3) + 1);
        assert_eq!(out.ranges.y, ((input.bits.y >> 2) & 3) + 1);
        assert_eq!(out.ranges.z, 0);
        assert_eq!(out.ranges.w, (input.bits.w >> 3) & 3);
        let n = input.bits.x % 5;
        let first = [10, 10, 20, 30][..].get(n as usize).copied();
        assert_eq!(out.features.x, first.unwrap_or(n * 100));
        assert_eq!(out.features.y, (input.bits.y & 15).next_multiple_of(5));
        let a = shaders::probe::small_pair(input.bits.x, input.bits.y);
        let b = shaders::probe::small_pair(input.bits.z, input.bits.w);
        let lanes = [(a.x, b.x), (a.y, b.y)];
        let every = |f: fn(i32, i32) -> bool| lanes.iter().all(|&(a, b)| f(a, b));
        let expected = u32::from(every(|a, b| a < b))
            | u32::from(every(|a, b| a >= b)) << 1
            | u32::from(!every(|a, b| a < b)) << 2
            | u32::from(every(|a, b| a == b)) << 3;
        assert_eq!(out.features.z, expected);
        let steps = input.bits.z & 7;
        let mean = (0..steps).map(|i| i * 3).sum::<u32>() / steps.max(1);
        assert_eq!(out.features.w, mean * 1000 + steps);
        let restored = out.restored.xyz() - input.c.xyz();
        assert!(restored.length() < 1e-5, "{restored:?}");
        for draw in [out.random.y, out.random.z, out.random.w] {
            assert!((0.0..1.0).contains(&f32::from_bits(draw)));
        }
    }
}

/// The probe and the helpers, compiled the way Blade's build script compiles
/// its shaders, and decoded into this crate's Naga.
fn probe_module() -> naga::Module {
    static MODULE: OnceLock<naga::Module> = OnceLock::new();
    MODULE
        .get_or_init(|| {
            let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("shader_helpers");
            let sources = dir.join("shaders");
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&sources).expect("create the shader directory");
            for (name, text) in shaders::HELPERS {
                std::fs::write(sources.join(format!("{name}.rs")), text).expect("write a helper");
            }
            std::fs::write(sources.join("probe.rs"), shaders::PROBE).expect("write the probe");
            // As `rustc` sees the helpers here.
            let cfg = match cfg!(debug_assertions) {
                true => synaga::Cfg::new().with("debug_assertions"),
                false => synaga::Cfg::new(),
            };
            let built = synaga::build::Shaders::new()
                .dir(&sources)
                .bindings(synaga::build::Bindings::Host)
                .cfg(cfg)
                // The module is compiled while the test runs, so the generated
                // layout file is written too late for anything to include it.
                // `the_probe_structs_are_laid_out_the_same_on_the_host_and_the_gpu`
                // makes the same comparison at run time, against the module
                // rather than synaga's model of it, which is the stronger of the
                // two. Said here so the build does not warn about a check that
                // is made by other means.
                .layout_checks_included()
                .emit_to(&dir)
                .unwrap_or_else(|err| panic!("{err}"));
            let [probe] = &built[..] else {
                panic!("one shader, the probe: {built:?}");
            };
            let bytes = std::fs::read(&probe.output_path).expect("read the probe");
            synaga_shader::ir::decode(&bytes).expect("decode the probe")
        })
        .clone()
}

#[test]
fn the_helpers_compile_for_the_gpu() {
    let module = probe_module();
    assert!(module.entry_points.iter().any(|ep| ep.name == "main"));
}

/// `rustc`'s layout of the probe's structs, against the layout the module it
/// was compiled to has them read with.
///
/// Blade's own shaders get this from `check_layout!`, which asserts at compile
/// time what the build script writes. This module is built while the test
/// runs, so there was nothing for that to include, and the warning the build
/// script raised about it was true. The same comparison is made here instead,
/// against the module rather than against synaga's model of `rustc`, which is
/// the stronger of the two: this reads the offsets the GPU will read, not what
/// produced them.
///
/// The size and every field's offset are what is compared. The struct's own
/// alignment is not, because it is not what the GPU uses here: `inputs` and
/// `outputs` are bound as whole buffers, so where the buffer starts is the
/// host's decision and no field moves. Naga's own layouter would say `Input`
/// is aligned 16, since WGSL wants a `vec4<f32>` on 16, and `Bindings::Host`
/// is the opt-out from exactly that rule which lets a `#[repr(C)]` struct be
/// read as written. Comparing against it would report the opt-out as a bug.
///
/// Every field is named, since `offset_of!` takes the name as a literal and
/// there is no way to ask `rustc` for the offsets of a struct it was not given
/// by name.
macro_rules! assert_laid_out {
    ($ty:ty { $($field:ident),* $(,)? }) => {{
        let module = probe_module();
        let name = stringify!($ty);
        let (_, ty) = module
            .types
            .iter()
            .find(|&(_, ty)| ty.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("the probe has no struct named '{name}'"));
        let naga::TypeInner::Struct { span, ref members } = ty.inner else {
            panic!("'{name}' is not a struct in the probe");
        };
        assert_eq!(
            size_of::<$ty>(),
            span as usize,
            "'{name}' is {span} bytes on the GPU and {} on the host",
            size_of::<$ty>(),
        );
        let found: Vec<_> = members.iter().map(|m| m.name.as_deref().unwrap()).collect();
        let offsets: Vec<usize> = members.iter().map(|m| m.offset as usize).collect();
        assert_eq!(found, [$(stringify!($field)),*], "the fields of '{name}' differ");
        assert_eq!(
            offsets,
            [$(offset_of!($ty, $field)),*],
            "the fields of '{name}' are at other offsets on the GPU",
        );
    }};
}

#[test]
fn the_probe_structs_are_laid_out_the_same_on_the_host_and_the_gpu() {
    assert_laid_out!(Input { a, b, c, bits });
    assert_laid_out!(Output {
        random,
        material,
        brdf,
        sample,
        densities,
        arc,
        rotated,
        restored,
        srgb,
        normal,
        tangent,
        bitangent,
        affine,
        linear,
        ray,
        cast,
        control,
        ranges,
        features,
    });
}

#[derive(blade_macros::ShaderData)]
struct ProbeData {
    inputs: gpu::BufferPiece,
    outputs: gpu::BufferPiece,
}

fn run_on_gpu(inputs: &[Input]) -> Vec<Output> {
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()) }.expect("GPU context");
    let input_bytes: &[u8] = bytemuck::cast_slice(inputs);
    let output_size = (inputs.len() * size_of::<Output>()) as u64;
    let input_buffer = context.create_buffer(gpu::BufferDesc {
        name: "probe inputs",
        size: input_bytes.len() as u64,
        memory: gpu::Memory::Shared,
    });
    let output_buffer = context.create_buffer(gpu::BufferDesc {
        name: "probe outputs",
        size: output_size,
        memory: gpu::Memory::Shared,
    });
    unsafe {
        std::ptr::copy_nonoverlapping(input_bytes.as_ptr(), input_buffer.data(), input_bytes.len());
    }
    context.sync_buffer(
        input_buffer.into(),
        input_bytes.len() as u64,
        gpu::BufferTarget::Data,
    );

    let shader = context.create_shader(gpu::ShaderDesc {
        source: "",
        naga_module: Some(probe_module()),
    });
    let layout = <ProbeData as gpu::ShaderData>::layout();
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "probe",
        data_layouts: &[&layout],
        compute: shader.at("main"),
    });
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "probe",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    if let mut compute = encoder.compute("probe")
        && let mut pass = compute.with(&pipeline)
    {
        pass.bind(
            0,
            &ProbeData {
                inputs: input_buffer.into(),
                outputs: output_buffer.into(),
            },
        );
        pass.dispatch([(inputs.len() as u32).div_ceil(64), 1, 1]);
    }
    let sync_point = context.submit(&mut encoder);
    assert!(context.wait_for(&sync_point, 10_000).unwrap());

    let written = unsafe {
        std::slice::from_raw_parts(output_buffer.data() as *const u8, output_size as usize)
    };
    let outputs = bytemuck::cast_slice::<u8, Output>(written).to_vec();

    context.destroy_command_encoder(&mut encoder);
    context.destroy_compute_pipeline(&mut pipeline);
    context.destroy_buffer(output_buffer);
    context.destroy_buffer(input_buffer);
    outputs
}

/// How far a float the GPU computes may be from the CPU's: some relative
/// error for the functions WGSL allows several ULPs, such as `pow`, which the
/// BRDF raises to the fifth power, and an absolute floor for values near
/// zero.
fn close(cpu: f32, gpu: f32) -> bool {
    (cpu - gpu).abs() <= 1e-4 + 1e-3 * cpu.abs()
}

#[test]
#[ignore = "requires a working GPU context"]
fn the_helpers_agree_on_the_cpu_and_the_gpu() {
    let inputs = inputs();
    let gpu_outputs = run_on_gpu(&inputs);
    let mut differences = Vec::new();
    for (index, (&input, gpu)) in inputs.iter().zip(&gpu_outputs).enumerate() {
        let cpu = probe(input);
        let exact = [
            ("random", cpu.random, gpu.random),
            ("control", cpu.control, gpu.control),
            ("ranges", cpu.ranges, gpu.ranges),
            ("features", cpu.features, gpu.features),
        ];
        for (name, cpu, gpu) in exact {
            if cpu != gpu {
                differences.push(format!(
                    "{index}: {name} is {cpu:?} on the CPU, {gpu:?} on the GPU"
                ));
            }
        }
        if cpu.cast != gpu.cast {
            differences.push(format!(
                "{index}: cast is {:?} on the CPU, {:?} on the GPU",
                cpu.cast, gpu.cast
            ));
        }
        let floats = [
            ("material", cpu.material, gpu.material),
            ("brdf", cpu.brdf, gpu.brdf),
            ("sample", cpu.sample, gpu.sample),
            ("densities", cpu.densities, gpu.densities),
            ("arc", cpu.arc, gpu.arc),
            ("rotated", cpu.rotated, gpu.rotated),
            ("restored", cpu.restored, gpu.restored),
            ("srgb", cpu.srgb, gpu.srgb),
            ("normal", cpu.normal, gpu.normal),
            ("tangent", cpu.tangent, gpu.tangent),
            ("bitangent", cpu.bitangent, gpu.bitangent),
            ("affine", cpu.affine, gpu.affine),
            ("linear", cpu.linear, gpu.linear),
            ("ray", cpu.ray, gpu.ray),
        ];
        for (name, cpu, gpu) in floats {
            let lanes: [f32; 4] = cpu.into();
            let gpu_lanes: [f32; 4] = gpu.into();
            if !lanes.iter().zip(gpu_lanes).all(|(&c, g)| close(c, g)) {
                differences.push(format!(
                    "{index}: {name} is {cpu:?} on the CPU, {gpu:?} on the GPU"
                ));
            }
        }
    }
    assert!(
        differences.is_empty(),
        "{} of {} values differ:\n{}",
        differences.len(),
        inputs.len(),
        differences[..differences.len().min(20)].join("\n")
    );
}
