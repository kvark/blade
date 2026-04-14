//! Fast matrix multiplication using cooperative matrix operations.
//!
//! Computes C = A * B for 64x64 matrices using cooperative
//! matrix tiles (tensor cores / simdgroup matrix), then verifies
//! against a CPU reference.
//!
//! Adapts to the device's supported tile size (8 or 16) and scalar
//! type (f32 or f16 inputs with f32 accumulator).
//!
//! Requires VK_KHR_cooperative_matrix (Vulkan) or Apple7+ (Metal).

use blade_graphics as gpu;
use gpu::ShaderData as _;
use std::mem;

const M: u32 = 64;
const N: u32 = 64;
const K: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Params {
    m: u32,
    n: u32,
    k: u32,
}

#[derive(blade_macros::ShaderData)]
struct MatmulData {
    matrix_a: gpu::BufferPiece,
    matrix_b: gpu::BufferPiece,
    matrix_c: gpu::BufferPiece,
    params: gpu::BufferPiece,
}

fn main() {
    env_logger::init();

    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            validation: cfg!(debug_assertions),
            ..Default::default()
        })
        .expect("Failed to init GPU context")
    };

    let caps = context.capabilities();
    let cm = caps.cooperative_matrix;
    // Prefer f32 inputs, fall back to f16 inputs with f32 accumulator.
    let (tile, f16_input) = if cm.f32_tile > 0 {
        (cm.f32_tile, false)
    } else if cm.f16_tile > 0 {
        (cm.f16_tile, true)
    } else {
        eprintln!(
            "Cooperative matrix not supported on this device ({}).",
            context.device_information().device_name
        );
        eprintln!("Requires VK_KHR_cooperative_matrix (Vulkan) or Apple7+ (Metal).");
        return;
    };
    let input_type = if f16_input { "f16" } else { "f32" };
    println!(
        "Device: {} (cooperative matrix {tile}x{tile}, {input_type} input)",
        context.device_information().device_name
    );

    // Specialize shader source for the device's capabilities
    let coop_type = format!("coop_mat{tile}x{tile}");
    let source_template = include_str!("matmul.wgsl");
    let source = source_template
        .replace("ENABLE_F16", if f16_input { "enable f16;" } else { "" })
        .replace("COOP_MAT", &coop_type)
        .replace("INPUT_SCALAR", input_type)
        .replace("TILE_SIZE", &format!("{tile}u"));

    // Create shader and pipeline
    let shader = context.create_shader(gpu::ShaderDesc {
        source: &source,
        naga_module: None,
    });
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "matmul",
        data_layouts: &[&MatmulData::layout()],
        compute: shader.at("main"),
    });

    // Prepare input matrices (always compute in f32, convert to f16 if needed)
    let a_f32: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_f32: Vec<f32> = (0..K * N).map(|i| (i % 11) as f32 * 0.1).collect();
    let c_data: Vec<f32> = vec![0.0; (M * N) as usize];
    let params = Params { m: M, n: N, k: K };

    let (a_bytes, b_bytes): (Vec<u8>, Vec<u8>) = if f16_input {
        let a_f16: Vec<half::f16> = a_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let b_f16: Vec<half::f16> = b_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        (
            bytemuck::cast_slice(&a_f16).to_vec(),
            bytemuck::cast_slice(&b_f16).to_vec(),
        )
    } else {
        (
            bytemuck::cast_slice(&a_f32).to_vec(),
            bytemuck::cast_slice(&b_f32).to_vec(),
        )
    };

    let buf_a = context.create_buffer(gpu::BufferDesc {
        name: "matrix_a",
        size: a_bytes.len() as u64,
        memory: gpu::Memory::Shared,
    });
    let buf_b = context.create_buffer(gpu::BufferDesc {
        name: "matrix_b",
        size: b_bytes.len() as u64,
        memory: gpu::Memory::Shared,
    });
    let buf_c = context.create_buffer(gpu::BufferDesc {
        name: "matrix_c",
        size: (c_data.len() * mem::size_of::<f32>()) as u64,
        memory: gpu::Memory::Shared,
    });
    let buf_params = context.create_buffer(gpu::BufferDesc {
        name: "params",
        size: mem::size_of::<Params>() as u64,
        memory: gpu::Memory::Shared,
    });

    // Upload data
    unsafe {
        std::ptr::copy_nonoverlapping(a_bytes.as_ptr(), buf_a.data(), a_bytes.len());
        std::ptr::copy_nonoverlapping(b_bytes.as_ptr(), buf_b.data(), b_bytes.len());
        std::ptr::copy_nonoverlapping(c_data.as_ptr() as *const u8, buf_c.data(), c_data.len() * 4);
        std::ptr::copy_nonoverlapping(
            &params as *const Params as *const u8,
            buf_params.data(),
            mem::size_of::<Params>(),
        );
    }

    // Dispatch
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "matmul",
        buffer_count: 1,
        queue: gpu::QueueType::Main,
    });
    encoder.start();
    {
        let mut pass = encoder.compute("matmul");
        let mut pe = pass.with(&pipeline);
        pe.bind(
            0,
            &MatmulData {
                matrix_a: buf_a.into(),
                matrix_b: buf_b.into(),
                matrix_c: buf_c.into(),
                params: buf_params.into(),
            },
        );
        pe.dispatch([M / tile, N / tile, 1]);
    }
    let sp = context.submit(&mut encoder, &[]);
    let _ = context.wait_for(&sp, !0);

    // Read back results
    let result =
        unsafe { std::slice::from_raw_parts(buf_c.data() as *const f32, (M * N) as usize) };

    // CPU reference
    let mut reference = vec![0.0f32; (M * N) as usize];
    for i in 0..M {
        for j in 0..N {
            let mut sum = 0.0f32;
            for ki in 0..K {
                sum += a_f32[(i * K + ki) as usize] * b_f32[(ki * N + j) as usize];
            }
            reference[(i * N + j) as usize] = sum;
        }
    }

    // Verify (f16 inputs lose precision, so use a wider tolerance)
    let tolerance = if f16_input { 0.5 } else { 0.01 };
    let mut max_error = 0.0f32;
    for i in 0..(M * N) as usize {
        max_error = max_error.max((result[i] - reference[i]).abs());
    }

    println!("Matrix multiplication {M}x{K}x{N} complete.");
    println!("Max error vs CPU reference: {max_error:.6}");
    if max_error < tolerance {
        println!("PASS");
    } else {
        println!("FAIL (tolerance: {tolerance})");
    }

    // Print top-left 4x4 of result
    println!("Result (top-left 4x4):");
    for i in 0..4u32 {
        let row: Vec<String> = (0..4u32)
            .map(|j| format!("{:8.3}", result[(i * N + j) as usize]))
            .collect();
        println!("  [{}]", row.join(", "));
    }

    // Cleanup
    context.destroy_buffer(buf_a);
    context.destroy_buffer(buf_b);
    context.destroy_buffer(buf_c);
    context.destroy_buffer(buf_params);
    context.destroy_compute_pipeline(&mut pipeline);
    context.destroy_command_encoder(&mut encoder);
}
