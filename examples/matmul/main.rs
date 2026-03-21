//! Fast matrix multiplication using cooperative matrix operations.
//!
//! Computes C = A * B for 64x64 f32 matrices using 8x8 cooperative
//! matrix tiles (tensor cores / simdgroup matrix), then verifies
//! against a CPU reference.
//!
//! Requires VK_KHR_cooperative_matrix (Vulkan) or Apple7+ (Metal).

use blade_graphics as gpu;
use gpu::ShaderData as _;
use std::mem;

const M: u32 = 64;
const N: u32 = 64;
const K: u32 = 64;
const TILE: u32 = 8;

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
    if !caps.cooperative_matrix {
        eprintln!(
            "Cooperative matrix not supported on this device ({}).",
            context.device_information().device_name
        );
        eprintln!("Requires VK_KHR_cooperative_matrix (Vulkan) or Apple7+ (Metal).");
        return;
    }
    println!(
        "Device: {} (cooperative matrix supported)",
        context.device_information().device_name
    );

    // Create shader and pipeline
    let source = include_str!("matmul.wgsl");
    let shader = context.create_shader(gpu::ShaderDesc {
        source,
        naga_module: None,
    });
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "matmul",
        data_layouts: &[&MatmulData::layout()],
        compute: shader.at("main"),
    });

    // Prepare input matrices
    let a_data: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..K * N).map(|i| (i % 11) as f32 * 0.1).collect();
    let c_data: Vec<f32> = vec![0.0; (M * N) as usize];
    let params = Params { m: M, n: N, k: K };

    let buf_a = context.create_buffer(gpu::BufferDesc {
        name: "matrix_a",
        size: (a_data.len() * mem::size_of::<f32>()) as u64,
        memory: gpu::Memory::Shared,
    });
    let buf_b = context.create_buffer(gpu::BufferDesc {
        name: "matrix_b",
        size: (b_data.len() * mem::size_of::<f32>()) as u64,
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
        std::ptr::copy_nonoverlapping(a_data.as_ptr() as *const u8, buf_a.data(), a_data.len() * 4);
        std::ptr::copy_nonoverlapping(b_data.as_ptr() as *const u8, buf_b.data(), b_data.len() * 4);
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
        pe.dispatch([M / TILE, N / TILE, 1]);
    }
    let sp = context.submit(&mut encoder);
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
                sum += a_data[(i * K + ki) as usize] * b_data[(ki * N + j) as usize];
            }
            reference[(i * N + j) as usize] = sum;
        }
    }

    // Verify
    let mut max_error = 0.0f32;
    for i in 0..(M * N) as usize {
        max_error = max_error.max((result[i] - reference[i]).abs());
    }

    println!("Matrix multiplication {M}x{K}x{N} complete.");
    println!("Max error vs CPU reference: {max_error:.6}");
    if max_error < 0.01 {
        println!("PASS");
    } else {
        println!("FAIL (tolerance: 0.01)");
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
