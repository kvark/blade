//! Is `Memory::Download` worth having next to `Memory::Shared`?
//!
//! Both are host visible, so either can back a readback. They ask the
//! allocator for different things though: `Shared` takes whatever host visible
//! memory comes first, which on a discrete GPU is uncached write-combined,
//! while `Download` asks for `HOST_CACHED`. Writing to write-combined memory is
//! fast and reading back from it is not, because the reads miss the cache and
//! nothing prefetches them.
//!
//! That only matters if the CPU actually reads what it downloads, which is the
//! case this exists for: scoring passes that scan every value they pull back.
//!
//! Run with:
//!   cargo test --release --test download_memory -- --ignored --nocapture
#![cfg(not(gles))]

use blade_graphics as gpu;
use std::time;

/// Large enough to leave any cache behind, small enough to stay quick.
const MEGABYTES: usize = 64;
const SIZE: usize = MEGABYTES * 1024 * 1024;
/// Repeats, to see past a noisy first pass.
const PASSES: usize = 3;

/// Sum every byte, which is what a scoring pass does in effect: touch all of
/// it, in order, once.
///
/// `read_volatile` keeps the optimiser from noticing the sum is unused and
/// deleting the loads that are the entire point of the measurement.
fn scan(pointer: *const u8, length: usize) -> u64 {
    let mut total = 0u64;
    let words = length / 8;
    let as_u64 = pointer as *const u64;
    for index in 0..words {
        total = total.wrapping_add(unsafe { as_u64.add(index).read_volatile() });
    }
    total
}

fn time_scan(name: &str, pointer: *const u8, length: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..PASSES {
        let started = time::Instant::now();
        let checksum = scan(pointer, length);
        let elapsed = started.elapsed().as_secs_f64();
        assert!(checksum != u64::MAX, "the scan was optimised away");
        best = best.min(elapsed);
    }
    let rate = length as f64 / best / 1.0e9;
    println!("{name:<10}{:>8.1} ms{:>9.2} GB/s", best * 1000.0, rate);
    rate
}

#[test]
#[ignore = "requires a working GPU context"]
fn download_memory_reads_faster_than_shared() {
    let context = match unsafe { gpu::Context::init(gpu::ContextDesc::default()) } {
        Ok(c) => c,
        Err(e) => {
            println!("Skipping: no GPU context: {e:?}");
            return;
        }
    };
    let info = context.device_information();
    println!("GPU: {} ({:?})", info.device_name, info.driver_name);
    if info.is_software_emulated {
        println!("Skipping: a software rasterizer has no interesting memory types");
        return;
    }

    let source = context.create_buffer(gpu::BufferDesc {
        name: "download-source",
        size: SIZE as u64,
        memory: gpu::Memory::Device,
        transient: false,
    });
    let shared = context.create_buffer(gpu::BufferDesc {
        name: "download-shared",
        size: SIZE as u64,
        memory: gpu::Memory::Shared,
        transient: false,
    });
    let download = context.create_buffer(gpu::BufferDesc {
        name: "download-cached",
        size: SIZE as u64,
        memory: gpu::Memory::Download,
        transient: false,
    });

    // Fill the device buffer through an upload staging buffer, so both
    // readbacks are fed by the same kind of transfer the real path uses.
    let staging = context.create_buffer(gpu::BufferDesc {
        name: "download-staging",
        size: SIZE as u64,
        memory: gpu::Memory::Upload,
        transient: false,
    });
    unsafe {
        std::ptr::write_bytes(staging.data(), 0x5a, SIZE);
    }

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "download-memory",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    {
        let mut transfer = encoder.transfer("fill");
        transfer.copy_buffer_to_buffer(staging.at(0), source.at(0), SIZE as u64);
    }
    {
        let mut transfer = encoder.transfer("readback");
        transfer.copy_buffer_to_buffer(source.at(0), shared.at(0), SIZE as u64);
        transfer.copy_buffer_to_buffer(source.at(0), download.at(0), SIZE as u64);
    }
    let sync_point = context.submit(&mut encoder);
    assert!(context.wait_for(&sync_point, 20_000).unwrap());

    println!("\nCPU scan in place, {MEGABYTES} MiB, best of {PASSES}:");
    let shared_rate = time_scan("Shared", shared.data(), SIZE);
    let download_rate = time_scan("Download", download.data(), SIZE);
    println!(
        "\nDownload is {:.1}x faster to read",
        download_rate / shared_rate
    );
    // On a discrete GPU this is orders of magnitude; on unified memory both
    // land on the same heap and it is a wash. Either is fine — what would not
    // be is `Download` losing, which is what happens if it stops asking the
    // allocator for a cached mapping.
    assert!(
        download_rate >= shared_rate * 0.9,
        "Download reads at {download_rate:.2} GB/s against Shared at {shared_rate:.2}, \
         so it is no longer getting cached memory"
    );

    // Scalar loads are the worst way to read write-combined memory, and a bulk
    // copy is the obvious way out: `memcpy` uses wide loads, which such memory
    // handles far better than one word at a time. Worth knowing how much of
    // the gap that recovers before concluding a cached mapping is required.
    println!("\nsame, copied into ordinary memory first:");
    let mut scratch = vec![0u8; SIZE];
    for (name, buffer) in [("Shared", &shared), ("Download", &download)] {
        let mut best = f64::INFINITY;
        for _ in 0..PASSES {
            let started = time::Instant::now();
            unsafe {
                std::ptr::copy_nonoverlapping(buffer.data(), scratch.as_mut_ptr(), SIZE);
            }
            let checksum = scan(scratch.as_ptr(), SIZE);
            assert!(checksum != u64::MAX, "the scan was optimised away");
            best = best.min(started.elapsed().as_secs_f64());
        }
        println!(
            "{name:<10}{:>8.1} ms{:>9.2} GB/s",
            best * 1000.0,
            SIZE as f64 / best / 1.0e9
        );
    }

    context.destroy_buffer(staging);
    context.destroy_buffer(download);
    context.destroy_buffer(shared);
    context.destroy_buffer(source);
    context.destroy_command_encoder(&mut encoder);
}
