//! Opt-in, matched vehicle captures and frame profiles. No golden updates.
//!
//! Run `BLADE_CAPTURE_VEHICLE=1 BLADE_PROFILE_GPU=1 cargo test --test gpu_examples compare_vehicle_denoising
//! -- --ignored --nocapture`. Optional BLADE_CAPTURE_SIZE (WIDTHxHEIGHT), CASE,
//! FRAMES, PASSES and TAPS select short profiling/quality experiments. These
//! use controlled vehicle assets, not the interactive game's physics or presentation.
use crate::{PbrHarness, snapshot};
use blade_graphics as gpu;

#[test]
#[ignore = "manual vehicle denoiser capture and GPU profile"]
fn compare_vehicle_denoising() {
    if std::env::var_os("BLADE_CAPTURE_VEHICLE").is_none() {
        println!("Skipping: set BLADE_CAPTURE_VEHICLE=1 for manual vehicle captures");
        return;
    }
    // Match the other ray-tracing fixtures: macOS CI can report capabilities
    // yet throw uncatchable Objective-C exceptions when building an AS.
    if cfg!(target_os = "macos") && std::env::var_os("CI").is_some() {
        println!("Skipping: vehicle ray tracing capture not supported on macOS CI");
        return;
    }
    let (width, height) = std::env::var("BLADE_CAPTURE_SIZE").map_or((640, 480), |value| {
        let (width, height) = value.split_once('x').unwrap();
        (width.parse().unwrap(), height.parse().unwrap())
    });
    let size = gpu::Extent {
        width,
        height,
        depth: 1,
    };
    let timing = std::env::var_os("BLADE_PROFILE_GPU").is_some();
    let moving = std::env::var_os("BLADE_CAPTURE_MOTION").is_some();
    let turning = std::env::var("BLADE_CAPTURE_MOTION").is_ok_and(|mode| mode == "turn");
    let spatial_passes =
        std::env::var("BLADE_CAPTURE_PASSES").map_or(4, |value| value.parse().unwrap());
    let taps = std::env::var("BLADE_CAPTURE_TAPS").map_or(2, |value| value.parse().unwrap());
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            timing,
            ..Default::default()
        })
        .unwrap()
    };
    let harness = PbrHarness::new(context, "vehicle-denoiser-diagnostic", true);
    let context = &harness.context;
    let mut objects = Vec::new();
    for (file, positions) in [
        ("ground.gltf", vec![glam::vec3(0.0, 0.0, 0.0)]),
        ("raceFuture-body.glb", vec![glam::vec3(0.0, 0.55, 0.1)]),
        (
            "wheelRacing.glb",
            vec![
                glam::vec3(-0.5, 0.75, 0.7),
                glam::vec3(0.5, 0.75, 0.7),
                glam::vec3(-0.5, 0.75, -0.8),
                glam::vec3(0.5, 0.75, -0.8),
            ],
        ),
    ] {
        let (model, task) = harness.asset_hub.models.load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("examples/vehicle/data")
                .join(file),
            blade_render::model::Meta {
                generate_tangents: true,
                front_face: blade_render::model::FrontFace::CounterClockwise,
            },
        );
        task.join();
        for position in positions {
            let mut object = blade_render::Object::from(model);
            object.transform.x.w = position.x;
            object.transform.y.w = position.y;
            object.transform.z.w = position.z;
            object.prev_transform = object.transform;
            objects.push(object);
        }
    }
    let position = if moving {
        glam::vec3(0.0, 4.0, 2.2)
    } else {
        glam::vec3(3.6, 3.0, 5.0)
    };
    let rotation = glam::Quat::from_mat4(
        &glam::camera::rh::view::look_at_mat4(position, glam::vec3(0.0, 0.6, 0.0), glam::Vec3::Y)
            .inverse(),
    );
    let camera = blade_render::Camera {
        pos: position.to_array().into(),
        rot: rotation.into(),
        fov_y: 0.8,
        depth: 100.0,
        fov: None,
    };
    let base_transforms: Vec<_> = objects.iter().map(|object| object.transform).collect();
    let mut directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/denoiser-diagnostic")
        .join(format!("{width}x{height}"));
    if moving {
        directory = directory.join(if turning { "turning" } else { "moving" });
    }
    std::fs::create_dir_all(&directory).unwrap();
    let mut images = Vec::new();
    for (label, mode, denoise, reset_debug, frames) in [
        (
            "reference",
            blade_render::RenderMode::Canonical,
            false,
            false,
            128,
        ),
        ("raw", blade_render::RenderMode::RealTime, false, true, 32),
        (
            "current-engine",
            blade_render::RenderMode::RealTime,
            true,
            true,
            32,
        ),
        (
            "history-preserved",
            blade_render::RenderMode::RealTime,
            true,
            false,
            32,
        ),
    ] {
        if std::env::var("BLADE_CAPTURE_CASE").is_ok_and(|case| case != label) {
            continue;
        }
        let frames =
            std::env::var("BLADE_CAPTURE_FRAMES").map_or(frames, |value| value.parse().unwrap());
        assert!(frames > 4);
        let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: label,
            buffer_count: 1,
            manual_barriers: false,
        });
        encoder.start();
        let mut renderer = blade_render::RayTracer::new(
            &mut encoder,
            context,
            harness.shaders.clone(),
            &harness.asset_hub.shaders,
            &blade_render::RenderConfig {
                surface_size: size,
                surface_info: gpu::SurfaceInfo {
                    format: gpu::TextureFormat::Rgba32Float,
                    alpha: gpu::AlphaMode::Ignored,
                },
                color_space: gpu::ColorSpace::Linear,
                max_debug_lines: 16,
            },
        );
        let mut temp = blade_render::FrameResources::default();
        harness.asset_hub.flush(&mut encoder, &mut temp.buffers);
        let sync = context.submit(&mut encoder);
        assert!(context.wait_for(&sync, !0).unwrap());
        let mut totals = std::collections::BTreeMap::<String, f64>::new();
        let mut cpu_build = 0.0;
        let mut cpu_encode = 0.0;
        let mut wall = 0.0;
        for frame in 0..frames {
            // Keep the last frame identical to the stationary reference, while
            // the following camera reveals ground previously hidden by the car.
            let offset = if moving && mode == blade_render::RenderMode::RealTime {
                0.12 * (frame as f32 - (frames - 1) as f32)
            } else {
                0.0
            };
            let mut frame_camera = camera;
            frame_camera.pos.z += offset;
            let vehicle_motion = vehicle_motion(offset / 0.12, turning);
            for (index, (object, &base)) in objects.iter_mut().zip(&base_transforms).enumerate() {
                object.transform = base;
                if index != 0 {
                    let base = glam::Mat4::from_cols(
                        base.x.into(),
                        base.y.into(),
                        base.z.into(),
                        glam::Vec4::W,
                    )
                    .transpose();
                    let rows = (vehicle_motion * base).transpose().to_cols_array_2d();
                    object.transform = [rows[0], rows[1], rows[2]].into();
                }
                if frame == 0 {
                    object.flip();
                }
            }
            let start = std::time::Instant::now();
            encoder.start();
            renderer.build_scene(
                &mut encoder,
                &mut objects,
                None,
                &harness.asset_hub,
                context,
                &mut temp,
            );
            let built = start.elapsed().as_secs_f64();
            renderer.prepare(
                &mut encoder,
                &frame_camera,
                blade_render::FrameConfig {
                    frozen: false,
                    debug_draw: false,
                    reset_variance: reset_debug || frame == 0,
                    reset_reservoirs: frame == 0,
                    reset_accumulation: frame == 0,
                },
            );
            let mut rays = blade_helpers::default_ray_config();
            rays.environment_importance_sampling = false;
            rays.tap_count = taps;
            // Match the realtime estimator: direct environment + first-hit emission.
            rays.max_bounces = 0;
            rays.jitter_primary_rays = false;
            if mode == blade_render::RenderMode::Canonical {
                rays.num_brdf_samples = 4;
            }
            renderer.render(
                &mut encoder,
                mode,
                blade_render::DebugConfig::default(),
                rays,
                denoise.then_some(blade_render::DenoiserConfig {
                    num_passes: spatial_passes,
                    temporal_weight: 0.1,
                }),
            );
            let encoded = start.elapsed().as_secs_f64();
            let sync = context.submit(&mut encoder);
            assert!(context.wait_for(&sync, !0).unwrap());
            let elapsed = start.elapsed().as_secs_f64();
            if frame >= 4 {
                cpu_build += built;
                cpu_encode += encoded - built;
                wall += elapsed;
                if timing {
                    for (name, duration) in encoder.last_timing().pass_durations() {
                        *totals.entry(name.to_owned()).or_default() += duration.as_secs_f64();
                    }
                }
            }
            for buffer in temp.buffers.drain(..) {
                context.destroy_buffer(buffer);
            }
            for accel in temp.acceleration_structures.drain(..) {
                context.destroy_acceleration_structure(accel);
            }
            for object in &mut objects {
                object.flip();
            }
        }
        let count = f64::from(frames - 4);
        println!(
            "{label}: wall {:.3} ms, CPU build {:.3}, CPU render encode {:.3}",
            wall * 1000.0 / count,
            cpu_build * 1000.0 / count,
            cpu_encode * 1000.0 / count
        );
        for (name, seconds) in totals {
            println!("  {name}: {:.3} ms", seconds * 1000.0 / count);
        }
        encoder.start();
        if moving && mode == blade_render::RenderMode::RealTime {
            verify_motion(&renderer, context, &mut encoder, size, camera, turning);
            encoder.start();
        }
        let target = snapshot::OffscreenTarget::new(context, size, gpu::TextureFormat::Rgba32Float);
        encoder.init_texture(target.texture);
        {
            let mut pass = encoder.render(
                "capture",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: target.view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            );
            renderer.post_proc(
                &mut pass,
                blade_render::DebugConfig::default(),
                blade_render::PostProcConfig {
                    tone_map: false,
                    ..Default::default()
                },
                &[],
                &[],
            );
        }
        let bytes = target.read_pixels(context, &mut encoder);
        let pixels: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&bytes).to_vec();
        // Default post-processing has unit exposure and white level, making
        // tone mapping an identity. Only clamp and apply the display transfer;
        // no image-space noise reduction is performed during capture.
        let display: Vec<u8> = pixels
            .iter()
            .flat_map(|p| {
                p.iter().enumerate().map(|(i, &v)| {
                    let linear = v.clamp(0.0, 1.0);
                    let srgb = if i == 3 {
                        1.0
                    } else if linear <= 0.0031308 {
                        12.92 * linear
                    } else {
                        1.055 * linear.powf(1.0 / 2.4) - 0.055
                    };
                    (255.0 * srgb).round() as u8
                })
            })
            .collect();
        let suffix = if spatial_passes == 4 && taps == 2 {
            String::new()
        } else {
            format!("-{spatial_passes}passes-{taps}taps")
        };
        let file = std::fs::File::create(directory.join(format!("{label}{suffix}.png"))).unwrap();
        let mut png = png::Encoder::new(file, size.width, size.height);
        png.set_color(png::ColorType::Rgba);
        png.set_depth(png::BitDepth::Eight);
        png.write_header()
            .unwrap()
            .write_image_data(&display)
            .unwrap();
        images.push((label, pixels));
        target.destroy(context);
        renderer.destroy(context);
        context.destroy_command_encoder(&mut encoder);
        for object in &mut objects {
            object.destroy_gpu(context);
        }
    }
    let mut errors = std::collections::BTreeMap::new();
    for (label, pixels) in &images[1..] {
        let reference = &images[0].1;
        let mut squared_error = 0.0;
        let mut high_frequency_error = 0.0;
        let mut count = 0;
        for y in 1..size.height - 1 {
            for x in 1..size.width - 1 {
                let i = (y * size.width + x) as usize;
                // Reject background and sharp reference edges for the noise metric.
                let right = i + 1;
                let below = i + size.width as usize;
                if reference[i][..3].iter().all(|&value| value >= 0.999)
                    || (0..3).any(|c| {
                        (reference[i][c] - reference[right][c]).abs() > 0.02
                            || (reference[i][c] - reference[below][c]).abs() > 0.02
                    })
                {
                    continue;
                }
                for c in 0..3 {
                    let e = pixels[i][c] - reference[i][c];
                    let next_e = pixels[right][c] - reference[right][c];
                    squared_error += f64::from(e * e);
                    high_frequency_error += f64::from((e - next_e).powi(2));
                    count += 1;
                }
            }
        }
        let rmse = (squared_error / f64::from(count)).sqrt();
        let noise = (high_frequency_error / f64::from(count)).sqrt();
        errors.insert(*label, (rmse, noise));
        println!(
            "{label}: interior linear RMSE {:.6}, residual neighbor RMS {:.6}, channels {count}",
            rmse, noise,
        );
    }
    if let (Some(raw), Some(filtered), Some(control)) = (
        errors.get("raw"),
        errors.get("current-engine"),
        errors.get("history-preserved"),
    ) {
        assert!(
            filtered.0 < raw.0 * 0.5,
            "denoising did not reduce reference error: {errors:?}"
        );
        assert!(
            filtered.1 < raw.1 * if moving { 0.15 } else { 0.3 },
            "denoising did not reduce noise: {errors:?}"
        );
        assert!(
            (filtered.0 - control.0).abs() < raw.0 * 0.02,
            "debug statistics reset invalidated denoiser history: {errors:?}"
        );
    }
    println!("Captures: {}", directory.display());
    harness.destroy();
}

#[derive(blade_macros::ShaderData)]
struct MotionProbeData {
    t_depth: gpu::TextureView,
    t_motion: gpu::TextureView,
    output: gpu::BufferPiece,
}

fn vehicle_motion(phase: f32, turning: bool) -> glam::Mat4 {
    glam::Mat4::from_scale_rotation_translation(
        if turning {
            glam::vec3(
                1.0 + 0.004 * phase,
                1.0 - 0.002 * phase,
                1.0 + 0.003 * phase,
            )
        } else {
            glam::Vec3::ONE
        },
        glam::Quat::from_rotation_y(if turning { 0.025 * phase } else { 0.0 }),
        glam::vec3(0.0, 0.0, 0.12 * phase),
    )
}

fn verify_motion(
    renderer: &blade_render::RayTracer,
    context: &gpu::Context,
    encoder: &mut gpu::CommandEncoder,
    size: gpu::Extent,
    camera: blade_render::Camera,
    turning: bool,
) {
    use gpu::ShaderData as _;
    let count = (size.width * size.height) as usize;
    let output = context.create_buffer(gpu::BufferDesc {
        name: "motion probe",
        size: (count * 16) as u64,
        memory: gpu::Memory::Shared,
    });
    let source = format!(
        "{}\n{}",
        snapshot::shader_source("gbuf.inc.wgsl"),
        include_str!("../shaders/motion_probe.wgsl")
    );
    let shader = context.create_shader(gpu::ShaderDesc {
        source: &source,
        naga_module: None,
    });
    let mut pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "motion probe",
        data_layouts: &[&MotionProbeData::layout()],
        compute: shader.at("probe"),
    });
    let views = renderer.view_gbuffer();
    {
        let mut pass = encoder.compute("verify moving geometry and camera");
        let mut pc = pass.with(&pipeline);
        pc.bind(
            0,
            &MotionProbeData {
                t_depth: views.depth,
                t_motion: views.motion,
                output: output.into(),
            },
        );
        pc.dispatch([size.width.div_ceil(8), size.height.div_ceil(8), 1]);
    }
    let sync = context.submit(encoder);
    assert!(context.wait_for(&sync, !0).unwrap());
    let pixels = unsafe { std::slice::from_raw_parts(output.data().cast::<[f32; 4]>(), count) };
    let position = glam::Vec3::from(camera.pos);
    let rotation = glam::Quat::from(camera.rot);
    let previous_camera = position - glam::vec3(0.0, 0.0, 0.12);
    let half_size = glam::vec2(size.width as f32, size.height as f32) * 0.5;
    let tangent =
        glam::vec2(size.width as f32 / size.height as f32, 1.0) * (0.5 * camera.fov_y).tan();
    let mut counts = [0; 2];
    let mut max_error = [0.0f32; 2];
    for (index, &[mx, my, previous_depth, depth]) in pixels.iter().enumerate() {
        if depth == 0.0 {
            continue;
        }
        let center = glam::vec2(
            (index % size.width as usize) as f32 + 0.5,
            (index / size.width as usize) as f32 + 0.5,
        );
        let ndc = (center - half_size) / half_size;
        let direction =
            rotation * glam::vec3(ndc.x * tangent.x, -ndc.y * tangent.y, -1.0).normalize();
        let point = position + direction * depth;
        let class = if point.y.abs() < 0.002 {
            0
        } else if point.y > 0.1 && point.y < 2.5 && point.x.abs() < 1.5 && point.z.abs() < 2.0 {
            1
        } else {
            continue;
        };
        let previous_point = if class == 0 {
            point
        } else {
            vehicle_motion(-1.0, turning).transform_point3(point)
        };
        let local = rotation.inverse() * (previous_point - previous_camera);
        let projected =
            (glam::vec2(local.x, -local.y) / -local.z / tangent + glam::Vec2::ONE) * half_size;
        let error = (glam::vec2(mx, my) - (projected - center))
            .abs()
            .max_element();
        assert!(
            error < 0.1,
            "class {class}, pixel {index}: motion error {error}"
        );
        let distance = previous_point.distance(previous_camera);
        assert!(
            (previous_depth - distance).abs() < (0.001 * distance).max(0.01),
            "class {class}: previous depth {previous_depth} != {distance}"
        );
        counts[class] += 1;
        max_error[class] = max_error[class].max(error);
    }
    assert!(
        counts.into_iter().all(|count| count > 1000),
        "insufficient ground/car pixels: {counts:?}"
    );
    println!("Motion verified: ground/car pixels {counts:?}, max error {max_error:?} pixels");
    context.destroy_compute_pipeline(&mut pipeline);
    context.destroy_buffer(output);
}
