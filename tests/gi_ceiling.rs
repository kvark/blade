//! How much of a path-traced image can direct lighting alone explain?
//!
//! A relightable point-cloud model shades a primitive by evaluating a BRDF
//! against the environment light with a visibility term. That is exactly what
//! this renderer does at `max_bounces = 0`: next event estimation at the
//! primary hit and nothing else. Every bit of light the reference adds beyond
//! that arrives only after bouncing off another surface, and no direct-lighting
//! model can represent it.
//!
//! So the PSNR between a `max_bounces = 0` render and a converged one is an
//! upper bound on the quality any such reconstruction can reach, whatever the
//! primitive or the optimiser. It is worth knowing before building one.
//!
//! Two numbers make the bound readable, and both are reported:
//!
//! - the *noise floor*: two independent converged renders of the same scene,
//!   which bounds how much of any gap is just Monte Carlo variance;
//! - the *bounce ladder*: one, two, ... bounces against the reference, which
//!   shows how fast the residual decays and hence whether a one-bounce
//!   extension would be worth its cost.
//!
//! The ceiling is a property of the scene, not of the renderer: an open scene
//! under a distant light barely interreflects, while an enclosed one does
//! little else. The scenes below are chosen to bracket that range.
//!
//! Run with:
//!   cargo test --release --test gi_ceiling -- --ignored --nocapture
//!
//! Environment:
//!   GI_FRAMES  paths accumulated per render (default 512)
//!   GI_DUMP    directory to write the rendered images into
#![cfg(not(gles))]
#![allow(dead_code)]

mod pbr_scene;
mod snapshot;

use blade_graphics as gpu;

const SIZE: gpu::Extent = gpu::Extent {
    width: 256,
    height: 192,
    depth: 1,
};
/// The renderer writes linear values and the hardware encodes them, so the
/// readback holds display-referred sRGB — the same space the reconstruction
/// track measures PSNR in.
const FORMAT: gpu::TextureFormat = gpu::TextureFormat::Rgba8UnormSrgb;
/// Bounce counts to compare against the reference.
const LADDER: [u32; 7] = [0, 1, 2, 3, 4, 6, 8];
/// Path length treated as converged ground truth.
const REFERENCE_BOUNCES: u32 = 16;

fn frame_count() -> usize {
    std::env::var("GI_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512)
}

/// Pixels showing the environment directly, which every render reproduces
/// exactly because the environment *is* the light.
///
/// Leaving them in makes the score depend on how much empty frame the camera
/// happens to include, which is a property of the shot rather than of the
/// representation. The uniform environment encodes to pure white, so a pixel
/// is foreground exactly when it is not.
fn foreground_mask(reference: &[u8]) -> Vec<bool> {
    reference
        .chunks(4)
        .map(|texel| texel[..3] != [255, 255, 255])
        .collect()
}

/// Peak signal-to-noise ratio over the color channels of two 8-bit images,
/// restricted to the pixels selected by `mask`.
fn psnr(a: &[u8], b: &[u8], mask: Option<&[bool]>) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    let mut count = 0usize;
    for (index, (texel_a, texel_b)) in a.chunks(4).zip(b.chunks(4)).enumerate() {
        if let Some(m) = mask {
            if !m[index] {
                continue;
            }
        }
        for (component_a, component_b) in texel_a[..3].iter().zip(texel_b[..3].iter()) {
            let diff = *component_a as f64 - *component_b as f64;
            sum += diff * diff;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    let mse = sum / count as f64;
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

fn encode_normal(v: [f32; 3]) -> u32 {
    let quantize = |f: f32| ((f.clamp(-1.0, 1.0) * 127.0 + 0.5) as i8) as u8 as u32;
    quantize(v[0]) | (quantize(v[1]) << 8) | (quantize(v[2]) << 16)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / length, v[1] / length, v[2] / length]
}

/// A flat quad spanned by `right` and `up`, facing along their cross product.
///
/// Wound counter-clockwise as seen from the facing side, matching the sphere
/// in `pbr_scene`, so the ray tracer's flat normals agree with the shading
/// ones.
fn quad(center: [f32; 3], right: [f32; 3], up: [f32; 3]) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    let normal = normalize(cross(right, up));
    let tangent = normalize(right);
    let corner = |sr: f32, su: f32, tex_coords: [f32; 2]| blade_render::Vertex {
        position: [
            center[0] + sr * right[0] + su * up[0],
            center[1] + sr * right[1] + su * up[1],
            center[2] + sr * right[2] + su * up[2],
        ],
        bitangent_sign: 1.0,
        tex_coords,
        normal: encode_normal(normal),
        tangent: encode_normal(tangent),
    };
    let vertices = vec![
        corner(-1.0, -1.0, [0.0, 0.0]),
        corner(1.0, -1.0, [1.0, 0.0]),
        corner(1.0, 1.0, [1.0, 1.0]),
        corner(-1.0, 1.0, [0.0, 1.0]),
    ];
    (vertices, vec![0, 1, 2, 0, 2, 3])
}

fn wall(name: &str, center: [f32; 3], right: [f32; 3], up: [f32; 3]) -> blade_render::ProceduralGeometry {
    let (vertices, indices) = quad(center, right, up);
    blade_render::ProceduralGeometry {
        name: name.to_string(),
        vertices,
        indices,
        // A neutral diffuse wall: bright enough to bounce a lot of light,
        // which is the point of the enclosed scene.
        base_color_factor: [0.7, 0.7, 0.7, 1.0],
        metalness: 0.0,
        roughness: 1.0,
        emissive_factor: [0.0; 3],
    }
}

/// The lit part of the material grid, without the emissive row.
fn material_spheres(roughness_range: [f32; 2]) -> Vec<blade_render::ProceduralGeometry> {
    pbr_scene::material_grid(roughness_range)
        .into_iter()
        .filter(|geometry| geometry.emissive_factor == [0.0; 3])
        .collect()
}

/// A lone white sphere in a uniform environment: the white furnace test.
///
/// With an energy conserving BRDF and enough bounces this has to converge to
/// the environment radiance exactly, making the sphere vanish into the
/// background. Whatever darkness is left is energy the renderer dropped, so
/// this doubles as a check on the bounce ladder itself.
fn furnace(roughness: f32) -> Vec<blade_render::ProceduralGeometry> {
    let (vertices, indices) = pbr_scene::sphere([0.0, 0.0, 0.0], 2.0);
    vec![blade_render::ProceduralGeometry {
        name: "furnace-sphere".to_string(),
        vertices,
        indices,
        base_color_factor: [1.0, 1.0, 1.0, 1.0],
        metalness: 0.0,
        roughness,
        emissive_factor: [0.0; 3],
    }]
}

/// The material grid standing on a single diffuse floor.
///
/// The common case for a game asset: an object outdoors, lit by the sky, with
/// one large surface nearby to bounce light back up into it.
fn grid_on_floor() -> Vec<blade_render::ProceduralGeometry> {
    let mut geometries = material_spheres([0.3, 1.0]);
    geometries.push(wall(
        "floor",
        [0.0, -3.0, 3.0],
        [6.0, 0.0, 0.0],
        [0.0, 0.0, -6.0],
    ));
    geometries
}

/// The material grid inside a diffuse room that is open towards the camera.
///
/// Five walls bounce light around while the sixth is left out, so the
/// environment can actually get in. Sealing the box instead would leave the
/// interior black, which measures nothing.
fn room() -> Vec<blade_render::ProceduralGeometry> {
    let mut geometries = material_spheres([0.3, 1.0]);
    geometries.push(wall(
        "floor",
        [0.0, -4.0, 3.0],
        [5.0, 0.0, 0.0],
        [0.0, 0.0, -6.0],
    ));
    geometries.push(wall(
        "ceiling",
        [0.0, 4.0, 3.0],
        [5.0, 0.0, 0.0],
        [0.0, 0.0, 6.0],
    ));
    geometries.push(wall(
        "back",
        [0.0, 0.0, -3.0],
        [5.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
    ));
    geometries.push(wall(
        "left",
        [-5.0, 0.0, 3.0],
        [0.0, 0.0, -6.0],
        [0.0, 4.0, 0.0],
    ));
    geometries.push(wall(
        "right",
        [5.0, 0.0, 3.0],
        [0.0, 0.0, 6.0],
        [0.0, 4.0, 0.0],
    ));
    geometries
}

struct Scene {
    name: &'static str,
    note: &'static str,
    geometries: Vec<blade_render::ProceduralGeometry>,
}

fn scenes() -> Vec<Scene> {
    vec![
        Scene {
            name: "furnace",
            note: "one white sphere, uniform environment (energy check)",
            geometries: furnace(0.6),
        },
        Scene {
            name: "grid-open",
            note: "material grid, nothing else to bounce off",
            geometries: material_spheres([0.3, 1.0]),
        },
        Scene {
            name: "grid-emissive",
            note: "same, plus the emissive row lighting its neighbours",
            geometries: pbr_scene::material_grid([0.3, 1.0]),
        },
        Scene {
            name: "grid-floor",
            note: "material grid standing on a diffuse floor",
            geometries: grid_on_floor(),
        },
        Scene {
            name: "room",
            note: "material grid in a room open towards the camera",
            geometries: room(),
        },
    ]
}

struct Harness {
    context: std::sync::Arc<gpu::Context>,
    choir: std::sync::Arc<choir::Choir>,
    workers: Vec<choir::WorkerHandle>,
    asset_hub: blade_render::AssetHub,
    shaders: blade_render::Shaders,
}

impl Harness {
    fn new(context: gpu::Context) -> Self {
        let context = std::sync::Arc::new(context);
        let choir = choir::Choir::new();
        let workers = (0..2)
            .map(|i| choir.add_worker(&format!("gi-ceiling-{i}")))
            .collect();
        let cache_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-assets")
            .join("gi-ceiling");
        let asset_hub = blade_render::AssetHub::new(&cache_path, &choir, &context);
        let (shaders, shader_task) = blade_render::Shaders::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("blade-render")
                .join("code")
                .as_ref(),
            &asset_hub,
            true,
        );
        shader_task.join();
        Self {
            context,
            choir,
            workers,
            asset_hub,
            shaders,
        }
    }

    fn destroy(mut self) {
        self.asset_hub.destroy();
        self.workers.clear();
        drop(self.choir);
    }
}

fn ray_config(max_bounces: u32) -> blade_render::RayConfig {
    blade_render::RayConfig {
        num_environment_samples: 1,
        num_brdf_samples: 4,
        // the dummy environment map carries no importance sampling data
        environment_importance_sampling: false,
        max_bounces,
        max_accumulated_samples: 0,
        tap_count: 2,
        tap_radius: 16,
        tap_confidence_near: 8,
        tap_confidence_far: 4,
        t_start: 0.01,
        pairwise_mis: true,
        defensive_mis: 0.1,
    }
}

/// Accumulate `frames` independent paths per pixel and read the result back.
///
/// `frame_index` inside the renderer keeps climbing across calls, so two
/// invocations with the same arguments still draw different random numbers.
/// That is what makes the noise floor measurable.
#[allow(clippy::too_many_arguments)]
fn accumulate(
    context: &gpu::Context,
    encoder: &mut gpu::CommandEncoder,
    renderer: &mut blade_render::RayTracer,
    target: &snapshot::OffscreenTarget,
    asset_hub: &blade_render::AssetHub,
    objects: &[blade_render::Object],
    camera: &blade_render::Camera,
    max_bounces: u32,
    frames: usize,
) -> Vec<u8> {
    let debug_config = blade_render::DebugConfig::default();
    encoder.start();
    let mut temp = blade_render::FrameResources::default();
    for frame_index in 0..frames {
        renderer.build_scene(encoder, objects, None, asset_hub, context, &mut temp);
        renderer.prepare(
            encoder,
            camera,
            blade_render::FrameConfig {
                frozen: false,
                debug_draw: false,
                reset_variance: frame_index == 0,
                reset_reservoirs: frame_index == 0,
                reset_accumulation: frame_index == 0,
            },
        );
        renderer.render(
            encoder,
            blade_render::RenderMode::Canonical,
            debug_config,
            ray_config(max_bounces),
            None,
        );
    }

    if let mut pass = encoder.render(
        "gi-ceiling",
        gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: target.view,
                init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        },
    ) {
        renderer.post_proc(
            &mut pass,
            debug_config,
            blade_render::PostProcConfig::default(),
            &[],
            &[],
        );
    }

    let pixels = target.read_pixels(context, encoder);
    for buffer in temp.buffers {
        context.destroy_buffer(buffer);
    }
    for acceleration_structure in temp.acceleration_structures {
        context.destroy_acceleration_structure(acceleration_structure);
    }
    pixels
}

#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn measure_gi_ceiling() {
    if cfg!(target_os = "macos") {
        println!("Skipping: ray tracing not usable on macOS CI");
        return;
    }
    let context = unsafe {
        match gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            ..Default::default()
        }) {
            Ok(c) => c,
            Err(e) => {
                println!("Skipping: no ray tracing context: {e:?}");
                return;
            }
        }
    };
    if !context
        .capabilities()
        .ray_query
        .contains(gpu::ShaderVisibility::COMPUTE)
    {
        println!("Skipping: ray_query compute not supported");
        return;
    }
    let info = context.device_information();
    println!("GPU: {} ({:?})", info.device_name, info.driver_name);
    assert!(
        !info.is_software_emulated,
        "refusing to measure on a software rasterizer"
    );

    let frames = frame_count();
    let dump = std::env::var("GI_DUMP").ok().map(std::path::PathBuf::from);
    if let Some(ref dir) = dump {
        std::fs::create_dir_all(dir).unwrap();
    }
    println!("{frames} accumulated paths per render, {}x{}\n", SIZE.width, SIZE.height);

    let harness = Harness::new(context);
    let context = std::sync::Arc::clone(&harness.context);
    let target = snapshot::OffscreenTarget::new(&context, SIZE, FORMAT);
    let camera = pbr_scene::camera();

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "gi-ceiling",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    let mut renderer = blade_render::RayTracer::new(
        &mut encoder,
        &context,
        harness.shaders.clone(),
        &harness.asset_hub.shaders,
        &blade_render::RenderConfig {
            surface_size: SIZE,
            surface_info: gpu::SurfaceInfo {
                format: FORMAT,
                alpha: gpu::AlphaMode::Ignored,
            },
            color_space: gpu::ColorSpace::Linear,
            max_debug_lines: 16,
        },
    );

    // Cook every scene up front so a single asset flush covers them all.
    let models = scenes()
        .into_iter()
        .map(|scene| {
            let model = harness
                .asset_hub
                .models
                .baker
                .create_model(scene.name, scene.geometries);
            (scene.name, scene.note, harness.asset_hub.models.insert(model))
        })
        .collect::<Vec<_>>();
    let mut flush_buffers = Vec::new();
    harness
        .asset_hub
        .flush(&mut encoder, &mut flush_buffers);
    encoder.init_texture(target.texture);
    let sync_point = context.submit(&mut encoder);
    assert!(context.wait_for(&sync_point, 10_000).unwrap());

    let mut summary = Vec::new();
    for (name, note, handle) in models {
        let objects = vec![blade_render::Object::from(handle)];
        println!("== {name}: {note} ==");

        let reference = accumulate(
            &context,
            &mut encoder,
            &mut renderer,
            &target,
            &harness.asset_hub,
            &objects,
            &camera,
            REFERENCE_BOUNCES,
            frames,
        );
        // An independent estimate of the same image: whatever separates the
        // two is variance, not bounce depth, so no gap below this line means
        // anything.
        let reference_again = accumulate(
            &context,
            &mut encoder,
            &mut renderer,
            &target,
            &harness.asset_hub,
            &objects,
            &camera,
            REFERENCE_BOUNCES,
            frames,
        );
        let mask = foreground_mask(&reference);
        let covered = mask.iter().filter(|m| **m).count();
        let coverage = covered as f64 / mask.len() as f64;
        let mask = if covered == 0 {
            // A white sphere in a white furnace converges to the environment
            // exactly, so there is no foreground to separate. That is the
            // result, not a failure.
            println!("  (no foreground: the scene is indistinguishable from the environment)");
            None
        } else {
            Some(mask)
        };
        let floor = psnr(&reference, &reference_again, mask.as_deref());
        println!(
            "  foreground                  {:5.1} % of the frame",
            100.0 * coverage
        );
        println!("  noise floor                 {floor:6.2} dB");

        let mut direct = f64::NAN;
        for bounces in LADDER {
            let image = accumulate(
                &context,
                &mut encoder,
                &mut renderer,
                &target,
                &harness.asset_hub,
                &objects,
                &camera,
                bounces,
                frames,
            );
            let value = psnr(&image, &reference, mask.as_deref());
            let whole = psnr(&image, &reference, None);
            let label = match bounces {
                0 => "direct only".to_string(),
                1 => "+1 bounce".to_string(),
                n => format!("+{n} bounces"),
            };
            println!("  {label:<26}{value:6.2} dB   (whole frame {whole:.2})");
            if bounces == 0 {
                direct = value;
            }
            if let Some(ref dir) = dump {
                snapshot::save_image(&dir.join(format!("{name}-b{bounces}.png")), &image, SIZE);
            }
        }
        if let Some(ref dir) = dump {
            snapshot::save_image(&dir.join(format!("{name}-reference.png")), &reference, SIZE);
        }
        summary.push((name, direct, floor));
        println!();
    }

    println!("== ceiling for a direct-lighting-only model ==");
    for (name, direct, floor) in summary {
        println!("  {name:<16}{direct:6.2} dB   (noise floor {floor:.2} dB)");
    }

    renderer.destroy(&context);
    for buffer in flush_buffers {
        context.destroy_buffer(buffer);
    }
    context.destroy_command_encoder(&mut encoder);
    target.destroy(&context);
    harness.destroy();
}
