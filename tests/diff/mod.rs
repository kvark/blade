//! Differentiable rendering of the material grid.

use super::{PbrHarness, SHADER_DIR, pbr_scene};
use blade_graphics as gpu;
use blade_render::diff::{
    AdamConfig, DiffConfig, DiffShaders, DiffTracer, MaterialParamFlags, MaterialParams,
};

const SIZE: gpu::Extent = gpu::Extent {
    width: 40,
    height: 32,
    depth: 1,
};
// Different from any frame index the gradients are taken at.
const TARGET_FRAMES: std::ops::Range<u32> = 1_000_000..1_000_032;
const CONVERGED_FRAMES: std::ops::Range<u32> = 3_000_000..3_000_064;

/// Row 0 column 2: a dielectric sphere.
const DIELECTRIC: usize = 2;
/// Row 2 column 0: the smoothest metal.
const FULL_METAL: usize = 2 * pbr_scene::COLUMNS;
/// Row 2 column 1: a smooth metal.
const METAL: usize = 2 * pbr_scene::COLUMNS + 1;
/// Row 3 column 0: an emissive sphere.
const EMISSIVE: usize = 3 * pbr_scene::COLUMNS;

const SKY_SIZE: [u32; 2] = [8, 4];

/// Light from above and one side, so that the response of the materials
/// depends on the direction, unlike under a uniform environment.
fn sky_texels() -> Vec<[u8; 4]> {
    let mut texels = Vec::new();
    for y in 0..SKY_SIZE[1] {
        for x in 0..SKY_SIZE[0] {
            let brightness: u8 = [255, 160, 60, 10][y as usize];
            let warm = if x == 3 { 255 } else { brightness };
            texels.push([warm, brightness, (u16::from(brightness) * 3 / 4) as u8, 255]);
        }
    }
    texels
}

struct DiffScene {
    harness: PbrHarness,
    encoder: gpu::CommandEncoder,
    renderer: blade_render::RayTracer,
    objects: Vec<blade_render::Object>,
    temp: blade_render::FrameResources,
    diff: DiffTracer,
    camera: blade_render::Camera,
    config: DiffConfig,
}

impl DiffScene {
    fn new(cache_name: &str) -> Option<Self> {
        if cfg!(target_os = "macos") && std::env::var_os("CI").is_some() {
            println!("Skipping: ray tracing is not supported on macOS CI");
            return None;
        }
        let context = match unsafe {
            gpu::Context::init(gpu::ContextDesc {
                ray_tracing: true,
                ..Default::default()
            })
        } {
            Ok(context) => context,
            Err(e) => {
                println!("Skipping: GPU context with ray tracing not available: {e:?}");
                return None;
            }
        };
        let caps = context.capabilities();
        if !caps.ray_query.contains(gpu::ShaderVisibility::COMPUTE)
            || !caps.binding_array
            || !caps.shader_float32_atomic
        {
            println!("Skipping: ray queries, binding arrays or float32 atomics not supported");
            return None;
        }

        let harness = PbrHarness::new(context, cache_name, true);
        let context = std::sync::Arc::clone(&harness.context);
        let (diff_shaders, task) = DiffShaders::load(SHADER_DIR.as_ref(), &harness.asset_hub);
        if harness.workers.is_empty() {
            task.join_active();
        } else {
            task.join();
        }

        let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "diff",
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
                    format: gpu::TextureFormat::Rgba32Float,
                    alpha: gpu::AlphaMode::Ignored,
                },
                color_space: gpu::ColorSpace::Linear,
                max_debug_lines: 16,
            },
        );
        let mut objects = vec![blade_render::Object::from(
            harness.create_grid_model([0.3, 1.0]),
        )];
        let sky = harness.asset_hub.textures.baker.create_texture(
            "diff-sky",
            SKY_SIZE[0],
            SKY_SIZE[1],
            &sky_texels(),
        );
        let sky = harness.asset_hub.textures.insert(sky);
        let mut temp = blade_render::FrameResources::default();
        harness.asset_hub.flush(&mut encoder, &mut temp.buffers);
        renderer.build_scene(
            &mut encoder,
            &mut objects,
            Some(sky),
            &harness.asset_hub,
            &context,
            &mut temp,
        );
        let diff = DiffTracer::new(
            &mut encoder,
            &context,
            &diff_shaders,
            &harness.asset_hub.shaders,
            &renderer,
        );
        let mut scene = Self {
            harness,
            encoder,
            renderer,
            objects,
            temp,
            diff,
            camera: pbr_scene::camera(),
            config: DiffConfig {
                num_environment_samples: 1,
                samples_per_pixel: 2,
                max_bounces: 2,
                ..Default::default()
            },
        };
        scene.submit();
        assert_eq!(
            scene.diff.material_count(),
            pbr_scene::COLUMNS * pbr_scene::ROWS
        );
        Some(scene)
    }

    fn submit(&mut self) {
        let sync_point = self.harness.context.submit(&mut self.encoder);
        assert!(self.harness.context.wait_for(&sync_point, !0).unwrap());
        self.encoder.start();
    }

    /// Render many frames into the image, with the given frame indices.
    fn render_converged(&mut self, frames: std::ops::Range<u32>) {
        let start = frames.start;
        for frame in frames {
            self.diff.render(
                &mut self.encoder,
                &self.renderer,
                &self.camera,
                &self.config,
                frame,
                frame != start,
            );
        }
    }

    fn capture_target(&mut self) {
        self.render_converged(TARGET_FRAMES);
        self.diff.capture_target(&mut self.encoder);
        self.submit();
    }

    /// Loss of a converged image, rendered with the given frames.
    fn converged_loss(&mut self, frames: std::ops::Range<u32>) -> f32 {
        self.render_converged(frames);
        self.diff.evaluate_loss(&mut self.encoder);
        self.submit();
        self.diff.loss()
    }

    /// One noisy loss and gradient evaluation, adding to the gradients.
    ///
    /// The image the loss is taken on gets more paths than the gradient,
    /// since its noise multiplies that of the gradient.
    fn backward(&mut self, iteration: u32, primal_samples: u32, adjoint_samples: u32) {
        let primal = DiffConfig {
            samples_per_pixel: primal_samples,
            ..self.config
        };
        let adjoint = DiffConfig {
            samples_per_pixel: adjoint_samples,
            ..self.config
        };
        self.diff.render(
            &mut self.encoder,
            &self.renderer,
            &self.camera,
            &primal,
            2 * iteration,
            false,
        );
        self.diff.backward(
            &mut self.encoder,
            &self.renderer,
            &self.camera,
            &adjoint,
            2 * iteration + 1,
        );
    }

    fn destroy(mut self) {
        let context = std::sync::Arc::clone(&self.harness.context);
        self.diff.destroy(&context);
        for buffer in self.temp.buffers.drain(..) {
            context.destroy_buffer(buffer);
        }
        for acceleration_structure in self.temp.acceleration_structures.drain(..) {
            context.destroy_acceleration_structure(acceleration_structure);
        }
        for object in self.objects.iter_mut() {
            object.destroy_gpu(&context);
        }
        self.renderer.destroy(&context);
        context.destroy_command_encoder(&mut self.encoder);
        self.harness.destroy();
    }
}

/// Selects one float of the material parameters.
type Component = fn(&mut MaterialParams) -> &mut f32;

/// The gradient of the loss by a few material factors must agree with
/// central differences of the loss of converged images.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_material_gradients_match_finite_differences() {
    let Some(mut scene) = DiffScene::new("diff-gradients") else {
        return;
    };
    scene.capture_target();

    let checks: [(&str, usize, Component, f32); 4] = [
        ("base color", DIELECTRIC, |p| &mut p.base_color[0], -0.3),
        ("metalness", FULL_METAL, |p| &mut p.metalness, -0.4),
        ("roughness", METAL, |p| &mut p.roughness, 0.25),
        ("emissive", EMISSIVE, |p| &mut p.emissive[1], 0.5),
    ];
    for &(_, entry, component, offset) in &checks {
        *component(&mut scene.diff.material_params_mut()[entry]) += offset;
    }

    const ITERATIONS: u32 = 256;
    for iteration in 0..ITERATIONS {
        scene.backward(iteration, 8, 4);
    }
    scene.submit();
    let grads = scene.diff.material_grads().to_vec();

    const H: f32 = 0.05;
    // Common random numbers for both sides of each difference.
    let frames = 2_000_000..2_000_512;
    for &(name, entry, component, _) in &checks {
        let mut grad = grads[entry];
        let analytic = *component(&mut grad) / ITERATIONS as f32;
        *component(&mut scene.diff.material_params_mut()[entry]) += H;
        let above = scene.converged_loss(frames.clone());
        *component(&mut scene.diff.material_params_mut()[entry]) -= 2.0 * H;
        let below = scene.converged_loss(frames.clone());
        *component(&mut scene.diff.material_params_mut()[entry]) += H;
        let numeric = (above - below) / (2.0 * H);
        println!("d loss / d {name}: gradient {analytic:e}, central difference {numeric:e}");
        assert!(
            analytic * numeric > 0.0 && (analytic - numeric).abs() <= 0.25 * numeric.abs(),
            "d loss / d {name}: gradient {analytic:e} disagrees with {numeric:e}"
        );
    }

    // Parameters that are at their true values have much smaller gradients.
    let untouched = grads[0].base_color[0].abs() / ITERATIONS as f32;
    let mut perturbed = grads[DIELECTRIC];
    let perturbed = checks[0].2(&mut perturbed).abs() / ITERATIONS as f32;
    assert!(
        untouched < 0.2 * perturbed,
        "an unperturbed sphere has a gradient of {untouched:e}, against {perturbed:e}"
    );
    scene.destroy();
}

/// Optimizing the base colors of the dielectric row recovers them.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_recovers_base_colors() {
    let Some(mut scene) = DiffScene::new("diff-recovery") else {
        return;
    };
    scene.capture_target();
    let truth = scene.diff.material_params().to_vec();
    for params in &mut scene.diff.material_params_mut()[..pbr_scene::COLUMNS] {
        params.base_color = [0.4, 0.4, 0.4];
    }
    let error = |params: &[MaterialParams]| {
        let mut sum = 0.0;
        for (p, t) in params.iter().zip(&truth).take(pbr_scene::COLUMNS) {
            for c in 0..3 {
                sum += (p.base_color[c] - t.base_color[c]).abs();
            }
        }
        sum / (3 * pbr_scene::COLUMNS) as f32
    };
    let initial_error = error(scene.diff.material_params());
    // The loss of a single step is mostly the noise of its image.
    let initial_loss = scene.converged_loss(CONVERGED_FRAMES);

    let adam = AdamConfig {
        learning_rate: 0.03,
        ..Default::default()
    };
    for iteration in 0..60 {
        scene.backward(iteration, 2, 2);
        scene.diff.step(
            &mut scene.encoder,
            Some((adam, MaterialParamFlags::BASE_COLOR)),
            None,
        );
        scene.submit();
    }
    let final_error = error(scene.diff.material_params());
    let final_loss = scene.converged_loss(CONVERGED_FRAMES);
    println!(
        "base color error {initial_error:.3} -> {final_error:.3}, loss {initial_loss:e} -> {final_loss:e}"
    );
    for (p, t) in scene
        .diff
        .material_params()
        .iter()
        .zip(&truth)
        .take(pbr_scene::COLUMNS)
    {
        println!("  {:?} for {:?}", p.base_color, t.base_color);
    }
    assert!(
        final_error < 0.25 * initial_error,
        "the base color error only went from {initial_error} to {final_error}"
    );
    // Nothing else is optimized.
    for (p, t) in scene.diff.material_params().iter().zip(&truth) {
        assert_eq!(
            (p.metalness, p.roughness, p.emissive),
            (t.metalness, t.roughness, t.emissive)
        );
    }
    assert!(
        final_loss < 0.25 * initial_loss,
        "the loss only went from {initial_loss} to {final_loss}"
    );
    scene.destroy();
}

/// Optimizing the environment recovers the radiance of the texels the camera
/// sees directly, behind the grid. The others are only seen in reflections
/// off small spheres, which leaves them underdetermined.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_recovers_environment() {
    const STEPS: u32 = 150;
    let Some(mut scene) = DiffScene::new("diff-environment") else {
        return;
    };
    scene.capture_target();
    let size = scene.diff.environment_size();
    assert_eq!([size.width, size.height], SKY_SIZE);
    let texel_count = (size.width * size.height) as usize;
    let bytes = (texel_count * 16) as u64;
    let context = std::sync::Arc::clone(&scene.harness.context);
    let staging = context.create_buffer(gpu::BufferDesc {
        name: "diff-environment",
        size: bytes,
        memory: gpu::Memory::Shared,
    });
    let texels = |staging: &gpu::Buffer| unsafe {
        std::slice::from_raw_parts_mut(staging.data() as *mut [f32; 4], texel_count)
    };
    let download = |scene: &mut DiffScene| {
        scene
            .encoder
            .transfer("diff-environment")
            .copy_buffer_to_buffer(scene.diff.environment_params(), staging.into(), bytes);
        scene.submit();
        texels(&staging).to_vec()
    };
    let truth = download(&mut scene);
    for (texel, expected) in truth.iter().zip(sky_texels()) {
        for c in 0..4 {
            assert!((texel[c] - f32::from(expected[c]) / 255.0).abs() < 1.0e-6);
        }
    }
    // Flatten the sky into its average.
    let mut average = [0.0; 3];
    for texel in &truth {
        for c in 0..3 {
            average[c] += texel[c] / texel_count as f32;
        }
    }
    for texel in texels(&staging).iter_mut() {
        *texel = [average[0], average[1], average[2], 1.0];
    }
    context.sync_buffer(staging.into(), bytes, gpu::BufferTarget::Data);
    scene
        .encoder
        .transfer("diff-environment")
        .copy_buffer_to_buffer(staging.into(), scene.diff.environment_params(), bytes);
    scene.submit();

    // The camera looks down -Z, at the seam of the map, and its field of
    // view stays within the two middle rows.
    let width = size.width as usize;
    let visible = [width, 2 * width - 1, 2 * width, 3 * width - 1];
    let error = |texels: &[[f32; 4]]| {
        let mut max: f32 = 0.0;
        for &index in &visible {
            for c in 0..3 {
                max = max.max((texels[index][c] - truth[index][c]).abs());
            }
        }
        max
    };
    let initial_error = error(&download(&mut scene));
    let initial_loss = scene.converged_loss(CONVERGED_FRAMES);
    let adam = AdamConfig {
        learning_rate: 0.05,
        ..Default::default()
    };
    for iteration in 0..STEPS {
        scene.backward(iteration, 2, 2);
        scene.diff.step(&mut scene.encoder, None, Some(adam));
        scene.submit();
    }
    let recovered = download(&mut scene);
    let final_error = error(&recovered);
    let final_loss = scene.converged_loss(CONVERGED_FRAMES);
    println!(
        "visible sky error {initial_error:.3} -> {final_error:.3}, loss {initial_loss:e} -> {final_loss:e}"
    );
    for &index in &visible {
        println!(
            "  {:?} for {:?}",
            &recovered[index][..3],
            &truth[index][..3]
        );
    }
    assert!(
        final_error < 0.05,
        "the visible sky error only went from {initial_error} to {final_error}"
    );
    assert!(
        final_loss < 0.25 * initial_loss,
        "the loss only went from {initial_loss} to {final_loss}"
    );
    // Untouched by the optimizer.
    assert!(recovered.iter().all(|texel| texel[3] == 1.0));
    context.destroy_buffer(staging);
    scene.destroy();
}
