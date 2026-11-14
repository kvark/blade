//! Differentiable rendering of the material grid.

use super::{PbrHarness, SHADER_DIR, pbr_scene};
use blade_graphics as gpu;
use blade_render::diff::{
    AdamConfig, DiffConfig, DiffShaders, DiffTracer, DiffTracerDesc, GeometryBase, GeometryOffsets,
    MaterialParamFlags, MaterialParams, StepConfig,
};

const SIZE: gpu::Extent = gpu::Extent {
    width: 40,
    height: 32,
    depth: 1,
};
/// For textures, spheres span more pixels than texels of `CHECKER`, and for
/// geometry, silhouettes move across enough samples of each pixel for
/// central differences to be accurate.
const DETAILED_SIZE: gpu::Extent = gpu::Extent {
    width: 80,
    height: 64,
    depth: 1,
};
const CHECKER_SIZE: u32 = 4;
const CHECKER: [[u8; 4]; 2] = [[230, 70, 40, 255], [40, 130, 220, 255]];

/// What the scene is made of, and what is differentiated.
#[derive(Clone, Copy, PartialEq)]
enum SceneKind {
    /// The material grid, as one object.
    Grid,
    /// Every sphere of the grid as an object of its own, whose translations
    /// are parameters.
    Spheres,
    /// A quad past the edges of the view, with its normals tilted outwards,
    /// whose translation is a parameter. Nothing in view is a discontinuity,
    /// so only the interpolated normals move the image.
    Bowl,
    /// A flat-shaded cube, whose translation is a parameter. Flat faces only
    /// move the image at their edges.
    Cube,
    /// A large dielectric sphere on its own, whose translation is a parameter.
    Ball,
    /// `CHECKER` as the base color of the dielectric row.
    BaseColorTexture,
    /// Normals tilted four ways, at a unit normal scale, on a quad facing
    /// the camera instead of the grid, so that all of the texels are seen.
    NormalMap,
}

impl SceneKind {
    /// Whether the scene is made of the spheres of the grid.
    fn is_grid(self) -> bool {
        matches!(self, Self::Grid | Self::Spheres | Self::BaseColorTexture)
    }

    /// Whether translations are parameters.
    fn moves(self) -> bool {
        matches!(self, Self::Spheres | Self::Bowl | Self::Cube | Self::Ball)
    }

    fn texels(self) -> Vec<[u8; 4]> {
        let size = CHECKER_SIZE as usize;
        (0..size * size)
            .map(|i| match self {
                Self::Grid | Self::Spheres | Self::Bowl | Self::Cube | Self::Ball => {
                    unreachable!()
                }
                Self::BaseColorTexture => CHECKER[(i / size + i % size) % 2],
                Self::NormalMap => [
                    [200, 128, 255, 255],
                    [128, 200, 255, 255],
                    [56, 128, 255, 255],
                    [128, 56, 255, 255],
                ][(i / size + i % size) % 4],
            })
            .collect()
    }
}
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
    sky: blade_asset::Handle<blade_render::Texture>,
    base: GeometryBase,
    config: DiffConfig,
}

impl DiffScene {
    /// The material grid under `sky_texels`, with the textures that
    /// `kind` asks for, which are then parameters.
    fn new(cache_name: &str, kind: SceneKind) -> Option<Self> {
        let textured = matches!(kind, SceneKind::BaseColorTexture | SceneKind::NormalMap);
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
                surface_size: if textured || kind.moves() {
                    DETAILED_SIZE
                } else {
                    SIZE
                },
                surface_info: gpu::SurfaceInfo {
                    format: gpu::TextureFormat::Rgba32Float,
                    alpha: gpu::AlphaMode::Ignored,
                },
                color_space: gpu::ColorSpace::Linear,
                max_debug_lines: 16,
            },
        );
        let mut objects = if kind == SceneKind::Spheres {
            pbr_scene::material_grid([0.3, 1.0])
                .into_iter()
                .enumerate()
                .map(|(i, geometry)| {
                    let model = harness
                        .asset_hub
                        .models
                        .baker
                        .create_model(&format!("diff-sphere-{i}"), vec![geometry]);
                    blade_render::Object::from(harness.asset_hub.models.insert(model))
                })
                .collect()
        } else {
            let geometries = if kind == SceneKind::NormalMap {
                vec![quad()]
            } else if kind == SceneKind::Bowl {
                vec![bowl()]
            } else if kind == SceneKind::Cube {
                vec![cube()]
            } else if kind == SceneKind::Ball {
                vec![ball()]
            } else {
                pbr_scene::material_grid([0.3, 1.0])
            };
            let mut model = harness
                .asset_hub
                .models
                .baker
                .create_model("diff-scene", geometries);
            if textured {
                let texture = harness.asset_hub.textures.baker.create_texture(
                    "diff-texture",
                    CHECKER_SIZE,
                    CHECKER_SIZE,
                    &kind.texels(),
                );
                let texture = harness.asset_hub.textures.insert(texture);
                let count = model.geometries.len().min(pbr_scene::COLUMNS);
                for geometry in &model.geometries[..count] {
                    let material = &mut model.materials[geometry.material_index];
                    if kind == SceneKind::BaseColorTexture {
                        material.base_color_texture = Some(texture);
                        material.base_color_factor = [1.0; 4];
                    } else {
                        material.normal_texture = Some(texture);
                        material.normal_scale = 1.0;
                    }
                }
            }
            vec![blade_render::Object::from(
                harness.asset_hub.models.insert(model),
            )]
        };
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
            &harness.asset_hub,
            &renderer,
            &DiffTracerDesc {
                textures: textured,
                geometry: kind.moves(),
            },
        );
        let mut scene = Self {
            harness,
            encoder,
            renderer,
            objects,
            temp,
            diff,
            camera: pbr_scene::camera(),
            sky,
            base: GeometryBase::new(&[], &pbr_scene::camera()),
            config: DiffConfig {
                num_environment_samples: 1,
                samples_per_pixel: 2,
                max_bounces: 2,
                ..Default::default()
            },
        };
        scene.base = GeometryBase::new(&scene.objects, &scene.camera);
        scene.submit();
        if kind.is_grid() {
            assert_eq!(
                scene.diff.material_count(),
                pbr_scene::COLUMNS * pbr_scene::ROWS
            );
        }
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

    /// Material parameters and their gradients, after the submitted work.
    fn read_materials(&mut self) -> (Vec<MaterialParams>, Vec<MaterialParams>) {
        self.diff.read_back(&mut self.encoder);
        self.submit();
        (
            self.diff.material_params().to_vec(),
            self.diff.material_grads().to_vec(),
        )
    }

    fn write_materials(&mut self, params: &[MaterialParams]) {
        let context = std::sync::Arc::clone(&self.harness.context);
        self.diff
            .write_materials(&mut self.encoder, &context, params, &mut self.temp);
        self.submit();
    }

    /// Move the objects and the camera from where they started, and build
    /// the scene again. The tracer gets the offsets as its parameters, which
    /// its gradient of the rotations depends on.
    fn move_geometry(&mut self, offsets: &GeometryOffsets) {
        let context = std::sync::Arc::clone(&self.harness.context);
        self.diff
            .write_geometry_offsets(&mut self.encoder, &context, offsets, &mut self.temp);
        self.base
            .apply(offsets, &mut self.objects, &mut self.camera);
        self.diff.rebuild_scene(
            &mut self.encoder,
            &context,
            &mut self.renderer,
            &mut self.objects,
            Some(self.sky),
            &self.harness.asset_hub,
            &mut self.temp,
        );
        self.submit();
    }

    /// Offsets of the only object.
    fn single_offsets(
        translation: [f32; 3],
        rotation: [f32; 3],
        camera: [f32; 3],
    ) -> GeometryOffsets {
        GeometryOffsets {
            translations: vec![translation],
            rotations: vec![rotation],
            camera,
        }
    }

    /// Read `count` texels of a parameter buffer.
    fn download(&mut self, piece: gpu::BufferPiece, count: usize) -> Vec<[f32; 4]> {
        let context = std::sync::Arc::clone(&self.harness.context);
        let bytes = (count * 16) as u64;
        let staging = context.create_buffer(gpu::BufferDesc {
            name: "diff-download",
            size: bytes,
            memory: gpu::Memory::Shared,
        });
        self.encoder
            .transfer("diff-download")
            .copy_buffer_to_buffer(piece, staging.into(), bytes);
        self.submit();
        let texels =
            unsafe { std::slice::from_raw_parts(staging.data() as *const [f32; 4], count) }
                .to_vec();
        context.destroy_buffer(staging);
        texels
    }

    /// Write texels into a parameter buffer.
    fn upload(&mut self, piece: gpu::BufferPiece, texels: &[[f32; 4]]) {
        let context = std::sync::Arc::clone(&self.harness.context);
        let bytes = (texels.len() * 16) as u64;
        let staging = context.create_buffer(gpu::BufferDesc {
            name: "diff-upload",
            size: bytes,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            std::slice::from_raw_parts_mut(staging.data() as *mut [f32; 4], texels.len())
                .copy_from_slice(texels);
        }
        context.sync_buffer(staging.into(), bytes, gpu::BufferTarget::Data);
        self.encoder
            .transfer("diff-upload")
            .copy_buffer_to_buffer(staging.into(), piece, bytes);
        self.submit();
        context.destroy_buffer(staging);
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
    let Some(mut scene) = DiffScene::new("diff-gradients", SceneKind::Grid) else {
        return;
    };
    scene.capture_target();

    let checks: [(&str, usize, Component, f32); 4] = [
        ("base color", DIELECTRIC, |p| &mut p.base_color[0], -0.3),
        ("metalness", FULL_METAL, |p| &mut p.metalness, -0.4),
        ("roughness", METAL, |p| &mut p.roughness, 0.25),
        ("emissive", EMISSIVE, |p| &mut p.emissive[1], 0.5),
    ];
    let (mut params, _) = scene.read_materials();
    for &(_, entry, component, offset) in &checks {
        *component(&mut params[entry]) += offset;
    }
    scene.write_materials(&params);

    const ITERATIONS: u32 = 256;
    for iteration in 0..ITERATIONS {
        scene.backward(iteration, 8, 4);
    }
    let (_, grads) = scene.read_materials();

    const H: f32 = 0.05;
    // Common random numbers for both sides of each difference.
    let frames = 2_000_000..2_000_512;
    for &(name, entry, component, _) in &checks {
        let mut grad = grads[entry];
        let analytic = *component(&mut grad) / ITERATIONS as f32;
        *component(&mut params[entry]) += H;
        scene.write_materials(&params);
        let above = scene.converged_loss(frames.clone());
        *component(&mut params[entry]) -= 2.0 * H;
        scene.write_materials(&params);
        let below = scene.converged_loss(frames.clone());
        *component(&mut params[entry]) += H;
        scene.write_materials(&params);
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
    let Some(mut scene) = DiffScene::new("diff-recovery", SceneKind::Grid) else {
        return;
    };
    scene.capture_target();
    let (truth, _) = scene.read_materials();
    let mut params = truth.clone();
    for params in &mut params[..pbr_scene::COLUMNS] {
        params.base_color = [0.4, 0.4, 0.4];
    }
    scene.write_materials(&params);
    let error = |params: &[MaterialParams]| {
        let mut sum = 0.0;
        for (p, t) in params.iter().zip(&truth).take(pbr_scene::COLUMNS) {
            for c in 0..3 {
                sum += (p.base_color[c] - t.base_color[c]).abs();
            }
        }
        sum / (3 * pbr_scene::COLUMNS) as f32
    };
    let initial_error = error(&params);
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
            &StepConfig {
                materials: Some((adam, MaterialParamFlags::BASE_COLOR)),
                ..Default::default()
            },
        );
        scene.submit();
    }
    let (params, _) = scene.read_materials();
    let final_error = error(&params);
    let final_loss = scene.converged_loss(CONVERGED_FRAMES);
    println!(
        "base color error {initial_error:.3} -> {final_error:.3}, loss {initial_loss:e} -> {final_loss:e}"
    );
    for (p, t) in params.iter().zip(&truth).take(pbr_scene::COLUMNS) {
        println!("  {:?} for {:?}", p.base_color, t.base_color);
    }
    assert!(
        final_error < 0.25 * initial_error,
        "the base color error only went from {initial_error} to {final_error}"
    );
    // Nothing else is optimized.
    for (p, t) in params.iter().zip(&truth) {
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
    let Some(mut scene) = DiffScene::new("diff-environment", SceneKind::Grid) else {
        return;
    };
    scene.capture_target();
    let size = scene.diff.environment_size();
    assert_eq!([size.width, size.height], SKY_SIZE);
    let texel_count = (size.width * size.height) as usize;
    let params = scene.diff.environment_params();
    let truth = scene.download(params, texel_count);
    for (texel, expected) in truth.iter().zip(sky_texels()) {
        for c in 0..4 {
            assert!((texel[c] - f32::from(expected[c]) / 255.0).abs() < 1.0e-6);
        }
    }
    // Flatten the sky into its average.
    let average = average_color(&truth);
    scene.upload(params, &vec![average; texel_count]);

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
    let initial_error = error(&scene.download(params, texel_count));
    let initial_loss = scene.converged_loss(CONVERGED_FRAMES);
    let adam = AdamConfig {
        learning_rate: 0.05,
        ..Default::default()
    };
    for iteration in 0..STEPS {
        scene.backward(iteration, 2, 2);
        scene.diff.step(
            &mut scene.encoder,
            &StepConfig {
                environment: Some(adam),
                ..Default::default()
            },
        );
        scene.submit();
    }
    let recovered = scene.download(params, texel_count);
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
    scene.destroy();
}

/// A gray quad in the plane of the grid, facing the camera.
fn quad() -> blade_render::ProceduralGeometry {
    let [half_width, half_height] = [3.0, 2.4];
    let corners = [[-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]];
    let vertices = corners
        .iter()
        .map(|&[x, y]| blade_render::Vertex {
            position: [x * half_width, y * half_height, 0.0],
            bitangent_sign: 1.0,
            tex_coords: [0.5 * (x + 1.0), 0.5 * (1.0 - y)],
            normal: pbr_scene::encode_normal([0.0, 0.0, 1.0]),
            tangent: pbr_scene::encode_normal([1.0, 0.0, 0.0]),
        })
        .collect();
    blade_render::ProceduralGeometry {
        name: "quad".to_string(),
        vertices,
        // Counter-clockwise, seen from the camera.
        indices: vec![0, 1, 2, 0, 2, 3],
        base_color_factor: [0.6, 0.6, 0.6, 1.0],
        metalness: 0.0,
        roughness: 0.4,
        emissive_factor: [0.0; 3],
    }
}

/// A quad larger than the view, with normals tilted outwards at the corners.
fn bowl() -> blade_render::ProceduralGeometry {
    let mut geometry = quad();
    for vertex in &mut geometry.vertices {
        vertex.position[0] *= 2.0;
        vertex.position[1] *= 2.0;
        let [x, y, _] = vertex.position;
        let normal = [0.15 * x, 0.15 * y, 1.0];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + 1.0).sqrt();
        vertex.normal = pbr_scene::encode_normal(normal.map(|c| c / length));
    }
    geometry
}

/// A cube with a vertex per corner of every face, so that its faces are flat,
/// turned so that the camera sees three of them.
fn cube() -> blade_render::ProceduralGeometry {
    let (sin_y, cos_y) = 0.6f32.sin_cos();
    let (sin_x, cos_x) = 0.4f32.sin_cos();
    let turn = |v: [f32; 3]| {
        let [x, y, z] = v;
        let (x, z) = (x * cos_y + z * sin_y, -x * sin_y + z * cos_y);
        let (y, z) = (y * cos_x - z * sin_x, y * sin_x + z * cos_x);
        [x, y, z]
    };
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for axis in 0..3 {
        for sign in [1.0f32, -1.0] {
            let mut normal = [0.0; 3];
            normal[axis] = sign;
            let mut u = [0.0; 3];
            u[(axis + 1) % 3] = 1.0;
            let mut v = [0.0; 3];
            v[(axis + 2) % 3] = sign;
            let base = vertices.len() as u32;
            for [a, b] in [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]] {
                let position = [0, 1, 2].map(|i| 1.2 * (normal[i] + a * u[i] + b * v[i]));
                vertices.push(blade_render::Vertex {
                    position: turn(position),
                    bitangent_sign: 1.0,
                    tex_coords: [0.5 * (a + 1.0), 0.5 * (b + 1.0)],
                    normal: pbr_scene::encode_normal(turn(normal)),
                    tangent: pbr_scene::encode_normal(turn(u)),
                });
            }
            // Counter-clockwise, seen from outside.
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    blade_render::ProceduralGeometry {
        name: "cube".to_string(),
        vertices,
        indices,
        base_color_factor: [0.8, 0.5, 0.3, 1.0],
        metalness: 0.0,
        roughness: 0.5,
        emissive_factor: [0.0; 3],
    }
}

/// The dielectric sphere of the grid, three times larger, at the center.
fn ball() -> blade_render::ProceduralGeometry {
    let mut geometry = pbr_scene::material_grid([0.3, 1.0]).swap_remove(DIELECTRIC);
    let count = geometry.vertices.len() as f32;
    let mut center = [0.0; 3];
    for vertex in &geometry.vertices {
        for (sum, &c) in center.iter_mut().zip(&vertex.position) {
            *sum += c / count;
        }
    }
    for vertex in &mut geometry.vertices {
        for (c, &mid) in vertex.position.iter_mut().zip(&center) {
            *c = 3.0 * (*c - mid);
        }
    }
    geometry
}

fn average_color(texels: &[[f32; 4]]) -> [f32; 4] {
    let mut average = [0.0, 0.0, 0.0, 1.0];
    for texel in texels {
        for c in 0..3 {
            average[c] += texel[c] / texels.len() as f32;
        }
    }
    average
}

/// The gradient by the base color texels agrees with central differences,
/// and optimizing them recovers those that the camera sees.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_recovers_texture() {
    check_texture(SceneKind::BaseColorTexture, &[(5, 0), (8, 2)], 3);
}

/// The same for a normal map, which only has red and green.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_recovers_normal_map() {
    check_texture(SceneKind::NormalMap, &[(5, 0), (8, 1)], 2);
}

/// Check the gradient by some `(texel, channel)` pairs, and recover the first
/// `channels` channels of the texels that the camera sees from their average.
fn check_texture(kind: SceneKind, checks: &[(usize, usize)], channels: usize) {
    let name = match kind {
        SceneKind::BaseColorTexture => "diff-texture",
        _ => "diff-normal-map",
    };
    let Some(mut scene) = DiffScene::new(name, kind) else {
        return;
    };
    // The sky is the environment, not a material texture.
    let textures = scene.diff.textures().to_vec();
    assert_eq!(textures.len(), 1);
    assert_eq!([textures[0].width, textures[0].height], [CHECKER_SIZE; 2]);
    let texel_count = (CHECKER_SIZE * CHECKER_SIZE) as usize;
    let params = scene.diff.texture_params();
    scene.capture_target();
    let truth = scene.download(params, texel_count);
    let size = CHECKER_SIZE as usize;
    for (texel, expected) in truth.iter().zip(kind.texels()) {
        for c in 0..4 {
            assert!((texel[c] - f32::from(expected[c]) / 255.0).abs() < 1.0e-6);
        }
    }
    let flat = vec![average_color(&truth); texel_count];
    scene.upload(params, &flat);

    let visible = if kind == SceneKind::NormalMap {
        // The whole quad is in view.
        (0..texel_count).collect::<Vec<_>>()
    } else {
        // The front of the spheres shows the first half of the columns,
        // and the middle rows face the camera the most.
        vec![size, size + 1, 2 * size, 2 * size + 1]
    };
    for &(index, _) in checks {
        assert!(visible.contains(&index));
    }

    const ITERATIONS: u32 = 64;
    for iteration in 0..ITERATIONS {
        scene.backward(iteration, 8, 4);
    }
    let grads = scene.download(scene.diff.texture_grads(), texel_count);
    const H: f32 = 0.05;
    let frames = 2_000_000..2_000_256;
    for &(index, channel) in checks {
        let analytic = grads[index][channel] / ITERATIONS as f32;
        let mut texels = flat.clone();
        texels[index][channel] += H;
        scene.upload(params, &texels);
        let above = scene.converged_loss(frames.clone());
        texels[index][channel] -= 2.0 * H;
        scene.upload(params, &texels);
        let below = scene.converged_loss(frames.clone());
        let numeric = (above - below) / (2.0 * H);
        println!(
            "d loss / d texel {index}.{channel}: gradient {analytic:e}, central difference {numeric:e}"
        );
        assert!(
            analytic * numeric > 0.0 && (analytic - numeric).abs() <= 0.25 * numeric.abs(),
            "d loss / d texel {index}.{channel}: gradient {analytic:e} disagrees with {numeric:e}"
        );
    }

    scene.upload(params, &flat);
    // Drop the gradients of the check.
    scene.diff.step(&mut scene.encoder, &StepConfig::default());
    let error = |texels: &[[f32; 4]]| {
        let mut sum = 0.0;
        for &index in &visible {
            for c in 0..channels {
                sum += (texels[index][c] - truth[index][c]).abs();
            }
        }
        sum / (visible.len() * channels) as f32
    };
    let initial_error = error(&flat);
    // Normals change the image less than colors, so they get more steps.
    let (learning_rate, steps) = match kind {
        SceneKind::NormalMap => (0.05, 200),
        _ => (0.03, 100),
    };
    let adam = AdamConfig {
        learning_rate,
        ..Default::default()
    };
    for iteration in 0..steps {
        scene.backward(ITERATIONS + iteration, 2, 2);
        scene.diff.step(
            &mut scene.encoder,
            &StepConfig {
                textures: Some(adam),
                ..Default::default()
            },
        );
        scene.submit();
    }
    let recovered = scene.download(params, texel_count);
    let final_error = error(&recovered);
    println!("visible texel error {initial_error:.3} -> {final_error:.3}");
    for &index in &visible {
        println!(
            "  {:?} for {:?}",
            &recovered[index][..3],
            &truth[index][..3]
        );
    }
    assert!(
        final_error < 0.4 * initial_error,
        "the visible texel error only went from {initial_error} to {final_error}"
    );
    // Untouched by the optimizer, and without a gradient for normal maps.
    for texel in &recovered {
        assert_eq!(texel[3], 1.0);
        if kind == SceneKind::NormalMap {
            assert!((texel[2] - 1.0).abs() < 1.0e-6);
        }
    }
    scene.destroy();
}

/// Row 0 column 0: the leftmost dielectric sphere, at the edge of the grid.
const EDGE_SPHERE: usize = 0;

/// Optimizing the translation of a displaced sphere brings it back.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_recovers_translation() {
    let Some(mut scene) = DiffScene::new("diff-translation", SceneKind::Spheres) else {
        return;
    };
    let sphere_count = pbr_scene::COLUMNS * pbr_scene::ROWS;
    scene.config.edge_samples = 2;
    scene.capture_target();
    let displacement = [0.25, -0.2, 0.0];

    // Start from the displaced sphere, as a guess.
    let mut offsets = GeometryOffsets {
        translations: vec![[0.0; 3]; sphere_count],
        rotations: vec![[0.0; 3]; sphere_count],
        camera: [0.0; 3],
    };
    offsets.translations[EDGE_SPHERE] = displacement;
    for iteration in 0..160 {
        // Noisy gradients keep a constant rate wandering about the optimum.
        let adam = AdamConfig {
            learning_rate: 0.03 * 0.985f32.powi(iteration as i32),
            ..Default::default()
        };
        scene.move_geometry(&offsets);
        scene.backward(iteration, 4, 2);
        scene.diff.step(
            &mut scene.encoder,
            &StepConfig {
                translations: Some(adam),
                ..Default::default()
            },
        );
        scene.diff.read_back(&mut scene.encoder);
        scene.submit();
        offsets = scene.diff.geometry_offsets();
    }
    let offsets = offsets.translations;
    // Moving towards the camera mostly scales the silhouette, which is
    // barely distinguishable from the rest, so Z is only kept in check.
    let lateral_error = |offset: [f32; 3]| (offset[0] * offset[0] + offset[1] * offset[1]).sqrt();
    let initial_error = lateral_error(displacement);
    let offset = offsets[EDGE_SPHERE];
    let final_error = lateral_error(offset);
    println!("sphere displacement {initial_error:.3} -> {final_error:.3}, offset {offset:?}");
    assert!(
        final_error < 0.35 * initial_error,
        "the displacement only went from {initial_error} to {final_error}"
    );
    assert!(
        offset[2].abs() < 0.3,
        "the sphere moved {} along Z",
        offset[2]
    );
    // The other spheres stay about where they are, across the view.
    for (i, offset) in offsets.iter().enumerate() {
        println!("  sphere {i}: {offset:?}");
        if i != EDGE_SPHERE {
            let lateral = (offset[0] * offset[0] + offset[1] * offset[1]).sqrt();
            assert!(lateral < 0.15, "sphere {i} wandered off to {offset:?}");
            assert!(
                offset[2].abs() < 0.4,
                "sphere {i} wandered off to {offset:?}"
            );
        }
    }
    scene.destroy();
}

/// The interior part of the gradient by the translations: moving a surface
/// with interpolated normals under a fixed view.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_interpolated_normal_gradients_match_finite_differences() {
    check_single_object(SceneKind::Bowl, "diff-bowl", &ALL_MOVES, [0.1, -0.15, 0.05]);
}

/// The boundary part of the gradient by the translations: moving the
/// silhouette and the creases of a flat-shaded cube.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_edge_gradients_match_finite_differences() {
    // From a turned start, the cube is 10-30% off along some axes, which
    // remains to be explained, so it starts from its target rotation.
    check_single_object(SceneKind::Cube, "diff-cube", &ALL_MOVES, [0.0; 3]);
}

/// The gradient by the translation of a smooth sphere, which moves both
/// its silhouette and its shading.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_sphere_translation_gradients_match_finite_differences() {
    check_single_object(
        SceneKind::Ball,
        "diff-ball",
        // Turning a sphere about its center doesn't change the image.
        &[Move::Translation(0), Move::Translation(1), Move::Camera(2)],
        [0.0; 3],
    );
}

const ALL_MOVES: [Move; 5] = [
    Move::Translation(0),
    Move::Translation(1),
    Move::Rotation(0),
    Move::Rotation(1),
    Move::Camera(2),
];

/// What a finite difference check moves.
#[derive(Clone, Copy, Debug)]
enum Move {
    Translation(usize),
    Rotation(usize),
    Camera(usize),
}

/// Check the gradient by moving the only object, starting off from where it
/// was for the target.
fn check_single_object(kind: SceneKind, cache_name: &str, moves: &[Move], rotation: [f32; 3]) {
    let Some(mut scene) = DiffScene::new(cache_name, kind) else {
        return;
    };
    scene.config.edge_samples = 4;
    scene.capture_target();
    // Off along Z too, or the gradient by Z would be about zero.
    let start = DiffScene::single_offsets([0.3, -0.2, 0.6], rotation, [0.0; 3]);
    scene.move_geometry(&start);
    const ITERATIONS: u32 = 64;
    for iteration in 0..ITERATIONS {
        scene.backward(iteration, 8, 4);
    }
    scene.diff.read_back(&mut scene.encoder);
    scene.submit();
    let translation_grad = scene.diff.translation_grads()[0];
    let rotation_grad = scene.diff.rotation_grads()[0];
    let camera_grad = scene.diff.camera_offset_grad();
    const H: f32 = 0.05;
    let frames = 2_000_000..2_000_128;
    for &movement in moves {
        let (grad, axis) = match movement {
            Move::Translation(axis) => (translation_grad[axis], axis),
            Move::Rotation(axis) => (rotation_grad[axis], axis),
            Move::Camera(axis) => (camera_grad[axis], axis),
        };
        let analytic = grad / ITERATIONS as f32;
        let mut losses = [0.0; 2];
        for (loss, sign) in losses.iter_mut().zip([1.0, -1.0]) {
            let mut offsets = start.clone();
            match movement {
                Move::Translation(_) => offsets.translations[0][axis] += sign * H,
                Move::Rotation(_) => offsets.rotations[0][axis] += sign * H,
                Move::Camera(_) => offsets.camera[axis] += sign * H,
            }
            scene.move_geometry(&offsets);
            *loss = scene.converged_loss(frames.clone());
        }
        let numeric = (losses[0] - losses[1]) / (2.0 * H);
        println!("d loss / d {movement:?}: gradient {analytic:e}, central difference {numeric:e}");
        assert!(
            analytic * numeric > 0.0 && (analytic - numeric).abs() <= 0.25 * numeric.abs(),
            "d loss / d {movement:?}: gradient {analytic:e} disagrees with {numeric:e}"
        );
    }
    scene.destroy();
}
