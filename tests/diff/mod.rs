//! Differentiable rendering of the material grid.

use super::{PbrHarness, SHADER_DIR, pbr_scene};
use blade_graphics as gpu;
use blade_render::diff::{
    AdamConfig, DiffConfig, DiffLight, DiffShaders, DiffTracer, DiffTracerDesc, GeometryBase,
    GeometryOffsets, LightParamFlags, MaterialParamFlags, MaterialParams, StepConfig,
};
use blade_render::large_steps::{LargeSteps, LargeStepsConfig};
use blade_render::marching::{SdfGrid, SdfOptimizer, SdfOptimizerConfig};

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
    /// The same sphere, with the positions of its vertices as parameters.
    Blob,
    /// A sphere just above the view, in front of a wall that it shades from
    /// part of the sky. The camera only sees it in the shadow, and the
    /// translations of both are parameters.
    Shadow,
    /// The wall of `Shadow` under a ceiling above the view, with the sphere
    /// between them. A light below the ceiling only lights it, so the
    /// camera sees the shadow of the sphere on it by a bounce off the wall.
    Ceiling,
    /// The wall and the sphere of `Shadow`, lit by a small emissive sphere
    /// where the point light of the tests of lights is.
    AreaLight,
    /// A small quad, an open mesh, whose translation is a parameter. Its
    /// silhouette is its boundary, whose edges only have one triangle.
    Panel,
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
        matches!(
            self,
            Self::Spheres
                | Self::Bowl
                | Self::Cube
                | Self::Ball
                | Self::Blob
                | Self::Panel
                | Self::Shadow
                | Self::Ceiling
                | Self::AreaLight
        )
    }

    fn texels(self) -> Vec<[u8; 4]> {
        let size = CHECKER_SIZE as usize;
        (0..size * size)
            .map(|i| match self {
                Self::BaseColorTexture => CHECKER[(i / size + i % size) % 2],
                Self::NormalMap => [
                    [200, 128, 255, 255],
                    [128, 200, 255, 255],
                    [56, 128, 255, 255],
                    [128, 56, 255, 255],
                ][(i / size + i % size) % 4],
                _ => unreachable!("the scene has no texture"),
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
    desc: DiffTracerDesc,
}

impl DiffScene {
    /// The material grid under `sky_texels`, with the textures that
    /// `kind` asks for, which are then parameters.
    fn new(cache_name: &str, kind: SceneKind) -> Option<Self> {
        Self::with_views(cache_name, kind, 1)
    }

    fn with_views(cache_name: &str, kind: SceneKind, views: u32) -> Option<Self> {
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
        let mut objects = if matches!(
            kind,
            SceneKind::Shadow | SceneKind::Ceiling | SceneKind::AreaLight
        ) {
            let room = if kind == SceneKind::Ceiling {
                vec![wall(), ceiling()]
            } else {
                vec![wall()]
            };
            let mut parts = vec![("diff-wall", room), ("diff-shading-ball", vec![small_ball()])];
            if kind == SceneKind::AreaLight {
                parts.push(("diff-area-light", vec![light_ball()]));
            }
            parts
                .into_iter()
                .map(|(name, geometries)| {
                    let model = harness
                        .asset_hub
                        .models
                        .baker
                        .create_model(name, geometries);
                    blade_render::Object::from(harness.asset_hub.models.insert(model))
                })
                .collect()
        } else if kind == SceneKind::Spheres {
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
            } else if kind == SceneKind::Ball || kind == SceneKind::Blob {
                vec![ball()]
            } else if kind == SceneKind::Panel {
                vec![panel()]
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
        let desc = DiffTracerDesc {
            textures: textured,
            geometry: kind.moves(),
            vertices: kind == SceneKind::Blob,
            views,
        };
        let diff = DiffTracer::new(
            &mut encoder,
            &context,
            &diff_shaders,
            &harness.asset_hub,
            &renderer,
            &desc,
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

            desc,
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
        self.diff.view_loss()
    }

    /// The loss of the expected image, without the part that the noise of
    /// the image adds: the product of the differences of two images from
    /// separate frames with the target.
    fn unbiased_loss(&mut self, frames: std::ops::Range<u32>) -> f32 {
        let size = self.diff.image_size();
        let count = (size.width * size.height) as usize;
        let half = frames.start + (frames.end - frames.start) / 2;
        let render = |scene: &mut Self, range: std::ops::Range<u32>| {
            // In chunks, which keeps the recorded commands short.
            let start = range.start;
            for chunk in range.clone().step_by(256) {
                for frame in chunk..(chunk + 256).min(range.end) {
                    scene.diff.render(
                        &mut scene.encoder,
                        &scene.renderer,
                        &scene.camera,
                        &scene.config,
                        frame,
                        frame != start,
                    );
                }
                scene.submit();
            }
            scene.download(scene.diff.image(), count)
        };
        let first = render(self, frames.start..half);
        let second = render(self, half..frames.end);
        let target = self.download(self.diff.target(), count);
        let mut sum = 0.0f64;
        for ((a, b), t) in first.iter().zip(&second).zip(&target) {
            for c in 0..3 {
                sum += f64::from(a[c] - t[c]) * f64::from(b[c] - t[c]);
            }
        }
        (sum / (3 * count) as f64) as f32
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

    fn set_lights(&mut self, lights: &[DiffLight]) {
        let context = std::sync::Arc::clone(&self.harness.context);
        self.diff
            .set_lights(&mut self.encoder, &context, lights, &mut self.temp);
        self.submit();
    }

    fn read_lights(&mut self) -> (Vec<DiffLight>, Vec<DiffLight>) {
        self.diff.read_back_lights(&mut self.encoder);
        self.submit();
        (
            self.diff.lights().to_vec(),
            self.diff.light_grads().to_vec(),
        )
    }

    /// Take the light of the sky out, leaving that of the lights.
    fn darken_sky(&mut self) {
        let size = self.diff.environment_size();
        let count = (size.width * size.height) as usize;
        let env = self.diff.environment_params();
        self.upload(env, &vec![[0.0, 0.0, 0.0, 1.0]; count]);
    }

    /// Offset the vertex slots of the blob by `offset` of their positions
    /// from where they were loaded, apply them, and build the scene again.
    fn shape_blob(&mut self, offset: impl Fn([f32; 3]) -> [f32; 3]) -> Vec<[f32; 4]> {
        self.diff.read_back_vertices(&mut self.encoder);
        self.submit();
        let offsets = self
            .diff
            .vertex_bases()
            .iter()
            .map(|base| {
                if base[3] == 0.0 {
                    return [0.0; 4];
                }
                let [x, y, z] = offset([base[0], base[1], base[2]]);
                [x, y, z, 0.0]
            })
            .collect::<Vec<_>>();
        let context = std::sync::Arc::clone(&self.harness.context);
        self.diff
            .write_vertex_offsets(&mut self.encoder, &context, &offsets, &mut self.temp);
        self.apply_vertices();
        offsets
    }

    /// Replace the model of the only object by `mesh`, with the same
    /// material, and start a tracer for it, which keeps the target.
    fn set_mesh(&mut self, mesh: &blade_render::remesh::Mesh) {
        let context = std::sync::Arc::clone(&self.harness.context);
        let old = self.objects[0].model;
        let material = &self.harness.asset_hub.models[old].materials[0];
        let mut geometry = mesh.to_geometry("diff-remeshed");
        geometry.base_color_factor = material.base_color_factor;
        geometry.metalness = material.metalness;
        geometry.roughness = material.roughness;
        geometry.emissive_factor = material.emissive_factor;
        let model = self
            .harness
            .asset_hub
            .models
            .baker
            .create_model("diff-remeshed", vec![geometry]);
        self.objects[0].model = self.harness.asset_hub.models.insert(model);
        self.harness
            .asset_hub
            .flush(&mut self.encoder, &mut self.temp.buffers);
        self.renderer.build_scene(
            &mut self.encoder,
            &mut self.objects,
            Some(self.sky),
            &self.harness.asset_hub,
            &context,
            &mut self.temp,
        );
        // Nothing that is recorded can use the buffers of the old tracer.
        self.submit();
        self.diff.rebuild(
            &mut self.encoder,
            &context,
            &self.harness.asset_hub,
            &self.renderer,
            &self.desc,
        );
        self.submit();
    }

    /// The mesh of the only object, as it is now.
    fn current_mesh(&mut self) -> blade_render::remesh::Mesh {
        self.diff.read_back_meshes(&mut self.encoder);
        self.submit();
        let mut meshes = self.diff.current_meshes(&self.harness.asset_hub);
        assert_eq!(meshes.len(), 1);
        meshes.pop().unwrap().mesh
    }

    /// Write the offsets of the vertices into the model, and build the scene
    /// again.
    fn apply_vertices(&mut self) {
        let context = std::sync::Arc::clone(&self.harness.context);
        self.diff.apply_vertices(&mut self.encoder);
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

    /// Step the offsets of the vertices by `optimizer`, on the host, which
    /// starts from the offsets and the triangles as they are if it's `None`.
    fn large_step(&mut self, optimizer: &mut Option<LargeSteps>, config: &LargeStepsConfig) {
        self.diff.gather_vertex_grads(&mut self.encoder);
        self.diff.read_back_meshes(&mut self.encoder);
        self.submit();
        let optimizer = optimizer.get_or_insert_with(|| {
            let offsets = self
                .diff
                .vertex_offsets()
                .iter()
                .map(|o| glam::Vec3::new(o[0], o[1], o[2]))
                .collect();
            let triangles = self.diff.slot_triangles(&self.harness.asset_hub);
            LargeSteps::new(offsets, &triangles, config.lambda)
        });
        let grads: Vec<_> = self
            .diff
            .vertex_grads()
            .iter()
            .map(|g| glam::Vec3::new(g[0], g[1], g[2]))
            .collect();
        optimizer.step(&grads, config);
        let offsets: Vec<_> = optimizer
            .offsets()
            .iter()
            .map(|o| [o.x, o.y, o.z, 0.0])
            .collect();
        // Clears the gradients.
        self.diff.step(&mut self.encoder, &StepConfig::default());
        let context = std::sync::Arc::clone(&self.harness.context);
        self.diff
            .write_vertex_offsets(&mut self.encoder, &context, &offsets, &mut self.temp);
        self.apply_vertices();
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

    // Pixels are tent filtered, which spreads the noise of the loss over
    // more paths, and the gradient by metalness, the smallest, needs more.
    const ITERATIONS: u32 = 512;
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

/// A quad that leaves the sky around it in view, turned a little.
///
/// Its edges are slanted in the image: an edge along the pixel grid keeps
/// the same place within the pixels it crosses, all along, and then the loss
/// changes its slope every time the edge moves across a pixel, which central
/// differences over a step close to a pixel average out.
fn panel() -> blade_render::ProceduralGeometry {
    let mut geometry = quad();
    let (sin_z, cos_z) = 0.35f32.sin_cos();
    let (sin_y, cos_y) = 0.3f32.sin_cos();
    let turn = |[x, y, z]: [f32; 3]| {
        let (x, y) = (x * cos_z - y * sin_z, x * sin_z + y * cos_z);
        [x * cos_y + z * sin_y, y, -x * sin_y + z * cos_y]
    };
    for vertex in &mut geometry.vertices {
        vertex.position = turn(vertex.position.map(|c| 0.4 * c));
        vertex.normal = pbr_scene::encode_normal(turn([0.0, 0.0, 1.0]));
        vertex.tangent = pbr_scene::encode_normal(turn([1.0, 0.0, 0.0]));
    }
    geometry
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
/// turned so that the camera sees three of them. The turn about the view
/// keeps its edges off the pixel grid, see `panel`.
fn cube() -> blade_render::ProceduralGeometry {
    cuboid([1.2; 3], [0.0; 3], [0.6, 0.4, 0.35], [0.8, 0.5, 0.3, 1.0])
}

/// A wall behind the middle of the view, facing the camera.
fn wall() -> blade_render::ProceduralGeometry {
    cuboid(
        [4.0, 3.0, 0.2],
        [0.0, 0.0, -1.6],
        [0.0; 3],
        [0.7, 0.7, 0.7, 1.0],
    )
}

/// A slab above the view and the sphere, over the wall and in front of it.
fn ceiling() -> blade_render::ProceduralGeometry {
    cuboid(
        [4.0, 0.1, 2.2],
        [0.0, 6.6, 0.6],
        [0.0; 3],
        [0.7, 0.7, 0.7, 1.0],
    )
}

/// Where the emissive sphere of `SceneKind::AreaLight` is, and its radius.
const AREA_LIGHT_CENTER: [f32; 3] = [0.0, 8.0, -0.2];
const AREA_LIGHT_RADIUS: f32 = 0.1;

/// A small sphere, that emits about as much as a point light of 800 at its
/// center: its radiance times the area of its disk.
fn light_ball() -> blade_render::ProceduralGeometry {
    let mut geometry = ball();
    // `ball` has a radius of 1.5.
    let scale = AREA_LIGHT_RADIUS / 1.5;
    for vertex in &mut geometry.vertices {
        let [x, y, z] = vertex.position.map(|c| c * scale);
        vertex.position = [
            x + AREA_LIGHT_CENTER[0],
            y + AREA_LIGHT_CENTER[1],
            z + AREA_LIGHT_CENTER[2],
        ];
    }
    let radiance = 800.0 / (std::f32::consts::PI * AREA_LIGHT_RADIUS * AREA_LIGHT_RADIUS);
    geometry.base_color_factor = [0.0, 0.0, 0.0, 1.0];
    geometry.emissive_factor = [radiance; 3];
    geometry
}

/// The dielectric sphere of the grid, a little larger, above the view and
/// close to the top of the wall.
fn small_ball() -> blade_render::ProceduralGeometry {
    let mut geometry = ball();
    for vertex in &mut geometry.vertices {
        let [x, y, z] = vertex.position.map(|c| c * 0.5);
        vertex.position = [x, y + 4.2, z - 0.8];
    }
    geometry
}

/// A box with flat faces, turned about Y, then X, then Z by `turn`.
fn cuboid(
    half: [f32; 3],
    center: [f32; 3],
    turn: [f32; 3],
    color: [f32; 4],
) -> blade_render::ProceduralGeometry {
    let (sin_y, cos_y) = turn[0].sin_cos();
    let (sin_x, cos_x) = turn[1].sin_cos();
    let (sin_z, cos_z) = turn[2].sin_cos();
    let turn = |v: [f32; 3]| {
        let [x, y, z] = v;
        let (x, z) = (x * cos_y + z * sin_y, -x * sin_y + z * cos_y);
        let (y, z) = (y * cos_x - z * sin_x, y * sin_x + z * cos_x);
        let (x, y) = (x * cos_z - y * sin_z, x * sin_z + y * cos_z);
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
                let position = [0, 1, 2].map(|i| half[i] * (normal[i] + a * u[i] + b * v[i]));
                let [x, y, z] = turn(position);
                vertices.push(blade_render::Vertex {
                    position: [x + center[0], y + center[1], z + center[2]],
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
        name: "cuboid".to_string(),
        vertices,
        indices,
        base_color_factor: color,
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
        SceneKind::NormalMap => (0.05, 320),
        _ => (0.03, 100),
    };
    for iteration in 0..steps {
        // Noisy gradients keep a constant rate wandering about the optimum.
        let adam = AdamConfig {
            learning_rate: learning_rate * 0.99f32.powi(iteration as i32),
            ..Default::default()
        };
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
    check_single_object(
        SceneKind::Cube,
        "diff-cube",
        &ALL_MOVES,
        [-0.08, 0.12, -0.06],
    );
}

/// The boundary edges of an open mesh have one triangle each, which has to
/// sample them.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_open_mesh_gradients_match_finite_differences() {
    // The camera along Z barely changes the loss here.
    check_single_object(SceneKind::Panel, "diff-panel", &ALL_MOVES[..4], [0.0; 3]);
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

/// Carrying the hits of the paths onto the edges of their triangles finds
/// the same gradients, of the edges of the cube and of the outline of the
/// sphere, without visiting every edge.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_projective_edges_match_finite_differences() {
    check_single_object_with(
        SceneKind::Cube,
        "diff-projective-cube",
        &ALL_MOVES,
        [-0.08, 0.12, -0.06],
        true,
    );
    check_single_object_with(
        SceneKind::Ball,
        "diff-projective-ball",
        &[Move::Translation(0), Move::Translation(1), Move::Camera(2)],
        [0.0; 3],
        true,
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
    check_single_object_with(kind, cache_name, moves, rotation, false);
}

fn check_single_object_with(
    kind: SceneKind,
    cache_name: &str,
    moves: &[Move],
    rotation: [f32; 3],
    projective: bool,
) {
    let Some(mut scene) = DiffScene::new(cache_name, kind) else {
        return;
    };
    scene.config.edge_samples = 4;
    scene.config.projective_edges = projective;
    scene.capture_target();
    // Off along Z too, or the gradient by Z would be about zero.
    let start = DiffScene::single_offsets([0.3, -0.2, 0.6], rotation, [0.0; 3]);
    scene.move_geometry(&start);
    const ITERATIONS: u32 = 128;
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
    let mut results = Vec::new();
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
        results.push((movement, analytic, numeric));
    }
    // The noise of the gradient is about the same for every axis, so one that
    // is small next to the others is held to the size of the largest.
    let largest = results.iter().map(|&(_, _, n)| n.abs()).fold(0.0, f32::max);
    for (movement, analytic, numeric) in results {
        assert!(
            (analytic - numeric).abs() <= 0.25 * numeric.abs() + 0.05 * largest,
            "d loss / d {movement:?}: gradient {analytic:e} disagrees with {numeric:e}"
        );
    }
    scene.destroy();
}

/// Moving a sphere that the camera doesn't see only moves the shadow that it
/// casts on a wall, which takes sampling the edges that the light at the
/// wall comes past.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_shadow_gradients_match_finite_differences() {
    // Direct light only.
    check_shadow_gradients("diff-shadow", 0, 1, 0.0);
}

/// With bounces, the light that the sphere reflects onto the wall moves
/// with it too, along with the shadows that it casts on what the wall
/// reflects.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_indirect_gradients_match_finite_differences() {
    check_shadow_gradients("diff-indirect", 2, 2, 0.0);
}

/// An emissive sphere out of view is the only light of the wall, which the
/// camera sees move with it: the light that it casts is a boundary, like a
/// shadow, which is all that moves.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_emitter_gradients_match_finite_differences() {
    check_shadow_gradients("diff-emitter", 1, 1, 4.0);
}

/// The shadow on the wall moves with the sphere, as with drawing the edges
/// from the whole scene, when the edges are found by carrying the hits of
/// directions drawn by the BSDF of the wall onto them. With bounces, this
/// agrees with drawing edges too, though too noisily to test as quickly.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_projective_shadows_match_finite_differences() {
    check_shadow_gradients_with("diff-projective-shadow", 0, 1, 0.0, true);
}

/// With bounces, the shadows on what the wall reflects move too, and the
/// edges of the sphere as seen from the hits of the bounced paths.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_projective_indirect_gradients_match_finite_differences() {
    check_shadow_gradients_with("diff-projective-indirect", 2, 2, 0.0, true);
}

/// With the sphere emitting `emission`.
fn check_shadow_gradients(cache_name: &str, bounces: u32, depth: u32, emission: f32) {
    check_shadow_gradients_with(cache_name, bounces, depth, emission, false);
}

fn check_shadow_gradients_with(
    cache_name: &str,
    bounces: u32,
    depth: u32,
    emission: f32,
    projective: bool,
) {
    let Some(mut scene) = DiffScene::new(cache_name, SceneKind::Shadow) else {
        return;
    };
    const BALL: usize = 1;
    scene.config.edge_samples = 4;
    scene.config.max_bounces = bounces;
    scene.config.secondary_edge_depth = depth;
    if emission > 0.0 {
        let (mut params, _) = scene.read_materials();
        assert_eq!(params.len(), 2);
        params[BALL].emissive = [emission; 3];
        scene.write_materials(&params);
        // The sphere is the only light.
        {
            let size = scene.diff.environment_size();
            let count = (size.width * size.height) as usize;
            let env = scene.diff.environment_params();
            scene.upload(env, &vec![[0.0, 0.0, 0.0, 1.0]; count]);
        }
    }
    scene.capture_target();
    let mut start = GeometryOffsets {
        translations: vec![[0.0; 3]; 2],
        rotations: vec![[0.0; 3]; 2],
        camera: [0.0; 3],
    };
    start.translations[BALL] = [0.3, 0.25, 0.0];
    scene.move_geometry(&start);

    const ITERATIONS: u32 = 256;
    let mut grads = Vec::new();
    for secondary in [0, 32] {
        // Few of the directions from the wall come to the sphere, which
        // each costs less than an edge drawn from the scene.
        scene.config.secondary_edge_samples = if projective { 4 * secondary } else { secondary };
        scene.config.projective_edges = projective && secondary != 0;
        // Guided by what the edges brought in a few rounds first.
        scene.config.guided_edges = secondary != 0;
        if secondary != 0 {
            for round in 0..4 {
                for iteration in 0..8 {
                    scene.backward(ITERATIONS + 8 * round + iteration, 4, 2);
                }
                scene
                    .diff
                    .refresh_edge_guide(&mut scene.encoder, &scene.renderer);
            }
            scene.diff.step(&mut scene.encoder, &StepConfig::default());
        }
        for iteration in 0..ITERATIONS {
            scene.backward(iteration, 8, 4);
            if secondary != 0 && iteration % 16 == 15 {
                scene
                    .diff
                    .refresh_edge_guide(&mut scene.encoder, &scene.renderer);
            }
        }
        scene.diff.read_back(&mut scene.encoder);
        scene.submit();
        grads.push(scene.diff.translation_grads()[BALL]);
        // Clear the gradients for the next round.
        scene.diff.step(&mut scene.encoder, &StepConfig::default());
    }

    const H: f32 = 0.05;
    let frames = 2_000_000..2_002_048;
    let [without_grad, with_grad] = [grads[0], grads[1]];
    let mut results = Vec::new();
    for axis in 0..2 {
        let mut losses = [0.0; 2];
        for (loss, sign) in losses.iter_mut().zip([1.0, -1.0]) {
            let mut offsets = start.clone();
            offsets.translations[BALL][axis] += sign * H;
            scene.move_geometry(&offsets);
            // The light of the wall is found by few paths, and the noise
            // that it leaves in the image adds to the loss, and moves with
            // the sphere, which tilts central differences of it.
            *loss = scene.unbiased_loss(frames.clone());
        }
        let numeric = (losses[0] - losses[1]) / (2.0 * H);
        let without = without_grad[axis] / ITERATIONS as f32;
        let with = with_grad[axis] / ITERATIONS as f32;
        println!(
            "d loss / d ball {axis}: {without:e} without shadows, {with:e} with, central difference {numeric:e}"
        );
        results.push((axis, without, with, numeric));
    }
    // Few of the edges drawn from the scene are silhouettes from the wall,
    // which leaves this noisier than the others, about the same for both
    // axes, so the smaller one is also held to the size of the larger.
    let largest = results.iter().map(|r| r.3.abs()).fold(0.0, f32::max);
    let mut interreflection = 0.0f32;
    for (axis, without, with, numeric) in results {
        assert!(
            (with - numeric).abs() <= 0.3 * numeric.abs() + 0.1 * largest,
            "d loss / d ball {axis}: gradient {with:e} disagrees with {numeric:e}"
        );
        // With bounces, the light that the sphere reflects moves with it
        // without the edges.
        assert!(
            bounces > 0 || without.abs() < 0.1 * numeric.abs(),
            "d loss / d ball {axis}: {without:e} without shadows, against {numeric:e}"
        );
        interreflection = interreflection.max(without.abs() / largest);
    }
    if bounces > 0 && emission == 0.0 {
        assert!(
            interreflection > 0.2,
            "the light reflected by the sphere barely moves with it"
        );
    }
    scene.destroy();
}

/// Scale the blob about its center along X by `stretch`, as offsets.
fn stretch(stretch: f32) -> impl Fn([f32; 3]) -> [f32; 3] {
    move |[x, _, _]| [x * stretch, 0.0, 0.0]
}

/// The gradient by the positions of the vertices agrees with central
/// differences, along the direction of stretching the blob.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_vertex_gradients_match_finite_differences() {
    let Some(mut scene) = DiffScene::new("diff-blob", SceneKind::Blob) else {
        return;
    };
    scene.config.edge_samples = 4;
    scene.capture_target();
    const STRETCH: f32 = 0.15;
    let direction = scene.shape_blob(stretch(1.0));
    scene.shape_blob(stretch(STRETCH));
    const ITERATIONS: u32 = 128;
    for iteration in 0..ITERATIONS {
        scene.backward(iteration, 8, 4);
    }
    scene.diff.gather_vertex_grads(&mut scene.encoder);
    scene.diff.read_back_vertices(&mut scene.encoder);
    scene.submit();
    // Along the stretch, by its amount.
    let analytic = scene
        .diff
        .vertex_grads()
        .iter()
        .zip(&direction)
        .map(|(g, d)| g[0] * d[0] + g[1] * d[1] + g[2] * d[2])
        .sum::<f32>()
        / ITERATIONS as f32;
    // The normals are stored in 8 bits, which a step has to be well past.
    const H: f32 = 0.05;
    let frames = 2_000_000..2_000_128;
    let mut losses = [0.0; 2];
    for (loss, sign) in losses.iter_mut().zip([1.0, -1.0]) {
        scene.shape_blob(stretch(STRETCH + sign * H));
        *loss = scene.converged_loss(frames.clone());
    }
    let numeric = (losses[0] - losses[1]) / (2.0 * H);
    println!("d loss / d stretch: gradient {analytic:e}, central difference {numeric:e}");
    assert!(
        analytic * numeric > 0.0 && (analytic - numeric).abs() <= 0.25 * numeric.abs(),
        "d loss / d stretch: gradient {analytic:e} disagrees with {numeric:e}"
    );
    scene.destroy();
}

/// Optimizing the positions of the vertices brings a stretched blob back to
/// the sphere that it was.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_recovers_shape() {
    let Some(mut scene) = DiffScene::new("diff-shape", SceneKind::Blob) else {
        return;
    };
    scene.config.edge_samples = 2;
    scene.capture_target();
    const STRETCH: f32 = 0.25;
    scene.shape_blob(stretch(STRETCH));
    // How far the vertices are from the sphere of the target, on average.
    // Sliding along the surface barely changes the image, so that isn't
    // held against them.
    let measure = |scene: &mut DiffScene| {
        scene.diff.read_back_vertices(&mut scene.encoder);
        scene.submit();
        let (mut sum, mut count) = (0.0, 0.0);
        for (base, offset) in scene
            .diff
            .vertex_bases()
            .iter()
            .zip(scene.diff.vertex_offsets())
        {
            if base[3] != 0.0 {
                let p = [0, 1, 2].map(|c| base[c] + offset[c]);
                let radius = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                let target = (base[0] * base[0] + base[1] * base[1] + base[2] * base[2]).sqrt();
                sum += (radius - target).abs();
                count += 1.0;
            }
        }
        sum / count
    };
    let initial = measure(&mut scene);
    for iteration in 0..120 {
        scene.backward(iteration, 4, 2);
        let adam = AdamConfig {
            learning_rate: 0.01 * 0.98f32.powi(iteration as i32),
            ..Default::default()
        };
        scene.diff.step(
            &mut scene.encoder,
            &StepConfig {
                vertices: Some((adam, 1.0e-2)),
                ..Default::default()
            },
        );
        scene.apply_vertices();
    }
    let error = measure(&mut scene);
    println!("distance from the sphere {initial:.4} -> {error:.4}");
    assert!(
        error < 0.3 * initial,
        "the distance from the sphere only went from {initial} to {error}"
    );
    scene.destroy();
}

/// How far the outline of `mesh`, its extents across the view as the camera
/// sees them, is from that of the ellipsoid that `stretch` makes of the
/// blob, on average, in the units of the scene at its center. A single view
/// leaves the depth of the shape, and its back, to the smoothness.
fn outline_error(mesh: &blade_render::remesh::Mesh, stretch: f32) -> f32 {
    const RADIUS: f32 = 1.5;
    let distance = pbr_scene::camera().pos.z;
    let extents = |points: &mut dyn Iterator<Item = [f32; 3]>| {
        let mut extents = [f32::MIN; 4];
        for p in points {
            let scale = distance / (distance - p[2]);
            let projected = [p[0] * scale, p[1] * scale];
            extents[0] = extents[0].max(projected[0]);
            extents[1] = extents[1].max(-projected[0]);
            extents[2] = extents[2].max(projected[1]);
            extents[3] = extents[3].max(-projected[1]);
        }
        extents
    };
    let target = extents(&mut (0..64 * 128).map(|i| {
        let theta = std::f32::consts::PI * ((i / 128) as f32 + 0.5) / 64.0;
        let phi = std::f32::consts::TAU * (i % 128) as f32 / 128.0;
        [
            RADIUS * (1.0 + stretch) * theta.sin() * phi.cos(),
            RADIUS * theta.cos(),
            RADIUS * theta.sin() * phi.sin(),
        ]
    }));
    let current = extents(&mut mesh.positions.iter().copied());
    (0..4).map(|i| (current[i] - target[i]).abs()).sum::<f32>() / 4.0
}

/// The result of fitting a mesh to the stretched blob.
struct Fit {
    loss: f32,
    outline: f32,
    triangles: usize,
}

/// Fit an icosahedron to a stretched sphere, remeshing it between rounds of
/// optimizing its vertices to each of `edge_lengths`, or not at all.
fn fit_icosahedron(cache_name: &str, edge_lengths: &[f32]) -> Option<Fit> {
    const STRETCH: f32 = 0.3;
    let mut scene = DiffScene::new(cache_name, SceneKind::Blob)?;
    scene.config.edge_samples = 4;
    scene.shape_blob(stretch(STRETCH));
    scene.capture_target();
    scene.set_mesh(&blade_render::remesh::icosahedron(1.5));
    let mut iteration = 0;
    // Steps that are a fraction of the edges, which get shorter.
    let mut rate = 0.02;
    for round in 0..=edge_lengths.len() {
        if round > 0 {
            let length = edge_lengths[round - 1];
            let mesh = scene.current_mesh().remesh(length, 4);
            assert!(mesh.is_manifold());
            scene.set_mesh(&mesh);
            rate = 0.025 * length;
        }
        for step in 0..40 {
            scene.backward(iteration, 8, 4);
            iteration += 1;
            // Every remeshing starts the optimizer over, whose first steps go
            // as far as later ones on less of a gradient, so they ramp up.
            let warm_up = ((step + 1) as f32 / 10.0).min(1.0);
            let adam = AdamConfig {
                learning_rate: rate * warm_up * 0.96f32.powi(step),
                ..Default::default()
            };
            scene.diff.step(
                &mut scene.encoder,
                &StepConfig {
                    vertices: Some((adam, 1.0e-2)),
                    ..Default::default()
                },
            );
            scene.apply_vertices();
        }
    }
    let loss = scene.converged_loss(3_000_000..3_000_064);
    let mesh = scene.current_mesh();
    let fit = Fit {
        loss,
        outline: outline_error(&mesh, STRETCH),
        triangles: mesh.triangles.len(),
    };
    scene.destroy();
    Some(fit)
}

/// Remeshing between rounds of optimization lets a coarse mesh refine as it
/// takes the shape, which it can't with its own few triangles.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_remeshing_refines_the_fit() {
    let start = outline_error(&blade_render::remesh::icosahedron(1.5), 0.3);
    let Some(fixed) = fit_icosahedron("diff-fit-fixed", &[]) else {
        return;
    };
    // The same number of steps, in rounds between remeshing.
    let Some(remeshed) = fit_icosahedron("diff-fit-remeshed", &[0.8, 0.5, 0.35, 0.35]) else {
        return;
    };
    println!(
        "outline {start:.3} at the start; fixed: loss {:e}, outline {:.3}, {} triangles; \
         remeshed: loss {:e}, outline {:.3}, {} triangles",
        fixed.loss,
        fixed.outline,
        fixed.triangles,
        remeshed.loss,
        remeshed.outline,
        remeshed.triangles,
    );
    assert!(
        remeshed.loss < 0.6 * fixed.loss,
        "remeshing only got the loss to {}, against {} without",
        remeshed.loss,
        fixed.loss
    );
    assert!(
        remeshed.outline < 0.3 * start,
        "the outline only went from {start} to {}",
        remeshed.outline
    );
    assert!(remeshed.triangles > 10 * fixed.triangles);
}

/// The camera of the grid, and the same from the right, from above, from
/// behind and from below, all looking at the center.
fn view_cameras() -> [blade_render::Camera; 5] {
    let front = pbr_scene::camera();
    let distance = front.pos.z;
    let half = std::f32::consts::FRAC_1_SQRT_2;
    let side = blade_render::Camera {
        pos: [distance, 0.0, 0.0].into(),
        rot: mint::Quaternion {
            v: [0.0, half, 0.0].into(),
            s: half,
        },
        ..front
    };
    let top = blade_render::Camera {
        pos: [0.0, distance, 0.0].into(),
        rot: mint::Quaternion {
            v: [-half, 0.0, 0.0].into(),
            s: half,
        },
        ..front
    };
    let back = blade_render::Camera {
        pos: [0.0, 0.0, -distance].into(),
        rot: mint::Quaternion {
            v: [0.0, 1.0, 0.0].into(),
            s: 0.0,
        },
        ..front
    };
    let bottom = blade_render::Camera {
        pos: [0.0, -distance, 0.0].into(),
        rot: mint::Quaternion {
            v: [half, 0.0, 0.0].into(),
            s: half,
        },
        ..front
    };
    [front, side, top, back, bottom]
}

/// How the vertices of the blob are optimized by [`fit_depth`].
#[derive(Clone, Copy, PartialEq)]
enum VertexOptimizer {
    /// Adam on every offset, with smoothness.
    Adam,
    /// [`LargeSteps`] on the host.
    LargeSteps,
}

/// The outcome of [`fit_depth`].
struct DepthFit {
    /// How much deeper along Z the blob became, where the target is 0.3.
    stretch: f32,
    /// Sum of the losses of the views, of converged images.
    loss: f32,
}

/// Fit the blob to itself stretched along the view of the front camera,
/// from the first `views` of [`view_cameras`].
fn fit_depth(cache_name: &str, views: u32, optimizer: VertexOptimizer) -> Option<DepthFit> {
    const STRETCH: f32 = 0.3;
    let mut scene = DiffScene::with_views(cache_name, SceneKind::Blob, views)?;
    scene.config.edge_samples = 2;
    let cameras = view_cameras();
    scene.shape_blob(|[_, _, z]| [0.0, 0.0, z * STRETCH]);
    for view in 0..views {
        scene.diff.select_view(view);
        scene.camera = cameras[view as usize];
        scene.capture_target();
    }
    scene.shape_blob(|_| [0.0; 3]);
    let mut large_steps = None;
    for iteration in 0..160 {
        for view in 0..views {
            scene.diff.select_view(view);
            scene.camera = cameras[view as usize];
            scene.backward(iteration * views + view, 4, 2);
        }
        let decay = 0.985f32.powi(iteration as i32);
        match optimizer {
            VertexOptimizer::Adam => {
                let adam = AdamConfig {
                    learning_rate: 0.04 * decay,
                    ..Default::default()
                };
                scene.diff.step(
                    &mut scene.encoder,
                    &StepConfig {
                        vertices: Some((adam, 1.0e-2)),
                        ..Default::default()
                    },
                );
                scene.apply_vertices();
            }
            VertexOptimizer::LargeSteps => {
                // The steps are of the largest change, which spreads.
                let config = LargeStepsConfig {
                    learning_rate: 0.2 * decay,
                    ..Default::default()
                };
                scene.large_step(&mut large_steps, &config);
            }
        }
    }
    scene.diff.read_back_vertices(&mut scene.encoder);
    scene.submit();
    let stretch = depth_stretch(&scene.diff);
    let mut loss = 0.0;
    for view in 0..views {
        scene.diff.select_view(view);
        scene.camera = cameras[view as usize];
        loss += scene.converged_loss(TARGET_FRAMES);
    }
    scene.destroy();
    Some(DepthFit { stretch, loss })
}

/// How much deeper along Z the blob is than it was loaded.
fn depth_stretch(diff: &DiffTracer) -> f32 {
    let (mut near, mut far) = (f32::MAX, f32::MIN);
    let (mut base_near, mut base_far) = (f32::MAX, f32::MIN);
    for (base, offset) in diff.vertex_bases().iter().zip(diff.vertex_offsets()) {
        if base[3] != 0.0 {
            near = near.min(base[2] + offset[2]);
            far = far.max(base[2] + offset[2]);
            base_near = base_near.min(base[2]);
            base_far = base_far.max(base[2]);
        }
    }
    (far - near) / (base_far - base_near) - 1.0
}

/// One view can't tell how deep a shape is, which more views from around it
/// can, with their gradients adding up.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_views_recover_depth() {
    let Some(single) = fit_depth("diff-depth-single", 1, VertexOptimizer::Adam) else {
        return;
    };
    let Some(multiple) = fit_depth("diff-depth-multiple", 3, VertexOptimizer::Adam) else {
        return;
    };
    println!(
        "stretch along the view of 0.3 found as {:.3} from one view, {:.3} from three",
        single.stretch, multiple.stretch
    );
    assert!(
        single.stretch.abs() < 0.05,
        "one view found a stretch of {}",
        single.stretch
    );
    assert!(
        (multiple.stretch - 0.3).abs() < 0.1,
        "three views found a stretch of {}",
        multiple.stretch
    );
}

/// The steps of [`LargeSteps`] move the whole surface smoothly, which gets
/// closer to the target than Adam on every vertex, without smoothing that
/// pulls the surface away from it.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_large_steps_fit_closer() {
    let Some(adam) = fit_depth("diff-steps-adam", 3, VertexOptimizer::Adam) else {
        return;
    };
    let Some(large) = fit_depth("diff-steps-large", 3, VertexOptimizer::LargeSteps) else {
        return;
    };
    println!(
        "stretch of 0.3 found as {:.3} by Adam, {:.3} by large steps, with losses {:.2e} and {:.2e}",
        adam.stretch, large.stretch, adam.loss, large.loss
    );
    assert!(
        (large.stretch - 0.3).abs() < 0.03,
        "large steps found a stretch of {}",
        large.stretch
    );
    assert!(
        large.loss < 0.3 * adam.loss,
        "large steps got to a loss of {}, against {} of Adam",
        large.loss,
        adam.loss
    );
}

/// Drawing the edges for shadows by a guide made of what they brought
/// before gets the same gradient as drawing them uniformly, with less
/// noise, since few of the edges cast the shadow.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_guided_edges_reduce_variance() {
    let Some(mut scene) = DiffScene::new("diff-guided", SceneKind::Shadow) else {
        return;
    };
    const BALL: usize = 1;
    const BATCHES: usize = 16;
    const BATCH: u32 = 8;
    scene.config.edge_samples = 4;
    scene.config.max_bounces = 0;
    scene.config.secondary_edge_samples = 8;
    scene.capture_target();
    let mut start = GeometryOffsets {
        translations: vec![[0.0; 3]; 2],
        rotations: vec![[0.0; 3]; 2],
        camera: [0.0; 3],
    };
    start.translations[BALL] = [0.3, 0.25, 0.0];
    scene.move_geometry(&start);

    let mut iteration = 0;
    let mut stats = Vec::new();
    for guided in [false, true] {
        scene.config.guided_edges = guided;
        if guided {
            for _ in 0..4 {
                for _ in 0..BATCH {
                    scene.backward(iteration, 4, 2);
                    iteration += 1;
                }
                scene.diff.step(&mut scene.encoder, &StepConfig::default());
                scene
                    .diff
                    .refresh_edge_guide(&mut scene.encoder, &scene.renderer);
            }
        }
        let mut batches = Vec::new();
        for _ in 0..BATCHES {
            for _ in 0..BATCH {
                scene.backward(iteration, 4, 2);
                iteration += 1;
            }
            scene.diff.read_back(&mut scene.encoder);
            scene.submit();
            let grad = scene.diff.translation_grads()[BALL];
            batches.push([grad[0] / BATCH as f32, grad[1] / BATCH as f32]);
            scene.diff.step(&mut scene.encoder, &StepConfig::default());
            if guided {
                scene
                    .diff
                    .refresh_edge_guide(&mut scene.encoder, &scene.renderer);
            }
        }
        let mut mean = [0.0f32; 2];
        let mut variance = [0.0f32; 2];
        for axis in 0..2 {
            mean[axis] = batches.iter().map(|b| b[axis]).sum::<f32>() / BATCHES as f32;
            variance[axis] = batches
                .iter()
                .map(|b| (b[axis] - mean[axis]).powi(2))
                .sum::<f32>()
                / (BATCHES - 1) as f32;
        }
        println!(
            "{}: mean {mean:?}, deviation {:?}",
            if guided { "guided" } else { "uniform" },
            variance.map(f32::sqrt)
        );
        stats.push((mean, variance));
    }
    let [
        (uniform_mean, uniform_variance),
        (guided_mean, guided_variance),
    ] = [stats[0], stats[1]];
    for axis in 0..2 {
        let error = ((uniform_variance[axis] + guided_variance[axis]) / BATCHES as f32).sqrt();
        assert!(
            (uniform_mean[axis] - guided_mean[axis]).abs() < 3.0 * error,
            "axis {axis}: the guided mean {} is off the uniform {}",
            guided_mean[axis],
            uniform_mean[axis]
        );
    }
    let total = |v: [f32; 2]| v[0] + v[1];
    assert!(
        total(guided_variance) < 0.5 * total(uniform_variance),
        "the variance went from {} to {} with the guide",
        total(uniform_variance),
        total(guided_variance)
    );
    scene.destroy();
}

/// Pairing the paths with their twins turned about the normal at the first
/// hit gets the same gradient by the texels of a normal map, with less
/// noise for as many paths.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_antithetic_paths_reduce_variance() {
    let Some(mut scene) = DiffScene::new("diff-antithetic", SceneKind::NormalMap) else {
        return;
    };
    const BATCHES: usize = 16;
    const BATCH: u32 = 4;
    let texel_count = (CHECKER_SIZE * CHECKER_SIZE) as usize;
    let params = scene.diff.texture_params();
    scene.capture_target();
    let truth = scene.download(params, texel_count);
    scene.upload(params, &vec![average_color(&truth); texel_count]);

    let mut iteration = 0;
    let mut stats = Vec::new();
    for antithetic in [false, true] {
        scene.config.antithetic = antithetic;
        let mut batches = Vec::new();
        for _ in 0..BATCHES {
            for _ in 0..BATCH {
                scene.backward(iteration, 4, 2);
                iteration += 1;
            }
            let grads = scene.download(scene.diff.texture_grads(), texel_count);
            batches.push(grads.iter().flat_map(|g| [g[0], g[1]]).collect::<Vec<_>>());
            scene.diff.step(&mut scene.encoder, &StepConfig::default());
        }
        let count = batches[0].len();
        let mean: Vec<f32> = (0..count)
            .map(|k| batches.iter().map(|b| b[k]).sum::<f32>() / BATCHES as f32)
            .collect();
        let variance: Vec<f32> = (0..count)
            .map(|k| {
                batches
                    .iter()
                    .map(|b| (b[k] - mean[k]).powi(2))
                    .sum::<f32>()
                    / (BATCHES - 1) as f32
            })
            .collect();
        stats.push((mean, variance));
    }
    let (independent_mean, independent_variance) = &stats[0];
    let (antithetic_mean, antithetic_variance) = &stats[1];
    let total = |v: &[f32]| v.iter().sum::<f32>();
    // How far the means are apart, in units of their expected spread.
    let mut spread = 0.0;
    let mut expected = 0.0;
    for k in 0..independent_mean.len() {
        spread += (independent_mean[k] - antithetic_mean[k]).powi(2);
        expected += (independent_variance[k] + antithetic_variance[k]) / BATCHES as f32;
    }
    println!(
        "variance of the texel gradients {:e} independent, {:e} antithetic; means apart by {:.2} of their spread",
        total(independent_variance),
        total(antithetic_variance),
        spread / expected
    );
    assert!(
        spread < 2.0 * expected,
        "the means are apart by {} of their spread",
        spread / expected
    );
    assert!(
        total(antithetic_variance) < 0.7 * total(independent_variance),
        "antithetic paths left the variance at {} from {}",
        total(antithetic_variance),
        total(independent_variance)
    );
    scene.destroy();
}

/// The values of a signed distance grid, through the vertices of the mesh
/// of its zero level, find two spheres from one capsule around both, which
/// takes a change of topology that moving the vertices of one mesh can't.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_signed_distance_splits_the_shape() {
    // These see the gap between the spheres, and all around them.
    const VIEWS: [u32; 4] = [0, 2, 3, 4];
    let Some(mut scene) = DiffScene::with_views("diff-sdf", SceneKind::Blob, 5) else {
        return;
    };
    scene.config.edge_samples = 2;
    let cameras = view_cameras();
    let grid_at = |distance: &dyn Fn(glam::Vec3) -> f32| {
        SdfGrid::new(
            [25, 15, 15],
            glam::Vec3::new(-2.4, -1.4, -1.4),
            0.2,
            distance,
        )
    };
    let spheres = grid_at(&|p| {
        let left = (p - glam::Vec3::new(-1.1, 0.0, 0.0)).length();
        let right = (p - glam::Vec3::new(1.1, 0.0, 0.0)).length();
        left.min(right) - 0.8
    });
    scene.set_mesh(&spheres.extract().mesh);
    for view in VIEWS {
        scene.diff.select_view(view);
        scene.camera = cameras[view as usize];
        scene.capture_target();
    }

    let mut grid = grid_at(&|p| {
        let axis = glam::Vec3::new(p.x.clamp(-1.1, 1.1), 0.0, 0.0);
        (p - axis).length() - 0.7
    });
    let mut optimizer = SdfOptimizer::new(&grid);
    let start = blade_render::marching::components(&grid.extract().mesh);
    let mut losses = Vec::new();
    for iteration in 0..80u32 {
        let extraction = grid.extract();
        scene.set_mesh(&extraction.mesh);
        for (k, &view) in VIEWS.iter().enumerate() {
            scene.diff.select_view(view);
            scene.camera = cameras[view as usize];
            scene.backward(iteration * VIEWS.len() as u32 + k as u32, 4, 2);
        }
        scene.diff.gather_vertex_grads(&mut scene.encoder);
        scene.diff.read_back_meshes(&mut scene.encoder);
        scene.submit();
        losses.push(scene.diff.loss());
        // The slot of every vertex of the mesh, by the corners of its
        // triangles, which the model has in the same order.
        let slot_triangles = scene.diff.slot_triangles(&scene.harness.asset_hub);
        assert_eq!(slot_triangles.len(), extraction.mesh.triangles.len());
        let mut slots = vec![u32::MAX; extraction.mesh.positions.len()];
        for (t, s) in extraction.mesh.triangles.iter().zip(&slot_triangles) {
            for k in 0..3 {
                slots[t[k] as usize] = s[k];
            }
        }
        let vertex_grads = scene.diff.vertex_grads();
        let position_grads: Vec<glam::Vec3> = slots
            .iter()
            .map(|&slot| match vertex_grads.get(slot as usize) {
                Some(g) => glam::Vec3::new(g[0], g[1], g[2]),
                None => glam::Vec3::ZERO,
            })
            .collect();
        let mut value_grads = vec![0.0; grid.values.len()];
        extraction.backpropagate(&grid, &position_grads, &mut value_grads);
        optimizer.step(
            &mut grid,
            &value_grads,
            &SdfOptimizerConfig {
                learning_rate: 0.04 * 0.98f32.powi(iteration as i32),
                smoothness: 0.1,
                limit: 2.0,
                ..Default::default()
            },
        );
        // Clear the gradients.
        scene.diff.step(&mut scene.encoder, &StepConfig::default());
    }
    let end = blade_render::marching::components(&grid.extract().mesh);
    let early = losses[..8].iter().sum::<f32>() / 8.0;
    let late = losses[losses.len() - 8..].iter().sum::<f32>() / 8.0;
    println!("parts {start:?} -> {end:?}, loss {early:.3e} -> {late:.3e}");
    scene.destroy();
    let closed_spheres = |parts: &[blade_render::marching::Component]| {
        parts.iter().all(|p| p.euler_characteristic == 2)
    };
    assert!(start.len() == 1 && closed_spheres(&start));
    assert!(
        end.len() == 2 && closed_spheres(&end),
        "the shape didn't split into two spheres"
    );
    for part in &end {
        let [x, y, z] = part.center;
        assert!(
            (x.abs() - 1.1).abs() < 0.1 && y.abs() < 0.1 && z.abs() < 0.1,
            "a part is at {:?}",
            part.center
        );
    }
    assert!(late < 0.5 * early);
}

/// Drawing points on an emitter finds the light that it casts with much
/// less noise than finding it by the BSDF alone, and the same on average.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_emitter_sampling_reduces_noise() {
    let Some(mut scene) = DiffScene::new("diff-emitter-sampling", SceneKind::Shadow) else {
        return;
    };
    scene.config.max_bounces = 1;
    let (mut params, _) = scene.read_materials();
    params[1].emissive = [4.0; 3];
    scene.write_materials(&params);
    let size = scene.diff.environment_size();
    let env = scene.diff.environment_params();
    scene.upload(
        env,
        &vec![[0.0, 0.0, 0.0, 1.0]; (size.width * size.height) as usize],
    );
    let image_size = scene.diff.image_size();
    let count = (image_size.width * image_size.height) as usize;
    let mut results = Vec::new();
    for samples in [0, 1] {
        scene.config.emitter_samples = samples;
        let mut halves = Vec::new();
        for half in 0..2u32 {
            scene.render_converged(half * 64..(half + 1) * 64);
            scene.submit();
            halves.push(scene.download(scene.diff.image(), count));
        }
        // The difference of two images with separate noise.
        let noise = halves[0]
            .iter()
            .zip(&halves[1])
            .map(|(a, b)| (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f32>())
            .sum::<f32>()
            / (6 * count) as f32;
        let mean = halves
            .iter()
            .flatten()
            .map(|t| t[0] + t[1] + t[2])
            .sum::<f32>()
            / (6 * count) as f32;
        results.push((mean, noise));
    }
    let [(bsdf_mean, bsdf_noise), (light_mean, light_noise)] = [results[0], results[1]];
    println!(
        "mean radiance {bsdf_mean:.4} by the BSDF, {light_mean:.4} with the emitters drawn; variance {bsdf_noise:.3e} and {light_noise:.3e}"
    );
    // The variance is of a pixel of either half, and the colors are gray.
    let spread = ((bsdf_noise + light_noise) / (2 * count) as f32).sqrt();
    assert!(
        (bsdf_mean - light_mean).abs() < 4.0 * spread + 1.0e-3 * bsdf_mean,
        "the means are {bsdf_mean} and {light_mean}"
    );
    assert!(light_noise < 0.2 * bsdf_noise);
    scene.destroy();
}

/// The recursive control variate of the image takes much of the noise of
/// its image out of the gradient by the base colors, which keeps its mean.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_control_variate_reduces_variance() {
    let Some(mut scene) = DiffScene::new("diff-control", SceneKind::Grid) else {
        return;
    };
    const ITERATIONS: u32 = 48;
    scene.capture_target();
    let (truth, _) = scene.read_materials();
    let mut params = truth.clone();
    for params in &mut params[..pbr_scene::COLUMNS] {
        params.base_color = [0.4, 0.4, 0.4];
    }
    scene.write_materials(&params);
    let mut iteration = 0;
    let mut stats = Vec::new();
    for weight in [0.0, 0.9] {
        scene.config.control_variate = weight;
        let mut samples = Vec::new();
        for k in 0..8 + ITERATIONS {
            scene.backward(iteration, 2, 2);
            iteration += 1;
            let (_, grads) = scene.read_materials();
            // Clear the gradients.
            scene.diff.step(&mut scene.encoder, &StepConfig::default());
            if k >= 8 {
                samples.push(
                    grads[..pbr_scene::COLUMNS]
                        .iter()
                        .flat_map(|g| g.base_color)
                        .collect::<Vec<_>>(),
                );
            }
        }
        let count = samples[0].len();
        let mean: Vec<f32> = (0..count)
            .map(|k| samples.iter().map(|s| s[k]).sum::<f32>() / samples.len() as f32)
            .collect();
        let variance: Vec<f32> = (0..count)
            .map(|k| {
                samples
                    .iter()
                    .map(|s| (s[k] - mean[k]).powi(2))
                    .sum::<f32>()
                    / (samples.len() - 1) as f32
            })
            .collect();
        stats.push((mean, variance));
    }
    let (plain_mean, plain_variance) = &stats[0];
    let (control_mean, control_variance) = &stats[1];
    let total = |v: &[f32]| v.iter().sum::<f32>();
    let apart: f32 = plain_mean
        .iter()
        .zip(control_mean)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    // The control variate averages the noise of the images over time, so
    // its mean is about as far off as that of the plain one.
    let expected = 2.0 * total(plain_variance) / ITERATIONS as f32;
    println!(
        "variance of the base color gradients {:e} plain, {:e} with the control variate; means apart by {:.2} of their spread",
        total(plain_variance),
        total(control_variance),
        apart / expected
    );
    assert!(
        apart < 3.0 * expected,
        "the means are apart by {}",
        apart / expected
    );
    assert!(total(control_variance) < 0.5 * total(plain_variance));

    // As the parameters change, it still finds them.
    let error = |params: &[MaterialParams]| {
        let mut sum = 0.0;
        for (p, t) in params.iter().zip(&truth).take(pbr_scene::COLUMNS) {
            for c in 0..3 {
                sum += (p.base_color[c] - t.base_color[c]).abs();
            }
        }
        sum / (3 * pbr_scene::COLUMNS) as f32
    };
    let mut errors = Vec::new();
    for weight in [0.0, 0.9] {
        scene.config.control_variate = weight;
        scene.write_materials(&params);
        for _ in 0..40 {
            scene.backward(iteration, 2, 2);
            iteration += 1;
            scene.diff.step(
                &mut scene.encoder,
                &StepConfig {
                    materials: Some((
                        AdamConfig {
                            learning_rate: 0.03,
                            ..Default::default()
                        },
                        MaterialParamFlags::BASE_COLOR,
                    )),
                    ..Default::default()
                },
            );
            scene.submit();
        }
        let (found, _) = scene.read_materials();
        errors.push(error(&found));
    }
    println!(
        "base color error {:.3} -> {:.3} plain, {:.3} with the control variate",
        error(&params),
        errors[0],
        errors[1]
    );
    assert!(errors[1] < 0.25 * error(&params));
    scene.destroy();
}

/// A panel that lets part of the sky behind it through, as a thin wall: the
/// gradient by how much agrees with central differences, and optimizing it
/// finds how much.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_recovers_transmission() {
    let Some(mut scene) = DiffScene::new("diff-transmission", SceneKind::Panel) else {
        return;
    };
    const TRUTH: f32 = 0.6;
    let (mut params, _) = scene.read_materials();
    params[0].transmission = TRUTH;
    scene.write_materials(&params);
    scene.capture_target();
    params[0].transmission = 0.2;
    scene.write_materials(&params);

    const ITERATIONS: u32 = 128;
    for iteration in 0..ITERATIONS {
        scene.backward(iteration, 8, 4);
    }
    let (_, grads) = scene.read_materials();
    let analytic = grads[0].transmission / ITERATIONS as f32;
    const H: f32 = 0.05;
    let frames = 2_000_000..2_000_256;
    let mut losses = [0.0; 2];
    for (loss, sign) in losses.iter_mut().zip([1.0, -1.0]) {
        params[0].transmission = 0.2 + sign * H;
        scene.write_materials(&params);
        *loss = scene.converged_loss(frames.clone());
    }
    let numeric = (losses[0] - losses[1]) / (2.0 * H);
    println!("d loss / d transmission: gradient {analytic:e}, central difference {numeric:e}");
    assert!(
        analytic * numeric > 0.0 && (analytic - numeric).abs() <= 0.15 * numeric.abs(),
        "d loss / d transmission: gradient {analytic:e} disagrees with {numeric:e}"
    );

    params[0].transmission = 0.2;
    scene.write_materials(&params);
    // Clear the gradients.
    scene.diff.step(&mut scene.encoder, &StepConfig::default());
    for iteration in 0..100 {
        scene.backward(ITERATIONS + iteration, 4, 2);
        scene.diff.step(
            &mut scene.encoder,
            &StepConfig {
                materials: Some((
                    AdamConfig {
                        learning_rate: 0.04 * 0.99f32.powi(iteration as i32),
                        ..Default::default()
                    },
                    MaterialParamFlags::TRANSMISSION,
                )),
                ..Default::default()
            },
        );
        scene.submit();
    }
    let (found, _) = scene.read_materials();
    println!("transmission 0.2 -> {} for {TRUTH}", found[0].transmission);
    assert!((found[0].transmission - TRUTH).abs() < 0.05);
    scene.destroy();
}

/// A ball of glass refracts the sky behind it, entering and leaving it: the
/// gradient by its index of refraction agrees with central differences,
/// and optimizing it finds the index.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_recovers_index_of_refraction() {
    let Some(mut scene) = DiffScene::new("diff-refraction", SceneKind::Ball) else {
        return;
    };
    const TRUTH: f32 = 1.5;
    const START: f32 = 1.3;
    let (mut params, _) = scene.read_materials();
    params[0].transmission = 1.0;
    params[0].ior = TRUTH;
    scene.write_materials(&params);
    scene.capture_target();
    params[0].ior = START;
    scene.write_materials(&params);

    // The gradient through two refractions is noisy.
    const ITERATIONS: u32 = 512;
    // In batches, which add up on the host in double precision.
    let mut total = 0.0f64;
    for batch in 0..ITERATIONS / 32 {
        for iteration in 0..32 {
            scene.backward(batch * 32 + iteration, 8, 4);
        }
        let (_, grads) = scene.read_materials();
        total += f64::from(grads[0].ior);
        scene.diff.step(&mut scene.encoder, &StepConfig::default());
    }
    let analytic = (total / f64::from(ITERATIONS)) as f32;
    // The paths depend on the index through their refraction, so the
    // images on both sides don't share their noise, which takes many
    // frames, and the loss of the expected image.
    let h = 0.05;
    let frames = 2_000_000..2_004_096;
    let mut losses = [0.0; 2];
    for (loss, sign) in losses.iter_mut().zip([1.0, -1.0]) {
        params[0].ior = START + sign * h;
        scene.write_materials(&params);
        *loss = scene.unbiased_loss(frames.clone());
    }
    let numeric = (losses[0] - losses[1]) / (2.0 * h);
    println!("d loss / d ior: gradient {analytic:e}, central difference {numeric:e}");
    assert!(
        analytic * numeric > 0.0 && (analytic - numeric).abs() <= 0.2 * numeric.abs(),
        "d loss / d ior: gradient {analytic:e} disagrees with {numeric:e}"
    );

    params[0].ior = START;
    scene.write_materials(&params);
    // Clear the gradients.
    scene.diff.step(&mut scene.encoder, &StepConfig::default());
    for iteration in 0..100 {
        scene.backward(ITERATIONS + iteration, 4, 2);
        scene.diff.step(
            &mut scene.encoder,
            &StepConfig {
                materials: Some((
                    AdamConfig {
                        learning_rate: 0.01 * 0.98f32.powi(iteration as i32),
                        ..Default::default()
                    },
                    MaterialParamFlags::IOR,
                )),
                ..Default::default()
            },
        );
        scene.submit();
    }
    let (found, _) = scene.read_materials();
    println!(
        "index of refraction {START} -> {} for {TRUTH}",
        found[0].ior
    );
    assert!((found[0].ior - TRUTH).abs() < 0.05);
    scene.destroy();
}

/// The control variate of the image keeps up with moving objects, by
/// rendering the previous scene, which the ray tracer keeps: the gradient
/// by the translation of a sphere that moves a little every step keeps its
/// mean, with less noise.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_control_variate_follows_motion() {
    let Some(mut scene) = DiffScene::new("diff-control-motion", SceneKind::Ball) else {
        return;
    };
    const ITERATIONS: u32 = 48;
    // More samples of the gradient than of the image, whose noise the
    // control variate takes out.
    scene.config.edge_samples = 8;
    scene.capture_target();
    let start = [0.3, -0.2, 0.6];
    let mut iteration = 0;
    let mut stats = Vec::new();
    for weight in [0.0, 0.9] {
        scene.config.control_variate = weight;
        let mut samples = Vec::new();
        for k in 0..8 + ITERATIONS {
            // Back and forth, as optimizing it would.
            let jitter = if k % 2 == 0 { 0.002 } else { -0.002 };
            let offsets = DiffScene::single_offsets(
                [start[0] + jitter, start[1], start[2]],
                [0.0; 3],
                [0.0; 3],
            );
            scene.move_geometry(&offsets);
            scene.backward(iteration, 1, 8);
            iteration += 1;
            scene.diff.read_back(&mut scene.encoder);
            scene.submit();
            let grad = scene.diff.translation_grads()[0];
            scene.diff.step(&mut scene.encoder, &StepConfig::default());
            if k >= 8 {
                samples.push([grad[0], grad[1]]);
            }
        }
        let count = samples.len() as f32;
        let mean = [0, 1].map(|c| samples.iter().map(|s| s[c]).sum::<f32>() / count);
        let variance = [0, 1].map(|c| {
            samples
                .iter()
                .map(|s| (s[c] - mean[c]).powi(2))
                .sum::<f32>()
                / (count - 1.0)
        });
        println!(
            "control variate {weight}: mean {mean:?}, deviation {:?}",
            variance.map(f32::sqrt)
        );
        stats.push((mean, variance));
    }
    let [
        (plain_mean, plain_variance),
        (control_mean, control_variance),
    ] = [stats[0], stats[1]];
    let total = |v: [f32; 2]| v[0] + v[1];
    let apart =
        (plain_mean[0] - control_mean[0]).powi(2) + (plain_mean[1] - control_mean[1]).powi(2);
    // The estimates of the control variate average the noise of the images
    // over time, so its mean is about as far off as that of the plain one.
    let expected = 2.0 * total(plain_variance) / ITERATIONS as f32;
    println!("means apart by {:.2} of their spread", apart / expected);
    assert!(
        apart < 3.0 * expected,
        "the means are apart by {}",
        apart / expected
    );
    assert!(total(control_variance) < 0.7 * total(plain_variance));
    scene.destroy();
}

/// A parameter of the scene of the lights, for central differences.
#[derive(Clone, Copy, Debug, PartialEq)]
enum LightParam {
    Ball(usize),
    Position(usize),
    Intensity,
    Direction(usize),
}

/// The scene of the shadow, lit by `light` alone, which the sphere shades
/// the wall from, with the sphere moved by `ball`.
fn set_light_scene(scene: &mut DiffScene, ball: [f32; 3], light: DiffLight) {
    let mut offsets = GeometryOffsets {
        translations: vec![[0.0; 3]; 2],
        rotations: vec![[0.0; 3]; 2],
        camera: [0.0; 3],
    };
    offsets.translations[1] = ball;
    scene.move_geometry(&offsets);
    scene.set_lights(&[light]);
}

/// The gradients by the light, and by the sphere, which only moves the
/// shadow that the light casts, against central differences of the loss.
fn check_light_gradients(
    scene: &mut DiffScene,
    target: DiffLight,
    start: DiffLight,
    params: &[(LightParam, f32)],
) {
    check_light_gradients_with(scene, target, start, params, &[], 8);
}

/// Like `check_light_gradients`, and also the gradients by `alone` without
/// the secondary edge samples, which have to disagree with the central
/// differences: the shadows that they find are what makes up the gradient.
fn check_light_gradients_with(
    scene: &mut DiffScene,
    target: DiffLight,
    start: DiffLight,
    params: &[(LightParam, f32)],
    alone: &[LightParam],
    primal_samples: u32,
) {
    const BALL: usize = 1;
    scene.darken_sky();
    set_light_scene(scene, [0.0; 3], target);
    scene.capture_target();
    let ball = [0.3, 0.25, 0.0];
    set_light_scene(scene, ball, start);

    const ITERATIONS: u32 = 256;
    // The gradients only add up until an optimizer step reads them, so
    // those without the secondary edge samples come first, and are taken
    // out of the sum after them.
    // The sums after every batch, of which the spread tells the noise.
    const BATCH: u32 = 32;
    let guided = scene.config.guided_edges;
    let read = |scene: &mut DiffScene| {
        scene.diff.read_back(&mut scene.encoder);
        scene.submit();
        let ball_grad = scene.diff.translation_grads()[BALL];
        let (_, light_grads) = scene.read_lights();
        (ball_grad, light_grads[0])
    };
    let gradients = |scene: &mut DiffScene, iterations: std::ops::Range<u32>| {
        let mut sums = Vec::new();
        for iteration in iterations {
            scene.backward(iteration, primal_samples, 4);
            if guided && iteration % 16 == 15 {
                scene
                    .diff
                    .refresh_edge_guide(&mut scene.encoder, &scene.renderer);
            }
            if (iteration + 1) % BATCH == 0 {
                sums.push(read(scene));
            }
        }
        sums
    };
    // The guide comes from what the edges brought in a few rounds first,
    // whose gradients are left out.
    let mut warm = None;
    if guided {
        for round in 0..4 {
            for iteration in 0..8 {
                scene.backward(3 * ITERATIONS + 8 * round + iteration, 4, 2);
            }
            scene
                .diff
                .refresh_edge_guide(&mut scene.encoder, &scene.renderer);
        }
        warm = Some(read(scene));
    }
    let analytic_of = |(ball_grad, light_grad): ([f32; 4], DiffLight), param| match param {
        LightParam::Ball(axis) => ball_grad[axis],
        LightParam::Position(axis) => light_grad.position[axis],
        LightParam::Intensity => light_grad.intensity[0],
        LightParam::Direction(axis) => light_grad.direction[axis],
    };
    let mut without = None;
    if !alone.is_empty() {
        let samples = scene.config.secondary_edge_samples;
        scene.config.secondary_edge_samples = 0;
        without = Some(gradients(scene, ITERATIONS..2 * ITERATIONS));
        scene.config.secondary_edge_samples = samples;
    }
    let clock = std::time::Instant::now();
    let with = gradients(scene, 0..ITERATIONS);
    println!(
        "{ITERATIONS} gradient iterations in {:.1} s",
        clock.elapsed().as_secs_f32()
    );
    let zero = ([0.0; 4], DiffLight::point([0.0; 3], [0.0; 3]));

    let frames = 2_000_000..2_002_048;
    let mut results = Vec::new();
    for &(param, h) in params {
        // The mean of the batches since `last`, and its standard error.
        let mean_of = |sums: &[([f32; 4], DiffLight)], mut last: f32| {
            let batches = sums
                .iter()
                .map(|&sums| {
                    let sum = analytic_of(sums, param);
                    let batch = (sum - last) / BATCH as f32;
                    last = sum;
                    batch
                })
                .collect::<Vec<_>>();
            let count = batches.len() as f32;
            let mean = batches.iter().sum::<f32>() / count;
            let spread = batches.iter().map(|b| (b - mean).powi(2)).sum::<f32>() / (count - 1.0);
            (mean, (spread / count).sqrt(), last)
        };
        let warmed = analytic_of(warm.unwrap_or(zero), param);
        let (before, before_error, last) = match &without {
            Some(sums) => mean_of(sums, warmed),
            None => (0.0, 0.0, warmed),
        };
        let (analytic, error, _) = mean_of(&with, last);
        let mut losses = [0.0; 2];
        for (loss, sign) in losses.iter_mut().zip([1.0, -1.0]) {
            let mut moved = ball;
            let mut light = start;
            match param {
                LightParam::Ball(axis) => moved[axis] += sign * h,
                LightParam::Position(axis) => light.position[axis] += sign * h,
                LightParam::Intensity => light.intensity[0] += sign * h,
                LightParam::Direction(axis) => light.direction[axis] += sign * h,
            }
            set_light_scene(scene, moved, light);
            *loss = scene.unbiased_loss(frames.clone());
        }
        let numeric = (losses[0] - losses[1]) / (2.0 * h);
        if without.is_some() {
            println!(
                "d loss / d {param:?}: {analytic:e} ± {error:e}, {before:e} ± {before_error:e} without secondary edges, central difference {numeric:e}"
            );
        } else {
            println!("d loss / d {param:?}: {analytic:e} ± {error:e}, central difference {numeric:e}");
        }
        results.push((param, analytic, before, numeric));
    }
    let largest = results.iter().map(|r| r.3.abs()).fold(0.0, f32::max);
    let close = |gradient: f32, numeric: f32| {
        (gradient - numeric).abs() <= 0.25 * numeric.abs() + 0.05 * largest
    };
    for (param, analytic, before, numeric) in results {
        assert!(
            close(analytic, numeric),
            "d loss / d {param:?}: gradient {analytic:e} disagrees with {numeric:e}"
        );
        if alone.iter().any(|&p| p == param) {
            assert!(
                !close(before, numeric),
                "d loss / d {param:?}: {before:e} without secondary edges is already {numeric:e}"
            );
        }
    }
}

/// A point light high above the wall, past the sphere, which the camera
/// sees by the shadow on the wall alone. Moving the light or the sphere
/// moves the shadow, and the light also lights the wall by its distance
/// and its direction.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_point_light_gradients_match_finite_differences() {
    let Some(mut scene) = DiffScene::new("diff-point-light", SceneKind::Shadow) else {
        return;
    };
    scene.config.max_bounces = 0;
    scene.config.edge_samples = 4;
    check_light_gradients(
        &mut scene,
        DiffLight::point([0.0, 8.0, -0.2], [800.0; 3]),
        DiffLight::point([0.3, 7.7, 0.1], [700.0, 800.0, 900.0]),
        &[
            (LightParam::Ball(0), 0.05),
            (LightParam::Ball(1), 0.05),
            (LightParam::Position(0), 0.1),
            (LightParam::Position(1), 0.1),
            (LightParam::Position(2), 0.1),
            (LightParam::Intensity, 20.0),
        ],
    );
    scene.destroy();
}

/// A spot light, whose cone fades out on the wall, through a bounce: moving
/// it and turning it moves the cone and the shadow of the sphere, which
/// also brings light to the wall from where the spot light lights it.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_spot_light_gradients_match_finite_differences() {
    let Some(mut scene) = DiffScene::new("diff-spot-light", SceneKind::Shadow) else {
        return;
    };
    scene.config.max_bounces = 1;
    scene.config.edge_samples = 4;
    scene.config.secondary_edge_samples = 8;
    let spot = |position: [f32; 3], direction: [f32; 3]| {
        DiffLight::spot(position, [800.0; 3], direction, 0.1, 0.25)
    };
    check_light_gradients(
        &mut scene,
        spot([0.0, 8.0, -0.2], [0.0, -7.6, -1.2]),
        spot([0.2, 7.8, 0.0], [0.3, -7.6, -1.0]),
        &[
            (LightParam::Ball(0), 0.05),
            (LightParam::Position(0), 0.1),
            (LightParam::Position(1), 0.1),
            (LightParam::Direction(0), 0.1),
            (LightParam::Direction(2), 0.1),
        ],
    );
    scene.destroy();
}

/// A spot light under the ceiling, which only lights the ceiling, and the
/// camera only sees the wall, by the light that the ceiling sends it. The
/// shadow of the sphere on the ceiling moves with the light and the sphere,
/// which only the edges drawn at the hits on the wall find: from there, the
/// silhouette of the sphere is seen from the light on another surface.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_indirect_light_shadow_gradients_match_finite_differences() {
    let Some(mut scene) = DiffScene::new("diff-indirect-light", SceneKind::Ceiling) else {
        return;
    };
    scene.config.max_bounces = 1;
    scene.config.edge_samples = 2;
    scene.config.secondary_edge_samples = 16;
    // Few of the edges of the scene are silhouettes from the light.
    scene.config.guided_edges = true;
    let spot = |position: [f32; 3]| {
        DiffLight::spot(position, [800.0; 3], [0.0, 1.0, 0.0], 0.35, 0.5)
    };
    check_light_gradients_with(
        &mut scene,
        spot([0.0, 2.0, -0.2]),
        spot([0.2, 2.1, -0.1]),
        &[
            (LightParam::Ball(0), 0.05),
            (LightParam::Ball(2), 0.05),
            (LightParam::Position(0), 0.1),
            (LightParam::Position(2), 0.1),
        ],
        &[LightParam::Ball(2)],
        32,
    );
    scene.destroy();
}

/// The scene of `diff_point_light_gradients_match_finite_differences`, lit
/// by a small emissive sphere instead, by the points drawn on it, which
/// cast the shadow of the sphere like a point light. The edges drawn at the
/// hits on the wall find it too, as the discontinuity of the light from the
/// directions to the emitter, but rarely, as it's small.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_area_light_gradients_match_finite_differences() {
    let Some(mut scene) = DiffScene::new("diff-area-light", SceneKind::AreaLight) else {
        return;
    };
    const BALL: usize = 1;
    const LIGHT: usize = 2;
    scene.config.max_bounces = 0;
    scene.config.edge_samples = 4;
    scene.config.emitter_samples = 1;
    scene.config.emitters_by_area = true;
    scene.darken_sky();
    let place = |scene: &mut DiffScene, ball: [f32; 3], light: [f32; 3]| {
        let mut offsets = GeometryOffsets {
            translations: vec![[0.0; 3]; 3],
            rotations: vec![[0.0; 3]; 3],
            camera: [0.0; 3],
        };
        offsets.translations[BALL] = ball;
        offsets.translations[LIGHT] = [0, 1, 2].map(|i| light[i] - AREA_LIGHT_CENTER[i]);
        scene.move_geometry(&offsets);
    };
    place(&mut scene, [0.0; 3], AREA_LIGHT_CENTER);
    scene.capture_target();
    let ball = [0.3, 0.25, 0.0];
    let light = [0.3, 7.7, 0.1];
    place(&mut scene, ball, light);

    const ITERATIONS: u32 = 256;
    const BATCH: u32 = 32;
    let mut sums = Vec::new();
    let clock = std::time::Instant::now();
    for iteration in 0..ITERATIONS {
        scene.backward(iteration, 8, 4);
        if (iteration + 1) % BATCH == 0 {
            scene.diff.read_back(&mut scene.encoder);
            scene.submit();
            let grads = scene.diff.translation_grads();
            sums.push([grads[BALL], grads[LIGHT]]);
        }
    }
    println!(
        "{ITERATIONS} gradient iterations in {:.1} s",
        clock.elapsed().as_secs_f32()
    );

    let params = [
        (LightParam::Ball(0), 0.05),
        (LightParam::Ball(1), 0.05),
        (LightParam::Position(0), 0.1),
        (LightParam::Position(1), 0.1),
        (LightParam::Position(2), 0.1),
    ];
    let frames = 2_000_000..2_002_048;
    let mut results = Vec::new();
    for (param, h) in params {
        let (object, axis) = match param {
            LightParam::Ball(axis) => (0, axis),
            LightParam::Position(axis) => (1, axis),
            _ => unreachable!(),
        };
        let mut last = 0.0;
        let batches = sums
            .iter()
            .map(|sum| {
                let batch = (sum[object][axis] - last) / BATCH as f32;
                last = sum[object][axis];
                batch
            })
            .collect::<Vec<_>>();
        let count = batches.len() as f32;
        let analytic = batches.iter().sum::<f32>() / count;
        let spread = batches.iter().map(|b| (b - analytic).powi(2)).sum::<f32>() / (count - 1.0);
        let error = (spread / count).sqrt();
        let mut losses = [0.0; 2];
        for (loss, sign) in losses.iter_mut().zip([1.0, -1.0]) {
            let mut moved = [ball, light];
            moved[object][axis] += sign * h;
            place(&mut scene, moved[0], moved[1]);
            *loss = scene.unbiased_loss(frames.clone());
        }
        let numeric = (losses[0] - losses[1]) / (2.0 * h);
        println!("d loss / d {param:?}: {analytic:e} ± {error:e}, central difference {numeric:e}");
        results.push((param, analytic, numeric));
    }
    let largest = results.iter().map(|r| r.2.abs()).fold(0.0, f32::max);
    for (param, analytic, numeric) in results {
        assert!(
            (analytic - numeric).abs() <= 0.25 * numeric.abs() + 0.05 * largest,
            "d loss / d {param:?}: gradient {analytic:e} disagrees with {numeric:e}"
        );
    }
    scene.destroy();
}

/// Optimizing the position and the intensity of a point light brings them
/// back to those that lit the target, by the light on the wall and the
/// shadow of the sphere on it.
#[test]
#[ignore = "requires a working GPU context with ray tracing and float32 atomics"]
fn diff_recovers_point_light() {
    let Some(mut scene) = DiffScene::new("diff-recover-light", SceneKind::Shadow) else {
        return;
    };
    scene.config.max_bounces = 0;
    scene.config.edge_samples = 2;
    scene.darken_sky();
    let target = DiffLight::point([0.0, 8.0, -0.2], [800.0; 3]);
    scene.set_lights(&[target]);
    scene.capture_target();
    let start = DiffLight::point([0.6, 7.4, 0.3], [600.0; 3]);
    scene.set_lights(&[start]);
    let distance = |light: &DiffLight| {
        light
            .position
            .iter()
            .zip(target.position)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    };
    let step = StepConfig {
        lights: Some((
            AdamConfig {
                learning_rate: 0.02,
                ..Default::default()
            },
            LightParamFlags::POSITION,
        )),
        ..Default::default()
    };
    let intensity_step = StepConfig {
        lights: Some((
            AdamConfig {
                learning_rate: 20.0,
                ..Default::default()
            },
            LightParamFlags::INTENSITY,
        )),
        ..Default::default()
    };
    let mut first_loss = None;
    // The distance to the wall trades off with the intensity, but for the
    // shadow, which takes a while to settle.
    for iteration in 0..600 {
        scene.backward(iteration, 4, 2);
        // Positions and intensities take steps of their own scales, on
        // alternate iterations.
        let config = if iteration % 2 == 0 {
            &step
        } else {
            &intensity_step
        };
        scene.diff.step(&mut scene.encoder, config);
        if iteration % 100 == 0 {
            scene.submit();
            let loss = scene.diff.loss();
            first_loss.get_or_insert(loss);
            let (lights, _) = scene.read_lights();
            println!(
                "iteration {iteration}: loss {loss:e}, light at {:?}, intensity {:?}",
                lights[0].position, lights[0].intensity
            );
        }
    }
    scene.submit();
    let loss = scene.diff.loss();
    let (lights, _) = scene.read_lights();
    let light = lights[0];
    println!(
        "final: loss {loss:e}, light at {:?}, intensity {:?}",
        light.position, light.intensity
    );
    let start_distance = distance(&start);
    assert!(
        distance(&light) < 0.25 * start_distance,
        "the light ended {} away, from {start_distance}",
        distance(&light)
    );
    for c in 0..3 {
        assert!(
            (light.intensity[c] - target.intensity[c]).abs() < 0.1 * target.intensity[c],
            "intensity {c} ended at {}",
            light.intensity[c]
        );
    }
    assert!(
        loss < 0.1 * first_loss.unwrap(),
        "the loss went from {first_loss:?} to {loss}"
    );
    scene.destroy();
}
