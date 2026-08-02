//! Generate synthetic ground truth for relightable reconstruction.
//!
//! Reconstructing materials and lighting from images is ill posed: reproducing
//! the input views constrains the *product* of albedo and illumination, not the
//! split between them. Photographs cannot tell you whether a solver got the
//! split right, because nobody knows what the right answer was. Synthetic data
//! can, because the scene was authored: the albedo, the roughness, the normals
//! and the light are all known before a single image is rendered.
//!
//! So this writes, for one scene:
//!
//! - the radiance seen from N viewpoints under M different environments, as
//!   linear floating point, with the tone map off, so it is data and not a
//!   picture;
//! - the material and geometry the renderer used to produce it — albedo,
//!   roughness, shading normal, ray distance — read straight out of the
//!   G-buffer;
//! - a manifest tying the poses, the environments and the files together.
//!
//! Several environments is the point rather than a convenience. Under a single
//! illumination the decomposition is genuinely ambiguous and only priors break
//! the tie; under two or more it is close to determined. Having them one
//! re-render apart turns "how many lighting conditions does this need" into a
//! curve that can be measured instead of argued.
//!
//! Held-out illumination is what the data is *for*: train on some environments,
//! relight into one that was never seen, and compare against the render of it
//! that already exists here. Training-view error is not evidence — it is
//! satisfied exactly by baking all the light into the albedo.
//!
//! Run with:
//!   RELIGHT_OUT=/tmp/relight cargo test --release --test relight_data -- --ignored --nocapture
//!
//! Environment:
//!   RELIGHT_OUT     output directory (required)
//!   RELIGHT_VIEWS   camera positions around the scene (default 12)
//!   RELIGHT_FRAMES  paths accumulated per image (default 512)
//!   RELIGHT_SIZE    "WxH" (default 320x240)
//!   RELIGHT_BOUNCES path length, 0 for direct lighting only (default 4)
#![cfg(not(gles))]
#![allow(dead_code)]

mod pbr_scene;
mod snapshot;

use blade_graphics as gpu;

/// Equirectangular resolution of the authored environments.
const ENV_SIZE: (u32, u32) = (256, 128);
/// The composed radiance, untouched. `Capture::Hdr` in the snapshot tests.
const HDR_FORMAT: gpu::TextureFormat = gpu::TextureFormat::Rgba32Float;

fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn parsed<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    env_var(name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn output_size() -> gpu::Extent {
    let spec = env_var("RELIGHT_SIZE").unwrap_or_else(|| "320x240".to_string());
    let (w, h) = spec
        .split_once('x')
        .expect("RELIGHT_SIZE must look like WxH");
    gpu::Extent {
        width: w.parse().expect("bad width"),
        height: h.parse().expect("bad height"),
        depth: 1,
    }
}

// ---------------------------------------------------------------- environments

/// Direction an equirectangular texel looks at.
///
/// Mirrors `map_equirect_uv_to_dir` in the renderer's `env-light.inc.wgsl`. It
/// has to agree exactly, or the light in the manifest is not the light that was
/// rendered.
fn env_direction(u: f32, v: f32) -> glam::Vec3 {
    let yaw = std::f32::consts::PI * (0.5 - v);
    let pitch = 2.0 * std::f32::consts::PI * (u - 0.5);
    glam::Vec3::new(yaw.cos() * pitch.sin(), yaw.sin(), yaw.cos() * pitch.cos())
}

fn direction_from_angles(azimuth_deg: f32, elevation_deg: f32) -> glam::Vec3 {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.to_radians();
    glam::Vec3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos())
}

struct Environment {
    name: &'static str,
    /// Linear radiance arriving from a direction.
    radiance: Box<dyn Fn(glam::Vec3) -> [f32; 3]>,
}

/// A concentrated source over a dim sky.
///
/// The edge is softened over a few degrees: a hard disc in an 8-bit map
/// aliases into the importance sampling, and a slightly soft sun is the more
/// realistic of the two anyway.
fn sun(
    azimuth: f32,
    elevation: f32,
    color: [f32; 3],
    sky: [f32; 3],
) -> Box<dyn Fn(glam::Vec3) -> [f32; 3]> {
    let dir = direction_from_angles(azimuth, elevation);
    Box::new(move |d: glam::Vec3| {
        let cosine = d.dot(dir);
        let t = ((cosine - 0.9986) / 0.0012).clamp(0.0, 1.0);
        [
            sky[0] + (color[0] - sky[0]) * t,
            sky[1] + (color[1] - sky[1]) * t,
            sky[2] + (color[2] - sky[2]) * t,
        ]
    })
}

fn environments() -> Vec<Environment> {
    // Radiance, not display values. A real sun is orders of magnitude brighter
    // than the sky it sits in, and that contrast is the whole of what makes a
    // specular highlight identifiable: a lobe narrow enough to be worth
    // recovering only shows up when there is something bright and small for it
    // to reflect. Eight bit environments cannot carry it.
    vec![
        Environment {
            name: "uniform",
            radiance: Box::new(|_| [0.6, 0.6, 0.6]),
        },
        Environment {
            name: "sun-east",
            radiance: sun(60.0, 35.0, [120.0, 114.0, 102.0], [0.10, 0.12, 0.16]),
        },
        Environment {
            name: "sun-west",
            radiance: sun(-75.0, 25.0, [150.0, 135.0, 112.0], [0.10, 0.12, 0.16]),
        },
        Environment {
            // Overcast sky over dark ground: nearly all the light comes from
            // above, which is the case that makes a normal recoverable.
            name: "sky-dome",
            radiance: Box::new(|d: glam::Vec3| {
                let t = (0.5 * (d.y + 1.0)).clamp(0.0, 1.0).powf(0.7);
                let v = 0.04 + (0.85 - 0.04) * t;
                [v, v * 0.99, v * 0.95]
            }),
        },
        Environment {
            // Two tinted lobes from opposite sides. Colour is what breaks the
            // "red wall under white light" tie, so one condition has to carry
            // it or the ablation cannot see the difference.
            name: "studio",
            radiance: {
                let key = direction_from_angles(45.0, 20.0);
                let fill = direction_from_angles(-120.0, 10.0);
                Box::new(move |d: glam::Vec3| {
                    let kw = d.dot(key).max(0.0).powf(200.0);
                    let fw = d.dot(fill).max(0.0).powf(60.0);
                    [
                        0.03 + 40.0 * kw + 6.0 * fw,
                        0.03 + 34.0 * kw + 9.0 * fw,
                        0.04 + 24.0 * kw + 18.0 * fw,
                    ]
                })
            },
        },
    ]
}

/// Sample an environment into the float texels the renderer will light with.
fn bake_environment(env: &Environment) -> Vec<[f32; 4]> {
    let (width, height) = ENV_SIZE;
    let mut texels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let rgb = (env.radiance)(env_direction(u, v));
            texels.push([rgb[0], rgb[1], rgb[2], 1.0]);
        }
    }
    texels
}

/// A look at an environment, for eyes rather than for the solver.
///
/// Reinhard rather than a clamp, so a sun a hundred times brighter than the
/// sky is still distinguishable from one a thousand times brighter.
fn environment_preview(texels: &[[f32; 4]]) -> Vec<u8> {
    texels
        .iter()
        .flat_map(|texel| {
            let mut out = [255u8; 4];
            for i in 0..3 {
                let mapped = texel[i] / (1.0 + texel[i]);
                let encoded = if mapped <= 0.003_130_8 {
                    12.92 * mapped
                } else {
                    1.055 * mapped.powf(1.0 / 2.4) - 0.055
                };
                out[i] = (encoded.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            out
        })
        .collect()
}

// ----------------------------------------------------------------------- scene

/// The material grid on a floor.
///
/// The grid sweeps roughness across the columns and metalness down the rows, so
/// a solver that only works for rough dielectrics fails visibly in one corner
/// rather than uniformly. The floor is there because the interreflection it
/// adds is what separates a direct-lighting model from a converged render.
fn scene() -> Vec<blade_render::ProceduralGeometry> {
    let mut geometries = pbr_scene::material_spheres([0.15, 1.0]);
    geometries.push(pbr_scene::wall(
        "floor",
        [0.0, -3.0, 0.0],
        [7.0, 0.0, 0.0],
        [0.0, 0.0, -7.0],
        0.6,
    ));
    geometries
}

/// Cameras on an arc around the scene, all looking at its middle.
///
/// An arc rather than a full circle: the floor means the interesting variation
/// is in front of and above the grid, and a viewpoint from behind sees nothing
/// but the backs of the spheres.
fn views(count: usize) -> Vec<blade_render::Camera> {
    let target = glam::Vec3::new(0.0, -0.4, 0.0);
    let radius = 11.0;
    (0..count)
        .map(|i| {
            let t = if count == 1 {
                0.5
            } else {
                i as f32 / (count - 1) as f32
            };
            let azimuth = (-55.0 + 110.0 * t).to_radians();
            // Sweep the elevation too, so the set is not a degenerate ring.
            let elevation = (8.0 + 26.0 * (std::f32::consts::PI * t).sin()).to_radians();
            let eye = target
                + radius
                    * glam::Vec3::new(
                        elevation.cos() * azimuth.sin(),
                        elevation.sin(),
                        elevation.cos() * azimuth.cos(),
                    );

            // The camera looks down its local -Z, so build the frame that
            // sends -Z to the view direction and read it back as a rotation.
            let back = (eye - target).normalize();
            let right = glam::Vec3::Y.cross(back).normalize();
            let up = back.cross(right);
            let rotation = glam::Quat::from_mat3(&glam::Mat3::from_cols(right, up, back));

            blade_render::Camera {
                pos: eye.into(),
                rot: mint::Quaternion {
                    v: [rotation.x, rotation.y, rotation.z].into(),
                    s: rotation.w,
                },
                fov_y: 0.8,
                depth: 100.0,
                fov: None,
            }
        })
        .collect()
}

// ------------------------------------------------------------------- rendering

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
            .map(|i| choir.add_worker(&format!("relight-{i}")))
            .collect();
        let cache_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-assets")
            .join("relight-data");
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

#[derive(blade_macros::ShaderData)]
struct RelightGBufferData {
    t_depth: gpu::TextureView,
    t_basis: gpu::TextureView,
    t_flat_normal: gpu::TextureView,
    t_diffuse_albedo: gpu::TextureView,
    t_specular_f0: gpu::TextureView,
    out_material: gpu::TextureView,
    out_geometry: gpu::TextureView,
    out_specular: gpu::TextureView,
}

/// A storage texture the probe writes into, plus the buffer it reads back to.
struct ProbeTarget {
    texture: gpu::Texture,
    view: gpu::TextureView,
    readback: gpu::Buffer,
    size: gpu::Extent,
}

impl ProbeTarget {
    fn new(context: &gpu::Context, size: gpu::Extent, name: &str) -> Self {
        let texture = context.create_texture(gpu::TextureDesc {
            name,
            format: HDR_FORMAT,
            size,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            // A compute pass cannot write a texture without this, and the
            // failure is a silently black image rather than an error.
            usage: gpu::TextureUsage::STORAGE | gpu::TextureUsage::COPY,
            sample_count: 1,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name,
                format: HDR_FORMAT,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );
        let readback = context.create_buffer(gpu::BufferDesc {
            name,
            size: (size.width * size.height) as u64 * 16,
            memory: gpu::Memory::Shared,
        });
        Self {
            texture,
            view,
            readback,
            size,
        }
    }

    fn read(&self) -> Vec<f32> {
        let count = (self.size.width * self.size.height * 4) as usize;
        let mut out = vec![0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.readback.data() as *const f32,
                out.as_mut_ptr(),
                count,
            );
        }
        out
    }

    fn destroy(self, context: &gpu::Context) {
        context.destroy_buffer(self.readback);
        context.destroy_texture_view(self.view);
        context.destroy_texture(self.texture);
    }
}

fn ray_config() -> blade_render::RayConfig {
    blade_render::RayConfig {
        num_environment_samples: 1,
        num_brdf_samples: 4,
        // The environments have a concentrated source in them, which uniform
        // sphere sampling would find only by luck.
        environment_importance_sampling: true,
        // One bounce short of where the open scenes converge, measured by the
        // GI ceiling harness; the floor here needs the indirect light. Set to
        // zero for direct lighting only, which is what a relightable model
        // without an indirect term can actually represent.
        max_bounces: parsed("RELIGHT_BOUNCES", 4),
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

// --------------------------------------------------------------------- writing

/// The renderer's display path, on the CPU.
///
/// Previews come from the captured radiance rather than a second render: it is
/// the same signal one stage later, and re-rendering would only add noise that
/// makes the preview disagree with the data it illustrates.
fn to_display(radiance: &[f32]) -> Vec<u8> {
    // `PostProcConfig::default()` has a white level of 1, which makes the
    // renderer's extended Reinhard an identity, so the display path is the
    // transfer encoding and the clamp. Matching it keeps a preview comparable
    // with a `Capture::Display` render of the same frame.
    let encode = |v: f32| {
        if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };
    radiance
        .chunks_exact(4)
        .flat_map(|texel| {
            let mut out = [255u8; 4];
            for i in 0..3 {
                let v = encode(texel[i].max(0.0)).clamp(0.0, 1.0);
                out[i] = (v * 255.0).round() as u8;
            }
            out
        })
        .collect()
}

fn write_f32(path: &std::path::Path, data: &[f32]) {
    std::fs::write(path, bytemuck::cast_slice(data)).unwrap();
}

// ------------------------------------------------------------------------ main

#[test]
#[ignore = "requires a working GPU context with ray tracing"]
fn generate_relighting_dataset() {
    let Some(out_dir) = env_var("RELIGHT_OUT").map(std::path::PathBuf::from) else {
        println!("Skipping: set RELIGHT_OUT to the directory to write into");
        return;
    };
    if cfg!(target_os = "macos") {
        println!("Skipping: ray tracing not usable on macOS CI");
        return;
    }
    let context = unsafe {
        match gpu::Context::init(gpu::ContextDesc {
            ray_tracing: true,
            validation: env_var("RELIGHT_VALIDATE").is_some(),
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
    assert!(
        !info.is_software_emulated,
        "refusing to generate a dataset on a software rasterizer"
    );

    let size = output_size();
    let frames: usize = parsed("RELIGHT_FRAMES", 512);
    let view_count: usize = parsed("RELIGHT_VIEWS", 12);
    let cameras = views(view_count);
    let mut envs = environments();
    if let Some(limit) = env_var("RELIGHT_ENVS").and_then(|v| v.parse::<usize>().ok()) {
        envs.truncate(limit);
    }
    println!(
        "GPU: {}\n{} views x {} environments at {}x{}, {frames} paths each",
        info.device_name,
        cameras.len(),
        envs.len(),
        size.width,
        size.height
    );

    std::fs::create_dir_all(out_dir.join("env")).unwrap();

    let harness = Harness::new(context);
    let context = std::sync::Arc::clone(&harness.context);
    let hdr = snapshot::OffscreenTarget::new(&context, size, HDR_FORMAT);
    let material = ProbeTarget::new(&context, size, "relight-material");
    let geometry = ProbeTarget::new(&context, size, "relight-geometry");
    let specular = ProbeTarget::new(&context, size, "relight-specular");

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "relight-data",
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
            surface_size: size,
            surface_info: gpu::SurfaceInfo {
                format: HDR_FORMAT,
                alpha: gpu::AlphaMode::Ignored,
            },
            color_space: gpu::ColorSpace::Linear,
            max_debug_lines: 16,
        },
    );

    let probe_shader = context.create_shader(gpu::ShaderDesc {
        source: include_str!("shaders/relight_gbuffer.wgsl"),
        naga_module: None,
    });
    let probe_layout = <RelightGBufferData as gpu::ShaderData>::layout();
    let mut probe_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
        name: "relight-gbuffer",
        data_layouts: &[&probe_layout],
        compute: probe_shader.at("probe"),
    });

    // Cook the scene and the environments, then flush once.
    let model = harness
        .asset_hub
        .models
        .baker
        .create_model("relight-scene", scene());
    let model = harness.asset_hub.models.insert(model);
    let objects = vec![blade_render::Object::from(model)];

    let env_handles = envs
        .iter()
        .map(|env| {
            let texels = bake_environment(env);
            // The radiance as floats, for whatever consumes the dataset, and a
            // tone mapped picture next to it for looking at. A PNG cannot be
            // the source of truth here: the sun is a hundred times brighter
            // than the sky and would clip to the same white.
            write_f32(
                &out_dir.join("env").join(format!("{}.f32", env.name)),
                bytemuck::cast_slice(&texels),
            );
            snapshot::save_image(
                &out_dir.join("env").join(format!("{}.png", env.name)),
                &environment_preview(&texels),
                gpu::Extent {
                    width: ENV_SIZE.0,
                    height: ENV_SIZE.1,
                    depth: 1,
                },
            );
            let texture = harness
                .asset_hub
                .textures
                .baker
                .create_texture_hdr(env.name, ENV_SIZE.0, ENV_SIZE.1, &texels);
            harness.asset_hub.textures.insert(texture)
        })
        .collect::<Vec<_>>();

    let mut flush_buffers = Vec::new();
    harness.asset_hub.flush(&mut encoder, &mut flush_buffers);
    encoder.init_texture(hdr.texture);
    encoder.init_texture(material.texture);
    encoder.init_texture(geometry.texture);
    encoder.init_texture(specular.texture);
    let sync_point = context.submit(&mut encoder);
    assert!(context.wait_for(&sync_point, 30_000).unwrap());

    let debug_config = blade_render::DebugConfig::default();
    let post_proc = blade_render::PostProcConfig {
        // The whole point: hand back radiance, not a picture.
        tone_map: false,
        ..Default::default()
    };

    let mut manifest = String::new();
    manifest.push_str("# Synthetic ground truth for relightable reconstruction.\n");
    manifest.push_str("# Generated by blade's tests/relight_data.rs.\n\n[dataset]\n");
    manifest.push_str(&format!(
        "width = {}\nheight = {}\nsamples_per_pixel = {frames}\nmax_bounces = {}\n",
        size.width,
        size.height,
        ray_config().max_bounces
    ));
    manifest.push_str(
        "radiance = \"linear rgba32f, tone map off, alpha unused\"\n\
         material = \"rgba32f: diffuse albedo rgb, roughness a\"\n\
         geometry = \"rgba32f: shading normal xyz, ray distance a (negative on a miss)\"\n\
         specular = \"rgba32f: specular reflectance at normal incidence, rgb\"\n\
         environment = \"linear rgba32f, equirectangular, matches map_equirect_uv_to_dir\"\n\n",
    );
    for env in &envs {
        manifest.push_str(&format!(
            "[[environment]]\nname = \"{}\"\nradiance = \"env/{}.f32\"\npreview = \"env/{}.png\"\n\n",
            env.name, env.name, env.name
        ));
    }

    for (index, camera) in cameras.iter().enumerate() {
        let dir = out_dir.join(format!("view_{index:03}"));
        std::fs::create_dir_all(&dir).unwrap();
        let mut radiance_rows = Vec::new();

        for (env, handle) in envs.iter().zip(&env_handles) {
            encoder.start();
            let mut temp = blade_render::FrameResources::default();
            for frame_index in 0..frames {
                renderer.build_scene(
                    &mut encoder,
                    &objects,
                    Some(*handle),
                    &harness.asset_hub,
                    &context,
                    &mut temp,
                );
                renderer.prepare(
                    &mut encoder,
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
                    &mut encoder,
                    blade_render::RenderMode::Canonical,
                    debug_config,
                    ray_config(),
                    None,
                );
            }
            {
                let mut pass = encoder.render(
                    "relight-compose",
                    gpu::RenderTargetSet {
                        colors: &[gpu::RenderTarget {
                            view: hdr.view,
                            init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                            finish_op: gpu::FinishOp::Store,
                        }],
                        depth_stencil: None,
                    },
                );
                renderer.post_proc(&mut pass, debug_config, post_proc, &[], &[]);
            }
            let bytes = hdr.read_pixels(&context, &mut encoder);
            let values: &[f32] = bytemuck::cast_slice(&bytes);

            write_f32(&dir.join(format!("{}.f32", env.name)), values);
            snapshot::save_image(
                &dir.join(format!("{}.png", env.name)),
                &to_display(values),
                size,
            );
            radiance_rows.push(format!(
                "  {{ environment = \"{}\", file = \"view_{index:03}/{}.f32\", preview = \"view_{index:03}/{}.png\" }}",
                env.name, env.name, env.name
            ));

            for buffer in temp.buffers {
                context.destroy_buffer(buffer);
            }
            for acceleration_structure in temp.acceleration_structures {
                context.destroy_acceleration_structure(acceleration_structure);
            }
        }

        // The G-buffer is filled by the real-time path; the canonical one
        // resolves its hits inside the path tracer and never writes it. So it
        // takes one real-time frame to describe this pose, which is cheap and
        // safe to do here because every radiance capture for the view has
        // already been read back.
        //
        // It depends on the camera alone, so once per view is enough.
        encoder.start();
        let mut gbuffer_temp = blade_render::FrameResources::default();
        renderer.build_scene(
            &mut encoder,
            &objects,
            env_handles.first().copied(),
            &harness.asset_hub,
            &context,
            &mut gbuffer_temp,
        );
        renderer.prepare(
            &mut encoder,
            camera,
            blade_render::FrameConfig {
                frozen: false,
                debug_draw: false,
                reset_variance: true,
                reset_reservoirs: true,
                reset_accumulation: true,
            },
        );
        renderer.render(
            &mut encoder,
            blade_render::RenderMode::RealTime,
            debug_config,
            ray_config(),
            None,
        );
        {
            let views = renderer.view_gbuffer();
            let mut pass = encoder.compute("relight-gbuffer");
            let mut commands = pass.with(&probe_pipeline);
            commands.bind(
                0,
                &RelightGBufferData {
                    t_depth: views.depth,
                    t_basis: views.basis,
                    t_flat_normal: views.flat_normal,
                    t_diffuse_albedo: views.diffuse_albedo,
                    t_specular_f0: views.specular_f0,
                    out_material: material.view,
                    out_geometry: geometry.view,
                    out_specular: specular.view,
                },
            );
            commands.dispatch([size.width.div_ceil(8), size.height.div_ceil(8), 1]);
        }
        {
            let mut transfer = encoder.transfer("relight-gbuffer-readback");
            transfer.copy_texture_to_buffer(
                material.texture.into(),
                material.readback.into(),
                size.width * 16,
                size,
            );
            transfer.copy_texture_to_buffer(
                geometry.texture.into(),
                geometry.readback.into(),
                size.width * 16,
                size,
            );
            transfer.copy_texture_to_buffer(
                specular.texture.into(),
                specular.readback.into(),
                size.width * 16,
                size,
            );
        }
        let sync_point = context.submit(&mut encoder);
        assert!(
            context
                .wait_for(&sync_point, snapshot::READBACK_TIMEOUT_MS)
                .unwrap()
        );

        for buffer in gbuffer_temp.buffers {
            context.destroy_buffer(buffer);
        }
        for acceleration_structure in gbuffer_temp.acceleration_structures {
            context.destroy_acceleration_structure(acceleration_structure);
        }

        let material_data = material.read();
        let geometry_data = geometry.read();
        write_f32(&dir.join("material.f32"), &material_data);
        write_f32(&dir.join("geometry.f32"), &geometry_data);
        write_f32(&dir.join("specular.f32"), &specular.read());

        let covered = geometry_data
            .chunks_exact(4)
            .filter(|texel| texel[3] > 0.0)
            .count();
        let position: [f32; 3] = camera.pos.into();
        let rotation = camera.rot;
        manifest.push_str(&format!(
            "[[view]]\nindex = {index}\nposition = [{:.6}, {:.6}, {:.6}]\n\
             orientation = [{:.6}, {:.6}, {:.6}, {:.6}]\nfov_y = {:.6}\n\
             coverage = {:.4}\nmaterial = \"view_{index:03}/material.f32\"\n\
             geometry = \"view_{index:03}/geometry.f32\"\n\
             specular = \"view_{index:03}/specular.f32\"\nradiance = [\n{}\n]\n\n",
            position[0],
            position[1],
            position[2],
            rotation.v.x,
            rotation.v.y,
            rotation.v.z,
            rotation.s,
            camera.fov_y,
            covered as f64 / (size.width * size.height) as f64,
            radiance_rows.join(",\n"),
        ));
        println!(
            "view {index:>3}: {:.1} % covered",
            100.0 * covered as f64 / (size.width * size.height) as f64
        );
    }

    std::fs::write(out_dir.join("manifest.toml"), manifest).unwrap();
    println!("\nwrote {} views to {}", cameras.len(), out_dir.display());

    context.destroy_compute_pipeline(&mut probe_pipeline);
    renderer.destroy(&context);
    for buffer in flush_buffers {
        context.destroy_buffer(buffer);
    }
    context.destroy_command_encoder(&mut encoder);
    material.destroy(&context);
    geometry.destroy(&context);
    specular.destroy(&context);
    hdr.destroy(&context);
    harness.destroy();
}
