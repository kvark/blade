use std::{collections::HashMap, fs, mem, ptr, time};

const MAX_RESOURCES: u32 = 1000;

#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    pub screen_size: blade_graphics::Extent,
    pub surface_format: blade_graphics::TextureFormat,
    pub max_debug_lines: u32,
}

struct DummyResources {
    size: blade_graphics::Extent,
    white_texture: blade_graphics::Texture,
    white_view: blade_graphics::TextureView,
}

struct Samplers {
    linear: blade_graphics::Sampler,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(u32)]
pub enum DebugMode {
    None = 0,
    Depth = 1,
    Normal = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct RayConfig {
    pub num_environment_samples: u32,
    pub temporal_history: u32,
}

// Has to match the shader!
#[repr(C)]
#[derive(Debug)]
struct DebugVariance {
    color_sum: [f32; 3],
    pad: u32,
    color2_sum: [f32; 3],
    count: u32,
}

struct DebugRender {
    capacity: u32,
    buffer: blade_graphics::Buffer,
    variance_buffer: blade_graphics::Buffer,
    pipeline: blade_graphics::RenderPipeline,
}

struct Targets {
    main: blade_graphics::Texture,
    main_view: blade_graphics::TextureView,
    depth: blade_graphics::Texture,
    depth_view: blade_graphics::TextureView,
    basis: blade_graphics::Texture,
    basis_view: blade_graphics::TextureView,
    albedo: blade_graphics::Texture,
    albedo_view: blade_graphics::TextureView,
}

impl Targets {
    fn create_target(
        name: &str,
        format: blade_graphics::TextureFormat,
        size: blade_graphics::Extent,
        gpu: &blade_graphics::Context,
    ) -> (blade_graphics::Texture, blade_graphics::TextureView) {
        let texture = gpu.create_texture(blade_graphics::TextureDesc {
            name,
            format,
            size,
            dimension: blade_graphics::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: blade_graphics::TextureUsage::RESOURCE | blade_graphics::TextureUsage::STORAGE,
        });
        let view = gpu.create_texture_view(blade_graphics::TextureViewDesc {
            name,
            texture,
            format,
            dimension: blade_graphics::ViewDimension::D2,
            subresources: &blade_graphics::TextureSubresources::default(),
        });
        (texture, view)
    }

    fn new(
        size: blade_graphics::Extent,
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
    ) -> Self {
        let (main, main_view) = Self::create_target(
            "main",
            blade_graphics::TextureFormat::Rgba16Float,
            size,
            gpu,
        );
        encoder.init_texture(main);
        let (depth, depth_view) =
            Self::create_target("depth", blade_graphics::TextureFormat::R32Float, size, gpu);
        encoder.init_texture(depth);
        let (basis, basis_view) = Self::create_target(
            "basis",
            blade_graphics::TextureFormat::Rgba8Snorm,
            size,
            gpu,
        );
        encoder.init_texture(basis);
        let (albedo, albedo_view) = Self::create_target(
            "basis",
            blade_graphics::TextureFormat::Rgba8Unorm,
            size,
            gpu,
        );
        encoder.init_texture(albedo);
        Self {
            main,
            main_view,
            depth,
            depth_view,
            basis,
            basis_view,
            albedo,
            albedo_view,
        }
    }

    fn destroy(&self, gpu: &blade_graphics::Context) {
        gpu.destroy_texture_view(self.main_view);
        gpu.destroy_texture(self.main);
        gpu.destroy_texture_view(self.depth_view);
        gpu.destroy_texture(self.depth);
        gpu.destroy_texture_view(self.basis_view);
        gpu.destroy_texture(self.basis);
        gpu.destroy_texture_view(self.albedo_view);
        gpu.destroy_texture(self.albedo);
    }
}

/// Blade Renderer is a comprehensive rendering solution for
/// end user applications.
///
/// It takes care of the shaders, geometry buffers, acceleration structures,
/// dummy resources, and debug drawing.
///
/// It doesn't:
///   - manage or submit any command encoders
///   - know about the window to display on
pub struct Renderer {
    config: RenderConfig,
    targets: Targets,
    shader_modified_time: Option<time::SystemTime>,
    fill_pipeline: blade_graphics::ComputePipeline,
    main_pipeline: blade_graphics::ComputePipeline,
    blit_pipeline: blade_graphics::RenderPipeline,
    scene: super::Scene,
    acceleration_structure: blade_graphics::AccelerationStructure,
    environment_view: blade_graphics::TextureView,
    dummy: DummyResources,
    hit_buffer: blade_graphics::Buffer,
    vertex_buffers: blade_graphics::BufferArray<MAX_RESOURCES>,
    index_buffers: blade_graphics::BufferArray<MAX_RESOURCES>,
    textures: blade_graphics::TextureArray<MAX_RESOURCES>,
    samplers: Samplers,
    reservoir_buffer: blade_graphics::Buffer,
    debug: DebugRender,
    is_tlas_dirty: bool,
    are_reservoirs_dirty: bool,
    screen_size: blade_graphics::Extent,
    frame_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct CameraParams {
    position: [f32; 3],
    depth: f32,
    orientation: [f32; 4],
    fov: [f32; 2],
    mouse_pos: [i32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct MainParams {
    frame_index: u32,
    debug_mode: u32,
    num_environment_samples: u32,
    temporal_history: u32,
}

#[derive(blade_macros::ShaderData)]
struct FillData<'a> {
    camera: CameraParams,
    acc_struct: blade_graphics::AccelerationStructure,
    hit_entries: blade_graphics::BufferPiece,
    index_buffers: &'a blade_graphics::BufferArray<MAX_RESOURCES>,
    vertex_buffers: &'a blade_graphics::BufferArray<MAX_RESOURCES>,
    textures: &'a blade_graphics::TextureArray<MAX_RESOURCES>,
    sampler_linear: blade_graphics::Sampler,
    debug_buf: blade_graphics::BufferPiece,
    out_depth: blade_graphics::TextureView,
    out_basis: blade_graphics::TextureView,
    out_albedo: blade_graphics::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct MainData {
    camera: CameraParams,
    parameters: MainParams,
    acc_struct: blade_graphics::AccelerationStructure,
    env_map: blade_graphics::TextureView,
    in_depth: blade_graphics::TextureView,
    in_basis: blade_graphics::TextureView,
    in_albedo: blade_graphics::TextureView,
    debug_buf: blade_graphics::BufferPiece,
    reservoirs: blade_graphics::BufferPiece,
    output: blade_graphics::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct BlitData {
    input: blade_graphics::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct DebugData {
    camera: CameraParams,
    debug_buf: blade_graphics::BufferPiece,
}

#[repr(C)]
#[derive(Debug)]
struct HitEntry {
    index_buf: u32,
    vertex_buf: u32,
    rotation: [i8; 4],
    //geometry_to_object: mint::RowMatrix3x4<f32>,
    base_color_texture: u32,
    base_color_factor: [u8; 4],
}

struct ShaderPipelines {
    fill: blade_graphics::ComputePipeline,
    main: blade_graphics::ComputePipeline,
    blit: blade_graphics::RenderPipeline,
    debug: blade_graphics::RenderPipeline,
    debug_line_size: u32,
    debug_buffer_size: u32,
    reservoir_size: u32,
}

const SHADER_PATH: &str = "blade-render/shader.wgsl";

impl ShaderPipelines {
    fn init(config: &RenderConfig, gpu: &blade_graphics::Context) -> Result<Self, &'static str> {
        let source = fs::read_to_string(SHADER_PATH).unwrap();
        let shader = gpu.try_create_shader(blade_graphics::ShaderDesc { source: &source })?;
        let fill_layout = <FillData as blade_graphics::ShaderData>::layout();
        let main_layout = <MainData as blade_graphics::ShaderData>::layout();
        let blit_layout = <BlitData as blade_graphics::ShaderData>::layout();
        let debug_layout = <DebugData as blade_graphics::ShaderData>::layout();

        Ok(Self {
            fill: gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
                name: "fill-gbuf",
                data_layouts: &[&fill_layout],
                compute: shader.at("fill_gbuf"),
            }),
            main: gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
                name: "ray-trace",
                data_layouts: &[&main_layout],
                compute: shader.at("main"),
            }),
            blit: gpu.create_render_pipeline(blade_graphics::RenderPipelineDesc {
                name: "main",
                data_layouts: &[&blit_layout],
                primitive: blade_graphics::PrimitiveState {
                    topology: blade_graphics::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                vertex: shader.at("blit_vs"),
                fragment: shader.at("blit_fs"),
                color_targets: &[config.surface_format.into()],
                depth_stencil: None,
            }),
            debug: gpu.create_render_pipeline(blade_graphics::RenderPipelineDesc {
                name: "debug",
                data_layouts: &[&debug_layout],
                vertex: shader.at("debug_vs"),
                primitive: blade_graphics::PrimitiveState {
                    topology: blade_graphics::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: None,
                fragment: shader.at("debug_fs"),
                color_targets: &[config.surface_format.into()],
            }),
            debug_line_size: shader.get_struct_size("DebugLine"),
            debug_buffer_size: shader.get_struct_size("DebugBuffer"),
            reservoir_size: shader.get_struct_size("StoredReservoir"),
        })
    }
}

impl Renderer {
    /// Create a new renderer with a given configuration.
    ///
    /// Panics if the system is not compatible.
    /// Records initialization routines into the given command encoder.
    pub fn new(
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
        config: &RenderConfig,
    ) -> Self {
        let capabilities = gpu.capabilities();
        assert!(capabilities
            .ray_query
            .contains(blade_graphics::ShaderVisibility::COMPUTE));

        let shader_modified_time = fs::metadata(SHADER_PATH).and_then(|m| m.modified()).ok();
        let sp = ShaderPipelines::init(config, gpu).unwrap();

        let debug_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "debug",
            size: (sp.debug_buffer_size + (config.max_debug_lines - 1) * sp.debug_line_size) as u64,
            memory: blade_graphics::Memory::Device,
        });
        let variance_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "variance",
            size: mem::size_of::<DebugVariance>() as u64,
            memory: blade_graphics::Memory::Shared,
        });
        unsafe {
            ptr::write_bytes(variance_buffer.data(), 0, mem::size_of::<DebugVariance>());
        }

        let total_reservoirs =
            config.screen_size.width as usize * config.screen_size.height as usize;
        let reservoir_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
            name: "reservoirs",
            size: sp.reservoir_size as u64 * total_reservoirs as u64,
            memory: blade_graphics::Memory::Device,
        });

        let targets = Targets::new(config.screen_size, encoder, gpu);

        let dummy = {
            let size = blade_graphics::Extent {
                width: 1,
                height: 1,
                depth: 1,
            };
            let white_texture = gpu.create_texture(blade_graphics::TextureDesc {
                name: "dummy/white",
                format: blade_graphics::TextureFormat::Rgba8Unorm,
                size,
                array_layer_count: 1,
                mip_level_count: 1,
                dimension: blade_graphics::TextureDimension::D2,
                usage: blade_graphics::TextureUsage::COPY | blade_graphics::TextureUsage::RESOURCE,
            });
            let white_view = gpu.create_texture_view(blade_graphics::TextureViewDesc {
                name: "dummy/white",
                texture: white_texture,
                format: blade_graphics::TextureFormat::Rgba8Unorm,
                dimension: blade_graphics::ViewDimension::D2,
                subresources: &blade_graphics::TextureSubresources::default(),
            });
            encoder.init_texture(white_texture);
            DummyResources {
                size,
                white_texture,
                white_view,
            }
        };

        let samplers = Samplers {
            linear: gpu.create_sampler(blade_graphics::SamplerDesc {
                name: "linear",
                address_modes: [blade_graphics::AddressMode::ClampToEdge; 3],
                mag_filter: blade_graphics::FilterMode::Linear,
                min_filter: blade_graphics::FilterMode::Linear,
                mipmap_filter: blade_graphics::FilterMode::Linear,
                ..Default::default()
            }),
        };

        Self {
            config: *config,
            targets,
            scene: super::Scene::default(),
            shader_modified_time,
            fill_pipeline: sp.fill,
            main_pipeline: sp.main,
            blit_pipeline: sp.blit,
            acceleration_structure: blade_graphics::AccelerationStructure::default(),
            environment_view: dummy.white_view,
            dummy,
            hit_buffer: blade_graphics::Buffer::default(),
            vertex_buffers: blade_graphics::BufferArray::new(),
            index_buffers: blade_graphics::BufferArray::new(),
            textures: blade_graphics::TextureArray::new(),
            samplers,
            reservoir_buffer,
            debug: DebugRender {
                capacity: config.max_debug_lines,
                buffer: debug_buffer,
                variance_buffer,
                pipeline: sp.debug,
            },
            is_tlas_dirty: true,
            are_reservoirs_dirty: true,
            screen_size: config.screen_size,
            frame_index: 0,
        }
    }

    /// Destroy all internally managed GPU resources.
    pub fn destroy(&mut self, gpu: &blade_graphics::Context) {
        // internal resources
        self.targets.destroy(gpu);
        if self.hit_buffer != blade_graphics::Buffer::default() {
            gpu.destroy_buffer(self.hit_buffer);
        }
        gpu.destroy_acceleration_structure(self.acceleration_structure);
        // dummy resources
        gpu.destroy_texture_view(self.dummy.white_view);
        gpu.destroy_texture(self.dummy.white_texture);
        // samplers
        gpu.destroy_sampler(self.samplers.linear);
        // buffers
        gpu.destroy_buffer(self.debug.buffer);
        gpu.destroy_buffer(self.debug.variance_buffer);
        gpu.destroy_buffer(self.reservoir_buffer);
    }

    pub fn merge_scene(&mut self, scene: super::Scene) {
        self.scene = scene;
    }

    /// Check if any shaders need to be hot reloaded, and do it.
    pub fn hot_reload(
        &mut self,
        gpu: &blade_graphics::Context,
        sync_point: &blade_graphics::SyncPoint,
    ) -> bool {
        if let Some(ref mut last_mod_time) = self.shader_modified_time {
            let cur_mod_time = fs::metadata(SHADER_PATH).unwrap().modified().unwrap();
            if *last_mod_time != cur_mod_time {
                log::info!("Hot-reloading shaders...");
                *last_mod_time = cur_mod_time;
                if let Ok(sp) = ShaderPipelines::init(&self.config, gpu) {
                    gpu.wait_for(sync_point, !0);
                    self.fill_pipeline = sp.fill;
                    self.main_pipeline = sp.main;
                    self.blit_pipeline = sp.blit;
                    self.debug.pipeline = sp.debug;
                    return true;
                }
            }
        }
        false
    }

    /// Prepare to render a frame.
    pub fn prepare(
        &mut self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        asset_hub: &crate::AssetHub,
        gpu: &blade_graphics::Context,
        temp_buffers: &mut Vec<blade_graphics::Buffer>,
        enable_debug: bool,
    ) {
        if self.is_tlas_dirty {
            self.is_tlas_dirty = false;
            if self.acceleration_structure != blade_graphics::AccelerationStructure::default() {
                temp_buffers.push(self.hit_buffer);
                //TODO: delay this or stall the GPU
                gpu.destroy_acceleration_structure(self.acceleration_structure);
            }

            let (tlas, geometry_count) = self.scene.build_top_level_acceleration_structure(
                command_encoder,
                &asset_hub.models,
                gpu,
                temp_buffers,
            );
            self.acceleration_structure = tlas;
            log::info!("Preparing ray tracing with {geometry_count} geometries in total");
            let mut transfers = command_encoder.transfer();

            {
                // init the dummy
                let staging = gpu.create_buffer(blade_graphics::BufferDesc {
                    name: "dummy staging",
                    size: 4,
                    memory: blade_graphics::Memory::Upload,
                });
                unsafe {
                    ptr::write(staging.data() as *mut _, [!0u8; 4]);
                }
                transfers.copy_buffer_to_texture(
                    staging.into(),
                    4,
                    self.dummy.white_texture.into(),
                    self.dummy.size,
                );
                temp_buffers.push(staging);
            }
            {
                // init the debug buffer
                let data = [2, 0, 0, 0, self.debug.capacity, 0];
                let size = 4 * data.len() as u64;
                let staging = gpu.create_buffer(blade_graphics::BufferDesc {
                    name: "debug buf staging",
                    size,
                    memory: blade_graphics::Memory::Upload,
                });
                unsafe {
                    ptr::write(staging.data() as *mut _, data);
                }
                transfers.copy_buffer_to_buffer(staging.into(), self.debug.buffer.into(), size);
                temp_buffers.push(staging);
            }

            let hit_staging = {
                // init the hit buffer
                let hit_size = (geometry_count as usize * mem::size_of::<HitEntry>()) as u64;
                self.hit_buffer = gpu.create_buffer(blade_graphics::BufferDesc {
                    name: "hit entries",
                    size: hit_size,
                    memory: blade_graphics::Memory::Device,
                });
                let staging = gpu.create_buffer(blade_graphics::BufferDesc {
                    name: "hit staging",
                    size: hit_size,
                    memory: blade_graphics::Memory::Upload,
                });
                temp_buffers.push(staging);
                transfers.copy_buffer_to_buffer(staging.at(0), self.hit_buffer.at(0), hit_size);
                staging
            };

            self.textures.clear();
            let dummy_white = self.textures.alloc(self.dummy.white_view);
            let mut texture_indices = HashMap::new();

            self.vertex_buffers.clear();
            self.index_buffers.clear();
            let mut geometry_index = 0;
            for object in self.scene.objects.iter() {
                let model = &asset_hub.models[object.model];
                let rotation = {
                    let col_matrix = mint::ColumnMatrix3x4::from(object.transform);
                    let m3 = glam::Mat3::from_cols(
                        col_matrix.x.into(),
                        col_matrix.y.into(),
                        col_matrix.z.into(),
                    );
                    let m3_normal = m3.inverse().transpose();
                    let quat = glam::Quat::from_mat3(&m3_normal);
                    let qv = glam::Vec4::from(quat) * 127.0;
                    [qv.x as i8, qv.y as i8, qv.z as i8, qv.w as i8]
                };
                for geometry in model.geometries.iter() {
                    let material = &model.materials[geometry.material_index];
                    let vertex_offset =
                        geometry.vertex_range.start as u64 * mem::size_of::<crate::Vertex>() as u64;
                    let hit_entry = HitEntry {
                        index_buf: match geometry.index_type {
                            Some(_) => self
                                .index_buffers
                                .alloc(model.index_buffer.at(geometry.index_offset)),
                            None => !0,
                        },
                        vertex_buf: self
                            .vertex_buffers
                            .alloc(model.vertex_buffer.at(vertex_offset)),
                        rotation,
                        base_color_texture: match material.base_color_texture {
                            Some(handle) => *texture_indices.entry(handle).or_insert_with(|| {
                                let texture = &asset_hub.textures[handle];
                                self.textures.alloc(texture.view)
                            }),
                            None => dummy_white,
                        },
                        base_color_factor: {
                            let c = material.base_color_factor;
                            [
                                (c[0] * 255.0) as u8,
                                (c[1] * 255.0) as u8,
                                (c[2] * 255.0) as u8,
                                (c[3] * 255.0) as u8,
                            ]
                        },
                    };
                    log::debug!("Entry[{geometry_index}] = {hit_entry:?}");
                    unsafe {
                        ptr::write(
                            (hit_staging.data() as *mut HitEntry).add(geometry_index),
                            hit_entry,
                        );
                    }
                    geometry_index += 1;
                }
            }
            assert_eq!(geometry_index, geometry_count as usize);
        }

        self.frame_index += 1;
        self.environment_view = match self.scene.environment_map {
            Some(handle) => asset_hub.textures[handle].view,
            None => self.dummy.white_view,
        };

        let mut transfer = command_encoder.transfer();
        if enable_debug {
            // reset the debug line count
            transfer.fill_buffer(self.debug.buffer.at(4), 4, 0);
            transfer.fill_buffer(self.debug.buffer.at(20), 4, 1);
            // copy the previous frame variance
            transfer.copy_buffer_to_buffer(
                self.debug.buffer.at(32),
                self.debug.variance_buffer.into(),
                mem::size_of::<DebugVariance>() as u64,
            );
        } else {
            // reset the open bit, variance accumulator
            transfer.fill_buffer(self.debug.buffer.at(20), 12 + 32, 0);
        }
        if self.are_reservoirs_dirty {
            self.are_reservoirs_dirty = false;
            let total_reservoirs = self.screen_size.width as u64 * self.screen_size.height as u64;
            transfer.fill_buffer(self.reservoir_buffer.into(), total_reservoirs, 0);
        }
    }

    fn make_camera_params(
        &self,
        camera: &super::Camera,
        mouse_pos: Option<[i32; 2]>,
    ) -> CameraParams {
        let fov_x = camera.fov_y * self.screen_size.width as f32 / self.screen_size.height as f32;
        CameraParams {
            position: camera.pos.into(),
            depth: camera.depth,
            orientation: camera.rot.into(),
            fov: [fov_x, camera.fov_y],
            mouse_pos: match mouse_pos {
                Some(p) => [p[0], self.screen_size.height as i32 - p[1]],
                None => [-1; 2],
            },
        }
    }

    /// Ray trace the scene.
    ///
    /// The result is stored internally in an HDR render target.
    pub fn ray_trace(
        &self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        camera: &super::Camera,
        debug_mode: DebugMode,
        mouse_pos: Option<[i32; 2]>,
        ray_config: RayConfig,
    ) {
        assert!(!self.is_tlas_dirty);

        if let mut pass = command_encoder.compute() {
            let mut pc = pass.with(&self.fill_pipeline);
            let wg_size = self.fill_pipeline.get_workgroup_size();
            let group_count = [
                (self.screen_size.width + wg_size[0] - 1) / wg_size[0],
                (self.screen_size.height + wg_size[1] - 1) / wg_size[1],
                1,
            ];

            pc.bind(
                0,
                &FillData {
                    camera: self.make_camera_params(camera, mouse_pos),
                    acc_struct: self.acceleration_structure,
                    hit_entries: self.hit_buffer.into(),
                    index_buffers: &self.index_buffers,
                    vertex_buffers: &self.vertex_buffers,
                    textures: &self.textures,
                    sampler_linear: self.samplers.linear,
                    debug_buf: self.debug.buffer.into(),
                    out_depth: self.targets.depth_view,
                    out_basis: self.targets.basis_view,
                    out_albedo: self.targets.albedo_view,
                },
            );
            pc.dispatch(group_count);
        }

        if let mut pass = command_encoder.compute() {
            let mut pc = pass.with(&self.main_pipeline);
            let wg_size = self.main_pipeline.get_workgroup_size();
            let group_count = [
                (self.screen_size.width + wg_size[0] - 1) / wg_size[0],
                (self.screen_size.height + wg_size[1] - 1) / wg_size[1],
                1,
            ];

            pc.bind(
                0,
                &MainData {
                    camera: self.make_camera_params(camera, mouse_pos),
                    parameters: MainParams {
                        frame_index: self.frame_index,
                        debug_mode: debug_mode as u32,
                        num_environment_samples: ray_config.num_environment_samples,
                        temporal_history: ray_config.temporal_history,
                    },
                    acc_struct: self.acceleration_structure,
                    env_map: self.environment_view,
                    in_depth: self.targets.depth_view,
                    in_basis: self.targets.basis_view,
                    in_albedo: self.targets.albedo_view,
                    debug_buf: self.debug.buffer.into(),
                    reservoirs: self.reservoir_buffer.into(),
                    output: self.targets.main_view,
                },
            );
            pc.dispatch(group_count);
        }
    }

    /// Blit the rendering result into a specified render pass.
    pub fn blit(&self, pass: &mut blade_graphics::RenderCommandEncoder, camera: &super::Camera) {
        if let mut pc = pass.with(&self.blit_pipeline) {
            pc.bind(
                0,
                &BlitData {
                    input: self.targets.main_view,
                },
            );
            pc.draw(0, 3, 0, 1);
        }
        if let mut pc = pass.with(&self.debug.pipeline) {
            pc.bind(
                0,
                &DebugData {
                    camera: self.make_camera_params(camera, None),
                    debug_buf: self.debug.buffer.into(),
                },
            );
            pc.draw_indirect(self.debug.buffer.at(0));
        }
    }

    pub fn read_debug_std_deviation(&self) -> Option<mint::Vector3<f32>> {
        let dv = unsafe { &*(self.debug.variance_buffer.data() as *const DebugVariance) };
        if dv.count == 0 {
            return None;
        }
        let sum_avg = glam::Vec3::from(dv.color_sum) / (dv.count as f32);
        let sum2_avg = glam::Vec3::from(dv.color2_sum) / (dv.count as f32);
        let variance = sum2_avg - sum_avg * sum_avg;
        Some(mint::Vector3 {
            x: variance.x.sqrt(),
            y: variance.y.sqrt(),
            z: variance.z.sqrt(),
        })
    }
}
