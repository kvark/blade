use std::{fs, mem, ptr, time};

const MAX_RESOURCES: u32 = 1000;

#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    pub screen_size: blade::Extent,
    pub surface_format: blade::TextureFormat,
    pub max_debug_lines: u32,
}

struct DummyResources {
    size: blade::Extent,
    white_texture: blade::Texture,
    white_view: blade::TextureView,
}

struct Samplers {
    linear: blade::Sampler,
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

struct DebugRender {
    capacity: u32,
    buffer: blade::Buffer,
    pipeline: blade::RenderPipeline,
}

struct Targets {
    main: blade::Texture,
    main_view: blade::TextureView,
    depth: blade::Texture,
    depth_view: blade::TextureView,
    basis: blade::Texture,
    basis_view: blade::TextureView,
    albedo: blade::Texture,
    albedo_view: blade::TextureView,
}

impl Targets {
    fn create_target(
        name: &str,
        format: blade::TextureFormat,
        size: blade::Extent,
        gpu: &blade::Context,
    ) -> (blade::Texture, blade::TextureView) {
        let texture = gpu.create_texture(blade::TextureDesc {
            name,
            format,
            size,
            dimension: blade::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: blade::TextureUsage::RESOURCE | blade::TextureUsage::STORAGE,
        });
        let view = gpu.create_texture_view(blade::TextureViewDesc {
            name,
            texture,
            format,
            dimension: blade::ViewDimension::D2,
            subresources: &blade::TextureSubresources::default(),
        });
        (texture, view)
    }

    fn new(size: blade::Extent, encoder: &mut blade::CommandEncoder, gpu: &blade::Context) -> Self {
        let (main, main_view) =
            Self::create_target("main", blade::TextureFormat::Rgba16Float, size, gpu);
        encoder.init_texture(main);
        let (depth, depth_view) =
            Self::create_target("depth", blade::TextureFormat::R32Float, size, gpu);
        encoder.init_texture(depth);
        let (basis, basis_view) =
            Self::create_target("basis", blade::TextureFormat::Rgba8Snorm, size, gpu);
        encoder.init_texture(basis);
        let (albedo, albedo_view) =
            Self::create_target("basis", blade::TextureFormat::Rgba8Unorm, size, gpu);
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

    fn destroy(&self, gpu: &blade::Context) {
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

pub struct Renderer {
    config: RenderConfig,
    targets: Targets,
    shader_modified_time: Option<time::SystemTime>,
    fill_pipeline: blade::ComputePipeline,
    main_pipeline: blade::ComputePipeline,
    blit_pipeline: blade::RenderPipeline,
    scene: super::Scene,
    acceleration_structure: blade::AccelerationStructure,
    dummy: DummyResources,
    hit_buffer: blade::Buffer,
    vertex_buffers: blade::BufferArray<MAX_RESOURCES>,
    index_buffers: blade::BufferArray<MAX_RESOURCES>,
    textures: blade::TextureArray<MAX_RESOURCES>,
    samplers: Samplers,
    reservoir_buffer: blade::Buffer,
    debug: DebugRender,
    is_tlas_dirty: bool,
    are_reservoirs_dirty: bool,
    screen_size: blade::Extent,
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
    acc_struct: blade::AccelerationStructure,
    hit_entries: blade::BufferPiece,
    index_buffers: &'a blade::BufferArray<MAX_RESOURCES>,
    vertex_buffers: &'a blade::BufferArray<MAX_RESOURCES>,
    textures: &'a blade::TextureArray<MAX_RESOURCES>,
    sampler_linear: blade::Sampler,
    debug_buf: blade::BufferPiece,
    out_depth: blade::TextureView,
    out_basis: blade::TextureView,
    out_albedo: blade::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct MainData {
    camera: CameraParams,
    parameters: MainParams,
    acc_struct: blade::AccelerationStructure,
    in_depth: blade::TextureView,
    in_basis: blade::TextureView,
    in_albedo: blade::TextureView,
    debug_buf: blade::BufferPiece,
    reservoirs: blade::BufferPiece,
    output: blade::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct BlitData {
    input: blade::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct DebugData {
    camera: CameraParams,
    debug_buf: blade::BufferPiece,
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
    fill: blade::ComputePipeline,
    main: blade::ComputePipeline,
    blit: blade::RenderPipeline,
    debug: blade::RenderPipeline,
    debug_line_size: u32,
    debug_buffer_size: u32,
    reservoir_size: u32,
}

const SHADER_PATH: &str = "blade-render/shader.wgsl";

impl ShaderPipelines {
    fn init(config: &RenderConfig, gpu: &blade::Context) -> Result<Self, &'static str> {
        let source = fs::read_to_string(SHADER_PATH).unwrap();
        let shader = gpu.try_create_shader(blade::ShaderDesc { source: &source })?;
        let fill_layout = <FillData as blade::ShaderData>::layout();
        let main_layout = <MainData as blade::ShaderData>::layout();
        let blit_layout = <BlitData as blade::ShaderData>::layout();
        let debug_layout = <DebugData as blade::ShaderData>::layout();

        Ok(Self {
            fill: gpu.create_compute_pipeline(blade::ComputePipelineDesc {
                name: "fill-gbuf",
                data_layouts: &[&fill_layout],
                compute: shader.at("fill_gbuf"),
            }),
            main: gpu.create_compute_pipeline(blade::ComputePipelineDesc {
                name: "ray-trace",
                data_layouts: &[&main_layout],
                compute: shader.at("main"),
            }),
            blit: gpu.create_render_pipeline(blade::RenderPipelineDesc {
                name: "main",
                data_layouts: &[&blit_layout],
                primitive: blade::PrimitiveState {
                    topology: blade::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                vertex: shader.at("blit_vs"),
                fragment: shader.at("blit_fs"),
                color_targets: &[config.surface_format.into()],
                depth_stencil: None,
            }),
            debug: gpu.create_render_pipeline(blade::RenderPipelineDesc {
                name: "debug",
                data_layouts: &[&debug_layout],
                vertex: shader.at("debug_vs"),
                primitive: blade::PrimitiveState {
                    topology: blade::PrimitiveTopology::LineList,
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
    pub fn new(
        encoder: &mut blade::CommandEncoder,
        gpu: &blade::Context,
        config: &RenderConfig,
    ) -> Self {
        let capabilities = gpu.capabilities();
        assert!(capabilities
            .ray_query
            .contains(blade::ShaderVisibility::COMPUTE));

        let shader_modified_time = fs::metadata(SHADER_PATH).and_then(|m| m.modified()).ok();
        let sp = ShaderPipelines::init(config, gpu).unwrap();

        let debug_buffer = gpu.create_buffer(blade::BufferDesc {
            name: "debug",
            size: (sp.debug_buffer_size + (config.max_debug_lines - 1) * sp.debug_line_size) as u64,
            memory: blade::Memory::Device,
        });
        let total_reservoirs =
            config.screen_size.width as usize * config.screen_size.height as usize;
        let reservoir_buffer = gpu.create_buffer(blade::BufferDesc {
            name: "reservoirs",
            size: sp.reservoir_size as u64 * total_reservoirs as u64,
            memory: blade::Memory::Device,
        });

        let targets = Targets::new(config.screen_size, encoder, gpu);

        let dummy = {
            let size = blade::Extent {
                width: 1,
                height: 1,
                depth: 1,
            };
            let white_texture = gpu.create_texture(blade::TextureDesc {
                name: "dummy/white",
                format: blade::TextureFormat::Rgba8Unorm,
                size,
                array_layer_count: 1,
                mip_level_count: 1,
                dimension: blade::TextureDimension::D2,
                usage: blade::TextureUsage::COPY | blade::TextureUsage::RESOURCE,
            });
            let white_view = gpu.create_texture_view(blade::TextureViewDesc {
                name: "dummy/white",
                texture: white_texture,
                format: blade::TextureFormat::Rgba8Unorm,
                dimension: blade::ViewDimension::D2,
                subresources: &blade::TextureSubresources::default(),
            });
            encoder.init_texture(white_texture);
            DummyResources {
                size,
                white_texture,
                white_view,
            }
        };

        let samplers = Samplers {
            linear: gpu.create_sampler(blade::SamplerDesc {
                name: "linear",
                address_modes: [blade::AddressMode::ClampToEdge; 3],
                mag_filter: blade::FilterMode::Linear,
                min_filter: blade::FilterMode::Linear,
                mipmap_filter: blade::FilterMode::Linear,
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
            acceleration_structure: blade::AccelerationStructure::default(),
            dummy,
            hit_buffer: blade::Buffer::default(),
            vertex_buffers: blade::BufferArray::new(),
            index_buffers: blade::BufferArray::new(),
            textures: blade::TextureArray::new(),
            samplers,
            reservoir_buffer,
            debug: DebugRender {
                capacity: config.max_debug_lines,
                buffer: debug_buffer,
                pipeline: sp.debug,
            },
            is_tlas_dirty: true,
            are_reservoirs_dirty: true,
            screen_size: config.screen_size,
            frame_index: 0,
        }
    }

    pub fn destroy(&mut self, gpu: &blade::Context) {
        // scene
        for texture in self.scene.textures.drain(..) {
            gpu.destroy_texture_view(texture.view);
            gpu.destroy_texture(texture.texture);
        }
        for mut object in self.scene.objects.drain(..) {
            for geometry in object.geometries.drain(..) {
                gpu.destroy_buffer(geometry.vertex_buf);
                if geometry.index_type.is_some() {
                    gpu.destroy_buffer(geometry.index_buf);
                }
            }
            gpu.destroy_acceleration_structure(object.acceleration_structure);
        }
        // internal resources
        self.targets.destroy(gpu);
        if self.hit_buffer != blade::Buffer::default() {
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
        gpu.destroy_buffer(self.reservoir_buffer);
    }

    pub fn merge_scene(&mut self, scene: super::Scene) {
        self.scene = scene;
    }

    pub fn hot_reload(&mut self, gpu: &blade::Context, sync_point: &blade::SyncPoint) -> bool {
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

    pub fn prepare(
        &mut self,
        command_encoder: &mut blade::CommandEncoder,
        gpu: &blade::Context,
        temp_buffers: &mut Vec<blade::Buffer>,
        enable_debug: bool,
    ) {
        if self.is_tlas_dirty {
            self.is_tlas_dirty = false;
            if self.acceleration_structure != blade::AccelerationStructure::default() {
                temp_buffers.push(self.hit_buffer);
                //TODO: delay this or stall the GPU
                gpu.destroy_acceleration_structure(self.acceleration_structure);
            }

            let (tlas, geometry_count) = self.scene.build_top_level_acceleration_structure(
                command_encoder,
                gpu,
                temp_buffers,
            );
            self.acceleration_structure = tlas;
            log::info!("Preparing ray tracing with {geometry_count} geometries in total");
            let mut transfers = command_encoder.transfer();

            {
                // init the dummy
                let staging = gpu.create_buffer(blade::BufferDesc {
                    name: "dummy staging",
                    size: 4,
                    memory: blade::Memory::Upload,
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
                let staging = gpu.create_buffer(blade::BufferDesc {
                    name: "debug buf staging",
                    size,
                    memory: blade::Memory::Upload,
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
                self.hit_buffer = gpu.create_buffer(blade::BufferDesc {
                    name: "hit entries",
                    size: hit_size,
                    memory: blade::Memory::Device,
                });
                let staging = gpu.create_buffer(blade::BufferDesc {
                    name: "hit staging",
                    size: hit_size,
                    memory: blade::Memory::Upload,
                });
                temp_buffers.push(staging);
                transfers.copy_buffer_to_buffer(staging.at(0), self.hit_buffer.at(0), hit_size);
                staging
            };

            self.textures.clear();
            let dummy_white = self.textures.alloc(self.dummy.white_view);
            let mut texture_indices = Vec::with_capacity(self.scene.textures.len());
            for texture in self.scene.textures.iter() {
                texture_indices.push(self.textures.alloc(texture.view));
            }

            self.vertex_buffers.clear();
            self.index_buffers.clear();
            let mut geometry_index = 0;
            for object in self.scene.objects.iter() {
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
                for geometry in object.geometries.iter() {
                    let material = &self.scene.materials[geometry.material_index];
                    let hit_entry = HitEntry {
                        index_buf: match geometry.index_type {
                            Some(_) => self.index_buffers.alloc(geometry.index_buf.at(0)),
                            None => !0,
                        },
                        vertex_buf: self.vertex_buffers.alloc(geometry.vertex_buf.at(0)),
                        rotation,
                        base_color_texture: if material.base_color_texture_index == !0 {
                            dummy_white
                        } else {
                            texture_indices[material.base_color_texture_index]
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
        let mut transfer = command_encoder.transfer();
        if enable_debug {
            // reset the debug line count
            transfer.fill_buffer(self.debug.buffer.at(4), 4, 0);
            transfer.fill_buffer(self.debug.buffer.at(20), 4, 1);
        } else {
            transfer.fill_buffer(self.debug.buffer.at(20), 4, 0);
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

    pub fn ray_trace(
        &self,
        command_encoder: &mut blade::CommandEncoder,
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

    pub fn blit(&self, pass: &mut blade::RenderCommandEncoder, camera: &super::Camera) {
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
}
