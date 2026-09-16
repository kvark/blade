//! wgpu custom backend implemented on `blade-graphics`.
//!
//! This is study scaffolding: the public wgpu API is unchanged, wgpu-core is
//! skipped, and Blade issues its usual pass-boundary barriers. CPU cost of the
//! two backends is compared at the `blade-wgpu::*` profiling scopes.

mod command;
mod conv;
mod lifecycle;

use std::{
    fmt,
    future::ready,
    pin::Pin,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use blade_graphics as gpu;
use wgpu::custom::{
    AdapterInterface, BindGroupInterface, BindGroupLayoutInterface, BlasInterface,
    BufferInterface, BufferMapCallback, BufferMappedRangeInterface, CommandBufferInterface,
    CommandEncoderInterface, ComputePassInterface, ComputePipelineInterface, DeviceInterface,
    DispatchAdapter, DispatchBindGroup, DispatchBindGroupLayout, DispatchBlas,
    DispatchBuffer, DispatchBufferMappedRange, DispatchCommandBuffer, DispatchCommandEncoder,
    DispatchComputePass, DispatchComputePipeline, DispatchDevice, DispatchExternalTexture,
    DispatchPipelineCache, DispatchPipelineLayout, DispatchQuerySet, DispatchQueue,
    DispatchQueueWriteBuffer,
    DispatchRenderBundle, DispatchRenderBundleEncoder, DispatchRenderPass, DispatchRenderPipeline,
    DispatchSampler, DispatchShaderModule, DispatchSurface, DispatchTexture, DispatchTextureView,
    DispatchTlas, ExternalTextureInterface, InstanceInterface, PipelineCacheInterface,
    PipelineLayoutInterface, QuerySetInterface, QueueInterface, QueueWriteBufferInterface,
    RenderBundleEncoderInterface,
    RenderBundleInterface, RenderPipelineInterface,
    RequestAdapterFuture, RequestDeviceFuture, SamplerInterface, ShaderModuleInterface,
    TextureInterface, TextureViewInterface, TlasInterface,
};

pub use command::{BladeCommandBuffer, BladeCommandEncoder, BladeComputePass, BladeRenderPass};
pub use lifecycle::{
    reset as lifecycle_reset, take as take_encoder_lifecycle, EncoderLifecycle,
};

fn todo(what: &str) -> ! {
    unimplemented!("blade-wgpu: {what}")
}

/// Shared Blade context plus the last submitted work.
pub struct Shared {
    pub context: gpu::Context,
    last_submit: Mutex<Option<gpu::SyncPoint>>,
    map_callbacks: Mutex<Vec<BufferMapCallback>>,
    submit_index: Mutex<u64>,
    pending_inits: Mutex<Vec<gpu::Texture>>,
    keep_alive: Mutex<Vec<gpu::Buffer>>,
    /// Recycled Blade encoders. Creating one allocates two 1 MiB scratch
    /// buffers and descriptor pools; the native Blade loop reuses them via
    /// `start` after waiting for that encoder's last submit. Destroying them
    /// inside `Queue::submit` was the host-cost outlier versus wgpu-core.
    encoder_pool: Mutex<Vec<PooledEncoder>>,
}

struct PooledEncoder {
    encoder: command::SendEnc,
    inflight: Option<gpu::SyncPoint>,
}

impl fmt::Debug for Shared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shared")
            .field("device", &self.context.device_information().device_name)
            .finish()
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        for enc in self.encoder_pool.lock().unwrap().drain(..) {
            let mut encoder = enc.encoder.0;
            let start = std::time::Instant::now();
            self.context.destroy_command_encoder(&mut encoder);
            lifecycle::note_destroy(lifecycle::elapsed_ns(start));
        }
    }
}

/// wgpu `Instance` implementation backed by one Blade context.
#[derive(Debug, Clone)]
pub struct BladeInstance {
    shared: Arc<Shared>,
}

impl BladeInstance {
    /// Create a Blade context and wrap it as a custom wgpu instance.
    pub fn create(desc: &wgpu::InstanceDescriptor) -> Self {
        Self::create_with(desc, false)
    }

    /// `presentation` must be true if the caller will create a swapchain.
    pub fn create_with(desc: &wgpu::InstanceDescriptor, presentation: bool) -> Self {
        profiling::scope!("blade-wgpu::Instance::create");
        let validation = desc.flags.contains(wgpu::InstanceFlags::VALIDATION)
            || desc.flags.contains(wgpu::InstanceFlags::DEBUG);
        let context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation,
                validation,
                ..Default::default()
            })
        }
        .expect("blade-wgpu: failed to init blade-graphics context");
        Self {
            shared: Arc::new(Shared {
                context,
                last_submit: Mutex::new(None),
                map_callbacks: Mutex::new(Vec::new()),
                submit_index: Mutex::new(0),
                pending_inits: Mutex::new(Vec::new()),
                keep_alive: Mutex::new(Vec::new()),
                encoder_pool: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Wrap this backend as a `wgpu::Instance`.
    pub fn into_wgpu(self) -> wgpu::Instance {
        wgpu::Instance::from_custom(self)
    }
}

/// Convenience constructor used by examples and Bevy.
pub fn instance() -> wgpu::Instance {
    BladeInstance::create(&wgpu::InstanceDescriptor::new_without_display_handle()).into_wgpu()
}

impl InstanceInterface for BladeInstance {
    fn new(desc: wgpu::InstanceDescriptor) -> Self
    where
        Self: Sized,
    {
        Self::create(&desc)
    }

    unsafe fn create_surface(
        &self,
        _target: wgpu::SurfaceTargetUnsafe,
    ) -> Result<DispatchSurface, wgpu::CreateSurfaceError> {
        todo("create_surface")
    }

    fn request_adapter(
        &self,
        _options: &wgpu::RequestAdapterOptions<'_, '_>,
    ) -> Pin<Box<dyn RequestAdapterFuture>> {
        Box::pin(ready(Ok(DispatchAdapter::custom(BladeAdapter {
            shared: self.shared.clone(),
        }))))
    }

    fn poll_all_devices(&self, force_wait: bool) -> bool {
        if force_wait {
            wait_for_gpu(&self.shared);
        }
        true
    }

    fn wgsl_language_features(&self) -> wgpu::WgslLanguageFeatures {
        wgpu::WgslLanguageFeatures::empty()
    }

    fn enumerate_adapters(
        &self,
        _backends: wgpu::Backends,
    ) -> Pin<Box<dyn wgpu::custom::EnumerateAdapterFuture>> {
        let adapter = DispatchAdapter::custom(BladeAdapter {
            shared: self.shared.clone(),
        });
        Box::pin(ready(vec![adapter]))
    }
}

#[derive(Debug, Clone)]
pub struct BladeAdapter {
    shared: Arc<Shared>,
}

fn adapter_info(shared: &Shared) -> wgpu::AdapterInfo {
    let info = shared.context.device_information();
    let mut out = wgpu::AdapterInfo::new(
        if info.is_software_emulated {
            wgpu::DeviceType::Cpu
        } else {
            wgpu::DeviceType::DiscreteGpu
        },
        if cfg!(target_vendor = "apple") {
            wgpu::Backend::Metal
        } else {
            wgpu::Backend::Vulkan
        },
    );
    out.name = info.device_name.clone();
    out.driver = info.driver_name.clone();
    out.driver_info = info.driver_info.clone();
    out
}

impl AdapterInterface for BladeAdapter {
    fn request_device(
        &self,
        _desc: &wgpu::DeviceDescriptor<'_>,
    ) -> Pin<Box<dyn RequestDeviceFuture>> {
        let device = DispatchDevice::custom(BladeDevice {
            shared: self.shared.clone(),
        });
        let queue = DispatchQueue::custom(BladeQueue {
            shared: self.shared.clone(),
        });
        Box::pin(ready(Ok((device, queue))))
    }

    fn is_surface_supported(&self, _surface: &DispatchSurface) -> bool {
        false
    }

    fn features(&self) -> wgpu::Features {
        conv::advertised_features()
    }

    fn limits(&self) -> wgpu::Limits {
        conv::advertised_limits()
    }

    fn downlevel_capabilities(&self) -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities::default()
    }

    fn get_info(&self) -> wgpu::AdapterInfo {
        adapter_info(&self.shared)
    }

    fn get_texture_format_features(
        &self,
        _format: wgpu::TextureFormat,
    ) -> wgpu::TextureFormatFeatures {
        wgpu::TextureFormatFeatures {
            allowed_usages: wgpu::TextureUsages::all(),
            flags: wgpu::TextureFormatFeatureFlags::empty(),
        }
    }

    fn get_presentation_timestamp(&self) -> wgpu::PresentationTimestamp {
        wgpu::PresentationTimestamp::INVALID_TIMESTAMP
    }

    fn cooperative_matrix_properties(&self) -> Vec<wgpu::CooperativeMatrixProperties> {
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct BladeDevice {
    shared: Arc<Shared>,
}

fn wait_for_gpu(shared: &Shared) {
    if let Some(sp) = shared.last_submit.lock().unwrap().clone() {
        let _ = shared.context.wait_for(&sp, !0);
    }
    for callback in shared.map_callbacks.lock().unwrap().drain(..) {
        callback(Ok(()));
    }
    for buf in shared.keep_alive.lock().unwrap().drain(..) {
        shared.context.destroy_buffer(buf);
    }
}

fn map_memory(usage: wgpu::BufferUsages, mapped_at_creation: bool) -> gpu::Memory {
    // Host-visible for Queue::write_buffer without a staging path. Download
    // is only for MAP_READ targets that the GPU writes and the CPU reads.
    if usage.contains(wgpu::BufferUsages::MAP_READ) && !mapped_at_creation {
        gpu::Memory::Download
    } else {
        let _ = mapped_at_creation;
        gpu::Memory::Shared
    }
}

fn acquire_encoder(shared: &Shared) -> gpu::CommandEncoder {
    let start = std::time::Instant::now();
    let popped = shared.encoder_pool.lock().unwrap().pop();
    let (mut encoder, allocated) = match popped {
        Some(pooled) => {
            if let Some(sp) = pooled.inflight {
                let _ = shared.context.wait_for(&sp, !0);
            }
            (pooled.encoder.0, false)
        }
        None => (
            shared.context.create_command_encoder(gpu::CommandEncoderDesc {
                name: "blade-wgpu-encoder",
                buffer_count: 2,
                manual_barriers: false,
                barrier_scope: gpu::BarrierScope::PassKind,
            }),
            true,
        ),
    };
    lifecycle::note_create(lifecycle::elapsed_ns(start), allocated);
    encoder.start();
    init_pending(shared, &mut encoder);
    encoder
}

fn recycle_encoder(shared: &Shared, encoder: gpu::CommandEncoder, inflight: gpu::SyncPoint) {
    shared.encoder_pool.lock().unwrap().push(PooledEncoder {
        encoder: command::SendEnc(encoder),
        inflight: Some(inflight),
    });
}

fn submit_encoder(shared: &Shared, encoder: &mut gpu::CommandEncoder) -> gpu::SyncPoint {
    let start = std::time::Instant::now();
    let sp = shared.context.submit(encoder);
    lifecycle::note_vk_submit(lifecycle::elapsed_ns(start));
    sp
}

fn init_pending(shared: &Shared, encoder: &mut gpu::CommandEncoder) {
    for texture in shared.pending_inits.lock().unwrap().drain(..) {
        encoder.init_texture(texture);
    }
}

fn leaked_name(binding: u32) -> &'static str {
    Box::leak(format!("b{binding}").into_boxed_str())
}

fn leaked_name_loc(location: u32) -> &'static str {
    Box::leak(format!("l{location}").into_boxed_str())
}

fn map_binding_ty(ty: &wgpu::BindingType) -> gpu::ShaderBinding {
    match *ty {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { .. },
            ..
        } => gpu::ShaderBinding::Buffer,
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            ..
        } => gpu::ShaderBinding::UniformBuffer,
        wgpu::BindingType::Sampler(_) => gpu::ShaderBinding::Sampler,
        wgpu::BindingType::Texture { .. } | wgpu::BindingType::StorageTexture { .. } => {
            gpu::ShaderBinding::Texture
        }
        wgpu::BindingType::AccelerationStructure { .. } => gpu::ShaderBinding::AccelerationStructure,
        wgpu::BindingType::ExternalTexture => todo("ExternalTexture bindings"),
    }
}

fn map_binding(entry: &wgpu::BindGroupLayoutEntry) -> gpu::ShaderBinding {
    let base = map_binding_ty(&entry.ty);
    match (base, entry.count) {
        (gpu::ShaderBinding::Texture, Some(n)) => gpu::ShaderBinding::TextureArray { count: n.get() },
        (gpu::ShaderBinding::Buffer, Some(n)) => gpu::ShaderBinding::BufferArray { count: n.get() },
        (other, _) => other,
    }
}

struct LayoutParts {
    layout: gpu::ShaderDataLayout,
    dynamic: Vec<u32>,
}

fn layout_from_entries(entries: &[wgpu::BindGroupLayoutEntry]) -> LayoutParts {
    if entries.is_empty() {
        return LayoutParts {
            layout: gpu::ShaderDataLayout {
                bindings: Vec::new(),
            },
            dynamic: Vec::new(),
        };
    }
    let max = entries.iter().map(|e| e.binding).max().unwrap_or(0);
    let mut bindings = vec![("unused", gpu::ShaderBinding::Unused); max as usize + 1];
    let mut dynamic = Vec::new();
    for entry in entries {
        bindings[entry.binding as usize] = (leaked_name(entry.binding), map_binding(entry));
        if let wgpu::BindingType::Buffer {
            has_dynamic_offset: true,
            ..
        } = entry.ty
        {
            dynamic.push(entry.binding);
        }
    }
    dynamic.sort_unstable();
    LayoutParts {
        layout: gpu::ShaderDataLayout { bindings },
        dynamic,
    }
}

impl DeviceInterface for BladeDevice {
    fn features(&self) -> wgpu::Features {
        conv::advertised_features()
    }
    fn limits(&self) -> wgpu::Limits {
        conv::advertised_limits()
    }
    fn adapter_info(&self) -> wgpu::AdapterInfo {
        adapter_info(&self.shared)
    }

    fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor<'_>,
        _shader_bound_checks: wgpu::ShaderRuntimeChecks,
    ) -> DispatchShaderModule {
        profiling::scope!("blade-wgpu::create_shader_module");
        let source = match desc.source {
            wgpu::ShaderSource::Wgsl(ref src) => src.as_ref(),
            _ => todo("non-WGSL shader source"),
        };
        let shader = self.shared.context.create_shader(gpu::ShaderDesc {
            source,
            naga_module: None,
        });
        DispatchShaderModule::custom(BladeShaderModule {
            shader: Arc::new(shader),
        })
    }

    unsafe fn create_shader_module_passthrough(
        &self,
        _desc: &wgpu::ShaderModuleDescriptorPassthrough<'_>,
    ) -> DispatchShaderModule {
        todo("create_shader_module_passthrough")
    }

    fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor<'_>,
    ) -> DispatchBindGroupLayout {
        let parts = layout_from_entries(desc.entries);
        DispatchBindGroupLayout::custom(BladeBindGroupLayout {
            layout: parts.layout,
            dynamic: parts.dynamic,
        })
    }

    fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor<'_>) -> DispatchBindGroup {
        profiling::scope!("blade-wgpu::create_bind_group");
        let bgl = desc
            .layout
            .as_custom::<BladeBindGroupLayout>()
            .expect("bind group layout is not blade-wgpu");
        let mut slots = Vec::with_capacity(desc.entries.len());
        for entry in desc.entries {
            match entry.resource {
                wgpu::BindingResource::Buffer(ref bb) => {
                    let buf = bb
                        .buffer
                        .as_custom::<BladeBuffer>()
                        .expect("bind group buffer is not blade-wgpu");
                    let size = bb.size.map(|s| s.get()).unwrap_or(0);
                    let uniform = matches!(
                        bgl.layout.bindings.get(entry.binding as usize),
                        Some((_, gpu::ShaderBinding::UniformBuffer))
                    );
                    let dynamic = bgl.dynamic.binary_search(&entry.binding).is_ok();
                    slots.push(Slot::Buffer {
                        binding: entry.binding,
                        piece: buf.raw.at(bb.offset),
                        size,
                        dynamic,
                        uniform,
                    });
                }
                wgpu::BindingResource::TextureView(view) => {
                    let view = view
                        .as_custom::<BladeTextureView>()
                        .expect("bind group view is not blade-wgpu");
                    slots.push(Slot::Texture {
                        binding: entry.binding,
                        view: view.raw,
                    });
                }
                wgpu::BindingResource::Sampler(sampler) => {
                    let sampler = sampler
                        .as_custom::<BladeSampler>()
                        .expect("bind group sampler is not blade-wgpu");
                    slots.push(Slot::Sampler {
                        binding: entry.binding,
                        sampler: sampler.raw,
                    });
                }
                _ => todo("bind group resource"),
            }
        }
        DispatchBindGroup::custom(BladeBindGroup { slots })
    }

    fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor<'_>,
    ) -> DispatchPipelineLayout {
        let mut layouts = Vec::new();
        let mut dynamics = Vec::new();
        for bgl in desc.bind_group_layouts {
            let bgl = bgl
                .as_ref()
                .expect("null bind group layout")
                .as_custom::<BladeBindGroupLayout>()
                .expect("pipeline layout BGL is not blade-wgpu");
            layouts.push(bgl.layout.clone());
            dynamics.push(bgl.dynamic.clone());
        }
        DispatchPipelineLayout::custom(BladePipelineLayout { layouts, dynamics })
    }

    fn create_render_pipeline(
        &self,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> DispatchRenderPipeline {
        profiling::scope!("blade-wgpu::create_render_pipeline");
        let vs_mod = desc
            .vertex
            .module
            .as_custom::<BladeShaderModule>()
            .expect("vertex shader is not blade-wgpu");
        let owned_layouts: Vec<gpu::ShaderDataLayout> = match desc.layout {
            Some(layout) => layout
                .as_custom::<BladePipelineLayout>()
                .expect("pipeline layout is not blade-wgpu")
                .layouts
                .clone(),
            None => Vec::new(),
        };
        let layout_refs: Vec<&gpu::ShaderDataLayout> = owned_layouts.iter().collect();

        let mut vertex_layouts: Vec<gpu::VertexLayout> = Vec::new();
        for (slot, buf_layout) in desc.vertex.buffers.iter().enumerate() {
            let Some(buf_layout) = buf_layout else {
                continue;
            };
            let mut attributes = Vec::new();
            for attr in buf_layout.attributes {
                attributes.push((
                    leaked_name_loc(attr.shader_location),
                    gpu::VertexAttribute {
                        offset: attr.offset as u32,
                        format: conv::vertex_format(attr.format),
                    },
                ));
            }
            vertex_layouts.push(gpu::VertexLayout {
                attributes,
                stride: buf_layout.array_stride as u32,
            });
            let _ = slot;
        }
        // Rebuild fetches after layouts are owned.
        let fetch_states: Vec<gpu::VertexFetchState> = vertex_layouts
            .iter()
            .zip(desc.vertex.buffers.iter().flatten())
            .map(|(layout, b)| gpu::VertexFetchState {
                layout,
                instanced: b.step_mode == wgpu::VertexStepMode::Instance,
            })
            .collect();

        let vs_entry = vs_mod
            .shader
            .resolve_vertex_entry_point(desc.vertex.entry_point);
        let color_targets: Vec<gpu::ColorTargetState> = desc
            .fragment
            .as_ref()
            .map(|fs| {
                fs.targets
                    .iter()
                    .filter_map(|t| {
                        t.as_ref().map(|ct| gpu::ColorTargetState {
                            format: conv::texture_format(ct.format),
                            blend: ct.blend.map(|b| gpu::BlendState {
                                color: conv::blend_component(b.color),
                                alpha: conv::blend_component(b.alpha),
                            }),
                            write_mask: conv::color_writes(ct.write_mask),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let depth_stencil = desc.depth_stencil.as_ref().map(|ds| gpu::DepthStencilState {
            format: conv::texture_format(ds.format),
            depth_write_enabled: ds.depth_write_enabled.unwrap_or(false),
            depth_compare: conv::compare(ds.depth_compare.unwrap_or(wgpu::CompareFunction::Always)),
            stencil: gpu::StencilState {
                front: conv::stencil_face(ds.stencil.front),
                back: conv::stencil_face(ds.stencil.back),
                read_mask: ds.stencil.read_mask,
                write_mask: ds.stencil.write_mask,
            },
            bias: gpu::DepthBiasState {
                constant: ds.bias.constant,
                slope_scale: ds.bias.slope_scale,
                clamp: ds.bias.clamp,
            },
        });

        let fs_mod;
        let fragment = if let Some(fs) = &desc.fragment {
            fs_mod = fs
                .module
                .as_custom::<BladeShaderModule>()
                .expect("fragment shader is not blade-wgpu");
            Some(fs_mod.shader.at(fs_mod.shader.resolve_fragment_entry_point(fs.entry_point)))
        } else {
            fs_mod = vs_mod;
            None
        };
        let _ = &fs_mod;

        let pipeline = self
            .shared
            .context
            .create_render_pipeline(gpu::RenderPipelineDesc {
                name: desc.label.unwrap_or("blade-wgpu-render"),
                data_layouts: &layout_refs,
                vertex: vs_mod.shader.at(vs_entry),
                vertex_fetches: &fetch_states,
                primitive: gpu::PrimitiveState {
                    topology: conv::topology(desc.primitive.topology),
                    front_face: conv::front_face(desc.primitive.front_face),
                    cull_mode: desc.primitive.cull_mode.map(conv::cull),
                    unclipped_depth: desc.primitive.unclipped_depth,
                    wireframe: desc.primitive.polygon_mode == wgpu::PolygonMode::Line,
                },
                depth_stencil,
                fragment,
                color_targets: &color_targets,
                multisample_state: gpu::MultisampleState {
                    sample_count: desc.multisample.count,
                    sample_mask: desc.multisample.mask,
                    alpha_to_coverage: desc.multisample.alpha_to_coverage_enabled,
                },
            });
        let dynamics = match desc.layout {
            Some(layout) => layout
                .as_custom::<BladePipelineLayout>()
                .map(|l| l.dynamics.clone())
                .unwrap_or_default(),
            None => Vec::new(),
        };
        DispatchRenderPipeline::custom(BladeRenderPipeline {
            pipeline: Arc::new(pipeline),
            layouts: owned_layouts,
            dynamics,
        })
    }

    fn create_mesh_pipeline(
        &self,
        _desc: &wgpu::MeshPipelineDescriptor<'_>,
    ) -> DispatchRenderPipeline {
        todo("create_mesh_pipeline")
    }

    fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> DispatchComputePipeline {
        profiling::scope!("blade-wgpu::create_compute_pipeline");
        let module = desc
            .module
            .as_custom::<BladeShaderModule>()
            .expect("shader module is not blade-wgpu");
        let (owned_layouts, dynamics) = match desc.layout {
            Some(layout) => {
                let layout = layout
                    .as_custom::<BladePipelineLayout>()
                    .expect("pipeline layout is not blade-wgpu");
                (layout.layouts.clone(), layout.dynamics.clone())
            }
            None => (Vec::new(), Vec::new()),
        };
        let layout_refs: Vec<&gpu::ShaderDataLayout> = owned_layouts.iter().collect();
        let entry = module
            .shader
            .resolve_compute_entry_point(desc.entry_point);
        let pipeline = self
            .shared
            .context
            .create_compute_pipeline(gpu::ComputePipelineDesc {
                name: desc.label.unwrap_or("blade-wgpu-compute"),
                data_layouts: &layout_refs,
                compute: module.shader.at(entry),
            });
        DispatchComputePipeline::custom(BladeComputePipeline {
            pipeline: Arc::new(pipeline),
            layouts: owned_layouts,
            dynamics,
        })
    }

    unsafe fn create_pipeline_cache(
        &self,
        _desc: &wgpu::PipelineCacheDescriptor<'_>,
    ) -> DispatchPipelineCache {
        todo("create_pipeline_cache")
    }

    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> DispatchBuffer {
        profiling::scope!("blade-wgpu::create_buffer");
        let memory = map_memory(desc.usage, desc.mapped_at_creation);
        let raw = self.shared.context.create_buffer(gpu::BufferDesc {
            name: desc.label.unwrap_or("blade-wgpu-buffer"),
            size: desc.size.max(4),
            memory,
        });
        if !raw.data().is_null() {
            unsafe {
                std::ptr::write_bytes(raw.data(), 0, desc.size as usize);
            }
        }
        DispatchBuffer::custom(BladeBuffer {
            raw,
            size: desc.size,
            usage: desc.usage,
            mapped: Mutex::new(desc.mapped_at_creation),
            destroyed: Mutex::new(false),
            shared: self.shared.clone(),
        })
    }

    fn create_texture(&self, desc: &wgpu::TextureDescriptor<'_>) -> DispatchTexture {
        profiling::scope!("blade-wgpu::create_texture");
        let format = conv::texture_format(desc.format);
        let raw = self.shared.context.create_texture(gpu::TextureDesc {
            name: desc.label.unwrap_or("blade-wgpu-texture"),
            format,
            size: conv::extent3d(desc.size),
            array_layer_count: if desc.dimension == wgpu::TextureDimension::D3 {
                1
            } else {
                desc.size.depth_or_array_layers.max(1)
            },
            mip_level_count: desc.mip_level_count.max(1),
            sample_count: desc.sample_count.max(1),
            dimension: conv::texture_dimension(desc.dimension),
            usage: conv::texture_usage(desc.usage),
            external: None,
        });
        self.shared.pending_inits.lock().unwrap().push(raw);
        DispatchTexture::custom(BladeTexture {
            raw,
            size: desc.size,
            format: desc.format,
            usage: desc.usage,
            mip_level_count: desc.mip_level_count.max(1),
            sample_count: desc.sample_count.max(1),
            dimension: desc.dimension,
            shared: self.shared.clone(),
            destroyed: Mutex::new(false),
        })
    }
    fn create_external_texture(
        &self,
        _desc: &wgpu::ExternalTextureDescriptor<'_>,
        _planes: &[&wgpu::TextureView],
    ) -> DispatchExternalTexture {
        todo("create_external_texture")
    }
    fn create_blas(
        &self,
        _desc: &wgpu::CreateBlasDescriptor<'_>,
        _sizes: wgpu::BlasGeometrySizeDescriptors,
    ) -> (Option<u64>, DispatchBlas) {
        todo("create_blas")
    }
    fn create_tlas(&self, _desc: &wgpu::CreateTlasDescriptor<'_>) -> DispatchTlas {
        todo("create_tlas")
    }
    fn create_sampler(&self, desc: &wgpu::SamplerDescriptor<'_>) -> DispatchSampler {
        let raw = self.shared.context.create_sampler(gpu::SamplerDesc {
            name: desc.label.unwrap_or("blade-wgpu-sampler"),
            address_modes: [
                conv::address_mode(desc.address_mode_u),
                conv::address_mode(desc.address_mode_v),
                conv::address_mode(desc.address_mode_w),
            ],
            mag_filter: conv::filter_mode(desc.mag_filter),
            min_filter: conv::filter_mode(desc.min_filter),
            mipmap_filter: conv::mipmap_filter(desc.mipmap_filter),
            lod_min_clamp: desc.lod_min_clamp,
            lod_max_clamp: Some(desc.lod_max_clamp),
            compare: desc.compare.map(conv::compare),
            anisotropy_clamp: desc.anisotropy_clamp as u32,
            border_color: desc.border_color.map(|c| match c {
                wgpu::SamplerBorderColor::TransparentBlack => gpu::TextureColor::TransparentBlack,
                wgpu::SamplerBorderColor::OpaqueBlack => gpu::TextureColor::OpaqueBlack,
                wgpu::SamplerBorderColor::OpaqueWhite => gpu::TextureColor::White,
                wgpu::SamplerBorderColor::Zero => gpu::TextureColor::TransparentBlack,
            }),
        });
        DispatchSampler::custom(BladeSampler { raw })
    }
    fn create_query_set(&self, desc: &wgpu::QuerySetDescriptor<'_>) -> DispatchQuerySet {
        DispatchQuerySet::custom(BladeQuerySet {
            ty: desc.ty,
            count: desc.count,
        })
    }

    fn create_command_encoder(
        &self,
        desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> DispatchCommandEncoder {
        profiling::scope!("blade-wgpu::create_command_encoder");
        let _ = desc;
        let encoder = acquire_encoder(&self.shared);
        DispatchCommandEncoder::custom(BladeCommandEncoder::new(encoder))
    }

    fn create_render_bundle_encoder(
        &self,
        _desc: &wgpu::RenderBundleEncoderDescriptor<'_>,
    ) -> DispatchRenderBundleEncoder {
        todo("create_render_bundle_encoder")
    }

    fn set_device_lost_callback(&self, _device_lost_callback: wgpu::custom::BoxDeviceLostCallback) {}
    fn on_uncaptured_error(&self, _handler: Arc<dyn wgpu::UncapturedErrorHandler>) {}
    fn push_error_scope(&self, _filter: wgpu::ErrorFilter) -> u32 {
        0
    }
    fn pop_error_scope(&self, _index: u32) -> Pin<Box<dyn wgpu::custom::PopErrorScopeFuture>> {
        Box::pin(ready(None))
    }
    unsafe fn start_graphics_debugger_capture(&self) {}
    unsafe fn stop_graphics_debugger_capture(&self) {}

    fn poll(
        &self,
        poll_type: wgpu::wgt::PollType<u64>,
    ) -> Result<wgpu::PollStatus, wgpu::PollError> {
        profiling::scope!("blade-wgpu::Device::poll");
        match poll_type {
            wgpu::wgt::PollType::Poll => Ok(wgpu::PollStatus::Poll),
            wgpu::wgt::PollType::Wait { .. } => {
                wait_for_gpu(&self.shared);
                Ok(wgpu::PollStatus::QueueEmpty)
            }
        }
    }

    fn get_internal_counters(&self) -> wgpu::InternalCounters {
        wgpu::InternalCounters {
            core: Default::default(),
            hal: Default::default(),
        }
    }
    fn generate_allocator_report(&self) -> Option<wgpu::AllocatorReport> {
        None
    }
    fn destroy(&self) {}
}

#[derive(Debug, Clone)]
pub struct BladeQueue {
    shared: Arc<Shared>,
}

impl QueueInterface for BladeQueue {
    fn write_buffer(&self, buffer: &DispatchBuffer, offset: wgpu::BufferAddress, data: &[u8]) {
        profiling::scope!("blade-wgpu::Queue::write_buffer");
        let buf = buffer
            .as_custom::<BladeBuffer>()
            .expect("write_buffer: not blade-wgpu");
        let ptr = buf.raw.data();
        assert!(
            !ptr.is_null(),
            "blade-wgpu: Queue::write_buffer requires host-visible memory"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset as usize), data.len());
        }
    }

    fn create_staging_buffer(
        &self,
        size: wgpu::BufferSize,
    ) -> Option<DispatchQueueWriteBuffer> {
        Some(DispatchQueueWriteBuffer::custom(BladeQueueWriteBuffer {
            data: vec![0u8; size.get() as usize].into_boxed_slice(),
        }))
    }
    fn validate_write_buffer(
        &self,
        _buffer: &DispatchBuffer,
        _offset: wgpu::BufferAddress,
        _size: wgpu::BufferSize,
    ) -> Option<()> {
        Some(())
    }
    fn write_staging_buffer(
        &self,
        buffer: &DispatchBuffer,
        offset: wgpu::BufferAddress,
        staging_buffer: DispatchQueueWriteBuffer,
    ) {
        profiling::scope!("blade-wgpu::Queue::write_staging_buffer");
        let staging = staging_buffer
            .as_custom::<BladeQueueWriteBuffer>()
            .expect("staging buffer is not blade-wgpu");
        self.write_buffer(buffer, offset, &staging.data);
    }
    fn write_texture(
        &self,
        texture: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        data_layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        profiling::scope!("blade-wgpu::Queue::write_texture");
        let tex = texture
            .texture
            .as_custom::<BladeTexture>()
            .expect("write_texture: not blade-wgpu");
        let bytes_per_row = data_layout.bytes_per_row.unwrap_or(size.width * 4);
        let staging = self.shared.context.create_buffer(gpu::BufferDesc {
            name: "write-texture-staging",
            size: data.len().max(4) as u64,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), staging.data(), data.len());
        }
        let mut encoder = acquire_encoder(&self.shared);
        {
            let mut pass = encoder.transfer("write-texture");
            pass.copy_buffer_to_texture(
                staging.at(data_layout.offset),
                bytes_per_row,
                gpu::TexturePiece {
                    texture: tex.raw,
                    mip_level: texture.mip_level,
                    array_layer: texture.origin.z,
                    origin: [texture.origin.x, texture.origin.y, 0],
                },
                conv::extent3d(wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: size.depth_or_array_layers,
                }),
            );
        }
        let sp = submit_encoder(&self.shared, &mut encoder);
        *self.shared.last_submit.lock().unwrap() = Some(sp.clone());
        self.shared.keep_alive.lock().unwrap().push(staging);
        recycle_encoder(&self.shared, encoder, sp);
    }

    fn submit(
        &self,
        command_buffers: &mut dyn Iterator<Item = DispatchCommandBuffer>,
    ) -> u64 {
        profiling::scope!("blade-wgpu::Queue::submit");
        let mut last = *self.shared.submit_index.lock().unwrap();
        for cb in command_buffers {
            let blade = cb
                .as_custom::<BladeCommandBuffer>()
                .expect("submit: command buffer is not blade-wgpu");
            let mut encoder = blade
                .take_encoder()
                .expect("command buffer already submitted");
            let sp = submit_encoder(&self.shared, &mut encoder);
            *self.shared.last_submit.lock().unwrap() = Some(sp.clone());
            recycle_encoder(&self.shared, encoder, sp);
            last += 1;
        }
        *self.shared.submit_index.lock().unwrap() = last;
        last
    }

    fn get_timestamp_period(&self) -> f32 {
        1.0
    }
    fn on_submitted_work_done(&self, callback: wgpu::custom::BoxSubmittedWorkDoneCallback) {
        wait_for_gpu(&self.shared);
        callback();
    }
    fn compact_blas(&self, _blas: &DispatchBlas) -> (Option<u64>, DispatchBlas) {
        todo("compact_blas")
    }
    fn present(&self, _detail: &wgpu::custom::DispatchSurfaceOutputDetail) {
        todo("present")
    }
}

pub struct BladeShaderModule {
    shader: Arc<gpu::Shader>,
}

impl fmt::Debug for BladeShaderModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BladeShaderModule").finish()
    }
}

impl ShaderModuleInterface for BladeShaderModule {
    fn get_compilation_info(&self) -> Pin<Box<dyn wgpu::custom::ShaderCompilationInfoFuture>> {
        Box::pin(ready(wgpu::CompilationInfo {
            messages: Vec::new(),
        }))
    }
}

#[derive(Debug, Clone)]
pub struct BladeBindGroupLayout {
    layout: gpu::ShaderDataLayout,
    dynamic: Vec<u32>,
}

impl BindGroupLayoutInterface for BladeBindGroupLayout {}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Slot {
    Buffer {
        binding: u32,
        piece: gpu::BufferPiece,
        size: u64,
        dynamic: bool,
        uniform: bool,
    },
    Texture {
        binding: u32,
        view: gpu::TextureView,
    },
    Sampler {
        binding: u32,
        sampler: gpu::Sampler,
    },
}

#[derive(Debug)]
pub struct BladeBindGroup {
    pub(crate) slots: Vec<Slot>,
}

impl BindGroupInterface for BladeBindGroup {}

impl gpu::ShaderData for BladeBindGroup {
    fn layout() -> gpu::ShaderDataLayout {
        panic!("blade-wgpu bind groups use the wgpu layout, not ShaderData::layout()")
    }

    fn fill(&self, mut ctx: gpu::PipelineContext) {
        use gpu::ShaderBindable as _;
        for slot in &self.slots {
            match *slot {
                Slot::Buffer {
                    binding,
                    mut piece,
                    size,
                    uniform,
                    ..
                } => {
                    if size != 0 {
                        piece.size = size;
                    } else if uniform {
                        let remaining = piece.buffer.size().saturating_sub(piece.offset);
                        piece.size = remaining.min(65536).max(4);
                    }
                    piece.bind_to(&mut ctx, binding);
                }
                Slot::Texture { binding, view } => view.bind_to(&mut ctx, binding),
                Slot::Sampler { binding, sampler } => sampler.bind_to(&mut ctx, binding),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BladePipelineLayout {
    layouts: Vec<gpu::ShaderDataLayout>,
    dynamics: Vec<Vec<u32>>,
}

impl PipelineLayoutInterface for BladePipelineLayout {}

#[derive(Debug)]
pub struct BladeComputePipeline {
    pub(crate) pipeline: Arc<gpu::ComputePipeline>,
    layouts: Vec<gpu::ShaderDataLayout>,
    dynamics: Vec<Vec<u32>>,
}

impl ComputePipelineInterface for BladeComputePipeline {
    fn get_bind_group_layout(&self, index: u32) -> DispatchBindGroupLayout {
        let i = index as usize;
        DispatchBindGroupLayout::custom(BladeBindGroupLayout {
            layout: self
                .layouts
                .get(i)
                .cloned()
                .unwrap_or_default(),
            dynamic: self.dynamics.get(i).cloned().unwrap_or_default(),
        })
    }
}

#[derive(Debug)]
pub struct BladeBuffer {
    pub(crate) raw: gpu::Buffer,
    pub(crate) size: u64,
    usage: wgpu::BufferUsages,
    mapped: Mutex<bool>,
    destroyed: Mutex<bool>,
    shared: Arc<Shared>,
}

impl BladeBuffer {
    fn destroy_once(&self) {
        let mut destroyed = self.destroyed.lock().unwrap();
        if !*destroyed {
            *destroyed = true;
            self.shared.context.destroy_buffer(self.raw);
        }
    }
}

impl Drop for BladeBuffer {
    fn drop(&mut self) {
        self.destroy_once();
    }
}

impl BufferInterface for BladeBuffer {
    fn map_async(
        &self,
        _mode: wgpu::MapMode,
        _range: std::ops::Range<wgpu::BufferAddress>,
        callback: BufferMapCallback,
    ) {
        *self.mapped.lock().unwrap() = true;
        self.shared.map_callbacks.lock().unwrap().push(callback);
    }

    fn get_mapped_range(
        &self,
        sub_range: std::ops::Range<wgpu::BufferAddress>,
    ) -> Result<DispatchBufferMappedRange, wgpu::MapRangeError> {
        assert!(
            *self.mapped.lock().unwrap(),
            "blade-wgpu: get_mapped_range on an unmapped buffer"
        );
        let ptr = self.raw.data();
        assert!(!ptr.is_null(), "blade-wgpu: buffer is not host-visible");
        let offset = sub_range.start as usize;
        let len = (sub_range.end - sub_range.start) as usize;
        Ok(DispatchBufferMappedRange::custom(BladeMappedRange {
            ptr: unsafe { NonNull::new_unchecked(ptr.add(offset)) },
            len,
        }))
    }

    fn unmap(&self) {
        *self.mapped.lock().unwrap() = false;
    }

    fn destroy(&self) {
        self.destroy_once();
    }

    fn size(&self) -> wgpu::BufferAddress {
        self.size
    }

    fn usage(&self) -> wgpu::BufferUsages {
        self.usage
    }
}

#[derive(Debug)]
struct BladeMappedRange {
    ptr: NonNull<u8>,
    len: usize,
}

unsafe impl Send for BladeMappedRange {}
unsafe impl Sync for BladeMappedRange {}

impl BufferMappedRangeInterface for BladeMappedRange {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn read_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    unsafe fn write_slice(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        unsafe { wgpu::WriteOnly::new(NonNull::slice_from_raw_parts(self.ptr, self.len)) }
    }
}

// Unused-for-compute marker types so the remaining dispatch traits can be named.
macro_rules! empty_debug {
    ($name:ident) => {
        #[derive(Debug)]
        pub struct $name;
    };
}

#[derive(Debug)]
pub struct BladeTexture {
    pub(crate) raw: gpu::Texture,
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    mip_level_count: u32,
    sample_count: u32,
    dimension: wgpu::TextureDimension,
    shared: Arc<Shared>,
    destroyed: Mutex<bool>,
}

impl BladeTexture {
    fn destroy_once(&self) {
        let mut destroyed = self.destroyed.lock().unwrap();
        if !*destroyed {
            *destroyed = true;
            self.shared.context.destroy_texture(self.raw);
        }
    }
}

impl Drop for BladeTexture {
    fn drop(&mut self) {
        self.destroy_once();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BladeTextureView {
    pub(crate) raw: gpu::TextureView,
}

#[derive(Debug, Clone, Copy)]
pub struct BladeSampler {
    pub(crate) raw: gpu::Sampler,
}

#[derive(Debug)]
pub struct BladeQuerySet {
    ty: wgpu::QueryType,
    count: u32,
}

pub struct BladeRenderPipeline {
    pub(crate) pipeline: Arc<gpu::RenderPipeline>,
    layouts: Vec<gpu::ShaderDataLayout>,
    dynamics: Vec<Vec<u32>>,
}

impl fmt::Debug for BladeRenderPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BladeRenderPipeline").finish()
    }
}

#[derive(Debug)]
struct BladeQueueWriteBuffer {
    data: Box<[u8]>,
}

unsafe impl Send for BladeQueueWriteBuffer {}
unsafe impl Sync for BladeQueueWriteBuffer {}

impl QueueWriteBufferInterface for BladeQueueWriteBuffer {
    fn len(&self) -> usize {
        self.data.len()
    }

    unsafe fn write_slice(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        let ptr = NonNull::new(self.data.as_mut_ptr()).expect("staging buffer");
        unsafe { wgpu::WriteOnly::new(NonNull::slice_from_raw_parts(ptr, self.data.len())) }
    }
}

empty_debug!(BladePipelineCache);
empty_debug!(BladeExternalTexture);
empty_debug!(BladeBlas);
empty_debug!(BladeTlas);
empty_debug!(BladeRenderBundle);
empty_debug!(BladeRenderBundleEncoder);

impl TextureInterface for BladeTexture {
    fn create_view(&self, desc: &wgpu::TextureViewDescriptor<'_>) -> DispatchTextureView {
        let format = desc.format.unwrap_or(self.format);
        let dimension = desc.dimension.unwrap_or(match self.dimension {
            wgpu::TextureDimension::D1 => wgpu::TextureViewDimension::D1,
            wgpu::TextureDimension::D2 => wgpu::TextureViewDimension::D2,
            wgpu::TextureDimension::D3 => wgpu::TextureViewDimension::D3,
        });
        let view = self.shared.context.create_texture_view(
            self.raw,
            gpu::TextureViewDesc {
                name: desc.label.unwrap_or("blade-wgpu-view"),
                format: conv::texture_format(format),
                dimension: conv::view_dimension(dimension),
                subresources: &gpu::TextureSubresources {
                    base_mip_level: desc.base_mip_level,
                    mip_level_count: desc
                        .mip_level_count
                        .and_then(std::num::NonZeroU32::new),
                    base_array_layer: desc.base_array_layer,
                    array_layer_count: desc
                        .array_layer_count
                        .and_then(std::num::NonZeroU32::new),
                },
            },
        );
        DispatchTextureView::custom(BladeTextureView { raw: view })
    }
    fn destroy(&self) {
        self.destroy_once();
    }
    fn size(&self) -> wgpu::Extent3d {
        self.size
    }
    fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }
    fn sample_count(&self) -> u32 {
        self.sample_count
    }
    fn dimension(&self) -> wgpu::TextureDimension {
        self.dimension
    }
    fn format(&self) -> wgpu::TextureFormat {
        self.format
    }
    fn usage(&self) -> wgpu::TextureUsages {
        self.usage
    }
}
impl TextureViewInterface for BladeTextureView {}
impl SamplerInterface for BladeSampler {}
impl QuerySetInterface for BladeQuerySet {
    fn destroy(&self) {}
    fn ty(&self) -> wgpu::QueryType {
        self.ty
    }
    fn count(&self) -> u32 {
        self.count
    }
}
impl RenderPipelineInterface for BladeRenderPipeline {
    fn get_bind_group_layout(&self, index: u32) -> DispatchBindGroupLayout {
        let i = index as usize;
        DispatchBindGroupLayout::custom(BladeBindGroupLayout {
            layout: self
                .layouts
                .get(i)
                .cloned()
                .unwrap_or_default(),
            dynamic: self.dynamics.get(i).cloned().unwrap_or_default(),
        })
    }
}
impl PipelineCacheInterface for BladePipelineCache {
    fn get_data(&self) -> Option<Vec<u8>> {
        None
    }
}
impl ExternalTextureInterface for BladeExternalTexture {
    fn destroy(&self) {}
}
impl BlasInterface for BladeBlas {
    fn prepare_compact_async(&self, _callback: wgpu::custom::BlasCompactCallback) {}
    fn ready_for_compaction(&self) -> bool {
        false
    }
}
impl TlasInterface for BladeTlas {}
impl RenderBundleInterface for BladeRenderBundle {}
impl CommandBufferInterface for crate::command::BladeCommandBuffer {}
