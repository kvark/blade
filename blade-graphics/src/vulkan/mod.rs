use ash::{
    extensions::{ext, khr},
    vk,
};
use std::{num::NonZeroU32, ptr, sync::Mutex};

mod command;
mod init;
mod pipeline;
mod resource;

struct Instance {
    core: ash::Instance,
    debug_utils: ext::DebugUtils,
    get_physical_device_properties2: khr::GetPhysicalDeviceProperties2,
}

#[derive(Clone)]
struct RayTracingDevice {
    acceleration_structure: khr::AccelerationStructure,
}

#[derive(Clone)]
struct Device {
    core: ash::Device,
    timeline_semaphore: khr::TimelineSemaphore,
    dynamic_rendering: khr::DynamicRendering,
    ray_tracing: Option<RayTracingDevice>,
}

struct MemoryManager {
    allocator: gpu_alloc::GpuAllocator<vk::DeviceMemory>,
    slab: slab::Slab<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
    valid_ash_memory_types: u32,
}

struct Queue {
    raw: vk::Queue,
    timeline_semaphore: vk::Semaphore,
    present_semaphore: vk::Semaphore,
    last_progress: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    image_index: u32,
    image: vk::Image,
    view: vk::ImageView,
    format: crate::TextureFormat,
    acquire_semaphore: vk::Semaphore,
    target_size: [u16; 2],
}

impl Frame {
    pub fn texture(&self) -> Texture {
        Texture {
            raw: self.image,
            memory_handle: !0,
            target_size: self.target_size,
            format: self.format,
        }
    }

    pub fn texture_view(&self) -> TextureView {
        TextureView {
            raw: self.view,
            target_size: self.target_size,
            aspects: crate::TexelAspects::COLOR,
        }
    }
}

struct Surface {
    raw: vk::SurfaceKHR,
    frames: Vec<Frame>,
    next_semaphore: vk::Semaphore,
    swapchain: vk::SwapchainKHR,
    extension: khr::Swapchain,
}

fn map_timeout(millis: u32) -> u64 {
    if millis == !0 {
        !0
    } else {
        millis as u64 * 1_000_000
    }
}

pub struct Context {
    memory: Mutex<MemoryManager>,
    device: Device,
    queue_family_index: u32,
    queue: Mutex<Queue>,
    surface: Option<Mutex<Surface>>,
    _physical_device: vk::PhysicalDevice,
    naga_flags: naga::back::spv::WriterFlags,
    instance: Instance,
    _entry: ash::Entry,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Buffer {
    raw: vk::Buffer,
    memory_handle: usize,
    mapped_data: *mut u8,
}

impl Default for Buffer {
    fn default() -> Self {
        Self {
            raw: vk::Buffer::null(),
            memory_handle: !0,
            mapped_data: ptr::null_mut(),
        }
    }
}

impl Buffer {
    pub fn data(&self) -> *mut u8 {
        self.mapped_data
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BlockInfo {
    bytes: u8,
    width: u8,
    height: u8,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Texture {
    raw: vk::Image,
    memory_handle: usize,
    target_size: [u16; 2],
    format: crate::TextureFormat,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct TextureView {
    raw: vk::ImageView,
    target_size: [u16; 2],
    aspects: crate::TexelAspects,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Sampler {
    raw: vk::Sampler,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct AccelerationStructure {
    raw: vk::AccelerationStructureKHR,
    buffer: vk::Buffer,
    memory_handle: usize,
}

#[derive(Debug, Default)]
struct DescriptorSetLayout {
    raw: vk::DescriptorSetLayout,
    update_template: vk::DescriptorUpdateTemplate,
    template_size: u32,
    template_offsets: Box<[u32]>,
}

#[derive(Debug)]
struct PipelineLayout {
    raw: vk::PipelineLayout,
    descriptor_set_layouts: Vec<DescriptorSetLayout>,
}

pub struct PipelineContext<'a> {
    update_data: &'a mut [u8],
    template_offsets: &'a [u32],
}

#[derive(Debug)]
pub struct ComputePipeline {
    raw: vk::Pipeline,
    layout: PipelineLayout,
    wg_size: [u32; 3],
}

impl ComputePipeline {
    pub fn get_workgroup_size(&self) -> [u32; 3] {
        self.wg_size
    }
}

pub struct RenderPipeline {
    raw: vk::Pipeline,
    layout: PipelineLayout,
}

#[derive(Clone, Copy, Debug)]
struct CommandBuffer {
    raw: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
}

#[derive(Debug, PartialEq)]
struct Presentation {
    image_index: u32,
    acquire_semaphore: vk::Semaphore,
}

pub struct CommandEncoder {
    pool: vk::CommandPool,
    buffers: Box<[CommandBuffer]>,
    device: Device,
    update_data: Vec<u8>,
    present: Option<Presentation>,
}
pub struct TransferCommandEncoder<'a> {
    raw: vk::CommandBuffer,
    device: &'a Device,
}
pub struct AccelerationStructureCommandEncoder<'a> {
    raw: vk::CommandBuffer,
    device: &'a Device,
}
pub struct ComputeCommandEncoder<'a> {
    cmd_buf: CommandBuffer,
    device: &'a Device,
    update_data: &'a mut Vec<u8>,
}
pub struct RenderCommandEncoder<'a> {
    cmd_buf: CommandBuffer,
    device: &'a Device,
    update_data: &'a mut Vec<u8>,
}
pub struct PipelineEncoder<'a, 'p> {
    cmd_buf: CommandBuffer,
    layout: &'p PipelineLayout,
    bind_point: vk::PipelineBindPoint,
    device: &'a Device,
    update_data: &'a mut Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct SyncPoint {
    progress: u64,
}

#[hidden_trait::expose]
impl crate::traits::CommandDevice for Context {
    type CommandEncoder = CommandEncoder;
    type SyncPoint = SyncPoint;

    fn create_command_encoder(&self, desc: super::CommandEncoderDesc) -> CommandEncoder {
        //TODO: these numbers are arbitrary, needs to be replaced by
        // an abstraction from gpu-alloc, if possible.
        const ROUGH_SET_COUNT: u32 = 60000;
        const DESCRIPTOR_SIZES: &[vk::DescriptorPoolSize] = &[
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                descriptor_count: ROUGH_SET_COUNT * crate::limits::PLAIN_DATA_SIZE,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: ROUGH_SET_COUNT,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 2 * ROUGH_SET_COUNT,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: ROUGH_SET_COUNT,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: ROUGH_SET_COUNT,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: ROUGH_SET_COUNT,
            },
        ];

        let pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe {
            self.device
                .core
                .create_command_pool(&pool_info, None)
                .unwrap()
        };
        let cmd_buf_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .command_buffer_count(desc.buffer_count);
        let cmd_buffers = unsafe {
            self.device
                .core
                .allocate_command_buffers(&cmd_buf_info)
                .unwrap()
        };
        let buffers = cmd_buffers
            .into_iter()
            .map(|raw| {
                if !desc.name.is_empty() {
                    self.set_object_name(vk::ObjectType::COMMAND_BUFFER, raw, desc.name);
                };
                let mut inline_uniform_block_info =
                    vk::DescriptorPoolInlineUniformBlockCreateInfoEXT::builder()
                        .max_inline_uniform_block_bindings(ROUGH_SET_COUNT);
                let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                    .max_sets(ROUGH_SET_COUNT)
                    .pool_sizes(DESCRIPTOR_SIZES)
                    .push_next(&mut inline_uniform_block_info);
                let descriptor_pool = unsafe {
                    self.device
                        .core
                        .create_descriptor_pool(&descriptor_pool_info, None)
                        .unwrap()
                };
                CommandBuffer {
                    raw,
                    descriptor_pool,
                }
            })
            .collect();
        CommandEncoder {
            pool,
            buffers,
            device: self.device.clone(),
            update_data: Vec::new(),
            present: None,
        }
    }

    fn destroy_command_encoder(&self, command_encoder: CommandEncoder) {
        for cmd_buf in command_encoder.buffers.into_iter() {
            let raw_cmd_buffers = [cmd_buf.raw];
            unsafe {
                self.device
                    .core
                    .free_command_buffers(command_encoder.pool, &raw_cmd_buffers);
                self.device
                    .core
                    .destroy_descriptor_pool(cmd_buf.descriptor_pool, None);
            }
        }
        unsafe {
            self.device
                .core
                .destroy_command_pool(command_encoder.pool, None)
        };
    }

    fn submit(&self, encoder: &mut CommandEncoder) -> SyncPoint {
        let raw_cmd_buf = encoder.finish();
        let mut queue = self.queue.lock().unwrap();
        queue.last_progress += 1;
        let progress = queue.last_progress;
        let command_buffers = [raw_cmd_buf];
        let wait_values_all = [0];
        let mut wait_semaphores_all = [vk::Semaphore::null()];
        let wait_stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let signal_semaphores_all = [queue.timeline_semaphore, queue.present_semaphore];
        let signal_values_all = [progress, 0];
        let (num_wait_semaphores, num_signal_sepahores) = match encoder.present {
            Some(ref presentation) => {
                wait_semaphores_all[0] = presentation.acquire_semaphore;
                (1, 2)
            }
            None => (0, 1),
        };
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(&wait_values_all[..num_wait_semaphores])
            .signal_semaphore_values(&signal_values_all[..num_signal_sepahores]);
        let vk_info = vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores_all[..num_wait_semaphores])
            .wait_dst_stage_mask(&wait_stages[..num_wait_semaphores])
            .signal_semaphores(&signal_semaphores_all[..num_signal_sepahores])
            .push_next(&mut timeline_info);
        unsafe {
            self.device
                .core
                .queue_submit(queue.raw, &[vk_info.build()], vk::Fence::null())
                .unwrap();
        }

        if let Some(presentation) = encoder.present.take() {
            let surface = self.surface.as_ref().unwrap().lock().unwrap();
            let swapchains = [surface.swapchain];
            let image_indices = [presentation.image_index];
            let wait_semaphores = [queue.present_semaphore];
            let present_info = vk::PresentInfoKHR::builder()
                .swapchains(&swapchains)
                .image_indices(&image_indices)
                .wait_semaphores(&wait_semaphores);
            unsafe {
                surface
                    .extension
                    .queue_present(queue.raw, &present_info)
                    .unwrap()
            };
        }

        SyncPoint { progress }
    }

    fn wait_for(&self, sp: &SyncPoint, timeout_ms: u32) -> bool {
        //Note: technically we could get away without locking the queue,
        // but also this isn't time-sensitive, so it's fine.
        let timeline_semaphore = self.queue.lock().unwrap().timeline_semaphore;
        let semaphores = [timeline_semaphore];
        let semaphore_values = [sp.progress];
        let wait_info = vk::SemaphoreWaitInfoKHR::builder()
            .semaphores(&semaphores)
            .values(&semaphore_values);
        let timeout_ns = map_timeout(timeout_ms);
        unsafe {
            self.device
                .timeline_semaphore
                .wait_semaphores(&wait_info, timeout_ns)
                .is_ok()
        }
    }
}

fn map_texture_format(format: crate::TextureFormat) -> vk::Format {
    use crate::TextureFormat as Tf;
    match format {
        Tf::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        Tf::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        Tf::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        Tf::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        Tf::Depth32Float => vk::Format::D32_SFLOAT,
    }
}

fn map_aspects(aspects: crate::TexelAspects) -> vk::ImageAspectFlags {
    let mut flags = vk::ImageAspectFlags::empty();
    if aspects.contains(crate::TexelAspects::COLOR) {
        flags |= vk::ImageAspectFlags::COLOR;
    }
    if aspects.contains(crate::TexelAspects::DEPTH) {
        flags |= vk::ImageAspectFlags::DEPTH;
    }
    if aspects.contains(crate::TexelAspects::STENCIL) {
        flags |= vk::ImageAspectFlags::STENCIL;
    }
    flags
}

fn map_extent_3d(extent: &crate::Extent) -> vk::Extent3D {
    vk::Extent3D {
        width: extent.width,
        height: extent.height,
        depth: extent.depth,
    }
}

fn map_subresource_range(
    subresources: &crate::TextureSubresources,
    aspects: crate::TexelAspects,
) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: map_aspects(aspects),
        base_mip_level: subresources.base_mip_level,
        level_count: subresources
            .mip_level_count
            .map_or(vk::REMAINING_MIP_LEVELS, NonZeroU32::get),
        base_array_layer: subresources.base_array_layer,
        layer_count: subresources
            .array_layer_count
            .map_or(vk::REMAINING_ARRAY_LAYERS, NonZeroU32::get),
    }
}

fn map_comparison(fun: crate::CompareFunction) -> vk::CompareOp {
    use crate::CompareFunction as Cf;
    match fun {
        Cf::Never => vk::CompareOp::NEVER,
        Cf::Less => vk::CompareOp::LESS,
        Cf::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        Cf::Equal => vk::CompareOp::EQUAL,
        Cf::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        Cf::Greater => vk::CompareOp::GREATER,
        Cf::NotEqual => vk::CompareOp::NOT_EQUAL,
        Cf::Always => vk::CompareOp::ALWAYS,
    }
}

fn map_index_type(index_type: crate::IndexType) -> vk::IndexType {
    match index_type {
        crate::IndexType::U16 => vk::IndexType::UINT16,
        crate::IndexType::U32 => vk::IndexType::UINT32,
    }
}

fn map_vertex_format(vertex_format: crate::VertexFormat) -> vk::Format {
    use crate::VertexFormat as Vf;
    match vertex_format {
        Vf::F32Vec3 => vk::Format::R32G32B32_SFLOAT,
    }
}

struct BottomLevelAccelerationStructureInput {
    max_primitive_counts: Box<[u32]>,
    build_range_infos: Box<[vk::AccelerationStructureBuildRangeInfoKHR]>,
    _geometries: Box<[vk::AccelerationStructureGeometryKHR]>,
    build_info: vk::AccelerationStructureBuildGeometryInfoKHR,
}

impl Device {
    fn get_device_address(&self, piece: &crate::BufferPiece) -> u64 {
        let vk_info = vk::BufferDeviceAddressInfo::builder().buffer(piece.buffer.raw);
        let base = unsafe { self.core.get_buffer_device_address(&vk_info) };
        base + piece.offset
    }

    fn map_acceleration_structure_meshes(
        &self,
        meshes: &[crate::AccelerationStructureMesh],
    ) -> BottomLevelAccelerationStructureInput {
        let mut max_primitive_counts = Vec::with_capacity(meshes.len());
        let mut build_range_infos = Vec::with_capacity(meshes.len());
        let mut geometries = Vec::with_capacity(meshes.len());
        for mesh in meshes {
            max_primitive_counts.push(mesh.triangle_count);
            build_range_infos.push(vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: mesh.triangle_count,
                primitive_offset: 0,
                first_vertex: 0,
                transform_offset: 0,
            });

            let mut triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                .vertex_format(map_vertex_format(mesh.vertex_format))
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: self.get_device_address(&mesh.vertex_data),
                })
                .vertex_stride(mesh.vertex_stride as u64)
                .max_vertex(mesh.vertex_count.saturating_sub(1))
                .build();
            if let Some(index_type) = mesh.index_type {
                triangles.index_type = map_index_type(index_type);
                triangles.index_data = vk::DeviceOrHostAddressConstKHR {
                    device_address: self.get_device_address(&mesh.index_data),
                };
            }
            if mesh.transform_data.buffer.raw != vk::Buffer::null() {
                triangles.transform_data = vk::DeviceOrHostAddressConstKHR {
                    device_address: self.get_device_address(&mesh.transform_data),
                };
            }

            let geometry = vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                .flags(if mesh.is_opaque {
                    vk::GeometryFlagsKHR::OPAQUE
                } else {
                    vk::GeometryFlagsKHR::empty()
                })
                .build();
            geometries.push(geometry);
        }
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries)
            .build();

        BottomLevelAccelerationStructureInput {
            max_primitive_counts: max_primitive_counts.into_boxed_slice(),
            build_range_infos: build_range_infos.into_boxed_slice(),
            _geometries: geometries.into_boxed_slice(),
            build_info,
        }
    }
}
