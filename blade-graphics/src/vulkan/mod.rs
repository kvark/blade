use ash::{
    extensions::{ext, khr},
    vk,
};
use naga::back::spv;
use std::{ffi, num::NonZeroU32, sync::Mutex};

mod command;
mod pipeline;
mod resource;

struct InstanceExt {
    debug_utils: ext::DebugUtils,
    get_physical_device_properties2: khr::GetPhysicalDeviceProperties2,
}

struct DeviceExt {
    draw_indirect_count: Option<khr::DrawIndirectCount>,
    timeline_semaphore: Option<khr::TimelineSemaphore>,
}

struct MemoryManager {
    allocator: gpu_alloc::GpuAllocator<vk::DeviceMemory>,
    slab: slab::Slab<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
    valid_ash_memory_types: u32,
}

pub struct Context {
    memory: Mutex<MemoryManager>,
    device_ext: DeviceExt,
    device: ash::Device,
    queue: Mutex<vk::Queue>,
    physical_device: vk::PhysicalDevice,
    naga_flags: spv::WriterFlags,
    instance_ext: InstanceExt,
    instance: ash::Instance,
    entry: ash::Entry,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Buffer {
    raw: vk::Buffer,
    memory_handle: usize,
    mapped_data: *mut u8,
}

impl Buffer {
    pub fn data(&self) -> *mut u8 {
        self.mapped_data
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Texture {
    raw: vk::Image,
    memory_handle: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureView {
    raw: vk::ImageView,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampler {
    raw: vk::Sampler,
}

#[derive(Debug, Default)]
struct DescriptorSetLayout {
    raw: vk::DescriptorSetLayout,
    update_template: vk::DescriptorUpdateTemplate,
}

#[derive(Debug)]
struct PipelineLayout {
    raw: vk::PipelineLayout,
    descriptor_set_layouts: Vec<DescriptorSetLayout>,
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
}

pub struct CommandEncoder {
    raw: vk::CommandBuffer,
    device: ash::Device,
    plain_data: Vec<u8>,
}
pub struct TransferCommandEncoder<'a> {
    raw: vk::CommandBuffer,
    device: &'a ash::Device,
}
pub struct ComputeCommandEncoder<'a> {
    raw: vk::CommandBuffer,
    device: &'a ash::Device,
    plain_data: &'a mut Vec<u8>,
}
pub struct RenderCommandEncoder<'a> {
    raw: vk::CommandBuffer,
    device: &'a ash::Device,
    plain_data: &'a mut Vec<u8>,
}
pub struct PipelineEncoder<'a> {
    raw: vk::CommandBuffer,
    device: &'a ash::Device,
    plain_data: &'a mut Vec<u8>,
}

pub struct SyncPoint {}

struct AdapterCapabilities {
    properties: vk::PhysicalDeviceProperties,
}

unsafe fn inspect_adapter(
    phd: vk::PhysicalDevice,
    _instance: &ash::Instance,
    extensions: &InstanceExt,
    _driver_api_version: u32,
) -> Option<AdapterCapabilities> {
    let mut inline_uniform_block_properties =
        vk::PhysicalDeviceInlineUniformBlockPropertiesEXT::default();
    let mut properties2_khr =
        vk::PhysicalDeviceProperties2KHR::builder().push_next(&mut inline_uniform_block_properties);
    extensions
        .get_physical_device_properties2
        .get_physical_device_properties2(phd, &mut properties2_khr);

    let properties = properties2_khr.properties;
    let name = ffi::CStr::from_ptr(properties.device_name.as_ptr());
    log::info!("Adapter {:?}", name);
    if inline_uniform_block_properties.max_inline_uniform_block_size
        < crate::limits::PLAIN_DATA_SIZE
        || inline_uniform_block_properties.max_descriptor_set_inline_uniform_blocks == 0
    {
        log::info!(
            "\tRejected for inline uniform blocks: {:?}",
            inline_uniform_block_properties
        );
        return None;
    }

    Some(AdapterCapabilities { properties })
}

impl Context {
    pub unsafe fn init(desc: super::ContextDesc) -> Result<Self, super::NotSupportedError> {
        let entry = match ash::Entry::load() {
            Ok(entry) => entry,
            Err(err) => {
                log::error!("Missing Vulkan entry points: {:?}", err);
                return Err(super::NotSupportedError);
            }
        };
        let driver_api_version = match entry.try_enumerate_instance_version() {
            // Vulkan 1.1+
            Ok(Some(version)) => version,
            Ok(None) => return Err(super::NotSupportedError),
            Err(err) => {
                log::error!("try_enumerate_instance_version: {:?}", err);
                return Err(super::NotSupportedError);
            }
        };

        let _supported_layers = match entry.enumerate_instance_layer_properties() {
            Ok(layers) => layers,
            Err(err) => {
                log::error!("enumerate_instance_layer_properties: {:?}", err);
                return Err(super::NotSupportedError);
            }
        };

        let mut layers: Vec<&'static ffi::CStr> = Vec::new();
        if desc.validation {
            layers.push(ffi::CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap());
        }

        let supported_instance_extension_properties =
            match entry.enumerate_instance_extension_properties(None) {
                Ok(extensions) => extensions,
                Err(err) => {
                    log::error!("enumerate_instance_extension_properties: {:?}", err);
                    return Err(super::NotSupportedError);
                }
            };
        let supported_instance_extensions = supported_instance_extension_properties
            .iter()
            .map(|ext_prop| ffi::CStr::from_ptr(ext_prop.extension_name.as_ptr()))
            .collect::<Vec<_>>();
        let is_vulkan_portability =
            supported_instance_extensions.contains(&vk::KhrPortabilityEnumerationFn::name());

        let instance = {
            let mut create_flags = vk::InstanceCreateFlags::empty();

            let mut instance_extensions = vec![
                ext::DebugUtils::name(),
                vk::KhrGetPhysicalDeviceProperties2Fn::name(),
            ];

            for inst_ext in instance_extensions.iter() {
                if !supported_instance_extensions.contains(inst_ext) {
                    log::error!("Extension {:?} is not supported", inst_ext);
                    return Err(super::NotSupportedError);
                }
            }
            if is_vulkan_portability {
                log::info!("Enabling Vulkan Portability");
                instance_extensions.push(vk::KhrPortabilityEnumerationFn::name());
                create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }

            let app_info = vk::ApplicationInfo::builder()
                .engine_name(ffi::CStr::from_bytes_with_nul(b"blade\0").unwrap())
                .engine_version(1)
                .api_version(vk::HEADER_VERSION_COMPLETE);
            let str_pointers = layers
                .iter()
                .chain(instance_extensions.iter())
                .map(|&s| s.as_ptr())
                .collect::<Vec<_>>();
            let (layer_strings, extension_strings) = str_pointers.split_at(layers.len());
            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .flags(create_flags)
                .enabled_layer_names(layer_strings)
                .enabled_extension_names(extension_strings);
            entry.create_instance(&create_info, None).unwrap()
        };

        let instance_ext = InstanceExt {
            debug_utils: ext::DebugUtils::new(&entry, &instance),
            get_physical_device_properties2: khr::GetPhysicalDeviceProperties2::new(
                &entry, &instance,
            ),
        };

        let physical_devices = instance.enumerate_physical_devices().unwrap();
        let (physical_device, capabilities) = physical_devices
            .into_iter()
            .find_map(|phd| {
                inspect_adapter(phd, &instance, &instance_ext, driver_api_version)
                    .map(|caps| (phd, caps))
            })
            .ok_or(super::NotSupportedError)?;

        let family_index = 0; //TODO

        let device = {
            let family_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(family_index)
                .queue_priorities(&[1.0])
                .build();
            let family_infos = [family_info];

            let mut device_extensions = vec![
                vk::ExtInlineUniformBlockFn::name(),
                vk::KhrPushDescriptorFn::name(),
                vk::KhrDescriptorUpdateTemplateFn::name(),
            ];
            if is_vulkan_portability {
                device_extensions.push(vk::KhrPortabilitySubsetFn::name());
            }

            let str_pointers = device_extensions
                .iter()
                .map(|&s| s.as_ptr())
                .collect::<Vec<_>>();

            let mut ext_inline_uniform_block =
                vk::PhysicalDeviceInlineUniformBlockFeaturesEXT::builder()
                    .inline_uniform_block(true);
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&family_infos)
                .enabled_extension_names(&str_pointers)
                .push_next(&mut ext_inline_uniform_block);
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };

        let device_ext = DeviceExt {
            draw_indirect_count: None,
            timeline_semaphore: None,
        };

        let memory_manager = {
            let mem_properties = instance.get_physical_device_memory_properties(physical_device);
            let memory_types =
                &mem_properties.memory_types[..mem_properties.memory_type_count as usize];
            let limits = &capabilities.properties.limits;
            let config = gpu_alloc::Config::i_am_prototyping(); //TODO?

            let properties = gpu_alloc::DeviceProperties {
                max_memory_allocation_count: limits.max_memory_allocation_count,
                max_memory_allocation_size: u64::max_value(), // TODO
                non_coherent_atom_size: limits.non_coherent_atom_size,
                memory_types: memory_types
                    .iter()
                    .map(|memory_type| gpu_alloc::MemoryType {
                        props: gpu_alloc::MemoryPropertyFlags::from_bits_truncate(
                            memory_type.property_flags.as_raw() as u8,
                        ),
                        heap: memory_type.heap_index,
                    })
                    .collect(),
                memory_heaps: mem_properties.memory_heaps
                    [..mem_properties.memory_heap_count as usize]
                    .iter()
                    .map(|&memory_heap| gpu_alloc::MemoryHeap {
                        size: memory_heap.size,
                    })
                    .collect(),
                buffer_device_address: false,
            };

            let known_memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED
                | vk::MemoryPropertyFlags::LAZILY_ALLOCATED;
            let valid_ash_memory_types = memory_types.iter().enumerate().fold(0, |u, (i, mem)| {
                if known_memory_flags.contains(mem.property_flags) {
                    u | (1 << i)
                } else {
                    u
                }
            });
            MemoryManager {
                allocator: gpu_alloc::GpuAllocator::new(config, properties),
                slab: slab::Slab::new(),
                valid_ash_memory_types,
            }
        };

        let queue = device.get_device_queue(family_index, 0);

        let mut naga_flags = spv::WriterFlags::ADJUST_COORDINATE_SPACE;
        if desc.validation {
            naga_flags |= spv::WriterFlags::DEBUG;
        }

        Ok(Context {
            memory: Mutex::new(memory_manager),
            device_ext,
            device,
            queue: Mutex::new(queue),
            physical_device,
            naga_flags,
            instance_ext,
            instance,
            entry,
        })
    }

    pub fn create_command_encoder(&self, _desc: super::CommandEncoderDesc) -> CommandEncoder {
        CommandEncoder {
            raw: vk::CommandBuffer::null(),
            device: self.device.clone(),
            plain_data: Vec::new(),
        }
    }

    pub fn submit(&self, _encoder: &mut CommandEncoder) -> SyncPoint {
        unimplemented!()
    }

    pub fn wait_for(&self, _sp: SyncPoint, _timeout_ms: u32) -> bool {
        unimplemented!()
    }

    fn set_object_name(&self, object_type: vk::ObjectType, object: impl vk::Handle, name: &str) {
        let name_cstr = ffi::CString::new(name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_type(object_type)
            .object_handle(object.as_raw())
            .object_name(&name_cstr);
        let _ = unsafe {
            self.instance_ext
                .debug_utils
                .set_debug_utils_object_name(self.device.handle(), &name_info)
        };
    }
}

bitflags::bitflags! {
    struct FormatAspects: u32 {
        const COLOR = 0 << 1;
        const DEPTH = 1 << 1;
        const STENCIL = 1 << 2;
    }
}

struct FormatInfo {
    raw: vk::Format,
    aspects: FormatAspects,
}

fn describe_format(format: crate::TextureFormat) -> FormatInfo {
    use crate::TextureFormat as Tf;
    let (raw, aspects) = match format {
        Tf::Rgba8Unorm => (vk::Format::R8G8B8A8_UNORM, FormatAspects::COLOR),
        Tf::Bgra8UnormSrgb => (vk::Format::B8G8R8A8_SRGB, FormatAspects::COLOR),
    };
    FormatInfo { raw, aspects }
}

fn map_aspects(aspects: FormatAspects) -> vk::ImageAspectFlags {
    let mut flags = vk::ImageAspectFlags::empty();
    if aspects.contains(FormatAspects::COLOR) {
        flags |= vk::ImageAspectFlags::COLOR;
    }
    if aspects.contains(FormatAspects::DEPTH) {
        flags |= vk::ImageAspectFlags::DEPTH;
    }
    if aspects.contains(FormatAspects::STENCIL) {
        flags |= vk::ImageAspectFlags::STENCIL;
    }
    flags
}

fn map_extent_3d(extent: crate::Extent) -> vk::Extent3D {
    vk::Extent3D {
        width: extent.width,
        height: extent.height,
        depth: extent.depth,
    }
}

fn map_subresource_range(
    subresources: &crate::TextureSubresources,
    aspects: FormatAspects,
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
