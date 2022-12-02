use ash::{
    extensions::{ext, khr},
    vk,
};
use naga::back::spv;
use std::{ffi, mem, num::NonZeroU32, sync::Mutex};

mod command;
mod pipeline;
mod resource;

struct Instance {
    core: ash::Instance,
    debug_utils: ext::DebugUtils,
    get_physical_device_properties2: khr::GetPhysicalDeviceProperties2,
}

#[derive(Clone)]
struct Device {
    core: ash::Device,
    timeline_semaphore: khr::TimelineSemaphore,
    dynamic_rendering: khr::DynamicRendering,
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
            aspects: FormatAspects::COLOR,
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
    millis as u64 * 1_000_000
}

pub struct Context {
    memory: Mutex<MemoryManager>,
    device: Device,
    queue_family_index: u32,
    queue: Mutex<Queue>,
    surface: Option<Mutex<Surface>>,
    physical_device: vk::PhysicalDevice,
    naga_flags: spv::WriterFlags,
    instance: Instance,
    _entry: ash::Entry,
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
struct BlockInfo {
    bytes: u8,
    width: u8,
    height: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Texture {
    raw: vk::Image,
    memory_handle: usize,
    target_size: [u16; 2],
    format: crate::TextureFormat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureView {
    raw: vk::ImageView,
    target_size: [u16; 2],
    aspects: FormatAspects,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampler {
    raw: vk::Sampler,
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

pub struct CommandEncoder {
    pool: vk::CommandPool,
    buffers: Box<[CommandBuffer]>,
    device: Device,
    update_data: Vec<u8>,
    present_index: Option<u32>,
}
pub struct TransferCommandEncoder<'a> {
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
    device: &'a Device,
    update_data: &'a mut Vec<u8>,
}

pub struct SyncPoint {
    progress: u64,
}

struct AdapterCapabilities {
    properties: vk::PhysicalDeviceProperties,
}

unsafe fn inspect_adapter(
    phd: vk::PhysicalDevice,
    instance: &Instance,
    driver_api_version: u32,
) -> Option<AdapterCapabilities> {
    let mut inline_uniform_block_properties =
        vk::PhysicalDeviceInlineUniformBlockPropertiesEXT::default();
    let mut timeline_semaphore_properties =
        vk::PhysicalDeviceTimelineSemaphorePropertiesKHR::default();
    let mut properties2_khr = vk::PhysicalDeviceProperties2KHR::builder()
        .push_next(&mut inline_uniform_block_properties)
        .push_next(&mut timeline_semaphore_properties);
    instance
        .get_physical_device_properties2
        .get_physical_device_properties2(phd, &mut properties2_khr);

    let api_version = properties2_khr
        .properties
        .api_version
        .min(driver_api_version);
    if api_version < vk::API_VERSION_1_1 {
        log::info!("\tRejected for API version {}", api_version);
        return None;
    }

    let mut inline_uniform_block_features =
        vk::PhysicalDeviceInlineUniformBlockFeaturesEXT::default();
    let mut timeline_semaphore_features = vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR::default();
    let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeaturesKHR::default();
    let mut features2_khr = vk::PhysicalDeviceFeatures2::builder()
        .push_next(&mut inline_uniform_block_features)
        .push_next(&mut timeline_semaphore_features)
        .push_next(&mut dynamic_rendering_features);
    instance
        .get_physical_device_properties2
        .get_physical_device_features2(phd, &mut features2_khr);

    let properties = properties2_khr.properties;
    let name = ffi::CStr::from_ptr(properties.device_name.as_ptr());
    log::info!("Adapter {:?}", name);

    if inline_uniform_block_properties.max_inline_uniform_block_size
        < crate::limits::PLAIN_DATA_SIZE
        || inline_uniform_block_properties.max_descriptor_set_inline_uniform_blocks == 0
        || inline_uniform_block_features.inline_uniform_block == 0
    {
        log::info!(
            "\tRejected for inline uniform blocks. Properties = {:?}, Features = {:?}",
            inline_uniform_block_properties,
            inline_uniform_block_features,
        );
        return None;
    }

    if timeline_semaphore_features.timeline_semaphore == 0 {
        log::info!(
            "\tRejected for timeline semaphore. Properties = {:?}, Features = {:?}",
            timeline_semaphore_properties,
            timeline_semaphore_features,
        );
        return None;
    }

    if dynamic_rendering_features.dynamic_rendering == 0 {
        log::info!(
            "\tRejected for dynamic rendering. Features = {:?}",
            dynamic_rendering_features,
        );
        return None;
    }

    Some(AdapterCapabilities { properties })
}

impl Context {
    unsafe fn init_impl(
        desc: super::ContextDesc,
        surface_handles: Option<(
            raw_window_handle::RawWindowHandle,
            raw_window_handle::RawDisplayHandle,
        )>,
    ) -> Result<Self, super::NotSupportedError> {
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

        let core_instance = {
            let mut create_flags = vk::InstanceCreateFlags::empty();

            let mut instance_extensions = vec![
                ext::DebugUtils::name(),
                vk::KhrGetPhysicalDeviceProperties2Fn::name(),
            ];
            if let Some((_, rdh)) = surface_handles {
                instance_extensions.extend(
                    ash_window::enumerate_required_extensions(rdh)
                        .unwrap()
                        .iter()
                        .map(|&ptr| ffi::CStr::from_ptr(ptr)),
                );
            }

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

        let instance = Instance {
            debug_utils: ext::DebugUtils::new(&entry, &core_instance),
            get_physical_device_properties2: khr::GetPhysicalDeviceProperties2::new(
                &entry,
                &core_instance,
            ),
            core: core_instance,
        };

        let physical_devices = instance.core.enumerate_physical_devices().unwrap();
        let (physical_device, capabilities) = physical_devices
            .into_iter()
            .find_map(|phd| {
                inspect_adapter(phd, &instance, driver_api_version).map(|caps| (phd, caps))
            })
            .ok_or(super::NotSupportedError)?;

        let queue_family_index = 0; //TODO

        let device_core = {
            let family_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&[1.0])
                .build();
            let family_infos = [family_info];

            let mut device_extensions = vec![
                vk::ExtInlineUniformBlockFn::name(),
                vk::KhrTimelineSemaphoreFn::name(),
                vk::KhrDescriptorUpdateTemplateFn::name(),
                vk::KhrDynamicRenderingFn::name(),
            ];
            if surface_handles.is_some() {
                device_extensions.push(vk::KhrSwapchainFn::name());
            }
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
            let mut khr_timeline_semaphore =
                vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR::builder().timeline_semaphore(true);
            let mut khr_dynamic_rendering =
                vk::PhysicalDeviceDynamicRenderingFeaturesKHR::builder().dynamic_rendering(true);
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&family_infos)
                .enabled_extension_names(&str_pointers)
                .push_next(&mut ext_inline_uniform_block)
                .push_next(&mut khr_timeline_semaphore)
                .push_next(&mut khr_dynamic_rendering);
            instance
                .core
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };

        let device = Device {
            timeline_semaphore: khr::TimelineSemaphore::new(&instance.core, &device_core),
            dynamic_rendering: khr::DynamicRendering::new(&instance.core, &device_core),
            core: device_core,
        };

        let memory_manager = {
            let mem_properties = instance
                .core
                .get_physical_device_memory_properties(physical_device);
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

        let queue = device.core.get_device_queue(queue_family_index, 0);
        let last_progress = 0;
        let mut timeline_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(last_progress);
        let timeline_semaphore_create_info =
            vk::SemaphoreCreateInfo::builder().push_next(&mut timeline_info);
        let timeline_semaphore = unsafe {
            device
                .core
                .create_semaphore(&timeline_semaphore_create_info, None)
                .unwrap()
        };
        let present_semaphore_create_info = vk::SemaphoreCreateInfo::builder();
        let present_semaphore = unsafe {
            device
                .core
                .create_semaphore(&present_semaphore_create_info, None)
                .unwrap()
        };

        let surface = surface_handles.map(|(rwh, rdh)| {
            let extension = khr::Swapchain::new(&instance.core, &device.core);
            let raw = ash_window::create_surface(&entry, &instance.core, rdh, rwh, None).unwrap();
            let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
            let next_semaphore = unsafe {
                device
                    .core
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            };
            Mutex::new(Surface {
                raw,
                frames: Vec::new(),
                next_semaphore,
                swapchain: vk::SwapchainKHR::null(),
                extension,
            })
        });

        let mut naga_flags = spv::WriterFlags::ADJUST_COORDINATE_SPACE;
        if desc.validation {
            naga_flags |= spv::WriterFlags::DEBUG;
        }

        Ok(Context {
            memory: Mutex::new(memory_manager),
            device,
            queue_family_index,
            queue: Mutex::new(Queue {
                raw: queue,
                timeline_semaphore,
                present_semaphore,
                last_progress,
            }),
            surface,
            physical_device,
            naga_flags,
            instance,
            _entry: entry,
        })
    }

    pub unsafe fn init(desc: super::ContextDesc) -> Result<Self, super::NotSupportedError> {
        Self::init_impl(desc, None)
    }

    pub unsafe fn init_windowed<
        I: raw_window_handle::HasRawWindowHandle + raw_window_handle::HasRawDisplayHandle,
    >(
        window: &I,
        desc: super::ContextDesc,
    ) -> Result<Self, super::NotSupportedError> {
        let handles = (window.raw_window_handle(), window.raw_display_handle());
        Self::init_impl(desc, Some(handles))
    }

    pub fn create_command_encoder(&self, desc: super::CommandEncoderDesc) -> CommandEncoder {
        //TODO: these numbers are arbitrary, needs to be replaced by
        // an abstraction from gpu-alloc, if possible.
        const ROUGH_SET_COUNT: u32 = 100;
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
            present_index: None,
        }
    }

    pub fn destroy_command_encoder(&self, command_encoder: CommandEncoder) {
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

    pub fn submit(&self, encoder: &mut CommandEncoder) -> SyncPoint {
        let raw_cmd_buf = encoder.finish();
        let mut queue = self.queue.lock().unwrap();
        queue.last_progress += 1;
        let progress = queue.last_progress;
        let command_buffers = [raw_cmd_buf];
        let semaphores_all = [queue.timeline_semaphore, queue.present_semaphore];
        let signal_values_all = [progress, 0];
        let num_sepahores = if encoder.present_index.is_some() {
            2
        } else {
            1
        };
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::builder()
            .signal_semaphore_values(&signal_values_all[..num_sepahores]);
        let vk_info = vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .signal_semaphores(&semaphores_all[..num_sepahores])
            .push_next(&mut timeline_info);
        unsafe {
            self.device
                .core
                .queue_submit(queue.raw, &[vk_info.build()], vk::Fence::null())
                .unwrap();
        }

        if let Some(image_index) = encoder.present_index.take() {
            let surface = self.surface.as_ref().unwrap().lock().unwrap();
            let swapchains = [surface.swapchain];
            let image_indices = [image_index];
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

    pub fn wait_for(&self, sp: SyncPoint, timeout_ms: u32) -> bool {
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

    fn set_object_name(&self, object_type: vk::ObjectType, object: impl vk::Handle, name: &str) {
        let name_cstr = ffi::CString::new(name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_type(object_type)
            .object_handle(object.as_raw())
            .object_name(&name_cstr);
        let _ = unsafe {
            self.instance
                .debug_utils
                .set_debug_utils_object_name(self.device.core.handle(), &name_info)
        };
    }
}

impl Context {
    pub fn resize(&self, config: crate::SurfaceConfig) -> crate::TextureFormat {
        let mut surface = self.surface.as_ref().unwrap().lock().unwrap();
        let queue_families = [self.queue_family_index];
        let format = crate::TextureFormat::Bgra8UnormSrgb;
        let format_info = describe_format(format);
        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.raw)
            .min_image_count(config.frame_count)
            .image_format(format_info.raw)
            .image_extent(vk::Extent2D {
                width: config.size.width,
                height: config.size.height,
            })
            .image_array_layers(1)
            .image_usage(resource::map_texture_usage(
                config.usage,
                FormatAspects::COLOR,
            ))
            .queue_family_indices(&queue_families)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .old_swapchain(surface.swapchain);
        surface.swapchain = unsafe {
            surface
                .extension
                .create_swapchain(&create_info, None)
                .unwrap()
        };

        for frame in surface.frames.drain(..) {
            unsafe {
                self.device.core.destroy_image_view(frame.view, None);
                self.device
                    .core
                    .destroy_semaphore(frame.acquire_semaphore, None);
            }
        }
        let images = unsafe {
            surface
                .extension
                .get_swapchain_images(surface.swapchain)
                .unwrap()
        };
        let target_size = [config.size.width as u16, config.size.height as u16];
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        for (index, image) in images.into_iter().enumerate() {
            let view_create_info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format_info.raw)
                .subresource_range(subresource_range);
            let view = unsafe {
                self.device
                    .core
                    .create_image_view(&view_create_info, None)
                    .unwrap()
            };
            let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
            let acquire_semaphore = unsafe {
                self.device
                    .core
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            };
            surface.frames.push(Frame {
                image_index: index as u32,
                image,
                view,
                format,
                acquire_semaphore,
                target_size,
            });
        }
        format
    }

    pub fn acquire_frame(&self) -> Frame {
        let mut surface = self.surface.as_ref().unwrap().lock().unwrap();
        let acquire_semaphore = surface.next_semaphore;
        let (index, _suboptimal) = unsafe {
            surface
                .extension
                .acquire_next_image(surface.swapchain, !0, acquire_semaphore, vk::Fence::null())
                .unwrap()
        };
        surface.next_semaphore = mem::replace(
            &mut surface.frames[index as usize].acquire_semaphore,
            acquire_semaphore,
        );
        surface.frames[index as usize]
    }
}

bitflags::bitflags! {
    struct FormatAspects: u8 {
        const COLOR = 0 << 1;
        const DEPTH = 1 << 1;
        const STENCIL = 1 << 2;
    }
}

struct FormatInfo {
    raw: vk::Format,
    aspects: FormatAspects,
    block: BlockInfo,
}

fn describe_format(format: crate::TextureFormat) -> FormatInfo {
    use crate::TextureFormat as Tf;
    let (raw, aspects, bytes) = match format {
        Tf::Rgba8Unorm => (vk::Format::R8G8B8A8_UNORM, FormatAspects::COLOR, 4),
        Tf::Bgra8UnormSrgb => (vk::Format::B8G8R8A8_SRGB, FormatAspects::COLOR, 4),
    };
    FormatInfo {
        raw,
        aspects,
        block: BlockInfo {
            bytes,
            width: 1,
            height: 1,
        },
    }
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

fn map_extent_3d(extent: &crate::Extent) -> vk::Extent3D {
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
