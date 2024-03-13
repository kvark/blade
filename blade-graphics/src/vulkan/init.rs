use ash::{
    extensions::{ext, khr},
    vk,
};
use naga::back::spv;
use std::{ffi, fs, mem, sync::Mutex};

mod db {
    pub mod intel {
        pub const VENDOR: u32 = 0x8086;
    }
}
mod layer {
    use std::ffi::CStr;
    pub const KHRONOS_VALIDATION: &CStr =
        unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };
    pub const MESA_OVERLAY: &CStr =
        unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_MESA_overlay\0") };
}

const REQUIRED_DEVICE_EXTENSIONS: &[&ffi::CStr] = &[
    vk::ExtInlineUniformBlockFn::name(),
    vk::KhrTimelineSemaphoreFn::name(),
    vk::KhrDescriptorUpdateTemplateFn::name(),
    vk::KhrDynamicRenderingFn::name(),
];

#[derive(Debug)]
struct AdapterCapabilities {
    api_version: u32,
    properties: vk::PhysicalDeviceProperties,
    queue_family_index: u32,
    layered: bool,
    ray_tracing: bool,
    buffer_marker: bool,
    shader_info: bool,
}

struct SystemBugs {
    /// https://gitlab.freedesktop.org/mesa/mesa/-/issues/4688
    intel_unable_to_present_on_xorg: bool,
}

// See https://github.com/canonical/nvidia-prime/blob/587c5012be9dddcc17ab4d958f10a24fa3342b4d/prime-select#L56
fn is_nvidia_prime_forced() -> bool {
    match fs::read_to_string("/etc/prime-discrete") {
        Ok(contents) => contents == "on\n",
        Err(_) => false,
    }
}

unsafe fn inspect_adapter(
    phd: vk::PhysicalDevice,
    instance: &super::Instance,
    driver_api_version: u32,
    bugs: &SystemBugs,
    surface: Option<vk::SurfaceKHR>,
) -> Option<AdapterCapabilities> {
    let supported_extension_properties = instance
        .core
        .enumerate_device_extension_properties(phd)
        .unwrap();
    let supported_extensions = supported_extension_properties
        .iter()
        .map(|ext_prop| ffi::CStr::from_ptr(ext_prop.extension_name.as_ptr()))
        .collect::<Vec<_>>();
    for extension in REQUIRED_DEVICE_EXTENSIONS {
        if !supported_extensions.contains(extension) {
            log::warn!(
                "Rejected for device extension {:?} not supported",
                extension
            );
            return None;
        }
    }

    let mut inline_uniform_block_properties =
        vk::PhysicalDeviceInlineUniformBlockPropertiesEXT::default();
    let mut timeline_semaphore_properties =
        vk::PhysicalDeviceTimelineSemaphorePropertiesKHR::default();
    let mut descriptor_indexing_properties =
        vk::PhysicalDeviceDescriptorIndexingPropertiesEXT::default();
    let mut acceleration_structure_properties =
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut portability_subset_properties =
        vk::PhysicalDevicePortabilitySubsetPropertiesKHR::default();
    let mut properties2_khr = vk::PhysicalDeviceProperties2KHR::builder()
        .push_next(&mut inline_uniform_block_properties)
        .push_next(&mut timeline_semaphore_properties)
        .push_next(&mut descriptor_indexing_properties)
        .push_next(&mut acceleration_structure_properties)
        .push_next(&mut portability_subset_properties);
    instance
        .get_physical_device_properties2
        .get_physical_device_properties2(phd, &mut properties2_khr);

    let api_version = properties2_khr
        .properties
        .api_version
        .min(driver_api_version);
    if api_version < vk::API_VERSION_1_1 {
        log::warn!("\tRejected for API version {}", api_version);
        return None;
    }

    let queue_family_index = 0; //TODO
    if let Some(surface) = surface {
        let khr = instance.surface.as_ref()?;
        if khr.get_physical_device_surface_support(phd, queue_family_index, surface) != Ok(true) {
            log::warn!("Rejected for not presenting to the window surface");
            return None;
        }
        if bugs.intel_unable_to_present_on_xorg
            && properties2_khr.properties.vendor_id == db::intel::VENDOR
        {
            log::warn!("Rejecting Intel for not presenting when Nvidia is present (on Linux)");
            return None;
        }
    }

    let mut inline_uniform_block_features =
        vk::PhysicalDeviceInlineUniformBlockFeaturesEXT::default();
    let mut timeline_semaphore_features = vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR::default();
    let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeaturesKHR::default();
    let mut descriptor_indexing_features =
        vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::default();
    let mut buffer_device_address_features =
        vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR::default();
    let mut acceleration_structure_features =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
    let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
    let mut features2_khr = vk::PhysicalDeviceFeatures2::builder()
        .push_next(&mut inline_uniform_block_features)
        .push_next(&mut timeline_semaphore_features)
        .push_next(&mut dynamic_rendering_features)
        .push_next(&mut descriptor_indexing_features)
        .push_next(&mut buffer_device_address_features)
        .push_next(&mut acceleration_structure_features)
        .push_next(&mut ray_query_features);
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
        log::warn!(
            "\tRejected for inline uniform blocks. Properties = {:?}, Features = {:?}",
            inline_uniform_block_properties,
            inline_uniform_block_features,
        );
        return None;
    }

    if timeline_semaphore_features.timeline_semaphore == 0 {
        log::warn!(
            "\tRejected for timeline semaphore. Properties = {:?}, Features = {:?}",
            timeline_semaphore_properties,
            timeline_semaphore_features,
        );
        return None;
    }

    if dynamic_rendering_features.dynamic_rendering == 0 {
        log::warn!(
            "\tRejected for dynamic rendering. Features = {:?}",
            dynamic_rendering_features,
        );
        return None;
    }

    let ray_tracing = if !supported_extensions.contains(&vk::KhrAccelerationStructureFn::name())
        || !supported_extensions.contains(&vk::KhrRayQueryFn::name())
    {
        log::info!("No ray tracing extensions are supported");
        false
    } else if descriptor_indexing_properties.max_per_stage_update_after_bind_resources == vk::FALSE
        || descriptor_indexing_features.descriptor_binding_partially_bound == vk::FALSE
        || descriptor_indexing_features.shader_storage_buffer_array_non_uniform_indexing
            == vk::FALSE
        || descriptor_indexing_features.shader_sampled_image_array_non_uniform_indexing == vk::FALSE
    {
        log::info!(
            "No ray tracing because of the descriptor indexing. Properties = {:?}. Features = {:?}",
            descriptor_indexing_properties,
            descriptor_indexing_features
        );
        false
    } else if buffer_device_address_features.buffer_device_address == vk::FALSE {
        log::info!(
            "No ray tracing because of the buffer device address. Features = {:?}",
            buffer_device_address_features
        );
        false
    } else if acceleration_structure_properties.max_geometry_count == 0
        || acceleration_structure_features.acceleration_structure == vk::FALSE
    {
        log::info!("No ray tracing because of the acceleration structure. Properties = {:?}. Features = {:?}",
            acceleration_structure_properties, acceleration_structure_features);
        false
    } else if ray_query_features.ray_query == vk::FALSE {
        log::info!(
            "No ray tracing because of the ray query. Features = {:?}",
            ray_query_features
        );
        false
    } else {
        log::info!("Ray tracing is supported");
        log::debug!("Ray tracing properties: {acceleration_structure_properties:#?}");
        true
    };

    let buffer_marker = supported_extensions.contains(&vk::AmdBufferMarkerFn::name());
    let shader_info = supported_extensions.contains(&vk::AmdShaderInfoFn::name());

    Some(AdapterCapabilities {
        api_version,
        properties,
        queue_family_index,
        layered: portability_subset_properties.min_vertex_input_binding_stride_alignment != 0,
        ray_tracing,
        buffer_marker,
        shader_info,
    })
}

impl super::Context {
    unsafe fn init_impl(
        desc: crate::ContextDesc,
        surface_handles: Option<(
            raw_window_handle::RawWindowHandle,
            raw_window_handle::RawDisplayHandle,
        )>,
    ) -> Result<Self, crate::NotSupportedError> {
        let entry = match ash::Entry::load() {
            Ok(entry) => entry,
            Err(err) => {
                log::error!("Missing Vulkan entry points: {:?}", err);
                return Err(crate::NotSupportedError);
            }
        };
        let driver_api_version = match entry.try_enumerate_instance_version() {
            // Vulkan 1.1+
            Ok(Some(version)) => version,
            Ok(None) => return Err(crate::NotSupportedError),
            Err(err) => {
                log::error!("try_enumerate_instance_version: {:?}", err);
                return Err(crate::NotSupportedError);
            }
        };

        let supported_layers = match entry.enumerate_instance_layer_properties() {
            Ok(layers) => layers,
            Err(err) => {
                log::error!("enumerate_instance_layer_properties: {:?}", err);
                return Err(crate::NotSupportedError);
            }
        };
        let supported_layer_names = supported_layers
            .iter()
            .map(|properties| ffi::CStr::from_ptr(properties.layer_name.as_ptr()))
            .collect::<Vec<_>>();

        let mut layers: Vec<&'static ffi::CStr> = Vec::new();
        let mut requested_layers = Vec::<&ffi::CStr>::new();
        if desc.validation {
            requested_layers.push(layer::KHRONOS_VALIDATION);
        }
        if desc.overlay {
            requested_layers.push(layer::MESA_OVERLAY);
        }
        for name in requested_layers {
            if supported_layer_names.contains(&name) {
                layers.push(name);
            } else {
                log::warn!("Requested layer is not found: {:?}", name);
            }
        }

        let supported_instance_extension_properties =
            match entry.enumerate_instance_extension_properties(None) {
                Ok(extensions) => extensions,
                Err(err) => {
                    log::error!("enumerate_instance_extension_properties: {:?}", err);
                    return Err(crate::NotSupportedError);
                }
            };
        let supported_instance_extensions = supported_instance_extension_properties
            .iter()
            .map(|ext_prop| ffi::CStr::from_ptr(ext_prop.extension_name.as_ptr()))
            .collect::<Vec<_>>();

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
                    log::error!("Instance extension {:?} is not supported", inst_ext);
                    return Err(crate::NotSupportedError);
                }
            }
            if supported_instance_extensions.contains(&vk::KhrPortabilityEnumerationFn::name()) {
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

        let is_xorg = match surface_handles {
            Some((_, raw_window_handle::RawDisplayHandle::Xlib(_))) => true,
            Some((_, raw_window_handle::RawDisplayHandle::Xcb(_))) => true,
            _ => false,
        };
        let bugs = SystemBugs {
            intel_unable_to_present_on_xorg: is_xorg && is_nvidia_prime_forced(),
        };

        let vk_surface = surface_handles.map(|(rwh, rdh)| {
            ash_window::create_surface(&entry, &core_instance, rdh, rwh, None).unwrap()
        });

        let instance = super::Instance {
            debug_utils: ext::DebugUtils::new(&entry, &core_instance),
            get_physical_device_properties2: khr::GetPhysicalDeviceProperties2::new(
                &entry,
                &core_instance,
            ),
            surface: if surface_handles.is_some() {
                Some(khr::Surface::new(&entry, &core_instance))
            } else {
                None
            },
            core: core_instance,
        };

        let physical_devices = instance.core.enumerate_physical_devices().unwrap();
        let (physical_device, capabilities) = physical_devices
            .into_iter()
            .find_map(|phd| {
                inspect_adapter(phd, &instance, driver_api_version, &bugs, vk_surface)
                    .map(|caps| (phd, caps))
            })
            .ok_or(crate::NotSupportedError)?;

        log::debug!("Adapter {:#?}", capabilities);

        let device_core = {
            let family_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(capabilities.queue_family_index)
                .queue_priorities(&[1.0])
                .build();
            let family_infos = [family_info];

            let mut device_extensions = REQUIRED_DEVICE_EXTENSIONS.to_vec();
            if surface_handles.is_some() {
                device_extensions.push(vk::KhrSwapchainFn::name());
            }
            if capabilities.layered {
                log::info!("Enabling Vulkan Portability");
                device_extensions.push(vk::KhrPortabilitySubsetFn::name());
            }
            if capabilities.ray_tracing {
                if capabilities.api_version < vk::API_VERSION_1_2 {
                    device_extensions.push(vk::ExtDescriptorIndexingFn::name());
                    device_extensions.push(vk::KhrBufferDeviceAddressFn::name());
                    device_extensions.push(vk::KhrShaderFloatControlsFn::name());
                    device_extensions.push(vk::KhrSpirv14Fn::name());
                }
                device_extensions.push(vk::KhrDeferredHostOperationsFn::name());
                device_extensions.push(vk::KhrAccelerationStructureFn::name());
                device_extensions.push(vk::KhrRayQueryFn::name());
            }
            if capabilities.buffer_marker {
                device_extensions.push(vk::AmdBufferMarkerFn::name());
            }
            if capabilities.shader_info {
                device_extensions.push(vk::AmdShaderInfoFn::name());
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
            let mut device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&family_infos)
                .enabled_extension_names(&str_pointers)
                .push_next(&mut ext_inline_uniform_block)
                .push_next(&mut khr_timeline_semaphore)
                .push_next(&mut khr_dynamic_rendering);

            let mut ext_descriptor_indexing;
            let mut khr_buffer_device_address;
            let mut khr_acceleration_structure;
            let mut khr_ray_query;
            if capabilities.ray_tracing {
                ext_descriptor_indexing =
                    vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::builder()
                        .shader_storage_buffer_array_non_uniform_indexing(true)
                        .shader_sampled_image_array_non_uniform_indexing(true)
                        .descriptor_binding_partially_bound(true);
                khr_buffer_device_address =
                    vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR::builder()
                        .buffer_device_address(true);
                khr_acceleration_structure =
                    vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                        .acceleration_structure(true);
                khr_ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::builder().ray_query(true);
                device_create_info = device_create_info
                    .push_next(&mut ext_descriptor_indexing)
                    .push_next(&mut khr_buffer_device_address)
                    .push_next(&mut khr_acceleration_structure)
                    .push_next(&mut khr_ray_query);
            }

            instance
                .core
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };

        let device = super::Device {
            timeline_semaphore: khr::TimelineSemaphore::new(&instance.core, &device_core),
            dynamic_rendering: khr::DynamicRendering::new(&instance.core, &device_core),
            ray_tracing: if capabilities.ray_tracing {
                Some(super::RayTracingDevice {
                    acceleration_structure: khr::AccelerationStructure::new(
                        &instance.core,
                        &device_core,
                    ),
                })
            } else {
                None
            },
            buffer_marker: if capabilities.buffer_marker && desc.validation {
                //TODO: https://github.com/ash-rs/ash/issues/768
                Some(vk::AmdBufferMarkerFn::load(|name| unsafe {
                    mem::transmute(
                        instance
                            .core
                            .get_device_proc_addr(device_core.handle(), name.as_ptr()),
                    )
                }))
            } else {
                None
            },
            shader_info: if capabilities.shader_info {
                Some(vk::AmdShaderInfoFn::load(|name| unsafe {
                    mem::transmute(
                        instance
                            .core
                            .get_device_proc_addr(device_core.handle(), name.as_ptr()),
                    )
                }))
            } else {
                None
            },
            core: device_core,
            //TODO: detect GPU family
            workarounds: super::Workarounds {
                extra_sync_src_access: vk::AccessFlags::TRANSFER_WRITE,
                extra_sync_dst_access: vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            },
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
                buffer_device_address: capabilities.ray_tracing,
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
            super::MemoryManager {
                allocator: gpu_alloc::GpuAllocator::new(config, properties),
                slab: slab::Slab::new(),
                valid_ash_memory_types,
            }
        };

        let queue = device
            .core
            .get_device_queue(capabilities.queue_family_index, 0);
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

        let surface = vk_surface.map(|raw| {
            let extension = khr::Swapchain::new(&instance.core, &device.core);
            let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
            let next_semaphore = unsafe {
                device
                    .core
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            };
            Mutex::new(super::Surface {
                raw,
                frames: Vec::new(),
                next_semaphore,
                swapchain: vk::SwapchainKHR::null(),
                extension,
            })
        });

        let mut naga_flags = spv::WriterFlags::FORCE_POINT_SIZE;
        if desc.validation {
            naga_flags |= spv::WriterFlags::DEBUG;
        }

        Ok(super::Context {
            memory: Mutex::new(memory_manager),
            device,
            queue_family_index: capabilities.queue_family_index,
            queue: Mutex::new(super::Queue {
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

    pub unsafe fn init(desc: crate::ContextDesc) -> Result<Self, crate::NotSupportedError> {
        Self::init_impl(desc, None)
    }

    pub unsafe fn init_windowed<
        I: raw_window_handle::HasRawWindowHandle + raw_window_handle::HasRawDisplayHandle,
    >(
        window: &I,
        desc: crate::ContextDesc,
    ) -> Result<Self, crate::NotSupportedError> {
        let handles = (window.raw_window_handle(), window.raw_display_handle());
        Self::init_impl(desc, Some(handles))
    }

    pub(super) fn set_object_name(
        &self,
        object_type: vk::ObjectType,
        object: impl vk::Handle,
        name: &str,
    ) {
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

    pub fn capabilities(&self) -> crate::Capabilities {
        crate::Capabilities {
            ray_query: match self.device.ray_tracing {
                Some(_) => crate::ShaderVisibility::all(),
                None => crate::ShaderVisibility::empty(),
            },
        }
    }
}

impl super::Context {
    pub fn resize(&self, config: crate::SurfaceConfig) -> crate::TextureFormat {
        let surface_khr = self.instance.surface.as_ref().unwrap();
        let mut surface = self.surface.as_ref().unwrap().lock().unwrap();

        let capabilities = unsafe {
            surface_khr
                .get_physical_device_surface_capabilities(self.physical_device, surface.raw)
                .unwrap()
        };
        if config.size.width < capabilities.min_image_extent.width
            || config.size.width > capabilities.max_image_extent.width
            || config.size.height < capabilities.min_image_extent.height
            || config.size.height > capabilities.max_image_extent.height
        {
            log::warn!(
                "Requested size {}x{} is outside of surface capabilities",
                config.size.width,
                config.size.height
            );
        }

        let effective_frame_count = config.frame_count.max(capabilities.min_image_count).min(
            if capabilities.max_image_count != 0 {
                capabilities.max_image_count
            } else {
                !0
            },
        );
        if effective_frame_count != config.frame_count {
            log::warn!(
                "Requested frame count {} is outside of surface capabilities, clamping to {}",
                config.frame_count,
                effective_frame_count,
            );
        }

        let queue_families = [self.queue_family_index];
        //TODO: consider supported color spaces by Vulkan
        let format = match config.color_space {
            crate::ColorSpace::Linear => crate::TextureFormat::Bgra8UnormSrgb,
            crate::ColorSpace::Srgb => crate::TextureFormat::Bgra8Unorm,
        };
        let vk_format = super::map_texture_format(format);
        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.raw)
            .min_image_count(effective_frame_count)
            .image_format(vk_format)
            .image_extent(vk::Extent2D {
                width: config.size.width,
                height: config.size.height,
            })
            .image_array_layers(1)
            .image_usage(super::resource::map_texture_usage(
                config.usage,
                crate::TexelAspects::COLOR,
            ))
            .queue_family_indices(&queue_families)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .old_swapchain(surface.swapchain);
        let new_swapchain = unsafe {
            surface
                .extension
                .create_swapchain(&create_info, None)
                .unwrap()
        };

        // destroy the old swapchain
        unsafe {
            surface.extension.destroy_swapchain(surface.swapchain, None);
        }
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
                .get_swapchain_images(new_swapchain)
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
                .format(vk_format)
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
            surface.frames.push(super::Frame {
                image_index: index as u32,
                image,
                view,
                format,
                acquire_semaphore,
                target_size,
            });
        }
        surface.swapchain = new_swapchain;
        format
    }

    pub fn acquire_frame(&self) -> super::Frame {
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
