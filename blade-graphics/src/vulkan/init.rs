use ash::vk::Handle as _;
use ash::{amd, ext, khr, vk};
use naga::back::spv;
use std::{ffi, sync::Mutex};

use crate::NotSupportedError;

// TODO: Remove once `ash` includes VK_KHR_unified_image_layouts bindings.
// See https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_unified_image_layouts.html
mod unified_image_layouts {
    use ash::vk;
    use std::ffi;

    pub const NAME: &ffi::CStr = c"VK_KHR_unified_image_layouts";
    const STRUCTURE_TYPE: i32 = 1000466000;

    #[repr(C)]
    pub struct PhysicalDeviceFeatures {
        pub s_type: vk::StructureType,
        pub p_next: *mut ffi::c_void,
        pub unified_image_layouts: vk::Bool32,
        pub unified_image_layouts_video: vk::Bool32,
    }
    impl Default for PhysicalDeviceFeatures {
        fn default() -> Self {
            Self {
                s_type: vk::StructureType::from_raw(STRUCTURE_TYPE),
                p_next: std::ptr::null_mut(),
                unified_image_layouts: vk::FALSE,
                unified_image_layouts_video: vk::FALSE,
            }
        }
    }
    unsafe impl vk::TaggedStructure for PhysicalDeviceFeatures {
        const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(STRUCTURE_TYPE);
    }
    unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for PhysicalDeviceFeatures {}
    unsafe impl vk::ExtendsDeviceCreateInfo for PhysicalDeviceFeatures {}
}

mod db {
    pub mod intel {
        pub const VENDOR: u32 = 0x8086;
    }
    pub mod nvidia {
        pub const VENDOR: u32 = 0x10DE;
    }
    pub mod qualcomm {
        pub const VENDOR: u32 = 0x5143;
    }
    pub mod pci_class {
        /// VGA compatible controller
        pub const VGA: &str = "0x0300";
        /// 3D controller (e.g. NVIDIA dGPU in PRIME configurations)
        pub const DISPLAY_3D: &str = "0x0302";
    }
}
mod layer {
    use std::ffi::CStr;
    pub const KHRONOS_VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";
    pub const MESA_OVERLAY: &CStr = c"VK_LAYER_MESA_overlay";
}

const REQUIRED_DEVICE_EXTENSIONS: &[&ffi::CStr] = &[
    vk::KHR_TIMELINE_SEMAPHORE_NAME,
    vk::KHR_DESCRIPTOR_UPDATE_TEMPLATE_NAME,
    vk::KHR_DYNAMIC_RENDERING_NAME,
];

fn is_promoted_instance_extension(name: &ffi::CStr, api_version: u32) -> bool {
    if api_version < vk::API_VERSION_1_1 {
        return false;
    }

    name == vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME
        || name == vk::KHR_GET_SURFACE_CAPABILITIES2_NAME
}

#[derive(Debug)]
struct RayTracingCapabilities {
    min_scratch_buffer_alignment: u64,
}

#[derive(Debug)]
struct SystemBugs {
    /// https://github.com/kvark/blade/issues/117
    intel_fix_descriptor_pool_leak: bool,
}

#[derive(Debug)]
struct AdapterCapabilities {
    api_version: u32,
    properties: vk::PhysicalDeviceProperties,
    device_information: crate::DeviceInformation,
    queue_family_index: u32,
    layered: bool,
    binding_array: bool,
    ray_tracing: Option<RayTracingCapabilities>,
    buffer_device_address: bool,
    max_inline_uniform_block_size: u32,
    buffer_marker: bool,
    shader_info: bool,
    full_screen_exclusive: bool,
    external_memory: bool,
    timing: bool,
    dual_source_blending: bool,
    shader_float16: bool,
    cooperative_matrix: crate::CooperativeMatrix,
    unified_image_layouts: bool,
    memory_budget: bool,
    bugs: SystemBugs,
}

impl AdapterCapabilities {
    fn to_capabilities(&self) -> crate::Capabilities {
        crate::Capabilities {
            binding_array: self.binding_array,
            ray_query: match self.ray_tracing {
                Some(_) => crate::ShaderVisibility::all(),
                None => crate::ShaderVisibility::empty(),
            },
            sample_count_mask: (self.properties.limits.framebuffer_color_sample_counts
                & self.properties.limits.framebuffer_depth_sample_counts)
                .as_raw(),
            dual_source_blending: self.dual_source_blending,
            shader_float16: self.shader_float16,
            cooperative_matrix: self.cooperative_matrix,
        }
    }
}

/// Display server on the current Linux system.
/// Used to determine which GPU can present in PRIME configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DisplayServer {
    X11,
    Wayland,
    Other,
}

/// Detect the active display server from environment variables.
fn detect_display_server() -> DisplayServer {
    // WAYLAND_DISPLAY is set by Wayland compositors for native clients
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        return DisplayServer::Wayland;
    }
    // DISPLAY is set by X11
    if std::env::var_os("DISPLAY").is_some() {
        return DisplayServer::X11;
    }
    DisplayServer::Other
}

/// Read GPU vendor IDs from sysfs for all PCI display devices.
///
/// Scans `/sys/bus/pci/devices/*/class` for VGA controllers and
/// 3D controllers, then reads their vendor IDs.
/// Returns a deduplicated list of vendor IDs.
fn detect_gpu_vendors() -> Vec<u32> {
    let mut vendors = Vec::new();
    let entries = match std::fs::read_dir("/sys/bus/pci/devices") {
        Ok(entries) => entries,
        Err(_) => return vendors,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let class = match std::fs::read_to_string(path.join("class")) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let class = class.trim();
        let is_gpu =
            class.starts_with(db::pci_class::VGA) || class.starts_with(db::pci_class::DISPLAY_3D);
        if !is_gpu {
            continue;
        }
        let vendor = match std::fs::read_to_string(path.join("vendor")) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Ok(id) = u32::from_str_radix(vendor.trim().trim_start_matches("0x"), 16)
            && !vendors.contains(&id)
        {
            vendors.push(id);
        }
    }
    vendors
}

/// Check if a physical device cannot present in the current multi-GPU configuration.
///
/// On Linux systems with both Intel and NVIDIA GPUs (PRIME topology):
/// - X11: Intel iGPU cannot present (Mesa bug <https://gitlab.freedesktop.org/mesa/mesa/-/issues/4688>)
/// - Wayland: NVIDIA dGPU cannot present (surface owned by compositor's GPU)
///
/// Returns `true` if the device must be rejected for presentation.
fn is_presentation_broken(
    vendor_id: u32,
    gpu_vendors: &[u32],
    display_server: DisplayServer,
) -> bool {
    let has_intel = gpu_vendors.contains(&db::intel::VENDOR);
    let has_nvidia = gpu_vendors.contains(&db::nvidia::VENDOR);
    if !has_intel || !has_nvidia {
        return false;
    }
    match (display_server, vendor_id) {
        // Intel cannot present on X11 when NVIDIA is also present
        (DisplayServer::X11, db::intel::VENDOR) => true,
        // NVIDIA cannot present on Wayland when Intel is also present
        (DisplayServer::Wayland, db::nvidia::VENDOR) => true,
        _ => false,
    }
}

fn inspect_adapter(
    phd: vk::PhysicalDevice,
    instance: &super::Instance,
    driver_api_version: u32,
    desc: &crate::ContextDesc,
    gpu_vendors: &[u32],
    display_server: DisplayServer,
) -> Result<AdapterCapabilities, String> {
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

    let mut driver_properties = vk::PhysicalDeviceDriverPropertiesKHR::default();
    let mut properties2_khr = vk::PhysicalDeviceProperties2KHR::default()
        .push_next(&mut inline_uniform_block_properties)
        .push_next(&mut timeline_semaphore_properties)
        .push_next(&mut descriptor_indexing_properties)
        .push_next(&mut acceleration_structure_properties)
        .push_next(&mut portability_subset_properties)
        .push_next(&mut driver_properties);
    unsafe {
        instance
            .get_physical_device_properties2
            .get_physical_device_properties2(phd, &mut properties2_khr);
    }

    let properties = properties2_khr.properties;
    let name = unsafe { ffi::CStr::from_ptr(properties.device_name.as_ptr()) };
    log::info!("Adapter: {:?}", name);

    if let Some(device_id) = desc.device_id
        && device_id != properties.device_id
    {
        return Err(format!(
            "device ID 0x{:X} doesn't match requested 0x{:X}",
            properties.device_id, device_id,
        ));
    }

    let api_version = properties.api_version.min(driver_api_version);
    if api_version < vk::API_VERSION_1_1 {
        return Err(format!("Vulkan API version {} is below 1.1", api_version));
    }

    let supported_extension_properties = unsafe {
        instance
            .core
            .enumerate_device_extension_properties(phd)
            .unwrap()
    };
    let supported_extensions = supported_extension_properties
        .iter()
        .map(|ext_prop| unsafe { ffi::CStr::from_ptr(ext_prop.extension_name.as_ptr()) })
        .collect::<Vec<_>>();
    for extension in REQUIRED_DEVICE_EXTENSIONS {
        if !supported_extensions.contains(extension) {
            return Err(format!(
                "required device extension {:?} is not supported",
                extension
            ));
        }
    }

    let bugs = SystemBugs {
        intel_fix_descriptor_pool_leak: cfg!(windows) && properties.vendor_id == db::intel::VENDOR,
    };

    let queue_family_index = 0; //TODO
    if desc.presentation
        && is_presentation_broken(properties.vendor_id, gpu_vendors, display_server)
    {
        return Err(format!(
            "cannot present on {:?} in Intel+NVIDIA PRIME configuration",
            display_server,
        ));
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
    let mut cooperative_matrix_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut vulkan_memory_model_features = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
    let mut float16_int8_features = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut storage_16bit_features = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut unified_image_layouts_features =
        unified_image_layouts::PhysicalDeviceFeatures::default();
    let mut features2_khr = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut inline_uniform_block_features)
        .push_next(&mut timeline_semaphore_features)
        .push_next(&mut dynamic_rendering_features)
        .push_next(&mut descriptor_indexing_features)
        .push_next(&mut buffer_device_address_features)
        .push_next(&mut acceleration_structure_features)
        .push_next(&mut ray_query_features)
        .push_next(&mut cooperative_matrix_features)
        .push_next(&mut vulkan_memory_model_features)
        .push_next(&mut float16_int8_features)
        .push_next(&mut storage_16bit_features)
        .push_next(&mut unified_image_layouts_features);
    unsafe {
        instance
            .get_physical_device_properties2
            .get_physical_device_features2(phd, &mut features2_khr)
    };

    let dual_source_blending = features2_khr.features.dual_src_blend != 0;
    let shader_float16 = float16_int8_features.shader_float16 != 0;

    let has_inline_ub = supported_extensions.contains(&vk::EXT_INLINE_UNIFORM_BLOCK_NAME)
        && inline_uniform_block_properties.max_descriptor_set_inline_uniform_blocks > 0
        && inline_uniform_block_features.inline_uniform_block != 0;
    // Adreno 740 (Qualcomm) has a driver bug: inline uniform blocks combined
    // with inter-stage varyings cause "Failed to link shaders" at pipeline creation.
    let max_inline_uniform_block_size =
        if has_inline_ub && properties.vendor_id != db::qualcomm::VENDOR {
            log::info!(
                "Inline uniform blocks enabled (max size per binding: {} bytes)",
                inline_uniform_block_properties.max_inline_uniform_block_size,
            );
            inline_uniform_block_properties.max_inline_uniform_block_size
        } else {
            log::info!(
                "Inline uniform blocks disabled (supported={}, vendor=0x{:X}). Using UBO fallback.",
                has_inline_ub,
                properties.vendor_id,
            );
            0
        };

    if timeline_semaphore_features.timeline_semaphore == 0 {
        return Err("timeline semaphore feature is not supported".to_string());
    }

    if dynamic_rendering_features.dynamic_rendering == 0 {
        return Err("dynamic rendering feature is not supported".to_string());
    }

    let external_memory = supported_extensions.contains(&vk::KHR_EXTERNAL_MEMORY_NAME);
    let external_memory = external_memory
        && supported_extensions.contains(if cfg!(target_os = "windows") {
            &vk::KHR_EXTERNAL_MEMORY_WIN32_NAME
        } else {
            &vk::KHR_EXTERNAL_MEMORY_FD_NAME
        });

    let timing = if properties.limits.timestamp_compute_and_graphics == vk::FALSE {
        log::info!("No timing because of queue support");
        false
    } else {
        true
    };

    let buffer_device_address = buffer_device_address_features.buffer_device_address == vk::TRUE
        && (properties.api_version >= vk::API_VERSION_1_2
            || supported_extensions.contains(&vk::KHR_BUFFER_DEVICE_ADDRESS_NAME));

    let supports_descriptor_indexing = api_version >= vk::API_VERSION_1_2
        || supported_extensions.contains(&vk::EXT_DESCRIPTOR_INDEXING_NAME);
    let binding_array = supports_descriptor_indexing
        && descriptor_indexing_features.descriptor_binding_partially_bound == vk::TRUE
        && descriptor_indexing_features.shader_storage_buffer_array_non_uniform_indexing
            == vk::TRUE
        && descriptor_indexing_features.shader_sampled_image_array_non_uniform_indexing == vk::TRUE;

    let ray_tracing = if !desc.ray_tracing {
        log::info!("Ray tracing disabled by configuration");
        None
    } else if !supported_extensions.contains(&vk::KHR_ACCELERATION_STRUCTURE_NAME)
        || !supported_extensions.contains(&vk::KHR_RAY_QUERY_NAME)
    {
        log::info!("No ray tracing extensions are supported");
        None
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
        None
    } else if buffer_device_address_features.buffer_device_address == vk::FALSE {
        log::info!(
            "No ray tracing because of the buffer device address. Features = {:?}",
            buffer_device_address_features
        );
        None
    } else if acceleration_structure_properties.max_geometry_count == 0
        || acceleration_structure_features.acceleration_structure == vk::FALSE
    {
        log::info!(
            "No ray tracing because of the acceleration structure. Properties = {:?}. Features = {:?}",
            acceleration_structure_properties,
            acceleration_structure_features
        );
        None
    } else if ray_query_features.ray_query == vk::FALSE {
        log::info!(
            "No ray tracing because of the ray query. Features = {:?}",
            ray_query_features
        );
        None
    } else {
        log::info!("Ray tracing is supported");
        log::debug!("Ray tracing properties: {acceleration_structure_properties:#?}");
        Some(RayTracingCapabilities {
            min_scratch_buffer_alignment: acceleration_structure_properties
                .min_acceleration_structure_scratch_offset_alignment
                as u64,
        })
    };

    let cooperative_matrix = if !supported_extensions.contains(&vk::KHR_COOPERATIVE_MATRIX_NAME) {
        log::info!("No cooperative matrix extension support");
        crate::CooperativeMatrix::default()
    } else if cooperative_matrix_features.cooperative_matrix == vk::FALSE {
        log::info!(
            "No cooperative matrix feature support. Features = {:?}",
            cooperative_matrix_features
        );
        crate::CooperativeMatrix::default()
    } else if vulkan_memory_model_features.vulkan_memory_model == vk::FALSE {
        log::info!(
            "No Vulkan memory model support (required for cooperative matrix). Features = {:?}",
            vulkan_memory_model_features
        );
        crate::CooperativeMatrix::default()
    } else {
        // Query supported cooperative matrix configurations and find
        // square float configurations (Naga supports 8x8 and 16x16).
        let coop_props = unsafe {
            instance
                .cooperative_matrix
                .get_physical_device_cooperative_matrix_properties(phd)
                .unwrap_or_default()
        };
        let find_tile = |a_type, b_type, c_type, result_type| {
            [8u32, 16].into_iter().find(|&size| {
                coop_props.iter().any(|p| {
                    p.m_size == size
                        && p.n_size == size
                        && p.k_size == size
                        && p.a_type == a_type
                        && p.b_type == b_type
                        && p.c_type == c_type
                        && p.result_type == result_type
                        && p.scope == vk::ScopeKHR::SUBGROUP
                })
            })
        };
        let f32t = vk::ComponentTypeKHR::FLOAT32;
        let f16t = vk::ComponentTypeKHR::FLOAT16;
        let f32_tile = find_tile(f32t, f32t, f32t, f32t).unwrap_or(0);
        let f16_tile = if float16_int8_features.shader_float16 != 0
            && storage_16bit_features.storage_buffer16_bit_access != 0
        {
            find_tile(f16t, f16t, f32t, f32t).unwrap_or(0)
        } else {
            0
        };
        let cm = crate::CooperativeMatrix { f32_tile, f16_tile };
        if cm.is_supported() {
            log::info!(
                "Cooperative matrix: f32 tile={}, f16 tile={}",
                cm.f32_tile,
                cm.f16_tile,
            );
        } else {
            log::info!(
                "Cooperative matrix extension present but no usable config. Properties: {:?}",
                coop_props
            );
        }
        cm
    };
    // Auto-enable shader_float16 when cooperative matrix has f16 support.
    let shader_float16 = shader_float16 || cooperative_matrix.f16_tile > 0;

    let buffer_marker = supported_extensions.contains(&vk::AMD_BUFFER_MARKER_NAME);
    let shader_info = supported_extensions.contains(&vk::AMD_SHADER_INFO_NAME);
    let full_screen_exclusive = supported_extensions.contains(&vk::EXT_FULL_SCREEN_EXCLUSIVE_NAME);
    let memory_budget = supported_extensions.contains(&vk::EXT_MEMORY_BUDGET_NAME);

    let device_information = unsafe {
        crate::DeviceInformation {
            is_software_emulated: properties.device_type == vk::PhysicalDeviceType::CPU,
            device_name: ffi::CStr::from_ptr(properties.device_name.as_ptr())
                .to_string_lossy()
                .to_string(),
            driver_name: ffi::CStr::from_ptr(driver_properties.driver_name.as_ptr())
                .to_string_lossy()
                .to_string(),
            driver_info: ffi::CStr::from_ptr(driver_properties.driver_info.as_ptr())
                .to_string_lossy()
                .to_string(),
        }
    };

    Ok(AdapterCapabilities {
        api_version,
        properties,
        device_information,
        queue_family_index,
        layered: portability_subset_properties.min_vertex_input_binding_stride_alignment != 0,
        binding_array,
        ray_tracing,
        buffer_device_address,
        max_inline_uniform_block_size,
        buffer_marker,
        shader_info,
        full_screen_exclusive,
        external_memory,
        timing,
        dual_source_blending,
        shader_float16,
        cooperative_matrix,
        unified_image_layouts: supported_extensions.contains(&unified_image_layouts::NAME)
            && unified_image_layouts_features.unified_image_layouts == vk::TRUE,
        memory_budget,
        bugs,
    })
}

impl super::VulkanInstance {
    unsafe fn create(desc: &crate::ContextDesc) -> Result<Self, NotSupportedError> {
        let entry = match unsafe { ash::Entry::load() } {
            Ok(entry) => entry,
            Err(err) => {
                log::error!("Missing Vulkan entry points: {:?}", err);
                return Err(crate::PlatformError::loading(err).into());
            }
        };
        let driver_api_version = match unsafe { entry.try_enumerate_instance_version() } {
            // Vulkan 1.1+
            Ok(Some(version)) => version,
            Ok(None) => return Err(NotSupportedError::NoSupportedDeviceFound),
            Err(err) => {
                log::error!("try_enumerate_instance_version: {:?}", err);
                return Err(crate::PlatformError::init(err).into());
            }
        };

        if let Some(ref xr) = desc.xr {
            let reqs = xr
                .instance
                .graphics_requirements::<openxr::Vulkan>(xr.system_id)
                .unwrap();
            let driver_api_version_xr = openxr::Version::new(
                vk::api_version_major(driver_api_version) as u16,
                vk::api_version_minor(driver_api_version) as u16,
                vk::api_version_patch(driver_api_version),
            );
            if driver_api_version_xr < reqs.min_api_version_supported
                || driver_api_version_xr.major() > reqs.max_api_version_supported.major()
            {
                return Err(NotSupportedError::NoSupportedDeviceFound);
            }
        }

        let supported_layers = match unsafe { entry.enumerate_instance_layer_properties() } {
            Ok(layers) => layers,
            Err(err) => {
                log::error!("enumerate_instance_layer_properties: {:?}", err);
                return Err(crate::PlatformError::init(err).into());
            }
        };
        let supported_layer_names = supported_layers
            .iter()
            .map(|properties| unsafe { ffi::CStr::from_ptr(properties.layer_name.as_ptr()) })
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
            match unsafe { entry.enumerate_instance_extension_properties(None) } {
                Ok(extensions) => extensions,
                Err(err) => {
                    log::error!("enumerate_instance_extension_properties: {:?}", err);
                    return Err(crate::PlatformError::init(err).into());
                }
            };
        let supported_instance_extensions = supported_instance_extension_properties
            .iter()
            .map(|ext_prop| unsafe { ffi::CStr::from_ptr(ext_prop.extension_name.as_ptr()) })
            .collect::<Vec<_>>();

        let core_instance = {
            let mut create_flags = vk::InstanceCreateFlags::empty();

            let mut instance_extensions = vec![
                vk::EXT_DEBUG_UTILS_NAME,
                vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME,
            ];
            if desc.presentation {
                instance_extensions.push(vk::KHR_SURFACE_NAME);
                instance_extensions.push(vk::KHR_GET_SURFACE_CAPABILITIES2_NAME);
                let candidates = [
                    vk::KHR_WAYLAND_SURFACE_NAME,
                    vk::KHR_XCB_SURFACE_NAME,
                    vk::KHR_XLIB_SURFACE_NAME,
                    vk::KHR_WIN32_SURFACE_NAME,
                    vk::KHR_ANDROID_SURFACE_NAME,
                    vk::EXT_SWAPCHAIN_COLORSPACE_NAME,
                ];
                for candidate in candidates.iter() {
                    if supported_instance_extensions.contains(candidate) {
                        log::info!("Presentation support: {:?}", candidate);
                        instance_extensions.push(candidate);
                    }
                }
            }

            let mut enabled_instance_extensions = Vec::with_capacity(instance_extensions.len());
            for inst_ext in instance_extensions.drain(..) {
                if supported_instance_extensions.contains(&inst_ext) {
                    enabled_instance_extensions.push(inst_ext);
                } else if is_promoted_instance_extension(inst_ext, driver_api_version) {
                    log::info!(
                        "Skipping promoted instance extension {:?} (core version {:x})",
                        inst_ext,
                        driver_api_version
                    );
                } else {
                    log::error!("Instance extension {:?} is not supported", inst_ext);
                    return Err(NotSupportedError::NoSupportedDeviceFound);
                }
            }
            if supported_instance_extensions.contains(&vk::KHR_PORTABILITY_ENUMERATION_NAME) {
                log::info!("Enabling Vulkan Portability");
                enabled_instance_extensions.push(vk::KHR_PORTABILITY_ENUMERATION_NAME);
                create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }

            let app_info = vk::ApplicationInfo::default()
                .engine_name(c"blade")
                .engine_version(1)
                .api_version(driver_api_version.min(vk::HEADER_VERSION_COMPLETE));
            let str_pointers = layers
                .iter()
                .chain(enabled_instance_extensions.iter())
                .map(|&s| s.as_ptr())
                .collect::<Vec<_>>();
            let (layer_strings, extension_strings) = str_pointers.split_at(layers.len());
            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .flags(create_flags)
                .enabled_layer_names(layer_strings)
                .enabled_extension_names(extension_strings);
            if let Some(ref xr_desc) = desc.xr {
                let get_instance_proc_addr: openxr::sys::platform::VkGetInstanceProcAddr =
                    unsafe { std::mem::transmute(entry.static_fn().get_instance_proc_addr) };
                let raw_instance = unsafe {
                    xr_desc
                        .instance
                        .create_vulkan_instance(
                            xr_desc.system_id,
                            get_instance_proc_addr,
                            &create_info as *const _ as *const _,
                        )
                        .map_err(|_| NotSupportedError::NoSupportedDeviceFound)?
                        .map_err(|raw| crate::PlatformError::init(vk::Result::from_raw(raw)))?
                };
                unsafe {
                    ash::Instance::load(
                        entry.static_fn(),
                        vk::Instance::from_raw(raw_instance as _),
                    )
                }
            } else {
                unsafe { entry.create_instance(&create_info, None) }
                    .map_err(crate::PlatformError::init)?
            }
        };

        let instance =
            super::Instance {
                _debug_utils: ext::debug_utils::Instance::new(&entry, &core_instance),
                get_physical_device_properties2:
                    khr::get_physical_device_properties2::Instance::new(&entry, &core_instance),
                cooperative_matrix: khr::cooperative_matrix::Instance::new(&entry, &core_instance),
                get_surface_capabilities2: if desc.presentation {
                    Some(khr::get_surface_capabilities2::Instance::new(
                        &entry,
                        &core_instance,
                    ))
                } else {
                    None
                },
                surface: if desc.presentation {
                    Some(khr::surface::Instance::new(&entry, &core_instance))
                } else {
                    None
                },
                core: core_instance,
            };

        Ok(super::VulkanInstance {
            entry,
            instance,
            driver_api_version,
        })
    }
}

fn inspect_devices(
    instance: &super::Instance,
    driver_api_version: u32,
    default_phd: Option<vk::PhysicalDevice>,
) -> Vec<crate::DeviceReport> {
    let physical_devices =
        unsafe { instance.core.enumerate_physical_devices() }.unwrap_or_default();

    // Probe all capabilities unconditionally:
    //  - `ray_tracing` must be enabled to discover RT support
    //  - `presentation` is left off, so the PRIME topology check is skipped
    //    (it would reject devices based on the display server, not device features)
    let inspect_desc = crate::ContextDesc {
        ray_tracing: true,
        ..Default::default()
    };

    physical_devices
        .into_iter()
        .map(|phd| {
            let properties = unsafe { instance.core.get_physical_device_properties(phd) };
            let device_id = properties.device_id;
            let device_name = unsafe {
                ffi::CStr::from_ptr(properties.device_name.as_ptr())
                    .to_string_lossy()
                    .to_string()
            };

            let is_default = default_phd == Some(phd);
            let result = inspect_adapter(
                phd,
                instance,
                driver_api_version,
                &inspect_desc,
                &[],
                DisplayServer::Other,
            );

            let (status, information) = match result {
                Ok(caps) => {
                    let capabilities = caps.to_capabilities();
                    let status = crate::DeviceReportStatus::Available {
                        is_default,
                        caps: capabilities,
                    };
                    (status, caps.device_information)
                }
                Err(reason) => (
                    crate::DeviceReportStatus::Rejected(reason),
                    crate::DeviceInformation {
                        is_software_emulated: properties.device_type == vk::PhysicalDeviceType::CPU,
                        device_name,
                        driver_name: String::new(),
                        driver_info: String::new(),
                    },
                ),
            };

            crate::DeviceReport {
                device_id,
                information,
                status,
            }
        })
        .collect()
}

impl Drop for super::VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            self.instance.core.destroy_instance(None);
        }
    }
}

impl super::Context {
    pub unsafe fn init(desc: crate::ContextDesc) -> Result<Self, NotSupportedError> {
        let inner = unsafe { super::VulkanInstance::create(&desc)? };

        // Detect GPU vendors from sysfs to identify PRIME multi-GPU topologies.
        // Only needed when presentation is requested (for the PRIME rejection check).
        let gpu_vendors = if desc.presentation {
            let vendors = detect_gpu_vendors();
            log::info!(
                "PCI GPU vendors: {:?}",
                vendors
                    .iter()
                    .map(|v| format!("0x{v:04X}"))
                    .collect::<Vec<_>>(),
            );
            vendors
        } else {
            Vec::new()
        };
        let display_server = detect_display_server();

        let (physical_device, capabilities) = if let Some(ref xr_desc) = desc.xr {
            let xr_physical_device = unsafe {
                xr_desc
                    .instance
                    .vulkan_graphics_device(
                        xr_desc.system_id,
                        inner.instance.core.handle().as_raw() as _,
                    )
                    .map_err(|_| NotSupportedError::NoSupportedDeviceFound)?
            };
            let physical_device = vk::PhysicalDevice::from_raw(xr_physical_device as _);
            let capabilities = inspect_adapter(
                physical_device,
                &inner.instance,
                inner.driver_api_version,
                &desc,
                &gpu_vendors,
                display_server,
            )
            .map_err(|_| NotSupportedError::NoSupportedDeviceFound)?;
            (physical_device, capabilities)
        } else {
            unsafe { inner.instance.core.enumerate_physical_devices() }
                .map_err(crate::PlatformError::init)?
                .into_iter()
                .find_map(|phd| {
                    inspect_adapter(
                        phd,
                        &inner.instance,
                        inner.driver_api_version,
                        &desc,
                        &gpu_vendors,
                        display_server,
                    )
                    .ok()
                    .map(|caps| (phd, caps))
                })
                .ok_or(NotSupportedError::NoSupportedDeviceFound)?
        };

        log::debug!("Adapter {:#?}", capabilities);
        let mut min_buffer_alignment = 1;
        if let Some(ref rt) = capabilities.ray_tracing {
            min_buffer_alignment = min_buffer_alignment.max(rt.min_scratch_buffer_alignment);
        }

        let device_core = {
            let family_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(capabilities.queue_family_index)
                .queue_priorities(&[1.0]);
            let family_infos = [family_info];

            let mut device_extensions = REQUIRED_DEVICE_EXTENSIONS.to_vec();
            if capabilities.max_inline_uniform_block_size > 0 {
                device_extensions.push(vk::EXT_INLINE_UNIFORM_BLOCK_NAME);
            }
            if desc.presentation {
                device_extensions.push(vk::KHR_SWAPCHAIN_NAME);
            }
            if capabilities.layered {
                log::info!("Enabling Vulkan Portability");
                device_extensions.push(vk::KHR_PORTABILITY_SUBSET_NAME);
            }
            let needs_descriptor_indexing =
                capabilities.binding_array || capabilities.ray_tracing.is_some();
            if needs_descriptor_indexing && capabilities.api_version < vk::API_VERSION_1_2 {
                device_extensions.push(vk::EXT_DESCRIPTOR_INDEXING_NAME);
            }
            if capabilities.ray_tracing.is_some() {
                if capabilities.api_version < vk::API_VERSION_1_2 {
                    device_extensions.push(vk::KHR_BUFFER_DEVICE_ADDRESS_NAME);
                    device_extensions.push(vk::KHR_SHADER_FLOAT_CONTROLS_NAME);
                    device_extensions.push(vk::KHR_SPIRV_1_4_NAME);
                }
                device_extensions.push(vk::KHR_DEFERRED_HOST_OPERATIONS_NAME);
                device_extensions.push(vk::KHR_ACCELERATION_STRUCTURE_NAME);
                device_extensions.push(vk::KHR_RAY_QUERY_NAME);
            } else if capabilities.buffer_device_address
                && capabilities.api_version < vk::API_VERSION_1_2
            {
                device_extensions.push(vk::KHR_BUFFER_DEVICE_ADDRESS_NAME);
            }
            if capabilities.buffer_marker {
                device_extensions.push(vk::AMD_BUFFER_MARKER_NAME);
            }
            if capabilities.shader_info {
                device_extensions.push(vk::AMD_SHADER_INFO_NAME);
            }
            if capabilities.full_screen_exclusive {
                device_extensions.push(vk::EXT_FULL_SCREEN_EXCLUSIVE_NAME);
            }
            if capabilities.external_memory {
                device_extensions.push(vk::KHR_EXTERNAL_MEMORY_NAME);
                device_extensions.push(if cfg!(target_os = "windows") {
                    vk::KHR_EXTERNAL_MEMORY_WIN32_NAME
                } else {
                    vk::KHR_EXTERNAL_MEMORY_FD_NAME
                });
            }
            if capabilities.cooperative_matrix.is_supported() {
                device_extensions.push(vk::KHR_COOPERATIVE_MATRIX_NAME);
                if capabilities.api_version < vk::API_VERSION_1_2 {
                    device_extensions.push(vk::KHR_VULKAN_MEMORY_MODEL_NAME);
                }
            }
            if capabilities.memory_budget {
                device_extensions.push(vk::EXT_MEMORY_BUDGET_NAME);
            }
            if capabilities.unified_image_layouts {
                // TODO: Replace with ash constant once available.
                device_extensions.push(unified_image_layouts::NAME);
            }

            let str_pointers = device_extensions
                .iter()
                .map(|&s| s.as_ptr())
                .collect::<Vec<_>>();

            let mut ext_inline_uniform_block = vk::PhysicalDeviceInlineUniformBlockFeaturesEXT {
                inline_uniform_block: vk::TRUE,
                ..Default::default()
            };
            let mut khr_timeline_semaphore = vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR {
                timeline_semaphore: vk::TRUE,
                ..Default::default()
            };
            let mut khr_dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeaturesKHR {
                dynamic_rendering: vk::TRUE,
                ..Default::default()
            };
            let mut device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&family_infos)
                .enabled_extension_names(&str_pointers)
                .push_next(&mut khr_timeline_semaphore)
                .push_next(&mut khr_dynamic_rendering);
            if capabilities.max_inline_uniform_block_size > 0 {
                device_create_info = device_create_info.push_next(&mut ext_inline_uniform_block);
            }

            let mut ext_descriptor_indexing;
            if needs_descriptor_indexing {
                ext_descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeaturesEXT {
                    shader_storage_buffer_array_non_uniform_indexing: vk::TRUE,
                    shader_sampled_image_array_non_uniform_indexing: vk::TRUE,
                    descriptor_binding_partially_bound: vk::TRUE,
                    ..Default::default()
                };
                device_create_info = device_create_info.push_next(&mut ext_descriptor_indexing);
            }

            let mut khr_buffer_device_address;
            if capabilities.buffer_device_address {
                khr_buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR {
                    buffer_device_address: vk::TRUE,
                    ..Default::default()
                };
                device_create_info = device_create_info.push_next(&mut khr_buffer_device_address);
            }

            let mut khr_acceleration_structure;
            let mut khr_ray_query;
            if capabilities.ray_tracing.is_some() {
                khr_acceleration_structure = vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
                    acceleration_structure: vk::TRUE,
                    ..Default::default()
                };
                khr_ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR {
                    ray_query: vk::TRUE,
                    ..Default::default()
                };
                device_create_info = device_create_info
                    .push_next(&mut khr_acceleration_structure)
                    .push_next(&mut khr_ray_query);
            }

            let mut khr_float16_int8;
            let mut storage_16bit;
            if capabilities.shader_float16 {
                khr_float16_int8 = vk::PhysicalDeviceShaderFloat16Int8Features {
                    shader_float16: vk::TRUE,
                    ..Default::default()
                };
                device_create_info = device_create_info.push_next(&mut khr_float16_int8);
            }
            if capabilities.cooperative_matrix.f16_tile > 0 {
                storage_16bit = vk::PhysicalDevice16BitStorageFeatures {
                    storage_buffer16_bit_access: vk::TRUE,
                    uniform_and_storage_buffer16_bit_access: vk::TRUE,
                    ..Default::default()
                };
                device_create_info = device_create_info.push_next(&mut storage_16bit);
            }

            let mut khr_cooperative_matrix;
            let mut vulkan_memory_model;
            if capabilities.cooperative_matrix.is_supported() {
                khr_cooperative_matrix = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR {
                    cooperative_matrix: vk::TRUE,
                    ..Default::default()
                };
                vulkan_memory_model = vk::PhysicalDeviceVulkanMemoryModelFeatures {
                    vulkan_memory_model: vk::TRUE,
                    ..Default::default()
                };
                device_create_info = device_create_info
                    .push_next(&mut khr_cooperative_matrix)
                    .push_next(&mut vulkan_memory_model);
            }

            // TODO: Replace with ash typed struct once available.
            let mut khr_unified_image_layouts;
            if capabilities.unified_image_layouts {
                khr_unified_image_layouts = unified_image_layouts::PhysicalDeviceFeatures {
                    unified_image_layouts: vk::TRUE,
                    ..Default::default()
                };
                device_create_info = device_create_info.push_next(&mut khr_unified_image_layouts);
            }

            let mut core_features = vk::PhysicalDeviceFeatures::default();
            if capabilities.dual_source_blending {
                core_features.dual_src_blend = vk::TRUE;
            }

            let mut device_features2 =
                vk::PhysicalDeviceFeatures2::default().features(core_features);

            device_create_info = device_create_info.push_next(&mut device_features2);

            if let Some(ref xr_desc) = desc.xr {
                let get_instance_proc_addr: openxr::sys::platform::VkGetInstanceProcAddr =
                    unsafe { std::mem::transmute(inner.entry.static_fn().get_instance_proc_addr) };
                unsafe {
                    let raw_device = xr_desc
                        .instance
                        .create_vulkan_device(
                            xr_desc.system_id,
                            get_instance_proc_addr,
                            physical_device.as_raw() as *const ffi::c_void,
                            &device_create_info as *const _ as *const ffi::c_void,
                        )
                        .map_err(|_| NotSupportedError::NoSupportedDeviceFound)?
                        .map_err(|raw| crate::PlatformError::init(vk::Result::from_raw(raw)))?;
                    ash::Device::load(
                        inner.instance.core.fp_v1_0(),
                        vk::Device::from_raw(raw_device as _),
                    )
                }
            } else {
                unsafe {
                    inner
                        .instance
                        .core
                        .create_device(physical_device, &device_create_info, None)
                        .map_err(crate::PlatformError::init)?
                }
            }
        };

        let instance = &inner.instance;
        let device = super::Device {
            swapchain: if desc.presentation {
                Some(khr::swapchain::Device::new(&instance.core, &device_core))
            } else {
                None
            },
            debug_utils: ext::debug_utils::Device::new(&instance.core, &device_core),
            timeline_semaphore: khr::timeline_semaphore::Device::new(&instance.core, &device_core),
            dynamic_rendering: khr::dynamic_rendering::Device::new(&instance.core, &device_core),
            ray_tracing: if let Some(ref caps) = capabilities.ray_tracing {
                Some(super::RayTracingDevice {
                    acceleration_structure: khr::acceleration_structure::Device::new(
                        &instance.core,
                        &device_core,
                    ),
                    scratch_buffer_alignment: caps.min_scratch_buffer_alignment,
                })
            } else {
                None
            },
            buffer_device_address: capabilities.buffer_device_address,
            max_inline_uniform_block_size: capabilities.max_inline_uniform_block_size,
            buffer_marker: if capabilities.buffer_marker && desc.validation {
                Some(amd::buffer_marker::Device::new(
                    &instance.core,
                    &device_core,
                ))
            } else {
                None
            },
            shader_info: if capabilities.shader_info {
                Some(amd::shader_info::Device::new(&instance.core, &device_core))
            } else {
                None
            },
            full_screen_exclusive: if desc.presentation && capabilities.full_screen_exclusive {
                Some(ext::full_screen_exclusive::Device::new(
                    &instance.core,
                    &device_core,
                ))
            } else {
                None
            },
            external_memory: if capabilities.external_memory {
                #[cfg(not(target_os = "windows"))]
                use khr::external_memory_fd::Device;
                #[cfg(target_os = "windows")]
                use khr::external_memory_win32::Device;

                Some(Device::new(&instance.core, &device_core))
            } else {
                None
            },
            core: device_core,
            device_information: capabilities.device_information,
            command_scope: if desc.capture {
                Some(super::CommandScopeDevice {})
            } else {
                None
            },
            timing: if desc.timing && capabilities.timing {
                Some(super::TimingDevice {
                    period: capabilities.properties.limits.timestamp_period,
                })
            } else {
                None
            },
            //TODO: detect GPU family
            workarounds: super::Workarounds {
                extra_sync_src_access: vk::AccessFlags::TRANSFER_WRITE,
                extra_sync_dst_access: vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | if capabilities.ray_tracing.is_some() {
                        vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
                    } else {
                        vk::AccessFlags::NONE
                    },
                extra_descriptor_pool_create_flags: if capabilities
                    .bugs
                    .intel_fix_descriptor_pool_leak
                {
                    vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET
                } else {
                    vk::DescriptorPoolCreateFlags::empty()
                },
            },
        };

        let memory_manager = {
            let mem_properties = unsafe {
                inner
                    .instance
                    .core
                    .get_physical_device_memory_properties(physical_device)
            };
            let memory_types =
                &mem_properties.memory_types[..mem_properties.memory_type_count as usize];
            let limits = &capabilities.properties.limits;
            let config = gpu_alloc::Config::i_am_prototyping(); //TODO?

            let properties = gpu_alloc::DeviceProperties {
                max_memory_allocation_count: limits.max_memory_allocation_count,
                max_memory_allocation_size: u64::MAX, // TODO
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
                buffer_device_address: capabilities.buffer_device_address,
            };

            let known_memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED
                | vk::MemoryPropertyFlags::LAZILY_ALLOCATED;
            let valid_ash_memory_types = memory_types.iter().enumerate().fold(0, |u, (i, mem)| {
                if !known_memory_flags.contains(mem.property_flags) {
                    log::debug!(
                        "Skipping memory type={} for having unknown flags: {:?}",
                        i,
                        mem.property_flags & !known_memory_flags
                    );
                    u
                } else if mem
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && !mem
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
                {
                    //TODO: see if and how we can support this
                    log::debug!("Skipping memory type={} for lack of host coherency", i);
                    u
                } else {
                    u | (1 << i)
                }
            });
            super::MemoryManager {
                allocator: gpu_alloc::GpuAllocator::new(config, properties),
                slab: slab::Slab::new(),
                valid_ash_memory_types,
            }
        };

        let queue = unsafe {
            device
                .core
                .get_device_queue(capabilities.queue_family_index, 0)
        };
        let last_progress = 0;
        let mut timeline_info = vk::SemaphoreTypeCreateInfo {
            semaphore_type: vk::SemaphoreType::TIMELINE,
            initial_value: last_progress,
            ..Default::default()
        };
        let timeline_semaphore_create_info =
            vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
        let timeline_semaphore = unsafe {
            device
                .core
                .create_semaphore(&timeline_semaphore_create_info, None)
                .unwrap()
        };

        let mut naga_flags = spv::WriterFlags::FORCE_POINT_SIZE;
        let shader_debug_path = if desc.validation || desc.capture {
            use std::{env, fs};
            naga_flags |= spv::WriterFlags::DEBUG;
            let dir = env::temp_dir().join("blade");
            let _ = fs::create_dir(&dir);
            Some(dir)
        } else {
            None
        };

        let xr = if let Some(ref xr_desc) = desc.xr {
            let session_info = openxr::vulkan::SessionCreateInfo {
                instance: inner.instance.core.handle().as_raw() as _,
                physical_device: physical_device.as_raw() as _,
                device: device.core.handle().as_raw() as _,
                queue_family_index: capabilities.queue_family_index,
                queue_index: 0,
            };
            match unsafe {
                xr_desc
                    .instance
                    .create_session::<openxr::Vulkan>(xr_desc.system_id, &session_info)
            } {
                Ok((session, frame_wait, frame_stream)) => {
                    let view_type = openxr::ViewConfigurationType::PRIMARY_STEREO;
                    let environment_blend_mode = xr_desc
                        .instance
                        .enumerate_environment_blend_modes(xr_desc.system_id, view_type)
                        .ok()
                        .and_then(|modes| modes.first().copied())
                        .unwrap_or(openxr::EnvironmentBlendMode::OPAQUE);
                    let space = match session.create_reference_space(
                        openxr::ReferenceSpaceType::LOCAL,
                        openxr::Posef::IDENTITY,
                    ) {
                        Ok(space) => space,
                        Err(e) => {
                            log::error!("Failed to create OpenXR local space: {e}");
                            return Err(NotSupportedError::NoSupportedDeviceFound);
                        }
                    };
                    Some(Mutex::new(super::XrSessionState {
                        instance: xr_desc.instance.clone(),
                        system_id: xr_desc.system_id,
                        session,
                        frame_wait,
                        frame_stream,
                        view_type,
                        environment_blend_mode,
                        space: Some(space),
                        predicted_display_time: None,
                    }))
                }
                Err(e) => {
                    log::error!("Failed to create OpenXR session: {e}");
                    return Err(NotSupportedError::NoSupportedDeviceFound);
                }
            }
        } else {
            None
        };

        Ok(super::Context {
            memory: Mutex::new(memory_manager),
            device,
            queue_family_index: capabilities.queue_family_index,
            queue: Mutex::new(super::Queue {
                raw: queue,
                timeline_semaphore,
                last_progress,
            }),
            physical_device,
            naga_flags,
            shader_debug_path,
            min_buffer_alignment,
            min_uniform_buffer_offset_alignment: capabilities
                .properties
                .limits
                .min_uniform_buffer_offset_alignment,
            sample_count_flags: capabilities
                .properties
                .limits
                .framebuffer_color_sample_counts
                & capabilities
                    .properties
                    .limits
                    .framebuffer_depth_sample_counts,
            dual_source_blending: capabilities.dual_source_blending,
            shader_float16: capabilities.shader_float16,
            cooperative_matrix: capabilities.cooperative_matrix,
            binding_array: capabilities.binding_array,
            memory_budget: capabilities.memory_budget,
            inner,
            xr,
        })
    }

    pub(super) fn set_object_name<T: vk::Handle>(&self, object: T, name: &str) {
        let name_cstr = ffi::CString::new(name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(object)
            .object_name(&name_cstr);
        let _ = unsafe {
            self.device
                .debug_utils
                .set_debug_utils_object_name(&name_info)
        };
    }

    pub fn capabilities(&self) -> crate::Capabilities {
        crate::Capabilities {
            binding_array: self.binding_array,
            ray_query: match self.device.ray_tracing {
                Some(_) => crate::ShaderVisibility::all(),
                None => crate::ShaderVisibility::empty(),
            },
            sample_count_mask: self.sample_count_flags.as_raw(),
            dual_source_blending: self.dual_source_blending,
            shader_float16: self.shader_float16,
            cooperative_matrix: self.cooperative_matrix,
        }
    }

    pub fn device_information(&self) -> &crate::DeviceInformation {
        &self.device.device_information
    }

    pub fn enumerate() -> Result<Vec<crate::DeviceReport>, NotSupportedError> {
        let desc = crate::ContextDesc::default();
        let inner = unsafe { super::VulkanInstance::create(&desc)? };
        Ok(inspect_devices(
            &inner.instance,
            inner.driver_api_version,
            None,
        ))
    }

    pub fn enumerate_devices(&self) -> Vec<crate::DeviceReport> {
        inspect_devices(
            &self.inner.instance,
            self.inner.driver_api_version,
            Some(self.physical_device),
        )
    }

    pub fn memory_stats(&self) -> crate::MemoryStats {
        if !self.memory_budget {
            return crate::MemoryStats::default();
        }

        let mut budget_properties = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut mem_properties2 =
            vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget_properties);

        unsafe {
            self.inner
                .instance
                .get_physical_device_properties2
                .get_physical_device_memory_properties2(self.physical_device, &mut mem_properties2);
        }

        // Copy what we need before accessing budget_properties
        let heap_count = mem_properties2.memory_properties.memory_heap_count as usize;
        let heap_flags: Vec<_> = mem_properties2.memory_properties.memory_heaps[..heap_count]
            .iter()
            .map(|h| h.flags)
            .collect();
        // Now mem_properties2 borrow is released, we can access budget_properties
        let _ = mem_properties2;

        let mut total_budget = 0u64;
        let mut total_usage = 0u64;
        for (i, flags) in heap_flags.iter().enumerate() {
            if flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                total_budget += budget_properties.heap_budget[i];
                total_usage += budget_properties.heap_usage[i];
            }
        }

        crate::MemoryStats {
            budget: total_budget,
            usage: total_usage,
        }
    }
}

impl Drop for super::Context {
    fn drop(&mut self) {
        unsafe {
            self.xr = None;
            if let Ok(queue) = self.queue.lock() {
                let _ = self.device.core.queue_wait_idle(queue.raw);
                self.device
                    .core
                    .destroy_semaphore(queue.timeline_semaphore, None);
            }
            self.device.core.destroy_device(None);
            // inner.drop() destroys the Vulkan instance
        }
    }
}
