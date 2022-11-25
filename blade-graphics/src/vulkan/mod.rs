use ash::{
    extensions::{ext, khr},
    vk,
};
use naga::back::spv;
use std::{ffi, marker::PhantomData, sync::Mutex};

mod command;
mod pipeline;
mod resource;

struct Extensions {
    debug_utils: ext::DebugUtils,
    draw_indirect_count: Option<khr::DrawIndirectCount>,
    timeline_semaphore: Option<khr::TimelineSemaphore>,
}

pub struct Context {
    device: ash::Device,
    queue: Mutex<vk::Queue>,
    physical_device: vk::PhysicalDevice,
    naga_flags: spv::WriterFlags,
    extensions: Extensions,
    instance: ash::Instance,
    entry: ash::Entry,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Buffer {
    raw: vk::Buffer,
}

impl Buffer {
    pub fn data(&self) -> *mut u8 {
        unimplemented!()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Texture {
    raw: vk::Image,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureView {
    raw: vk::ImageView,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampler {
    raw: vk::Sampler,
}

#[derive(Debug)]
struct DescriptorSetLayout {
    raw: vk::DescriptorSetLayout,
    update_template: vk::DescriptorUpdateTemplate,
}

#[derive(Debug)]
struct PipelineLayout {
    raw: vk::PipelineLayout,
    descriptor_set_layouts: Vec<Option<DescriptorSetLayout>>,
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
    data: Vec<u8>,
}
pub struct TransferCommandEncoder<'a> {
    phantom: PhantomData<&'a CommandEncoder>,
}
pub struct ComputePipelineContext<'a> {
    phantom: PhantomData<&'a CommandEncoder>,
}
pub struct RenderCommandEncoder<'a> {
    phantom: PhantomData<&'a CommandEncoder>,
}
pub struct RenderPipelineContext<'a> {
    phantom: PhantomData<&'a CommandEncoder>,
}

pub struct SyncPoint {}

fn inspect_adapter(
    _phd: vk::PhysicalDevice,
    _instance: &ash::Instance,
    _driver_api_version: u32,
) -> bool {
    //TODO
    true
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

            let mut instance_extensions: Vec<&'static ffi::CStr> = Vec::new();
            instance_extensions.push(ext::DebugUtils::name());
            instance_extensions.push(vk::KhrGetPhysicalDeviceProperties2Fn::name());

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

        let extensions = Extensions {
            debug_utils: ext::DebugUtils::new(&entry, &instance),
            draw_indirect_count: None,
            timeline_semaphore: None,
        };

        let physical_devices = instance.enumerate_physical_devices().unwrap();
        let physical_device = physical_devices
            .into_iter()
            .find(|&phd| inspect_adapter(phd, &instance, driver_api_version))
            .ok_or(super::NotSupportedError)?;

        let family_index = 0; //TODO

        let device = {
            let family_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(family_index)
                .queue_priorities(&[1.0])
                .build();
            let family_infos = [family_info];

            let mut device_extensions: Vec<&'static ffi::CStr> = Vec::new();
            if is_vulkan_portability {
                device_extensions.push(vk::KhrPortabilitySubsetFn::name());
            }

            let str_pointers = device_extensions
                .iter()
                .map(|&s| s.as_ptr())
                .collect::<Vec<_>>();

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&family_infos)
                .enabled_extension_names(&str_pointers);
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };

        let queue = device.get_device_queue(family_index, 0);

        let mut naga_flags = spv::WriterFlags::ADJUST_COORDINATE_SPACE;
        if desc.validation {
            naga_flags |= spv::WriterFlags::DEBUG;
        }

        Ok(Context {
            device,
            queue: Mutex::new(queue),
            physical_device,
            naga_flags,
            extensions,
            instance,
            entry,
        })
    }

    pub fn create_command_encoder(&self, desc: super::CommandEncoderDesc) -> CommandEncoder {
        CommandEncoder { data: Vec::new() }
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
            self.extensions
                .debug_utils
                .set_debug_utils_object_name(self.device.handle(), &name_info)
        };
    }
}
