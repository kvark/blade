use ash::vk;
use std::ffi;

pub const NAME: &ffi::CStr = c"VK_NV_cooperative_matrix2";

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Features {
    pub s_type: vk::StructureType,
    pub p_next: *mut ffi::c_void,
    pub workgroup_scope: vk::Bool32,
    pub flexible_dimensions: vk::Bool32,
    pub reductions: vk::Bool32,
    pub conversions: vk::Bool32,
    pub per_element_operations: vk::Bool32,
    pub tensor_addressing: vk::Bool32,
    pub block_loads: vk::Bool32,
}
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Properties {
    pub s_type: vk::StructureType,
    pub p_next: *mut ffi::c_void,
    pub max_workgroup_size: u32,
    pub max_dimension: u32,
    pub reserved_shared_memory: u32,
}
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FlexibleDimensions {
    pub s_type: vk::StructureType,
    pub p_next: *mut ffi::c_void,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_type: vk::ComponentTypeKHR,
    pub b_type: vk::ComponentTypeKHR,
    pub c_type: vk::ComponentTypeKHR,
    pub result_type: vk::ComponentTypeKHR,
    pub saturating: vk::Bool32,
    pub scope: vk::ScopeKHR,
    pub invocations: u32,
}
macro_rules! structure {
    ($ty:ty, $id:expr) => {
        impl Default for $ty {
            fn default() -> Self {
                Self {
                    s_type: vk::StructureType::from_raw($id),
                    ..unsafe { std::mem::zeroed() }
                }
            }
        }
        unsafe impl vk::TaggedStructure for $ty {
            const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw($id);
        }
    };
}
structure!(Features, 1_000_593_000);
structure!(FlexibleDimensions, 1_000_593_001);
structure!(Properties, 1_000_593_002);
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for Features {}
unsafe impl vk::ExtendsDeviceCreateInfo for Features {}
unsafe impl vk::ExtendsPhysicalDeviceProperties2 for Properties {}

pub type Query =
    unsafe extern "system" fn(vk::PhysicalDevice, *mut u32, *mut FlexibleDimensions) -> vk::Result;

pub unsafe fn load(entry: &ash::Entry, instance: vk::Instance) -> Option<Query> {
    unsafe {
        entry
            .get_instance_proc_addr(
                instance,
                c"vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV".as_ptr(),
            )
            .map(|p| std::mem::transmute::<unsafe extern "system" fn(), Query>(p))
    }
}

pub unsafe fn query(function: Query, device: vk::PhysicalDevice) -> Vec<FlexibleDimensions> {
    loop {
        let mut count = 0;
        if unsafe { function(device, &mut count, std::ptr::null_mut()) } != vk::Result::SUCCESS {
            return Vec::new();
        }
        let mut properties = vec![FlexibleDimensions::default(); count as usize];
        match unsafe { function(device, &mut count, properties.as_mut_ptr()) } {
            vk::Result::SUCCESS => {
                properties.truncate(count as usize);
                return properties;
            }
            vk::Result::INCOMPLETE => continue,
            _ => return Vec::new(),
        }
    }
}
