use std::{
    marker::PhantomData,
    ptr,
    sync::{Arc, Mutex},
    thread, time,
};

use metal::foreign_types::{ForeignType as _, ForeignTypeRef as _};

mod command;
mod pipeline;
mod resource;
mod surface;

const MAX_TIMESTAMPS: u64 = crate::limits::PASS_COUNT as u64 * 2;

pub type PlatformError = ();

pub struct Surface {
    view: *mut objc::runtime::Object,
    render_layer: metal::MetalLayer,
    info: crate::SurfaceInfo,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

#[derive(Debug)]
pub struct Frame {
    drawable: metal::MetalDrawable,
    texture: metal::Texture,
}

impl Frame {
    pub fn texture(&self) -> Texture {
        Texture {
            raw: self.texture.as_ptr(),
        }
    }

    pub fn texture_view(&self) -> TextureView {
        TextureView {
            raw: self.texture.as_ptr(),
        }
    }
}

#[derive(Debug, Clone)]
struct PrivateInfo {
    language_version: metal::MTLLanguageVersion,
    enable_debug_groups: bool,
    enable_dispatch_type: bool,
    timestamp_counter_set: Option<metal::CounterSet>,
}

pub struct Context {
    device: Mutex<metal::Device>,
    queue: Arc<Mutex<metal::CommandQueue>>,
    capture: Option<metal::CaptureManager>,
    info: PrivateInfo,
    device_information: crate::DeviceInformation,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Buffer {
    raw: *mut metal::MTLBuffer,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Default for Buffer {
    fn default() -> Self {
        Self {
            raw: ptr::null_mut(),
        }
    }
}

impl Buffer {
    fn as_ref(&self) -> &metal::BufferRef {
        unsafe { metal::BufferRef::from_ptr(self.raw) }
    }

    pub fn data(&self) -> *mut u8 {
        self.as_ref().contents() as *mut u8
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Texture {
    raw: *mut metal::MTLTexture,
}

unsafe impl Send for Texture {}
unsafe impl Sync for Texture {}

impl Default for Texture {
    fn default() -> Self {
        Self {
            raw: ptr::null_mut(),
        }
    }
}

impl Texture {
    fn as_ref(&self) -> &metal::TextureRef {
        unsafe { metal::TextureRef::from_ptr(self.raw) }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct TextureView {
    raw: *mut metal::MTLTexture,
}

unsafe impl Send for TextureView {}
unsafe impl Sync for TextureView {}

impl Default for TextureView {
    fn default() -> Self {
        Self {
            raw: ptr::null_mut(),
        }
    }
}

impl TextureView {
    fn as_ref(&self) -> &metal::TextureRef {
        unsafe { metal::TextureRef::from_ptr(self.raw) }
    }

    /// Create a TextureView from a raw Metal Texture.
    /// Does not keep a reference, need not being destoryed.
    pub fn from_metal_texture(raw: &metal::TextureRef) -> Self {
        Self { raw: raw.as_ptr() }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Sampler {
    raw: *mut metal::MTLSamplerState,
}

unsafe impl Send for Sampler {}
unsafe impl Sync for Sampler {}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            raw: ptr::null_mut(),
        }
    }
}

impl Sampler {
    fn as_ref(&self) -> &metal::SamplerStateRef {
        unsafe { metal::SamplerStateRef::from_ptr(self.raw) }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct AccelerationStructure {
    raw: *mut metal::MTLAccelerationStructure,
}

unsafe impl Send for AccelerationStructure {}
unsafe impl Sync for AccelerationStructure {}

impl Default for AccelerationStructure {
    fn default() -> Self {
        Self {
            raw: ptr::null_mut(),
        }
    }
}

impl AccelerationStructure {
    fn as_ref(&self) -> &metal::AccelerationStructureRef {
        unsafe { metal::AccelerationStructureRef::from_ptr(self.raw) }
    }
}

//TODO: make this copyable?
#[derive(Clone, Debug)]
pub struct SyncPoint {
    cmd_buf: metal::CommandBuffer,
}

#[derive(Debug)]
struct TimingData {
    pass_names: Vec<String>,
    sample_buffer: metal::CounterSampleBuffer,
}

#[derive(Debug)]
pub struct CommandEncoder {
    raw: Option<metal::CommandBuffer>,
    name: String,
    queue: Arc<Mutex<metal::CommandQueue>>,
    enable_debug_groups: bool,
    enable_dispatch_type: bool,
    has_open_debug_group: bool,
    timing_datas: Option<Box<[TimingData]>>,
    timings: crate::Timings,
}

#[derive(Debug)]
struct ShaderDataMapping {
    visibility: crate::ShaderVisibility,
    targets: Box<[u32]>,
}

#[derive(Debug)]
struct PipelineLayout {
    group_mappings: Box<[ShaderDataMapping]>,
    group_infos: Box<[crate::ShaderDataInfo]>,
    sizes_buffer_slot: Option<u32>,
}

#[derive(Debug)]
pub struct ComputePipeline {
    raw: metal::ComputePipelineState,
    name: String,
    #[allow(dead_code)]
    lib: metal::Library,
    layout: PipelineLayout,
    wg_size: metal::MTLSize,
    wg_memory_sizes: Box<[u32]>,
}

impl ComputePipeline {
    pub fn get_workgroup_size(&self) -> [u32; 3] {
        [
            self.wg_size.width as u32,
            self.wg_size.height as u32,
            self.wg_size.depth as u32,
        ]
    }
}

#[derive(Debug)]
pub struct RenderPipeline {
    raw: metal::RenderPipelineState,
    name: String,
    #[allow(dead_code)]
    vs_lib: metal::Library,
    #[allow(dead_code)]
    fs_lib: metal::Library,
    layout: PipelineLayout,
    primitive_type: metal::MTLPrimitiveType,
    triangle_fill_mode: metal::MTLTriangleFillMode,
    front_winding: metal::MTLWinding,
    cull_mode: metal::MTLCullMode,
    depth_clip_mode: metal::MTLDepthClipMode,
    depth_stencil: Option<(metal::DepthStencilState, super::DepthBiasState)>,
}

#[derive(Debug)]
pub struct TransferCommandEncoder<'a> {
    raw: metal::BlitCommandEncoder,
    phantom: PhantomData<&'a CommandEncoder>,
}

#[derive(Debug)]
pub struct AccelerationStructureCommandEncoder<'a> {
    raw: metal::AccelerationStructureCommandEncoder,
    phantom: PhantomData<&'a CommandEncoder>,
}

#[derive(Debug)]
pub struct ComputeCommandEncoder<'a> {
    raw: metal::ComputeCommandEncoder,
    phantom: PhantomData<&'a CommandEncoder>,
}

#[derive(Debug)]
pub struct RenderCommandEncoder<'a> {
    raw: metal::RenderCommandEncoder,
    phantom: PhantomData<&'a CommandEncoder>,
}

pub struct PipelineContext<'a> {
    //raw: metal::ArgumentEncoderRef,
    cs_encoder: Option<&'a metal::ComputeCommandEncoderRef>,
    vs_encoder: Option<&'a metal::RenderCommandEncoderRef>,
    fs_encoder: Option<&'a metal::RenderCommandEncoderRef>,
    targets: &'a [u32],
}

#[derive(Debug)]
pub struct ComputePipelineContext<'a> {
    encoder: &'a mut metal::ComputeCommandEncoder,
    wg_size: metal::MTLSize,
    group_mappings: &'a [ShaderDataMapping],
}

#[derive(Debug)]
pub struct RenderPipelineContext<'a> {
    encoder: &'a mut metal::RenderCommandEncoder,
    primitive_type: metal::MTLPrimitiveType,
    group_mappings: &'a [ShaderDataMapping],
}

fn map_texture_format(format: crate::TextureFormat) -> metal::MTLPixelFormat {
    use crate::TextureFormat as Tf;
    use metal::MTLPixelFormat::*;
    match format {
        Tf::R8Unorm => R8Unorm,
        Tf::Rg8Unorm => RG8Unorm,
        Tf::Rg8Snorm => RG8Snorm,
        Tf::Rgba8Unorm => RGBA8Unorm,
        Tf::Rgba8UnormSrgb => RGBA8Unorm_sRGB,
        Tf::Bgra8Unorm => BGRA8Unorm,
        Tf::Bgra8UnormSrgb => BGRA8Unorm_sRGB,
        Tf::Rgba8Snorm => RGBA8Snorm,
        Tf::R16Float => R16Float,
        Tf::Rg16Float => RG16Float,
        Tf::Rgba16Float => RGBA16Float,
        Tf::R32Float => R32Float,
        Tf::Rg32Float => RG32Float,
        Tf::Rgba32Float => RGBA32Float,
        Tf::R32Uint => R32Uint,
        Tf::Rg32Uint => RG32Uint,
        Tf::Rgba32Uint => RGBA32Uint,
        Tf::Depth32Float => Depth32Float,
        Tf::Bc1Unorm => BC1_RGBA,
        Tf::Bc1UnormSrgb => BC1_RGBA_sRGB,
        Tf::Bc2Unorm => BC2_RGBA,
        Tf::Bc2UnormSrgb => BC2_RGBA_sRGB,
        Tf::Bc3Unorm => BC3_RGBA,
        Tf::Bc3UnormSrgb => BC3_RGBA_sRGB,
        Tf::Bc4Unorm => BC4_RUnorm,
        Tf::Bc4Snorm => BC4_RSnorm,
        Tf::Bc5Unorm => BC5_RGUnorm,
        Tf::Bc5Snorm => BC5_RGSnorm,
        Tf::Bc6hUfloat => BC6H_RGBUfloat,
        Tf::Bc6hFloat => BC6H_RGBFloat,
        Tf::Bc7Unorm => BC7_RGBAUnorm,
        Tf::Bc7UnormSrgb => BC7_RGBAUnorm_sRGB,
        Tf::Rgb10a2Unorm => RGB10A2Unorm,
        Tf::Rg11b10Ufloat => RG11B10Float,
        Tf::Rgb9e5Ufloat => RGB9E5Float,
    }
}

fn map_compare_function(fun: crate::CompareFunction) -> metal::MTLCompareFunction {
    use crate::CompareFunction as Cf;
    use metal::MTLCompareFunction::*;
    match fun {
        Cf::Never => Never,
        Cf::Less => Less,
        Cf::LessEqual => LessEqual,
        Cf::Equal => Equal,
        Cf::GreaterEqual => GreaterEqual,
        Cf::Greater => Greater,
        Cf::NotEqual => NotEqual,
        Cf::Always => Always,
    }
}

fn map_index_type(ty: crate::IndexType) -> metal::MTLIndexType {
    match ty {
        crate::IndexType::U16 => metal::MTLIndexType::UInt16,
        crate::IndexType::U32 => metal::MTLIndexType::UInt32,
    }
}

fn map_vertex_format(
    format: crate::VertexFormat,
) -> (metal::MTLVertexFormat, metal::MTLAttributeFormat) {
    match format {
        crate::VertexFormat::F32 => (
            metal::MTLVertexFormat::Float,
            metal::MTLAttributeFormat::Float,
        ),
        crate::VertexFormat::F32Vec2 => (
            metal::MTLVertexFormat::Float2,
            metal::MTLAttributeFormat::Float2,
        ),
        crate::VertexFormat::F32Vec3 => (
            metal::MTLVertexFormat::Float3,
            metal::MTLAttributeFormat::Float3,
        ),
        crate::VertexFormat::F32Vec4 => (
            metal::MTLVertexFormat::Float4,
            metal::MTLAttributeFormat::Float4,
        ),
        crate::VertexFormat::U32 => (
            metal::MTLVertexFormat::UInt,
            metal::MTLAttributeFormat::UInt,
        ),
        crate::VertexFormat::U32Vec2 => (
            metal::MTLVertexFormat::UInt2,
            metal::MTLAttributeFormat::UInt2,
        ),
        crate::VertexFormat::U32Vec3 => (
            metal::MTLVertexFormat::UInt3,
            metal::MTLAttributeFormat::UInt3,
        ),
        crate::VertexFormat::U32Vec4 => (
            metal::MTLVertexFormat::UInt4,
            metal::MTLAttributeFormat::UInt4,
        ),
        crate::VertexFormat::I32 => (metal::MTLVertexFormat::Int, metal::MTLAttributeFormat::Int),
        crate::VertexFormat::I32Vec2 => (
            metal::MTLVertexFormat::Int2,
            metal::MTLAttributeFormat::Int2,
        ),
        crate::VertexFormat::I32Vec3 => (
            metal::MTLVertexFormat::Int3,
            metal::MTLAttributeFormat::Int3,
        ),
        crate::VertexFormat::I32Vec4 => (
            metal::MTLVertexFormat::Int4,
            metal::MTLAttributeFormat::Int4,
        ),
    }
}

impl Context {
    pub unsafe fn init(desc: super::ContextDesc) -> Result<Self, super::NotSupportedError> {
        if desc.validation {
            std::env::set_var("METAL_DEVICE_WRAPPER_TYPE", "1");
        }
        if desc.overlay {
            std::env::set_var("MTL_HUD_ENABLED", "1");
        }
        if desc.device_id != 0 {
            log::warn!("Unable to filter devices by ID");
        }

        let device = metal::Device::system_default()
            .ok_or(super::NotSupportedError::NoSupportedDeviceFound)?;
        let queue = device.new_command_queue();

        let auto_capture_everything = false;
        let capture = if desc.capture && auto_capture_everything {
            objc::rc::autoreleasepool(|| {
                let capture_manager = metal::CaptureManager::shared();
                let default_capture_scope = capture_manager.new_capture_scope_with_device(&device);
                capture_manager.set_default_capture_scope(&default_capture_scope);
                capture_manager.start_capture_with_scope(&default_capture_scope);
                default_capture_scope.begin_scope();
                Some(capture_manager.to_owned())
            })
        } else {
            None
        };
        let device_information = crate::DeviceInformation {
            is_software_emulated: false,
            device_name: device.name().to_string(),
            driver_name: "Metal".to_string(),
            driver_info: "".to_string(),
        };

        let mut timestamp_counter_set = None;
        if desc.timing {
            for counter_set in device.counter_sets() {
                if counter_set.name() == "timestamp" {
                    timestamp_counter_set = Some(counter_set);
                }
            }
            if timestamp_counter_set.is_none() {
                log::warn!("Timing counters are not supported by the device");
            } else if !device
                .supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary)
            {
                log::warn!("Timing counters do not support stage boundary");
                timestamp_counter_set = None;
            }
        }

        Ok(Context {
            device: Mutex::new(device),
            queue: Arc::new(Mutex::new(queue)),
            capture,
            info: PrivateInfo {
                //TODO: determine based on OS version
                language_version: metal::MTLLanguageVersion::V2_4,
                enable_debug_groups: desc.capture,
                enable_dispatch_type: true,
                timestamp_counter_set,
            },
            device_information,
        })
    }

    pub fn capabilities(&self) -> crate::Capabilities {
        let device = self.device.lock().unwrap();
        crate::Capabilities {
            ray_query: if device.supports_family(metal::MTLGPUFamily::Apple6) {
                crate::ShaderVisibility::all()
            } else if device.supports_family(metal::MTLGPUFamily::Mac2)
                || device.supports_family(metal::MTLGPUFamily::Metal3)
            {
                crate::ShaderVisibility::COMPUTE
            } else {
                crate::ShaderVisibility::empty()
            },
        }
    }

    pub fn device_information(&self) -> &crate::DeviceInformation {
        &self.device_information
    }

    /// Get an MTLDevice of this context.
    /// This is platform specific API.
    pub fn metal_device(&self) -> metal::Device {
        self.device.lock().unwrap().clone()
    }
}

#[hidden_trait::expose]
impl crate::traits::CommandDevice for Context {
    type CommandEncoder = CommandEncoder;
    type SyncPoint = SyncPoint;

    fn create_command_encoder(&self, desc: super::CommandEncoderDesc) -> CommandEncoder {
        let timing_datas = if let Some(ref counter_set) = self.info.timestamp_counter_set {
            let mut array = Vec::with_capacity(desc.buffer_count as usize);
            let csb_desc = metal::CounterSampleBufferDescriptor::new();
            csb_desc.set_counter_set(counter_set);
            csb_desc.set_storage_mode(metal::MTLStorageMode::Shared);
            csb_desc.set_sample_count(MAX_TIMESTAMPS);
            for i in 0..desc.buffer_count {
                csb_desc.set_label(&format!("{}/counter{}", desc.name, i));
                let sample_buffer = self
                    .device
                    .lock()
                    .unwrap()
                    .new_counter_sample_buffer_with_descriptor(&csb_desc)
                    .unwrap();
                array.push(TimingData {
                    sample_buffer,
                    pass_names: Vec::new(),
                });
            }
            Some(array.into_boxed_slice())
        } else {
            None
        };
        CommandEncoder {
            raw: None,
            name: desc.name.to_string(),
            queue: Arc::clone(&self.queue),
            enable_debug_groups: self.info.enable_debug_groups,
            enable_dispatch_type: self.info.enable_dispatch_type,
            has_open_debug_group: false,
            timing_datas,
            timings: Default::default(),
        }
    }

    fn destroy_command_encoder(&self, _command_encoder: &mut CommandEncoder) {}

    fn submit(&self, encoder: &mut CommandEncoder) -> SyncPoint {
        let cmd_buf = encoder.finish();
        cmd_buf.commit();
        SyncPoint { cmd_buf }
    }

    fn wait_for(&self, sp: &SyncPoint, timeout_ms: u32) -> bool {
        let start = time::Instant::now();
        loop {
            if let metal::MTLCommandBufferStatus::Completed = sp.cmd_buf.status() {
                return true;
            }
            if start.elapsed().as_millis() >= timeout_ms as u128 {
                return false;
            }
            thread::sleep(time::Duration::from_millis(1));
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Some(capture_manager) = self.capture.take() {
            if let Some(scope) = capture_manager.default_capture_scope() {
                scope.end_scope();
            }
            capture_manager.stop_capture();
        }
    }
}

fn make_bottom_level_acceleration_structure_desc(
    meshes: &[crate::AccelerationStructureMesh],
) -> metal::PrimitiveAccelerationStructureDescriptor {
    let mut geometry_descriptors = Vec::with_capacity(meshes.len());
    for mesh in meshes {
        let descriptor = metal::AccelerationStructureTriangleGeometryDescriptor::descriptor();
        descriptor.set_opaque(mesh.is_opaque);
        descriptor.set_vertex_buffer(Some(mesh.vertex_data.buffer.as_ref()));
        descriptor.set_vertex_buffer_offset(mesh.vertex_data.offset);
        descriptor.set_vertex_stride(mesh.vertex_stride as _);
        descriptor.set_triangle_count(mesh.triangle_count as _);
        if let Some(index_type) = mesh.index_type {
            descriptor.set_index_buffer(Some(mesh.index_data.buffer.as_ref()));
            descriptor.set_index_buffer_offset(mesh.index_data.offset);
            descriptor.set_index_type(map_index_type(index_type));
        }
        //TODO: requires macOS-13 ?
        if false {
            let (_, attribute_format) = map_vertex_format(mesh.vertex_format);
            descriptor.set_vertex_format(attribute_format);
            if !mesh.transform_data.buffer.raw.is_null() {
                descriptor
                    .set_transformation_matrix_buffer(Some(mesh.transform_data.buffer.as_ref()));
                descriptor.set_transformation_matrix_buffer_offset(mesh.transform_data.offset);
            }
        }
        geometry_descriptors.push(metal::AccelerationStructureGeometryDescriptor::from(
            descriptor,
        ));
    }

    let geometry_descriptor_array = metal::Array::from_owned_slice(&geometry_descriptors);
    let accel_descriptor = metal::PrimitiveAccelerationStructureDescriptor::descriptor();
    accel_descriptor.set_geometry_descriptors(geometry_descriptor_array);
    accel_descriptor
}

fn _print_class_methods(class: &objc::runtime::Class) {
    let mut count = 0;
    let methods = unsafe { objc::runtime::class_copyMethodList(class, &mut count) };
    println!("Class {} methods:", class.name());
    for i in 0..count {
        let method = unsafe { &**methods.add(i as usize) };
        println!("\t{}", method.name().name());
    }
}

fn _print_class_methods_by_name(class_name: &str) {
    let class = objc::runtime::Class::get(class_name).unwrap();
    _print_class_methods(class);
}

fn _print_class_methods_by_object(foreign_object: &impl metal::foreign_types::ForeignType) {
    let object = foreign_object.as_ptr() as *mut objc::runtime::Object;
    _print_class_methods(unsafe { &*object }.class());
}
