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

struct Surface {
    view: *mut objc::runtime::Object,
    render_layer: metal::MetalLayer,
    format: crate::TextureFormat,
}

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

struct DeviceInfo {
    language_version: metal::MTLLanguageVersion,
}

pub struct Context {
    device: Mutex<metal::Device>,
    queue: Arc<Mutex<metal::CommandQueue>>,
    surface: Option<Mutex<Surface>>,
    capture: Option<metal::CaptureManager>,
    info: DeviceInfo,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Buffer {
    raw: *mut metal::MTLBuffer,
}

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
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Sampler {
    raw: *mut metal::MTLSamplerState,
}

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
pub struct CommandEncoder {
    raw: Option<metal::CommandBuffer>,
    name: String,
    queue: Arc<Mutex<metal::CommandQueue>>,
}

#[derive(Debug)]
struct BindGroupInfo {
    visibility: crate::ShaderVisibility,
    targets: Box<[u32]>,
}

#[derive(Debug)]
struct PipelineLayout {
    bind_groups: Box<[BindGroupInfo]>,
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
    bind_groups: &'a [BindGroupInfo],
}

#[derive(Debug)]
pub struct RenderPipelineContext<'a> {
    encoder: &'a mut metal::RenderCommandEncoder,
    primitive_type: metal::MTLPrimitiveType,
    bind_groups: &'a [BindGroupInfo],
}

fn map_texture_format(format: crate::TextureFormat) -> metal::MTLPixelFormat {
    use crate::TextureFormat as Tf;
    use metal::MTLPixelFormat::*;
    match format {
        Tf::Rgba8Unorm => RGBA8Unorm,
        Tf::Rgba8UnormSrgb => RGBA8Unorm_sRGB,
        Tf::Bgra8UnormSrgb => BGRA8Unorm_sRGB,
        Tf::Rgba16Float => RGBA16Float,
        Tf::R32Float => R32Float,
        Tf::Depth32Float => Depth32Float,
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

fn map_attribute_format(format: crate::VertexFormat) -> metal::MTLAttributeFormat {
    match format {
        crate::VertexFormat::F32Vec3 => metal::MTLAttributeFormat::Float3,
    }
}

impl Context {
    pub unsafe fn init(desc: super::ContextDesc) -> Result<Self, super::NotSupportedError> {
        if desc.validation {
            std::env::set_var("METAL_DEVICE_WRAPPER_TYPE", "1");
        }
        let device = metal::Device::system_default().ok_or(super::NotSupportedError)?;
        let queue = device.new_command_queue();

        let capture = if desc.capture {
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

        Ok(Context {
            device: Mutex::new(device),
            queue: Arc::new(Mutex::new(queue)),
            surface: None,
            capture,
            info: DeviceInfo {
                //TODO: determine based on OS version
                language_version: metal::MTLLanguageVersion::V2_4,
            },
        })
    }

    pub unsafe fn init_windowed<
        I: raw_window_handle::HasRawWindowHandle + raw_window_handle::HasRawDisplayHandle,
    >(
        window: &I,
        desc: super::ContextDesc,
    ) -> Result<Self, super::NotSupportedError> {
        let mut context = Self::init(desc)?;

        let surface = match window.raw_window_handle() {
            #[cfg(target_os = "ios")]
            raw_window_handle::RawWindowHandle::UiKit(handle) => {
                Surface::from_view(handle.ui_view as *mut _)
            }
            #[cfg(target_os = "macos")]
            raw_window_handle::RawWindowHandle::AppKit(handle) => {
                Surface::from_view(handle.ns_view as *mut _)
            }
            _ => return Err(crate::NotSupportedError),
        };

        context.surface = Some(Mutex::new(surface));
        Ok(context)
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
}

#[hidden_trait::expose]
impl crate::traits::CommandDevice for Context {
    type CommandEncoder = CommandEncoder;
    type SyncPoint = SyncPoint;

    fn create_command_encoder(&self, desc: super::CommandEncoderDesc) -> CommandEncoder {
        CommandEncoder {
            raw: None,
            name: desc.name.to_string(),
            queue: Arc::clone(&self.queue),
        }
    }

    fn destroy_command_encoder(&self, _command_encoder: CommandEncoder) {}

    fn submit(&self, encoder: &mut CommandEncoder) -> SyncPoint {
        let cmd_buf = encoder.raw.take().unwrap();
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
            descriptor.set_vertex_format(map_attribute_format(mesh.vertex_format));
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
