use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;
use std::{mem, ptr};

struct Allocation {
    memory: vk::DeviceMemory,
    offset: u64,
    data: *mut u8,
    handle: usize,
}

impl Drop for super::Context {
    fn drop(&mut self) {
        let mut manager = self.memory.lock().unwrap();
        let blocks = manager.slab.drain().collect::<Vec<_>>();
        for block in blocks {
            unsafe {
                manager
                    .allocator
                    .dealloc(AshMemoryDevice::wrap(&self.device.core), block)
            };
        }
    }
}

impl super::Context {
    fn allocate_memory(
        &self,
        requirements: vk::MemoryRequirements,
        memory: crate::Memory,
    ) -> Allocation {
        let mut manager = self.memory.lock().unwrap();
        let device_address_usage = if self.device.ray_tracing.is_some() {
            gpu_alloc::UsageFlags::DEVICE_ADDRESS
        } else {
            gpu_alloc::UsageFlags::empty()
        };
        let alloc_usage = match memory {
            crate::Memory::Device => {
                gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS | device_address_usage
            }
            crate::Memory::Shared => {
                gpu_alloc::UsageFlags::HOST_ACCESS
                    | gpu_alloc::UsageFlags::DOWNLOAD
                    | gpu_alloc::UsageFlags::UPLOAD
                    | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
                    | device_address_usage
            }
            crate::Memory::Upload => {
                gpu_alloc::UsageFlags::HOST_ACCESS | gpu_alloc::UsageFlags::UPLOAD
            }
        };
        let memory_types = requirements.memory_type_bits & manager.valid_ash_memory_types;
        let mut block = unsafe {
            manager
                .allocator
                .alloc(
                    AshMemoryDevice::wrap(&self.device.core),
                    gpu_alloc::Request {
                        size: requirements.size,
                        align_mask: requirements.alignment - 1,
                        usage: alloc_usage,
                        memory_types,
                    },
                )
                .unwrap()
        };

        let data = match memory {
            crate::Memory::Device => ptr::null_mut(),
            crate::Memory::Shared | crate::Memory::Upload => unsafe {
                block
                    .map(
                        AshMemoryDevice::wrap(&self.device.core),
                        0,
                        requirements.size as usize,
                    )
                    .unwrap()
                    .as_ptr()
            },
        };
        Allocation {
            memory: *block.memory(),
            offset: block.offset(),
            data,
            handle: manager.slab.insert(block),
        }
    }

    fn free_memory(&self, handle: usize) {
        let mut manager = self.memory.lock().unwrap();
        let block = manager.slab.remove(handle);
        unsafe {
            manager
                .allocator
                .dealloc(AshMemoryDevice::wrap(&self.device.core), block);
        }
    }

    //TODO: move these into `ResourceDevice` trait when ready
    pub fn get_bottom_level_acceleration_structure_sizes(
        &self,
        meshes: &[crate::AccelerationStructureMesh],
    ) -> crate::AccelerationStructureSizes {
        let blas_input = self.device.map_acceleration_structure_meshes(meshes);
        let rt = self.device.ray_tracing.as_ref().unwrap();
        let sizes_raw = unsafe {
            rt.acceleration_structure
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &blas_input.build_info,
                    &blas_input.max_primitive_counts,
                )
        };
        crate::AccelerationStructureSizes {
            data: sizes_raw.acceleration_structure_size,
            scratch: sizes_raw.build_scratch_size,
        }
    }

    pub fn get_top_level_acceleration_structure_sizes(
        &self,
        instance_count: u32,
    ) -> crate::AccelerationStructureSizes {
        let geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder().build(),
            })
            .build();
        let geometries = [geometry];
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries)
            .build();

        let rt = self.device.ray_tracing.as_ref().unwrap();
        let sizes_raw = unsafe {
            rt.acceleration_structure
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[instance_count],
                )
        };
        crate::AccelerationStructureSizes {
            data: sizes_raw.acceleration_structure_size,
            scratch: sizes_raw.build_scratch_size,
        }
    }

    pub fn create_acceleration_structure_instance_buffer(
        &self,
        instances: &[crate::AccelerationStructureInstance],
        bottom_level: &[super::AccelerationStructure],
    ) -> super::Buffer {
        let buffer = self.create_buffer(crate::BufferDesc {
            name: "instance buffer",
            size: (instances.len().max(1) * mem::size_of::<vk::AccelerationStructureInstanceKHR>())
                as u64,
            memory: crate::Memory::Shared,
        });
        let rt = self.device.ray_tracing.as_ref().unwrap();
        for (i, instance) in instances.iter().enumerate() {
            let device_address_info = vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                .acceleration_structure(
                    bottom_level[instance.acceleration_structure_index as usize].raw,
                );
            let vk_instance = vk::AccelerationStructureInstanceKHR {
                transform: unsafe { mem::transmute(instance.transform) },
                instance_custom_index_and_mask: vk::Packed24_8::new(
                    instance.custom_index,
                    instance.mask as u8,
                ),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: unsafe {
                        rt.acceleration_structure
                            .get_acceleration_structure_device_address(&device_address_info)
                    },
                },
            };
            unsafe {
                ptr::write(
                    (buffer.data() as *mut vk::AccelerationStructureInstanceKHR).add(i),
                    vk_instance,
                );
            }
        }
        buffer
    }
}

#[hidden_trait::expose]
impl crate::traits::ResourceDevice for super::Context {
    type Buffer = super::Buffer;
    type Texture = super::Texture;
    type TextureView = super::TextureView;
    type Sampler = super::Sampler;
    type AccelerationStructure = super::AccelerationStructure;

    fn create_buffer(&self, desc: crate::BufferDesc) -> super::Buffer {
        use vk::BufferUsageFlags as Buf;
        let mut vk_info = vk::BufferCreateInfo::builder()
            .size(desc.size)
            .usage(
                Buf::TRANSFER_SRC
                    | Buf::TRANSFER_DST
                    | Buf::STORAGE_BUFFER
                    | Buf::INDEX_BUFFER
                    | Buf::VERTEX_BUFFER
                    | Buf::INDIRECT_BUFFER,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        if self.device.ray_tracing.is_some() {
            vk_info.usage |=
                Buf::SHADER_DEVICE_ADDRESS | Buf::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        }

        let raw = unsafe { self.device.core.create_buffer(&vk_info, None).unwrap() };
        let requirements = unsafe { self.device.core.get_buffer_memory_requirements(raw) };
        let allocation = self.allocate_memory(requirements, desc.memory);

        log::info!(
            "Creating buffer {:?} of size {}, name '{}', handle {:?}",
            raw,
            desc.size,
            desc.name,
            allocation.handle
        );
        unsafe {
            self.device
                .core
                .bind_buffer_memory(raw, allocation.memory, allocation.offset)
                .unwrap()
        };
        if !desc.name.is_empty() {
            self.set_object_name(vk::ObjectType::BUFFER, raw, desc.name);
        }

        super::Buffer {
            raw,
            memory_handle: allocation.handle,
            mapped_data: allocation.data,
        }
    }

    fn sync_buffer(&self, _buffer: super::Buffer) {}

    fn destroy_buffer(&self, buffer: super::Buffer) {
        log::info!(
            "Destroying buffer {:?}, handle {:?}",
            buffer.raw,
            buffer.memory_handle
        );
        unsafe { self.device.core.destroy_buffer(buffer.raw, None) };
        self.free_memory(buffer.memory_handle);
    }

    fn create_texture(&self, desc: crate::TextureDesc) -> super::Texture {
        let mut create_flags = vk::ImageCreateFlags::empty();
        if desc.dimension == crate::TextureDimension::D2
            && desc.size.depth % 6 == 0
            //&& desc.sample_count == 1
            && desc.size.width == desc.size.height
        {
            create_flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }

        let vk_info = vk::ImageCreateInfo::builder()
            .flags(create_flags)
            .image_type(map_texture_dimension(desc.dimension))
            .format(super::map_texture_format(desc.format))
            .extent(super::map_extent_3d(&desc.size))
            .mip_levels(desc.mip_level_count)
            .array_layers(desc.array_layer_count)
            .samples(vk::SampleCountFlags::from_raw(1)) // desc.sample_count
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(map_texture_usage(desc.usage, desc.format.aspects()))
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let raw = unsafe { self.device.core.create_image(&vk_info, None).unwrap() };
        let requirements = unsafe { self.device.core.get_image_memory_requirements(raw) };
        let allocation = self.allocate_memory(requirements, crate::Memory::Device);

        log::info!(
            "Creating texture {:?} of size {} and format {:?}, name '{}', handle {:?}",
            raw,
            desc.size,
            desc.format,
            desc.name,
            allocation.handle
        );
        unsafe {
            self.device
                .core
                .bind_image_memory(raw, allocation.memory, allocation.offset)
                .unwrap()
        };
        if !desc.name.is_empty() {
            self.set_object_name(vk::ObjectType::IMAGE, raw, desc.name);
        }

        super::Texture {
            raw,
            memory_handle: allocation.handle,
            target_size: [desc.size.width as u16, desc.size.height as u16],
            format: desc.format,
        }
    }

    fn destroy_texture(&self, texture: super::Texture) {
        log::info!(
            "Destroying texture {:?}, handle {:?}",
            texture.raw,
            texture.memory_handle
        );
        unsafe { self.device.core.destroy_image(texture.raw, None) };
        self.free_memory(texture.memory_handle);
    }

    fn create_texture_view(&self, desc: crate::TextureViewDesc) -> super::TextureView {
        let aspects = desc.format.aspects();
        let subresource_range = super::map_subresource_range(desc.subresources, aspects);
        let vk_info = vk::ImageViewCreateInfo::builder()
            .image(desc.texture.raw)
            .view_type(map_view_dimension(desc.dimension))
            .format(super::map_texture_format(desc.format))
            .subresource_range(subresource_range);

        let raw = unsafe { self.device.core.create_image_view(&vk_info, None).unwrap() };
        if !desc.name.is_empty() {
            self.set_object_name(vk::ObjectType::IMAGE_VIEW, raw, desc.name);
        }

        super::TextureView {
            raw,
            target_size: [
                (desc.texture.target_size[0] >> desc.subresources.base_mip_level).max(1),
                (desc.texture.target_size[1] >> desc.subresources.base_mip_level).max(1),
            ],
            aspects,
        }
    }

    fn destroy_texture_view(&self, view: super::TextureView) {
        unsafe { self.device.core.destroy_image_view(view.raw, None) };
    }

    fn create_sampler(&self, desc: crate::SamplerDesc) -> super::Sampler {
        let mut vk_info = vk::SamplerCreateInfo::builder()
            .mag_filter(map_filter_mode(desc.mag_filter))
            .min_filter(map_filter_mode(desc.min_filter))
            .mipmap_mode(map_mip_filter_mode(desc.mipmap_filter))
            .address_mode_u(map_address_mode(desc.address_modes[0]))
            .address_mode_v(map_address_mode(desc.address_modes[1]))
            .address_mode_w(map_address_mode(desc.address_modes[2]))
            .min_lod(desc.lod_min_clamp)
            .max_lod(desc.lod_max_clamp.unwrap_or(vk::LOD_CLAMP_NONE));

        if let Some(fun) = desc.compare {
            vk_info = vk_info
                .compare_enable(true)
                .compare_op(super::map_comparison(fun));
        }
        if desc.anisotropy_clamp > 1 {
            vk_info = vk_info
                .anisotropy_enable(true)
                .max_anisotropy(desc.anisotropy_clamp as f32);
        }
        if let Some(color) = desc.border_color {
            vk_info = vk_info.border_color(map_border_color(color));
        }

        let raw = unsafe { self.device.core.create_sampler(&vk_info, None).unwrap() };
        if !desc.name.is_empty() {
            self.set_object_name(vk::ObjectType::SAMPLER, raw, desc.name);
        }

        super::Sampler { raw }
    }

    fn destroy_sampler(&self, sampler: super::Sampler) {
        unsafe { self.device.core.destroy_sampler(sampler.raw, None) };
    }

    fn create_acceleration_structure(
        &self,
        desc: crate::AccelerationStructureDesc,
    ) -> super::AccelerationStructure {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(desc.size)
            .usage(vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.core.create_buffer(&buffer_info, None).unwrap() };
        let requirements = unsafe { self.device.core.get_buffer_memory_requirements(buffer) };
        let allocation = self.allocate_memory(requirements, crate::Memory::Device);

        unsafe {
            self.device
                .core
                .bind_buffer_memory(buffer, allocation.memory, allocation.offset)
                .unwrap()
        };

        let raw_ty = match desc.ty {
            crate::AccelerationStructureType::TopLevel => {
                vk::AccelerationStructureTypeKHR::TOP_LEVEL
            }
            crate::AccelerationStructureType::BottomLevel => {
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
            }
        };
        let vk_info = vk::AccelerationStructureCreateInfoKHR::builder()
            .ty(raw_ty)
            .buffer(buffer)
            .size(desc.size);

        let rt = self.device.ray_tracing.as_ref().unwrap();
        let raw = unsafe {
            rt.acceleration_structure
                .create_acceleration_structure(&vk_info, None)
                .unwrap()
        };

        if !desc.name.is_empty() {
            self.set_object_name(vk::ObjectType::BUFFER, buffer, desc.name);
            self.set_object_name(vk::ObjectType::ACCELERATION_STRUCTURE_KHR, raw, desc.name);
        }
        super::AccelerationStructure {
            raw,
            buffer,
            memory_handle: allocation.handle,
        }
    }

    fn destroy_acceleration_structure(&self, acceleration_structure: super::AccelerationStructure) {
        let rt = self.device.ray_tracing.as_ref().unwrap();
        unsafe {
            rt.acceleration_structure
                .destroy_acceleration_structure(acceleration_structure.raw, None);
            self.device
                .core
                .destroy_buffer(acceleration_structure.buffer, None);
        }
        self.free_memory(acceleration_structure.memory_handle);
    }
}

fn map_texture_dimension(dimension: crate::TextureDimension) -> vk::ImageType {
    match dimension {
        crate::TextureDimension::D1 => vk::ImageType::TYPE_1D,
        crate::TextureDimension::D2 => vk::ImageType::TYPE_2D,
        crate::TextureDimension::D3 => vk::ImageType::TYPE_3D,
    }
}

fn map_view_dimension(dimension: crate::ViewDimension) -> vk::ImageViewType {
    use crate::ViewDimension as Vd;
    match dimension {
        Vd::D1 => vk::ImageViewType::TYPE_1D,
        Vd::D1Array => vk::ImageViewType::TYPE_1D_ARRAY,
        Vd::D2 => vk::ImageViewType::TYPE_2D,
        Vd::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
        Vd::Cube => vk::ImageViewType::CUBE,
        Vd::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        Vd::D3 => vk::ImageViewType::TYPE_3D,
    }
}

pub(super) fn map_texture_usage(
    usage: crate::TextureUsage,
    aspects: crate::TexelAspects,
) -> vk::ImageUsageFlags {
    use vk::ImageUsageFlags as Iuf;

    let mut flags = Iuf::empty();
    if usage.contains(crate::TextureUsage::COPY) {
        flags |= Iuf::TRANSFER_SRC | Iuf::TRANSFER_DST;
    }
    if usage.contains(crate::TextureUsage::RESOURCE) {
        flags |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.contains(crate::TextureUsage::TARGET) {
        flags |= if aspects.contains(crate::TexelAspects::COLOR) {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        } else {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        };
    }
    if usage.intersects(crate::TextureUsage::STORAGE) {
        flags |= vk::ImageUsageFlags::STORAGE;
    }
    flags
}

fn map_filter_mode(mode: crate::FilterMode) -> vk::Filter {
    match mode {
        crate::FilterMode::Nearest => vk::Filter::NEAREST,
        crate::FilterMode::Linear => vk::Filter::LINEAR,
    }
}

fn map_mip_filter_mode(mode: crate::FilterMode) -> vk::SamplerMipmapMode {
    match mode {
        crate::FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        crate::FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

fn map_address_mode(mode: crate::AddressMode) -> vk::SamplerAddressMode {
    match mode {
        crate::AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        crate::AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        crate::AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        crate::AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        // wgt::AddressMode::MirrorClamp => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
    }
}

fn map_border_color(border_color: crate::TextureColor) -> vk::BorderColor {
    match border_color {
        crate::TextureColor::TransparentBlack => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        crate::TextureColor::OpaqueBlack => vk::BorderColor::FLOAT_OPAQUE_BLACK,
        crate::TextureColor::White => vk::BorderColor::FLOAT_OPAQUE_WHITE,
    }
}
