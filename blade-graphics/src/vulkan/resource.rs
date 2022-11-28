use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;

struct Allocation {
    memory: vk::DeviceMemory,
    offset: u64,
    handle: usize,
}

impl super::Context {
    fn allocate_memory(
        &self,
        requirements: vk::MemoryRequirements,
        memory: crate::Memory,
    ) -> Allocation {
        let mut manager = self.memory.lock().unwrap();
        let alloc_usage = match memory {
            crate::Memory::Device => gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            crate::Memory::Shared => {
                gpu_alloc::UsageFlags::HOST_ACCESS
                    | gpu_alloc::UsageFlags::DOWNLOAD
                    | gpu_alloc::UsageFlags::UPLOAD
                    | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
            }
            crate::Memory::Upload => {
                gpu_alloc::UsageFlags::HOST_ACCESS | gpu_alloc::UsageFlags::UPLOAD
            }
        };
        let memory_types = requirements.memory_type_bits & manager.valid_ash_memory_types;
        let block = unsafe {
            manager
                .allocator
                .alloc(
                    AshMemoryDevice::wrap(&self.device),
                    gpu_alloc::Request {
                        size: requirements.size,
                        align_mask: requirements.alignment - 1,
                        usage: alloc_usage,
                        memory_types,
                    },
                )
                .unwrap()
        };
        Allocation {
            memory: *block.memory(),
            offset: block.offset(),
            handle: manager.slab.insert(block),
        }
    }

    fn free_memory(&self, handle: usize) {
        let mut manager = self.memory.lock().unwrap();
        let block = manager.slab.remove(handle);
        unsafe {
            manager
                .allocator
                .dealloc(AshMemoryDevice::wrap(&self.device), block);
        }
    }

    pub fn create_buffer(&self, desc: crate::BufferDesc) -> super::Buffer {
        use vk::BufferUsageFlags as Buf;
        let vk_info = vk::BufferCreateInfo::builder()
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

        let raw = unsafe { self.device.create_buffer(&vk_info, None).unwrap() };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(raw) };
        let allocation = self.allocate_memory(requirements, desc.memory);

        unsafe {
            self.device
                .bind_buffer_memory(raw, allocation.memory, allocation.offset)
                .unwrap();
            if !desc.name.is_empty() {
                self.set_object_name(vk::ObjectType::BUFFER, raw, desc.name);
            }
        }

        super::Buffer {
            raw,
            memory_handle: allocation.handle,
        }
    }

    pub fn destroy_buffer(&self, buffer: super::Buffer) {
        unsafe { self.device.destroy_buffer(buffer.raw, None) };
        self.free_memory(buffer.memory_handle);
    }

    pub fn create_texture(&self, _desc: crate::TextureDesc) -> super::Texture {
        unimplemented!()
    }

    pub fn create_texture_view(&self, _desc: crate::TextureViewDesc) -> super::TextureView {
        unimplemented!()
    }

    pub fn create_sampler(&self, _desc: crate::SamplerDesc) -> super::Sampler {
        unimplemented!()
    }
}
