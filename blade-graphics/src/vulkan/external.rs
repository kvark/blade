//! Explicit Vulkan-to-Vulkan OPAQUE_FD imports and external ownership transfers.

use ash::vk;
use std::{
    os::fd::{AsRawFd, IntoRawFd, OwnedFd},
    ptr,
};

/// Allocation facts supplied by the exporting Vulkan device, not inferred from
/// the consumer's buffer requirements. Dedicated allocations are not supported.
#[derive(Clone, Copy, Debug)]
pub struct VulkanBufferImport {
    pub buffer_size: u64,
    pub allocation_size: u64,
    pub memory_offset: u64,
    pub memory_type_index: u32,
    pub device_uuid: [u8; 16],
    pub driver_uuid: [u8; 16],
}

impl super::Context {
    pub fn vulkan_device_uuids(&self) -> ([u8; 16], [u8; 16]) {
        let mut ids = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut ids);
        unsafe {
            self.inner
                .instance
                .get_physical_device_properties2
                .get_physical_device_properties2(self.physical_device, &mut properties);
        }
        (ids.device_uuid, ids.driver_uuid)
    }

    /// # Safety
    /// The descriptor must describe the non-dedicated Vulkan allocation exported
    /// by `fd`, with the same physical device/driver. No work may access imported
    /// bytes without an external ownership transfer and producer/consumer sync.
    /// Ownership of `fd` is consumed on success and closed on failure.
    pub unsafe fn import_vulkan_buffer_fd(
        &self,
        name: &str,
        info: VulkanBufferImport,
        fd: OwnedFd,
    ) -> Result<super::Buffer, String> {
        if self.device.external_memory.is_none() {
            return Err("external memory FD is unavailable".into());
        }
        if (info.device_uuid, info.driver_uuid) != self.vulkan_device_uuids() {
            return Err("external allocation belongs to a different device or driver".into());
        }
        if info.buffer_size == 0 || info.memory_type_index >= 32 {
            return Err("invalid external buffer size or memory type".into());
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let mut capabilities = vk::ExternalBufferProperties::default();
        unsafe {
            self.inner
                .instance
                .core
                .get_physical_device_external_buffer_properties(
                    self.physical_device,
                    &vk::PhysicalDeviceExternalBufferInfo::default()
                        .usage(usage)
                        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
                    &mut capabilities,
                );
        }
        let features = capabilities
            .external_memory_properties
            .external_memory_features;
        if !features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
            || features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
        {
            return Err("non-dedicated external storage buffers are unavailable".into());
        }
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let create = vk::BufferCreateInfo::default()
            .size(info.buffer_size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        let raw =
            unsafe { self.device.core.create_buffer(&create, None) }.map_err(|e| e.to_string())?;
        let requirements = unsafe { self.device.core.get_buffer_memory_requirements(raw) };
        if !valid_import_extent(info, requirements) {
            unsafe {
                self.device.core.destroy_buffer(raw, None);
            }
            return Err("external allocation does not satisfy the buffer requirements".into());
        }
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
            .fd(fd.as_raw_fd());
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(info.allocation_size)
            .memory_type_index(info.memory_type_index)
            .push_next(&mut import);
        let memory = match unsafe { self.device.core.allocate_memory(&allocate, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe {
                    self.device.core.destroy_buffer(raw, None);
                }
                return Err(error.to_string());
            }
        };
        let _ = fd.into_raw_fd(); // vkAllocateMemory now owns it.
        if let Err(error) = unsafe {
            self.device
                .core
                .bind_buffer_memory(raw, memory, info.memory_offset)
        } {
            unsafe {
                self.device.core.destroy_buffer(raw, None);
                self.device.core.free_memory(memory, None);
            }
            return Err(error.to_string());
        }
        let properties = unsafe {
            self.inner
                .instance
                .core
                .get_physical_device_memory_properties(self.physical_device)
        };
        let mut manager = self.memory.lock().unwrap();
        let block = unsafe {
            manager.allocator.import_memory(
                memory,
                info.memory_type_index,
                gpu_alloc_ash::memory_properties_from_ash(
                    properties.memory_types[info.memory_type_index as usize].property_flags,
                ),
                0,
                info.allocation_size,
            )
        };
        let handle = manager.slab.insert((block, name.into()));
        Ok(super::Buffer {
            raw,
            memory_handle: handle,
            mapped_data: ptr::null_mut(),
            size: info.buffer_size,
            external: None,
        })
    }

    /// Metadata for a non-dedicated buffer exported with Memory::External(Fd(None)).
    pub fn exported_vulkan_buffer_info(&self, buffer: super::Buffer) -> VulkanBufferImport {
        assert!(matches!(
            buffer.external,
            Some(crate::ExternalMemorySource::Fd(_))
        ));
        let manager = self.memory.lock().unwrap();
        let block = &manager.slab[buffer.memory_handle].0;
        let (device_uuid, driver_uuid) = self.vulkan_device_uuids();
        VulkanBufferImport {
            buffer_size: buffer.size,
            allocation_size: block.size(),
            memory_offset: block.offset(),
            memory_type_index: block.memory_type(),
            device_uuid,
            driver_uuid,
        }
    }

    /// # Safety
    /// The external producer has released this range to QUEUE_FAMILY_EXTERNAL,
    /// and its release is complete before this encoder is submitted. Keep the
    /// producer from reusing it until a matching release below completes.
    pub unsafe fn acquire_external_buffer(
        &self,
        encoder: &mut super::CommandEncoder,
        piece: crate::BufferPiece,
        size: u64,
    ) {
        unsafe {
            self.external_buffer_barrier(encoder, piece, size, true);
        }
    }

    /// # Safety
    /// This queue owns the range. Its external consumer may start only after
    /// completion of this release (a semaphore or an externally relayed fence).
    pub unsafe fn release_external_buffer(
        &self,
        encoder: &mut super::CommandEncoder,
        piece: crate::BufferPiece,
        size: u64,
    ) {
        unsafe {
            self.external_buffer_barrier(encoder, piece, size, false);
        }
    }

    unsafe fn external_buffer_barrier(
        &self,
        encoder: &mut super::CommandEncoder,
        piece: crate::BufferPiece,
        size: u64,
        acquire: bool,
    ) {
        assert_eq!(self.device.core.handle(), encoder.device.core.handle());
        assert!(
            size > 0
                && piece
                    .offset
                    .checked_add(size)
                    .is_some_and(|end| end <= piece.buffer.size)
        );
        let access = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(piece.buffer.raw)
            .offset(piece.offset)
            .size(size)
            .src_queue_family_index(if acquire {
                vk::QUEUE_FAMILY_EXTERNAL
            } else {
                self.queue_family_index
            })
            .dst_queue_family_index(if acquire {
                self.queue_family_index
            } else {
                vk::QUEUE_FAMILY_EXTERNAL
            })
            .src_access_mask(if acquire {
                vk::AccessFlags::empty()
            } else {
                access
            })
            .dst_access_mask(if acquire {
                access
            } else {
                vk::AccessFlags::empty()
            });
        unsafe {
            self.device.core.cmd_pipeline_barrier(
                encoder.buffers[0].raw,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
    }
}

fn valid_import_extent(info: VulkanBufferImport, requirements: vk::MemoryRequirements) -> bool {
    requirements.memory_type_bits & (1 << info.memory_type_index) != 0
        && info.memory_offset.is_multiple_of(requirements.alignment)
        && info
            .memory_offset
            .checked_add(requirements.size)
            .is_some_and(|end| end <= info.allocation_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_fd_import_checks_type_alignment_and_complete_extent() {
        let info = VulkanBufferImport {
            buffer_size: 128,
            allocation_size: 512,
            memory_offset: 256,
            memory_type_index: 2,
            device_uuid: [0; 16],
            driver_uuid: [0; 16],
        };
        let requirements = vk::MemoryRequirements {
            size: 256,
            alignment: 256,
            memory_type_bits: 4,
        };
        assert!(valid_import_extent(info, requirements));
        assert!(!valid_import_extent(
            VulkanBufferImport {
                memory_type_index: 1,
                ..info
            },
            requirements
        ));
        assert!(!valid_import_extent(
            VulkanBufferImport {
                memory_offset: 4,
                ..info
            },
            requirements
        ));
        assert!(!valid_import_extent(
            VulkanBufferImport {
                allocation_size: 511,
                ..info
            },
            requirements
        ));
        assert!(!valid_import_extent(
            VulkanBufferImport {
                memory_offset: u64::MAX - 255,
                ..info
            },
            requirements
        ));
    }
}
