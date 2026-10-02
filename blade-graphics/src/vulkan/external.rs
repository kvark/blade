//! External queue-family ownership transfers, separate from memory import.

use ash::vk;

impl super::CommandEncoder {
    /// Acquire the whole buffer from an external producer.
    /// The producer must release the whole buffer to QUEUE_FAMILY_EXTERNAL,
    /// and its release is complete before this encoder is submitted. Keep the
    /// producer from reusing it until a matching release below completes.
    pub fn acquire_external_buffer(&mut self, buffer: super::Buffer) {
        self.external_buffer_barrier(buffer, true);
    }

    /// Release the whole buffer to an external consumer.
    /// This queue must own the buffer. Its consumer may start only after
    /// completion of this release (a semaphore or an externally relayed fence).
    pub fn release_external_buffer(&mut self, buffer: super::Buffer) {
        self.external_buffer_barrier(buffer, false);
    }

    fn external_buffer_barrier(&mut self, buffer: super::Buffer, acquire: bool) {
        let access = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(buffer.raw)
            .size(vk::WHOLE_SIZE)
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
                self.buffers[0].raw,
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
