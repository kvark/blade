//! External queue-family ownership transfers, separate from memory import.

use ash::vk;

impl super::Context {
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
