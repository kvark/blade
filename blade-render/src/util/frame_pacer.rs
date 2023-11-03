use crate::render::FrameResources;
use std::mem;

/// Utility object that encapsulates the logic
/// of always rendering 1 frame at a time, and
/// cleaning up the temporary resources.
pub struct FramePacer {
    frame_index: usize,
    prev_resources: FrameResources,
    prev_sync_point: Option<blade_graphics::SyncPoint>,
    command_encoder: Option<blade_graphics::CommandEncoder>,
    next_resources: FrameResources,
}

impl FramePacer {
    pub fn new(context: &blade_graphics::Context) -> Self {
        let encoder = context.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        Self {
            frame_index: 0,
            prev_resources: FrameResources::default(),
            prev_sync_point: None,
            command_encoder: Some(encoder),
            next_resources: FrameResources::default(),
        }
    }

    #[profiling::function]
    pub fn wait_for_previous_frame(&mut self, context: &blade_graphics::Context) {
        if let Some(sp) = self.prev_sync_point.take() {
            context.wait_for(&sp, !0);
        }
        for buffer in self.prev_resources.buffers.drain(..) {
            context.destroy_buffer(buffer);
        }
        for accel_structure in self.prev_resources.acceleration_structures.drain(..) {
            context.destroy_acceleration_structure(accel_structure);
        }
    }

    pub fn last_sync_point(&self) -> Option<&blade_graphics::SyncPoint> {
        self.prev_sync_point.as_ref()
    }

    pub fn destroy(&mut self, context: &blade_graphics::Context) {
        self.wait_for_previous_frame(context);
        context.destroy_command_encoder(self.command_encoder.take().unwrap());
    }

    pub fn begin_frame(&mut self) -> (&mut blade_graphics::CommandEncoder, &mut FrameResources) {
        let encoder = self.command_encoder.as_mut().unwrap();
        encoder.start();
        (encoder, &mut self.next_resources)
    }

    pub fn end_frame(&mut self, context: &blade_graphics::Context) -> &blade_graphics::SyncPoint {
        let sync_point = context.submit(self.command_encoder.as_mut().unwrap());
        self.frame_index += 1;
        // Wait for the previous frame immediately - this ensures that we are
        // only processing one frame at a time, and yet not stalling.
        self.wait_for_previous_frame(context);
        self.prev_sync_point = Some(sync_point);
        mem::swap(&mut self.prev_resources, &mut self.next_resources);
        self.prev_sync_point.as_ref().unwrap()
    }
}
