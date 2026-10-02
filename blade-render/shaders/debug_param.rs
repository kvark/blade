use super::config::{DebugDrawFlags, DebugMode, DebugTextureFlags};
use synaga_shader::*;

/// An enum is not `Pod`, since not every `u32` is a `DebugMode`, but an
/// upload only needs every byte to be initialized.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::NoUninit)]
pub struct DebugParams {
    pub view_mode: DebugMode,
    pub draw_flags: DebugDrawFlags,
    pub texture_flags: DebugTextureFlags,
    pub _pad: u32,
    pub mouse_pos: Vec2<u32>,
}
