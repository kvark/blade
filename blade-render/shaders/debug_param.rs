use super::config::{DebugDrawFlags, DebugMode, DebugTextureFlags};
use synaga_shader::*;

/// `no_uninit`, since not every `u32` is a `DebugMode`: the host uploads
/// these, but can't read them back from bytes.
#[repr(C)]
#[derive(Shared)]
#[shared(no_uninit)]
pub struct DebugParams {
    pub view_mode: DebugMode,
    pub draw_flags: DebugDrawFlags,
    pub texture_flags: DebugTextureFlags,
    pub _pad: u32,
    pub mouse_pos: Vec2<u32>,
}
