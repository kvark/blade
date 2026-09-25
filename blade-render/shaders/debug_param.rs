use super::config::*;
use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct DebugParams {
    pub view_mode: u32,
    pub draw_flags: u32,
    pub texture_flags: u32,
    pub pad: u32,
    pub mouse_pos: vec2u,
}
