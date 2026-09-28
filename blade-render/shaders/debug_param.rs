use synaga_shader::*;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct DebugParams {
    pub view_mode: u32,
    pub draw_flags: u32,
    pub texture_flags: u32,
    pub _pad: u32,
    pub mouse_pos: Vec2<u32>,
}
