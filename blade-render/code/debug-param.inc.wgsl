#use DebugMode
#use DebugDrawFlags
#use DebugTextureFlags

struct DebugParams {
    view_mode: u32,
    pass_index: u32,
    draw_flags: u32,
    texture_flags: u32,
    mouse_pos: vec2<u32>,
};
