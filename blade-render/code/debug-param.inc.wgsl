#use DebugMode
#use DebugDrawFlags
#use DebugTextureFlags

struct DebugParams {
    view_mode: u32,
    draw_flags: u32,
    texture_flags: u32,
    pad: u32,
    mouse_pos: vec2<u32>,
};
var<uniform> debug: DebugParams;
