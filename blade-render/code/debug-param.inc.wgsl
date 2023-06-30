#use DebugMode
#use DebugFlags

struct DebugParams {
    view_mode: u32,
    flags: u32,
    mouse_pos: vec2<u32>,
};
var<uniform> debug: DebugParams;
