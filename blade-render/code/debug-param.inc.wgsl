struct DebugParams {
    view_mode: u32,
    flags: u32,
    mouse_pos: vec2<u32>,
};
var<uniform> debug: DebugParams;

//Must match host side `DebugMode`
const DEBUG_MODE_NONE: u32 = 0u;
const DEBUG_MODE_DEPTH: u32 = 1u;
const DEBUG_MODE_NORMAL: u32 = 2u;

//Must match host side `DebugFlags`
const DEBUG_FLAGS_GEOMETRY: u32 = 1u;
const DEBUG_FLAGS_RESTIR: u32 = 2u;
