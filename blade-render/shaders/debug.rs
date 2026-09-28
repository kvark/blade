use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct DebugPoint {
    pub pos: vec3,
    pub color: u32,
}

#[derive(Clone, Copy, Default)]
pub struct DebugLine {
    pub a: DebugPoint,
    pub b: DebugPoint,
}

#[derive(Clone, Copy, Default)]
pub struct DebugVariance {
    pub color_sum: vec3,
    pub color2_sum: vec3,
    pub count: u32,
}

#[derive(Clone, Copy, Default)]
pub struct DebugEntry {
    pub custom_index: u32,
    pub depth: f32,
    pub tex_coords: vec2,
    pub base_color_texture: u32,
    pub normal_texture: u32,
    pub pad: vec2u,
    pub position: vec3,
    pub flat_normal: vec3,
}

pub struct DebugBuffer {
    pub vertex_count: u32,
    pub instance_count: AtomicU32,
    pub first_vertex: u32,
    pub first_instance: u32,
    pub capacity: u32,
    pub open: u32,
    pub variance: DebugVariance,
    pub entry: DebugEntry,
    pub lines: [DebugLine],
}

pub static debug_buf: StorageMut<DebugBuffer> = binding();

pub fn debug_line(a: vec3, b: vec3, color: u32) {
    if debug_buf.open != 0u32 {
        let index = debug_buf.instance_count.fetch_add(1u32);
        if index < debug_buf.capacity {
            unsafe {
                debug_buf.get_mut().lines[(index) as usize] = DebugLine {
                    a: DebugPoint {
                        pos: a,
                        color: color,
                    },
                    b: DebugPoint {
                        pos: b,
                        color: color,
                    },
                };
            }
        } else {
            // ensure the final value is never above the capacity
            debug_buf.instance_count.fetch_sub(1u32);
        }
    }
}
