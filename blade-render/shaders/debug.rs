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
    pub instance_count: atomic<u32>,
    pub first_vertex: u32,
    pub first_instance: u32,
    pub capacity: u32,
    pub open: u32,
    pub variance: DebugVariance,
    pub entry: DebugEntry,
    pub lines: [DebugLine],
}

pub static mut debug_buf: StorageMut<DebugBuffer> = binding();

#[shader]
pub fn debug_line(a: vec3, b: vec3, color: u32) {
    if (debug_buf.open != 0u32) {
        let index = atomicAdd(debug_buf.instance_count, 1u32);
        if (index < debug_buf.capacity) {
            debug_buf.lines[(index) as usize] = DebugLine {
                a: DebugPoint {
                    pos: a,
                    color: color,
                },
                b: DebugPoint {
                    pos: b,
                    color: color,
                },
            };
        } else {
            // ensure the final value is never above the capacity
            atomicSub(debug_buf.instance_count, 1u32);
        }
    }
}
