use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct DebugPoint {
    pub pos: Vec3,
    pub color: u32,
}

#[derive(Clone, Copy, Default)]
pub struct DebugLine {
    pub a: DebugPoint,
    pub b: DebugPoint,
}

#[derive(Clone, Copy, Default)]
pub struct DebugVariance {
    pub color_sum: Vec3,
    pub color2_sum: Vec3,
    pub count: u32,
}

#[derive(Clone, Copy, Default)]
pub struct DebugEntry {
    pub custom_index: u32,
    pub depth: f32,
    pub tex_coords: Vec2,
    pub base_color_texture: u32,
    pub normal_texture: u32,
    pub _pad: Vec2<u32>,
    pub position: Vec3,
    pub flat_normal: Vec3,
}

/// The first four fields are the arguments of the indirect draw that shows
/// the lines, which the GPU reads rather than a shader.
#[allow(dead_code)]
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

pub fn debug_line(a: Vec3, b: Vec3, color: u32) {
    if debug_buf.open != 0 {
        let index = debug_buf.instance_count.fetch_add(1);
        if index < debug_buf.capacity {
            debug_buf.get_mut().lines[index as usize] = DebugLine {
                a: DebugPoint { pos: a, color },
                b: DebugPoint { pos: b, color },
            };
        } else {
            // ensure the final value is never above the capacity
            debug_buf.instance_count.fetch_sub(1);
        }
    }
}
