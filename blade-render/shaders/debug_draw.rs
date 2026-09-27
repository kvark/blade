use super::camera::*;
use super::debug::*;
use super::quaternion::*;
use synaga_shader::*;

#[derive(Clone, Copy, Debug, Default, Io)]
pub struct DebugVarying {
    #[builtin(position)]
    pos: vec4,
    #[location(0)]
    color: vec4,
    #[location(1)]
    dir: vec3,
}

pub static camera: Uniform<CameraParams> = binding();

pub static debug_lines: Storage<[DebugLine]> = binding();

pub static depth: texture_2d<f32> = binding();

/// The draw entry points only see the line array. This entry point is what
/// keeps `DebugBuffer` in the module, which is the layout the host uses to
/// size that buffer.
#[entry_point(compute, threads(1))]
pub fn debug_buffer_layout() {
    unsafe {
        debug_buf.get_mut().open = debug_buf.open;
    }
}

#[entry_point(vertex)]
pub fn debug_vs(
    #[builtin(vertex_index)] vertex_id: u32,
    #[builtin(instance_index)] instance_id: u32,
) -> DebugVarying {
    let line = debug_lines[(instance_id) as usize];
    let mut point = line.a;
    if (vertex_id != 0u32) {
        point = line.b;
    }

    let world_dir = point.pos - camera.position;
    let local_dir = qrot(qinv(camera.orientation), world_dir);
    let ndc = local_dir.xy() / tan(0.5 * camera.fov);

    let mut out = DebugVarying::default();
    out.pos = ((ndc).extend(0.0)).extend(-local_dir.z);
    out.color = unpack4x8unorm(point.color);
    out.dir = world_dir;
    return out;
}

#[entry_point(fragment)]
#[output(location(0))]
pub fn debug_fs(input: DebugVarying) -> vec4 {
    let geo_dim = depth.dimensions();
    let depth_itc = vec2i(
        (input.pos.x) as i32,
        (geo_dim.y) as i32 - (input.pos.y) as i32,
    );
    let stored = depth.load(depth_itc, 0).x;
    let alpha = select(
        0.8,
        0.2,
        stored != 0.0 && dot(input.dir, input.dir) > stored * stored,
    );
    return (input.color.xyz()).extend(alpha);
}
