use super::camera::CameraParams;
use super::debug::{DebugLine, debug_buf};
use super::quaternion::{qinv, qrot};
use synaga_shader::*;

#[derive(Clone, Copy, Debug, Default, Io)]
struct DebugVarying {
    #[builtin(position)]
    pos: Vec4,
    #[location(0)]
    color: Vec4,
    #[location(1)]
    dir: Vec3,
}

static camera: Uniform<CameraParams> = binding();
static debug_lines: Storage<[DebugLine]> = binding();
static depth: Texture2D<f32> = binding();

/// The draw entry points only see the line array. This entry point is what
/// keeps `DebugBuffer` in the module, which is the layout the host uses to
/// size that buffer.
#[entry_point(compute, threads(1))]
fn debug_buffer_layout() {
    debug_buf.get_mut().open = debug_buf.open;
}

#[entry_point(vertex)]
fn debug_vs(vertex_index: u32, instance_index: u32) -> DebugVarying {
    let line = debug_lines[instance_index as usize];
    let mut point = line.a;
    if vertex_index != 0 {
        point = line.b;
    }

    let world_dir = point.pos - camera.position;
    let local_dir = qrot(qinv(camera.orientation), world_dir);
    let ndc = local_dir.xy() / (0.5 * camera.fov).tan();

    DebugVarying {
        pos: ndc.extend(0.0).extend(-local_dir.z),
        color: unpack4x8unorm(point.color),
        dir: world_dir,
    }
}

#[entry_point(fragment)]
fn debug_fs(input: DebugVarying) -> Vec4 {
    let geo_dim = depth.dimensions();
    let depth_itc = vec2(input.pos.x as i32, geo_dim.y as i32 - input.pos.y as i32);
    let stored = depth.load(depth_itc, 0).x;
    let alpha = select(
        0.8,
        0.2,
        stored != 0.0 && input.dir.dot(input.dir) > stored * stored,
    );
    input.color.xyz().extend(alpha)
}
