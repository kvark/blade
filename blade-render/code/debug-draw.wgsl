#include "quaternion.inc.wgsl"
#include "debug.inc.wgsl"
#include "camera.inc.wgsl"

var<uniform> camera: CameraParams;

struct DebugVarying {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) dir: vec3<f32>,
}

@vertex
fn debug_vs(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> DebugVarying {
    let line = debug_buf.lines[instance_id];
    var point = line.a;
    if (vertex_id != 0u) {
        point = line.b;
    }

    let world_dir = point.pos - camera.position;
    let local_dir = qrot(qinv(camera.orientation), world_dir);
    let ndc = local_dir.xy / tan(0.5 * camera.fov);

    var out: DebugVarying;
    out.pos = vec4<f32>(ndc, 0.0, -local_dir.z);
    out.color = unpack4x8unorm(point.color);
    out.dir = world_dir;
    return out;
}

var depth: texture_2d<f32>;

@fragment
fn debug_fs(in: DebugVarying) -> @location(0) vec4<f32> {
    let geo_dim = textureDimensions(depth);
    let depth_itc = vec2<i32>(i32(in.pos.x), i32(geo_dim.y) - i32(in.pos.y));
    let depth = textureLoad(depth, depth_itc, 0).x;
    let alpha = select(0.8, 0.2, depth != 0.0 && dot(in.dir, in.dir) > depth*depth);
    return vec4<f32>(in.color.xyz, alpha);
}
