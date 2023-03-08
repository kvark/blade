const MAX_BOUNCES: i32 = 3;

struct Parameters {
    cam_position: vec3<f32>,
    depth: f32,
    cam_orientation: vec4<f32>,
    fov: vec2<f32>,
};

var<uniform> parameters: Parameters;
var acc_struct: acceleration_structure;
var output: texture_storage_2d<rgba8unorm, write>;

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_size = textureDimensions(output);
    let half_size = vec2<f32>(target_size >> vec2<u32>(1u));
    let ndc = vec2<f32>(1.0, -1.0) * (vec2<f32>(global_id.xy) - half_size) / half_size;
    if (any(global_id.xy > target_size)) {
        return;
    }

    let local_dir = vec3<f32>(ndc * tan(parameters.fov), 1.0);
    let world_dir = normalize(qrot(parameters.cam_orientation, local_dir));

    var rq: ray_query;
    var ray_pos = parameters.cam_position;
    var ray_dir = world_dir;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, parameters.depth, ray_pos, ray_dir));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);

    var color = vec4<f32>(0.0);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        color = vec4<f32>(fract(intersection.t));
    }
    textureStore(output, global_id.xy, color);
}

@vertex
fn draw_vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(vi & 1u) * 4.0 - 1.0, f32(vi & 2u) * 2.0 - 1.0, 0.0, 1.0);
}

var input: texture_2d<f32>;

@fragment
fn draw_fs(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(input, vec2<i32>(frag_coord.xy), 0);
}
