const MAX_BOUNCES: i32 = 3;

struct Parameters {
    cam_position: vec3<f32>,
    depth: f32,
    cam_orientation: vec4<f32>,
    fov: vec2<f32>,
    torus_radius: f32,
    rotation_angle: f32,
};

var<uniform> parameters: Parameters;
var acc_struct: acceleration_structure;

fn qmake(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    return vec4<f32>(axis * sin(angle), cos(angle));
}
fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}

fn get_miss_color(dir: vec3<f32>) -> vec4<f32> {
    var colors = array<vec4<f32>, 4>(vec4<f32>(1.0), vec4<f32>(0.6, 0.9, 0.3, 1.0), vec4<f32>(0.3, 0.6, 0.9, 1.0), vec4<f32>(0.0));
    var thresholds = array<f32, 4>(-0.8, -0.3, 0.4, 1.0);
    var i = 0;
    loop {
        if (dir.y <= thresholds[i]) {
            let t = (dir.y - thresholds[i - 1]) / (thresholds[i] - thresholds[i - 1]);
            return mix(colors[i - 1], colors[i], t);
        }
        i += 1;
        if (i >= 4) {
            break;
        }
    }
    return vec4<f32>(1.0);
}

fn get_torus_normal(world_point: vec3<f32>, intersection: RayIntersection) -> vec3<f32> {
    //Note: generally we'd store normals with the mesh data, but for the sake of
    // simplicity of this example it's computed analytically instead.
    let local_point = intersection.world_to_object * vec4<f32>(world_point, 1.0);
    let point_on_guiding_line = normalize(local_point.xy) * parameters.torus_radius;
    let world_point_on_guiding_line = intersection.object_to_world * vec4<f32>(point_on_guiding_line, 0.0, 1.0);
    return normalize(world_point - world_point_on_guiding_line);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) out_pos: vec2<f32>,
}

@vertex
fn draw_vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo: VertexOutput;
    vo.clip_pos = vec4<f32>(f32(vi & 1u) * 4.0 - 1.0, f32(vi & 2u) * 2.0 - 1.0, 0.0, 1.0);
    vo.out_pos = vo.clip_pos.xy;
    return vo;
}

@fragment
fn draw_fs(vi: VertexOutput) -> @location(0) vec4<f32> {
    let local_dir = vec3<f32>(vi.out_pos * tan(parameters.fov), 1.0);
    let world_dir = normalize(qrot(parameters.cam_orientation, local_dir));
    let rotator = qmake(vec3<f32>(0.0, 1.0, 0.0), parameters.rotation_angle);

    var num_bounces = 0;
    var rq: ray_query;
    var ray_pos = qrot(rotator, parameters.cam_position);
    var ray_dir = qrot(rotator, world_dir);
    loop {
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, parameters.depth, ray_pos, ray_dir));
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
        if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
            break;
        }

        ray_pos += ray_dir * intersection.t;
        let normal = get_torus_normal(ray_pos, intersection);
        ray_dir -= 2.0 * dot(ray_dir, normal) * normal;

        num_bounces += 1;
        if (num_bounces > MAX_BOUNCES) {
            break;
        }
    }
    
    return get_miss_color(ray_dir);
}
