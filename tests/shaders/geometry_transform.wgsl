enable wgpu_ray_query;

var acc_struct: acceleration_structure;
var<storage, read> rays: array<vec4<f32>>;
var<storage, read_write> hits: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var query: ray_query;
    rayQueryInitialize(&query, acc_struct, RayDesc(
        RAY_FLAG_NONE, 0xFFu, 0.01, 10.0,
        rays[2u * id.x].xyz, rays[2u * id.x + 1u].xyz,
    ));
    rayQueryProceed(&query);
    let hit = rayQueryGetCommittedIntersection(&query);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        hits[id.x] = vec4<f32>(-1.0);
    } else {
        hits[id.x] = vec4<f32>(hit.t, hit.barycentrics, f32(hit.geometry_index));
    }
}
