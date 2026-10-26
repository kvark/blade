var t_depth: texture_2d<f32>;
var t_motion: texture_2d<f32>;
var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(t_depth);
    if (any(id.xy >= size)) { return; }
    let motion = textureLoad(t_motion, id.xy, 0);
    output[id.y * size.x + id.x] = vec4<f32>(motion.xy / MOTION_SCALE, motion.z, textureLoad(t_depth, id.xy, 0).x);
}
