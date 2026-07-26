// Sample the ray tracer's G-buffer views and pack what a consumer would want
// into one float target, so a test can read it back in a single copy.
//
// Binding the views is the point: it checks they are usable as sampled
// textures from outside the renderer, not merely that they are non-null.

var t_depth: texture_2d<f32>;
var t_basis: texture_2d<f32>;
var t_flat_normal: texture_2d<f32>;
var t_diffuse_albedo: texture_2d<f32>;
var t_specular_f0: texture_2d<f32>;
var output: texture_storage_2d<rgba32float, write>;

// Matches `qrot` in the renderer's quaternion.inc.wgsl.
fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

@compute @workgroup_size(8, 8, 1)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(t_depth, 0);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let texel = vec2<i32>(i32(id.x), i32(id.y));

    let depth = textureLoad(t_depth, texel, 0).x;
    let basis = textureLoad(t_basis, texel, 0);
    let flat_normal = textureLoad(t_flat_normal, texel, 0).xyz;
    let albedo = textureLoad(t_diffuse_albedo, texel, 0).xyz;
    let specular = textureLoad(t_specular_f0, texel, 0);

    // The shading normal is the tangent frame applied to +Z.
    let shading_normal = qrot(basis, vec3<f32>(0.0, 0.0, 1.0));

    textureStore(output, texel, vec4<f32>(
        depth,
        specular.w,
        albedo.x,
        // How far the shading normal and the geometric one agree. Both should
        // be unit length, so this lands in [-1, 1] and is near 1 on the flat
        // parts of the scene.
        dot(shading_normal, flat_normal),
    ));
}
