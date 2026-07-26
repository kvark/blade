// Pack the ray tracer's G-buffer into the ground truth a relighting
// reconstruction is scored against.
//
// Two targets, because what a consumer wants does not fit in four channels:
// the material it has to recover, and the geometry it has to agree with.

var t_depth: texture_2d<f32>;
var t_basis: texture_2d<f32>;
var t_flat_normal: texture_2d<f32>;
var t_diffuse_albedo: texture_2d<f32>;
var t_specular_f0: texture_2d<f32>;
// Diffuse albedo in RGB, roughness in alpha.
var out_material: texture_storage_2d<rgba32float, write>;
// Shading normal in XYZ, ray distance in alpha. Negative distance means the
// ray left the scene, so a consumer can build a coverage mask from it without
// a separate channel.
var out_geometry: texture_storage_2d<rgba32float, write>;

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
    let albedo = textureLoad(t_diffuse_albedo, texel, 0).xyz;
    let specular = textureLoad(t_specular_f0, texel, 0);

    // The tangent frame is a quaternion; the shading normal is it applied
    // to +Z. On a miss the frame is not meaningful, so fall back to the flat
    // normal rather than storing a rotated axis that means nothing.
    let hit = depth > 0.0;
    let shading_normal = select(
        textureLoad(t_flat_normal, texel, 0).xyz,
        qrot(basis, vec3<f32>(0.0, 0.0, 1.0)),
        hit,
    );

    textureStore(out_material, texel, vec4<f32>(albedo, specular.w));
    textureStore(out_geometry, texel, vec4<f32>(
        shading_normal,
        select(-1.0, depth, hit),
    ));
}
