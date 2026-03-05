struct Params {
    mvp: mat4x4<f32>,
    tint: vec4<f32>,
};
var<storage, read> globals: Params;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    var positions = array<vec3<f32>, 12>(
        vec3<f32>(0.0, 0.6, 0.0),
        vec3<f32>(0.5, -0.3, 0.4),
        vec3<f32>(-0.5, -0.3, 0.4),
        vec3<f32>(0.0, 0.6, 0.0),
        vec3<f32>(-0.5, -0.3, 0.4),
        vec3<f32>(0.0, -0.3, -0.5),
        vec3<f32>(0.0, 0.6, 0.0),
        vec3<f32>(0.0, -0.3, -0.5),
        vec3<f32>(0.5, -0.3, 0.4),
        vec3<f32>(-0.5, -0.3, 0.4),
        vec3<f32>(0.5, -0.3, 0.4),
        vec3<f32>(0.0, -0.3, -0.5),
    );
    var colors = array<vec4<f32>, 12>(
        vec4<f32>(1.0, 0.2, 0.2, 1.0),
        vec4<f32>(0.2, 1.0, 0.2, 1.0),
        vec4<f32>(0.2, 0.2, 1.0, 1.0),
        vec4<f32>(1.0, 0.2, 0.2, 1.0),
        vec4<f32>(0.2, 0.2, 1.0, 1.0),
        vec4<f32>(1.0, 1.0, 0.2, 1.0),
        vec4<f32>(1.0, 0.2, 0.2, 1.0),
        vec4<f32>(1.0, 1.0, 0.2, 1.0),
        vec4<f32>(0.2, 1.0, 0.2, 1.0),
        vec4<f32>(0.2, 0.2, 1.0, 1.0),
        vec4<f32>(0.2, 1.0, 0.2, 1.0),
        vec4<f32>(1.0, 1.0, 0.2, 1.0),
    );

    var out: VsOut;
    out.position = globals.mvp * vec4<f32>(positions[vertex_index], 1.0);
    out.color = colors[vertex_index];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color * globals.tint;
}
