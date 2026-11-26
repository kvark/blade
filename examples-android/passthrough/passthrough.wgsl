// The scene: a spinning pyramid, with an opaque alpha, over passthrough.

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
    var colors = array<vec4<f32>, 4>(
        vec4<f32>(1.0, 0.8, 0.2, 1.0),
        vec4<f32>(0.2, 0.8, 1.0, 1.0),
        vec4<f32>(0.6, 1.0, 0.3, 1.0),
        vec4<f32>(1.0, 1.0, 1.0, 1.0),
    );

    var out: VsOut;
    out.position = globals.mvp * vec4<f32>(positions[vertex_index], 1.0);
    out.color = colors[vertex_index / 3u];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color * globals.tint;
}

// The HUD: a translucent panel with a bar that moves with the time, and
// the depth of the environment of the left eye, when there is one, which
// the compositor draws on a quad in front of the head.

struct HudParams {
    time: f32,
    has_depth: u32,
    pad: vec2<f32>,
};
var<uniform> hud: HudParams;
var depth_tex: texture_depth_2d_array;

struct HudOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn hud_vs(@builtin(vertex_index) vertex_index: u32) -> HudOut {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var out: HudOut;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

// Distance outside of a rectangle of half size `half` with rounded corners.
fn rounded_box(p: vec2<f32>, half: vec2<f32>, radius: f32) -> f32 {
    let q = abs(p) - half + radius;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn hud_fs(in: HudOut) -> @location(0) vec4<f32> {
    let p = in.uv * 2.0 - 1.0;
    let panel = rounded_box(p, vec2<f32>(0.98, 0.95), 0.15);
    if (panel > 0.0) {
        return vec4<f32>(0.0);
    }
    var color = vec3<f32>(0.05, 0.05, 0.08);
    var alpha = 0.7;
    // A bar along the bottom, filled by the time.
    let fill = 0.5 + 0.5 * sin(hud.time);
    if (in.uv.y > 0.8 && in.uv.y < 0.88 && in.uv.x > 0.06 && in.uv.x < 0.94) {
        let t = (in.uv.x - 0.06) / 0.88;
        color = select(vec3<f32>(0.2), vec3<f32>(1.0, 0.3, 0.2), t < fill);
        alpha = 0.9;
    }
    // The depth of the environment in a window on the right, brighter
    // where it is nearer.
    let window = vec4<f32>(0.5, 0.08, 0.94, 0.72);
    if (in.uv.x > window.x && in.uv.x < window.z && in.uv.y > window.y && in.uv.y < window.w) {
        if (hud.has_depth != 0u) {
            let size = vec2<f32>(textureDimensions(depth_tex));
            let local = (in.uv - window.xy) / (window.zw - window.xy);
            let texel = vec2<i32>(local * (size - 1.0));
            let depth = textureLoad(depth_tex, texel, 0, 0);
            color = vec3<f32>(clamp((1.0 - depth) * 20.0, 0.0, 1.0));
        } else {
            color = vec3<f32>(0.3, 0.1, 0.1);
        }
        alpha = 1.0;
    }
    // Premultiplied.
    return vec4<f32>(color * alpha, alpha);
}
