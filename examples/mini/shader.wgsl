var input: texture_2d<f32>;
var output: texture_storage_2d<rgba8unorm, write>;

var<uniform> modulator: vec4<f32>;

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let src_dim = vec2<u32>(textureDimensions(input).xy);
    if (any(global_id.xy * 2u >= src_dim)) {
        return;
    }
    let tc = vec2<i32>(global_id.xy);
    var sum = textureLoad(input, tc * 2, 0);
    var count = 1.0;
    if (global_id.x * 2u + 1u < src_dim.x) {
        sum += textureLoad(input, tc * 2 + vec2<i32>(1, 0), 0);
        count += 1.0;
    }
    if (global_id.y * 2u + 1u < src_dim.y) {
        sum += textureLoad(input, tc * 2 + vec2<i32>(0, 1), 0);
        count += 1.0;
    }
    if (all(global_id.xy * 2u + 1u < src_dim)) {
        sum += textureLoad(input, tc * 2 + vec2<i32>(1, 1), 0);
        count += 1.0;
    }
    textureStore(output, tc, sum * modulator / count);
}
