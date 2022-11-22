#header

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let src_dim = vec2<u32>(textureDimensions($input).xy);
    if (any(global_id.xy * 2u >= src_dim)) {
        return;
    }
    var sum = textureLoad($input, global_id.xy * 2u, 0);
    var count = 1.0;
    if (global_id.x * 2u + 1u < src_dim.x) {
        sum += textureLoad($input, global_id.xy * 2u + vec2<u32>(1u, 0u), 0);
        count += 1.0;
    }
    if (global_id.y * 2u + 1u < src_dim.y) {
        sum += textureLoad($input, global_id.xy * 2u + vec2<u32>(0u, 1u), 0);
        count += 1.0;
    }
    if (all(global_id.xy * 2u + 1u < src_dim)) {
        sum += textureLoad($input, global_id.xy * 2u + vec2<u32>(1u, 1u), 0);
        count += 1.0;
    }
    textureStore($output, global_id.xy, sum * $modulator / count);
}
