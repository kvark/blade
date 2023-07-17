struct Params {
    extent: vec2<u32>,
}

var<uniform> params: Params;
var input: texture_2d<f32>;
var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id.xy >= params.extent)) {
        return;
    }

    //TODO: wavelet kernel
    let radiance = textureLoad(input, vec2<i32>(global_id.xy), 0);
    textureStore(output, global_id.xy, radiance);
}
