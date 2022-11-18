@vertex
fn vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
