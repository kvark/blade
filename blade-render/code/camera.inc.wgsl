struct CameraParams {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>,
    fov: vec2<f32>,
    target_size: vec2<u32>,
}

fn get_ray_direction(cp: CameraParams, pixel: vec2<i32>) -> vec3<f32> {
    let half_size = 0.5 * vec2<f32>(cp.target_size);
    let ndc = (vec2<f32>(pixel) + vec2<f32>(0.5) - half_size) / half_size;
    // Right-handed coordinate system with X=right, Y=up, and Z=towards the camera
    let local_dir = vec3<f32>(ndc * tan(0.5 * cp.fov), -1.0);
    return normalize(qrot(cp.orientation, local_dir));
}

fn get_projected_pixel(cp: CameraParams, point: vec3<f32>) -> vec2<i32> {
    let local_dir = qrot(qinv(cp.orientation), point - cp.position);
    if local_dir.z >= 0.0 {
        return vec2<i32>(-1);
    }
    let ndc = local_dir.xy / (-local_dir.z * tan(0.5 * cp.fov));
    let half_size = 0.5 * vec2<f32>(cp.target_size);
    return vec2<i32>((ndc + vec2<f32>(1.0)) * half_size);
}
