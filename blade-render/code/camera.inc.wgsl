struct CameraParams {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>,
    fov: vec2<f32>,
    film_offset: vec2<f32>,
    target_size: vec2<u32>,
}

const VFLIP: vec2<f32> = vec2<f32>(1.0, -1.0);

// Direction of the ray through a point on the film, in pixel units.
fn get_ray_direction_at(cp: CameraParams, film_pos: vec2<f32>) -> vec3<f32> {
    let half_size = 0.5 * vec2<f32>(cp.target_size);
    let ndc = (film_pos - half_size) / half_size;
    // Right-handed coordinate system with X=right, Y=up, and Z=towards the camera
    let local_dir = vec3<f32>(cp.film_offset + VFLIP * ndc * tan(0.5 * cp.fov), -1.0);
    return normalize(qrot(cp.orientation, local_dir));
}

fn get_ray_direction(cp: CameraParams, pixel: vec2<i32>) -> vec3<f32> {
    return get_ray_direction_at(cp, vec2<f32>(pixel) + vec2<f32>(0.5));
}

fn get_projected_pixel_float(cp: CameraParams, point: vec3<f32>) -> vec2<f32> {
    let local_dir = qrot(qinv(cp.orientation), point - cp.position);
    if local_dir.z >= 0.0 {
        return vec2<f32>(-1.0);
    }
    let slope = local_dir.xy / -local_dir.z;
    let ndc = VFLIP * (slope - cp.film_offset) / tan(0.5 * cp.fov);
    let half_size = 0.5 * vec2<f32>(cp.target_size);
    return (ndc + vec2<f32>(1.0)) * half_size;
}

fn get_projected_pixel(cp: CameraParams, point: vec3<f32>) -> vec2<i32> {
    return vec2<i32>(get_projected_pixel_float(cp, point));
}
