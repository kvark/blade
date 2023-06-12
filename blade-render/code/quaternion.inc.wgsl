fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}
fn qinv(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz,q.w);
}
fn make_quat(m: mat3x3<f32>) -> vec4<f32> {
    let w = sqrt(1.0 + m[0].x + m[1].y + m[2].z) / 2.0;
    let v = vec3<f32>(m[1].z - m[2].y, m[2].x - m[0].z, m[0].y - m[1].x);
    return vec4<f32>(v / (4.0 * w), w);
}
