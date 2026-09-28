use synaga_shader::*;

pub fn qrot(q: Vec4, v: Vec3) -> Vec3 {
    v + 2.0 * q.xyz().cross(q.xyz().cross(v) + q.w * v)
}

pub fn qinv(q: Vec4) -> Vec4 {
    (-q.xyz()).extend(q.w)
}

pub fn shortest_arc_quat(a: Vec3, b: Vec3) -> Vec4 {
    if a.dot(b) < -0.99999 {
        // Choose the axis of rotation that doesn't align with the vectors
        select(
            vec4(1.0, 0.0, 0.0, 0.0),
            vec4(0.0, 1.0, 0.0, 0.0),
            a.x.abs() > a.y.abs(),
        )
    } else {
        a.cross(b).extend(1.0 + a.dot(b)).normalize()
    }
}
