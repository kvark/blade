use synaga_shader::*;

#[shader]
pub fn qrot(q: vec4, v: vec3) -> vec3 {
    return v + 2.0 * cross(q.xyz(), cross(q.xyz(), v) + q.w * v);
}

#[shader]
pub fn qinv(q: vec4) -> vec4 {
    return (-q.xyz()).extend(q.w);
}

#[shader]
pub fn make_quat(m: mat3x3) -> vec4 {
    let mut q = vec4::default();
    if (m[2].z < 0.0) {
        // x^2 + y^2 >= z^2 + w^2
        let dif10 = m[1].y - m[0].x;
        let omm22 = 1.0 - m[2].z;
        if (dif10 < 0.0) {
            // x^2 >= y^2
            q = vec4(
                omm22 - dif10,
                m[0].y + m[1].x,
                m[0].z + m[2].x,
                m[1].z - m[2].y,
            );
        } else {
            // y^2 >= x^2
            q = vec4(
                m[0].y + m[1].x,
                omm22 + dif10,
                m[1].z + m[2].y,
                m[2].x - m[0].z,
            );
        }
    } else {
        // z^2 + w^2 >= x^2 + y^2
        let sum10 = m[1].y + m[0].x;
        let opm22 = 1.0 + m[2].z;
        if (sum10 < 0.0) {
            // z^2 >= w^2
            q = vec4(
                m[0].z + m[2].x,
                m[1].z + m[2].y,
                opm22 - sum10,
                m[0].y - m[1].x,
            );
        } else {
            // w^2 >= z^2
            q = vec4(
                m[1].z - m[2].y,
                m[2].x - m[0].z,
                m[0].y - m[1].x,
                opm22 + sum10,
            );
        }
    }
    return normalize(q);
}

#[shader]
pub fn shortest_arc_quat(a: vec3, b: vec3) -> vec4 {
    if (dot(a, b) < -0.99999) {
        // Choose the axis of rotation that doesn't align with the vectors
        return select(
            vec4(1.0, 0.0, 0.0, 0.0),
            vec4(0.0, 1.0, 0.0, 0.0),
            abs(a.x) > abs(a.y),
        );
    } else {
        return normalize((cross(a, b)).extend(1.0 + dot(a, b)));
    }
}
