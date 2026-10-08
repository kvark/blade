use synaga_shader::*;

/// A rotation kept as a unit quaternion in a `Vec4`: the vector part in
/// `xyz`, and the scalar in `w`.
pub trait Quaternion {
    /// `v` rotated.
    fn rotate(self, v: Vec3) -> Vec3;
    /// The rotation back.
    fn inv(self) -> Self;
}

impl Quaternion for Vec4 {
    fn rotate(self, v: Vec3) -> Vec3 {
        v + 2.0 * self.xyz().cross(self.xyz().cross(v) + self.w * v)
    }

    fn inv(self) -> Self {
        (-self.xyz()).extend(self.w)
    }
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
