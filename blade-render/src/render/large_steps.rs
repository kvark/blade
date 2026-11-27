//! Optimizing vertex positions with large steps, after Nicolet, Jacobson and
//! Jakob, "Large Steps in Inverse Rendering of Geometry" (2021).
//!
//! The gradient by the positions of vertices is mostly at silhouettes and
//! where shading changes fastest, and Adam moves every vertex on its own, so
//! the surface crumples where the gradient is, and barely moves elsewhere.
//! Instead, the positions are `x = (I + lambda L)^-1 u` of the parameters
//! `u`, with the Laplacian `L` of the mesh, so that a change of the
//! parameters is a smooth change of the positions. The parameters take steps
//! of Adam with one scale for all, which keeps that smoothness.
//!
//! This runs on the host, between reading the gradients of the vertices
//! back, see [`super::diff::DiffTracer::vertex_grads`], and writing their
//! offsets, see [`super::diff::DiffTracer::write_vertex_offsets`].

use glam::Vec3;

/// The vertices of a mesh, and who neighbors whom, by slot.
#[derive(Clone, Debug, Default)]
pub struct Adjacency {
    /// The neighbors of every vertex, which every edge adds both ways.
    pub neighbors: Vec<Vec<u32>>,
}

impl Adjacency {
    pub fn from_triangles(vertex_count: usize, triangles: &[[u32; 3]]) -> Self {
        let mut neighbors = vec![Vec::new(); vertex_count];
        for t in triangles {
            for i in 0..3 {
                let (a, b) = (t[i] as usize, t[(i + 1) % 3] as usize);
                if !neighbors[a].contains(&(b as u32)) {
                    neighbors[a].push(b as u32);
                }
                if !neighbors[b].contains(&(a as u32)) {
                    neighbors[b].push(a as u32);
                }
            }
        }
        Self { neighbors }
    }

    /// `(I + lambda L) v`, with the combinatorial Laplacian `L = D - A`.
    fn apply(&self, lambda: f32, v: &[Vec3], out: &mut [Vec3]) {
        for (i, neighbors) in self.neighbors.iter().enumerate() {
            let sum: Vec3 = neighbors.iter().map(|&j| v[j as usize]).sum();
            out[i] = v[i] + lambda * (neighbors.len() as f32 * v[i] - sum);
        }
    }

    /// Solve `(I + lambda L) x = b` by conjugate gradients, from `x`.
    pub fn solve(&self, lambda: f32, b: &[Vec3], x: &mut [Vec3]) {
        let n = b.len();
        let mut ax = vec![Vec3::ZERO; n];
        self.apply(lambda, x, &mut ax);
        let mut r: Vec<Vec3> = b.iter().zip(&ax).map(|(b, a)| *b - *a).collect();
        let mut p = r.clone();
        // The three coordinates are independent systems with the same matrix.
        let dot = |a: &[Vec3], b: &[Vec3]| -> Vec3 { a.iter().zip(b).map(|(a, b)| *a * *b).sum() };
        let mut rr = dot(&r, &r);
        let scale = dot(b, b).max_element().max(1.0e-30);
        let mut ap = vec![Vec3::ZERO; n];
        for _ in 0..1000 {
            if rr.max_element() <= 1.0e-12 * scale {
                break;
            }
            self.apply(lambda, &p, &mut ap);
            let pap = dot(&p, &ap);
            let alpha = Vec3::select(pap.cmpgt(Vec3::ZERO), rr / pap, Vec3::ZERO);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let next = dot(&r, &r);
            let beta = Vec3::select(rr.cmpgt(Vec3::ZERO), next / rr, Vec3::ZERO);
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rr = next;
        }
    }
}

/// The settings of [`LargeSteps`].
#[derive(Clone, Copy, Debug)]
pub struct LargeStepsConfig {
    /// Weight of the Laplacian, where larger is smoother. Nicolet et al.
    /// use tens to hundreds for meshes of thousands of vertices.
    pub lambda: f32,
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl Default for LargeStepsConfig {
    fn default() -> Self {
        Self {
            lambda: 20.0,
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
        }
    }
}

/// The state of optimizing the offsets of the vertices of a mesh.
pub struct LargeSteps {
    adjacency: Adjacency,
    /// The parameters, `(I + lambda L) x`.
    params: Vec<Vec3>,
    /// The offsets of the vertices.
    offsets: Vec<Vec3>,
    moment: Vec<Vec3>,
    /// The second moment, one for all, so that steps stay smooth.
    variance: f32,
    step: i32,
    lambda: f32,
}

impl LargeSteps {
    /// Start from `offsets`, for a mesh of `triangles` over them.
    pub fn new(offsets: Vec<Vec3>, triangles: &[[u32; 3]], lambda: f32) -> Self {
        let adjacency = Adjacency::from_triangles(offsets.len(), triangles);
        let mut params = vec![Vec3::ZERO; offsets.len()];
        adjacency.apply(lambda, &offsets, &mut params);
        Self {
            moment: vec![Vec3::ZERO; offsets.len()],
            adjacency,
            params,
            offsets,
            variance: 0.0,
            step: 0,
            lambda,
        }
    }

    pub fn offsets(&self) -> &[Vec3] {
        &self.offsets
    }

    /// Take a step along `grads`, the derivative of the loss by the offsets.
    pub fn step(&mut self, grads: &[Vec3], config: &LargeStepsConfig) {
        assert_eq!(grads.len(), self.offsets.len());
        assert_eq!(
            config.lambda, self.lambda,
            "lambda changed under the parameters"
        );
        // By the parameters: the matrix is symmetric.
        let mut d_params = vec![Vec3::ZERO; grads.len()];
        self.adjacency.solve(self.lambda, grads, &mut d_params);
        self.step += 1;
        let largest = d_params
            .iter()
            .map(|g| g.length_squared())
            .fold(0.0f32, f32::max);
        self.variance = config.beta2 * self.variance + (1.0 - config.beta2) * largest;
        let bias1 = 1.0 - config.beta1.powi(self.step);
        let bias2 = 1.0 - config.beta2.powi(self.step);
        let scale = config.learning_rate / ((self.variance / bias2).sqrt() + config.epsilon);
        for ((param, moment), grad) in self.params.iter_mut().zip(&mut self.moment).zip(&d_params) {
            *moment = config.beta1 * *moment + (1.0 - config.beta1) * *grad;
            *param -= scale * *moment / bias1;
        }
        let params = self.params.clone();
        self.adjacency
            .solve(self.lambda, &params, &mut self.offsets);
    }

    /// Carry the state over to a new mesh, such as after remeshing, by taking
    /// the moment of the closest of `old_positions` for every one of
    /// `new_positions`, whose offsets start from zero.
    pub fn remap(
        &self,
        old_positions: &[Vec3],
        new_positions: &[Vec3],
        triangles: &[[u32; 3]],
    ) -> Self {
        let mut next = Self::new(
            vec![Vec3::ZERO; new_positions.len()],
            triangles,
            self.lambda,
        );
        for (moment, p) in next.moment.iter_mut().zip(new_positions) {
            let closest = old_positions
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    p.distance_squared(*a.1)
                        .total_cmp(&p.distance_squared(*b.1))
                })
                .map(|(i, _)| i);
            if let Some(i) = closest {
                *moment = self.moment[i];
            }
        }
        next.variance = self.variance;
        next.step = self.step;
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> (usize, Vec<[u32; 3]>) {
        // A fan around a center, which every vertex neighbors.
        let triangles = (0..n)
            .map(|i| [0, 1 + i as u32, 1 + ((i + 1) % n) as u32])
            .collect();
        (n + 1, triangles)
    }

    #[test]
    fn solve_inverts_the_system() {
        let (count, triangles) = ring(8);
        let adjacency = Adjacency::from_triangles(count, &triangles);
        let x_true: Vec<Vec3> = (0..count)
            .map(|i| Vec3::new(i as f32, (i * i) as f32 * 0.1, -(i as f32)))
            .collect();
        let mut b = vec![Vec3::ZERO; count];
        adjacency.apply(5.0, &x_true, &mut b);
        let mut x = vec![Vec3::ZERO; count];
        adjacency.solve(5.0, &b, &mut x);
        for (a, e) in x.iter().zip(&x_true) {
            assert!(a.distance(*e) < 1.0e-3, "{a} for {e}");
        }
    }

    #[test]
    fn a_step_spreads_over_the_neighborhood() {
        let (count, triangles) = ring(8);
        let mut steps = LargeSteps::new(vec![Vec3::ZERO; count], &triangles, 10.0);
        // Pull on one rim vertex only.
        let mut grads = vec![Vec3::ZERO; count];
        grads[1] = Vec3::new(0.0, 0.0, -1.0);
        steps.step(
            &grads,
            &LargeStepsConfig {
                lambda: 10.0,
                ..Default::default()
            },
        );
        let offsets = steps.offsets();
        assert!(offsets[1].z > 0.0);
        // The rest moves the same way, less.
        for (i, o) in offsets.iter().enumerate().skip(2) {
            assert!(o.z > 0.0 && o.z < offsets[1].z, "vertex {i} moved by {o}");
        }
    }
}
