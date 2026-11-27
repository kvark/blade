//! Meshes of the zero level of a signed distance grid, whose vertices move
//! with its values, for optimizing shapes of any topology, after Shen et al.
//! "Deep Marching Tetrahedra" (2021).
//!
//! Every cube of the grid is split into six tetrahedra along its diagonal,
//! and every edge of a tetrahedron whose ends are on either side of zero
//! has a vertex where the values, interpolated linearly, are zero. The
//! gradient by the positions of these vertices, such as from
//! [`super::diff::DiffTracer::vertex_grads`], goes to the values at the
//! ends of their edges, which moves the surface, and with it the topology,
//! as the values cross zero.

use super::remesh::Mesh;
use glam::Vec3;
use std::collections::HashMap;

/// Values of a signed distance on the points of a regular grid, negative
/// inside of the shape.
#[derive(Clone, Debug)]
pub struct SdfGrid {
    /// Number of points along every axis.
    pub size: [usize; 3],
    /// Position of the first point.
    pub origin: Vec3,
    /// Distance between neighboring points.
    pub spacing: f32,
    /// X first, then Y, then Z.
    pub values: Vec<f32>,
}

/// The corners of the six tetrahedra of a cube, by the bits of their
/// offsets along X, Y and Z, around the diagonal from 0 to 7.
const TETRAHEDRA: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
];

/// A mesh of the zero level of an [`SdfGrid`], see [`SdfGrid::extract`].
#[derive(Clone, Debug, Default)]
pub struct Extraction {
    /// Its triangles face outside, towards the positive values.
    pub mesh: Mesh,
    /// The points of the grid on the inside and the outside of the edge of
    /// every vertex.
    pub sources: Vec<[u32; 2]>,
}

impl SdfGrid {
    /// A grid with the values of `distance` at its points.
    pub fn new(
        size: [usize; 3],
        origin: Vec3,
        spacing: f32,
        distance: impl Fn(Vec3) -> f32,
    ) -> Self {
        let mut grid = Self {
            size,
            origin,
            spacing,
            values: Vec::with_capacity(size[0] * size[1] * size[2]),
        };
        for z in 0..size[2] {
            for y in 0..size[1] {
                for x in 0..size[0] {
                    let p = grid.point(grid.index([x, y, z]));
                    grid.values.push(distance(p));
                }
            }
        }
        grid
    }

    pub fn index(&self, [x, y, z]: [usize; 3]) -> usize {
        (z * self.size[1] + y) * self.size[0] + x
    }

    fn coordinates(&self, index: usize) -> [usize; 3] {
        let x = index % self.size[0];
        let y = (index / self.size[0]) % self.size[1];
        let z = index / (self.size[0] * self.size[1]);
        [x, y, z]
    }

    pub fn point(&self, index: usize) -> Vec3 {
        let [x, y, z] = self.coordinates(index);
        self.origin + self.spacing * Vec3::new(x as f32, y as f32, z as f32)
    }

    /// Where the edge between the points `inside` and `outside` crosses
    /// zero, and how far along it from `inside`.
    fn crossing(&self, inside: usize, outside: usize) -> (Vec3, f32) {
        let (si, so) = (self.values[inside], self.values[outside]);
        let t = si / (si - so);
        let a = self.point(inside);
        (a + t * (self.point(outside) - a), t)
    }

    fn is_inside(&self, index: usize) -> bool {
        self.values[index] < 0.0
    }

    /// The mesh of the zero level, with a vertex for every edge that
    /// crosses it, shared by the tetrahedra around that edge, which makes it
    /// closed where the shape is inside of the grid.
    pub fn extract(&self) -> Extraction {
        let mut extraction = Extraction::default();
        let mut vertices = HashMap::<(u32, u32), u32>::new();
        let mut vertex = |inside: usize, outside: usize, extraction: &mut Extraction| -> u32 {
            *vertices
                .entry((inside as u32, outside as u32))
                .or_insert_with(|| {
                    let (p, _) = self.crossing(inside, outside);
                    extraction.mesh.positions.push(p.into());
                    extraction.sources.push([inside as u32, outside as u32]);
                    (extraction.mesh.positions.len() - 1) as u32
                })
        };
        let [sx, sy, sz] = self.size;
        for z in 0..sz.saturating_sub(1) {
            for y in 0..sy.saturating_sub(1) {
                for x in 0..sx.saturating_sub(1) {
                    let corner = |bits: usize| {
                        self.index([x + (bits & 1), y + ((bits >> 1) & 1), z + ((bits >> 2) & 1)])
                    };
                    for tetrahedron in TETRAHEDRA {
                        let points = tetrahedron.map(corner);
                        let (inside, outside): (Vec<usize>, Vec<usize>) =
                            points.iter().partition(|&&p| self.is_inside(p));
                        let polygon = match (inside.len(), outside.len()) {
                            (1, 3) => vec![
                                vertex(inside[0], outside[0], &mut extraction),
                                vertex(inside[0], outside[1], &mut extraction),
                                vertex(inside[0], outside[2], &mut extraction),
                            ],
                            (3, 1) => vec![
                                vertex(inside[0], outside[0], &mut extraction),
                                vertex(inside[1], outside[0], &mut extraction),
                                vertex(inside[2], outside[0], &mut extraction),
                            ],
                            // Around the four edges between the two pairs.
                            (2, 2) => vec![
                                vertex(inside[0], outside[0], &mut extraction),
                                vertex(inside[0], outside[1], &mut extraction),
                                vertex(inside[1], outside[1], &mut extraction),
                                vertex(inside[1], outside[0], &mut extraction),
                            ],
                            _ => continue,
                        };
                        let center = |set: &[usize]| {
                            set.iter().map(|&p| self.point(p)).sum::<Vec3>() / set.len() as f32
                        };
                        let outwards = center(&outside) - center(&inside);
                        let position = |v: u32| Vec3::from(extraction.mesh.positions[v as usize]);
                        for k in 1..polygon.len() - 1 {
                            let mut triangle = [polygon[0], polygon[k], polygon[k + 1]];
                            let normal = (position(triangle[1]) - position(triangle[0]))
                                .cross(position(triangle[2]) - position(triangle[0]));
                            if normal.dot(outwards) < 0.0 {
                                triangle.swap(1, 2);
                            }
                            extraction.mesh.triangles.push(triangle);
                        }
                    }
                }
            }
        }
        extraction
    }
}

impl Extraction {
    /// Add the derivative of a loss by the values of `grid`, which this was
    /// extracted from, to `value_grads`, given its derivative by the
    /// positions of the vertices.
    pub fn backpropagate(&self, grid: &SdfGrid, position_grads: &[Vec3], value_grads: &mut [f32]) {
        assert_eq!(position_grads.len(), self.sources.len());
        assert_eq!(value_grads.len(), grid.values.len());
        for (&[inside, outside], grad) in self.sources.iter().zip(position_grads) {
            let (inside, outside) = (inside as usize, outside as usize);
            let (si, so) = (grid.values[inside], grid.values[outside]);
            let denominator = (si - so) * (si - so);
            if denominator <= 0.0 {
                continue;
            }
            // `t = si / (si - so)` along the edge.
            let d_t = grad.dot(grid.point(outside) - grid.point(inside));
            value_grads[inside] += d_t * -so / denominator;
            value_grads[outside] += d_t * si / denominator;
        }
    }
}

/// Adam on the values of an [`SdfGrid`], with smoothness, which also keeps
/// the values that the surface doesn't reach in step with those around
/// them.
pub struct SdfOptimizer {
    moments: Vec<[f32; 2]>,
    step: i32,
}

/// The settings of [`SdfOptimizer`].
#[derive(Clone, Copy, Debug)]
pub struct SdfOptimizerConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    /// The fraction of the way to the mean of its neighbors that every
    /// value moves after each step.
    pub smoothness: f32,
    /// The values are kept within this distance of zero, in grid spacings.
    pub limit: f32,
}

impl Default for SdfOptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            smoothness: 0.0,
            limit: 4.0,
        }
    }
}

impl SdfOptimizer {
    pub fn new(grid: &SdfGrid) -> Self {
        Self {
            moments: vec![[0.0; 2]; grid.values.len()],
            step: 0,
        }
    }

    /// Step the values of `grid` along `grads`, their derivatives, and
    /// then smooth them.
    pub fn step(&mut self, grid: &mut SdfGrid, grads: &[f32], config: &SdfOptimizerConfig) {
        assert_eq!(grads.len(), grid.values.len());
        self.step += 1;
        let bias1 = 1.0 - config.beta1.powi(self.step);
        let bias2 = 1.0 - config.beta2.powi(self.step);
        let limit = config.limit * grid.spacing;
        for ((value, moment), &grad) in grid.values.iter_mut().zip(&mut self.moments).zip(grads) {
            if !grad.is_finite() {
                continue;
            }
            moment[0] = config.beta1 * moment[0] + (1.0 - config.beta1) * grad;
            moment[1] = config.beta2 * moment[1] + (1.0 - config.beta2) * grad * grad;
            let step = moment[0] / bias1 / ((moment[1] / bias2).sqrt() + config.epsilon);
            *value = (*value - config.learning_rate * step).clamp(-limit, limit);
        }
        if config.smoothness > 0.0 {
            // Outside of Adam, which would make as large a step of the
            // smoothness alone, where nothing else moves a value, as of the
            // gradient elsewhere.
            grid.values = grid.smoothed(config.smoothness);
        }
    }
}

impl SdfGrid {
    /// The values moved by `rate` towards the means of their neighbors.
    pub fn smoothed(&self, rate: f32) -> Vec<f32> {
        let [sx, sy, sz] = self.size;
        let mut values = self.values.clone();
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    for (axis, size) in [sx, sy, sz].into_iter().enumerate() {
                        for neighbor in [[x, y, z][axis].wrapping_sub(1), [x, y, z][axis] + 1] {
                            if neighbor < size {
                                let mut c = [x, y, z];
                                c[axis] = neighbor;
                                sum += self.values[self.index(c)];
                                count += 1.0;
                            }
                        }
                    }
                    // A grid of one cell has no neighbors.
                    if count == 0.0 {
                        continue;
                    }
                    let index = self.index([x, y, z]);
                    values[index] += rate * (sum / count - self.values[index]);
                }
            }
        }
        values
    }
}

/// A connected part of a mesh, see [`components`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Component {
    pub triangles: usize,
    /// `2 - 2 * genus` for a closed component.
    pub euler_characteristic: i64,
    /// The mean of its vertices.
    pub center: [f32; 3],
}

/// The connected parts of a mesh, largest first.
pub fn components(mesh: &Mesh) -> Vec<Component> {
    let mut parent: Vec<usize> = (0..mesh.positions.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for t in &mesh.triangles {
        for k in 0..3 {
            let (ra, rb) = (
                find(&mut parent, t[k] as usize),
                find(&mut parent, t[(k + 1) % 3] as usize),
            );
            parent[ra] = rb;
        }
    }
    // Vertices, edges and triangles of every part, by its root.
    let mut counts = std::collections::HashMap::<usize, [i64; 3]>::new();
    let mut sums = std::collections::HashMap::<usize, Vec3>::new();
    let mut edges = std::collections::HashSet::new();
    let mut vertices = std::collections::HashSet::new();
    for t in &mesh.triangles {
        let root = find(&mut parent, t[0] as usize);
        let entry = counts.entry(root).or_default();
        entry[2] += 1;
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if vertices.insert(a) {
                entry[0] += 1;
                *sums.entry(root).or_default() += Vec3::from(mesh.positions[a as usize]);
            }
            if edges.insert((a.min(b), a.max(b))) {
                entry[1] += 1;
            }
        }
    }
    let mut parts: Vec<Component> = counts
        .iter()
        .map(|(root, &[v, e, f])| Component {
            triangles: f as usize,
            euler_characteristic: v - e + f,
            center: (sums[root] / v as f32).into(),
        })
        .collect();
    parts.sort_by_key(|part| std::cmp::Reverse(part.triangles));
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_grid(radius: f32) -> SdfGrid {
        SdfGrid::new([12; 3], Vec3::splat(-1.1), 0.2, |p| p.length() - radius)
    }

    #[test]
    fn a_sphere_is_closed_and_round() {
        let grid = sphere_grid(0.8);
        let extraction = grid.extract();
        let mesh = &extraction.mesh;
        assert!(mesh.is_manifold());
        let parts = components(mesh);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].euler_characteristic, 2);
        for p in &mesh.positions {
            let r = Vec3::from(*p).length();
            assert!((r - 0.8).abs() < 0.05, "a vertex at {r}");
        }
        // Facing outwards.
        let volume: f32 = mesh
            .triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| Vec3::from(mesh.positions[i as usize]));
                a.dot(b.cross(c)) / 6.0
            })
            .sum();
        let expected = 4.0 / 3.0 * std::f32::consts::PI * 0.8f32.powi(3);
        assert!(
            (volume - expected).abs() < 0.1 * expected,
            "volume {volume}"
        );
    }

    #[test]
    fn a_torus_has_a_hole() {
        let grid = SdfGrid::new([16, 16, 8], Vec3::new(-1.5, -1.5, -0.7), 0.2, |p| {
            let ring = Vec3::new(p.x, p.y, 0.0).length() - 0.9;
            Vec3::new(ring, 0.0, p.z).length() - 0.4
        });
        let mesh = grid.extract().mesh;
        assert!(mesh.is_manifold());
        let parts = components(&mesh);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].euler_characteristic, 0);
    }

    #[test]
    fn backpropagation_matches_finite_differences() {
        let mut grid = sphere_grid(0.75);
        // A loss that is linear in the positions.
        let weight = |i: usize| Vec3::new(1.0 + i as f32 * 0.01, -0.5, 0.25);
        let loss = |grid: &SdfGrid| -> f32 {
            let extraction = grid.extract();
            extraction
                .mesh
                .positions
                .iter()
                .enumerate()
                .map(|(i, p)| weight(i).dot(Vec3::from(*p)))
                .sum()
        };
        let extraction = grid.extract();
        let position_grads: Vec<Vec3> = (0..extraction.sources.len()).map(weight).collect();
        let mut grads = vec![0.0; grid.values.len()];
        extraction.backpropagate(&grid, &position_grads, &mut grads);
        let [inside, outside] = extraction.sources[5];
        for index in [inside as usize, outside as usize] {
            let h = 1.0e-3;
            let value = grid.values[index];
            grid.values[index] = value + h;
            let plus = loss(&grid);
            grid.values[index] = value - h;
            let minus = loss(&grid);
            grid.values[index] = value;
            let numeric = (plus - minus) / (2.0 * h);
            assert!(
                (grads[index] - numeric).abs() < 1.0e-2 * numeric.abs().max(1.0),
                "{} against {numeric}",
                grads[index]
            );
        }
    }
}
