//! Isotropic remeshing of triangle meshes, after Botsch and Kobbelt, "A
//! Remeshing Approach to Multiresolution Modeling" (2004).
//!
//! Optimizing the positions of vertices, see [`super::diff`], keeps the
//! triangles they started with, which stretch and crowd as the shape
//! changes, and limit the detail it can take. Remeshing between steps of the
//! optimization brings the edges back to a target length, so the mesh can
//! start coarse and refine as it goes, like continuous remeshing does
//! (Palfinger, "Continuous Remeshing for Inverse Rendering", 2022). Remeshing
//! itself is a discrete step, which the gradient doesn't go through.
//!
//! Every pass works on a mesh that is a 2-manifold, possibly with boundary,
//! and keeps it one: edges on the boundary are only ever split.

use glam::Vec3;
use std::collections::{HashMap, HashSet};

/// A triangle mesh by its positions, with indices that go counter-clockwise
/// when seen from outside.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

struct Topology {
    /// The triangles along every edge.
    edge_faces: HashMap<(u32, u32), Vec<usize>>,
    /// The vertices around every vertex.
    neighbors: Vec<Vec<u32>>,
    /// The triangles around every vertex.
    vertex_faces: Vec<Vec<usize>>,
    boundary: Vec<bool>,
}

impl Topology {
    fn new(vertex_count: usize, triangles: &[[u32; 3]]) -> Self {
        let mut edge_faces = HashMap::<(u32, u32), Vec<usize>>::new();
        let mut neighbor_sets = vec![HashSet::new(); vertex_count];
        let mut vertex_faces = vec![Vec::new(); vertex_count];
        for (face, t) in triangles.iter().enumerate() {
            for i in 0..3 {
                let (a, b) = (t[i], t[(i + 1) % 3]);
                edge_faces.entry(edge_key(a, b)).or_default().push(face);
                neighbor_sets[a as usize].insert(b);
                neighbor_sets[b as usize].insert(a);
                vertex_faces[a as usize].push(face);
            }
        }
        let mut boundary = vec![false; vertex_count];
        for (&(a, b), faces) in &edge_faces {
            if faces.len() == 1 {
                boundary[a as usize] = true;
                boundary[b as usize] = true;
            }
        }
        let neighbors = neighbor_sets
            .into_iter()
            .map(|set| {
                let mut list: Vec<u32> = set.into_iter().collect();
                list.sort_unstable();
                list
            })
            .collect();
        Self {
            edge_faces,
            neighbors,
            vertex_faces,
            boundary,
        }
    }
}

impl Mesh {
    fn position(&self, index: u32) -> Vec3 {
        Vec3::from(self.positions[index as usize])
    }

    fn face_normal(&self, t: [u32; 3]) -> Vec3 {
        let [a, b, c] = t.map(|i| self.position(i));
        (b - a).cross(c - a)
    }

    /// Normals of the vertices, as the normalized sums of the normals of the
    /// triangles around them, by area.
    pub fn vertex_normals(&self) -> Vec<[f32; 3]> {
        let mut sums = vec![Vec3::ZERO; self.positions.len()];
        for &t in &self.triangles {
            let normal = self.face_normal(t);
            for i in t {
                sums[i as usize] += normal;
            }
        }
        sums.iter()
            .map(|n| n.normalize_or(Vec3::Z).to_array())
            .collect()
    }

    /// Mean length of the edges.
    pub fn mean_edge_length(&self) -> f32 {
        let topology = Topology::new(self.positions.len(), &self.triangles);
        let total: f32 = topology
            .edge_faces
            .keys()
            .map(|&(a, b)| self.position(a).distance(self.position(b)))
            .sum();
        total / topology.edge_faces.len().max(1) as f32
    }

    /// Whether every edge has one or two triangles along it, and no triangle
    /// is degenerate in its indices.
    pub fn is_manifold(&self) -> bool {
        let topology = Topology::new(self.positions.len(), &self.triangles);
        let edges_fine = topology.edge_faces.values().all(|faces| faces.len() <= 2);
        let faces_fine = self
            .triangles
            .iter()
            .all(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0]);
        edges_fine && faces_fine
    }

    /// Split every edge longer than `limit` at its middle, which splits
    /// the triangles along it, until none is left.
    fn split_long_edges(&mut self, limit: f32) {
        for _ in 0..8 {
            let mut middles = HashMap::<(u32, u32), u32>::new();
            for t in &self.triangles {
                for i in 0..3 {
                    let key = edge_key(t[i], t[(i + 1) % 3]);
                    if middles.contains_key(&key) {
                        continue;
                    }
                    let (a, b) = (self.position(key.0), self.position(key.1));
                    if a.distance(b) > limit {
                        middles.insert(key, self.positions.len() as u32);
                        self.positions.push(((a + b) * 0.5).to_array());
                    }
                }
            }
            if middles.is_empty() {
                return;
            }
            let mut triangles = Vec::with_capacity(self.triangles.len() * 2);
            for &t in &self.triangles {
                let m = [0, 1, 2].map(|i| middles.get(&edge_key(t[i], t[(i + 1) % 3])).copied());
                match m {
                    [None, None, None] => triangles.push(t),
                    [Some(ab), Some(bc), Some(ca)] => {
                        triangles.push([t[0], ab, ca]);
                        triangles.push([ab, t[1], bc]);
                        triangles.push([ca, bc, t[2]]);
                        triangles.push([ab, bc, ca]);
                    }
                    _ => {
                        // Rotate the triangle so that the first edge is split,
                        // and the second one too, if two of them are.
                        let rotation = (0..3)
                            .find(|&r| m[r].is_some() && (m[(r + 2) % 3].is_none()))
                            .unwrap();
                        let v = [0, 1, 2].map(|i| t[(i + rotation) % 3]);
                        let mid = [0, 1, 2].map(|i| m[(i + rotation) % 3]);
                        let ab = mid[0].unwrap();
                        match mid[1] {
                            None => {
                                triangles.push([v[0], ab, v[2]]);
                                triangles.push([ab, v[1], v[2]]);
                            }
                            Some(bc) => {
                                triangles.push([ab, v[1], bc]);
                                // Split the rest along the shorter diagonal.
                                let d0 = self.position(ab).distance(self.position(v[2]));
                                let d1 = self.position(bc).distance(self.position(v[0]));
                                if d0 < d1 {
                                    triangles.push([v[0], ab, v[2]]);
                                    triangles.push([ab, bc, v[2]]);
                                } else {
                                    triangles.push([v[0], ab, bc]);
                                    triangles.push([v[0], bc, v[2]]);
                                }
                            }
                        }
                    }
                }
            }
            self.triangles = triangles;
        }
    }

    /// Collapse every edge shorter than `limit` into its middle, as long as
    /// that keeps the mesh a manifold, doesn't flip a triangle, and doesn't
    /// make an edge longer than `upper`.
    fn collapse_short_edges(&mut self, limit: f32, upper: f32) {
        for _ in 0..8 {
            let topology = Topology::new(self.positions.len(), &self.triangles);
            let mut edges: Vec<(u32, u32)> = topology
                .edge_faces
                .keys()
                .copied()
                .filter(|&(a, b)| self.position(a).distance(self.position(b)) < limit)
                .collect();
            edges.sort_unstable();
            let mut locked = vec![false; self.positions.len()];
            let mut target = (0..self.positions.len() as u32).collect::<Vec<_>>();
            let mut removed = vec![false; self.triangles.len()];
            let mut any = false;
            for (a, b) in edges {
                if locked[a as usize] || locked[b as usize] {
                    continue;
                }
                if topology.boundary[a as usize] || topology.boundary[b as usize] {
                    continue;
                }
                // Only an edge of two triangles, which a mesh that isn't a
                // manifold may not have.
                if topology.edge_faces[&(a, b)].len() != 2 {
                    continue;
                }
                // The link condition: the neighbors that the ends share are
                // exactly the two opposite corners.
                let na = &topology.neighbors[a as usize];
                let nb = &topology.neighbors[b as usize];
                let shared = na.iter().filter(|v| nb.binary_search(v).is_ok()).count();
                if shared != 2 {
                    continue;
                }
                let middle = (self.position(a) + self.position(b)) * 0.5;
                let too_long = na
                    .iter()
                    .chain(nb)
                    .any(|&v| v != a && v != b && self.position(v).distance(middle) > upper);
                if too_long {
                    continue;
                }
                // The triangles that stay must not turn over.
                let flips = topology.vertex_faces[a as usize]
                    .iter()
                    .chain(&topology.vertex_faces[b as usize])
                    .any(|&face| {
                        let t = self.triangles[face];
                        if t.contains(&a) && t.contains(&b) {
                            return false;
                        }
                        let before = self.face_normal(t);
                        let moved = t.map(|i| if i == a || i == b { u32::MAX } else { i });
                        let p = moved.map(|i| {
                            if i == u32::MAX {
                                middle
                            } else {
                                self.position(i)
                            }
                        });
                        let after = (p[1] - p[0]).cross(p[2] - p[0]);
                        before.dot(after) <= 0.0
                    });
                if flips {
                    continue;
                }
                self.positions[a as usize] = middle.to_array();
                target[b as usize] = a;
                for &face in &topology.edge_faces[&edge_key(a, b)] {
                    removed[face] = true;
                }
                for &v in na.iter().chain(nb) {
                    locked[v as usize] = true;
                }
                locked[a as usize] = true;
                locked[b as usize] = true;
                any = true;
            }
            if !any {
                return;
            }
            self.triangles = self
                .triangles
                .iter()
                .zip(&removed)
                .filter(|&(_, &removed)| !removed)
                .map(|(t, _)| t.map(|i| target[i as usize]))
                .collect();
        }
    }

    /// Flip edges where that brings the valences of the four corners
    /// closer to 6, or 4 on the boundary.
    fn equalize_valences(&mut self) {
        for _ in 0..4 {
            let topology = Topology::new(self.positions.len(), &self.triangles);
            let target = |v: u32| if topology.boundary[v as usize] { 4 } else { 6 };
            let mut valence: Vec<i32> = topology.neighbors.iter().map(|n| n.len() as i32).collect();
            let mut edges: Vec<(u32, u32)> = topology.edge_faces.keys().copied().collect();
            edges.sort_unstable();
            let mut touched = vec![false; self.triangles.len()];
            let mut existing: HashSet<(u32, u32)> = topology.edge_faces.keys().copied().collect();
            let mut any = false;
            for (a, b) in edges {
                let faces = &topology.edge_faces[&(a, b)];
                if faces.len() != 2 || faces.iter().any(|&f| touched[f]) {
                    continue;
                }
                let (f0, f1) = (faces[0], faces[1]);
                let opposite = |face: usize| {
                    let t = self.triangles[face];
                    *t.iter().find(|&&v| v != a && v != b).unwrap()
                };
                let (c, d) = (opposite(f0), opposite(f1));
                if c == d || existing.contains(&edge_key(c, d)) {
                    continue;
                }
                let deviation = |va: i32, vb: i32, vc: i32, vd: i32| {
                    (va - target(a)).abs()
                        + (vb - target(b)).abs()
                        + (vc - target(c)).abs()
                        + (vd - target(d)).abs()
                };
                let (va, vb, vc, vd) = (
                    valence[a as usize],
                    valence[b as usize],
                    valence[c as usize],
                    valence[d as usize],
                );
                if deviation(va - 1, vb - 1, vc + 1, vd + 1) >= deviation(va, vb, vc, vd) {
                    continue;
                }
                // Keep the winding: `a, b, c` and `b, a, d` go around the same
                // way, and then `c, d, b`... become `a, d, c` and `d, b, c`.
                let t0 = self.triangles[f0];
                let forward = (0..3).any(|i| t0[i] == a && t0[(i + 1) % 3] == b);
                let (a, b) = if forward { (a, b) } else { (b, a) };
                let new0 = [a, d, c];
                let new1 = [d, b, c];
                // Only flip across a convex quadrilateral, without folding.
                let n_before =
                    self.face_normal(self.triangles[f0]) + self.face_normal(self.triangles[f1]);
                if self.face_normal(new0).dot(n_before) <= 0.0
                    || self.face_normal(new1).dot(n_before) <= 0.0
                {
                    continue;
                }
                self.triangles[f0] = new0;
                self.triangles[f1] = new1;
                touched[f0] = true;
                touched[f1] = true;
                existing.insert(edge_key(c, d));
                valence[a as usize] -= 1;
                valence[b as usize] -= 1;
                valence[c as usize] += 1;
                valence[d as usize] += 1;
                any = true;
            }
            if !any {
                return;
            }
        }
    }

    /// Move every vertex inside the mesh towards the mean of its neighbors,
    /// within the plane of its normal, which spaces them evenly over the
    /// surface without changing its shape much.
    fn relax(&mut self, amount: f32) {
        let topology = Topology::new(self.positions.len(), &self.triangles);
        let normals = self.vertex_normals();
        let positions = self.positions.clone();
        for (v, neighbors) in topology.neighbors.iter().enumerate() {
            if neighbors.is_empty() || topology.boundary[v] {
                continue;
            }
            let p = Vec3::from(positions[v]);
            let mean = neighbors
                .iter()
                .map(|&n| Vec3::from(positions[n as usize]))
                .sum::<Vec3>()
                / neighbors.len() as f32;
            let normal = Vec3::from(normals[v]);
            let shift = mean - p;
            self.positions[v] = (p + amount * (shift - normal * normal.dot(shift))).to_array();
        }
    }

    /// Drop the vertices that no triangle uses, keeping the order.
    fn compact(&mut self) {
        let mut used = vec![false; self.positions.len()];
        for t in &self.triangles {
            for &i in t {
                used[i as usize] = true;
            }
        }
        let mut map = vec![u32::MAX; self.positions.len()];
        let mut positions = Vec::new();
        for (i, p) in self.positions.iter().enumerate() {
            if used[i] {
                map[i] = positions.len() as u32;
                positions.push(*p);
            }
        }
        self.positions = positions;
        for t in &mut self.triangles {
            *t = t.map(|i| map[i as usize]);
        }
    }

    /// Bring the edges to about `edge_length`, between 4/5 and 4/3 of it.
    pub fn remesh(&self, edge_length: f32, iterations: u32) -> Mesh {
        let mut mesh = self.clone();
        let (lower, upper) = (0.8 * edge_length, 4.0 / 3.0 * edge_length);
        for _ in 0..iterations {
            mesh.split_long_edges(upper);
            mesh.collapse_short_edges(lower, upper);
            mesh.equalize_valences();
            mesh.relax(0.5);
        }
        mesh.compact();
        mesh
    }

    /// The mesh as a geometry for [`crate::model::Baker::create_model`],
    /// with smooth normals, and texture coordinates from its direction
    /// about the origin, since remeshing has none to keep.
    pub fn to_geometry(&self, name: &str) -> crate::ProceduralGeometry {
        let normals = self.vertex_normals();
        let encode = |v: Vec3| {
            let q = |f: f32| ((f.clamp(-1.0, 1.0) * 127.0).round() as i8) as u8 as u32;
            q(v.x) | (q(v.y) << 8) | (q(v.z) << 16)
        };
        let vertices = self
            .positions
            .iter()
            .zip(&normals)
            .map(|(&p, &n)| {
                let normal = Vec3::from(n);
                let tangent = normal.any_orthonormal_vector();
                let dir = Vec3::from(p).normalize_or(Vec3::Z);
                crate::Vertex {
                    position: p,
                    bitangent_sign: 1.0,
                    tex_coords: [
                        0.5 + dir.z.atan2(dir.x) / std::f32::consts::TAU,
                        dir.y.acos() / std::f32::consts::PI,
                    ],
                    normal: encode(normal),
                    tangent: encode(tangent),
                }
            })
            .collect();
        crate::ProceduralGeometry {
            name: name.to_string(),
            vertices,
            indices: self.triangles.iter().flatten().copied().collect(),
            ..Default::default()
        }
    }
}

/// A regular icosahedron of the given circumradius, about the origin.
pub fn icosahedron(radius: f32) -> Mesh {
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    let raw = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let positions = raw
        .iter()
        .map(|&p| (Vec3::from(p).normalize() * radius).to_array())
        .collect();
    let triangles = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    Mesh {
        positions,
        triangles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outward(mesh: &Mesh) -> bool {
        mesh.triangles.iter().all(|&t| {
            let center = t.iter().map(|&i| mesh.position(i)).sum::<Vec3>() / 3.0;
            mesh.face_normal(t).dot(center) > 0.0
        })
    }

    #[test]
    fn icosahedron_is_closed_and_outward() {
        let mesh = icosahedron(1.0);
        assert!(mesh.is_manifold());
        assert!(outward(&mesh));
        let topology = Topology::new(mesh.positions.len(), &mesh.triangles);
        assert!(topology.edge_faces.values().all(|f| f.len() == 2));
    }

    #[test]
    fn remeshing_refines_and_keeps_a_closed_surface() {
        let coarse = icosahedron(1.0);
        let target = 0.25;
        let fine = coarse.remesh(target, 6);
        assert!(fine.is_manifold());
        assert!(outward(&fine));
        let topology = Topology::new(fine.positions.len(), &fine.triangles);
        assert!(
            topology.edge_faces.values().all(|f| f.len() == 2),
            "it opened up"
        );
        assert!(fine.triangles.len() > 4 * coarse.triangles.len());
        let mean = fine.mean_edge_length();
        assert!(
            (0.8 * target..4.0 / 3.0 * target).contains(&mean),
            "edges are {mean} long on average"
        );
        // Closed, so V - E + F = 2.
        let euler = fine.positions.len() as i64 - topology.edge_faces.len() as i64
            + fine.triangles.len() as i64;
        assert_eq!(euler, 2);
        // The shape stays about the icosahedron, between its in- and
        // circumradius.
        for p in &fine.positions {
            let r = Vec3::from(*p).length();
            assert!((0.7..=1.01).contains(&r), "a vertex is at {r}");
        }
    }

    #[test]
    fn remeshing_coarsens() {
        let fine = icosahedron(1.0).remesh(0.15, 6);
        let coarse = fine.remesh(0.5, 6);
        assert!(coarse.is_manifold());
        assert!(outward(&coarse));
        assert!(coarse.triangles.len() * 4 < fine.triangles.len());
    }

    #[test]
    fn remeshing_an_open_mesh_keeps_its_boundary() {
        let mesh = Mesh {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        let fine = mesh.remesh(0.2, 4);
        assert!(fine.is_manifold());
        for p in &fine.positions {
            assert!(p[2].abs() < 1.0e-6);
            assert!((-1.0e-6..=1.0 + 1.0e-6).contains(&p[0]));
            assert!((-1.0e-6..=1.0 + 1.0e-6).contains(&p[1]));
        }
        assert!(fine.triangles.iter().all(|&t| fine.face_normal(t).z > 0.0));
    }
}
