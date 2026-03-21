#![cfg(target_os = "android")]

use std::collections::HashMap;

pub fn pack4x8snorm(v: [f32; 4]) -> u32 {
    v.iter().rev().fold(0u32, |u, f| {
        (u << 8) | (f.clamp(-1.0, 1.0) * 127.0 + 0.5) as i8 as u8 as u32
    })
}

pub fn encode_normal(n: [f32; 3]) -> u32 {
    pack4x8snorm([n[0], n[1], n[2], 0.0])
}

pub fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 1.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Simple hash-based noise for deterministic displacement.
pub fn hash_noise(seed: u32, x: f32, y: f32, z: f32) -> f32 {
    let ix = (x * 1000.0) as i32;
    let iy = (y * 1000.0) as i32;
    let iz = (z * 1000.0) as i32;
    let mut h = seed.wrapping_mul(374761393);
    h = h.wrapping_add((ix as u32).wrapping_mul(1103515245));
    h = h.wrapping_add((iy as u32).wrapping_mul(12345));
    h = h.wrapping_add((iz as u32).wrapping_mul(2654435761));
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    h ^= h >> 16;
    (h as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

/// Multi-octave noise for richer surface detail.
pub fn fbm_noise(seed: u32, x: f32, y: f32, z: f32, octaves: u32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut total_amplitude = 0.0;
    for i in 0..octaves {
        value += amplitude
            * hash_noise(
                seed.wrapping_add(i * 31),
                x * frequency,
                y * frequency,
                z * frequency,
            );
        total_amplitude += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    value / total_amplitude
}

/// Subdivide an icosphere. Returns (positions on unit sphere, triangle indices).
fn subdivide_icosphere(subdivisions: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut positions: Vec<[f32; 3]> = vec![
        normalize([-1.0, phi, 0.0]),
        normalize([1.0, phi, 0.0]),
        normalize([-1.0, -phi, 0.0]),
        normalize([1.0, -phi, 0.0]),
        normalize([0.0, -1.0, phi]),
        normalize([0.0, 1.0, phi]),
        normalize([0.0, -1.0, -phi]),
        normalize([0.0, 1.0, -phi]),
        normalize([phi, 0.0, -1.0]),
        normalize([phi, 0.0, 1.0]),
        normalize([-phi, 0.0, -1.0]),
        normalize([-phi, 0.0, 1.0]),
    ];
    let mut triangles: Vec<[u32; 3]> = vec![
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

    for _ in 0..subdivisions {
        let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut new_triangles = Vec::new();
        let get_midpoint = |positions: &mut Vec<[f32; 3]>,
                            cache: &mut HashMap<(u32, u32), u32>,
                            a: u32,
                            b: u32|
         -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&idx) = cache.get(&key) {
                return idx;
            }
            let pa = positions[a as usize];
            let pb = positions[b as usize];
            let mid = normalize([
                (pa[0] + pb[0]) * 0.5,
                (pa[1] + pb[1]) * 0.5,
                (pa[2] + pb[2]) * 0.5,
            ]);
            let idx = positions.len() as u32;
            positions.push(mid);
            cache.insert(key, idx);
            idx
        };
        for tri in triangles.iter() {
            let ab = get_midpoint(&mut positions, &mut midpoint_cache, tri[0], tri[1]);
            let bc = get_midpoint(&mut positions, &mut midpoint_cache, tri[1], tri[2]);
            let ca = get_midpoint(&mut positions, &mut midpoint_cache, tri[2], tri[0]);
            new_triangles.push([tri[0], ab, ca]);
            new_triangles.push([tri[1], bc, ab]);
            new_triangles.push([tri[2], ca, bc]);
            new_triangles.push([ab, bc, ca]);
        }
        triangles = new_triangles;
    }

    (positions, triangles)
}

/// Generate an icosphere with `subdivisions` levels and noise displacement.
/// `axis_scales` stretches the sphere along each axis before displacement for non-spherical shapes.
pub fn generate_asteroid_mesh(
    seed: u32,
    radius: f32,
    roughness: f32,
    subdivisions: u32,
    axis_scales: [f32; 3],
) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    let (positions, triangles) = subdivide_icosphere(subdivisions);

    // Apply axis scaling + multi-octave noise displacement
    let displaced: Vec<[f32; 3]> = positions
        .iter()
        .map(|p| {
            let noise = fbm_noise(seed, p[0], p[1], p[2], 3);
            let r = radius * (1.0 + roughness * noise);
            [
                p[0] * r * axis_scales[0],
                p[1] * r * axis_scales[1],
                p[2] * r * axis_scales[2],
            ]
        })
        .collect();

    // Build vertices with per-face normals for hard-edge rocky look
    let mut vertices = Vec::with_capacity(triangles.len() * 3);
    let mut indices = Vec::with_capacity(triangles.len() * 3);

    for tri in triangles.iter() {
        let p0 = displaced[tri[0] as usize];
        let p1 = displaced[tri[1] as usize];
        let p2 = displaced[tri[2] as usize];

        let face_normal = normalize(cross(sub(p1, p0), sub(p2, p0)));
        let encoded_normal = encode_normal(face_normal);

        let base_idx = vertices.len() as u32;
        for &pos in &[p0, p1, p2] {
            vertices.push(blade_render::Vertex {
                position: pos,
                bitangent_sign: 1.0,
                tex_coords: [0.0, 0.0],
                normal: encoded_normal,
                tangent: encode_normal([1.0, 0.0, 0.0]),
            });
        }
        indices.push(base_idx);
        indices.push(base_idx + 1);
        indices.push(base_idx + 2);
    }

    (vertices, indices)
}

/// Generate a planet with smooth normals, higher subdivision, and continent/ocean coloring.
pub fn generate_planet_model(
    engine: &mut blade_engine::Engine,
    radius: f32,
) -> blade_asset::Handle<blade_render::Model> {
    let seed = 12345u32;
    let roughness = 0.015;
    let (positions, triangles) = subdivide_icosphere(4);

    // Displace and compute per-vertex height for coloring
    let heights: Vec<f32> = positions
        .iter()
        .map(|p| fbm_noise(seed, p[0] * 2.0, p[1] * 2.0, p[2] * 2.0, 4))
        .collect();
    let displaced: Vec<[f32; 3]> = positions
        .iter()
        .zip(heights.iter())
        .map(|(p, &h)| {
            let r = radius * (1.0 + roughness * h);
            [p[0] * r, p[1] * r, p[2] * r]
        })
        .collect();

    // Smooth normals: accumulate face normals per vertex
    let mut vertex_normals = vec![[0.0f32; 3]; displaced.len()];
    for tri in &triangles {
        let p0 = displaced[tri[0] as usize];
        let p1 = displaced[tri[1] as usize];
        let p2 = displaced[tri[2] as usize];
        let face_normal = cross(sub(p1, p0), sub(p2, p0));
        for &idx in tri {
            let n = &mut vertex_normals[idx as usize];
            n[0] += face_normal[0];
            n[1] += face_normal[1];
            n[2] += face_normal[2];
        }
    }
    for n in &mut vertex_normals {
        *n = normalize(*n);
    }

    let ocean_threshold = 0.0;
    let ocean_color: [f32; 4] = [0.08, 0.15, 0.4, 1.0];
    let land_color: [f32; 4] = [0.2, 0.35, 0.15, 1.0];
    let ice_color: [f32; 4] = [0.8, 0.85, 0.9, 1.0];

    let mut ocean_verts = Vec::new();
    let mut ocean_idxs = Vec::new();
    let mut land_verts = Vec::new();
    let mut land_idxs = Vec::new();
    let mut ice_verts = Vec::new();
    let mut ice_idxs = Vec::new();

    for tri in &triangles {
        let p0 = displaced[tri[0] as usize];
        let p1 = displaced[tri[1] as usize];
        let p2 = displaced[tri[2] as usize];
        let n0 = vertex_normals[tri[0] as usize];
        let n1 = vertex_normals[tri[1] as usize];
        let n2 = vertex_normals[tri[2] as usize];
        let avg_height =
            (heights[tri[0] as usize] + heights[tri[1] as usize] + heights[tri[2] as usize]) / 3.0;
        let avg_y = (positions[tri[0] as usize][1]
            + positions[tri[1] as usize][1]
            + positions[tri[2] as usize][1])
            / 3.0;
        let (verts, idxs) = if avg_y.abs() > 0.85 {
            (&mut ice_verts, &mut ice_idxs)
        } else if avg_height > ocean_threshold {
            (&mut land_verts, &mut land_idxs)
        } else {
            (&mut ocean_verts, &mut ocean_idxs)
        };

        let base = verts.len() as u32;
        for (&pos, &normal) in [p0, p1, p2].iter().zip([n0, n1, n2].iter()) {
            verts.push(blade_render::Vertex {
                position: pos,
                bitangent_sign: 1.0,
                tex_coords: [0.0, 0.0],
                normal: encode_normal(normal),
                tangent: encode_normal([1.0, 0.0, 0.0]),
            });
        }
        idxs.extend_from_slice(&[base, base + 1, base + 2]);
    }

    let mut geometries = Vec::new();
    if !ocean_verts.is_empty() {
        geometries.push(blade_render::ProceduralGeometry {
            name: "planet_ocean".to_string(),
            vertices: ocean_verts,
            indices: ocean_idxs,
            base_color_factor: ocean_color,
        });
    }
    if !land_verts.is_empty() {
        geometries.push(blade_render::ProceduralGeometry {
            name: "planet_land".to_string(),
            vertices: land_verts,
            indices: land_idxs,
            base_color_factor: land_color,
        });
    }
    if !ice_verts.is_empty() {
        geometries.push(blade_render::ProceduralGeometry {
            name: "planet_ice".to_string(),
            vertices: ice_verts,
            indices: ice_idxs,
            base_color_factor: ice_color,
        });
    }

    engine.create_model("planet", geometries)
}

/// Generate a comet model (nucleus only — tail is a particle trail).
pub fn generate_comet_model(
    seed: u32,
    engine: &mut blade_engine::Engine,
    radius: f32,
) -> blade_asset::Handle<blade_render::Model> {
    let (verts, idxs) = generate_asteroid_mesh(seed, radius, 0.3, 2, [1.0, 1.0, 1.0]);
    engine.create_model(
        &format!("comet_{seed}"),
        vec![blade_render::ProceduralGeometry {
            name: "comet_nucleus".to_string(),
            vertices: verts,
            indices: idxs,
            base_color_factor: [0.9, 0.95, 1.0, 1.0],
        }],
    )
}

/// Generate a thin hexagonal prism along -Z for laser rendering.
pub fn generate_laser_mesh(length: f32, radius: f32) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    let sides = 6;
    let mut vertices = Vec::with_capacity(sides * 4);
    let mut indices = Vec::with_capacity(sides * 6);
    for i in 0..sides {
        let angle0 = (i as f32 / sides as f32) * std::f32::consts::TAU;
        let angle1 = ((i + 1) as f32 / sides as f32) * std::f32::consts::TAU;
        let (c0, s0) = (angle0.cos(), angle0.sin());
        let (c1, s1) = (angle1.cos(), angle1.sin());
        let n = normalize([(c0 + c1) * 0.5, (s0 + s1) * 0.5, 0.0]);
        let en = encode_normal(n);
        let base = vertices.len() as u32;
        for &(cx, sx, z) in &[
            (c0, s0, 0.0f32),
            (c1, s1, 0.0),
            (c0, s0, -length),
            (c1, s1, -length),
        ] {
            vertices.push(blade_render::Vertex {
                position: [cx * radius, sx * radius, z],
                bitangent_sign: 1.0,
                tex_coords: [0.0, 0.0],
                normal: en,
                tangent: encode_normal([0.0, 0.0, 1.0]),
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
    }
    (vertices, indices)
}

/// Compute quaternion [x, y, z, w] that rotates +Z to the given direction.
pub fn rotation_from_z_to(dir: [f32; 3]) -> [f32; 4] {
    let from = [0.0_f32, 0.0, 1.0];
    let d = from[0] * dir[0] + from[1] * dir[1] + from[2] * dir[2];
    if d > 0.9999 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if d < -0.9999 {
        return [0.0, 1.0, 0.0, 0.0];
    }
    let axis = normalize(cross(from, dir));
    let half_angle = (d.clamp(-1.0, 1.0)).acos() * 0.5;
    let s = half_angle.sin();
    [axis[0] * s, axis[1] * s, axis[2] * s, half_angle.cos()]
}

/// Rotate a vector by a quaternion [x, y, z, w].
pub fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    [
        v[0] + 2.0 * (q[3] * uv[0] + uuv[0]),
        v[1] + 2.0 * (q[3] * uv[1] + uuv[1]),
        v[2] + 2.0 * (q[3] * uv[2] + uuv[2]),
    ]
}
