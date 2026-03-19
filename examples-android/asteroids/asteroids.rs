#![cfg(target_os = "android")]

use std::{
    ffi::CString,
    fs,
    io::Read,
    path::Path,
    path::PathBuf,
    time::{Duration, Instant},
};

use android_activity::{AndroidApp, InputStatus, MainEvent, PollEvent};
use blade_graphics as gpu;
use log::info;
use openxr as xr;

// --- Mesh generation utilities ---

fn pack4x8snorm(v: [f32; 4]) -> u32 {
    v.iter().rev().fold(0u32, |u, f| {
        (u << 8) | (f.clamp(-1.0, 1.0) * 127.0 + 0.5) as i8 as u8 as u32
    })
}

fn encode_normal(n: [f32; 3]) -> u32 {
    pack4x8snorm([n[0], n[1], n[2], 0.0])
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 1.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Simple hash-based noise for deterministic displacement.
fn hash_noise(seed: u32, x: f32, y: f32, z: f32) -> f32 {
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
fn fbm_noise(seed: u32, x: f32, y: f32, z: f32, octaves: u32) -> f32 {
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

/// Generate an icosphere with `subdivisions` levels and noise displacement.
/// `axis_scales` stretches the sphere along each axis before displacement for non-spherical shapes.
fn generate_asteroid_mesh(
    seed: u32,
    radius: f32,
    roughness: f32,
    subdivisions: u32,
    axis_scales: [f32; 3],
) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    // Base icosahedron vertices
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let base_positions: Vec<[f32; 3]> = vec![
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

    let base_indices: Vec<[u32; 3]> = vec![
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

    let mut positions = base_positions;
    let mut triangles = base_indices;

    // Subdivide
    use std::collections::HashMap;
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
            let a = tri[0];
            let b = tri[1];
            let c = tri[2];
            let ab = get_midpoint(&mut positions, &mut midpoint_cache, a, b);
            let bc = get_midpoint(&mut positions, &mut midpoint_cache, b, c);
            let ca = get_midpoint(&mut positions, &mut midpoint_cache, c, a);
            new_triangles.push([a, ab, ca]);
            new_triangles.push([b, bc, ab]);
            new_triangles.push([c, ca, bc]);
            new_triangles.push([ab, bc, ca]);
        }
        triangles = new_triangles;
    }

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

// --- Asteroid field parameters ---

const ASTEROID_COUNT: usize = 200;
/// Half-width of the spawn slab perpendicular to the flow direction.
const FIELD_HALF_WIDTH: f32 = 40.0;
/// Half-height of the spawn slab.
const FIELD_HALF_HEIGHT: f32 = 20.0;
/// How far ahead of the player asteroids spawn.
const SPAWN_DISTANCE: f32 = 120.0;
/// How far behind the player asteroids are recycled.
const DESPAWN_BEHIND: f32 = 30.0;
/// Minimum distance from the player when spawning (initial spread).
const MIN_SPAWN_DISTANCE: f32 = 15.0;
/// Base speed of asteroid flow toward the player.
const FLOW_SPEED: f32 = 4.0;
/// Speed variation (+/- this fraction of FLOW_SPEED).
const SPEED_VARIATION: f32 = 0.4;

// --- Comet parameters ---
const COMET_COUNT: usize = 8;
const COMET_MIN_RADIUS: f32 = 200.0;
const COMET_MAX_RADIUS: f32 = 600.0;
const COMET_DESPAWN_RADIUS: f32 = 800.0;
const COMET_SPEED: f32 = 3.0;
const COMET_TAIL_SEGMENTS: usize = 4;
const COMET_TAIL_LENGTH: f32 = 8.0;
const COMET_NUCLEUS_RADIUS: f32 = 1.5;

struct Asteroid {
    object_handle: blade_engine::ObjectHandle,
}

struct AsteroidField {
    asteroids: Vec<Asteroid>,
    model_handles: Vec<blade_asset::Handle<blade_render::Model>>,
    /// The direction asteroids flow (toward the player). In XR, typically +Z (toward initial head).
    flow_dir: [f32; 3],
    next_seed: u32,
}

impl AsteroidField {
    fn new(engine: &mut blade_engine::Engine) -> Self {
        // Generate mesh variants with diverse shapes and roughness
        let variant_params: &[(f32, [f32; 3], [f32; 4])] = &[
            // (roughness, axis_scales, color)
            (0.35, [1.0, 1.0, 1.0], [0.45, 0.40, 0.35, 1.0]), // round rocky
            (0.50, [1.4, 0.7, 1.0], [0.50, 0.45, 0.38, 1.0]), // elongated
            (0.45, [1.0, 0.5, 1.2], [0.40, 0.38, 0.35, 1.0]), // flattened
            (0.60, [1.0, 1.0, 1.0], [0.35, 0.32, 0.30, 1.0]), // very rough dark
            (0.30, [0.8, 1.3, 0.9], [0.55, 0.50, 0.42, 1.0]), // tall lighter
            (0.55, [1.3, 0.8, 1.3], [0.42, 0.38, 0.32, 1.0]), // wide rough
            (0.40, [1.1, 1.1, 0.6], [0.48, 0.44, 0.38, 1.0]), // disc-like
            (0.65, [0.9, 0.9, 1.4], [0.38, 0.35, 0.30, 1.0]), // long rough dark
        ];
        let mut model_handles = Vec::new();
        for (i, &(roughness, axis_scales, color)) in variant_params.iter().enumerate() {
            let seed = (i * 7 + 42) as u32;
            let (vertices, indices) = generate_asteroid_mesh(seed, 1.0, roughness, 2, axis_scales);
            let handle = engine.create_model(
                &format!("asteroid_{i}"),
                vec![blade_render::ProceduralGeometry {
                    name: format!("asteroid_{i}"),
                    vertices,
                    indices,
                    base_color_factor: color,
                }],
            );
            model_handles.push(handle);
        }

        // Flow direction: asteroids come toward the player along -Z (standard forward in RH).
        // This gives the feeling of flying forward through a field.
        let flow_dir = [0.0, 0.0, 1.0]; // asteroids move in +Z toward the player at origin

        let mut field = AsteroidField {
            asteroids: Vec::with_capacity(ASTEROID_COUNT),
            model_handles,
            flow_dir,
            next_seed: 100,
        };

        // Spawn initial asteroids spread along the full depth of the field
        for _ in 0..ASTEROID_COUNT {
            field.spawn_asteroid(engine, true);
        }

        field
    }

    /// Spawn an asteroid ahead of the player.
    /// If `spread` is true, distribute along the full depth (for initial population).
    /// If false, spawn at the far edge (for recycling).
    fn spawn_asteroid(&mut self, engine: &mut blade_engine::Engine, spread: bool) {
        let seed = self.next_seed;
        self.next_seed += 1;

        let variant = (seed as usize) % self.model_handles.len();

        // Position: random XY within the slab, Z ahead of the player
        let x = hash_noise(seed, 0.1, 0.2, 0.3) * FIELD_HALF_WIDTH;
        let y = hash_noise(seed, 0.4, 0.5, 0.6) * FIELD_HALF_HEIGHT;
        let z_depth = if spread {
            // Distribute from MIN_SPAWN_DISTANCE to SPAWN_DISTANCE ahead
            let t = hash_noise(seed, 0.7, 0.8, 0.9).abs();
            -(MIN_SPAWN_DISTANCE + t * (SPAWN_DISTANCE - MIN_SPAWN_DISTANCE))
        } else {
            // Spawn at the far edge with slight variation
            -SPAWN_DISTANCE + hash_noise(seed, 0.7, 0.8, 0.9).abs() * 5.0
        };

        // Position in world space: flow_dir is +Z, so asteroids spawn at -Z (ahead)
        // and move toward +Z (toward/past the player)
        let pos = [x, y, z_depth];

        // Velocity: flow direction with some variation
        let speed = FLOW_SPEED * (1.0 + hash_noise(seed, 10.0, 11.0, 12.0) * SPEED_VARIATION);
        let velocity = [
            self.flow_dir[0] * speed + hash_noise(seed, 13.0, 14.0, 15.0) * 0.3,
            self.flow_dir[1] * speed + hash_noise(seed, 16.0, 17.0, 18.0) * 0.2,
            self.flow_dir[2] * speed + hash_noise(seed, 19.0, 20.0, 21.0) * 0.3,
        ];

        let angular_velocity = [
            hash_noise(seed, 30.0, 31.0, 32.0) * 0.5,
            hash_noise(seed, 33.0, 34.0, 35.0) * 0.5,
            hash_noise(seed, 36.0, 37.0, 38.0) * 0.5,
        ];

        let transform = blade_engine::Transform {
            position: mint::Vector3 {
                x: pos[0],
                y: pos[1],
                z: pos[2],
            },
            ..Default::default()
        };

        let handle = engine.add_object_with_model(
            "asteroid",
            self.model_handles[variant],
            transform,
            blade_engine::DynamicInput::Full,
        );

        // No colliders — asteroids pass through each other and never get stuck.

        // Set initial velocity on the rigid body
        engine.set_velocity(
            handle,
            mint::Vector3 {
                x: velocity[0],
                y: velocity[1],
                z: velocity[2],
            },
            mint::Vector3 {
                x: angular_velocity[0],
                y: angular_velocity[1],
                z: angular_velocity[2],
            },
        );

        self.asteroids.push(Asteroid {
            object_handle: handle,
        });
    }

    fn update(&mut self, engine: &mut blade_engine::Engine) {
        // Recycle asteroids that have passed behind the player
        let mut i = 0;
        while i < self.asteroids.len() {
            let pos = engine.get_object_position(self.asteroids[i].object_handle);
            // Asteroid has passed the player and is far enough behind
            let along_flow =
                pos.x * self.flow_dir[0] + pos.y * self.flow_dir[1] + pos.z * self.flow_dir[2];
            if along_flow > DESPAWN_BEHIND {
                let asteroid = self.asteroids.swap_remove(i);
                engine.remove_object(asteroid.object_handle);
                // Respawn at the far edge ahead
                self.spawn_asteroid(engine, false);
            } else {
                i += 1;
            }
        }
    }
}

// --- Comet mesh generation ---

/// Generate a comet model: bright nucleus + tapered tail made of cone segments.
/// The tail extends along +Z (will be oriented by the engine's transform).
fn generate_comet_model(
    seed: u32,
    engine: &mut blade_engine::Engine,
) -> blade_asset::Handle<blade_render::Model> {
    let mut geometries = Vec::new();

    // Nucleus: bright sphere
    {
        let (verts, idxs) =
            generate_asteroid_mesh(seed, COMET_NUCLEUS_RADIUS, 0.3, 2, [1.0, 1.0, 1.0]);
        geometries.push(blade_render::ProceduralGeometry {
            name: "comet_nucleus".to_string(),
            vertices: verts,
            indices: idxs,
            base_color_factor: [0.9, 0.95, 1.0, 1.0],
        });
    }

    // Tail: a series of cone frustum segments extending along +Z, getting wider and dimmer
    let ring_verts = 8;
    let tail_base_radius = COMET_NUCLEUS_RADIUS * 0.5;
    let tail_end_radius = COMET_NUCLEUS_RADIUS * 3.0;
    for seg in 0..COMET_TAIL_SEGMENTS {
        let t0 = seg as f32 / COMET_TAIL_SEGMENTS as f32;
        let t1 = (seg + 1) as f32 / COMET_TAIL_SEGMENTS as f32;
        let z0 = t0 * COMET_TAIL_LENGTH;
        let z1 = t1 * COMET_TAIL_LENGTH;
        // Radius grows from narrow to wide
        let r0 = tail_base_radius + t0 * (tail_end_radius - tail_base_radius);
        let r1 = tail_base_radius + t1 * (tail_end_radius - tail_base_radius);
        // Color fades from bright blue-white to dim
        let brightness = 1.0 - t0 * 0.7;
        let color = [0.5 * brightness, 0.7 * brightness, 1.0 * brightness, 1.0];

        let mut verts = Vec::with_capacity(ring_verts * 2);
        let mut idxs = Vec::new();

        for i in 0..ring_verts {
            let angle = (i as f32 / ring_verts as f32) * std::f32::consts::TAU;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            // Near ring vertex
            let pos0 = [cos_a * r0, sin_a * r0, z0];
            let n0 = normalize([cos_a, sin_a, -0.3]);
            verts.push(blade_render::Vertex {
                position: pos0,
                bitangent_sign: 1.0,
                tex_coords: [0.0, 0.0],
                normal: encode_normal(n0),
                tangent: encode_normal([0.0, 0.0, 1.0]),
            });

            // Far ring vertex
            let pos1 = [cos_a * r1, sin_a * r1, z1];
            verts.push(blade_render::Vertex {
                position: pos1,
                bitangent_sign: 1.0,
                tex_coords: [0.0, 0.0],
                normal: encode_normal(n0),
                tangent: encode_normal([0.0, 0.0, 1.0]),
            });

            // Two triangles forming a quad between this column and the next
            let next = (i + 1) % ring_verts;
            let a = (i * 2) as u32;
            let b = (i * 2 + 1) as u32;
            let c = (next * 2) as u32;
            let d = (next * 2 + 1) as u32;
            idxs.extend_from_slice(&[a, c, b, b, c, d]);
        }

        geometries.push(blade_render::ProceduralGeometry {
            name: format!("comet_tail_{seg}"),
            vertices: verts,
            indices: idxs,
            base_color_factor: color,
        });
    }

    engine.create_model(&format!("comet_{seed}"), geometries)
}

struct Comet {
    object_handle: blade_engine::ObjectHandle,
}

struct CometField {
    comets: Vec<Comet>,
    next_seed: u32,
}

impl CometField {
    fn new(engine: &mut blade_engine::Engine) -> Self {
        let mut field = CometField {
            comets: Vec::with_capacity(COMET_COUNT),
            next_seed: 50000,
        };
        for i in 0..COMET_COUNT {
            field.spawn_comet_stratified(engine, i);
        }
        field
    }

    /// Spawn a comet in a stratified sky sector.
    /// `sector` distributes comets evenly around the sky so they don't cluster.
    fn spawn_comet_stratified(&mut self, engine: &mut blade_engine::Engine, sector: usize) {
        let seed = self.next_seed;
        self.next_seed += 1;

        let model = generate_comet_model(seed, engine);

        // Stratified placement: divide the sky into COMET_COUNT sectors.
        // Each comet gets a base azimuth sector with jitter.
        let sector_width = std::f32::consts::TAU / COMET_COUNT as f32;
        let base_phi = sector as f32 * sector_width;
        let phi = base_phi + hash_noise(seed, 40.0, 50.0, 60.0) * sector_width * 0.4;
        // Elevation: spread between -60 and +60 degrees, jittered per comet
        let base_elev = -1.0 + 2.0 * (sector as f32 + 0.5) / COMET_COUNT as f32;
        let elev = (base_elev + hash_noise(seed, 70.0, 80.0, 90.0) * 0.3).clamp(-0.9, 0.9);
        let cos_elev = (1.0 - elev * elev).sqrt();
        let r = COMET_MIN_RADIUS
            + hash_noise(seed, 100.0, 110.0, 120.0).abs() * (COMET_MAX_RADIUS - COMET_MIN_RADIUS);
        let pos = [r * cos_elev * phi.cos(), r * elev, r * cos_elev * phi.sin()];

        // Direction: use widely-spaced hash inputs to avoid correlation.
        // Each comet gets a unique drift direction.
        let dir_seed = seed.wrapping_mul(2654435761); // Knuth multiplicative hash
        let vx = hash_noise(dir_seed, 200.0, 210.0, 220.0);
        let vy = hash_noise(dir_seed, 230.0, 240.0, 250.0) * 0.3;
        let vz = hash_noise(dir_seed, 260.0, 270.0, 280.0);
        let dir = normalize([vx, vy, vz]);
        let speed = COMET_SPEED * (0.5 + hash_noise(dir_seed, 290.0, 300.0, 310.0).abs());

        // Orient the comet so its tail (+Z local) points opposite to velocity.
        // We compute a quaternion that rotates +Z to -dir.
        let tail_dir = [-dir[0], -dir[1], -dir[2]];
        let orient = rotation_from_z_to(tail_dir);

        let transform = blade_engine::Transform {
            position: mint::Vector3 {
                x: pos[0],
                y: pos[1],
                z: pos[2],
            },
            orientation: mint::Quaternion {
                s: orient[3],
                v: mint::Vector3 {
                    x: orient[0],
                    y: orient[1],
                    z: orient[2],
                },
            },
        };

        let handle = engine.add_object_with_model(
            "comet",
            model,
            transform,
            blade_engine::DynamicInput::Full,
        );

        engine.set_velocity(
            handle,
            mint::Vector3 {
                x: dir[0] * speed,
                y: dir[1] * speed,
                z: dir[2] * speed,
            },
            mint::Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        );

        self.comets.push(Comet {
            object_handle: handle,
        });
    }

    fn update(&mut self, engine: &mut blade_engine::Engine) {
        let mut i = 0;
        while i < self.comets.len() {
            let pos = engine.get_object_position(self.comets[i].object_handle);
            let dist = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();
            if dist > COMET_DESPAWN_RADIUS {
                let sector = i; // reuse the same sector slot for even distribution
                let comet = self.comets.swap_remove(i);
                engine.remove_object(comet.object_handle);
                self.spawn_comet_stratified(engine, sector);
            } else {
                i += 1;
            }
        }
    }
}

/// Compute quaternion [x, y, z, w] that rotates +Z to the given direction.
fn rotation_from_z_to(dir: [f32; 3]) -> [f32; 4] {
    let from = [0.0_f32, 0.0, 1.0];
    let d = from[0] * dir[0] + from[1] * dir[1] + from[2] * dir[2];
    if d > 0.9999 {
        return [0.0, 0.0, 0.0, 1.0]; // identity
    }
    if d < -0.9999 {
        return [0.0, 1.0, 0.0, 0.0]; // 180 around Y
    }
    let axis = normalize(cross(from, dir));
    let half_angle = (d.clamp(-1.0, 1.0)).acos() * 0.5;
    let s = half_angle.sin();
    [axis[0] * s, axis[1] * s, axis[2] * s, half_angle.cos()]
}

// --- Laser beam mesh ---

/// Generate a thin hexagonal prism along -Z (aim direction) for laser rendering.
fn generate_laser_mesh(
    length: f32,
    radius: f32,
    color: [f32; 4],
) -> (Vec<blade_render::Vertex>, Vec<u32>) {
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
        // Near end (z=0) and far end (z=-length)
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
    let _ = color; // color is applied via base_color_factor on the model
    (vertices, indices)
}

// --- XR Input ---

const LASER_LENGTH: f32 = 200.0;
const LASER_HIT_RADIUS: f32 = 1.5;
const LASER_REDDENING_DURATION: f32 = 1.0;
const LASER_BEAM_RADIUS: f32 = 0.015;

struct HandState {
    trigger_pressed: bool,
    aim_pose: Option<xr::Posef>,
}

struct LaserHit {
    object_handle: blade_engine::ObjectHandle,
    timer: f32,
}

struct XrInput {
    action_set: xr::ActionSet,
    trigger_action: xr::Action<f32>,
    aim_action: xr::Action<xr::Posef>,
    left_aim_space: xr::Space,
    right_aim_space: xr::Space,
    hand_paths: [xr::Path; 2],
    hands: [HandState; 2],
    laser_hits: Vec<LaserHit>,
    laser_model: blade_asset::Handle<blade_render::Model>,
    aim_model: blade_asset::Handle<blade_render::Model>,
    laser_objects: [Option<blade_engine::ObjectHandle>; 2],
    laser_firing: [bool; 2],
}

impl XrInput {
    fn new(
        instance: &xr::Instance,
        session: &xr::Session<xr::Vulkan>,
        engine: &mut blade_engine::Engine,
    ) -> Self {
        let action_set = instance
            .create_action_set("gameplay", "Gameplay", 0)
            .unwrap();

        let trigger_action = action_set
            .create_action::<f32>(
                "trigger",
                "Trigger",
                &[
                    instance.string_to_path("/user/hand/left").unwrap(),
                    instance.string_to_path("/user/hand/right").unwrap(),
                ],
            )
            .unwrap();

        let aim_action = action_set
            .create_action::<xr::Posef>(
                "aim_pose",
                "Aim Pose",
                &[
                    instance.string_to_path("/user/hand/left").unwrap(),
                    instance.string_to_path("/user/hand/right").unwrap(),
                ],
            )
            .unwrap();

        // Suggest bindings for multiple controller profiles
        let left_trigger_path = instance
            .string_to_path("/user/hand/left/input/trigger/value")
            .unwrap();
        let right_trigger_path = instance
            .string_to_path("/user/hand/right/input/trigger/value")
            .unwrap();
        let left_aim_path = instance
            .string_to_path("/user/hand/left/input/aim/pose")
            .unwrap();
        let right_aim_path = instance
            .string_to_path("/user/hand/right/input/aim/pose")
            .unwrap();

        let trigger_aim_bindings = [
            xr::Binding::new(&trigger_action, left_trigger_path),
            xr::Binding::new(&trigger_action, right_trigger_path),
            xr::Binding::new(&aim_action, left_aim_path),
            xr::Binding::new(&aim_action, right_aim_path),
        ];

        // Try multiple interaction profiles for broad device compatibility
        let profiles = [
            "/interaction_profiles/oculus/touch_controller",
            "/interaction_profiles/meta/touch_controller_plus",
        ];
        for profile_path in profiles {
            if let Ok(profile) = instance.string_to_path(profile_path) {
                match instance.suggest_interaction_profile_bindings(profile, &trigger_aim_bindings)
                {
                    Ok(()) => info!("Bound input profile: {profile_path}"),
                    Err(e) => info!("Failed to bind {profile_path}: {e}"),
                }
            }
        }

        // Also bind KHR simple controller (uses select/click instead of trigger)
        if let (Ok(profile), Ok(left_select), Ok(right_select)) = (
            instance.string_to_path("/interaction_profiles/khr/simple_controller"),
            instance.string_to_path("/user/hand/left/input/select/click"),
            instance.string_to_path("/user/hand/right/input/select/click"),
        ) {
            let simple_bindings = [
                xr::Binding::new(&trigger_action, left_select),
                xr::Binding::new(&trigger_action, right_select),
                xr::Binding::new(&aim_action, left_aim_path),
                xr::Binding::new(&aim_action, right_aim_path),
            ];
            match instance.suggest_interaction_profile_bindings(profile, &simple_bindings) {
                Ok(()) => info!("Bound KHR simple controller profile"),
                Err(e) => info!("Failed to bind KHR simple: {e}"),
            }
        }

        session.attach_action_sets(&[&action_set]).unwrap();

        let left_hand = instance.string_to_path("/user/hand/left").unwrap();
        let right_hand = instance.string_to_path("/user/hand/right").unwrap();

        let left_aim_space = aim_action
            .create_space(session.clone(), left_hand, xr::Posef::IDENTITY)
            .unwrap();
        let right_aim_space = aim_action
            .create_space(session.clone(), right_hand, xr::Posef::IDENTITY)
            .unwrap();

        // Create laser beam models
        let (laser_verts, laser_idxs) =
            generate_laser_mesh(LASER_LENGTH, LASER_BEAM_RADIUS, [0.0, 1.0, 0.0, 1.0]);
        let laser_model = engine.create_model(
            "laser_beam",
            vec![blade_render::ProceduralGeometry {
                name: "laser_beam".to_string(),
                vertices: laser_verts,
                indices: laser_idxs,
                base_color_factor: [0.2, 1.0, 0.2, 1.0],
            }],
        );
        let (aim_verts, aim_idxs) =
            generate_laser_mesh(3.0, LASER_BEAM_RADIUS * 0.5, [0.1, 0.3, 0.6, 1.0]);
        let aim_model = engine.create_model(
            "aim_line",
            vec![blade_render::ProceduralGeometry {
                name: "aim_line".to_string(),
                vertices: aim_verts,
                indices: aim_idxs,
                base_color_factor: [0.1, 0.3, 0.6, 1.0],
            }],
        );

        XrInput {
            action_set,
            trigger_action,
            aim_action,
            left_aim_space,
            right_aim_space,
            hand_paths: [left_hand, right_hand],
            hands: [
                HandState {
                    trigger_pressed: false,
                    aim_pose: None,
                },
                HandState {
                    trigger_pressed: false,
                    aim_pose: None,
                },
            ],
            laser_hits: Vec::new(),
            laser_model,
            aim_model,
            laser_objects: [None, None],
            laser_firing: [false, false],
        }
    }

    fn sync(
        &mut self,
        engine: &mut blade_engine::Engine,
        session: &xr::Session<xr::Vulkan>,
        frame_count: u64,
    ) {
        if let Err(e) = session.sync_actions(&[xr::ActiveActionSet::new(&self.action_set)]) {
            if frame_count <= 5 || frame_count % 300 == 0 {
                info!("sync_actions failed: {e}");
            }
            return;
        }

        let aim_spaces = [&self.left_aim_space, &self.right_aim_space];
        for (i, aim_space) in aim_spaces.iter().enumerate() {
            let trigger_state = self.trigger_action.state(session, self.hand_paths[i]).ok();
            self.hands[i].trigger_pressed = trigger_state
                .map(|s| s.current_state > 0.5)
                .unwrap_or(false);

            self.hands[i].aim_pose = engine.xr_locate_space(aim_space);

            if frame_count <= 5 || frame_count % 300 == 0 {
                let trigger_val = trigger_state.map(|s| s.current_state).unwrap_or(-1.0);
                let has_pose = self.hands[i].aim_pose.is_some();
                info!("Hand {i}: trigger={trigger_val:.2}, pose={has_pose}",);
            }
        }
    }

    fn update(
        &mut self,
        engine: &mut blade_engine::Engine,
        asteroid_field: &mut AsteroidField,
        explosion_ps: usize,
        dt: f32,
    ) {
        // Update existing hit timers; remove stale hits (asteroid may have been recycled)
        let mut i = 0;
        while i < self.laser_hits.len() {
            let still_exists = asteroid_field
                .asteroids
                .iter()
                .any(|a| a.object_handle == self.laser_hits[i].object_handle);
            if !still_exists {
                self.laser_hits.swap_remove(i);
                continue;
            }
            self.laser_hits[i].timer += dt;
            if self.laser_hits[i].timer >= LASER_REDDENING_DURATION {
                // Destroy the asteroid
                let hit = self.laser_hits.swap_remove(i);
                if let Some(idx) = asteroid_field
                    .asteroids
                    .iter()
                    .position(|a| a.object_handle == hit.object_handle)
                {
                    let asteroid = asteroid_field.asteroids.swap_remove(idx);
                    let pos = engine.get_object_position(asteroid.object_handle);
                    engine.particle_burst(explosion_ps, 80, [pos.x, pos.y, pos.z]);
                    engine.remove_object(asteroid.object_handle);
                    asteroid_field.spawn_asteroid(engine, false);
                }
            } else {
                i += 1;
            }
        }

        // Check for new laser hits from each hand
        for hand in &self.hands {
            if !hand.trigger_pressed {
                continue;
            }
            let pose = match hand.aim_pose {
                Some(p) => p,
                None => continue,
            };

            // Ray origin and direction from aim pose
            let origin = [pose.position.x, pose.position.y, pose.position.z];
            let q = pose.orientation;
            // Aim pose forward is -Z in the pose's local space
            let dir = quat_rotate([q.x, q.y, q.z, q.w], [0.0, 0.0, -1.0]);

            // Test against all asteroids (simple sphere test)
            let mut closest_t = LASER_LENGTH;
            let mut closest_idx = None;
            for (idx, asteroid) in asteroid_field.asteroids.iter().enumerate() {
                // Skip already-hit asteroids
                if self
                    .laser_hits
                    .iter()
                    .any(|h| h.object_handle == asteroid.object_handle)
                {
                    continue;
                }
                let pos = engine.get_object_position(asteroid.object_handle);
                let to_asteroid = [pos.x - origin[0], pos.y - origin[1], pos.z - origin[2]];
                let along =
                    to_asteroid[0] * dir[0] + to_asteroid[1] * dir[1] + to_asteroid[2] * dir[2];
                if along < 0.0 || along > closest_t {
                    continue;
                }
                let perp_sq = (to_asteroid[0] - dir[0] * along).powi(2)
                    + (to_asteroid[1] - dir[1] * along).powi(2)
                    + (to_asteroid[2] - dir[2] * along).powi(2);
                if perp_sq < LASER_HIT_RADIUS * LASER_HIT_RADIUS {
                    closest_t = along;
                    closest_idx = Some(idx);
                }
            }

            if let Some(idx) = closest_idx {
                let handle = asteroid_field.asteroids[idx].object_handle;
                self.laser_hits.push(LaserHit {
                    object_handle: handle,
                    timer: 0.0,
                });
            }
        }
    }

    fn pose_to_transform(p: &xr::Posef) -> blade_engine::Transform {
        blade_engine::Transform {
            position: mint::Vector3 {
                x: p.position.x,
                y: p.position.y,
                z: p.position.z,
            },
            orientation: mint::Quaternion {
                s: p.orientation.w,
                v: mint::Vector3 {
                    x: p.orientation.x,
                    y: p.orientation.y,
                    z: p.orientation.z,
                },
            },
        }
    }

    fn update_laser_objects(&mut self, engine: &mut blade_engine::Engine) {
        for (i, hand) in self.hands.iter().enumerate() {
            let firing = hand.trigger_pressed;
            let has_pose = hand.aim_pose.is_some();

            // Remove if pose lost or model needs to change (trigger state changed)
            if let Some(handle) = self.laser_objects[i] {
                if !has_pose || firing != self.laser_firing[i] {
                    engine.remove_object(handle);
                    self.laser_objects[i] = None;
                }
            }

            if let Some(pose) = hand.aim_pose {
                let transform = Self::pose_to_transform(&pose);
                if let Some(handle) = self.laser_objects[i] {
                    engine.teleport_object(handle, transform);
                } else {
                    let model = if firing {
                        self.laser_model
                    } else {
                        self.aim_model
                    };
                    let handle = engine.add_object_with_model(
                        "laser",
                        model,
                        transform,
                        blade_engine::DynamicInput::Full,
                    );
                    self.laser_objects[i] = Some(handle);
                    self.laser_firing[i] = firing;
                }
            }
        }
    }
}

/// Rotate a vector by a quaternion [x, y, z, w].
fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    [
        v[0] + 2.0 * (q[3] * uv[0] + uuv[0]),
        v[1] + 2.0 * (q[3] * uv[1] + uuv[1]),
        v[2] + 2.0 * (q[3] * uv[2] + uuv[2]),
    ]
}

// --- App state ---

/// XR session visibility from the app's perspective.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Visibility {
    /// Session is synchronized but not visible (e.g. full overlay).
    Hidden,
    /// Session content is visible but app lacks input focus (e.g. Quest menu overlay).
    Visible,
    /// Session is fully focused — normal gameplay.
    Focused,
}

struct GameState {
    asteroid_field: AsteroidField,
    comet_field: CometField,
    explosion_ps: usize,
}

enum AppState {
    Idle,
    Running {
        rendered_frames: u64,
        game: GameState,
        visibility: Visibility,
    },
}

fn setup_game(engine: &mut blade_engine::Engine) -> GameState {
    // Ensure zero gravity for space
    engine.set_gravity(0.0);

    // Configure space lighting — no env map needed, space_sky + OpaqueBlack clear gives black sky
    engine.set_raster_config(blade_render::RasterConfig {
        clear_color: blade_graphics::TextureColor::OpaqueBlack,
        light_dir: mint::Vector3 {
            x: -0.5,
            y: 0.7,
            z: 0.5,
        },
        light_color: mint::Vector3 {
            x: 4.0,
            y: 3.8,
            z: 3.5,
        },
        ambient_color: mint::Vector3 {
            x: 0.02,
            y: 0.02,
            z: 0.03,
        },
        roughness: 0.7,
        metallic: 0.0,
        space_sky: true,
    });

    // Place a large planet in the distance
    {
        let planet_radius = 30.0;
        let (verts, idxs) = generate_asteroid_mesh(999, planet_radius, 0.02, 3, [1.0, 0.95, 1.0]);
        let planet_model = engine.create_model(
            "planet",
            vec![blade_render::ProceduralGeometry {
                name: "planet".to_string(),
                vertices: verts,
                indices: idxs,
                base_color_factor: [0.3, 0.35, 0.5, 1.0],
            }],
        );
        let planet_transform = blade_engine::Transform {
            position: mint::Vector3 {
                x: 30.0,
                y: -20.0,
                z: -80.0,
            },
            ..Default::default()
        };
        engine.add_object_with_model(
            "planet",
            planet_model,
            planet_transform,
            blade_engine::DynamicInput::Empty,
        );
    }

    let asteroid_field = AsteroidField::new(engine);
    let comet_field = CometField::new(engine);

    let explosion_effect = blade_particle::ParticleEffect {
        capacity: 2000,
        emitter: blade_particle::Emitter {
            rate: 0.0,
            burst_count: 0,
            shape: blade_particle::EmitterShape::Sphere { radius: 0.3 },
            cone_angle: std::f32::consts::PI,
        },
        particle: blade_particle::ParticleConfig {
            life: [0.3, 1.0],
            speed: [3.0, 12.0],
            scale: [0.03, 0.1],
            color: blade_particle::ColorConfig::Palette(vec![
                [255, 200, 50, 255],
                [255, 120, 20, 255],
                [200, 60, 10, 255],
            ]),
        },
    };
    let explosion_ps = engine.create_particle_system("explosions", &explosion_effect);

    GameState {
        asteroid_field,
        comet_field,
        explosion_ps,
    }
}

fn prepare_shader_dir(app: &AndroidApp) -> PathBuf {
    let internal_path = app
        .internal_data_path()
        .expect("No internal data path available");
    let shader_dir = internal_path.join("blade-render").join("code");
    fs::create_dir_all(&shader_dir).unwrap();
    copy_assets_to_dir(app, "", &shader_dir);

    shader_dir
}

fn copy_assets_to_dir(app: &AndroidApp, asset_dir_name: &str, output_dir: &Path) {
    let asset_manager = app.asset_manager();
    let dir_name = CString::new(asset_dir_name).unwrap();
    let mut asset_dir = asset_manager
        .open_dir(&dir_name)
        .unwrap_or_else(|| panic!("Unable to open asset dir '{asset_dir_name}'"));

    let mut copied = 0usize;
    for file_name in &mut asset_dir {
        let file_name = file_name.to_string_lossy();
        if !file_name.ends_with(".wgsl") {
            continue;
        }
        let asset_path = if asset_dir_name.is_empty() {
            file_name.to_string()
        } else {
            format!("{asset_dir_name}/{file_name}")
        };
        let asset_path_c = CString::new(asset_path.as_str()).unwrap();
        let mut asset = asset_manager
            .open(&asset_path_c)
            .unwrap_or_else(|| panic!("Unable to open asset '{asset_path}'"));
        let mut contents = Vec::with_capacity(asset.length());
        asset.read_to_end(&mut contents).unwrap();
        fs::write(output_dir.join(file_name.as_ref()), contents).unwrap();
        copied += 1;
    }
    assert!(
        copied > 0,
        "No WGSL files copied from asset dir '{asset_dir_name}'"
    );
}

#[no_mangle]
fn android_main(app: AndroidApp) {
    let xr_debug = std::env::var_os("BLADE_XR_DEBUG").is_some();

    macro_rules! mark {
        ($($arg:tt)*) => {{
            if xr_debug {
                let msg = format!($($arg)*);
                info!("{}", msg);
                eprintln!("{}", msg);
            }
        }};
    }

    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("blade-asteroids"),
    );
    mark!("XR mark: logger initialized");

    let entry = unsafe {
        xr::Entry::load()
            .expect("couldn't find the OpenXR loader; try enabling the \"static\" feature")
    };
    mark!("XR mark: OpenXR entry loaded");
    entry.initialize_android_loader().unwrap();
    mark!("XR mark: Android loader initialized");

    let available_extensions = entry.enumerate_extensions().unwrap();
    assert!(available_extensions.khr_vulkan_enable2);

    let mut enabled_extensions = xr::ExtensionSet::default();
    enabled_extensions.khr_vulkan_enable2 = true;
    enabled_extensions.khr_android_create_instance = true;
    if available_extensions.meta_touch_controller_plus {
        enabled_extensions.meta_touch_controller_plus = true;
        info!("Enabling XR_META_touch_controller_plus");
    }

    let xr_instance = entry
        .create_instance(
            &xr::ApplicationInfo {
                application_name: "Blade Asteroids",
                application_version: 0,
                engine_name: "Blade",
                engine_version: 0,
                api_version: xr::Version::new(1, 0, 0),
            },
            &enabled_extensions,
            &[],
        )
        .unwrap();
    mark!("XR mark: OpenXR instance created");

    let system = xr_instance
        .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
        .unwrap();
    mark!("XR mark: OpenXR system acquired");
    let shader_dir = prepare_shader_dir(&app);
    mark!("XR mark: shader dir prepared");
    let internal_path = app
        .internal_data_path()
        .expect("No internal data path available");
    let cache_path = internal_path.join("asset-cache");
    let mut engine = blade_engine::Engine::new(
        blade_engine::Presentation::Xr(gpu::XrDesc {
            instance: xr_instance.clone(),
            system_id: system,
        }),
        &blade_engine::config::Engine {
            shader_path: shader_dir.to_string_lossy().into_owned(),
            data_path: String::new(),
            cache_path: cache_path.to_string_lossy().into_owned(),
            time_step: 0.01,
            render_backend: blade_engine::config::RenderBackend::Rasterizer,
            gui_enabled: false,
        },
    );
    mark!("XR mark: engine created");

    let mut state = AppState::Idle;
    let mut xr_input: Option<XrInput> = None;
    let mut resumed = false;
    let mut destroy_requested = false;
    let mut event_storage = xr::EventDataBuffer::new();
    let mut last_heartbeat = Instant::now();
    let mut frame_start;
    let mut frame_time_sum = 0.0f64;
    let mut frame_time_max = 0.0f64;
    let mut frame_count_in_period = 0u32;
    mark!("XR mark: app initialized; entering main loop");

    'main_loop: loop {
        // Poll Android events. When not resumed and session not running,
        // use a longer timeout to avoid busy-waiting (hello_xr pattern).
        // android-activity handles input queue lifecycle internally,
        // preventing ANR from blocked main thread callbacks.
        let session_running = matches!(state, AppState::Running { .. });
        let timeout = if !resumed && !session_running {
            Some(Duration::from_millis(250))
        } else {
            Some(Duration::ZERO)
        };
        app.poll_events(timeout, |event| {
            match event {
                PollEvent::Main(main_event) => {
                    if xr_debug {
                        info!("Android event: {:?}", main_event);
                    }
                    match main_event {
                        MainEvent::Resume { .. } => resumed = true,
                        MainEvent::Pause => resumed = false,
                        MainEvent::Destroy => destroy_requested = true,
                        MainEvent::InputAvailable => {
                            // Drain input events to keep Android responsive.
                            if let Ok(mut iter) = app.input_events_iter() {
                                loop {
                                    if !iter.next(|_event| InputStatus::Unhandled) {
                                        break;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        });
        if destroy_requested {
            break 'main_loop;
        }

        if last_heartbeat.elapsed() >= Duration::from_secs(1) {
            if xr_debug {
                let (session_running, rendered_frames, asteroid_count) = match state {
                    AppState::Idle => (false, 0, 0),
                    AppState::Running {
                        rendered_frames,
                        ref game,
                        ..
                    } => (true, rendered_frames, game.asteroid_field.asteroids.len()),
                };
                let avg_ms = if frame_count_in_period > 0 {
                    frame_time_sum / frame_count_in_period as f64 * 1000.0
                } else {
                    0.0
                };
                info!(
                    "XR heartbeat: running={}, frames={}, avg={:.1}ms, max={:.1}ms, asteroids={}",
                    session_running,
                    rendered_frames,
                    avg_ms,
                    frame_time_max * 1000.0,
                    asteroid_count,
                );
            }
            frame_time_sum = 0.0;
            frame_time_max = 0.0;
            frame_count_in_period = 0;
            last_heartbeat = Instant::now();
        }

        while let Some(event) = xr_instance.poll_event(&mut event_storage).unwrap() {
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => {
                    if xr_debug {
                        info!("XR session state changed: {:?}", e.state());
                    }
                    match e.state() {
                        xr::SessionState::READY => {
                            if xr_debug {
                                info!("XR state READY -> begin session");
                            }
                            mark!("XR mark: setting up game");
                            let game = setup_game(&mut engine);
                            if xr_input.is_none() {
                                let session = engine.xr_session().expect("XR session required");
                                xr_input = Some(XrInput::new(&xr_instance, &session, &mut engine));
                            }
                            mark!("XR mark: game setup done");
                            state = AppState::Running {
                                rendered_frames: 0,
                                game,
                                visibility: Visibility::Hidden,
                            };
                            mark!("XR mark: calling engine.begin_xr");
                            engine.begin_xr();
                            mark!("XR mark: engine.begin_xr returned");
                        }
                        xr::SessionState::FOCUSED => {
                            if let AppState::Running {
                                ref mut visibility, ..
                            } = state
                            {
                                *visibility = Visibility::Focused;
                            }
                        }
                        xr::SessionState::VISIBLE => {
                            if let AppState::Running {
                                ref mut visibility, ..
                            } = state
                            {
                                *visibility = Visibility::Visible;
                            }
                        }
                        xr::SessionState::SYNCHRONIZED => {
                            if let AppState::Running {
                                ref mut visibility, ..
                            } = state
                            {
                                *visibility = Visibility::Hidden;
                            }
                        }
                        xr::SessionState::STOPPING => {
                            if xr_debug {
                                info!("XR state STOPPING -> end session");
                            }
                            // Clean up laser objects before ending
                            if let Some(ref mut input) = xr_input {
                                for obj in input.laser_objects.iter_mut() {
                                    if let Some(handle) = obj.take() {
                                        engine.remove_object(handle);
                                    }
                                }
                            }
                            engine.end_xr();
                            state = AppState::Idle;
                        }
                        xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                            if xr_debug {
                                info!("XR state {:?} -> exiting main loop", e.state());
                            }
                            break 'main_loop;
                        }
                        _ => {}
                    }
                }
                InstanceLossPending(_) => {
                    if xr_debug {
                        info!("XR instance loss pending -> exiting main loop");
                    }
                    break 'main_loop;
                }
                _ => {}
            }
        }

        match state {
            AppState::Idle => {
                // Session not running — throttle handled by poll_events timeout above.
            }
            AppState::Running {
                ref mut rendered_frames,
                ref mut game,
                visibility,
            } => {
                if visibility != Visibility::Focused {
                    // Not focused (menu overlay or app hidden).
                    // Submit frames to keep the runtime happy, but skip physics.
                    if !engine.render_xr() {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    continue;
                }
                frame_start = Instant::now();
                if let (Some(ref mut input), Some(session)) = (&mut xr_input, engine.xr_session()) {
                    input.sync(&mut engine, &session, *rendered_frames);
                    input.update(
                        &mut engine,
                        &mut game.asteroid_field,
                        game.explosion_ps,
                        0.016,
                    );
                    input.update_laser_objects(&mut engine);
                }
                game.asteroid_field.update(&mut engine);
                game.comet_field.update(&mut engine);
                engine.update(0.016);
                if engine.render_xr() {
                    let dt = frame_start.elapsed().as_secs_f64();
                    frame_time_sum += dt;
                    if dt > frame_time_max {
                        frame_time_max = dt;
                    }
                    frame_count_in_period += 1;
                    *rendered_frames += 1;
                    if xr_debug && (*rendered_frames <= 5 || *rendered_frames % 120 == 0) {
                        info!("XR frame submitted: {}", *rendered_frames);
                    }
                }
            }
        }
    }

    if matches!(state, AppState::Running { .. }) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.end_xr()));
    }
    engine.destroy();
}
