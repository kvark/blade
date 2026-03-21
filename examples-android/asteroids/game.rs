#![cfg(target_os = "android")]

use crate::mesh;

// --- Asteroid field parameters ---

const ASTEROID_COUNT: usize = 500;
const FIELD_HALF_WIDTH: f32 = 40.0;
const FIELD_HALF_HEIGHT: f32 = 20.0;
const SPAWN_DISTANCE: f32 = 120.0;
const DESPAWN_BEHIND: f32 = 30.0;
const MIN_SPAWN_DISTANCE: f32 = 15.0;
const FLOW_SPEED: f32 = 4.0;
const SPEED_VARIATION: f32 = 0.4;

// --- Asteroid health ---
const MAX_HEALTH: f32 = 1.0;
pub const LASER_DAMAGE: f32 = 0.016;
const HEALTH_REGEN: f32 = 0.2;
const DAMAGE_STAGES: usize = 5;

// --- Asteroid sizes ---
pub const SIZE_RADII: [f32; 3] = [0.5, 1.0, 2.0];
const SPLIT_COUNTS: [usize; 3] = [0, 2, 4];

// --- Comet parameters ---
const COMET_COUNT: usize = 20;
const COMET_MIN_RADIUS: f32 = 200.0;
const COMET_MAX_RADIUS: f32 = 600.0;
const COMET_DESPAWN_RADIUS: f32 = 800.0;
const COMET_SPEED: f32 = 3.0;
const COMET_NUCLEUS_RADIUS: f32 = 3.0;

// --- Asteroid ---

pub struct Asteroid {
    pub object_handle: blade_engine::ObjectHandle,
    variant: usize,
    pub size_class: usize,
    pub health: f32,
    color_stage: usize,
}

pub struct AsteroidField {
    pub asteroids: Vec<Asteroid>,
    /// model_handles[size_class][variant][color_stage]
    model_handles: Vec<Vec<Vec<blade_asset::Handle<blade_render::Model>>>>,
    flow_dir: [f32; 3],
    next_seed: u32,
}

impl AsteroidField {
    pub fn new(engine: &mut blade_engine::Engine) -> Self {
        let variant_params: &[(f32, [f32; 3], [f32; 4])] = &[
            (0.35, [1.0, 1.0, 1.0], [0.45, 0.40, 0.35, 1.0]),
            (0.50, [1.4, 0.7, 1.0], [0.50, 0.45, 0.38, 1.0]),
            (0.45, [1.0, 0.5, 1.2], [0.40, 0.38, 0.35, 1.0]),
            (0.60, [1.0, 1.0, 1.0], [0.35, 0.32, 0.30, 1.0]),
            (0.30, [0.8, 1.3, 0.9], [0.55, 0.50, 0.42, 1.0]),
            (0.55, [1.3, 0.8, 1.3], [0.42, 0.38, 0.32, 1.0]),
            (0.40, [1.1, 1.1, 0.6], [0.48, 0.44, 0.38, 1.0]),
            (0.65, [0.9, 0.9, 1.4], [0.38, 0.35, 0.30, 1.0]),
        ];
        let hot_color: [f32; 4] = [0.9, 0.15, 0.05, 1.0];
        let mut model_handles = Vec::new();
        for (sc, &radius) in SIZE_RADII.iter().enumerate() {
            let mut variants = Vec::new();
            for (i, &(roughness, axis_scales, color)) in variant_params.iter().enumerate() {
                let seed = (sc * 100 + i * 7 + 42) as u32;
                let (vertices, indices) =
                    mesh::generate_asteroid_mesh(seed, radius, roughness, 2, axis_scales);
                let mut stages = Vec::new();
                for stage in 0..DAMAGE_STAGES {
                    let t = stage as f32 / (DAMAGE_STAGES - 1) as f32;
                    let staged_color = [
                        color[0] + (hot_color[0] - color[0]) * t,
                        color[1] + (hot_color[1] - color[1]) * t,
                        color[2] + (hot_color[2] - color[2]) * t,
                        1.0,
                    ];
                    let handle = engine.create_model(
                        &format!("asteroid_s{sc}_v{i}_t{stage}"),
                        vec![blade_render::ProceduralGeometry {
                            name: format!("asteroid_s{sc}_v{i}_t{stage}"),
                            vertices: vertices.clone(),
                            indices: indices.clone(),
                            base_color_factor: staged_color,
                        }],
                    );
                    stages.push(handle);
                }
                variants.push(stages);
            }
            model_handles.push(variants);
        }

        let flow_dir = [0.0, 0.0, 1.0];

        let mut field = AsteroidField {
            asteroids: Vec::with_capacity(ASTEROID_COUNT),
            model_handles,
            flow_dir,
            next_seed: 100,
        };

        for _ in 0..ASTEROID_COUNT {
            field.spawn_asteroid(engine, true, None);
        }

        field
    }

    fn random_size_class(seed: u32) -> usize {
        let r = mesh::hash_noise(seed, 50.0, 51.0, 52.0).abs();
        if r < 0.5 {
            0
        } else if r < 0.85 {
            1
        } else {
            2
        }
    }

    fn num_variants(&self) -> usize {
        self.model_handles[0].len()
    }

    fn spawn_asteroid(
        &mut self,
        engine: &mut blade_engine::Engine,
        spread: bool,
        size_override: Option<usize>,
    ) {
        let seed = self.next_seed;
        self.next_seed += 1;

        let size_class = size_override.unwrap_or_else(|| Self::random_size_class(seed));
        let variant = (seed as usize) % self.num_variants();

        let x = mesh::hash_noise(seed, 0.1, 0.2, 0.3) * FIELD_HALF_WIDTH;
        let y = mesh::hash_noise(seed, 0.4, 0.5, 0.6) * FIELD_HALF_HEIGHT;
        let z_depth = if spread {
            let t = mesh::hash_noise(seed, 0.7, 0.8, 0.9).abs();
            -(MIN_SPAWN_DISTANCE + t * (SPAWN_DISTANCE - MIN_SPAWN_DISTANCE))
        } else {
            -SPAWN_DISTANCE + mesh::hash_noise(seed, 0.7, 0.8, 0.9).abs() * 5.0
        };

        let pos = [x, y, z_depth];

        let speed = FLOW_SPEED * (1.0 + mesh::hash_noise(seed, 10.0, 11.0, 12.0) * SPEED_VARIATION);
        let velocity = [
            self.flow_dir[0] * speed + mesh::hash_noise(seed, 13.0, 14.0, 15.0) * 0.3,
            self.flow_dir[1] * speed + mesh::hash_noise(seed, 16.0, 17.0, 18.0) * 0.2,
            self.flow_dir[2] * speed + mesh::hash_noise(seed, 19.0, 20.0, 21.0) * 0.3,
        ];

        let angular_velocity = [
            mesh::hash_noise(seed, 30.0, 31.0, 32.0) * 0.5,
            mesh::hash_noise(seed, 33.0, 34.0, 35.0) * 0.5,
            mesh::hash_noise(seed, 36.0, 37.0, 38.0) * 0.5,
        ];

        self.add_asteroid(engine, size_class, variant, pos, velocity, angular_velocity);
    }

    fn spawn_fragments(
        &mut self,
        engine: &mut blade_engine::Engine,
        parent_pos: [f32; 3],
        parent_vel: [f32; 3],
        child_size_class: usize,
        count: usize,
    ) {
        for _ in 0..count {
            let seed = self.next_seed;
            self.next_seed += 1;
            let variant = (seed as usize) % self.num_variants();

            let theta = mesh::hash_noise(seed, 60.0, 61.0, 62.0) * std::f32::consts::TAU;
            let cos_phi = mesh::hash_noise(seed, 63.0, 64.0, 65.0);
            let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
            let frag_dir = [sin_phi * theta.cos(), sin_phi * theta.sin(), cos_phi];

            let frag_speed = 3.0 + mesh::hash_noise(seed, 66.0, 67.0, 68.0).abs() * 5.0;
            let velocity = [
                parent_vel[0] + frag_dir[0] * frag_speed,
                parent_vel[1] + frag_dir[1] * frag_speed,
                parent_vel[2] + frag_dir[2] * frag_speed,
            ];

            let offset = SIZE_RADII[child_size_class] * 1.5;
            let pos = [
                parent_pos[0] + frag_dir[0] * offset,
                parent_pos[1] + frag_dir[1] * offset,
                parent_pos[2] + frag_dir[2] * offset,
            ];

            let angular_velocity = [
                mesh::hash_noise(seed, 70.0, 71.0, 72.0) * 2.0,
                mesh::hash_noise(seed, 73.0, 74.0, 75.0) * 2.0,
                mesh::hash_noise(seed, 76.0, 77.0, 78.0) * 2.0,
            ];

            self.add_asteroid(
                engine,
                child_size_class,
                variant,
                pos,
                velocity,
                angular_velocity,
            );
        }
    }

    fn add_asteroid(
        &mut self,
        engine: &mut blade_engine::Engine,
        size_class: usize,
        variant: usize,
        pos: [f32; 3],
        velocity: [f32; 3],
        angular_velocity: [f32; 3],
    ) {
        let color_stage = 0;
        let axis = mesh::normalize(angular_velocity);
        let angle = mesh::hash_noise(self.next_seed.wrapping_add(999), 80.0, 81.0, 82.0)
            * std::f32::consts::TAU;
        let half = angle * 0.5;
        let s = half.sin();
        let transform = blade_engine::Transform {
            position: mint::Vector3 {
                x: pos[0],
                y: pos[1],
                z: pos[2],
            },
            orientation: mint::Quaternion {
                s: half.cos(),
                v: mint::Vector3 {
                    x: axis[0] * s,
                    y: axis[1] * s,
                    z: axis[2] * s,
                },
            },
        };

        let handle = engine.add_object_with_model(
            "asteroid",
            self.model_handles[size_class][variant][color_stage],
            transform,
            blade_engine::DynamicInput::Full,
        );

        engine.add_ball_collider(handle, SIZE_RADII[size_class], 0.5);
        // Set velocity AFTER collider so the rigid body has proper inertia
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
            variant,
            size_class,
            health: MAX_HEALTH,
            color_stage,
        });
    }

    pub fn update(&mut self, engine: &mut blade_engine::Engine, explosion_ps: usize, dt: f32) {
        // Recycle asteroids that have passed behind the player
        let mut i = 0;
        while i < self.asteroids.len() {
            let pos = engine.get_object_position(self.asteroids[i].object_handle);
            let along_flow =
                pos.x * self.flow_dir[0] + pos.y * self.flow_dir[1] + pos.z * self.flow_dir[2];
            if along_flow > DESPAWN_BEHIND {
                let asteroid = self.asteroids.swap_remove(i);
                engine.remove_object(asteroid.object_handle);
                self.spawn_asteroid(engine, false, None);
            } else {
                i += 1;
            }
        }

        // Update health and check for explosions
        let mut i = 0;
        while i < self.asteroids.len() {
            let a = &mut self.asteroids[i];

            // Regenerate health over time
            a.health = (a.health + HEALTH_REGEN * dt).min(MAX_HEALTH);

            // Check if asteroid explodes
            if a.health <= 0.0 {
                let pos = engine.get_object_position(a.object_handle);
                let (linvel, _) = engine.get_velocity(a.object_handle);
                let parent_vel = [linvel.x, linvel.y, linvel.z];
                let parent_pos = [pos.x, pos.y, pos.z];
                let size_class = a.size_class;
                let burst_count = 40 + size_class as u32 * 30;
                engine.particle_burst(explosion_ps, burst_count, parent_pos);
                engine.remove_object(a.object_handle);
                self.asteroids.swap_remove(i);

                // Split into smaller asteroids if this was medium or large
                let split_count = SPLIT_COUNTS[size_class];
                if split_count > 0 {
                    self.spawn_fragments(
                        engine,
                        parent_pos,
                        parent_vel,
                        size_class - 1,
                        split_count,
                    );
                }
                self.spawn_asteroid(engine, false, None);
                continue;
            }

            // Update color stage based on damage
            let damage = 1.0 - a.health / MAX_HEALTH;
            let new_stage = (damage * (DAMAGE_STAGES - 1) as f32) as usize;
            let new_stage = new_stage.min(DAMAGE_STAGES - 1);
            if new_stage != a.color_stage {
                let transform = engine
                    .get_object_transform(a.object_handle, blade_engine::Prediction::LastKnown);
                let (linvel, angvel) = engine.get_velocity(a.object_handle);
                engine.remove_object(a.object_handle);
                let new_handle = engine.add_object_with_model(
                    "asteroid",
                    self.model_handles[a.size_class][a.variant][new_stage],
                    transform,
                    blade_engine::DynamicInput::Full,
                );
                engine.add_ball_collider(new_handle, SIZE_RADII[a.size_class], 0.5);
                engine.set_velocity(new_handle, linvel, angvel);
                a.object_handle = new_handle;
                a.color_stage = new_stage;
            }
            i += 1;
        }
    }
}

// --- Comets ---

struct Comet {
    object_handle: blade_engine::ObjectHandle,
    trail_ps: usize,
    velocity_dir: [f32; 3],
}

fn comet_trail_effect() -> blade_particle::ParticleEffect {
    blade_particle::ParticleEffect {
        capacity: 2000,
        emitter: blade_particle::Emitter {
            rate: 200.0,
            burst_count: 0,
            shape: blade_particle::EmitterShape::Sphere { radius: 1.0 },
            cone_angle: 0.4,
        },
        particle: blade_particle::ParticleConfig {
            life: [1.0, 4.0],
            speed: [2.0, 8.0],
            scale: [1.0, 4.0],
            color: blade_particle::ColorConfig::Palette(vec![
                [180, 200, 255, 255],
                [100, 150, 255, 200],
                [60, 100, 220, 150],
            ]),
        },
    }
}

pub struct CometField {
    comets: Vec<Comet>,
    next_seed: u32,
}

impl CometField {
    pub fn new(engine: &mut blade_engine::Engine) -> Self {
        let mut field = CometField {
            comets: Vec::with_capacity(COMET_COUNT),
            next_seed: 50000,
        };
        for i in 0..COMET_COUNT {
            field.spawn_comet_stratified(engine, i);
        }
        field
    }

    fn spawn_comet_stratified(&mut self, engine: &mut blade_engine::Engine, sector: usize) {
        let seed = self.next_seed;
        self.next_seed += 1;

        let model = mesh::generate_comet_model(seed, engine, COMET_NUCLEUS_RADIUS);

        let sector_width = std::f32::consts::TAU / COMET_COUNT as f32;
        let base_phi = sector as f32 * sector_width;
        let phi = base_phi + mesh::hash_noise(seed, 40.0, 50.0, 60.0) * sector_width * 0.4;
        let base_elev = -1.0 + 2.0 * (sector as f32 + 0.5) / COMET_COUNT as f32;
        let elev = (base_elev + mesh::hash_noise(seed, 70.0, 80.0, 90.0) * 0.3).clamp(-0.9, 0.9);
        let cos_elev = (1.0 - elev * elev).sqrt();
        let r = COMET_MIN_RADIUS
            + mesh::hash_noise(seed, 100.0, 110.0, 120.0).abs()
                * (COMET_MAX_RADIUS - COMET_MIN_RADIUS);
        let pos = [r * cos_elev * phi.cos(), r * elev, r * cos_elev * phi.sin()];

        // Direction: perpendicular to the radial vector
        let radial = mesh::normalize(pos);
        let dir_seed = seed.wrapping_mul(2654435761);
        let rand_vec = mesh::normalize([
            mesh::hash_noise(dir_seed, 200.0, 210.0, 220.0),
            mesh::hash_noise(dir_seed, 230.0, 240.0, 250.0),
            mesh::hash_noise(dir_seed, 260.0, 270.0, 280.0),
        ]);
        let tangent = mesh::cross(radial, rand_vec);
        let dir = mesh::normalize(tangent);
        let speed = COMET_SPEED * (0.5 + mesh::hash_noise(dir_seed, 290.0, 300.0, 310.0).abs());

        let tail_dir = [-dir[0], -dir[1], -dir[2]];
        let orient = mesh::rotation_from_z_to(tail_dir);

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

        let trail_ps = engine.create_particle_system("comet_trail", &comet_trail_effect());
        if let Some(ps) = engine.particle_system_mut(trail_ps) {
            ps.origin = pos;
            ps.axis = tail_dir;
        }

        self.comets.push(Comet {
            object_handle: handle,
            trail_ps,
            velocity_dir: dir,
        });
    }

    pub fn update(&mut self, engine: &mut blade_engine::Engine) {
        let mut i = 0;
        while i < self.comets.len() {
            let pos = engine.get_object_position(self.comets[i].object_handle);
            let dist = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();
            if dist > COMET_DESPAWN_RADIUS {
                let sector = i;
                let comet = self.comets.swap_remove(i);
                engine.remove_particle_system(comet.trail_ps);
                engine.remove_object(comet.object_handle);
                self.spawn_comet_stratified(engine, sector);
            } else {
                let dir = self.comets[i].velocity_dir;
                if let Some(ps) = engine.particle_system_mut(self.comets[i].trail_ps) {
                    ps.origin = [pos.x, pos.y, pos.z];
                    ps.axis = [-dir[0], -dir[1], -dir[2]];
                }
                i += 1;
            }
        }
    }
}

// --- Game state ---

pub struct GameState {
    pub asteroid_field: AsteroidField,
    pub comet_field: CometField,
    pub explosion_ps: usize,
    pub collision_ps: usize,
    pub laser_hit_ps: usize,
}

pub fn setup_game(engine: &mut blade_engine::Engine) -> GameState {
    engine.set_gravity(0.0);

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

    // Planet
    {
        let planet_radius = 30.0;
        let planet_model = mesh::generate_planet_model(engine, planet_radius);
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

    let collision_effect = blade_particle::ParticleEffect {
        capacity: 1000,
        emitter: blade_particle::Emitter {
            rate: 0.0,
            burst_count: 0,
            shape: blade_particle::EmitterShape::Sphere { radius: 0.2 },
            cone_angle: std::f32::consts::PI,
        },
        particle: blade_particle::ParticleConfig {
            life: [0.2, 0.5],
            speed: [2.0, 8.0],
            scale: [0.02, 0.05],
            color: blade_particle::ColorConfig::Palette(vec![
                [255, 220, 150, 255],
                [200, 180, 120, 255],
            ]),
        },
    };
    let collision_ps = engine.create_particle_system("collisions", &collision_effect);

    let laser_hit_effect = blade_particle::ParticleEffect {
        capacity: 500,
        emitter: blade_particle::Emitter {
            rate: 0.0,
            burst_count: 0,
            shape: blade_particle::EmitterShape::Point,
            cone_angle: std::f32::consts::FRAC_PI_2,
        },
        particle: blade_particle::ParticleConfig {
            life: [0.1, 0.3],
            speed: [2.0, 6.0],
            scale: [0.01, 0.04],
            color: blade_particle::ColorConfig::Palette(vec![
                [100, 255, 100, 255],
                [200, 255, 150, 255],
            ]),
        },
    };
    let laser_hit_ps = engine.create_particle_system("laser_hits", &laser_hit_effect);

    GameState {
        asteroid_field,
        comet_field,
        explosion_ps,
        collision_ps,
        laser_hit_ps,
    }
}
