#![allow(
    irrefutable_let_patterns,
    clippy::new_without_default,
    clippy::needless_borrowed_reference
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    //TODO: re-enable. Currently doesn't like "mem::size_of" on newer Rust
    //unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

mod dummy;
mod env_map;
pub use dummy::DummyResources;
pub use env_map::EnvironmentMap;

mod asset_hub;
pub mod model;
pub mod raster;
pub mod shader;
mod shaders;
pub mod texture;
pub mod util;

mod render;

pub use asset_hub::*;
pub use model::{Model, ProceduralGeometry};
pub use raster::{DirectionalShadowConfig, MAX_POINT_LIGHTS, PointLight, RasterConfig, Rasterizer};
pub use shader::Shader;
pub use shaders::{RenderConfig, Shaders};
pub use texture::Texture;
pub use util::FrameResources;

pub use render::*;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebugPoint {
    pub pos: [f32; 3],
    pub color: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebugLine {
    pub a: DebugPoint,
    pub b: DebugPoint,
}

// Has to match the `Vertex` in shaders
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Vertex {
    pub position: [f32; 3],
    pub bitangent_sign: f32,
    pub tex_coords: [f32; 2],
    pub normal: u32,
    pub tangent: u32,
}

/// Asymmetric field-of-view angles (in radians).
/// All angles are positive: left/down are measured from center toward left/down.
#[derive(Clone, Copy, Debug)]
pub struct Fov {
    pub left: f32,
    pub right: f32,
    pub up: f32,
    pub down: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub pos: mint::Vector3<f32>,
    pub rot: mint::Quaternion<f32>,
    pub fov_y: f32,
    pub depth: f32,
    /// Per-eye asymmetric FOV. When set, overrides `fov_y` for projection.
    pub fov: Option<Fov>,
}

impl Camera {
    /// Shift the projection by a subpixel offset without moving the camera.
    ///
    /// `jitter` is measured in target-pixel units, with positive X right and
    /// positive Y down. Path rays, the G-buffer, and motion vectors all consume
    /// the resulting asymmetric projection.
    pub fn with_projection_jitter(mut self, jitter: [f32; 2], target_size: [u32; 2]) -> Self {
        assert!(target_size.into_iter().all(|extent| extent != 0));
        assert!(jitter.into_iter().all(f32::is_finite));
        let (extent, center) = match self.fov {
            Some(fov) => {
                let tangent = [
                    fov.left.tan(),
                    fov.right.tan(),
                    fov.up.tan(),
                    fov.down.tan(),
                ];
                (
                    [
                        0.5 * (tangent[0] + tangent[1]),
                        0.5 * (tangent[2] + tangent[3]),
                    ],
                    [
                        0.5 * (tangent[1] - tangent[0]),
                        0.5 * (tangent[2] - tangent[3]),
                    ],
                )
            }
            None => {
                let y = (0.5 * self.fov_y).tan();
                (
                    [y * target_size[0] as f32 / target_size[1] as f32, y],
                    [0.0; 2],
                )
            }
        };
        let center = [
            center[0] + 2.0 * jitter[0] * extent[0] / target_size[0] as f32,
            center[1] - 2.0 * jitter[1] * extent[1] / target_size[1] as f32,
        ];
        self.fov = Some(Fov {
            left: (extent[0] - center[0]).atan(),
            right: (extent[0] + center[0]).atan(),
            up: (extent[1] + center[1]).atan(),
            down: (extent[1] - center[1]).atan(),
        });
        self
    }
}

pub struct Object {
    pub model: blade_asset::Handle<Model>,
    pub transform: blade_graphics::Transform,
    pub prev_transform: blade_graphics::Transform,
    /// Per-object color tint multiplied with the material's base_color_factor.
    /// Default: [1.0, 1.0, 1.0, 1.0] (no tint).
    pub color_tint: [f32; 4],
}

impl From<blade_asset::Handle<Model>> for Object {
    fn from(model: blade_asset::Handle<Model>) -> Self {
        Self {
            model,
            transform: blade_graphics::IDENTITY_TRANSFORM,
            prev_transform: blade_graphics::IDENTITY_TRANSFORM,
            color_tint: [1.0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
struct CameraParams {
    position: [f32; 3],
    depth: f32,
    orientation: [f32; 4],
    fov: [f32; 2],
    film_offset: [f32; 2],
    target_size: [u32; 2],
    _pad: [u32; 2],
}

impl CameraParams {
    fn new(camera: &Camera, target_size: [u32; 2]) -> Self {
        let (fov, film_offset) = match camera.fov {
            Some(fov) => {
                let tangent = [
                    fov.left.tan(),
                    fov.right.tan(),
                    fov.up.tan(),
                    fov.down.tan(),
                ];
                let half_extent = [
                    0.5 * (tangent[0] + tangent[1]),
                    0.5 * (tangent[2] + tangent[3]),
                ];
                (
                    [2.0 * half_extent[0].atan(), 2.0 * half_extent[1].atan()],
                    [
                        0.5 * (tangent[1] - tangent[0]),
                        0.5 * (tangent[2] - tangent[3]),
                    ],
                )
            }
            None => {
                let fov_x = 2.0
                    * ((camera.fov_y * 0.5).tan() * target_size[0] as f32 / target_size[1] as f32)
                        .atan();
                ([fov_x, camera.fov_y], [0.0; 2])
            }
        };
        Self {
            position: camera.pos.into(),
            depth: camera.depth,
            orientation: camera.rot.into(),
            fov,
            film_offset,
            target_size,
            _pad: [0; 2],
        }
    }
}

#[cfg(test)]
mod camera_tests {
    use super::*;

    fn camera(fov: Option<Fov>) -> Camera {
        Camera {
            pos: [0.0; 3].into(),
            rot: mint::Quaternion::from([0.0, 0.0, 0.0, 1.0]),
            fov_y: 0.8,
            depth: 100.0,
            fov,
        }
    }

    #[test]
    fn symmetric_camera_has_no_film_offset() {
        let params = CameraParams::new(&camera(None), [1600, 900]);
        assert_eq!(params.film_offset, [0.0; 2]);
        assert_eq!(params.fov[1], 0.8);
    }

    #[test]
    fn asymmetric_camera_preserves_each_frustum_edge() {
        let fov = Fov {
            left: 0.4,
            right: 0.6,
            up: 0.5,
            down: 0.3,
        };
        let params = CameraParams::new(&camera(Some(fov)), [1600, 900]);
        let extent = [(0.5 * params.fov[0]).tan(), (0.5 * params.fov[1]).tan()];
        for (actual, expected) in [
            (params.film_offset[0] - extent[0], -fov.left.tan()),
            (params.film_offset[0] + extent[0], fov.right.tan()),
            (params.film_offset[1] + extent[1], fov.up.tan()),
            (params.film_offset[1] - extent[1], -fov.down.tan()),
        ] {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn projection_jitter_is_measured_in_target_pixels() {
        let size = [128, 64];
        let jitter = [0.25, -0.125];
        let fov = camera(None)
            .with_projection_jitter(jitter, size)
            .fov
            .unwrap();
        let tangent = [
            fov.left.tan(),
            fov.right.tan(),
            fov.up.tan(),
            fov.down.tan(),
        ];
        let extent = [
            0.5 * (tangent[0] + tangent[1]),
            0.5 * (tangent[2] + tangent[3]),
        ];
        let center = [
            0.5 * (tangent[1] - tangent[0]),
            0.5 * (tangent[2] - tangent[3]),
        ];
        let expected = [
            2.0 * jitter[0] * extent[0] / size[0] as f32,
            -2.0 * jitter[1] * extent[1] / size[1] as f32,
        ];
        assert!((center[0] - expected[0]).abs() < 1e-6);
        assert!((center[1] - expected[1]).abs() < 1e-6);
    }
}
