//! A grid of spheres exercising the PBR material model.
//!
//! The columns vary the roughness, the rows vary the metalness,
//! and the last row is emissive. Both the rasterizer and the ray tracer
//! render it, so their results can be compared side by side.
#![cfg(not(gles))]
#![allow(dead_code)]

use std::f32::consts::PI;

pub const COLUMNS: usize = 5;
/// 3 rows of metalness values, plus one emissive row.
pub const ROWS: usize = 4;
const SPACING: f32 = 1.5;
const RADIUS: f32 = 0.5;
const SEGMENTS: usize = 32;
const RINGS: usize = 16;
/// Gold-ish, so that the metals are clearly tinted.
const BASE_COLOR: [f32; 4] = [0.95, 0.72, 0.35, 1.0];
const METALNESS_ROW: [f32; 3] = [0.0, 0.5, 1.0];
const EMISSIVE_COLORS: [[f32; 3]; COLUMNS] = [
    [1.0, 0.15, 0.1],
    [0.15, 1.0, 0.2],
    [0.1, 0.3, 1.0],
    [1.0, 0.9, 0.5],
    [0.3, 0.3, 0.3],
];

fn encode_normal(v: [f32; 3]) -> u32 {
    let quantize = |f: f32| ((f.clamp(-1.0, 1.0) * 127.0 + 0.5) as i8) as u8 as u32;
    quantize(v[0]) | (quantize(v[1]) << 8) | (quantize(v[2]) << 16)
}

/// Produce a UV sphere with normals and tangents.
fn sphere(center: [f32; 3], radius: f32) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity((SEGMENTS + 1) * (RINGS + 1));
    for ring in 0..=RINGS {
        let theta = PI * ring as f32 / RINGS as f32;
        let (sin_theta, cos_theta) = theta.sin_cos();
        for segment in 0..=SEGMENTS {
            let phi = 2.0 * PI * segment as f32 / SEGMENTS as f32;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let normal = [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi];
            vertices.push(blade_render::Vertex {
                position: [
                    center[0] + radius * normal[0],
                    center[1] + radius * normal[1],
                    center[2] + radius * normal[2],
                ],
                bitangent_sign: 1.0,
                tex_coords: [segment as f32 / SEGMENTS as f32, ring as f32 / RINGS as f32],
                normal: encode_normal(normal),
                tangent: encode_normal([-sin_phi, 0.0, cos_phi]),
            });
        }
    }

    // Note: counter-clockwise when looked at from the outside,
    // so that the flat normals of the ray tracer point outwards.
    let mut indices = Vec::with_capacity(SEGMENTS * RINGS * 6);
    let stride = (SEGMENTS + 1) as u32;
    for ring in 0..RINGS as u32 {
        for segment in 0..SEGMENTS as u32 {
            let base = ring * stride + segment;
            indices.extend_from_slice(&[base, base + 1, base + stride]);
            indices.extend_from_slice(&[base + 1, base + stride + 1, base + stride]);
        }
    }

    (vertices, indices)
}

/// Build the material grid, with the roughness interpolated
/// between the ends of `roughness_range` across the columns.
pub fn material_grid(roughness_range: [f32; 2]) -> Vec<blade_render::ProceduralGeometry> {
    let mut geometries = Vec::with_capacity(COLUMNS * ROWS);
    for row in 0..ROWS {
        // the row past the metals is the emissive one
        let metalness = METALNESS_ROW.get(row).copied();
        for (column, &emissive) in EMISSIVE_COLORS.iter().enumerate() {
            let center = [
                (column as f32 - 0.5 * (COLUMNS - 1) as f32) * SPACING,
                (0.5 * (ROWS - 1) as f32 - row as f32) * SPACING,
                0.0,
            ];
            let (vertices, indices) = sphere(center, RADIUS);
            let ratio = column as f32 / (COLUMNS - 1) as f32;
            geometries.push(blade_render::ProceduralGeometry {
                name: format!("sphere[{row}][{column}]"),
                vertices,
                indices,
                base_color_factor: match metalness {
                    Some(_) => BASE_COLOR,
                    None => [0.0, 0.0, 0.0, 1.0],
                },
                metalness: metalness.unwrap_or_default(),
                roughness: roughness_range[0] + ratio * (roughness_range[1] - roughness_range[0]),
                emissive_factor: match metalness {
                    Some(_) => [0.0; 3],
                    None => emissive,
                },
            });
        }
    }
    geometries
}

/// A camera that frames the whole grid.
pub fn camera() -> blade_render::Camera {
    let fov_y = 0.8f32;
    let height = ROWS as f32 * SPACING;
    blade_render::Camera {
        pos: [0.0, 0.0, 0.5 * height / (0.5 * fov_y).tan()].into(),
        rot: mint::Quaternion {
            v: [0.0; 3].into(),
            s: 1.0,
        },
        fov_y,
        depth: 100.0,
        fov: None,
    }
}
