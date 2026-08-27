//! Regenerate `animated_skin.glb` with:
//! `rustc tests/assets/generate_animated_skin.rs -o /tmp/generate-animated-skin && /tmp/generate-animated-skin tests/assets/animated_skin.glb`

use std::{env, fs, path::PathBuf};

fn push_f32(data: &mut Vec<u8>, values: impl IntoIterator<Item = f32>) {
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_u16(data: &mut Vec<u8>, values: impl IntoIterator<Item = u16>) {
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
}

fn main() {
    let output = PathBuf::from(env::args_os().nth(1).expect("output GLB path"));
    let mut bin = Vec::new();

    let positions = [
        [-0.45, 0.0, 0.0],
        [0.45, 0.0, 0.0],
        [-0.45, 1.0, 0.0],
        [0.45, 1.0, 0.0],
        [-0.45, 2.0, 0.0],
        [0.45, 2.0, 0.0],
    ];
    for value in positions {
        push_f32(&mut bin, value);
    }
    for _ in positions {
        push_f32(&mut bin, [0.6, 0.0, 0.8]);
    }
    for _ in positions {
        push_f32(&mut bin, [0.0, 1.0, 0.0, 1.0]);
    }
    for value in [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.5],
        [1.0, 0.5],
        [0.0, 1.0],
        [1.0, 1.0],
    ] {
        push_f32(&mut bin, value);
    }
    for value in [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ] {
        push_u16(&mut bin, value);
    }
    for value in [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ] {
        push_f32(&mut bin, value);
    }
    push_u16(&mut bin, [0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5]);

    let identity = [
        1.0, 0.0, 0.0, 0.0, // column 0
        0.0, 1.0, 0.0, 0.0, // column 1
        0.0, 0.0, 1.0, 0.0, // column 2
        0.0, 0.0, 0.0, 1.0, // column 3
    ];
    let inverse_tip_bind = [
        1.0, 0.0, 0.0, 0.0, // column 0
        0.0, 1.0, 0.0, 0.0, // column 1
        0.0, 0.0, 1.0, 0.0, // column 2
        0.0, -1.0, 0.0, 1.0, // column 3
    ];
    push_f32(&mut bin, identity);
    push_f32(&mut bin, inverse_tip_bind);
    push_f32(&mut bin, [0.0, 0.5, 1.0]);

    let half_45 = 22.5f32.to_radians();
    let half_35 = 17.5f32.to_radians();
    for value in [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, half_45.sin(), half_45.cos()],
        [0.0, 0.0, -half_35.sin(), half_35.cos()],
    ] {
        push_f32(&mut bin, value);
    }
    for value in [[1.0, 1.0, 1.0], [1.6, 0.7, 0.55], [0.7, 1.4, 1.5]] {
        push_f32(&mut bin, value);
    }
    assert_eq!(bin.len(), 680);

    let json = format!(
        r#"{{
  "asset": {{"version": "2.0", "generator": "blade animated skin fixture"}},
  "scene": 0,
  "scenes": [{{"nodes": [0, 1]}}],
  "nodes": [
    {{"name": "Mesh", "mesh": 0, "skin": 0}},
    {{"name": "RootJoint", "children": [2]}},
    {{"name": "TipJoint", "translation": [0, 1, 0]}}
  ],
  "meshes": [{{"name": "SkinnedStrip", "primitives": [{{
    "attributes": {{"POSITION": 0, "NORMAL": 1, "TANGENT": 2, "TEXCOORD_0": 3, "JOINTS_0": 4, "WEIGHTS_0": 5}},
    "indices": 6,
    "material": 0
  }}]}}],
  "materials": [{{"name": "Orange", "pbrMetallicRoughness": {{
    "baseColorFactor": [0.9, 0.3, 0.08, 1],
    "metallicFactor": 0,
    "roughnessFactor": 0.8
  }}}}],
  "skins": [{{"name": "TwoJointSkin", "inverseBindMatrices": 7, "skeleton": 1, "joints": [1, 2]}}],
  "animations": [{{"name": "BendAndScale", "samplers": [
    {{"input": 8, "output": 9, "interpolation": "LINEAR"}},
    {{"input": 8, "output": 10, "interpolation": "LINEAR"}}
  ], "channels": [
    {{"sampler": 0, "target": {{"node": 2, "path": "rotation"}}}},
    {{"sampler": 1, "target": {{"node": 2, "path": "scale"}}}}
  ]}}],
  "buffers": [{{"byteLength": {}}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 72, "target": 34962}},
    {{"buffer": 0, "byteOffset": 72, "byteLength": 72, "target": 34962}},
    {{"buffer": 0, "byteOffset": 144, "byteLength": 96, "target": 34962}},
    {{"buffer": 0, "byteOffset": 240, "byteLength": 48, "target": 34962}},
    {{"buffer": 0, "byteOffset": 288, "byteLength": 48, "target": 34962}},
    {{"buffer": 0, "byteOffset": 336, "byteLength": 96, "target": 34962}},
    {{"buffer": 0, "byteOffset": 432, "byteLength": 24, "target": 34963}},
    {{"buffer": 0, "byteOffset": 456, "byteLength": 128}},
    {{"buffer": 0, "byteOffset": 584, "byteLength": 12}},
    {{"buffer": 0, "byteOffset": 596, "byteLength": 48}},
    {{"buffer": 0, "byteOffset": 644, "byteLength": 36}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3", "min": [-0.45, 0, 0], "max": [0.45, 2, 0]}},
    {{"bufferView": 1, "componentType": 5126, "count": 6, "type": "VEC3"}},
    {{"bufferView": 2, "componentType": 5126, "count": 6, "type": "VEC4"}},
    {{"bufferView": 3, "componentType": 5126, "count": 6, "type": "VEC2"}},
    {{"bufferView": 4, "componentType": 5123, "count": 6, "type": "VEC4"}},
    {{"bufferView": 5, "componentType": 5126, "count": 6, "type": "VEC4"}},
    {{"bufferView": 6, "componentType": 5123, "count": 12, "type": "SCALAR"}},
    {{"bufferView": 7, "componentType": 5126, "count": 2, "type": "MAT4"}},
    {{"bufferView": 8, "componentType": 5126, "count": 3, "type": "SCALAR", "min": [0], "max": [1]}},
    {{"bufferView": 9, "componentType": 5126, "count": 3, "type": "VEC4"}},
    {{"bufferView": 10, "componentType": 5126, "count": 3, "type": "VEC3"}}
  ]
}}"#,
        bin.len()
    );
    let mut json_bytes = json.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }

    let total_length = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut glb = Vec::with_capacity(total_length);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total_length as u32).to_le_bytes());
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
    glb.extend_from_slice(&json_bytes);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E_4942u32.to_le_bytes());
    glb.extend_from_slice(&bin);
    assert_eq!(glb.len(), total_length);

    fs::write(output, glb).unwrap();
}
