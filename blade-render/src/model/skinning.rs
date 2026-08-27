use std::borrow::Cow;

/// Compact joint palettes fit in a WebGL2 uniform block. Must match
/// `MAX_JOINTS_PER_DRAW` in `blade-render/code/skin.inc.wgsl`.
pub const MAX_JOINTS_PER_DRAW: usize = 64;
const NO_INDEX: u32 = u32::MAX;

/// Rest-pose transform of one node in a skinned model.
///
/// Scale is part of the glTF rest pose: inverse-bind matrices are affine, and
/// clips may animate scale, including non-uniform scale.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SkinNode {
    pub parent_index: u32,
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}

impl SkinNode {
    fn local_transform(self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(
            self.scale,
            self.rotation.normalize(),
            self.translation,
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Skin {
    pub joints: Vec<u32>,
    pub inverse_bind_matrices: Vec<glam::Mat4>,
}

/// A fully evaluated model pose, expressed as affine node transforms.
///
/// Each transform is a 3x4 matrix indexed by glTF node index, matching both
/// the renderer's bind-pose nodes and `blade-engine`'s animation model.
#[derive(Clone, Debug, PartialEq)]
pub struct Pose {
    transforms: Vec<blade_graphics::Transform>,
}

impl Pose {
    /// Construct a pose from global node matrices.
    pub fn from_node_matrices(node_matrices: Vec<glam::Mat4>) -> Self {
        Self {
            transforms: node_matrices
                .into_iter()
                .map(super::mat4_to_transform)
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    pub fn node_transforms(&self) -> &[blade_graphics::Transform] {
        &self.transforms
    }

    pub fn matrix(&self, index: usize) -> glam::Mat4 {
        super::mat4_from_transform(&self.transforms[index])
    }
}

#[derive(blade_macros::Flat)]
pub(crate) struct CookedNode {
    pub parent_index: u32,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(blade_macros::Flat)]
pub(crate) struct CookedSkin<'a> {
    pub joints: Cow<'a, [u32]>,
    pub inverse_bind_matrices: Cow<'a, [[f32; 16]]>,
}

impl From<CookedNode> for SkinNode {
    fn from(node: CookedNode) -> Self {
        Self {
            parent_index: node.parent_index,
            translation: node.translation.into(),
            rotation: glam::Quat::from_array(node.rotation),
            scale: node.scale.into(),
        }
    }
}

impl From<CookedSkin<'_>> for Skin {
    fn from(skin: CookedSkin<'_>) -> Self {
        Self {
            joints: skin.joints.into_owned(),
            inverse_bind_matrices: skin
                .inverse_bind_matrices
                .iter()
                .map(glam::Mat4::from_cols_array)
                .collect(),
        }
    }
}

#[cfg(feature = "asset")]
pub(crate) fn cook_nodes(document: &gltf::Document) -> Vec<CookedNode> {
    let mut parents = vec![NO_INDEX; document.nodes().len()];
    for node in document.nodes() {
        for child in node.children() {
            let old = std::mem::replace(&mut parents[child.index()], node.index() as u32);
            assert_eq!(old, NO_INDEX, "glTF node has more than one parent");
        }
    }

    let mut warned_non_uniform = false;
    document
        .nodes()
        .map(|node| {
            let (translation, rotation, scale) = node.transform().decomposed();
            if !warned_non_uniform && is_non_uniform_scale(scale) {
                log::warn!(
                    "Node '{}' has a non-uniform rest scale; skinning assumes \
                     uniform scale, so its normals may be slightly skewed",
                    node.name().unwrap_or("")
                );
                warned_non_uniform = true;
            }
            CookedNode {
                parent_index: parents[node.index()],
                translation,
                rotation,
                scale,
            }
        })
        .collect()
}

fn is_non_uniform_scale(scale: [f32; 3]) -> bool {
    (scale[0] - scale[1]).abs() > 1.0e-4 || (scale[0] - scale[2]).abs() > 1.0e-4
}

#[cfg(feature = "asset")]
pub(crate) fn cook_skins<'a>(
    document: &gltf::Document,
    buffers: &[Vec<u8>],
) -> Vec<CookedSkin<'a>> {
    document
        .skins()
        .map(|skin| {
            let joints = skin.joints().map(|node| node.index() as u32).collect();
            let inverse_bind_matrices = skin
                .reader(|buffer| Some(&buffers[buffer.index()]))
                .read_inverse_bind_matrices()
                .map(|matrices| {
                    matrices
                        .map(|matrix| glam::Mat4::from_cols_array_2d(&matrix).to_cols_array())
                        .collect()
                })
                .unwrap_or_else(|| vec![[0.0; 16]; skin.joints().len()]);
            let mut cooked = CookedSkin {
                joints: Cow::Owned(joints),
                inverse_bind_matrices: Cow::Owned(inverse_bind_matrices),
            };
            if skin.inverse_bind_matrices().is_none() {
                for matrix in cooked.inverse_bind_matrices.to_mut() {
                    *matrix = glam::Mat4::IDENTITY.to_cols_array();
                }
            }
            assert_eq!(cooked.joints.len(), cooked.inverse_bind_matrices.len());
            cooked
        })
        .collect()
}

fn resolve_globals(nodes: &[SkinNode], locals: &[glam::Mat4]) -> Vec<glam::Mat4> {
    // Constant-time cycle detection: `state` marks nodes as unvisited,
    // on the current chain, or resolved.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum VisitState {
        Unvisited,
        OnStack,
        Resolved,
    }
    let mut globals = vec![glam::Mat4::IDENTITY; nodes.len()];
    let mut state = vec![VisitState::Unvisited; nodes.len()];
    for start in 0..nodes.len() {
        let mut chain = Vec::new();
        let mut current = start;
        loop {
            if state[current] != VisitState::Unvisited {
                assert_ne!(
                    state[current],
                    VisitState::OnStack,
                    "cycle in model node hierarchy"
                );
                break;
            }
            state[current] = VisitState::OnStack;
            chain.push(current);
            match nodes[current].parent_index {
                NO_INDEX => break,
                parent => current = parent as usize,
            }
        }
        while let Some(index) = chain.pop() {
            globals[index] = match nodes[index].parent_index {
                NO_INDEX => locals[index],
                parent => globals[parent as usize] * locals[index],
            };
            state[index] = VisitState::Resolved;
        }
    }
    globals
}

pub(crate) fn bind_pose(nodes: &[SkinNode]) -> Pose {
    let locals: Vec<_> = nodes
        .iter()
        .copied()
        .map(SkinNode::local_transform)
        .collect();
    Pose::from_node_matrices(resolve_globals(nodes, &locals))
}

impl super::Model {
    /// The pose to use, falling back to the bind pose when the provided one
    /// doesn't cover this model's nodes (e.g. it was evaluated from a
    /// different asset revision).
    pub(crate) fn matching_pose<'a>(&'a self, pose: Option<&'a Pose>) -> &'a Pose {
        match pose {
            Some(pose) if pose.len() == self.nodes.len() => pose,
            Some(pose) => {
                log::warn!(
                    "Pose with {} nodes doesn't match model '{}' with {} nodes; \
                     using the bind pose",
                    pose.len(),
                    self.name,
                    self.nodes.len()
                );
                &self.bind_pose
            }
            None => &self.bind_pose,
        }
    }

    pub(crate) fn geometry_transform(
        &self,
        geometry: &super::Geometry,
        pose: Option<&Pose>,
    ) -> blade_graphics::Transform {
        if geometry.skin_index.is_some() || pose.is_some() {
            self.matching_pose(pose).node_transforms()[geometry.node_index]
        } else {
            geometry.transform
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_out_of_order_node_hierarchy() {
        let nodes = [
            SkinNode {
                parent_index: 1,
                translation: glam::Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
            SkinNode {
                parent_index: NO_INDEX,
                translation: glam::Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
        ];
        let locals = [
            glam::Mat4::from_translation(glam::Vec3::X),
            glam::Mat4::from_translation(glam::Vec3::Y),
        ];
        let globals = resolve_globals(&nodes, &locals);
        assert_eq!(
            globals[0].transform_point3(glam::Vec3::ZERO),
            glam::Vec3::new(1.0, 1.0, 0.0)
        );
    }

    #[cfg(feature = "asset")]
    #[test]
    fn cooks_gltf_skin() {
        let source = br#"{
            "asset":{"version":"2.0"},
            "buffers":[{"byteLength":96,"uri":"animation.bin"}],
            "bufferViews":[
                {"buffer":0,"byteOffset":0,"byteLength":8},
                {"buffer":0,"byteOffset":8,"byteLength":24},
                {"buffer":0,"byteOffset":32,"byteLength":64}
            ],
            "accessors":[
                {"bufferView":0,"componentType":5126,"count":2,"type":"SCALAR"},
                {"bufferView":1,"componentType":5126,"count":2,"type":"VEC3"},
                {"bufferView":2,"componentType":5126,"count":1,"type":"MAT4"}
            ],
            "nodes":[{"children":[1]},{"name":"joint"}],
            "skins":[{"joints":[1],"inverseBindMatrices":2}],
            "animations":[{
                "name":"move",
                "samplers":[{"input":0,"output":1,"interpolation":"LINEAR"}],
                "channels":[{"sampler":0,"target":{"node":1,"path":"translation"}}]
            }]
        }"#;
        let gltf::Gltf { document, .. } = gltf::Gltf::from_slice(source).unwrap();
        let mut data = Vec::new();
        data.extend_from_slice(bytemuck::bytes_of(&0.0f32));
        data.extend_from_slice(bytemuck::bytes_of(&1.0f32));
        data.extend_from_slice(bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 2.0, 0.0, 0.0]));
        data.extend_from_slice(bytemuck::cast_slice(&glam::Mat4::IDENTITY.to_cols_array()));
        let buffers = vec![data];

        let nodes = cook_nodes(&document);
        assert_eq!(nodes[0].parent_index, NO_INDEX);
        assert_eq!(nodes[1].parent_index, 0);

        let skins = cook_skins(&document, &buffers);
        assert_eq!(skins.len(), 1);
        assert_eq!(skins[0].joints.as_ref(), &[1]);
        assert_eq!(
            skins[0].inverse_bind_matrices[0],
            glam::Mat4::IDENTITY.to_cols_array()
        );
    }
}
