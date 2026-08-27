use std::{borrow::Cow, fmt, sync::Arc};

const NO_INDEX: u32 = u32::MAX;

/// Animation keyframes loaded independently from the renderer's model.
#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    /// Clip duration in seconds.
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SkeletonNode {
    pub parent_index: u32,
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationProperty {
    Translation,
    Rotation,
    Scale,
}

impl AnimationProperty {
    pub fn component_count(self) -> usize {
        match self {
            Self::Translation | Self::Scale => 3,
            Self::Rotation => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationInterpolation {
    Linear,
    Step,
    CubicSpline,
}

#[derive(Clone, Debug)]
pub struct AnimationChannel {
    pub node_index: usize,
    pub property: AnimationProperty,
    pub interpolation: AnimationInterpolation,
    pub inputs: Vec<f32>,
    pub outputs: Vec<f32>,
}

/// Animation-system representation of a model.
///
/// Node indices match the source glTF document, the renderer's bind-pose
/// nodes, and [`blade_render::Pose`] matrices.
pub struct AnimationModel {
    pub clips: Vec<AnimationClip>,
    nodes: Vec<SkeletonNode>,
}

impl AnimationModel {
    pub fn animation_index(&self, name: &str) -> Option<usize> {
        self.clips.iter().position(|clip| clip.name == name)
    }
}

#[derive(blade_macros::Flat)]
struct CookedNode {
    parent_index: u32,
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

#[derive(blade_macros::Flat)]
struct CookedAnimationChannel<'a> {
    node_index: u32,
    property: u32,
    interpolation: u32,
    inputs: Cow<'a, [f32]>,
    outputs: Cow<'a, [f32]>,
}

#[derive(blade_macros::Flat)]
struct CookedAnimation<'a> {
    name: Cow<'a, [u8]>,
    duration: f32,
    channels: Vec<CookedAnimationChannel<'a>>,
}

#[derive(blade_macros::Flat)]
pub(crate) struct CookedAnimationModel<'a> {
    nodes: Vec<CookedNode>,
    clips: Vec<CookedAnimation<'a>>,
}

impl From<CookedNode> for SkeletonNode {
    fn from(node: CookedNode) -> Self {
        Self {
            parent_index: node.parent_index,
            translation: node.translation.into(),
            rotation: glam::Quat::from_array(node.rotation),
            scale: node.scale.into(),
        }
    }
}

impl From<CookedAnimation<'_>> for AnimationClip {
    fn from(animation: CookedAnimation<'_>) -> Self {
        Self {
            name: String::from_utf8_lossy(animation.name.as_ref()).into_owned(),
            duration: animation.duration,
            channels: animation
                .channels
                .into_iter()
                .map(|channel| AnimationChannel {
                    node_index: channel.node_index as usize,
                    property: match channel.property {
                        0 => AnimationProperty::Translation,
                        1 => AnimationProperty::Rotation,
                        2 => AnimationProperty::Scale,
                        _ => unreachable!("invalid animation property"),
                    },
                    interpolation: match channel.interpolation {
                        0 => AnimationInterpolation::Linear,
                        1 => AnimationInterpolation::Step,
                        2 => AnimationInterpolation::CubicSpline,
                        _ => unreachable!("invalid animation interpolation"),
                    },
                    inputs: channel.inputs.into_owned(),
                    outputs: channel.outputs.into_owned(),
                })
                .collect(),
        }
    }
}

fn cook_nodes(document: &gltf::Document) -> Vec<CookedNode> {
    let mut parents = vec![NO_INDEX; document.nodes().len()];
    for node in document.nodes() {
        for child in node.children() {
            let old = std::mem::replace(&mut parents[child.index()], node.index() as u32);
            assert_eq!(old, NO_INDEX, "glTF node has more than one parent");
        }
    }

    document
        .nodes()
        .map(|node| {
            let (translation, rotation, scale) = node.transform().decomposed();
            CookedNode {
                parent_index: parents[node.index()],
                translation,
                rotation,
                scale,
            }
        })
        .collect()
}

fn cook_animations<'a>(document: &gltf::Document, buffers: &[Vec<u8>]) -> Vec<CookedAnimation<'a>> {
    use gltf::animation::util::ReadOutputs;

    document
        .animations()
        .map(|animation| {
            let mut start_time = f32::INFINITY;
            let mut duration = 0.0f32;
            let mut channels = Vec::new();
            let mut warned_non_uniform = false;
            for channel in animation.channels() {
                let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
                let inputs: Vec<_> = reader.read_inputs().unwrap().collect();
                if inputs.is_empty() {
                    log::warn!("Ignoring an empty animation channel");
                    continue;
                }
                assert!(inputs.iter().all(|time| time.is_finite()));
                assert!(inputs.windows(2).all(|times| times[0] <= times[1]));
                start_time = start_time.min(inputs[0]);
                duration = duration.max(*inputs.last().unwrap());

                let (property, outputs): (u32, Vec<f32>) = match reader.read_outputs().unwrap() {
                    ReadOutputs::Translations(values) => {
                        (0, values.flat_map(|value| value.into_iter()).collect())
                    }
                    ReadOutputs::Rotations(values) => (
                        1,
                        values
                            .into_f32()
                            .flat_map(|value| value.into_iter())
                            .collect(),
                    ),
                    ReadOutputs::Scales(values) => {
                        let values: Vec<_> = values.collect();
                        if !warned_non_uniform
                            && values.iter().any(|&value| is_non_uniform_scale(value))
                        {
                            log::warn!(
                                "Animation '{}' animates non-uniform scale; skinning assumes \
                                 uniform scale, so its normals may be slightly skewed",
                                animation.name().unwrap_or("")
                            );
                            warned_non_uniform = true;
                        }
                        (
                            2,
                            values
                                .into_iter()
                                .flat_map(|value| value.into_iter())
                                .collect(),
                        )
                    }
                    ReadOutputs::MorphTargetWeights(_) => {
                        log::warn!(
                            "Ignoring morph-target channel in animation '{}'",
                            animation.name().unwrap_or("")
                        );
                        continue;
                    }
                };
                let interpolation = match channel.sampler().interpolation() {
                    gltf::animation::Interpolation::Linear => 0,
                    gltf::animation::Interpolation::Step => 1,
                    gltf::animation::Interpolation::CubicSpline => 2,
                };
                let component_count = if property == 1 { 4 } else { 3 };
                let sample_multiplier = if interpolation == 2 { 3 } else { 1 };
                assert_eq!(
                    outputs.len(),
                    inputs.len() * component_count * sample_multiplier,
                    "animation channel has mismatched input and output counts"
                );
                channels.push(CookedAnimationChannel {
                    node_index: channel.target().node().index() as u32,
                    property,
                    interpolation,
                    inputs: Cow::Owned(inputs),
                    outputs: Cow::Owned(outputs),
                });
            }
            if start_time.is_finite() {
                for channel in &mut channels {
                    for time in channel.inputs.to_mut() {
                        *time -= start_time;
                    }
                }
                duration -= start_time;
            }
            CookedAnimation {
                name: Cow::Owned(animation.name().unwrap_or("").as_bytes().to_owned()),
                duration,
                channels,
            }
        })
        .collect()
}

fn is_non_uniform_scale(scale: [f32; 3]) -> bool {
    (scale[0] - scale[1]).abs() > 1.0e-4 || (scale[0] - scale[2]).abs() > 1.0e-4
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Meta;

impl fmt::Display for Meta {
    fn fmt(&self, _formatter: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

pub(crate) struct Baker;

impl blade_asset::Baker for Baker {
    type Meta = Meta;
    type Data<'a> = CookedAnimationModel<'a>;
    type Output = AnimationModel;

    fn cook(
        &self,
        source: &[u8],
        extension: &str,
        _meta: Meta,
        cooker: Arc<blade_asset::Cooker<Self>>,
        _exe_context: &choir::ExecutionContext,
    ) {
        use base64::engine::{Engine as _, general_purpose::STANDARD as ENCODING_ENGINE};

        assert!(matches!(extension, "gltf" | "glb"));
        let gltf::Gltf { document, mut blob } = gltf::Gltf::from_slice(source).unwrap();
        let mut buffers = Vec::new();
        for buffer in document.buffers() {
            let mut data = match buffer.source() {
                gltf::buffer::Source::Uri(uri) => {
                    if let Some(rest) = uri.strip_prefix("data:") {
                        let (_, encoded) = rest.split_once(";base64,").unwrap();
                        ENCODING_ENGINE.decode(encoded).unwrap()
                    } else if let Some(rest) = uri.strip_prefix("file://") {
                        cooker.add_dependency(rest.as_ref())
                    } else if let Some(rest) = uri.strip_prefix("file:") {
                        cooker.add_dependency(rest.as_ref())
                    } else {
                        cooker.add_dependency(uri.as_ref())
                    }
                }
                gltf::buffer::Source::Bin => blob.take().unwrap(),
            };
            assert!(data.len() >= buffer.length());
            while data.len() % 4 != 0 {
                data.push(0);
            }
            buffers.push(data);
        }
        cooker.finish(CookedAnimationModel {
            nodes: cook_nodes(&document),
            clips: cook_animations(&document, &buffers),
        });
    }

    fn serve(
        &self,
        cooked: CookedAnimationModel<'_>,
        _exe_context: &choir::ExecutionContext,
    ) -> AnimationModel {
        AnimationModel {
            nodes: cooked.nodes.into_iter().map(SkeletonNode::from).collect(),
            clips: cooked.clips.into_iter().map(AnimationClip::from).collect(),
        }
    }

    fn delete(&self, _model: AnimationModel) {}
}

/// Playback state owned and evaluated by the engine, independently of rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimationPlayer {
    pub clip_index: usize,
    pub time: f32,
    pub speed: f32,
    pub looping: bool,
}

impl AnimationPlayer {
    pub fn new(clip_index: usize) -> Self {
        Self {
            clip_index,
            time: 0.0,
            speed: 1.0,
            looping: true,
        }
    }

    pub fn advance(&mut self, delta_time: f32) {
        self.time += delta_time * self.speed;
    }

    /// Evaluate this player into the concrete pose consumed by `blade-render`.
    pub fn pose(&self, model: &AnimationModel) -> Option<blade_render::Pose> {
        let clip = model.clips.get(self.clip_index)?;
        Some(evaluate(&model.nodes, clip, self.sample_time(clip)))
    }

    fn sample_time(&self, clip: &AnimationClip) -> f32 {
        let playback_time = if self.time.is_finite() {
            self.time
        } else {
            0.0
        };
        if clip.duration <= 0.0 {
            0.0
        } else if self.looping {
            playback_time.rem_euclid(clip.duration)
        } else {
            playback_time.clamp(0.0, clip.duration)
        }
    }
}

fn read_sample(channel: &AnimationChannel, key: usize, cubic_part: usize) -> &[f32] {
    let components = channel.property.component_count();
    let multiplier = if channel.interpolation == AnimationInterpolation::CubicSpline {
        3
    } else {
        1
    };
    let start = (key * multiplier + cubic_part) * components;
    &channel.outputs[start..start + components]
}

fn interpolate_channel(channel: &AnimationChannel, time: f32) -> [f32; 4] {
    use {AnimationInterpolation as Interpolation, AnimationProperty as Property};

    if channel.inputs.is_empty() {
        return [0.0; 4];
    }
    let last = channel.inputs.len() - 1;
    let (left, right, factor, delta) = if time <= channel.inputs[0] || last == 0 {
        (0, 0, 0.0, 0.0)
    } else if time >= channel.inputs[last] {
        (last, last, 0.0, 0.0)
    } else {
        let right = channel.inputs.partition_point(|&sample| sample <= time);
        let left = right - 1;
        let delta = channel.inputs[right] - channel.inputs[left];
        let factor = if delta > 0.0 {
            (time - channel.inputs[left]) / delta
        } else {
            0.0
        };
        (left, right, factor, delta)
    };

    let value_part = usize::from(channel.interpolation == Interpolation::CubicSpline);
    let a = read_sample(channel, left, value_part);
    let b = read_sample(channel, right, value_part);
    let components = channel.property.component_count();
    let mut result = [0.0; 4];
    match channel.interpolation {
        Interpolation::Step => result[..components].copy_from_slice(a),
        Interpolation::Linear if channel.property == Property::Rotation => {
            let qa = glam::Quat::from_array(a.try_into().unwrap()).normalize();
            let qb = glam::Quat::from_array(b.try_into().unwrap()).normalize();
            result = qa.slerp(qb, factor).normalize().to_array();
        }
        Interpolation::Linear => {
            for component in 0..components {
                result[component] = a[component] + factor * (b[component] - a[component]);
            }
        }
        Interpolation::CubicSpline => {
            let out_tangent = read_sample(channel, left, 2);
            let in_tangent = read_sample(channel, right, 0);
            let t2 = factor * factor;
            let t3 = t2 * factor;
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + factor;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            for component in 0..components {
                result[component] = h00 * a[component]
                    + h10 * delta * out_tangent[component]
                    + h01 * b[component]
                    + h11 * delta * in_tangent[component];
            }
        }
    }
    if channel.property == Property::Rotation {
        result = glam::Quat::from_array(result).normalize().to_array();
    }
    result
}

fn evaluate(nodes: &[SkeletonNode], clip: &AnimationClip, time: f32) -> blade_render::Pose {
    let mut locals = nodes.to_vec();
    for channel in &clip.channels {
        let value = interpolate_channel(channel, time);
        let node = &mut locals[channel.node_index];
        match channel.property {
            AnimationProperty::Translation => node.translation = glam::Vec3::from_slice(&value),
            AnimationProperty::Rotation => node.rotation = glam::Quat::from_array(value),
            AnimationProperty::Scale => node.scale = glam::Vec3::from_slice(&value),
        }
    }

    // Resolve the global transforms with constant-time cycle detection:
    // `state` marks nodes as unvisited, on the current chain, or resolved.
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
            let local = glam::Mat4::from_scale_rotation_translation(
                locals[index].scale,
                locals[index].rotation.normalize(),
                locals[index].translation,
            );
            globals[index] = match nodes[index].parent_index {
                NO_INDEX => local,
                parent => globals[parent as usize] * local,
            };
            state[index] = VisitState::Resolved;
        }
    }

    blade_render::Pose::from_node_matrices(globals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use {AnimationInterpolation as Interpolation, AnimationProperty as Property};

    fn channel(interpolation: Interpolation, outputs: Vec<f32>) -> AnimationChannel {
        AnimationChannel {
            node_index: 0,
            property: Property::Translation,
            interpolation,
            inputs: vec![0.0, 2.0],
            outputs,
        }
    }

    #[test]
    fn linearly_interpolates_keyframes() {
        let value = interpolate_channel(
            &channel(Interpolation::Linear, vec![0.0, 1.0, 2.0, 4.0, 5.0, 6.0]),
            0.5,
        );
        assert_eq!(value, [1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn step_interpolation_holds_the_left_keyframe() {
        let value = interpolate_channel(
            &channel(Interpolation::Step, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]),
            1.5,
        );
        assert_eq!(value, [1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn cubic_spline_uses_gltf_tangent_layout() {
        let value = interpolate_channel(
            &channel(
                Interpolation::CubicSpline,
                vec![
                    0.0, 0.0, 0.0, // key 0 in tangent
                    0.0, 0.0, 0.0, // key 0 value
                    0.0, 0.0, 0.0, // key 0 out tangent
                    0.0, 0.0, 0.0, // key 1 in tangent
                    2.0, 0.0, 0.0, // key 1 value
                    0.0, 0.0, 0.0, // key 1 out tangent
                ],
            ),
            1.0,
        );
        assert_eq!(value, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn playback_time_loops_or_clamps() {
        let clip = AnimationClip {
            name: String::new(),
            duration: 2.0,
            channels: Vec::new(),
        };
        let mut player = AnimationPlayer::new(0);
        player.time = 5.0;
        assert_eq!(player.sample_time(&clip), 1.0);
        player.looping = false;
        assert_eq!(player.sample_time(&clip), 2.0);
    }

    #[test]
    fn cooks_gltf_animation_model() {
        let source = br#"{
            "asset":{"version":"2.0"},
            "buffers":[{"byteLength":32,"uri":"animation.bin"}],
            "bufferViews":[
                {"buffer":0,"byteOffset":0,"byteLength":8},
                {"buffer":0,"byteOffset":8,"byteLength":24}
            ],
            "accessors":[
                {"bufferView":0,"componentType":5126,"count":2,"type":"SCALAR"},
                {"bufferView":1,"componentType":5126,"count":2,"type":"VEC3"}
            ],
            "nodes":[{"children":[1]},{"name":"joint"}],
            "animations":[{
                "name":"move",
                "samplers":[{"input":0,"output":1,"interpolation":"LINEAR"}],
                "channels":[{"sampler":0,"target":{"node":1,"path":"translation"}}]
            }]
        }"#;
        let gltf::Gltf { document, .. } = gltf::Gltf::from_slice(source).unwrap();
        let mut data = Vec::new();
        for value in [0.0f32, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let buffers = vec![data];

        let nodes = cook_nodes(&document);
        assert_eq!(nodes[0].parent_index, NO_INDEX);
        assert_eq!(nodes[1].parent_index, 0);
        let clips = cook_animations(&document, &buffers);
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].name.as_ref(), b"move");
        assert_eq!(clips[0].duration, 1.0);
        let clip = AnimationClip::from(clips.into_iter().next().unwrap());
        assert_eq!(clip.channels[0].property, AnimationProperty::Translation);
        assert_eq!(clip.channels[0].outputs[3], 2.0);
    }

    #[test]
    fn animation_model_survives_flat_asset_round_trip() {
        use blade_asset::Flat as _;

        let cooked = CookedAnimationModel {
            nodes: vec![CookedNode {
                parent_index: NO_INDEX,
                translation: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            }],
            clips: vec![CookedAnimation {
                name: Cow::Borrowed(b"walk"),
                duration: 2.0,
                channels: vec![CookedAnimationChannel {
                    node_index: 0,
                    property: 0,
                    interpolation: 0,
                    inputs: Cow::Borrowed(&[0.0, 2.0]),
                    outputs: Cow::Borrowed(&[0.0, 0.0, 0.0, 4.0, 0.0, 0.0]),
                }],
            }],
        };
        let mut bytes = vec![0u8; cooked.size()];
        unsafe { cooked.write(bytes.as_mut_ptr()) };
        let restored = unsafe { CookedAnimationModel::read(bytes.as_ptr()) };
        let model = AnimationModel {
            nodes: restored.nodes.into_iter().map(SkeletonNode::from).collect(),
            clips: restored
                .clips
                .into_iter()
                .map(AnimationClip::from)
                .collect(),
        };
        assert_eq!(model.nodes.len(), 1);
        assert_eq!(model.animation_index("walk"), Some(0));
        assert_eq!(model.clips[0].channels[0].outputs[3], 4.0);
    }

    #[test]
    fn evaluates_animated_skin_glb_fixture() {
        let source = include_bytes!("../../tests/assets/animated_skin.glb");
        let gltf::Gltf {
            document,
            blob: Some(blob),
        } = gltf::Gltf::from_slice(source).unwrap()
        else {
            panic!("fixture must contain a binary buffer")
        };
        let model = AnimationModel {
            nodes: cook_nodes(&document)
                .into_iter()
                .map(SkeletonNode::from)
                .collect(),
            clips: cook_animations(&document, &[blob])
                .into_iter()
                .map(AnimationClip::from)
                .collect(),
        };

        assert_eq!(model.animation_index("BendAndScale"), Some(0));
        assert_eq!(model.clips[0].channels.len(), 2);
        let mut player = AnimationPlayer::new(0);
        player.looping = false;
        player.time = 0.5;
        let pose = player.pose(&model).unwrap();
        let tip = pose.matrix(2);

        assert!((tip.x_axis.truncate().length() - 1.6).abs() < 1e-5);
        assert!((tip.y_axis.truncate().length() - 0.7).abs() < 1e-5);
        assert!((tip.z_axis.truncate().length() - 0.55).abs() < 1e-5);
        let offset = 0.7 * std::f32::consts::FRAC_1_SQRT_2;
        let expected = glam::Vec3::new(-offset, 1.0 + offset, 0.0);
        assert!(
            tip.transform_point3(glam::Vec3::Y)
                .abs_diff_eq(expected, 1e-5)
        );
    }

    #[test]
    fn evaluates_out_of_order_node_hierarchy() {
        let nodes = [
            SkeletonNode {
                parent_index: 1,
                translation: glam::Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
            SkeletonNode {
                parent_index: NO_INDEX,
                translation: glam::Vec3::Y,
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
        ];
        let clip = AnimationClip {
            name: "move".into(),
            duration: 1.0,
            channels: vec![AnimationChannel {
                node_index: 0,
                property: Property::Translation,
                interpolation: Interpolation::Linear,
                inputs: vec![0.0, 1.0],
                outputs: vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            }],
        };
        let pose = evaluate(&nodes, &clip, 0.5);
        let matrix = pose.matrix(0);
        assert_eq!(
            matrix.transform_point3(glam::Vec3::ZERO),
            glam::Vec3::new(1.0, 1.0, 0.0)
        );
    }
}
