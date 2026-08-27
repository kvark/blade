use crate::Vertex;
use blade_graphics as gpu;
use std::mem;

const IDENTITY_AFFINE_3X4: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub(crate) struct SkinningParams {
    pub post_transform: [f32; 12],
    pub joint_matrices: [[f32; 12]; crate::model::MAX_JOINTS_PER_DRAW],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct SkinDispatch {
    vertex_count: u32,
    _pad: [u32; 3],
}

#[derive(blade_macros::ShaderData)]
struct SkinData {
    skinning_params: SkinningParams,
    skin_dispatch: SkinDispatch,
    source: gpu::BufferPiece,
    skin_source: gpu::BufferPiece,
    destination: gpu::BufferPiece,
}

pub(crate) struct SkinJob {
    source: gpu::BufferPiece,
    skin_source: gpu::BufferPiece,
    destination: gpu::BufferPiece,
    vertex_count: u32,
    params: SkinningParams,
}

pub(crate) struct VertexCopy {
    source: gpu::BufferPiece,
    destination: gpu::BufferPiece,
    size: u64,
}

pub(crate) struct SkinPass {
    pipeline: gpu::ComputePipeline,
}

impl SkinPass {
    fn create_pipeline(shader: &gpu::Shader, gpu: &gpu::Context) -> gpu::ComputePipeline {
        shader.check_struct_size::<Vertex>();
        shader.check_struct_size::<crate::SkinVertex>();
        shader.check_struct_size::<SkinningParams>();
        shader.check_struct_size::<SkinDispatch>();
        let layout = <SkinData as gpu::ShaderData>::layout();
        gpu.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "skin",
            data_layouts: &[&layout],
            compute: shader.at("skin"),
        })
    }

    pub fn new(shader: &gpu::Shader, gpu: &gpu::Context) -> Option<Self> {
        if !gpu.capabilities().compute {
            return None;
        }
        Some(Self {
            pipeline: Self::create_pipeline(shader, gpu),
        })
    }

    pub fn destroy(&mut self, gpu: &gpu::Context) {
        gpu.destroy_compute_pipeline(&mut self.pipeline);
    }

    pub fn recreate(&mut self, shader: &gpu::Shader, gpu: &gpu::Context) {
        self.pipeline = Self::create_pipeline(shader, gpu);
    }

    fn dispatch(&self, encoder: &mut gpu::CommandEncoder, jobs: &[SkinJob]) {
        if jobs.is_empty() {
            return;
        }
        let mut pass = encoder.compute("skin");
        for job in jobs {
            let mut pc = pass.with(&self.pipeline);
            pc.bind(
                0,
                &SkinData {
                    skinning_params: job.params,
                    skin_dispatch: SkinDispatch {
                        vertex_count: job.vertex_count,
                        _pad: [0; 3],
                    },
                    source: job.source,
                    skin_source: job.skin_source,
                    destination: job.destination,
                },
            );
            pc.dispatch(self.pipeline.get_dispatch_for(gpu::Extent {
                width: job.vertex_count,
                height: 1,
                depth: 1,
            }));
        }
    }
}

fn affine_rows(matrix: glam::Mat4) -> [f32; 12] {
    let transform = crate::model::mat4_to_transform(matrix);
    [
        transform.x.x,
        transform.x.y,
        transform.x.z,
        transform.x.w,
        transform.y.x,
        transform.y.y,
        transform.y.z,
        transform.y.w,
        transform.z.x,
        transform.z.y,
        transform.z.z,
        transform.z.w,
    ]
}

pub(crate) fn make_skinning_params(
    model: &crate::Model,
    geometry: &crate::model::Geometry,
    pose: Option<&crate::Pose>,
) -> SkinningParams {
    make_skinning_params_with_post(model, geometry, pose, glam::Mat4::IDENTITY)
}

fn make_skinning_params_with_post(
    model: &crate::Model,
    geometry: &crate::model::Geometry,
    pose: Option<&crate::Pose>,
    post_transform: glam::Mat4,
) -> SkinningParams {
    let mut params = SkinningParams {
        post_transform: affine_rows(post_transform),
        joint_matrices: [IDENTITY_AFFINE_3X4; crate::model::MAX_JOINTS_PER_DRAW],
    };
    let Some(skin_index) = geometry.skin_index else {
        return params;
    };
    let pose = model.matching_pose(pose);
    let skin = &model.skins[skin_index];
    let inverse_mesh = pose.matrix(geometry.node_index).inverse();
    for (dst, &palette_index) in params
        .joint_matrices
        .iter_mut()
        .zip(&geometry.joint_palette)
    {
        let joint_index = palette_index as usize;
        let matrix = inverse_mesh
            * pose.matrix(skin.joints[joint_index] as usize)
            * skin.inverse_bind_matrices[joint_index];
        *dst = affine_rows(matrix);
    }
    params
}

pub(crate) fn model_needs_vertex_skin(model: &crate::Model) -> bool {
    model
        .geometries
        .iter()
        .any(|geometry| geometry.skin_index.is_some())
}

pub(crate) fn queue_model(
    model: &crate::Model,
    pose: Option<&crate::Pose>,
    post_transforms: Option<&[blade_graphics::Transform]>,
    destination: gpu::Buffer,
    destination_base: u64,
    jobs: &mut Vec<SkinJob>,
    copies: &mut Vec<VertexCopy>,
) {
    let vertex_size = mem::size_of::<crate::Vertex>() as u64;
    let skin_vertex_size = mem::size_of::<crate::SkinVertex>() as u64;
    for (geometry_index, geometry) in model.geometries.iter().enumerate() {
        let count = geometry.vertex_range.end - geometry.vertex_range.start;
        if count == 0 {
            continue;
        }
        let src_offset = geometry.vertex_range.start as u64 * vertex_size;
        let dst_offset = destination_base + src_offset;
        let source = model.vertex_buffer.at(src_offset);
        let skin_source = model
            .skin_vertex_buffer
            .at(geometry.vertex_range.start as u64 * skin_vertex_size);
        let destination = destination.at(dst_offset);
        if geometry.skin_index.is_some() || post_transforms.is_some() {
            let post_transform = post_transforms
                .map(|transforms| crate::model::mat4_from_transform(&transforms[geometry_index]))
                .unwrap_or(glam::Mat4::IDENTITY);
            jobs.push(SkinJob {
                source,
                skin_source,
                destination,
                vertex_count: count,
                params: make_skinning_params_with_post(model, geometry, pose, post_transform),
            });
        } else {
            copies.push(VertexCopy {
                source,
                destination,
                size: count as u64 * vertex_size,
            });
        }
    }
}

pub(crate) fn encode(
    encoder: &mut gpu::CommandEncoder,
    skin_pass: Option<&SkinPass>,
    copies: &[VertexCopy],
    jobs: &[SkinJob],
) {
    if !copies.is_empty() {
        let mut transfer = encoder.transfer("skin-copy");
        for copy in copies {
            transfer.copy_buffer_to_buffer(copy.source, copy.destination, copy.size);
        }
    }
    if let Some(skin_pass) = skin_pass {
        skin_pass.dispatch(encoder, jobs);
    } else {
        assert!(
            jobs.is_empty(),
            "compute skinning jobs were queued without a skin pipeline"
        );
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn joint_palette_fits_portable_uniform_limits() {
        assert_eq!(std::mem::size_of::<super::SkinningParams>(), 3120);
    }

    #[test]
    fn dispatch_params_match_uniform_alignment() {
        assert_eq!(std::mem::size_of::<super::SkinDispatch>(), 16);
    }
}
