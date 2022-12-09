use super::belt::{BeltDescriptor, BufferBelt};
use std::{collections::HashMap, fs};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Uniforms {}

#[derive(blade::ShaderData)]
struct ShaderData {
    uniforms: Uniforms,
    vertex_attributes: blade::BufferPiece,
    texture: blade::TextureView,
    sampler: blade::Sampler,
}

pub struct ScreenDescriptor {
    pub physical_size: (u32, u32),
    pub scale_factor: f32,
}

impl ScreenDescriptor {
    fn logical_size(&self) -> (u32, u32) {
        let logical_width = self.physical_size.0 as f32 / self.scale_factor;
        let logical_height = self.physical_size.1 as f32 / self.scale_factor;
        (logical_width as u32, logical_height as u32)
    }
}

struct GuiPrimitive {
    vertex_buf: blade::BufferPiece,
    index_buf: blade::BufferPiece,
}

struct GuiRender {
    pipeline: blade::RenderPipeline,
    belt: BufferBelt,
    primitives: Vec<GuiPrimitive>,
    last_user_texture_id: u64,
    textures: HashMap<egui::TextureId, blade::TextureView>,
}

impl GuiRender {
    pub fn new(context: &blade::Context, output_format: blade::TextureFormat) -> Self {
        let shader_source = fs::read_to_string("examples/particle/gui.wgsl").unwrap();
        let shader = context.create_shader(blade::ShaderDesc {
            source: &shader_source,
        });

        let data_layout = <ShaderData as blade::ShaderData>::layout();
        let pipeline = context.create_render_pipeline(blade::RenderPipelineDesc {
            name: "gui",
            data_layouts: &[&data_layout],
            vertex: shader.at("vs_main"),
            primitive: blade::PrimitiveState {
                topology: blade::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None, //TODO?
            fragment: shader.at("fs_main"),
            color_targets: &[blade::ColorTargetState {
                format: output_format,
                blend: Some(blade::BlendState::ALPHA_BLENDING),
                write_mask: blade::ColorWrites::all(),
            }],
        });

        let belt = BufferBelt::new(BeltDescriptor {
            memory: blade::Memory::Shared,
            min_chunk_size: 0x1000,
        });
        Self {
            pipeline,
            belt,
            primitives: Vec::new(),
            last_user_texture_id: 0,
            textures: Default::default(),
        }
    }

    fn execute(
        &self,
        pass: &mut blade::RenderCommandEncoder,
        paint_jobs: &[egui::epaint::ClippedPrimitive],
        sd: &ScreenDescriptor,
    ) {
        let mut pc = pass.with(&self.pipeline);
        assert_eq!(self.primitives.len(), paint_jobs.len());
        for (
            egui::ClippedPrimitive {
                clip_rect,
                primitive,
            },
            prim_data,
        ) in paint_jobs.iter().zip(self.primitives.iter())
        {
            {
                // Transform clip rect to physical pixels.
                let clip_min_x = sd.scale_factor * clip_rect.min.x;
                let clip_min_y = sd.scale_factor * clip_rect.min.y;
                let clip_max_x = sd.scale_factor * clip_rect.max.x;
                let clip_max_y = sd.scale_factor * clip_rect.max.y;

                // Make sure clip rect can fit within an `u32`.
                let clip_min_x = clip_min_x.clamp(0.0, sd.physical_size.0 as f32);
                let clip_min_y = clip_min_y.clamp(0.0, sd.physical_size.1 as f32);
                let clip_max_x = clip_max_x.clamp(clip_min_x, sd.physical_size.0 as f32);
                let clip_max_y = clip_max_y.clamp(clip_min_y, sd.physical_size.1 as f32);

                let clip_min_x = clip_min_x.trunc() as u32;
                let clip_min_y = clip_min_y.trunc() as u32;
                let clip_max_x = clip_max_x.ceil() as u32;
                let clip_max_y = clip_max_y.ceil() as u32;

                let width = (clip_max_x - clip_min_x).max(1);
                let height = (clip_max_y - clip_min_y).max(1);

                let width = width.min(sd.physical_size.0 - clip_min_x);
                let height = height.min(sd.physical_size.1 - clip_min_y);

                if width == 0 || height == 0 {
                    continue;
                }

                //rpass.set_scissor_rect(clip_min_x, clip_min_y, width, height);
            }

            if let egui::epaint::Primitive::Mesh(mesh) = primitive {
                //TODO:
                //let bind_group = self.get_texture_bind_group(mesh.texture_id)?;
                //rpass.set_bind_group(1, bind_group, &[]);

                //rpass.set_index_buffer(index_buffer.buffer.slice(..), wgpu::IndexFormat::Uint32);
                //rpass.set_vertex_buffer(0, vertex_buffer.buffer.slice(..));
                //pc.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
            }
        }
    }
}
