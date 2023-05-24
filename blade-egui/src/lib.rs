#![allow(
    irrefutable_let_patterns,
    clippy::new_without_default,
    // Conflicts with `pattern_type_mismatch`
    clippy::needless_borrowed_reference,
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

mod belt;

use belt::{BeltDescriptor, BufferBelt};
use std::{
    collections::hash_map::{Entry, HashMap},
    fs, mem, ptr,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Uniforms {
    screen_size: [f32; 2],
    padding: [f32; 2],
}

#[derive(blade_macros::ShaderData)]
struct Globals {
    r_uniforms: Uniforms,
    r_sampler: blade::Sampler,
}

#[derive(blade_macros::ShaderData)]
struct Locals {
    r_vertex_data: blade::BufferPiece,
    r_texture: blade::TextureView,
}

pub struct ScreenDescriptor {
    pub physical_size: (u32, u32),
    pub scale_factor: f32,
}

impl ScreenDescriptor {
    fn logical_size(&self) -> (f32, f32) {
        let logical_width = self.physical_size.0 as f32 / self.scale_factor;
        let logical_height = self.physical_size.1 as f32 / self.scale_factor;
        (logical_width, logical_height)
    }
}

struct GuiTexture {
    allocation: blade::Texture,
    view: blade::TextureView,
}

impl GuiTexture {
    fn create(context: &blade::Context, name: &str, size: blade::Extent) -> Self {
        let format = blade::TextureFormat::Rgba8UnormSrgb;
        let allocation = context.create_texture(blade::TextureDesc {
            name,
            format,
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: blade::TextureDimension::D2,
            usage: blade::TextureUsage::COPY | blade::TextureUsage::RESOURCE,
        });
        let view = context.create_texture_view(blade::TextureViewDesc {
            name,
            texture: allocation,
            format,
            dimension: blade::ViewDimension::D2,
            subresources: &blade::TextureSubresources::default(),
        });
        Self { allocation, view }
    }

    fn delete(self, context: &blade::Context) {
        context.destroy_texture(self.allocation);
        context.destroy_texture_view(self.view);
    }
}

//TODO: scissor test

/// GUI painter based on egui.
///
/// It can render egui primitives into a render pass.
pub struct GuiPainter {
    pipeline: blade::RenderPipeline,
    //TODO: find a better way to allocate temporary buffers.
    belt: BufferBelt,
    textures: HashMap<egui::TextureId, GuiTexture>,
    //TODO: this could also look better
    textures_dropped: Vec<GuiTexture>,
    textures_to_delete: Vec<(GuiTexture, blade::SyncPoint)>,
    sampler: blade::Sampler,
}

impl GuiPainter {
    /// Destroy the contents of the painter.
    pub fn destroy(&mut self, context: &blade::Context) {
        self.belt.destroy(context);
        for (_, gui_texture) in self.textures.drain() {
            gui_texture.delete(context);
        }
        context.destroy_sampler(self.sampler);
    }

    /// Create a new painter with a given GPU context.
    ///
    /// It supports renderpasses with only a color attachment,
    /// and this attachment format must be The `output_format`.
    pub fn new(context: &blade::Context, output_format: blade::TextureFormat) -> Self {
        let shader_source = fs::read_to_string("blade-egui/shader.wgsl").unwrap();
        let shader = context.create_shader(blade::ShaderDesc {
            source: &shader_source,
        });

        let globals_layout = <Globals as blade::ShaderData>::layout();
        let locals_layout = <Locals as blade::ShaderData>::layout();
        let pipeline = context.create_render_pipeline(blade::RenderPipelineDesc {
            name: "gui",
            data_layouts: &[&globals_layout, &locals_layout],
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

        let sampler = context.create_sampler(blade::SamplerDesc {
            name: "gui",
            address_modes: [blade::AddressMode::ClampToEdge; 3],
            mag_filter: blade::FilterMode::Linear,
            min_filter: blade::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            belt,
            textures: Default::default(),
            textures_dropped: Vec::new(),
            textures_to_delete: Vec::new(),
            sampler,
        }
    }

    fn triage_deletions(&mut self, context: &blade::Context) {
        let valid_pos = self
            .textures_to_delete
            .iter()
            .position(|&(_, ref sp)| !context.wait_for(sp, 0))
            .unwrap_or_default();
        for (texture, _) in self.textures_to_delete.drain(..valid_pos) {
            context.destroy_texture_view(texture.view);
            context.destroy_texture(texture.allocation);
        }
    }

    /// Updates the texture used by egui for the fonts etc.
    /// New textures should be added before the call to `execute()`,
    /// and old textures should be removed after.
    pub fn update_textures(
        &mut self,
        command_encoder: &mut blade::CommandEncoder,
        textures_delta: &egui::TexturesDelta,
        context: &blade::Context,
    ) {
        if textures_delta.set.is_empty() && textures_delta.free.is_empty() {
            return;
        }

        let mut copies = Vec::new();
        for &(texture_id, ref image_delta) in textures_delta.set.iter() {
            let src = match image_delta.image {
                egui::ImageData::Color(ref c) => self.belt.alloc_data(c.pixels.as_slice(), context),
                egui::ImageData::Font(ref a) => {
                    let color_iter = a.srgba_pixels(None);
                    let stage = self.belt.alloc(
                        (color_iter.len() * mem::size_of::<egui::Color32>()) as u64,
                        context,
                    );
                    let mut ptr = stage.data() as *mut egui::Color32;
                    for color in color_iter {
                        unsafe {
                            ptr::write(ptr, color);
                            ptr = ptr.offset(1);
                        }
                    }
                    stage
                }
            };

            let image_size = image_delta.image.size();
            let extent = blade::Extent {
                width: image_size[0] as u32,
                height: image_size[1] as u32,
                depth: 1,
            };

            let label = match texture_id {
                egui::TextureId::Managed(m) => format!("egui_image_{}", m),
                egui::TextureId::User(u) => format!("egui_user_image_{}", u),
            };

            let texture = match self.textures.entry(texture_id) {
                Entry::Occupied(mut o) => {
                    if image_delta.pos.is_none() {
                        let texture = GuiTexture::create(context, &label, extent);
                        command_encoder.init_texture(texture.allocation);
                        let old = o.insert(texture);
                        self.textures_dropped.push(old);
                    }
                    o.into_mut()
                }
                Entry::Vacant(v) => {
                    let texture = GuiTexture::create(context, &label, extent);
                    command_encoder.init_texture(texture.allocation);
                    v.insert(texture)
                }
            };

            let dst = blade::TexturePiece {
                texture: texture.allocation,
                mip_level: 0,
                array_layer: 0,
                origin: match image_delta.pos {
                    Some([x, y]) => [x as u32, y as u32, 0],
                    None => [0; 3],
                },
            };
            copies.push((src, dst, extent));
        }

        if let mut transfer = command_encoder.transfer() {
            for (src, dst, extent) in copies {
                transfer.copy_buffer_to_texture(src, 4 * extent.width, dst, extent);
            }
        }

        for texture_id in textures_delta.free.iter() {
            let texture = self.textures.remove(texture_id).unwrap();
            self.textures_dropped.push(texture);
        }

        self.triage_deletions(context);
    }

    /// Render the set of clipped primitives into a render pass.
    /// The `sd` must contain dimensions of the render target.
    pub fn paint(
        &mut self,
        pass: &mut blade::RenderCommandEncoder,
        paint_jobs: &[egui::epaint::ClippedPrimitive],
        sd: &ScreenDescriptor,
        context: &blade::Context,
    ) {
        let logical_size = sd.logical_size();
        let mut pc = pass.with(&self.pipeline);
        pc.bind(
            0,
            &Globals {
                r_uniforms: Uniforms {
                    screen_size: [logical_size.0, logical_size.1],
                    padding: [0.0; 2],
                },
                r_sampler: self.sampler,
            },
        );

        for clipped_prim in paint_jobs {
            let clip_rect = &clipped_prim.clip_rect;

            // Make sure clip rect can fit within an `u32`.
            let clip_min_x = (sd.scale_factor * clip_rect.min.x)
                .clamp(0.0, sd.physical_size.0 as f32)
                .trunc() as u32;
            let clip_min_y = (sd.scale_factor * clip_rect.min.y)
                .clamp(0.0, sd.physical_size.1 as f32)
                .trunc() as u32;
            let clip_max_x = (sd.scale_factor * clip_rect.max.x)
                .clamp(0.0, sd.physical_size.0 as f32)
                .ceil() as u32;
            let clip_max_y = (sd.scale_factor * clip_rect.max.y)
                .clamp(0.0, sd.physical_size.1 as f32)
                .ceil() as u32;

            if clip_max_x <= clip_min_x || clip_max_y == clip_min_y {
                continue;
            }

            pc.set_scissor_rect(&blade::ScissorRect {
                x: clip_min_x,
                y: clip_min_y,
                w: clip_max_x - clip_min_x,
                h: clip_max_y - clip_min_y,
            });

            if let egui::epaint::Primitive::Mesh(ref mesh) = clipped_prim.primitive {
                let texture = self.textures.get(&mesh.texture_id).unwrap();
                let index_buf = self.belt.alloc_data(&mesh.indices, context);
                let vertex_buf = self.belt.alloc_data(&mesh.vertices, context);

                pc.bind(
                    1,
                    &Locals {
                        r_vertex_data: vertex_buf,
                        r_texture: texture.view,
                    },
                );

                pc.draw_indexed(
                    index_buf,
                    blade::IndexType::U32,
                    mesh.indices.len() as u32,
                    0,
                    0,
                    1,
                );
            }
        }
    }

    /// Call this after submitting work at the given `sync_point`.
    pub fn after_submit(&mut self, sync_point: blade::SyncPoint) {
        self.textures_to_delete.extend(
            self.textures_dropped
                .drain(..)
                .map(|texture| (texture, sync_point.clone())),
        );
        self.belt.flush(sync_point);
    }
}
