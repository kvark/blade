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

const SHADER_SOURCE: &'static str = include_str!("../shader.wgsl");

use blade_util::{BufferBelt, BufferBeltDescriptor};
use std::{
    collections::hash_map::{Entry, HashMap},
    mem::size_of,
    ptr,
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
}

#[derive(blade_macros::ShaderData)]
struct Locals {
    r_vertex_data: blade_graphics::BufferPiece,
    r_texture: blade_graphics::TextureView,
    r_sampler: blade_graphics::Sampler,
}

#[derive(Debug, PartialEq)]
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
    allocation: blade_graphics::Texture,
    view: blade_graphics::TextureView,
    sampler: blade_graphics::Sampler,
}

impl GuiTexture {
    fn create(
        context: &blade_graphics::Context,
        name: &str,
        size: blade_graphics::Extent,
        options: egui::TextureOptions,
    ) -> Self {
        let format = blade_graphics::TextureFormat::Rgba8UnormSrgb;
        let allocation = context.create_texture(blade_graphics::TextureDesc {
            name,
            format,
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: blade_graphics::TextureDimension::D2,
            usage: blade_graphics::TextureUsage::COPY | blade_graphics::TextureUsage::RESOURCE,
        });
        let view = context.create_texture_view(
            allocation,
            blade_graphics::TextureViewDesc {
                name,
                format,
                dimension: blade_graphics::ViewDimension::D2,
                subresources: &blade_graphics::TextureSubresources::default(),
            },
        );

        let sampler = context.create_sampler(blade_graphics::SamplerDesc {
            name,
            address_modes: {
                let mode = match options.wrap_mode {
                    egui::TextureWrapMode::ClampToEdge => blade_graphics::AddressMode::ClampToEdge,
                    egui::TextureWrapMode::Repeat => blade_graphics::AddressMode::Repeat,
                    egui::TextureWrapMode::MirroredRepeat => {
                        blade_graphics::AddressMode::MirrorRepeat
                    }
                };
                [mode; 3]
            },
            mag_filter: match options.magnification {
                egui::TextureFilter::Nearest => blade_graphics::FilterMode::Nearest,
                egui::TextureFilter::Linear => blade_graphics::FilterMode::Linear,
            },
            min_filter: match options.minification {
                egui::TextureFilter::Nearest => blade_graphics::FilterMode::Nearest,
                egui::TextureFilter::Linear => blade_graphics::FilterMode::Linear,
            },
            mipmap_filter: match options.mipmap_mode {
                Some(it) => match it {
                    egui::TextureFilter::Nearest => blade_graphics::FilterMode::Nearest,
                    egui::TextureFilter::Linear => blade_graphics::FilterMode::Linear,
                },
                None => blade_graphics::FilterMode::Linear,
            },
            ..Default::default()
        });

        Self {
            allocation,
            view,
            sampler,
        }
    }

    fn delete(self, context: &blade_graphics::Context) {
        context.destroy_texture(self.allocation);
        context.destroy_texture_view(self.view);
        context.destroy_sampler(self.sampler);
    }
}

//TODO: scissor test

/// GUI painter based on egui.
///
/// It can render egui primitives into a render pass.
pub struct GuiPainter {
    pipeline: blade_graphics::RenderPipeline,
    //TODO: find a better way to allocate temporary buffers.
    belt: BufferBelt,
    textures: HashMap<egui::TextureId, GuiTexture>,
    //TODO: this could also look better
    textures_dropped: Vec<GuiTexture>,
    textures_to_delete: Vec<(GuiTexture, blade_graphics::SyncPoint)>,
}

impl GuiPainter {
    /// Destroy the contents of the painter.
    pub fn destroy(&mut self, context: &blade_graphics::Context) {
        context.destroy_render_pipeline(&mut self.pipeline);
        self.belt.destroy(context);
        for (_, gui_texture) in self.textures.drain() {
            gui_texture.delete(context);
        }
        for gui_texture in self.textures_dropped.drain(..) {
            gui_texture.delete(context);
        }
        for (gui_texture, _) in self.textures_to_delete.drain(..) {
            gui_texture.delete(context);
        }
    }

    /// Create a new painter with a given GPU context.
    ///
    /// It supports renderpasses with only a color attachment,
    /// and this attachment format must be The `output_format`.
    #[profiling::function]
    pub fn new(info: blade_graphics::SurfaceInfo, context: &blade_graphics::Context) -> Self {
        let shader = context.create_shader(blade_graphics::ShaderDesc {
            source: SHADER_SOURCE,
        });
        let globals_layout = <Globals as blade_graphics::ShaderData>::layout();
        let locals_layout = <Locals as blade_graphics::ShaderData>::layout();
        let pipeline = context.create_render_pipeline(blade_graphics::RenderPipelineDesc {
            name: "gui",
            data_layouts: &[&globals_layout, &locals_layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[],
            primitive: blade_graphics::PrimitiveState {
                topology: blade_graphics::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None, //TODO?
            fragment: shader.at("fs_main"),
            color_targets: &[blade_graphics::ColorTargetState {
                format: info.format,
                blend: Some(blade_graphics::BlendState::ALPHA_BLENDING),
                write_mask: blade_graphics::ColorWrites::all(),
            }],
        });

        let belt = BufferBelt::new(BufferBeltDescriptor {
            memory: blade_graphics::Memory::Shared,
            min_chunk_size: 0x1000,
            alignment: 4,
        });

        Self {
            pipeline,
            belt,
            textures: Default::default(),
            textures_dropped: Vec::new(),
            textures_to_delete: Vec::new(),
        }
    }

    #[profiling::function]
    fn triage_deletions(&mut self, context: &blade_graphics::Context) {
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
    #[profiling::function]
    pub fn update_textures(
        &mut self,
        command_encoder: &mut blade_graphics::CommandEncoder,
        textures_delta: &egui::TexturesDelta,
        context: &blade_graphics::Context,
    ) {
        if textures_delta.set.is_empty() && textures_delta.free.is_empty() {
            return;
        }

        let mut copies = Vec::new();
        for &(texture_id, ref image_delta) in textures_delta.set.iter() {
            let src = match image_delta.image {
                egui::ImageData::Color(ref c) => self.belt.alloc_pod(c.pixels.as_slice(), context),
                egui::ImageData::Font(ref a) => {
                    let color_iter = a.srgba_pixels(None);
                    let stage = self.belt.alloc(
                        (color_iter.len() * size_of::<egui::Color32>()) as u64,
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
            let extent = blade_graphics::Extent {
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
                        let texture =
                            GuiTexture::create(context, &label, extent, image_delta.options);
                        command_encoder.init_texture(texture.allocation);
                        let old = o.insert(texture);
                        self.textures_dropped.push(old);
                    }
                    o.into_mut()
                }
                Entry::Vacant(v) => {
                    let texture = GuiTexture::create(context, &label, extent, image_delta.options);
                    command_encoder.init_texture(texture.allocation);
                    v.insert(texture)
                }
            };

            let dst = blade_graphics::TexturePiece {
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

        if let mut transfer = command_encoder.transfer("update egui textures") {
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
    #[profiling::function]
    pub fn paint(
        &mut self,
        pass: &mut blade_graphics::RenderCommandEncoder,
        paint_jobs: &[egui::epaint::ClippedPrimitive],
        sd: &ScreenDescriptor,
        context: &blade_graphics::Context,
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
            },
        );

        for clipped_prim in paint_jobs {
            let clip_rect = &clipped_prim.clip_rect;

            // Make sure clip rect can fit within an `u32`.
            let clip_min_x = (sd.scale_factor * clip_rect.min.x)
                .clamp(0.0, sd.physical_size.0 as f32)
                .trunc() as i32;
            let clip_min_y = (sd.scale_factor * clip_rect.min.y)
                .clamp(0.0, sd.physical_size.1 as f32)
                .trunc() as i32;
            let clip_max_x = (sd.scale_factor * clip_rect.max.x)
                .clamp(0.0, sd.physical_size.0 as f32)
                .ceil() as i32;
            let clip_max_y = (sd.scale_factor * clip_rect.max.y)
                .clamp(0.0, sd.physical_size.1 as f32)
                .ceil() as i32;

            if clip_max_x <= clip_min_x || clip_max_y == clip_min_y {
                continue;
            }

            pc.set_scissor_rect(&blade_graphics::ScissorRect {
                x: clip_min_x,
                y: clip_min_y,
                w: (clip_max_x - clip_min_x) as u32,
                h: (clip_max_y - clip_min_y) as u32,
            });

            if let egui::epaint::Primitive::Mesh(ref mesh) = clipped_prim.primitive {
                let texture = self.textures.get(&mesh.texture_id).unwrap();
                let index_buf = self.belt.alloc_pod(&mesh.indices, context);
                let vertex_buf = self.belt.alloc_pod(&mesh.vertices, context);

                pc.bind(
                    1,
                    &Locals {
                        r_vertex_data: vertex_buf,
                        r_texture: texture.view,
                        r_sampler: texture.sampler,
                    },
                );

                pc.draw_indexed(
                    index_buf,
                    blade_graphics::IndexType::U32,
                    mesh.indices.len() as u32,
                    0,
                    0,
                    1,
                );
            }
        }
    }

    /// Call this after submitting work at the given `sync_point`.
    #[profiling::function]
    pub fn after_submit(&mut self, sync_point: &blade_graphics::SyncPoint) {
        self.textures_to_delete.extend(
            self.textures_dropped
                .drain(..)
                .map(|texture| (texture, sync_point.clone())),
        );
        self.belt.flush(sync_point);
    }
}
