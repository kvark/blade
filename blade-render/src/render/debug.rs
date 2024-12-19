use std::{cell::Cell, mem, ptr};

#[derive(Clone, Copy, Debug, Default)]
pub struct DebugBlit {
    pub input: blade_graphics::TextureView,
    pub mip_level: u32,
    pub target_offset: [i32; 2],
    pub target_size: [u32; 2],
}

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

#[derive(blade_macros::ShaderData)]
struct DebugDrawData {
    camera: super::CameraParams,
    debug_lines: blade_graphics::BufferPiece,
    depth: blade_graphics::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct DebugBlitParams {
    target_offset: [f32; 2],
    target_size: [f32; 2],
    mip_level: f32,
    unused: u32,
}

#[derive(blade_macros::ShaderData)]
struct DebugBlitData {
    input: blade_graphics::TextureView,
    samp: blade_graphics::Sampler,
    params: DebugBlitParams,
}

// Has to match the shader!
#[repr(C)]
#[derive(Debug)]
pub struct DebugVariance {
    pub color_sum: [f32; 3],
    pad: u32,
    pub color2_sum: [f32; 3],
    pub count: u32,
}

// Has to match the shader!
#[repr(C)]
#[derive(Debug)]
pub struct DebugEntry {
    pub custom_index: u32,
    pub depth: f32,
    pub tex_coords: [f32; 2],
    pub base_color_texture: u32,
    pub normal_texture: u32,
    pad: [u32; 2],
    pub position: [f32; 3],
    position_w: f32,
    pub normal: [f32; 3],
    normal_w: f32,
}

fn create_draw_pipeline(
    shader: &blade_graphics::Shader,
    format: blade_graphics::TextureFormat,
    gpu: &blade_graphics::Context,
) -> blade_graphics::RenderPipeline {
    shader.check_struct_size::<DebugPoint>();
    shader.check_struct_size::<DebugLine>();
    let layout = <DebugDrawData as blade_graphics::ShaderData>::layout();
    gpu.create_render_pipeline(blade_graphics::RenderPipelineDesc {
        name: "debug-draw",
        data_layouts: &[&layout],
        vertex: shader.at("debug_vs"),
        vertex_fetches: &[],
        primitive: blade_graphics::PrimitiveState {
            topology: blade_graphics::PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: None,
        fragment: Some(shader.at("debug_fs")),
        color_targets: &[blade_graphics::ColorTargetState {
            format,
            blend: Some(blade_graphics::BlendState::ALPHA_BLENDING),
            write_mask: blade_graphics::ColorWrites::all(),
        }],
        multisample_state: blade_graphics::MultisampleState::default(),
    })
}

fn create_blit_pipeline(
    shader: &blade_graphics::Shader,
    format: blade_graphics::TextureFormat,
    gpu: &blade_graphics::Context,
) -> blade_graphics::RenderPipeline {
    shader.check_struct_size::<DebugBlitParams>();
    let layout = <DebugBlitData as blade_graphics::ShaderData>::layout();
    gpu.create_render_pipeline(blade_graphics::RenderPipelineDesc {
        name: "debug-blit",
        data_layouts: &[&layout],
        vertex: shader.at("blit_vs"),
        vertex_fetches: &[],
        primitive: blade_graphics::PrimitiveState {
            topology: blade_graphics::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        fragment: Some(shader.at("blit_fs")),
        color_targets: &[format.into()],
        multisample_state: blade_graphics::MultisampleState::default(),
    })
}

pub struct DebugRender {
    capacity: u32,
    surface_format: blade_graphics::TextureFormat,
    buffer: blade_graphics::Buffer,
    variance_buffer: blade_graphics::Buffer,
    entry_buffer: blade_graphics::Buffer,
    cpu_lines_buffer: blade_graphics::Buffer,
    //Note: allows immutable `add_lines`
    cpu_lines_offset: Cell<u64>,
    draw_pipeline: blade_graphics::RenderPipeline,
    blit_pipeline: blade_graphics::RenderPipeline,
    line_size: u32,
    buffer_size: u32,
}

impl DebugRender {
    pub(super) fn init(
        encoder: &mut blade_graphics::CommandEncoder,
        gpu: &blade_graphics::Context,
        shader_draw: &blade_graphics::Shader,
        shader_blit: &blade_graphics::Shader,
        capacity: u32,
        surface_info: blade_graphics::SurfaceInfo,
    ) -> Self {
        let line_size = shader_draw.get_struct_size("DebugLine");
        let buffer_size = shader_draw.get_struct_size("DebugBuffer");
        let this = Self {
            capacity,
            surface_format: surface_info.format,
            buffer: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "debug",
                size: (buffer_size + capacity.saturating_sub(1) * line_size) as u64,
                memory: blade_graphics::Memory::Device,
            }),
            variance_buffer: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "variance",
                size: mem::size_of::<DebugVariance>() as u64,
                memory: blade_graphics::Memory::Shared,
            }),
            entry_buffer: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "debug entry",
                size: mem::size_of::<DebugEntry>() as u64,
                memory: blade_graphics::Memory::Shared,
            }),
            cpu_lines_buffer: gpu.create_buffer(blade_graphics::BufferDesc {
                name: "CPU debug lines",
                size: (capacity * line_size) as u64,
                memory: blade_graphics::Memory::Shared,
            }),
            cpu_lines_offset: Cell::new(0),
            draw_pipeline: create_draw_pipeline(shader_draw, surface_info.format, gpu),
            blit_pipeline: create_blit_pipeline(shader_blit, surface_info.format, gpu),
            line_size,
            buffer_size,
        };

        let init_data = [2u32, 0, 0, 0, capacity];
        let init_size = init_data.len() * mem::size_of::<u32>();
        assert!(init_size <= mem::size_of::<DebugEntry>());
        unsafe {
            ptr::write_bytes(
                this.variance_buffer.data(),
                0,
                mem::size_of::<DebugVariance>(),
            );
            ptr::write_bytes(this.entry_buffer.data(), 0, mem::size_of::<DebugEntry>());
            // piggyback on the staging buffers to upload the data
            ptr::copy_nonoverlapping(
                init_data.as_ptr(),
                this.entry_buffer.data() as *mut u32,
                init_data.len(),
            );
        }

        let mut transfers = encoder.transfer("upload debug");
        transfers.copy_buffer_to_buffer(
            this.entry_buffer.at(0),
            this.buffer.at(0),
            init_size as u64,
        );

        this
    }

    pub(super) fn destroy(&mut self, gpu: &blade_graphics::Context) {
        gpu.destroy_buffer(self.buffer);
        gpu.destroy_buffer(self.variance_buffer);
        gpu.destroy_buffer(self.entry_buffer);
        gpu.destroy_buffer(self.cpu_lines_buffer);
        gpu.destroy_render_pipeline(&mut self.draw_pipeline);
        gpu.destroy_render_pipeline(&mut self.blit_pipeline);
    }

    pub(super) fn recreate_draw_pipeline(
        &mut self,
        shader: &blade_graphics::Shader,
        gpu: &blade_graphics::Context,
    ) {
        assert_eq!(shader.get_struct_size("DebugLine"), self.line_size);
        assert_eq!(shader.get_struct_size("DebugBuffer"), self.buffer_size);
        self.draw_pipeline = create_draw_pipeline(shader, self.surface_format, gpu);
    }

    pub(super) fn recreate_blit_pipeline(
        &mut self,
        shader: &blade_graphics::Shader,
        gpu: &blade_graphics::Context,
    ) {
        self.draw_pipeline = create_blit_pipeline(shader, self.surface_format, gpu);
    }

    fn add_lines(&self, lines: &[DebugLine]) -> (blade_graphics::BufferPiece, u32) {
        let required_size = lines.len() as u64 * self.line_size as u64;
        let old_offset = self.cpu_lines_offset.get();
        let (original_offset, count) =
            if old_offset + required_size <= (self.capacity * self.line_size) as u64 {
                (old_offset, lines.len())
            } else {
                let count = lines.len().min(self.capacity as usize);
                if count < lines.len() {
                    log::warn!("Reducing the debug lines from {} to {}", lines.len(), count);
                }
                (0, count)
            };

        unsafe {
            ptr::copy_nonoverlapping(
                lines.as_ptr(),
                self.cpu_lines_buffer.data().add(original_offset as usize) as *mut DebugLine,
                count,
            );
        }

        self.cpu_lines_offset
            .set(original_offset + count as u64 * self.line_size as u64);
        (self.cpu_lines_buffer.at(original_offset), count as u32)
    }

    pub(super) fn render_lines(
        &self,
        debug_lines: &[DebugLine],
        camera: super::CameraParams,
        depth: blade_graphics::TextureView,
        pass: &mut blade_graphics::RenderCommandEncoder,
    ) {
        let mut pc = pass.with(&self.draw_pipeline);
        let lines_offset = 32 + mem::size_of::<DebugVariance>() + mem::size_of::<DebugEntry>();
        pc.bind(
            0,
            &DebugDrawData {
                camera,
                debug_lines: self.buffer.at(lines_offset as u64),
                depth,
            },
        );
        pc.draw_indirect(self.buffer.at(0));

        if !debug_lines.is_empty() {
            let (lines_buf, count) = self.add_lines(debug_lines);
            pc.bind(
                0,
                &DebugDrawData {
                    camera,
                    debug_lines: lines_buf,
                    depth,
                },
            );
            pc.draw(0, 2, 0, count);
        }
    }

    pub(super) fn render_blits(
        &self,
        debug_blits: &[DebugBlit],
        samp: blade_graphics::Sampler,
        screen_size: blade_graphics::Extent,
        pass: &mut blade_graphics::RenderCommandEncoder,
    ) {
        let mut pc = pass.with(&self.blit_pipeline);
        for db in debug_blits {
            pc.bind(
                0,
                &DebugBlitData {
                    input: db.input,
                    samp,
                    params: DebugBlitParams {
                        target_offset: [
                            db.target_offset[0] as f32 / screen_size.width as f32,
                            db.target_offset[1] as f32 / screen_size.height as f32,
                        ],
                        target_size: [
                            db.target_size[0] as f32 / screen_size.width as f32,
                            db.target_size[1] as f32 / screen_size.height as f32,
                        ],
                        mip_level: db.mip_level as f32,
                        unused: 0,
                    },
                },
            );
            pc.draw(0, 4, 0, 1);
        }
    }

    pub fn buffer_resource(&self) -> blade_graphics::BufferPiece {
        self.buffer.into()
    }

    pub fn enable_draw(&self, transfer: &mut blade_graphics::TransferCommandEncoder, enable: bool) {
        transfer.fill_buffer(self.buffer.at(20), 4, enable as _);
    }

    pub fn reset_lines(&self, transfer: &mut blade_graphics::TransferCommandEncoder) {
        transfer.fill_buffer(self.buffer.at(4), 4, 0);
    }

    pub fn reset_variance(&self, transfer: &mut blade_graphics::TransferCommandEncoder) {
        transfer.fill_buffer(
            self.buffer.at(32),
            mem::size_of::<DebugVariance>() as u64,
            0,
        );
    }

    /// Copy the previous frame variance into the CPU-shared buffer.
    pub fn update_variance(&self, transfer: &mut blade_graphics::TransferCommandEncoder) {
        transfer.copy_buffer_to_buffer(
            self.buffer.at(32),
            self.variance_buffer.into(),
            mem::size_of::<DebugVariance>() as u64,
        );
    }

    /// Copy the previous frame entry into the CPU-shared buffer.
    pub fn update_entry(&self, transfer: &mut blade_graphics::TransferCommandEncoder) {
        transfer.copy_buffer_to_buffer(
            self.buffer.at(32 + mem::size_of::<DebugVariance>() as u64),
            self.entry_buffer.into(),
            mem::size_of::<DebugEntry>() as u64,
        );
    }

    pub fn read_shared_data(&self) -> (&DebugVariance, &DebugEntry) {
        let db_v = unsafe { &*(self.variance_buffer.data() as *const DebugVariance) };
        let db_e = unsafe { &*(self.entry_buffer.data() as *const DebugEntry) };
        (db_v, db_e)
    }
}
