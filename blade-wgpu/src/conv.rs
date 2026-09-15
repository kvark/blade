//! wgpu ↔ blade type mapping.

use blade_graphics as gpu;

pub fn texture_format(format: wgpu::TextureFormat) -> gpu::TextureFormat {
    use wgpu::TextureFormat as W;
    match format {
        W::R8Unorm => gpu::TextureFormat::R8Unorm,
        W::R8Snorm => gpu::TextureFormat::R8Snorm,
        W::R8Uint => gpu::TextureFormat::R8Uint,
        W::Rg8Unorm => gpu::TextureFormat::Rg8Unorm,
        W::Rg8Snorm => gpu::TextureFormat::Rg8Snorm,
        W::Rgba8Unorm => gpu::TextureFormat::Rgba8Unorm,
        W::Rgba8UnormSrgb => gpu::TextureFormat::Rgba8UnormSrgb,
        W::Bgra8Unorm => gpu::TextureFormat::Bgra8Unorm,
        W::Bgra8UnormSrgb => gpu::TextureFormat::Bgra8UnormSrgb,
        W::Rgba8Snorm => gpu::TextureFormat::Rgba8Snorm,
        W::R16Uint => gpu::TextureFormat::R16Uint,
        W::Rg16Uint => gpu::TextureFormat::Rg16Uint,
        W::Rgba16Uint => gpu::TextureFormat::Rgba16Uint,
        W::R16Float => gpu::TextureFormat::R16Float,
        W::Rg16Float => gpu::TextureFormat::Rg16Float,
        W::Rgba16Float => gpu::TextureFormat::Rgba16Float,
        W::R32Float => gpu::TextureFormat::R32Float,
        W::Rg32Float => gpu::TextureFormat::Rg32Float,
        W::Rgba32Float => gpu::TextureFormat::Rgba32Float,
        W::R32Uint => gpu::TextureFormat::R32Uint,
        W::Rg32Uint => gpu::TextureFormat::Rg32Uint,
        W::Rgba32Uint => gpu::TextureFormat::Rgba32Uint,
        W::Depth16Unorm | W::Depth24Plus | W::Depth32Float => gpu::TextureFormat::Depth32Float,
        W::Depth24PlusStencil8 | W::Depth32FloatStencil8 => {
            gpu::TextureFormat::Depth32FloatStencil8Uint
        }
        W::Stencil8 => gpu::TextureFormat::Stencil8Uint,
        W::Bc1RgbaUnorm => gpu::TextureFormat::Bc1Unorm,
        W::Bc1RgbaUnormSrgb => gpu::TextureFormat::Bc1UnormSrgb,
        W::Bc2RgbaUnorm => gpu::TextureFormat::Bc2Unorm,
        W::Bc2RgbaUnormSrgb => gpu::TextureFormat::Bc2UnormSrgb,
        W::Bc3RgbaUnorm => gpu::TextureFormat::Bc3Unorm,
        W::Bc3RgbaUnormSrgb => gpu::TextureFormat::Bc3UnormSrgb,
        W::Bc4RUnorm => gpu::TextureFormat::Bc4Unorm,
        W::Bc4RSnorm => gpu::TextureFormat::Bc4Snorm,
        W::Bc5RgUnorm => gpu::TextureFormat::Bc5Unorm,
        W::Bc5RgSnorm => gpu::TextureFormat::Bc5Snorm,
        W::Bc6hRgbUfloat => gpu::TextureFormat::Bc6hUfloat,
        W::Bc6hRgbFloat => gpu::TextureFormat::Bc6hFloat,
        W::Bc7RgbaUnorm => gpu::TextureFormat::Bc7Unorm,
        W::Bc7RgbaUnormSrgb => gpu::TextureFormat::Bc7UnormSrgb,
        W::Rgb10a2Unorm => gpu::TextureFormat::Rgb10a2Unorm,
        W::Rg11b10Ufloat => gpu::TextureFormat::Rg11b10Ufloat,
        W::Rgb9e5Ufloat => gpu::TextureFormat::Rgb9e5Ufloat,
        other => crate::todo(&format!("texture format {other:?}")),
    }
}

pub fn texture_dimension(dim: wgpu::TextureDimension) -> gpu::TextureDimension {
    match dim {
        wgpu::TextureDimension::D1 => gpu::TextureDimension::D1,
        wgpu::TextureDimension::D2 => gpu::TextureDimension::D2,
        wgpu::TextureDimension::D3 => gpu::TextureDimension::D3,
    }
}

pub fn view_dimension(dim: wgpu::TextureViewDimension) -> gpu::ViewDimension {
    match dim {
        wgpu::TextureViewDimension::D1 => gpu::ViewDimension::D1,
        wgpu::TextureViewDimension::D2 => gpu::ViewDimension::D2,
        wgpu::TextureViewDimension::D2Array => gpu::ViewDimension::D2Array,
        wgpu::TextureViewDimension::Cube => gpu::ViewDimension::Cube,
        wgpu::TextureViewDimension::CubeArray => gpu::ViewDimension::CubeArray,
        wgpu::TextureViewDimension::D3 => gpu::ViewDimension::D3,
    }
}

pub fn texture_usage(usage: wgpu::TextureUsages) -> gpu::TextureUsage {
    let mut out = gpu::TextureUsage::empty();
    if usage.intersects(wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST) {
        out |= gpu::TextureUsage::COPY;
    }
    if usage.intersects(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        out |= gpu::TextureUsage::TARGET;
    }
    if usage.intersects(wgpu::TextureUsages::TEXTURE_BINDING) {
        out |= gpu::TextureUsage::RESOURCE;
    }
    if usage.intersects(wgpu::TextureUsages::STORAGE_BINDING) {
        out |= gpu::TextureUsage::STORAGE;
    }
    out
}

pub fn extent3d(size: wgpu::Extent3d) -> gpu::Extent {
    gpu::Extent {
        width: size.width.max(1),
        height: size.height.max(1),
        depth: size.depth_or_array_layers.max(1),
    }
}

pub fn address_mode(mode: wgpu::AddressMode) -> gpu::AddressMode {
    match mode {
        wgpu::AddressMode::ClampToEdge => gpu::AddressMode::ClampToEdge,
        wgpu::AddressMode::Repeat => gpu::AddressMode::Repeat,
        wgpu::AddressMode::MirrorRepeat => gpu::AddressMode::MirrorRepeat,
        wgpu::AddressMode::ClampToBorder => gpu::AddressMode::ClampToBorder,
    }
}

pub fn filter_mode(mode: wgpu::FilterMode) -> gpu::FilterMode {
    match mode {
        wgpu::FilterMode::Nearest => gpu::FilterMode::Nearest,
        wgpu::FilterMode::Linear => gpu::FilterMode::Linear,
    }
}

pub fn mipmap_filter(mode: wgpu::MipmapFilterMode) -> gpu::FilterMode {
    match mode {
        wgpu::MipmapFilterMode::Nearest => gpu::FilterMode::Nearest,
        wgpu::MipmapFilterMode::Linear => gpu::FilterMode::Linear,
    }
}

pub fn compare(fun: wgpu::CompareFunction) -> gpu::CompareFunction {
    match fun {
        wgpu::CompareFunction::Never => gpu::CompareFunction::Never,
        wgpu::CompareFunction::Less => gpu::CompareFunction::Less,
        wgpu::CompareFunction::Equal => gpu::CompareFunction::Equal,
        wgpu::CompareFunction::LessEqual => gpu::CompareFunction::LessEqual,
        wgpu::CompareFunction::Greater => gpu::CompareFunction::Greater,
        wgpu::CompareFunction::NotEqual => gpu::CompareFunction::NotEqual,
        wgpu::CompareFunction::GreaterEqual => gpu::CompareFunction::GreaterEqual,
        wgpu::CompareFunction::Always => gpu::CompareFunction::Always,
    }
}

pub fn vertex_format(format: wgpu::VertexFormat) -> gpu::VertexFormat {
    match format {
        wgpu::VertexFormat::Float32 => gpu::VertexFormat::F32,
        wgpu::VertexFormat::Float32x2 => gpu::VertexFormat::F32Vec2,
        wgpu::VertexFormat::Float32x3 => gpu::VertexFormat::F32Vec3,
        wgpu::VertexFormat::Float32x4 => gpu::VertexFormat::F32Vec4,
        wgpu::VertexFormat::Uint32 => gpu::VertexFormat::U32,
        wgpu::VertexFormat::Uint32x2 => gpu::VertexFormat::U32Vec2,
        wgpu::VertexFormat::Uint32x3 => gpu::VertexFormat::U32Vec3,
        wgpu::VertexFormat::Uint32x4 => gpu::VertexFormat::U32Vec4,
        wgpu::VertexFormat::Sint32 => gpu::VertexFormat::I32,
        wgpu::VertexFormat::Sint32x2 => gpu::VertexFormat::I32Vec2,
        wgpu::VertexFormat::Sint32x3 => gpu::VertexFormat::I32Vec3,
        wgpu::VertexFormat::Sint32x4 => gpu::VertexFormat::I32Vec4,
        other => crate::todo(&format!("vertex format {other:?}")),
    }
}

pub fn topology(t: wgpu::PrimitiveTopology) -> gpu::PrimitiveTopology {
    match t {
        wgpu::PrimitiveTopology::PointList => gpu::PrimitiveTopology::PointList,
        wgpu::PrimitiveTopology::LineList => gpu::PrimitiveTopology::LineList,
        wgpu::PrimitiveTopology::LineStrip => gpu::PrimitiveTopology::LineStrip,
        wgpu::PrimitiveTopology::TriangleList => gpu::PrimitiveTopology::TriangleList,
        wgpu::PrimitiveTopology::TriangleStrip => gpu::PrimitiveTopology::TriangleStrip,
    }
}

pub fn front_face(f: wgpu::FrontFace) -> gpu::FrontFace {
    match f {
        wgpu::FrontFace::Ccw => gpu::FrontFace::Ccw,
        wgpu::FrontFace::Cw => gpu::FrontFace::Cw,
    }
}

pub fn cull(face: wgpu::Face) -> gpu::Face {
    match face {
        wgpu::Face::Front => gpu::Face::Front,
        wgpu::Face::Back => gpu::Face::Back,
    }
}

pub fn blend_factor(f: wgpu::BlendFactor) -> gpu::BlendFactor {
    match f {
        wgpu::BlendFactor::Zero => gpu::BlendFactor::Zero,
        wgpu::BlendFactor::One => gpu::BlendFactor::One,
        wgpu::BlendFactor::Src => gpu::BlendFactor::Src,
        wgpu::BlendFactor::OneMinusSrc => gpu::BlendFactor::OneMinusSrc,
        wgpu::BlendFactor::SrcAlpha => gpu::BlendFactor::SrcAlpha,
        wgpu::BlendFactor::OneMinusSrcAlpha => gpu::BlendFactor::OneMinusSrcAlpha,
        wgpu::BlendFactor::Dst => gpu::BlendFactor::Dst,
        wgpu::BlendFactor::OneMinusDst => gpu::BlendFactor::OneMinusDst,
        wgpu::BlendFactor::DstAlpha => gpu::BlendFactor::DstAlpha,
        wgpu::BlendFactor::OneMinusDstAlpha => gpu::BlendFactor::OneMinusDstAlpha,
        wgpu::BlendFactor::SrcAlphaSaturated => gpu::BlendFactor::SrcAlphaSaturated,
        wgpu::BlendFactor::Constant => gpu::BlendFactor::Constant,
        wgpu::BlendFactor::OneMinusConstant => gpu::BlendFactor::OneMinusConstant,
        wgpu::BlendFactor::Src1 => gpu::BlendFactor::Src1,
        wgpu::BlendFactor::OneMinusSrc1 => gpu::BlendFactor::OneMinusSrc1,
        wgpu::BlendFactor::Src1Alpha => gpu::BlendFactor::Src1Alpha,
        wgpu::BlendFactor::OneMinusSrc1Alpha => gpu::BlendFactor::OneMinusSrc1Alpha,
        other => crate::todo(&format!("blend factor {other:?}")),
    }
}

pub fn blend_op(op: wgpu::BlendOperation) -> gpu::BlendOperation {
    match op {
        wgpu::BlendOperation::Add => gpu::BlendOperation::Add,
        wgpu::BlendOperation::Subtract => gpu::BlendOperation::Subtract,
        wgpu::BlendOperation::ReverseSubtract => gpu::BlendOperation::ReverseSubtract,
        wgpu::BlendOperation::Min => gpu::BlendOperation::Min,
        wgpu::BlendOperation::Max => gpu::BlendOperation::Max,
    }
}

pub fn blend_component(c: wgpu::BlendComponent) -> gpu::BlendComponent {
    gpu::BlendComponent {
        src_factor: blend_factor(c.src_factor),
        dst_factor: blend_factor(c.dst_factor),
        operation: blend_op(c.operation),
    }
}

pub fn color_writes(mask: wgpu::ColorWrites) -> gpu::ColorWrites {
    gpu::ColorWrites::from_bits_truncate(mask.bits())
}

pub fn stencil_op(op: wgpu::StencilOperation) -> gpu::StencilOperation {
    match op {
        wgpu::StencilOperation::Keep => gpu::StencilOperation::Keep,
        wgpu::StencilOperation::Zero => gpu::StencilOperation::Zero,
        wgpu::StencilOperation::Replace => gpu::StencilOperation::Replace,
        wgpu::StencilOperation::Invert => gpu::StencilOperation::Invert,
        wgpu::StencilOperation::IncrementClamp => gpu::StencilOperation::IncrementClamp,
        wgpu::StencilOperation::DecrementClamp => gpu::StencilOperation::DecrementClamp,
        wgpu::StencilOperation::IncrementWrap => gpu::StencilOperation::IncrementWrap,
        wgpu::StencilOperation::DecrementWrap => gpu::StencilOperation::DecrementWrap,
    }
}

pub fn stencil_face(s: wgpu::StencilFaceState) -> gpu::StencilFaceState {
    gpu::StencilFaceState {
        compare: compare(s.compare),
        fail_op: stencil_op(s.fail_op),
        depth_fail_op: stencil_op(s.depth_fail_op),
        pass_op: stencil_op(s.pass_op),
    }
}

pub fn color(c: wgpu::Color) -> gpu::TextureColor {
    gpu::TextureColor::Rgba([c.r as f32, c.g as f32, c.b as f32, c.a as f32])
}

pub fn index_type(format: wgpu::IndexFormat) -> gpu::IndexType {
    match format {
        wgpu::IndexFormat::Uint16 => gpu::IndexType::U16,
        wgpu::IndexFormat::Uint32 => gpu::IndexType::U32,
    }
}

pub fn advertised_features() -> wgpu::Features {
    wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        | wgpu::Features::FLOAT32_FILTERABLE
        | wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER
}

pub fn advertised_limits() -> wgpu::Limits {
    let mut limits = wgpu::Limits::default();
    limits.max_bind_groups = 8;
    limits.max_binding_array_elements_per_shader_stage = 0;
    limits.max_sampled_textures_per_shader_stage = 16;
    limits.max_samplers_per_shader_stage = 16;
    limits.max_storage_buffers_per_shader_stage = 16;
    limits.max_uniform_buffers_per_shader_stage = 16;
    limits.max_uniform_buffer_binding_size = 64 * 1024;
    limits.max_storage_buffer_binding_size = 1 << 28;
    limits.max_buffer_size = 1 << 30;
    limits.max_texture_dimension_1d = 8192;
    limits.max_texture_dimension_2d = 8192;
    limits.max_texture_dimension_3d = 2048;
    limits.max_texture_array_layers = 256;
    limits.max_color_attachments = 8;
    limits.max_color_attachment_bytes_per_sample = 32;
    limits.max_storage_textures_per_shader_stage = 16;
    limits.max_compute_workgroup_storage_size = 16384;
    limits.max_compute_invocations_per_workgroup = 256;
    limits.max_compute_workgroup_size_x = 256;
    limits.max_compute_workgroup_size_y = 256;
    limits.max_compute_workgroup_size_z = 64;
    limits.max_compute_workgroups_per_dimension = 65535;
    limits
}
