use std::{
    fmt, ptr, str,
    sync::{Arc, Mutex},
};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, blade_macros::Flat)]
struct TextureFormatWrap(blade::TextureFormat);

#[derive(blade_macros::Flat)]
pub struct CookedImage<'a> {
    name: &'a [u8],
    extent: [u32; 3],
    format: TextureFormatWrap,
    data: &'a [u8],
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Meta {
    pub format: blade::TextureFormat,
}

impl fmt::Display for Meta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.format, f)
    }
}

pub struct Texture {
    pub object: blade::Texture,
    pub view: blade::TextureView,
}

struct Initialization {
    dst: blade::Texture,
}

struct Transfer {
    stage: blade::Buffer,
    bytes_per_row: u32,
    dst: blade::Texture,
    extent: blade::Extent,
}

//TODO: consider this to be shared within the `AssetHub`?
#[derive(Default)]
struct PendingOperations {
    initializations: Vec<Initialization>,
    transfers: Vec<Transfer>,
}

pub struct Baker {
    gpu_context: Arc<blade::Context>,
    pending_operations: Mutex<PendingOperations>,
}

impl Baker {
    pub fn new(gpu_context: &Arc<blade::Context>) -> Self {
        Self {
            gpu_context: Arc::clone(gpu_context),
            pending_operations: Mutex::new(PendingOperations::default()),
        }
    }

    pub fn flush(
        &self,
        encoder: &mut blade::CommandEncoder,
        temp_buffers: &mut Vec<blade::Buffer>,
    ) {
        let mut pending_ops = self.pending_operations.lock().unwrap();
        for init in pending_ops.initializations.drain(..) {
            encoder.init_texture(init.dst);
        }
        if !pending_ops.transfers.is_empty() {
            let mut pass = encoder.transfer();
            for transfer in pending_ops.transfers.drain(..) {
                pass.copy_buffer_to_texture(
                    transfer.stage.into(),
                    transfer.bytes_per_row,
                    transfer.dst.into(),
                    transfer.extent,
                );
                temp_buffers.push(transfer.stage);
            }
        }
    }
}

impl blade_asset::Baker for Baker {
    type Meta = Meta;
    type Data<'a> = CookedImage<'a>;
    type Output = Texture;
    fn cook(
        &self,
        source: &[u8],
        extension: &str,
        meta: Meta,
        result: Arc<blade_asset::Cooked<CookedImage<'_>>>,
        _exe_context: choir::ExecutionContext,
    ) {
        use blade::TextureFormat as Tf;
        struct PlainImage {
            width: usize,
            height: usize,
            data: Box<[u8]>,
        }

        let src: PlainImage = match extension {
            #[cfg(feature = "asset")]
            "png" => {
                let options =
                    zune_core::options::DecoderOptions::default().png_set_add_alpha_channel(true);
                let mut decoder = zune_png::PngDecoder::new_with_options(source, options);
                let data = match decoder.decode().unwrap() {
                    zune_core::result::DecodingResult::U8(px) => px.into_boxed_slice(),
                    _ => panic!("Unsupported image"),
                };
                let info = decoder.get_info().unwrap();
                PlainImage {
                    width: info.width,
                    height: info.height,
                    data,
                }
            }
            #[cfg(feature = "asset")]
            "jpg" | "jpeg" => {
                let options = zune_core::options::DecoderOptions::default()
                    .jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGBA);
                let mut decoder = zune_jpeg::JpegDecoder::new_with_options(options, source);
                let data = decoder.decode().unwrap().into_boxed_slice();
                let info = decoder.info().unwrap();
                PlainImage {
                    width: info.width as usize,
                    height: info.height as usize,
                    data,
                }
            }
            other => panic!("Unknown texture extension: {}", other),
        };

        let mut buf = Vec::new();
        #[cfg(feature = "asset")]
        {
            profiling::scope!("compress");
            let dst_format = match meta.format {
                Tf::Bc1Unorm | Tf::Bc1UnormSrgb => texpresso::Format::Bc1,
                Tf::Bc2Unorm | Tf::Bc2UnormSrgb => texpresso::Format::Bc2,
                Tf::Bc3Unorm | Tf::Bc3UnormSrgb => texpresso::Format::Bc3,
                Tf::Bc4Unorm | Tf::Bc4Snorm => texpresso::Format::Bc4,
                Tf::Bc5Unorm | Tf::Bc5Snorm => texpresso::Format::Bc5,
                other => panic!("Unsupported destination format {:?}", other),
            };
            let dst_size = dst_format.compressed_size(src.width, src.height);
            buf.resize(dst_size, 0);
            let params = texpresso::Params::default();
            dst_format.compress(&src.data, src.width, src.height, params, &mut buf);
        }

        result.put(CookedImage {
            name: &[],
            extent: [src.width as u32, src.height as u32, 1],
            format: TextureFormatWrap(meta.format),
            data: &buf,
        });
    }

    fn serve(&self, image: CookedImage<'_>, _exe_context: choir::ExecutionContext) -> Self::Output {
        let name = str::from_utf8(image.name).unwrap();
        let extent = blade::Extent {
            width: image.extent[0],
            height: image.extent[1],
            depth: image.extent[2],
        };
        let texture = self.gpu_context.create_texture(blade::TextureDesc {
            name,
            format: image.format.0,
            size: extent,
            array_layer_count: 1,
            mip_level_count: 1, // TODO
            dimension: blade::TextureDimension::D2,
            usage: blade::TextureUsage::COPY | blade::TextureUsage::RESOURCE,
        });
        let stage = self.gpu_context.create_buffer(blade::BufferDesc {
            name: &format!("{name}/stage"),
            size: image.data.len() as u64,
            memory: blade::Memory::Upload,
        });
        unsafe {
            ptr::copy_nonoverlapping(image.data.as_ptr(), stage.data(), image.data.len());
        }

        let block_info = image.format.0.block_info();
        let mut pending_ops = self.pending_operations.lock().unwrap();
        pending_ops
            .initializations
            .push(Initialization { dst: texture });
        pending_ops.transfers.push(Transfer {
            stage,
            bytes_per_row: (image.extent[0] / block_info.dimensions.0 as u32)
                * block_info.size as u32,
            dst: texture,
            extent,
        });

        let view = self
            .gpu_context
            .create_texture_view(blade::TextureViewDesc {
                name,
                texture,
                format: image.format.0,
                dimension: blade::ViewDimension::D2,
                subresources: &Default::default(),
            });
        Texture {
            object: texture,
            view,
        }
    }

    fn delete(&self, texture: Self::Output) {
        self.gpu_context.destroy_texture_view(texture.view);
        self.gpu_context.destroy_texture(texture.object);
    }
}
