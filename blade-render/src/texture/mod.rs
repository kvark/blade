use std::{
    fmt, io, mem, ptr, slice, str,
    sync::{Arc, Mutex},
};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, blade_macros::Flat)]
struct TextureFormatWrap(blade_graphics::TextureFormat);

#[derive(blade_macros::Flat)]
pub struct CookedImage<'a> {
    name: &'a [u8],
    extent: [u32; 3],
    format: TextureFormatWrap,
    data: &'a [u8],
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Meta {
    pub format: blade_graphics::TextureFormat,
}

impl fmt::Display for Meta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.format, f)
    }
}

pub struct Texture {
    pub object: blade_graphics::Texture,
    pub view: blade_graphics::TextureView,
}

struct Initialization {
    dst: blade_graphics::Texture,
}

struct Transfer {
    stage: blade_graphics::Buffer,
    bytes_per_row: u32,
    dst: blade_graphics::Texture,
    extent: blade_graphics::Extent,
}

//TODO: consider this to be shared within the `AssetHub`?
#[derive(Default)]
struct PendingOperations {
    initializations: Vec<Initialization>,
    transfers: Vec<Transfer>,
}

pub struct Baker {
    gpu_context: Arc<blade_graphics::Context>,
    pending_operations: Mutex<PendingOperations>,
}

impl Baker {
    pub fn new(gpu_context: &Arc<blade_graphics::Context>) -> Self {
        Self {
            gpu_context: Arc::clone(gpu_context),
            pending_operations: Mutex::new(PendingOperations::default()),
        }
    }

    pub fn flush(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        temp_buffers: &mut Vec<blade_graphics::Buffer>,
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
        use blade_graphics::TextureFormat as Tf;
        enum PlainData {
            Ldr(Vec<[u8; 4]>),
            Hdr(Vec<[f32; 4]>),
        }
        struct PlainImage {
            width: usize,
            height: usize,
            data: PlainData,
        }

        let src: PlainImage = match extension {
            #[cfg(feature = "asset")]
            "png" => {
                let options =
                    zune_core::options::DecoderOptions::default().png_set_add_alpha_channel(true);
                let mut decoder = zune_png::PngDecoder::new_with_options(source, options);
                decoder.decode_headers().unwrap();
                let info = decoder.get_info().unwrap().clone();
                let mut data = vec![[0u8; 4]; info.width * info.height];
                decoder
                    .decode_into(unsafe {
                        slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
                    })
                    .unwrap();
                PlainImage {
                    width: info.width,
                    height: info.height,
                    data: PlainData::Ldr(data),
                }
            }
            #[cfg(feature = "asset")]
            "jpg" | "jpeg" => {
                let options = zune_core::options::DecoderOptions::default()
                    .jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGBA);
                let mut decoder = zune_jpeg::JpegDecoder::new_with_options(options, source);
                decoder.decode_headers().unwrap();
                let info = decoder.info().unwrap();
                let mut data = vec![[0u8; 4]; info.width as usize * info.height as usize];
                decoder
                    .decode_into(unsafe {
                        slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
                    })
                    .unwrap();
                PlainImage {
                    width: info.width as usize,
                    height: info.height as usize,
                    data: PlainData::Ldr(data),
                }
            }
            #[cfg(feature = "asset")]
            "exr" => {
                use exr::prelude::{ReadChannels as _, ReadLayers as _};
                struct RawImage {
                    width: usize,
                    data: Vec<[f32; 4]>,
                }
                let image = exr::image::read::read()
                    .no_deep_data()
                    .largest_resolution_level()
                    .rgba_channels(
                        |size, _| RawImage {
                            width: size.width(),
                            data: vec![[0f32; 4]; size.width() * size.height()],
                        },
                        |image, position, (r, g, b, a): (f32, f32, f32, f32)| {
                            image.data[position.y() * image.width + position.x()] = [r, g, b, a];
                        },
                    )
                    .first_valid_layer()
                    .all_attributes()
                    .from_buffered(io::Cursor::new(source))
                    .unwrap();
                PlainImage {
                    width: image.layer_data.size.width(),
                    height: image.layer_data.size.height(),
                    data: PlainData::Hdr(image.layer_data.channel_data.pixels.data),
                }
            }
            other => panic!("Unknown texture extension: {}", other),
        };

        let mut buf = Vec::new();
        #[cfg(feature = "asset")]
        match src.data {
            PlainData::Ldr(data) => {
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
                let data_raw =
                    unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
                dst_format.compress(data_raw, src.width, src.height, params, &mut buf);
            }
            PlainData::Hdr(data) => {
                //TODO: compress as BC6E
                assert_eq!(meta.format, blade_graphics::TextureFormat::Rgba32Float);
                let data_raw = unsafe {
                    slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * data[0].len() * mem::size_of::<f32>(),
                    )
                };
                buf.resize(data_raw.len(), 0);
                buf.copy_from_slice(data_raw);
            }
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
        let extent = blade_graphics::Extent {
            width: image.extent[0],
            height: image.extent[1],
            depth: image.extent[2],
        };
        let texture = self
            .gpu_context
            .create_texture(blade_graphics::TextureDesc {
                name,
                format: image.format.0,
                size: extent,
                array_layer_count: 1,
                mip_level_count: 1, // TODO
                dimension: blade_graphics::TextureDimension::D2,
                usage: blade_graphics::TextureUsage::COPY | blade_graphics::TextureUsage::RESOURCE,
            });
        let stage = self.gpu_context.create_buffer(blade_graphics::BufferDesc {
            name: &format!("{name}/stage"),
            size: image.data.len() as u64,
            memory: blade_graphics::Memory::Upload,
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
            .create_texture_view(blade_graphics::TextureViewDesc {
                name,
                texture,
                format: image.format.0,
                dimension: blade_graphics::ViewDimension::D2,
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
