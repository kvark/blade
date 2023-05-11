use blade_asset::Flat as _;
use std::{
    io, ptr, str,
    sync::{Arc, Mutex},
};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, blade_macros::Flat)]
struct TextureFormatWrap(blade::TextureFormat);

#[derive(blade_macros::Flat)]
pub struct CompressedImage<'a> {
    name: &'a [u8],
    extent: [u32; 3],
    format: TextureFormatWrap,
    data: &'a [u8],
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Meta {
    pub format: blade::TextureFormat,
}

#[cfg(feature = "asset")]
struct PlainImage {
    width: usize,
    height: usize,
    data: Box<[u8]>,
}

#[cfg(feature = "asset")]
impl PlainImage {
    fn from_png(input: &[u8]) -> Self {
        let mut decoder = png::Decoder::new(io::Cursor::new(input));
        decoder.set_transformations(png::Transformations::EXPAND);

        let mut reader = decoder
            .read_info()
            .expect("Failed to read PNG header. Is this really a PNG file?");

        // Preallocate the output buffer.
        let mut buf = vec![0; reader.output_buffer_size()];

        // Read the next frame. Currently this function should only called once.
        reader.next_frame(&mut buf).unwrap();

        let info = reader.info();
        if info.bit_depth != png::BitDepth::Eight {
            panic!("Only images with 8 bits per channel are supported");
        }

        // expand to rgba
        buf = match info.color_type {
            png::ColorType::Grayscale => buf[..]
                .iter()
                .flat_map(|&r| vec![r, r, r, 255])
                .collect::<Vec<u8>>(),
            png::ColorType::GrayscaleAlpha => buf[..]
                .chunks(2)
                .flat_map(|rg| vec![rg[0], rg[0], rg[0], rg[1]])
                .collect::<Vec<u8>>(),
            png::ColorType::Rgb => buf[..]
                .chunks(3)
                .flat_map(|rgb| vec![rgb[0], rgb[1], rgb[2], 255])
                .collect::<Vec<u8>>(),
            png::ColorType::Rgba => buf,
            _ => unreachable!(),
        };

        Self {
            width: info.width as usize,
            height: info.height as usize,
            data: buf.into_boxed_slice(),
        }
    }
}

pub struct Texture {
    _gpu: blade::Texture,
}

struct Transfer {
    stage: blade::Buffer,
    bytes_per_row: u32,
    dst: blade::Texture,
    extent: blade::Extent,
}

pub struct Baker {
    gpu_context: Arc<blade::Context>,
    pending_transfers: Mutex<Vec<Transfer>>,
}

impl Baker {
    pub fn new(gpu_context: &Arc<blade::Context>) -> Self {
        Self {
            gpu_context: Arc::clone(gpu_context),
            pending_transfers: Mutex::new(Vec::new()),
        }
    }

    pub fn flush(
        &self,
        encoder: &mut blade::TransferCommandEncoder,
        temp_buffers: &mut Vec<blade::Buffer>,
    ) {
        for transfer in self.pending_transfers.lock().unwrap().drain(..) {
            encoder.copy_buffer_to_texture(
                transfer.stage.into(),
                transfer.bytes_per_row,
                transfer.dst.into(),
                transfer.extent,
            );
            temp_buffers.push(transfer.stage);
        }
    }
}

impl blade_asset::Baker for Baker {
    type Meta = Meta;
    type Format = CompressedImage<'static>;
    type Output = Texture;
    fn cook(
        &self,
        source: &[u8],
        extension: &str,
        meta: Meta,
        result: Arc<blade_asset::SynCell<Vec<u8>>>,
        _exe_context: choir::ExecutionContext,
    ) {
        use blade::TextureFormat as Tf;
        let dst_format = match meta.format {
            Tf::Bc1Unorm | Tf::Bc1UnormSrgb => texpresso::Format::Bc1,
            Tf::Bc2Unorm | Tf::Bc2UnormSrgb => texpresso::Format::Bc2,
            Tf::Bc3Unorm | Tf::Bc3UnormSrgb => texpresso::Format::Bc3,
            Tf::Bc4Unorm | Tf::Bc4Snorm => texpresso::Format::Bc4,
            Tf::Bc5Unorm | Tf::Bc5Snorm => texpresso::Format::Bc5,
            other => panic!("Unsupported destination format {:?}", other),
        };

        match extension {
            #[cfg(feature = "asset")]
            "png" => {
                let src = PlainImage::from_png(source);
                let dst_size = dst_format.compressed_size(src.width, src.height);
                let mut buf = vec![0u8; dst_size];
                let params = texpresso::Params::default();
                dst_format.compress(&src.data, src.width, src.height, params, &mut buf);

                let image = CompressedImage {
                    name: &[],
                    extent: [src.width as u32, src.height as u32, 1],
                    format: TextureFormatWrap(meta.format),
                    data: &buf,
                };
                let mut dst_raw = vec![0u8; image.size()];
                unsafe { image.write(dst_raw.as_mut_ptr()) };
                *result.borrow_mut() = dst_raw;
            }
            other => panic!("Unknown texture extension: {}", other),
        }
    }

    fn serve(&self, cooked: &[u8]) -> Self::Output {
        let image = unsafe { CompressedImage::read(cooked.as_ptr()) };
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
        self.pending_transfers.lock().unwrap().push(Transfer {
            stage,
            bytes_per_row: (image.extent[0] / block_info.dimensions.0 as u32)
                * block_info.size as u32,
            dst: texture,
            extent,
        });
        Texture { _gpu: texture }
    }
}
