use blade_asset::Flat as _;
use std::{fs, io, path::Path, str, sync::Arc};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, blade_macros::Flat)]
struct TextureFormatWrap(blade::TextureFormat);

#[derive(blade_macros::Flat)]
struct Image<'a> {
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
struct RawImage {
    width: usize,
    height: usize,
    data: Box<[u8]>,
}

#[cfg(feature = "asset")]
impl RawImage {
    fn from_png(input: impl io::Read) -> Self {
        let mut decoder = png::Decoder::new(input);
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
    gpu: blade::Texture,
}

pub struct Baker {
    gpu: Arc<blade::Context>,
}

impl blade_asset::Baker for Baker {
    type Meta = Meta;
    type Output = Texture;
    fn cook(
        &self,
        src_path: &Path,
        meta: Meta,
        dst_path: &Path,
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

        let file_name = src_path.file_name().unwrap().to_str().unwrap();
        let input = fs::File::open(src_path).unwrap();
        match src_path.extension().unwrap().to_str().unwrap() {
            #[cfg(feature = "asset")]
            "png" => {
                let src = RawImage::from_png(input);
                let dst_size = dst_format.compressed_size(src.width, src.height);
                let mut buf = vec![0u8; dst_size];
                let params = texpresso::Params::default();
                dst_format.compress(&src.data, src.width, src.height, params, &mut buf);

                let image = Image {
                    name: file_name.as_bytes(),
                    extent: [src.width as u32, src.height as u32, 1],
                    format: TextureFormatWrap(meta.format),
                    data: &buf,
                };
                let mut dst_raw = vec![0u8; image.size()];
                unsafe { image.write(dst_raw.as_mut_ptr()) };
                fs::write(dst_path, &dst_raw).unwrap();
            }
            other => panic!("Unknown texture extension: {}", other),
        }
    }

    fn serve(&self, cooked: &[u8]) -> Self::Output {
        let image = unsafe { Image::read(cooked.as_ptr()) };
        let texture = self.gpu.create_texture(blade::TextureDesc {
            name: str::from_utf8(image.name).unwrap(),
            format: image.format.0,
            size: blade::Extent {
                width: image.extent[0],
                height: image.extent[1],
                depth: image.extent[2],
            },
            array_layer_count: 1,
            mip_level_count: 1, // TODO
            dimension: blade::TextureDimension::D2,
            usage: blade::TextureUsage::COPY | blade::TextureUsage::RESOURCE,
        });
        Texture { gpu: texture }
    }
}
