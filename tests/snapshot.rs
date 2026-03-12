use blade_graphics as gpu;
use std::io::BufReader;
use std::path::Path;

pub struct OffscreenTarget {
    pub texture: gpu::Texture,
    pub view: gpu::TextureView,
    pub readback: gpu::Buffer,
    pub size: gpu::Extent,
    pub format: gpu::TextureFormat,
}

impl OffscreenTarget {
    pub fn new(context: &gpu::Context, size: gpu::Extent, format: gpu::TextureFormat) -> Self {
        let texture = context.create_texture(gpu::TextureDesc {
            name: "snapshot-target",
            format,
            size,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
            sample_count: 1,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "snapshot-target",
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "snapshot-readback",
            size: (size.width * size.height) as u64 * 4,
            memory: gpu::Memory::Shared,
        });
        Self {
            texture,
            view,
            readback,
            size,
            format,
        }
    }

    pub fn read_pixels(
        &self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Vec<u8> {
        if let mut transfer = encoder.transfer("snapshot-readback") {
            transfer.copy_texture_to_buffer(
                self.texture.into(),
                self.readback.into(),
                self.size.width * 4,
                self.size,
            );
        }
        let sync_point = context.submit(encoder);
        assert!(
            context.wait_for(&sync_point, 5000),
            "GPU timed out during snapshot readback"
        );
        let byte_count = (self.size.width * self.size.height * 4) as usize;
        let mut pixels = vec![0u8; byte_count];
        unsafe {
            std::ptr::copy_nonoverlapping(self.readback.data(), pixels.as_mut_ptr(), byte_count);
        }
        pixels
    }

    pub fn destroy(self, context: &gpu::Context) {
        context.destroy_buffer(self.readback);
        context.destroy_texture_view(self.view);
        context.destroy_texture(self.texture);
    }
}

pub struct DiffReport {
    pub max_deviation: u8,
    pub rmse: f64,
    pub different_pixels: usize,
    pub total_pixels: usize,
}

impl std::fmt::Display for DiffReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Image diff: {}/{} pixels differ, max deviation = {}, RMSE = {:.2}",
            self.different_pixels, self.total_pixels, self.max_deviation, self.rmse
        )
    }
}

pub fn compare_images(
    actual: &[u8],
    expected: &[u8],
    size: gpu::Extent,
    tolerance: u8,
) -> Result<(), DiffReport> {
    let total_pixels = (size.width * size.height) as usize;
    assert_eq!(actual.len(), total_pixels * 4);
    assert_eq!(expected.len(), total_pixels * 4);

    let mut max_deviation: u8 = 0;
    let mut sum_sq: f64 = 0.0;
    let mut different_pixels = 0;

    for i in 0..total_pixels {
        let base = i * 4;
        let mut pixel_differs = false;
        for c in 0..4 {
            let a = actual[base + c];
            let e = expected[base + c];
            let diff = a.abs_diff(e);
            if diff > max_deviation {
                max_deviation = diff;
            }
            sum_sq += (diff as f64) * (diff as f64);
            if diff > tolerance {
                pixel_differs = true;
            }
        }
        if pixel_differs {
            different_pixels += 1;
        }
    }

    let rmse = (sum_sq / (total_pixels * 4) as f64).sqrt();

    if max_deviation > tolerance {
        Err(DiffReport {
            max_deviation,
            rmse,
            different_pixels,
            total_pixels,
        })
    } else {
        Ok(())
    }
}

pub fn load_reference(path: &Path) -> (Vec<u8>, gpu::Extent) {
    let file = std::fs::File::open(path).unwrap_or_else(|e| {
        panic!(
            "Failed to open reference image '{}': {}. \
             Run with BLADE_UPDATE_SNAPSHOTS=1 to generate it.",
            path.display(),
            e
        )
    });
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    let size = gpu::Extent {
        width: info.width,
        height: info.height,
        depth: 1,
    };
    (buf, size)
}

pub fn save_image(path: &Path, data: &[u8], size: gpu::Extent) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let file = std::fs::File::create(path).unwrap();
    let mut encoder = png::Encoder::new(file, size.width, size.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(data).unwrap();
}
