use blade_graphics as gpu;
use std::io::BufReader;
use std::path::Path;

const SSIM_THRESHOLD: f64 = 0.95;
const REFERENCE_DIR: &str = "tests/reference";
// SSIM constants for 8-bit images: C1 = (K1*L)^2, C2 = (K2*L)^2 where L=255
const C1: f64 = 6.5025; // (0.01 * 255)^2
const C2: f64 = 58.5225; // (0.03 * 255)^2
const BLOCK: usize = 8;

pub struct OffscreenTarget {
    pub texture: gpu::Texture,
    pub view: gpu::TextureView,
    pub readback: gpu::Buffer,
    pub size: gpu::Extent,
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
        }
    }

    pub fn read_pixels(
        &self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Vec<u8> {
        {
            let mut transfer = encoder.transfer("snapshot-readback");
            transfer.copy_texture_to_buffer(
                self.texture.into(),
                self.readback.into(),
                self.size.width * 4,
                self.size,
            );
        }
        let sync_point = context.submit(encoder, &[]);
        assert!(
            context.wait_for(&sync_point, 5000).unwrap(),
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

/// Compute mean SSIM between two RGBA images over non-overlapping 8x8 blocks.
/// Uses luminance: Y = 0.299*R + 0.587*G + 0.114*B
fn compute_ssim(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    let luminance = |rgba: &[u8]| -> f64 {
        0.299 * rgba[0] as f64 + 0.587 * rgba[1] as f64 + 0.114 * rgba[2] as f64
    };

    let blocks_x = width / BLOCK;
    let blocks_y = height / BLOCK;
    if blocks_x == 0 || blocks_y == 0 {
        return 1.0;
    }

    let n = (BLOCK * BLOCK) as f64;
    let mut ssim_sum = 0.0;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut sum_a = 0.0;
            let mut sum_b = 0.0;
            let mut sum_a2 = 0.0;
            let mut sum_b2 = 0.0;
            let mut sum_ab = 0.0;

            for dy in 0..BLOCK {
                for dx in 0..BLOCK {
                    let px = bx * BLOCK + dx;
                    let py = by * BLOCK + dy;
                    let off = (py * width + px) * 4;
                    let va = luminance(&a[off..]);
                    let vb = luminance(&b[off..]);
                    sum_a += va;
                    sum_b += vb;
                    sum_a2 += va * va;
                    sum_b2 += vb * vb;
                    sum_ab += va * vb;
                }
            }

            let mu_a = sum_a / n;
            let mu_b = sum_b / n;
            let var_a = sum_a2 / n - mu_a * mu_a;
            let var_b = sum_b2 / n - mu_b * mu_b;
            let cov_ab = sum_ab / n - mu_a * mu_b;

            let numerator = (2.0 * mu_a * mu_b + C1) * (2.0 * cov_ab + C2);
            let denominator = (mu_a * mu_a + mu_b * mu_b + C1) * (var_a + var_b + C2);
            ssim_sum += numerator / denominator;
        }
    }

    ssim_sum / (blocks_x * blocks_y) as f64
}

pub fn check(name: &str, pixels: &[u8], size: gpu::Extent) {
    let dir = Path::new(REFERENCE_DIR);
    let reference_path = dir.join(format!("{name}.png"));
    let actual_path = dir.join(format!("{name}_actual.png"));

    if std::env::var("BLADE_UPDATE_SNAPSHOTS").is_ok() {
        save_image(&reference_path, pixels, size);
        println!("Updated reference image: {}", reference_path.display());
        return;
    }

    let (reference, ref_size) = load_reference(&reference_path);
    assert_eq!(
        ref_size, size,
        "Reference image size mismatch: expected {:?}, got {:?}",
        size, ref_size
    );

    let ssim = compute_ssim(
        pixels,
        &reference,
        size.width as usize,
        size.height as usize,
    );
    println!("{name}: SSIM = {ssim:.4}");

    if ssim < SSIM_THRESHOLD {
        save_image(&actual_path, pixels, size);
        panic!(
            "{name} snapshot SSIM = {ssim:.4} (threshold {SSIM_THRESHOLD})\n\
             Actual output saved to: {}",
            actual_path.display()
        );
    }
}

fn load_reference(path: &Path) -> (Vec<u8>, gpu::Extent) {
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

fn save_image(path: &Path, data: &[u8], size: gpu::Extent) {
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
