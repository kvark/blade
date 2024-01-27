use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFile,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use std::error::Error;

pub fn print_err(error: &dyn Error) {
    eprint!("{}", error);

    let mut e = error.source();
    if e.is_some() {
        eprintln!(": ");
    } else {
        eprintln!();
    }

    while let Some(source) = e {
        eprintln!("\t{}", source);
        e = source.source();
    }
}

pub fn emit_annotated_error<E: Error>(ann_err: &naga::WithSpan<E>, filename: &str, source: &str) {
    let files = SimpleFile::new(filename, source);
    let config = term::Config::default();
    let writer = StandardStream::stderr(ColorChoice::Auto);

    let diagnostic = Diagnostic::error().with_labels(
        ann_err
            .spans()
            .map(|&(span, ref desc)| {
                Label::primary((), span.to_range().unwrap()).with_message(desc.to_owned())
            })
            .collect(),
    );

    term::emit(&mut writer.lock(), &config, &files, &diagnostic).expect("cannot write error");
}

impl super::TextureFormat {
    pub fn block_info(&self) -> super::TexelBlockInfo {
        fn uncompressed(size: u8) -> super::TexelBlockInfo {
            super::TexelBlockInfo {
                dimensions: (1, 1),
                size,
            }
        }
        fn cx_bc(size: u8) -> super::TexelBlockInfo {
            super::TexelBlockInfo {
                dimensions: (4, 4),
                size,
            }
        }
        match *self {
            Self::R8Unorm => uncompressed(1),
            Self::Rg8Unorm => uncompressed(2),
            Self::Rg8Snorm => uncompressed(2),
            Self::Rgba8Unorm => uncompressed(4),
            Self::Rgba8UnormSrgb => uncompressed(4),
            Self::Bgra8Unorm => uncompressed(4),
            Self::Bgra8UnormSrgb => uncompressed(4),
            Self::Rgba8Snorm => uncompressed(4),
            Self::R16Float => uncompressed(2),
            Self::Rgba16Float => uncompressed(8),
            Self::R32Float => uncompressed(4),
            Self::Rg32Float => uncompressed(8),
            Self::Rgba32Float => uncompressed(16),
            Self::R32Uint => uncompressed(4),
            Self::Rg32Uint => uncompressed(8),
            Self::Rgba32Uint => uncompressed(16),
            Self::Depth32Float => uncompressed(4),
            Self::Bc1Unorm => cx_bc(8),
            Self::Bc1UnormSrgb => cx_bc(8),
            Self::Bc2Unorm => cx_bc(16),
            Self::Bc2UnormSrgb => cx_bc(16),
            Self::Bc3Unorm => cx_bc(16),
            Self::Bc3UnormSrgb => cx_bc(16),
            Self::Bc4Unorm => cx_bc(8),
            Self::Bc4Snorm => cx_bc(8),
            Self::Bc5Unorm => cx_bc(16),
            Self::Bc5Snorm => cx_bc(16),
        }
    }

    pub fn aspects(&self) -> super::TexelAspects {
        match *self {
            Self::Depth32Float => super::TexelAspects::DEPTH,
            _ => super::TexelAspects::COLOR,
        }
    }
}

impl super::ComputePipeline {
    /// Return the dispatch group counts sufficient to cover the given extent.
    pub fn get_dispatch_for(&self, extent: super::Extent) -> [u32; 3] {
        let wg_size = self.get_workgroup_size();
        [
            (extent.width + wg_size[0] - 1) / wg_size[0],
            (extent.height + wg_size[1] - 1) / wg_size[1],
            (extent.depth + wg_size[2] - 1) / wg_size[2],
        ]
    }
}
