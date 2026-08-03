#![allow(
    // We don't use syntax sugar where it's not necessary.
    clippy::match_like_matches_macro,
    // Redundant matching is more explicit.
    clippy::redundant_pattern_matching,
    // Explicit lifetimes are often easier to reason about.
    clippy::needless_lifetimes,
    // No need for defaults in the internal types.
    clippy::new_without_default,
    // Matches are good and extendable, no need to make an exception here.
    clippy::single_match,
)]
#![warn(trivial_numeric_casts, unused_extern_crates)]

//! Blade Neural is a lean NPU abstraction in the spirit of `blade-graphics`.
//!
//! Unlike GPUs, NPUs are not exposed through a portable low-level command
//! buffer. Every vendor surface (Apple MPSGraph/Core ML, Windows DirectML,
//! Intel OpenVINO, Qualcomm QNN, the web's WebNN) is an *op-graph* API: you
//! describe a graph of primitive operations, compile it, bind tensors, and
//! execute. Blade Neural mirrors that lowest common denominator.
//!
//! ```
//! use blade_neural as bn;
//! let context = bn::Context::init(bn::ContextDesc::default()).unwrap();
//!
//! let mut builder = bn::GraphBuilder::new();
//! let x = builder.input("x", bn::TensorDesc::new(bn::DataType::F32, &[1, 3]));
//! let w = builder.constant(
//!     bn::TensorDesc::new(bn::DataType::F32, &[3, 2]),
//!     bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]),
//! );
//! let y = builder.relu(builder.matmul(x, w));
//! builder.output("y", y);
//!
//! let graph = context.compile_graph(&builder).unwrap();
//! let input = context.create_tensor(&bn::TensorDesc::new(bn::DataType::F32, &[1, 3]));
//! input.write_f32(&[1.0, 2.0, 3.0]);
//! let output = context.create_tensor(&bn::TensorDesc::new(bn::DataType::F32, &[1, 2]));
//! context.run(&graph, &[("x", &input)], &[("y", &output)]);
//! assert_eq!(output.read_f32(), vec![4.0, 5.0]);
//! ```

mod ir;
pub mod traits;

// The backend is selected at compile time, mirroring `blade-graphics`.
// Only the pure-Rust reference backend is wired up today; the commented
// arms below show where native backends slot in as they land.
//
// #[cfg_attr(all(neural_coreml, any(target_os = "macos", target_os = "ios")), path = "coreml/mod.rs")]
// #[cfg_attr(all(neural_directml, target_os = "windows"), path = "directml/mod.rs")]
// #[cfg_attr(neural_openvino, path = "openvino/mod.rs")]
#[cfg_attr(all(), path = "cpu/mod.rs")]
mod hal;

pub use hal::*;
pub use ir::{GraphBuilder, GraphIr, Node, Op};

use std::fmt;

/// Numeric type of a tensor's elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    /// Reserved for native backends; the reference backend does not compute in it yet.
    F16,
    I32,
    I8,
    U8,
}

impl DataType {
    /// Size of a single element, in bytes.
    pub fn size(&self) -> usize {
        match *self {
            Self::F32 | Self::I32 => 4,
            Self::F16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }
}

/// Description of a tensor: its element type and shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorDesc {
    pub dtype: DataType,
    pub shape: Box<[usize]>,
}

impl TensorDesc {
    pub fn new(dtype: DataType, shape: &[usize]) -> Self {
        Self {
            dtype,
            shape: shape.into(),
        }
    }
    /// Number of elements across all dimensions.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
    /// Total size of the tensor data, in bytes.
    pub fn byte_size(&self) -> usize {
        self.element_count() * self.dtype.size()
    }
}

/// A handle to a value flowing through the graph, returned by [`GraphBuilder`]
/// methods. It is a small `Copy` token, like a resource handle in `blade-graphics`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Operand(pub(crate) u32);

/// Options for initializing a [`Context`].
#[derive(Clone, Debug, Default)]
pub struct ContextDesc {
    /// Enable extra validation of the graph and execution.
    pub validation: bool,
}

/// Error returned when the platform cannot provide a neural context.
#[derive(Debug)]
pub enum NotSupportedError {
    PlatformNotSupported,
    NoSupportedDeviceFound,
}

impl fmt::Display for NotSupportedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::PlatformNotSupported => f.write_str("platform not supported"),
            Self::NoSupportedDeviceFound => f.write_str("no supported device found"),
        }
    }
}

impl std::error::Error for NotSupportedError {}

/// Error returned when a graph cannot be compiled by a backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// The graph refers to an output that no operation produces.
    UnknownOutput(String),
    /// A data type is not supported by this backend.
    UnsupportedDataType(DataType),
    /// An operation is not supported by this backend.
    UnsupportedOp(&'static str),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnknownOutput(ref name) => write!(f, "unknown graph output {:?}", name),
            Self::UnsupportedDataType(dt) => write!(f, "unsupported data type {:?}", dt),
            Self::UnsupportedOp(op) => write!(f, "unsupported operation {:?}", op),
        }
    }
}

impl std::error::Error for CompileError {}

#[cfg(test)]
mod tests;
