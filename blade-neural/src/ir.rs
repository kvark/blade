//! The backend-agnostic op-graph intermediate representation and its builder.
//!
//! This is the shared IR that every backend lowers from, analogous to how
//! `blade-graphics` shares a `naga::Module` across its shader backends.

use std::cell::{Ref, RefCell};

use crate::{Operand, TensorDesc};

/// A single operation in the graph.
///
/// Each variant carries the [`Operand`]s it reads. Operands always refer to
/// *earlier* nodes, so the natural index order of [`GraphIr::nodes`] is a valid
/// topological order.
#[derive(Clone, Debug)]
pub enum Op {
    /// A named graph input, bound at execution time.
    Input { name: String },
    /// A baked-in constant (e.g. weights).
    Constant { data: Box<[u8]> },
    /// 2D matrix multiply: `[M, K] x [K, N] -> [M, N]`.
    MatMul { a: Operand, b: Operand },
    /// Element-wise addition with numpy-style broadcasting.
    Add { a: Operand, b: Operand },
    /// Element-wise multiplication with numpy-style broadcasting.
    Mul { a: Operand, b: Operand },
    /// Element-wise `max(x, 0)`.
    Relu { x: Operand },
    /// Softmax along `axis`.
    Softmax { x: Operand, axis: usize },
    /// Reinterpret the same data under a new shape.
    Reshape { x: Operand },
}

impl Op {
    /// Stable name for diagnostics.
    pub fn name(&self) -> &'static str {
        match *self {
            Self::Input { .. } => "input",
            Self::Constant { .. } => "constant",
            Self::MatMul { .. } => "matmul",
            Self::Add { .. } => "add",
            Self::Mul { .. } => "mul",
            Self::Relu { .. } => "relu",
            Self::Softmax { .. } => "softmax",
            Self::Reshape { .. } => "reshape",
        }
    }
}

/// A node produces exactly one [`Operand`], whose id equals the node's index.
#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    /// Description of the operand this node produces (filled in by shape inference).
    pub desc: TensorDesc,
}

/// The compiled-into-able description of a graph: a topologically ordered list
/// of nodes plus the named outputs to read back.
#[derive(Clone, Debug, Default)]
pub struct GraphIr {
    pub nodes: Vec<Node>,
    pub outputs: Vec<(String, Operand)>,
}

/// Records operations into a [`GraphIr`]. Backend-agnostic and side-effect free
/// until handed to a backend for compilation.
///
/// Methods take `&self` (via interior mutability) so graphs read naturally as
/// nested expressions, e.g. `b.relu(b.add(b.matmul(x, w), bias))`.
///
/// Shape and type mismatches are programming errors and panic with a descriptive
/// message, in keeping with Blade's "you know what you are doing" stance.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    ir: RefCell<GraphIr>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the recorded IR (used by backends).
    pub fn ir(&self) -> Ref<'_, GraphIr> {
        self.ir.borrow()
    }

    /// The description of an existing operand.
    pub fn desc(&self, op: Operand) -> TensorDesc {
        self.ir.borrow().nodes[op.0 as usize].desc.clone()
    }

    fn push(&self, op: Op, desc: TensorDesc) -> Operand {
        let mut ir = self.ir.borrow_mut();
        let id = ir.nodes.len() as u32;
        ir.nodes.push(Node { op, desc });
        Operand(id)
    }

    /// Declare a named graph input.
    pub fn input(&self, name: &str, desc: TensorDesc) -> Operand {
        self.push(
            Op::Input {
                name: name.to_string(),
            },
            desc,
        )
    }

    /// Bake a constant tensor (e.g. weights) into the graph.
    pub fn constant(&self, desc: TensorDesc, data: &[u8]) -> Operand {
        assert_eq!(
            data.len(),
            desc.byte_size(),
            "constant data is {} bytes, but its shape needs {}",
            data.len(),
            desc.byte_size()
        );
        self.push(Op::Constant { data: data.into() }, desc)
    }

    /// 2D matrix multiply: `[M, K] x [K, N] -> [M, N]`.
    pub fn matmul(&self, a: Operand, b: Operand) -> Operand {
        let da = self.desc(a);
        let db = self.desc(b);
        assert!(
            da.shape.len() == 2 && db.shape.len() == 2,
            "matmul expects 2D operands, got {:?} and {:?}",
            da.shape,
            db.shape
        );
        assert_eq!(
            da.shape[1], db.shape[0],
            "matmul inner dimensions disagree: {:?} vs {:?}",
            da.shape, db.shape
        );
        let shape = [da.shape[0], db.shape[1]];
        self.push(Op::MatMul { a, b }, TensorDesc::new(da.dtype, &shape))
    }

    /// Element-wise addition with numpy-style broadcasting.
    pub fn add(&self, a: Operand, b: Operand) -> Operand {
        let desc = self.broadcast(a, b);
        self.push(Op::Add { a, b }, desc)
    }

    /// Element-wise multiplication with numpy-style broadcasting.
    pub fn mul(&self, a: Operand, b: Operand) -> Operand {
        let desc = self.broadcast(a, b);
        self.push(Op::Mul { a, b }, desc)
    }

    /// Element-wise `max(x, 0)`.
    pub fn relu(&self, x: Operand) -> Operand {
        let desc = self.desc(x);
        self.push(Op::Relu { x }, desc)
    }

    /// Softmax normalization along `axis`.
    pub fn softmax(&self, x: Operand, axis: usize) -> Operand {
        let desc = self.desc(x);
        assert!(
            axis < desc.shape.len(),
            "softmax axis {} out of range for shape {:?}",
            axis,
            desc.shape
        );
        self.push(Op::Softmax { x, axis }, desc)
    }

    /// Reinterpret a tensor under a new shape with the same element count.
    pub fn reshape(&self, x: Operand, shape: &[usize]) -> Operand {
        let src = self.desc(x);
        let new_count: usize = shape.iter().product();
        assert_eq!(
            src.element_count(),
            new_count,
            "reshape from {:?} to {:?} changes the element count",
            src.shape,
            shape
        );
        let desc = TensorDesc::new(src.dtype, shape);
        self.push(Op::Reshape { x }, desc)
    }

    /// Mark an operand as a named output of the graph.
    pub fn output(&self, name: &str, value: Operand) {
        self.ir.borrow_mut().outputs.push((name.to_string(), value));
    }

    /// Compute the broadcasted output description of an element-wise op.
    fn broadcast(&self, a: Operand, b: Operand) -> TensorDesc {
        let da = self.desc(a);
        let db = self.desc(b);
        assert_eq!(
            da.dtype, db.dtype,
            "element-wise operands disagree on type: {:?} vs {:?}",
            da.dtype, db.dtype
        );
        let shape = broadcast_shape(&da.shape, &db.shape);
        TensorDesc {
            dtype: da.dtype,
            shape,
        }
    }
}

/// Compute the numpy-style broadcasted shape of two shapes, or panic if they are
/// not broadcast-compatible.
pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Box<[usize]> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        // Align from the right; missing leading dims act as 1.
        let da = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            panic!("shapes {:?} and {:?} are not broadcast-compatible", a, b);
        };
    }
    out.into_boxed_slice()
}
