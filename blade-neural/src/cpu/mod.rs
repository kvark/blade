//! Pure-Rust reference backend.
//!
//! Always available, needs no drivers, and serves as the correctness oracle for
//! future native (NPU) backends. Tensors live in host memory and graphs are
//! evaluated by walking the IR in topological order.

mod ops;

use std::sync::{Arc, Mutex};

use crate::{
    CompileError, ContextDesc, DataType, GraphBuilder, GraphIr, NotSupportedError, Op, TensorDesc,
};

/// A neural context backed by the host CPU.
#[derive(Debug)]
pub struct Context {
    validation: bool,
}

/// Host-resident tensor storage. Cheap to clone (shares the underlying buffer).
#[derive(Clone, Debug)]
pub struct Tensor {
    desc: TensorDesc,
    data: Arc<Mutex<Vec<u8>>>,
}

impl Tensor {
    /// The tensor's element type and shape.
    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }
    /// Overwrite the tensor's raw bytes.
    pub fn write(&self, bytes: &[u8]) {
        let mut data = self.data.lock().unwrap();
        assert_eq!(bytes.len(), data.len(), "tensor write size mismatch");
        data.copy_from_slice(bytes);
    }
    /// Overwrite the tensor with `f32` values.
    pub fn write_f32(&self, values: &[f32]) {
        self.write(bytemuck::cast_slice(values));
    }
    /// Read back the tensor's raw bytes.
    pub fn read(&self) -> Vec<u8> {
        self.data.lock().unwrap().clone()
    }
    /// Read back the tensor as `f32` values.
    pub fn read_f32(&self) -> Vec<f32> {
        bytemuck::cast_slice(&self.read()).to_vec()
    }
}

/// A graph compiled for the reference backend (just the validated IR).
#[derive(Debug)]
pub struct Graph {
    ir: GraphIr,
}

impl Context {
    /// Initialize the reference backend. Always succeeds.
    pub fn init(desc: ContextDesc) -> Result<Self, NotSupportedError> {
        Ok(Self {
            validation: desc.validation,
        })
    }

    /// Compile a recorded graph.
    pub fn compile_graph(&self, builder: &GraphBuilder) -> Result<Graph, CompileError> {
        profiling::scope!("compile_graph");
        let ir = builder.ir();
        // The reference backend only computes in f32.
        for node in &ir.nodes {
            if node.desc.dtype != DataType::F32 {
                return Err(CompileError::UnsupportedDataType(node.desc.dtype));
            }
        }
        for (name, operand) in &ir.outputs {
            if operand.0 as usize >= ir.nodes.len() {
                return Err(CompileError::UnknownOutput(name.clone()));
            }
        }
        Ok(Graph {
            ir: GraphIr::clone(&ir),
        })
    }

    /// Allocate a zeroed tensor.
    pub fn create_tensor(&self, desc: &TensorDesc) -> Tensor {
        Tensor {
            desc: desc.clone(),
            data: Arc::new(Mutex::new(vec![0u8; desc.byte_size()])),
        }
    }

    /// Execute `graph`, reading `inputs` and writing `outputs`, matched by name.
    pub fn run(&self, graph: &Graph, inputs: &[(&str, &Tensor)], outputs: &[(&str, &Tensor)]) {
        profiling::scope!("run");
        let nodes = &graph.ir.nodes;
        let mut values: Vec<Vec<f32>> = Vec::with_capacity(nodes.len());

        for node in nodes {
            let value = match node.op {
                Op::Input { ref name } => {
                    let (_, tensor) = inputs
                        .iter()
                        .find(|(n, _)| *n == name.as_str())
                        .unwrap_or_else(|| panic!("missing input binding {:?}", name));
                    if self.validation {
                        assert_eq!(
                            &tensor.desc.shape, &node.desc.shape,
                            "input {:?} shape mismatch",
                            name
                        );
                    }
                    tensor.read_f32()
                }
                Op::Constant { ref data } => bytemuck::cast_slice::<u8, f32>(data).to_vec(),
                Op::MatMul { a, b } => {
                    let sa = &nodes[a.0 as usize].desc.shape;
                    let sb = &nodes[b.0 as usize].desc.shape;
                    ops::matmul(
                        &values[a.0 as usize],
                        &values[b.0 as usize],
                        sa[0],
                        sa[1],
                        sb[1],
                    )
                }
                Op::Add { a, b } => ops::broadcast_binary(
                    &values[a.0 as usize],
                    &nodes[a.0 as usize].desc.shape,
                    &values[b.0 as usize],
                    &nodes[b.0 as usize].desc.shape,
                    |x, y| x + y,
                ),
                Op::Mul { a, b } => ops::broadcast_binary(
                    &values[a.0 as usize],
                    &nodes[a.0 as usize].desc.shape,
                    &values[b.0 as usize],
                    &nodes[b.0 as usize].desc.shape,
                    |x, y| x * y,
                ),
                Op::Relu { x } => {
                    let mut v = values[x.0 as usize].clone();
                    ops::relu(&mut v);
                    v
                }
                Op::Softmax { x, axis } => {
                    ops::softmax(&values[x.0 as usize], &nodes[x.0 as usize].desc.shape, axis)
                }
                Op::Reshape { x } => values[x.0 as usize].clone(),
            };
            values.push(value);
        }

        for (name, operand) in &graph.ir.outputs {
            let (_, tensor) = outputs
                .iter()
                .find(|(n, _)| *n == name.as_str())
                .unwrap_or_else(|| panic!("missing output binding {:?}", name));
            tensor.write_f32(&values[operand.0 as usize]);
        }
    }
}

impl crate::traits::NeuralDevice for Context {
    type Graph = Graph;
    type Tensor = Tensor;

    fn compile_graph(&self, builder: &GraphBuilder) -> Result<Self::Graph, CompileError> {
        Context::compile_graph(self, builder)
    }
    fn create_tensor(&self, desc: &TensorDesc) -> Self::Tensor {
        Context::create_tensor(self, desc)
    }
    fn run(
        &self,
        graph: &Self::Graph,
        inputs: &[(&str, &Self::Tensor)],
        outputs: &[(&str, &Self::Tensor)],
    ) {
        Context::run(self, graph, inputs, outputs)
    }
}
