//! Abstract contract that every backend implements, mirroring
//! `blade-graphics::traits`. The concrete `Context`/`Tensor`/`Graph` types are
//! provided by the `cfg`-selected backend module and re-exported from the crate
//! root.

use std::fmt::Debug;

use crate::{CompileError, GraphBuilder, TensorDesc};

/// A neural device capable of compiling and running op-graphs.
pub trait NeuralDevice {
    /// A compiled, executable graph.
    type Graph: Send + Sync;
    /// Backing storage for a tensor, bound at execution time.
    type Tensor: Send + Sync + Clone + Debug;

    /// Compile the recorded graph into an executable form.
    fn compile_graph(&self, builder: &GraphBuilder) -> Result<Self::Graph, CompileError>;

    /// Allocate a tensor with the given description.
    fn create_tensor(&self, desc: &TensorDesc) -> Self::Tensor;

    /// Execute `graph`, reading from `inputs` and writing into `outputs`,
    /// matched by name.
    fn run(
        &self,
        graph: &Self::Graph,
        inputs: &[(&str, &Self::Tensor)],
        outputs: &[(&str, &Self::Tensor)],
    );
}
