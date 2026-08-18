# Blade Neural

[![Crates.io](https://img.shields.io/crates/v/blade-neural.svg?maxAge=2592000)](https://crates.io/crates/blade-neural)

Blade Neural is a lean, opinionated **NPU abstraction** in the same spirit as
[`blade-graphics`](../blade-graphics) is for GPUs: minimal, ergonomic, and
backed by native platform APIs selected at compile time.

## Why an op-graph, not a command buffer

Unlike GPUs, NPUs expose **no portable low-level command-buffer surface**. Every
vendor only exposes the accelerator through a *graph/model execution* API:

| Platform | Native op-graph surface |
| -------- | ----------------------- |
| Apple    | Core ML / BNNSGraph / MPSGraph (direct ANE access is private) |
| Windows  | DirectML → Windows ML / ONNX EPs |
| Intel    | OpenVINO (+ NPU plugin) |
| Qualcomm | QNN |
| Android  | NNAPI |
| Web      | WebNN (`MLGraphBuilder`) |

The portable lowest common denominator that is genuinely programmable across all
of them is an **op-graph builder**: describe a graph of primitive ops, compile
it, bind tensors, and execute. Blade Neural mirrors exactly that, so native
backends can lower onto each vendor API naturally.

## Usage

```rust
use blade_neural as bn;

let ctx = bn::Context::init(bn::ContextDesc::default()).unwrap();

let mut b = bn::GraphBuilder::new();
let x = b.input("x", bn::TensorDesc::new(bn::DataType::F32, &[1, 3]));
let w = b.constant(/* weights */);
let y = b.relu(b.matmul(x, w));
b.output("y", y);

let graph = ctx.compile_graph(&b).unwrap();
ctx.run(&graph, &[("x", &input)], &[("y", &output)]);
```

See [`examples/mlp.rs`](examples/mlp.rs) for a complete 2-layer network:

```bash
cargo run -p blade-neural --example mlp
```

## Backends

The backend is selected automatically by the host platform, like
`blade-graphics`.

| Backend | Status |
| ------- | ------ |
| CPU reference (pure Rust) | ✅ available everywhere; the correctness oracle |
| MPSGraph (Apple) | planned |
| DirectML (Windows) | planned |
| OpenVINO (Intel NPU) | planned |
| QNN (Qualcomm) | planned |

The MVP op set is `input`, `constant`, `matmul`, `add`, `mul`, `relu`,
`softmax`, and `reshape` — enough to express an MLP classifier. Convolution,
pooling, normalization, attention, quantized paths, and `blade-graphics` tensor
interop are future work.

## Scope

Inference only (no autodiff/training) for now. The crate is `f32`-first; other
data types are reserved in the API and accepted by future native backends.
