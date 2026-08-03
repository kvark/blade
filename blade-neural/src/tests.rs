use crate as bn;

fn f32_desc(shape: &[usize]) -> bn::TensorDesc {
    bn::TensorDesc::new(bn::DataType::F32, shape)
}

fn context() -> bn::Context {
    bn::Context::init(bn::ContextDesc { validation: true }).unwrap()
}

#[test]
fn matmul_bias_relu() {
    let ctx = context();
    let b = bn::GraphBuilder::new();
    let x = b.input("x", f32_desc(&[1, 3]));
    // 3x2 weight matrix (column-major view: identity-ish).
    let w = b.constant(
        f32_desc(&[3, 2]),
        bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]),
    );
    let bias = b.constant(f32_desc(&[2]), bytemuck::cast_slice(&[-10.0f32, 1.0]));
    let y = b.relu(b.add(b.matmul(x, w), bias));
    b.output("y", y);

    let graph = ctx.compile_graph(&b).unwrap();
    let xt = ctx.create_tensor(&f32_desc(&[1, 3]));
    xt.write_f32(&[1.0, 2.0, 3.0]);
    let yt = ctx.create_tensor(&f32_desc(&[1, 2]));
    ctx.run(&graph, &[("x", &xt)], &[("y", &yt)]);

    // matmul: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]; +bias = [-6, 6]; relu = [0, 6]
    assert_eq!(yt.read_f32(), vec![0.0, 6.0]);
}

#[test]
fn softmax_normalizes() {
    let ctx = context();
    let b = bn::GraphBuilder::new();
    let x = b.input("x", f32_desc(&[1, 3]));
    let y = b.softmax(x, 1);
    b.output("y", y);

    let graph = ctx.compile_graph(&b).unwrap();
    let xt = ctx.create_tensor(&f32_desc(&[1, 3]));
    xt.write_f32(&[1.0, 2.0, 3.0]);
    let yt = ctx.create_tensor(&f32_desc(&[1, 3]));
    ctx.run(&graph, &[("x", &xt)], &[("y", &yt)]);

    let out = yt.read_f32();
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax should sum to 1, got {sum}"
    );
    assert!(out[0] < out[1] && out[1] < out[2], "monotonic: {out:?}");
}

#[test]
fn broadcast_add() {
    let ctx = context();
    let b = bn::GraphBuilder::new();
    let x = b.input("x", f32_desc(&[2, 2]));
    let row = b.constant(f32_desc(&[2]), bytemuck::cast_slice(&[10.0f32, 20.0]));
    let y = b.add(x, row);
    b.output("y", y);

    let graph = ctx.compile_graph(&b).unwrap();
    let xt = ctx.create_tensor(&f32_desc(&[2, 2]));
    xt.write_f32(&[1.0, 2.0, 3.0, 4.0]);
    let yt = ctx.create_tensor(&f32_desc(&[2, 2]));
    ctx.run(&graph, &[("x", &xt)], &[("y", &yt)]);
    assert_eq!(yt.read_f32(), vec![11.0, 22.0, 13.0, 24.0]);
}

#[test]
fn reshape_preserves_data() {
    let ctx = context();
    let b = bn::GraphBuilder::new();
    let x = b.input("x", f32_desc(&[2, 3]));
    let y = b.reshape(x, &[3, 2]);
    b.output("y", y);

    let graph = ctx.compile_graph(&b).unwrap();
    let xt = ctx.create_tensor(&f32_desc(&[2, 3]));
    xt.write_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let yt = ctx.create_tensor(&f32_desc(&[3, 2]));
    ctx.run(&graph, &[("x", &xt)], &[("y", &yt)]);
    assert_eq!(yt.read_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn rejects_non_f32() {
    let ctx = context();
    let b = bn::GraphBuilder::new();
    let _ = b.input("x", bn::TensorDesc::new(bn::DataType::I32, &[2]));
    let err = ctx.compile_graph(&b).unwrap_err();
    assert_eq!(
        err,
        bn::CompileError::UnsupportedDataType(bn::DataType::I32)
    );
}

#[test]
#[should_panic(expected = "broadcast-compatible")]
fn incompatible_shapes_panic() {
    let b = bn::GraphBuilder::new();
    let x = b.input("x", f32_desc(&[2, 3]));
    let y = b.input("y", f32_desc(&[4, 5]));
    let _ = b.add(x, y);
}
