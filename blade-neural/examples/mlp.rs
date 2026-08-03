//! A tiny 2-layer MLP forward pass with baked-in weights, demonstrating the
//! full builder -> compile -> run flow of `blade-neural`.

use blade_neural as bn;

fn desc(shape: &[usize]) -> bn::TensorDesc {
    bn::TensorDesc::new(bn::DataType::F32, shape)
}

fn main() {
    let ctx = bn::Context::init(bn::ContextDesc::default()).unwrap();

    // Network: input[1,4] -> dense(4->3) + relu -> dense(3->2) -> softmax.
    let b = bn::GraphBuilder::new();
    let x = b.input("x", desc(&[1, 4]));

    let w1 = b.constant(
        desc(&[4, 3]),
        bytemuck::cast_slice(&[
            0.1f32, 0.2, -0.1, 0.0, 0.3, 0.1, -0.2, 0.1, 0.4, 0.5, -0.3, 0.2,
        ]),
    );
    let b1 = b.constant(desc(&[3]), bytemuck::cast_slice(&[0.1f32, -0.1, 0.0]));
    let h = b.relu(b.add(b.matmul(x, w1), b1));

    let w2 = b.constant(
        desc(&[3, 2]),
        bytemuck::cast_slice(&[0.3f32, -0.2, 0.1, 0.4, -0.5, 0.2]),
    );
    let b2 = b.constant(desc(&[2]), bytemuck::cast_slice(&[0.0f32, 0.1]));
    let y = b.softmax(b.add(b.matmul(h, w2), b2), 1);
    b.output("y", y);

    let graph = ctx.compile_graph(&b).unwrap();

    let input = ctx.create_tensor(&desc(&[1, 4]));
    input.write_f32(&[1.0, 0.5, -0.5, 2.0]);
    let output = ctx.create_tensor(&desc(&[1, 2]));
    ctx.run(&graph, &[("x", &input)], &[("y", &output)]);

    let probs = output.read_f32();
    println!("class probabilities: {:?}", probs);
    println!("sum = {:.6} (should be ~1.0)", probs.iter().sum::<f32>());
}
