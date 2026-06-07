//! Reference kernels for the pure-Rust backend. These operate on flat `f32`
//! slices and prioritize obvious correctness over speed — they are the oracle
//! that hardware backends are checked against.

use crate::ir::broadcast_shape;

/// 2D matrix multiply: `a` is `[m, k]`, `b` is `[k, n]`, output is `[m, n]`.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for inner in 0..k {
            let av = a[row * k + inner];
            if av == 0.0 {
                continue;
            }
            let b_row = &b[inner * n..inner * n + n];
            let out_row = &mut out[row * n..row * n + n];
            for col in 0..n {
                out_row[col] += av * b_row[col];
            }
        }
    }
    out
}

/// Element-wise op with numpy-style broadcasting from shapes `sa`/`sb`.
pub fn broadcast_binary(
    a: &[f32],
    sa: &[usize],
    b: &[f32],
    sb: &[usize],
    f: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let out_shape = broadcast_shape(sa, sb);
    let count: usize = out_shape.iter().product();
    let stride_a = broadcast_strides(sa, &out_shape);
    let stride_b = broadcast_strides(sb, &out_shape);

    let mut out = vec![0.0f32; count];
    let mut index = vec![0usize; out_shape.len()];
    for o in out.iter_mut() {
        let mut ia = 0;
        let mut ib = 0;
        for (d, &i) in index.iter().enumerate() {
            ia += i * stride_a[d];
            ib += i * stride_b[d];
        }
        *o = f(a[ia], b[ib]);
        increment(&mut index, &out_shape);
    }
    out
}

/// Strides into a source of shape `src` when iterating over `out`, with 0 stride
/// on broadcast (size-1 or missing) dimensions.
fn broadcast_strides(src: &[usize], out: &[usize]) -> Vec<usize> {
    let rank = out.len();
    let offset = rank - src.len();
    // Row-major strides of the source.
    let mut src_strides = vec![0usize; src.len()];
    let mut acc = 1;
    for i in (0..src.len()).rev() {
        src_strides[i] = acc;
        acc *= src[i];
    }
    let mut strides = vec![0usize; rank];
    for (d, stride) in strides.iter_mut().enumerate() {
        if d >= offset {
            let sd = d - offset;
            *stride = if src[sd] == 1 { 0 } else { src_strides[sd] };
        }
    }
    strides
}

/// Advance a multi-dimensional index in row-major order.
fn increment(index: &mut [usize], shape: &[usize]) {
    for d in (0..shape.len()).rev() {
        index[d] += 1;
        if index[d] < shape[d] {
            return;
        }
        index[d] = 0;
    }
}

/// Element-wise `max(x, 0)`, in place.
pub fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Numerically-stable softmax along `axis` of a tensor of shape `shape`.
pub fn softmax(x: &[f32], shape: &[usize], axis: usize) -> Vec<f32> {
    let axis_len = shape[axis];
    // Stride between successive elements along `axis`.
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();

    let mut out = vec![0.0f32; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_len * inner + i;
            let mut max = f32::NEG_INFINITY;
            for a in 0..axis_len {
                max = max.max(x[base + a * inner]);
            }
            let mut sum = 0.0f32;
            for a in 0..axis_len {
                let e = (x[base + a * inner] - max).exp();
                out[base + a * inner] = e;
                sum += e;
            }
            for a in 0..axis_len {
                out[base + a * inner] /= sum;
            }
        }
    }
    out
}
