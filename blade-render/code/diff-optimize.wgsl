// Loss and optimizer of the differentiable path tracer.
//
// Both go over flat arrays that can be longer than one dimension of
// workgroups can cover, so they are dispatched in rows of workgroups.

const WORKGROUP_SIZE: u32 = 64u;

fn flat_index(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return global_id.y * num_workgroups.x * WORKGROUP_SIZE + global_id.x;
}

// Has to match the host!
struct LossParams {
    pixel_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<uniform> loss_params: LossParams;
var<storage, read> image: array<vec4<f32>>;
var<storage, read> target_image: array<vec4<f32>>;
var<storage, read_write> image_adjoint: array<vec4<f32>>;
// Sum of the loss, which the host clears.
var<storage, read_write> loss: atomic<f32>;

var<workgroup> partial_loss: array<f32, 64>;

// Mean squared error over the color channels of all the pixels.
@compute @workgroup_size(64)
fn loss_l2(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let index = flat_index(global_id, num_workgroups);
    var error = 0.0;
    if (index < loss_params.pixel_count) {
        let scale = 1.0 / (3.0 * f32(loss_params.pixel_count));
        let diff = image[index].xyz - target_image[index].xyz;
        image_adjoint[index] = vec4<f32>(2.0 * scale * diff, 0.0);
        error = scale * dot(diff, diff);
    }
    // One atomic per workgroup.
    partial_loss[local_index] = error;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (local_index < stride) {
            partial_loss[local_index] += partial_loss[local_index + stride];
        }
        workgroupBarrier();
    }
    if (local_index == 0u) {
        atomicAdd(&loss, partial_loss[0]);
    }
}

// Has to match the host!
struct AdamParams {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    // `1 - beta^t` for the step `t`, correcting the bias of the moments.
    bias_correction1: f32,
    bias_correction2: f32,
    // Bounds that the parameters are projected into after the step.
    lower: f32,
    upper: f32,
    count: u32,
    // Parameters are grouped by `period`. Bit `i % period` of `mask` tells
    // whether parameter `i` is optimized, and of `upper_mask` whether it
    // is bounded by `upper`.
    period: u32,
    mask: u32,
    upper_mask: u32,
}

var<uniform> adam: AdamParams;
var<storage, read_write> params: array<f32>;
// Read and cleared for the next step.
var<storage, read_write> grads: array<f32>;
// First and second moments of each parameter.
var<storage, read_write> moments: array<vec2<f32>>;

@compute @workgroup_size(64)
fn adam_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let index = flat_index(global_id, num_workgroups);
    if (index >= adam.count) {
        return;
    }
    let grad = grads[index];
    grads[index] = 0.0;
    let bit = 1u << (index % adam.period);
    if ((adam.mask & bit) == 0u) {
        return;
    }
    let m = mix(grad, moments[index].x, adam.beta1);
    let v = mix(grad * grad, moments[index].y, adam.beta2);
    moments[index] = vec2<f32>(m, v);
    let step = (m / adam.bias_correction1) / (sqrt(v / adam.bias_correction2) + adam.epsilon);
    var value = max(params[index] - adam.learning_rate * step, adam.lower);
    if ((adam.upper_mask & bit) != 0u) {
        value = min(value, adam.upper);
    }
    params[index] = value;
}
