# Blade Examples

| Example   | graphics    | macros | util   | egui   | particle | asset  | render | helper | engine |
| --------- | ----------- | ------ | ------ | ------ | -------- | ------ | ------ | ------ | ------ |
| info      | :star:      |        |        |        |          |        |        |        |        |
| queue-priority | :star: | :star: |        |        |          |        |        |        |        |
| ray-query | :star: (RT) | :star: |        |        |          |        |        |        |        |
| particle  | :star:      | :star: |        | :star: | :star:   |        |        |        |        |
| scene     | :star: (RT) | :star: |        | :star: |          | :star: | :star: | :star: |        |
| vehicle   |             |        |        |        |          |        |        |        | :star: |
| move      |             |        |        |        |          |        |        | :star: | :star: |

`mini` and `init` were moved to GPU integration tests (`dispatch_gpu_test` and `env_map_gpu_test`) in [tests/gpu_examples.rs](../tests/gpu_examples.rs).

`cargo run --release --example queue-priority` compares foreground compute
latency while another logical device uses a normal or low system-wide queue
priority. On Vulkan, the low case requires `VK_KHR_global_priority`; the
example logs whether the request was applied and reports the p95 latency ratio.
