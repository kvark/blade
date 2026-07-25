# Experiment protocol

This document is the contract between the paper's claims and its artifacts. A
result is admissible only when its raw rows, machine metadata, exact source
revision, and analysis command are retained.

## Questions and factors

The evaluation separates four factors that are often conflated:

1. Barrier placement: every pass boundary versus only true hazards.
2. Memory scope: global versus named buffer/image ranges.
3. Stage/access scope: `ALL_COMMANDS` and generic memory access versus the
   producing and consuming stages/accesses.
4. Image policy: persistent `GENERAL` versus usage-specific optimal layouts.

The within-Blade benchmark varies only factor 1. Factors 2–4 are observed in a
matched wgpu implementation and its generated Vulkan commands. Because that
comparison also changes validation, lifetime management, bindings, and command
representation, it supports an end-to-end design comparison and explanatory
profiling—not single-factor causal attribution.

## Configuration IDs

| ID | Barrier placement | Resource scope | Stage/access scope | Image policy | Status |
|---|---|---|---|---|---|
| `B-auto` | every pass | global | broad | `GENERAL` | implemented |
| `B-hazard` | declared hazards | global | broad | `GENERAL` | implemented |
| `B-explicit-all` | every pass | global | broad | `GENERAL` | implemented as a control |
| `W-wgpu` | derived by wgpu | tracked resource usage | backend-generated | backend-generated | pending matched workload |

Blade emits a final global barrier when finishing every encoder, including in
manual mode. That barrier is common to all three implemented configurations.
`B-explicit-all` additionally emits a manual barrier before every pass, making
it command-for-command equivalent to the automatic barrier policy; it is a
control for benchmark and instrumentation artifacts rather than a distinct
synchronization strategy.

`W-wgpu` is the realistic tracked comparison. It must not be used alone to
attribute a difference to tracking, because validation, lifetime management,
bindings, and command representation also differ. Attribute host costs with
CPU profiles of wgpu's tracker paths, and explain device costs with captured
Vulkan barriers, layouts, and timelines. A direct `wgpu-hal` port is optional
if a particular result needs frontend decomposition; no new tracker in Blade
is required.

The matched wgpu program uses wgpu's native `IMMEDIATES` feature for the
16-byte per-pass parameters that Blade supplies as inline uniform data. This
avoids adding a tracked uniform-buffer resource only to the wgpu side. The
baseline is therefore native wgpu/wgpu-core, not browser WebGPU. CPU-only
collections disable timestamp queries; GPU-timed collections use native
encoder timestamps and do not use their host timings for primary CPU claims.

## Workload families

The headless `sync-bench` artifact initially provides:

| Workload | Resources | True inter-pass hazard | Purpose |
|---|---|---:|---|
| `compute-independent` | shared read-only input, unique outputs | no | cost of redundant compute barriers |
| `compute-chain` | ping-pong storage buffers | yes, every pass | cost when coarse placement is already exact |
| `graphics-independent` | unique color targets | no | cost of redundant graphics barriers |
| `graphics-chain` | one load/store color target with blending | yes, every pass | serialized graphics control |

Required extensions:

- dependency-density DAGs between these extremes;
- transfer→compute and compute→graphics stage transitions;
- attachment→sample, attachment→storage, and copy layout changes;
- resource-count sweeps independent of pass count;
- texture format, depth, MSAA, mip, and array-subresource sweeps;
- mixed graphics/compute workloads with known overlap opportunities.
- matched wgpu versions of the frozen workloads, using the same WGSL, resource
  sizes, pass graph, initialization, output checks, and iteration schedule.

Application workloads should include Bunnymark, the Blade particle system, a
fixed multi-pass renderer scene, and a compute pipeline. Freeze all scenes,
inputs, resolutions, and shader hashes before the final collection.

## Metrics

Collect separately:

- command-buffer start/reset/query-read CPU time;
- pass recording CPU time;
- queue submission CPU time;
- host wait time;
- GPU elapsed time spanning all benchmark passes;
- individual pass intervals where timestamp capacity permits;
- barrier count and resources covered;
- peak host memory used by tracking structures;
- GPU pipeline/barrier stalls and overlap from vendor profilers;
- relevant cache flush/invalidate and image-compression counters when exposed.

Do not use “the number of objects where animation starts to slow down” as a
paper metric. Use fixed workloads and report absolute time, effect size, and
uncertainty.

## Collection protocol

1. Record the exact Blade commit and confirm a clean worktree.
2. Record CPU, GPU, RAM, OS, kernel, Vulkan loader, API version, driver name and
   version, device extensions, feature bits, and monitor/display state.
3. State whether `VK_KHR_unified_image_layouts` is supported and enabled.
4. Use a release build. Disable validation, overlay, capture, and unrelated
   logging for performance runs.
5. Run a separate correctness pass with Vulkan validation and synchronization
   validation enabled. Check output hashes or numeric results.
6. Lock clocks or a stable power state where supported. Record the command and
   setting; never silently discard throttled samples.
7. Warm each configuration for at least 10 iterations. Collect at least 30
   measured iterations, increasing the count for sub-microsecond effects.
8. Counterbalance configuration order across repetitions. Retain order in the
   metadata so drift and thermal bias can be analyzed.
9. Run one configuration per process when comparing host memory and startup
   effects. Also run an in-process alternation as a sensitivity check.
10. Repeat a representative subset on a second day or reboot.
11. For the matched wgpu run, freeze the exact wgpu revision and capture one
    representative command stream per workload. Confirm that the capture
    contains the expected resource barriers and layouts before interpreting
    GPU-time differences.

GPU timestamps add commands and may perturb short workloads. Validate key
effects with an external profiler and an amortized-throughput run with only
begin/end timing.

## Hardware matrix

Use a tiered matrix:

**Minimum viable study**

- one recent AMD Vulkan device-driver pair;
- one recent NVIDIA Vulkan device-driver pair;
- the same primary operating system and benchmark revision on both;
- current stable drivers, with exact versions and extension lists retained.

This supports a two-device comparison, not a general claim about either vendor.

**Recommended cross-vendor study**

- two AMD architecture generations;
- two NVIDIA architecture generations;
- one AMD and one NVIDIA device repeated on a second driver/OS stack when
  practical, such as Linux and Windows;
- one integrated/UMA device as a sensitivity case if the paper discusses
  memory architecture.

Classify `VK_KHR_unified_image_layouts` from the recorded device-driver pair
rather than assuming support from the GPU model. Include both supporting and
non-supporting implementations if available, but do not block the study on a
new extension with sparse driver availability.

Do not pool vendor or driver results. Report every device-driver pair first,
then discuss repeated patterns. Apple/Metal is a separate case study because
its driver/framework may track hazards and it does not expose Vulkan image
layouts; it is not required for the core Vulkan claim. Intel Vulkan is a useful
stretch target, not a requirement for an AMD/NVIDIA-scoped paper.

## Statistical analysis

- Preserve every valid raw sample.
- Plot distributions and time-ordered samples before aggregation.
- Report median, interquartile range, and a 95% bootstrap confidence interval
  for the median and paired difference.
- For counterbalanced paired runs, analyze within-pair deltas.
- Report absolute nanoseconds/milliseconds alongside percentages.
- Define a practical-equivalence region before interpreting “no difference.”
- Correct for multiple comparisons when making device/workload-wide claims.
- Treat profiler counters as explanatory evidence, not independent benchmark
  repetitions.

No outlier may be removed solely because it weakens the conclusion. Exclusions
must follow a recorded event such as compilation, device loss, profiler
attachment, thermal throttling, or another preregistered rule.

## Raw data layout

Use:

```text
paper/data/
  raw/
    <collection-id>/
      metadata.txt
      vulkaninfo.txt
      order.txt
      <configuration>.csv
  derived/
    <analysis-version>/
      summary.csv
      figures/
      tables/
```

The initial benchmark emits comment-prefixed metadata followed by CSV rows:

```text
sample,workload,policy,passes,elements,rounds,width,height,start_ns,record_ns,submit_ns,wait_ns,gpu_ns,gpu_pass_count
```

Metadata includes `implementation` (`blade` or `wgpu`) so matched results remain
separate even when they share a backend and device.

Derived files must be reproducible from raw files without manual spreadsheet
edits. `paper/analyze.py` is the initial standard-library-only summarizer. It
reports both within-Blade policy differences and Blade-automatic versus
wgpu-tracked differences; its bootstrap seed is derived deterministically from
each configuration key.

## Correctness rules

- A no-barrier run is valid only when the workload graph contains no conflicting
  resource accesses.
- Dependent workloads must compare semantically equivalent synchronization.
- Initialization, host visibility, queue ownership, and presentation remain
  explicitly synchronized and are not counted as evidence that steady-state
  layout tracking is unnecessary.
- Validation is evidence against known API misuse, not a proof of correctness.
- Every experimental backend should pass deterministic output tests on all
  collected drivers before performance collection.

## Result gates

The paper can lose its `RESULTS PENDING` banner only after:

- the matched wgpu workloads reproduce Blade's workload graph and outputs;
- the paper consistently labels Blade-versus-wgpu results as end-to-end and
  reserves causal claims for the within-Blade barrier-placement control;
- representative Vulkan captures and CPU profiles support the explanation of
  tracked-versus-coarse behavior;
- at least the minimum hardware matrix is complete;
- raw data and analysis scripts reproduce every table and figure;
- negative and architecture-specific results are described alongside wins;
- the abstract contains absolute effects and uncertainty, not only ratios.
