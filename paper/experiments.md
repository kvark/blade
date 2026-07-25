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
| `B-auto` | every pass | global | broad | `GENERAL` | collected |
| `B-hazard` | declared hazards | global | broad | `GENERAL` | collected |
| `B-explicit-all` | every pass | global | broad | `GENERAL` | collected as a control |
| `W-wgpu` | derived by wgpu | tracked resource usage | backend-generated | backend-generated | collected |

Each of the three `B-*` rows is collected again with `BarrierScope::PassKind`,
giving six Blade configurations: placement crossed with scope, so neither axis
is a default for the other. The `-scoped` names in the data are `automatic-scoped`,
`hazard-only-scoped`, and `explicit-all-scoped`.

The `-scoped` variants promote factor 3 from observed-only to directly
controlled. The barrier is still one global memory barrier naming no resource,
but its stages and accesses are derived:

- source: the union of the pass kinds recorded since the previous barrier,
  kept as one bitmask on the encoder. Taking the union rather than the last
  pass alone is what keeps it correct when the application suppressed
  boundaries.
- destination: the kind of the pass being opened.

Neither needs anything from the caller, so the pass declaration is unchanged.
The one thing that cannot be derived is the destination of an explicitly placed
barrier, whose consumer has not been declared yet; `CommandEncoder::barrier`
therefore takes a `BarrierScope` argument, and the request is emitted when the
next pass opens so it can still name its consumer.

Because a `-scoped` policy places identical numbers of barriers at identical
boundaries to its unscoped twin, their difference is attributable to scope
alone — the first single-factor claim this study can make about factor 3.

Correctness evidence: the Khronos synchronization validation checks pass with
no hazards and no validation errors on all six workloads times six policies,
and every policy produces the same output hash. Validation earned its place
here: it caught that Blade's `extra_sync_*_access` driver workarounds add
transfer accesses that a narrowed stage mask does not support.

Result so far: the narrowing pays on NVIDIA (up to -5.3% of GPU span at fixed
placement) and not on either AMD device, which is the opposite of what the
RADV reading predicted. Reading a driver establishes which commands are
emitted, not what they cost.

Blade emits a barrier when finishing every encoder, including in manual mode;
its destination scope is always wide because it has to reach the next
submission and the host. That barrier is common to all six Blade
configurations. `B-explicit-all` additionally emits a manual barrier before
every pass, making it command-for-command equivalent to the automatic policy at
the same scope; it is a control for benchmark and instrumentation artifacts
rather than a distinct synchronization strategy.

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
| `mixed-independent` | alternating compute and render, unique per pass | no | scope and placement together across pass kinds |
| `mixed-chain` | alternating compute and render, each chained within its kind | yes, every pass | scope with placement held fixed |

The mixed families are Blade-only: the matched wgpu benchmark does not
implement them, and the collector runs wgpu on the four shared workloads only.
`mixed-chain` is the cleanest single-factor test of barrier scope, because
`B-hazard` places exactly the same barriers there as `B-auto`.

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
6. Lock clocks or force a stable performance level; see COLLECTING.md for the
   per-vendor commands. Record the command and setting; never silently discard
   throttled samples. Check the per-device control floor before interpreting
   any effect.
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

The practical-equivalence region was **not** preregistered. It is derived after
the fact from the `B-explicit-all` control, whose command stream is identical to
`B-auto`: excluding the one cell that the control itself rejects, the largest
control deviation observed on any device is 2.6%. GPU-span differences within
±3% are therefore reported as practically equivalent, and the derivation is
stated wherever the region is used. A future collection should preregister it.

- Preserve every valid raw sample.
- Plot distributions and time-ordered samples before aggregation.
- Report median, interquartile range, and a 95% bootstrap confidence interval
  for the median and paired difference.
- For counterbalanced paired runs, analyze within-pair deltas. `analyze.py`
  does not yet do this: it pools samples across repetitions and bootstraps
  unpaired. The repetition index is the blocking factor that captures clock
  state, so a paired estimator would tighten the noisy cells; it has not been
  needed for the claims made so far, because those rest on cells whose control
  floor is already below 1.5%.
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

Sweep collections append `__pNNNN` to the run ID, and their `order.csv` carries
a `passes` column. `analyze.py` already groups by `passes`, so a sweep needs no
special handling.

Derived files must be reproducible from raw files without manual spreadsheet
edits. `paper/analyze.py` is the initial standard-library-only summarizer. It
reports both within-Blade policy differences and Blade-automatic versus
wgpu-tracked differences; its bootstrap seed is derived deterministically from
each configuration key. `paper/build-tables.py` turns the raw collections
directly into the LaTeX tables that `main.tex` includes, so no number in the
paper is transcribed by hand.

## Known deviations in the current collections

1. The first `rubik` collection ran Blade on the Raphael integrated GPU and
   wgpu on the RX 7900 XT. The collector now aborts on this condition and
   collects every enumerated adapter in turn; `rubik` has been recollected as
   one collection per GPU with both implementations pinned. The defective
   collection is superseded and no longer retained.
2. Metal GPU timestamps are per pass rather than a span in Blade, and were
   returned as zero by wgpu for three of the four workloads. Metal device time
   is not reported; host wall time is used instead.
3. `20260725T160725Z-matrix` fails its own `B-explicit-all` control on
   `graphics-independent` (-37.4%, repetition medians spanning 3.2-5.6 ms for an
   identical command stream). The cell is reported and rejected, not deleted.
4. The RX 7900 XT's compute and mixed workloads run for about 200 microseconds,
   which is too short for its clocks to settle: repetition medians for an
   identical command stream vary by up to 16%, and its control reaches 6.5%.
   Only effects above roughly 10% are claimable on those cells. Its graphics
   workloads are stable to under 1%. Clock-locking or a longer workload would
   fix this and has not been done.
5. Only `zork` has a timestamp-free collection, so host-cost claims are
   established there and merely corroborated elsewhere.

## Control floor, per cell

The instrumentation control (`explicit-all` against `automatic` at global
scope) emits identical commands, so its disagreement bounds what can be
resolved. Nothing smaller may be claimed. `build-tables.py` computes it and
prints it as the `ctrl` column of the scope table.

It must be read **per cell, not per device**. Noise here is a property of the
workload on the device, not of the device: on the RX 7900 XT the compute cells
are worth 3.6-6.5% while the graphics and mixed cells are 0.1-0.8%, and on the
Intel part `graphics-chain` is 0.0% while `mixed-independent` is 48%. An
earlier version of this analysis took the worst cell per device as the device's
floor, which discarded every usable AMD and Intel cell and turned two
answerable questions into open ones.

The pattern behind the bad cells is short workloads on large idle GPUs: a
200-microsecond command buffer never takes the device out of its idle clock
state. Clock-locking (see COLLECTING.md) raises the number of usable cells and
is part of the protocol; it is not a precondition for a result whose cell floor
is already low.

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

- [x] the matched wgpu workloads reproduce Blade's workload graph and outputs
  (all 216 matrix runs and 768 sweep runs agree on the per-workload hash);
- [x] the paper consistently labels Blade-versus-wgpu results as end-to-end and
  reserves causal claims for the within-Blade barrier-placement control;
- [ ] representative Vulkan captures and CPU profiles support the explanation of
  tracked-versus-coarse behavior — **outstanding**; the Intel end-to-end gap is
  currently reported as unexplained rather than attributed;
- [x] at least the minimum hardware matrix is complete (NVIDIA and discrete
  AMD on Linux, plus two AMD integrated parts, Intel, and Apple as sensitivity
  cases);
- [x] raw data and analysis scripts reproduce every table (`build-tables.py`);
- [x] negative and architecture-specific results are described alongside wins
  (the Radeon 780M regression and the failed RADV scope prediction are both
  reported in the abstract);
- [x] the abstract contains absolute effects and uncertainty, not only ratios.

Application workloads remain outstanding independently of the banner.
