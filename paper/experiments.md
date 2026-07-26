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

The within-Blade benchmark controls factors 1 and 3 directly (placement, and
stage/access scope via `BarrierScope`). Factors 2 and 4 are observed in the
matched wgpu implementation and in captures of the commands both actually
emit. Because that comparison also changes validation, lifetime management,
bindings, and command representation, it supports an end-to-end design
comparison and explanatory profiling—not single-factor causal attribution.

What the captures established (see `capture-streams.py`, output `barriers.csv`):

- factor 1: wgpu emits exactly as many barriers as `B-hazard` — one on the
  independent workloads, four on the chains, against `B-auto`'s five.
- factor 2: every Blade barrier is a global `VkMemoryBarrier`; every wgpu
  barrier is a `VkBufferMemoryBarrier` or `VkImageMemoryBarrier`. Neither emits
  the other's kind anywhere.
- factor 4: no steady-state layout transition occurs in either. Blade emits no
  image barrier at all; wgpu's carry `COLOR_ATTACHMENT_OPTIMAL` on both sides.
  The layout difference is static, not a per-pass cost.

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

**The workload set is closed.** These six vary one factor — whether
consecutive passes carry a dependency — across the three pass kinds a
synchronization policy can distinguish, and the matched wgpu program covers the
four it can. Extensions considered and declined, with what each costs:

- *Application frames* (Bunnymark, the particle system, a multi-pass scene).
  Would test the decision boundary on real heterogeneity. Declined: scenes,
  content, and driver heuristics vary in ways this design exists to hold still,
  so it trades a controlled comparison for a plausible one. The boundary stays
  labelled a hypothesis.
- *MSAA targets.* The one case where RADV disables a compression path for
  `GENERAL`, so the one place a persistent-layout policy should cost bandwidth
  on AMD. Declined as niche in current renderers; the paper states the
  exception from driver source and does not test it.
- *Dependency-density DAGs, resource-count sweeps, format/depth/mip/array
  sweeps.* Each adds a factor the study does not claim about.

Adding any of these means reopening the collection on every machine. The bar
for doing so is a claim the paper wants to make and cannot.

## Implementation changes considered

Two came out of reading RADV alongside the results, and neither is measured
here.

- *Skip a pass-boundary barrier when nothing has been recorded since the last
  one.* `barrier_before` always emits. `since_barrier` is empty only
  immediately after an explicit `barrier()` with no pass in between, and an
  explicit barrier's destination scope is always `ALL_COMMANDS`, so skipping
  in that case is safe in both scopes and needs no resource state — one more
  bit on the encoder, not a table. It is not measured because no configuration
  in this study can reach the case: `manual_barriers` is on for exactly the
  policies that call `barrier()`, so automatic and explicit barriers never mix
  in the benchmark. Worth doing on its merits; it will not move any number
  here.
- *Avoiding RADV's global-barrier pessimism* — the unconditional
  `FLUSH_AND_INV_CB_META` / `FLUSH_AND_INV_DB_META`, and the coherence check
  that a memory barrier cannot use. Both are cleared only when the barrier
  carries an image, which is the state the design declines to keep. Measuring
  the narrowed scope already put an upper bound on what this is worth on AMD,
  and that bound is zero.

The remaining AMD lever is barrier placement, which is application knowledge
and is already exposed by `manual_barriers`.

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

No practical-equivalence region is used, and none was preregistered. An
earlier version of this document derived a single ±3% band from the worst
`B-explicit-all` control deviation; with every machine collected, control
floors run from 0.0% to 71.3%, so no single band describes them. Each claim is
compared instead against the control floor of the cell it is made in, which the
scope table reports beside it. A future collection should preregister the rule,
not the number.

- Preserve every valid raw sample.
- Plot distributions and time-ordered samples before aggregation.
- Report median, interquartile range, and a 95% bootstrap confidence interval
  for the median and paired difference.
- For counterbalanced paired runs, analyze within-pair deltas. `analyze.py`
  does not yet do this: it pools samples across repetitions and bootstraps
  unpaired. The repetition index is the blocking factor that captures clock
  state, so a paired estimator would tighten the noisy cells; it has not been
  needed for the claims made so far, because each is checked against its own
  cell's floor and the ones that fail are reported as failing.
- Report absolute nanoseconds/milliseconds alongside percentages.
- Compare “no difference” against the control floor of that cell, never
  against a global band.
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
4. The RX 7900 XT accelerates while it is being measured. Its clocks were
   pinned with `power_dpm_force_performance_level = high` -- `rocm-smi` in the
   collection records it, and also records a 35 MHz shader clock at idle
   underneath -- and its median block still runs 18.4% faster in its last
   third than in its first. Every one of its floors is above 2%. More warm-up,
   not more pinning, is what this needs.
5. Only `zork` has a timestamp-free collection, so host-cost claims are
   established there and merely corroborated elsewhere.

## Control floor, per cell

The instrumentation control (`explicit-all` against `automatic` at global
scope) emits identical commands, so its disagreement bounds what can be
resolved. Nothing smaller may be claimed. `build-tables.py` computes it and
prints it as the `ctrl` column of the scope table.

Two things about how it is computed have each been got wrong once.

It must be read **per cell, not per device**. Noise here is a property of the
workload on the device, not of the device: on the RX 7900 XT one compute cell
resolves to 0.1% and the other to 10.9%, and on the Intel part
`graphics-chain` is 1.7% while `mixed-independent` is 49.7%. An earlier version
took the worst cell per device as the device's floor, which discarded every
usable AMD and Intel cell and turned two answerable questions into open ones.

It is the **far end of the control's bootstrap interval, not its point
estimate**. On the Intel part's `compute-independent` cell the two identical
configurations agree to 2.8% on the median and their interval reaches 71.3%,
because that machine's independent-target blocks step down by nearly half
partway through the measurement and a median records when the step happened.
The point
estimate would have licensed a 14% placement reading from that cell; the
interval says the cell cannot resolve 14%. Switching to the interval moved the
count of cells with floors above 2% from 9 to 16 and withdrew one claim.

What the disqualified cells have in common was found by asking what the
control was reacting to. A block is one process launch, and the device
accelerates inside it: an RX 7900 XT `compute-chain` block opens at 245
microseconds, holds fourteen iterations, then steps down through 225, 217, 205
to 193 and is still descending when the block ends. Comparing the median of a
block's first third against its last third separates the machines the same way
the floors do -- -0.1% on the RTX 5070 and +0.0% on Raphael against -18.4% on
the RX 7900 XT and -55.3% in the worst Intel block. `build-tables.py` reports
it.

The single change that would most improve a repetition of this study is
therefore warm-up, and it is a flag rather than a code change: `--warmups 2000`
costs about a minute across a whole collection, against ten warm-ups today that
leave the RX 7900 XT still accelerating forty iterations later. Pinning clocks
is worth doing and is not sufficient on its own.

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
- [~] representative Vulkan captures and CPU profiles support the explanation
  of tracked-versus-coarse behavior. CPU profiles are collected
  (`profile-hosts.py`) and reported: resource tracking is 3-7% of wgpu's
  process CPU time, which rules it out as the explanation for a 2-12x host-cost
  gap. Command-stream captures of both implementations (`capture-streams.py`,
  which downloads RenderDoc if none is installed) confirm the emitted masks and
  counts, and settle factors 2 and 4: every Blade barrier is global and every
  wgpu barrier is resource-scoped, and neither performs a steady-state layout
  transition;
- [x] at least the minimum hardware matrix is complete (NVIDIA and discrete
  AMD on Linux, plus two AMD integrated parts, Intel, and Apple as sensitivity
  cases);
- [x] raw data and analysis scripts reproduce every table (`build-tables.py`);
- [x] negative and architecture-specific results are described alongside wins
  (the Radeon 780M regression and the failed RADV scope prediction are both
  reported in the abstract);
- [x] the abstract contains absolute effects and uncertainty, not only ratios.

Application workloads remain outstanding independently of the banner.
