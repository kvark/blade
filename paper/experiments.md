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

- barrier-call counts: wgpu emits exactly as many calls as `B-hazard` — one on
  the independent workloads, four on the chains, against `B-auto`'s five.
  Replaying Vulkan command buffers in queue-submission order shows the endpoint
  difference hidden by those counts: wgpu has an initial transition and no
  final barrier; `B-hazard` has no initial barrier and retains Blade's final
  encoder barrier. Both put three calls between the four dependent passes;
  neither puts one between independent passes.
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
| `B-hazard` | application-declared dependent workload | global | broad | `GENERAL` | collected |
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

Neither needs anything from the caller for automatically placed barriers, so
the pass declaration is unchanged. The destination of an explicitly placed
barrier cannot be derived because its consumer has not been declared.
`CommandEncoder::barrier` therefore takes a `BarrierScope` argument and emits
the request immediately with a pass-kind source and an `ALL_COMMANDS`
destination. It is not held until the next pass.

Because a `-scoped` policy places identical numbers of barriers at identical
boundaries to its unscoped twin, their difference is attributable to scope
alone — the first single-factor claim this study can make about factor 3.

Correctness evidence: a development-time Khronos synchronization-validation
run reportedly passed with no hazards or validation errors, and the retained
performance runs agree on the output hash they sampled. Validation earned its
place here: it caught that Blade's `extra_sync_*_access` driver workarounds add
transfer accesses that a narrowed stage mask does not support. The current raw
directories retain the sampled output hashes but not the
synchronization-validation logs; rerunning the automated correctness
collection, with the artifact's stronger all-output hash, and retaining its
logs is a publication gate.

Result so far: the narrowing pays on NVIDIA render boundaries and on the
RX 7900 XT's compute boundaries, but no AMD render-boundary saving resolves.
That is not the pattern the first RADV source reading predicted. Reading a
driver establishes which operations a barrier requests, not what they cost.

Blade emits a barrier when finishing every encoder, including in manual mode;
its destination scope is always wide because the next queue consumer is
unknown. Host visibility separately relies on queue completion and mapped
memory coherency/invalidation. That barrier is common to all six Blade
configurations. At **global** scope, `B-explicit-all` additionally emits a
manual barrier before every pass and produces the same barrier calls as
`B-auto`; it is the device-time control. At pass-kind scope the explicit
barrier has a wide destination while the automatic one knows its consumer, so
`B-explicit-all-scoped` is a crossed experimental configuration, not an
identical-command control.

`W-wgpu` is the realistic tracked comparison. It must not be used alone to
attribute a difference to tracking, because validation, lifetime management,
bindings, and command representation also differ. The current whole-process
profiles do not attribute host costs; a gated or batched record-and-submit
profile would be needed for that. Captured Vulkan barriers and layouts explain
what was requested without decomposing its cost. A direct `wgpu-hal` port is
optional if a particular result needs frontend decomposition; no new tracker
in Blade is required.

The matched wgpu program uses wgpu's native `IMMEDIATES` feature for the
16-byte per-pass parameters that Blade supplies as inline uniform data
(`IMMEDIATES` maps to push constants on Vulkan). This
avoids adding a tracked uniform-buffer resource only to the wgpu side. The
baseline is therefore native wgpu/wgpu-core, not browser WebGPU. CPU-only
collections disable timestamp queries; GPU-timed collections use native
encoder timestamps and do not use their host timings for primary CPU claims.
The benchmark also uses wgpu's trusted-shader API to select Blade's unchecked
shader-runtime-check policy; it is not a browser-safe WebGPU baseline.

## Workload families

The headless `sync-bench` artifact provides:

| Workload | Resources | True inter-pass hazard | Purpose |
|---|---|---:|---|
| `compute-independent` | shared read-only input, unique outputs | no | cost of redundant compute barriers |
| `compute-chain` | ping-pong storage buffers | yes, every pass | cost when coarse placement is already exact |
| `graphics-independent` | unique color targets | no | cost of redundant graphics barriers |
| `graphics-chain` | one load/store color target with blending | yes, every pass | serialized graphics control |
| `mixed-independent` | alternating compute and render, unique per pass | no | scope and placement together across pass kinds |
| `mixed-chain` | alternating compute and render, each chained within its kind | each pass from the third depends two positions back | scope with automatic placement held fixed |

The mixed families are Blade-only: the matched wgpu benchmark does not
implement them, and the collector runs wgpu on the four shared workloads only.
`mixed-chain` is a clean test of barrier scope when comparing
`B-auto-scoped` with `B-auto`, because placement is then identical. The
hazard-only policy places a barrier before every pass except the first and
therefore differs from automatic by the initial barrier, just like the
single-kind chains. Its barrier before the second mixed pass is conservative:
those first compute and render passes are independent, so `hazard-only` is not
an edge-minimal schedule for this one workload.

**The workload set is closed.** These six contrast independent resource sets
with chain dependency patterns across the pass kinds a synchronization policy
can distinguish, and the matched wgpu program covers the four it can.
Extensions considered and declined, with what each costs:

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
  carries an image, which is the state the design declines to keep. The
  narrowed scope measures what can be removed without naming a resource. It
  saves time on the discrete part's compute boundaries but not its render
  boundaries; that is not an upper bound on image-scoped barriers.

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
7. Warm each configuration until time-ordered samples stop drifting; the
   present fixed matrix uses 1000 warm-ups because the unretained pilot
   indicated that ten was insufficient. Collect at least 30 measured
   iterations inside each process.
8. Counterbalance configuration order across repetitions. Retain order in the
   metadata so drift and thermal bias can be analyzed.
9. Run one configuration per process when comparing host memory and startup
   effects. Also run an in-process alternation as a sensitivity check.
10. Use at least ten independently launched, randomized process repetitions
    for final inferential claims. The current three-repetition dataset is an
    exploratory minimum. Repeat a representative subset on a second day or
    reboot.
11. For the matched wgpu run, freeze the exact wgpu revision and capture one
    representative capture per workload. Confirm that the extracted barrier
    records contain the expected resource barriers and layouts before
    interpreting GPU-time differences; do not call the extracted table the
    complete command stream.

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
`B-explicit-all` control deviation, but controls vary substantially by cell.
Each claim is compared instead against the control floor of the cell it is made
in, which the generated figure reports beside it. This is a conservative
post-hoc stability diagnostic, not a confidence bound or equivalence test. A
future collection should preregister the rule rather than estimate it after
collection.

- Preserve every valid raw sample.
- Plot distributions and time-ordered samples before aggregation.
- One outer repetition is a randomized matrix block with one separately
  launched process per configuration. Pair a contender process with its
  `B-auto` process from the same outer block. The 30 samples inside either
  process share clocks, thermal state, and run order and are not 30
  independent repetitions.
- For each outer block, compare the contender process median with its paired
  `B-auto` process median. Report the median block-level percent effect and a 95%
  hierarchical-bootstrap interval that resamples blocks first and
  observations inside each selected block second. `analyze.py` and
  `build-tables.py` implement this deterministic estimator.
- The current within-process resampling treats the ordered observations as
  exchangeable. Inspect their autocorrelation and add a moving-block or
  process-median-only sensitivity analysis before treating the interval as
  inferential.
- Report the process count as well as the observation count. The current
  `n=3` process blocks expose observed instability but provide weak tail
  estimation; final collection should use at least ten.
- Report absolute nanoseconds/milliseconds alongside percentages.
- Read a directional effect only when its hierarchical interval lies wholly
  beyond the same-side control floor for that cell.
- No multiplicity-adjusted hypothesis tests are currently reported. Treat the
  device-by-workload matrix as exploratory effect estimation; add a
  preregistered correction before making family-wide null-rejection claims.
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

1. Each fixed-matrix configuration has three process repetitions. That is
   enough to reveal the pseudoreplication in the former pooled analysis but too
   few for strong tail estimation. Increase this before an archival run.
2. Metal GPU timestamps are per pass rather than a span in Blade, and were
   returned as zero by wgpu for three of the four workloads. Metal device time
   is not reported; host wall time is shown only as wall time.
3. Only `zork` has a timestamp-free collection. Cross-device host results use
   GPU-timed collections for direction; uninstrumented magnitude is established
   only by the `zork` sweep.
   Blade also uses 17 device timestamp writes and a window through its final
   barrier, while wgpu uses two writes ending with the final pass; their GPU
   spans are instrumented end-to-end comparisons, not matched timer streams.
4. `profile-hosts.py` samples the whole process, including a completion wait
   after each iteration. Those profiles describe whole-process shares but do
   not decompose the record-and-submit interval. The retained profiles were
   also not adapter-pinned: on `rubik`, Blade selected the Raphael iGPU while
   wgpu selected the RX 7900 XT. No between-implementation inference is drawn
   from those columns.
5. The new Metal matrix collection used shader parity, but Low Power Mode was
   enabled. The separate tracked/untracked Metal pilot retained only printed
   summaries from one process session.
6. At the 2026-07-27 audit, the Radeon 780M (`k6`) matrix, profile, and capture
   directories cited by the paper were absent from this checkout's
   `paper/data/raw`. `build-tables.py` now fails by default in that condition;
   `--allow-incomplete` exists only for partial audits.
7. The numerical `--shader-checks` sensitivity run and the development-time
   synchronization-validation logs are not present in the retained raw
   directories. The retained revision also hashed only one representative
   output in each independent family and used a mistyped FNV-1a multiplier in
   both implementations. The latter does not affect within-collection equality
   checks, but newly collected digest strings use the corrected standard
   multiplier and therefore differ from the old strings. During this audit the
   retained wgpu graphics shader also failed Vulkan SPIR-V validation under
   `VUID-StandaloneSpirv-None-10684`; the equivalent array-free shader now
   passes the automated correctness matrix. The retained graphics-chain
   readback row also saturated to all `0xff`, making its hash insensitive to
   missing late passes; the current shader keeps that row unsaturated through
   64 passes. The paper no longer quotes the unretained sensitivity result; the
   complete timing matrix must be
   recollected from the fixed shader, and the current all-output correctness
   hashes and validation logs must be published.
8. The current matrix has neither the planned in-process alternation
   sensitivity check nor a second-day/reboot repeat. Its inner bootstrap also
   does not model serial correlation among the 30 ordered observations. These
   are required before interpreting the resampling intervals as population
   confidence intervals.
9. The retained capture manifests identify their three hosts but do not record
   the adapter selected by either implementation, and they cover only
   `automatic`, `automatic-scoped`, and `hazard-only`. Their extracted request
   tables are valid observations of those streams, but they are not evidence
   from three named GPU models and do not cover the explicit/crossed controls.
   The current capture runner includes all six Blade policies, requires one new
   capture per run, records benchmark metadata, compares output hashes, and
   rejects different Blade/wgpu device names. Recollect with both adapter
   selectors pinned.

## Control floor, per cell

The instrumentation control (`explicit-all` against `automatic` at global
scope) produces the same barrier calls with identical arguments by
construction. The current capture set does not include this control;
`capture-streams.py` now includes it by default, so the final captures will
check the source-level equivalence directly.
Its device-time disagreement is used as a conservative post-hoc stability
threshold.
`build-tables.py` defines the floor as the larger absolute endpoint of that
cell's paired hierarchical interval.

The floor must be read **per cell, not per device**. It must also come from the
interval, not only the point estimate: process repetitions can occupy different
clock or interference regimes even when the median control effect is near
zero. A directional claim is admitted only when its entire interval lies
beyond the same-side floor for that cell. The generated numbers and figures are
the source of current floor values; this protocol deliberately does not copy
them into prose where a recollection can leave them stale.

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
  (the generated hash-group count reports zero conflicts);
- [x] the paper consistently labels Blade-versus-wgpu results as end-to-end and
  reserves causal claims for the within-Blade barrier-placement control;
- [~] host attribution is not complete. Existing `profile-hosts.py` data cover
  whole processes dominated by waits and cannot apportion the
  record-and-submit gap. The paper now treats them as diagnostic only; this
  becomes a blocker only if a tracking-specific host attribution claim is
  restored;
- [ ] a retained synchronization-validation collection covers all six Blade
  workloads and policies with no reported hazards or validation errors;
- [~] representative capture extraction confirms barrier masks, counts,
  pass-relative positions, resource scope, and steady-state layouts. The
  extracted `barriers.csv` files are not complete command streams; the retained
  manifests do not identify selected adapters or cover the explicit/crossed
  policies, and the Radeon 780M capture is currently missing from this
  checkout;
- [~] at least the minimum hardware matrix was collected (NVIDIA and discrete
  AMD on Linux, plus two AMD integrated parts, Intel, and Apple as sensitivity
  cases), but the Radeon 780M raw directory must be restored locally;
- [ ] raw data and analysis scripts reproduce every table. The generator now
  fails loudly because a cited collection is absent;
- [x] negative and architecture-specific results are described alongside wins
  (the Radeon 780M regression and the failed RADV scope prediction are both
  reported in the abstract);
- [x] the abstract contains measured host times and device effect estimates
  with uncertainty, not only unqualified ratios.

Application workloads remain outside the chosen scope. The unchecked
reproducibility, process-repetition, and validation-artifact items are
publication blockers. Host attribution is a blocker only for a
tracking-specific decomposition, which the current paper does not claim.
