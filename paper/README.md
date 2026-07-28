# Blade synchronization paper

This directory contains the working technical report and its experiment
protocol. The build is designed so every quantity drawn from a retained
matrix, sweep, profile, or capture is generated from the raw collections and
the prose cannot drift from the data. The explicitly exploratory Metal hazard
pilot is the exception because only its printed summaries survived. The build
currently fails deliberately until the cited Radeon 780M collection is
restored; the remaining open gates are stated where they bound a claim.

## Draft

Regenerate the tables before building, because both the tables and the numbers
the prose cites come out of the same script:

```bash
python3 build-tables.py
latexmk -pdf main.tex     # or: tectonic -X compile main.tex
```

`build-tables.py` writes `data/derived/tables/*.tex`, which `main.tex` pulls in
with `\input`. It uses only the standard library and re-uses `analyze.py` for
paired hierarchical-bootstrap intervals. It fails before writing tables if
`main.tex` cites a device whose matrix collection is absent. The
`--allow-incomplete` switch exists only for partial audits whose output is not
expected to build the paper.

`numbers.tex` is the part that keeps the prose honest: it defines every
measured quantity the text quotes as a LaTeX macro keyed by device, workload,
and comparison, so `\bmagci{rx7900xt}{compute-independent}{placement}` in a
sentence and the corresponding table cell use the same paired process-level
estimator. Citing a key the collections do not supply is a build error rather
than a blank. The unsigned forms (`\bmag`, `\bmagci`) are deliberately
undefined for cells whose interval spans zero, so "worth X%" cannot be written
about a cell that does not support it.

A full build needs `booktabs` and `xcolor`, which Debian's `texlive-latex-base`
does not include; `texlive-latex-recommended` or a `tectonic` binary covers
them.

The source uses a generic two-column article layout so that the work is not
tied to a venue before the evaluation is complete. It can later be moved to a
venue template without rewriting the content.

## Experiments

The experiment contract is in [experiments.md](experiments.md). The initial
headless benchmark is:

```bash
cargo run --release --example sync-bench -- \
  --workload compute-independent \
  --policy automatic
```

Raw measurements belong under `data/raw/` and generated tables under
`data/derived/`. Neither is tracked by git -- see [data/README.md](data/README.md).
Every result used by the paper must retain the raw rows and machine metadata
from which it was derived, which the collection directory does.

With Blade and the benchmark wgpu branch checked out as sibling directories,
one command collects everything this machine can contribute:

```bash
python3 paper/collect.py --wgpu ../wgpu
```

On Vulkan that first runs a small correctness matrix with the Khronos
synchronization-validation feature forced on, retains its output, and aborts on
an output-hash mismatch or validation error. It then runs the timing matrix
over every adapter, followed by a host CPU profile and RenderDoc captures of
both implementations. The last two need `perf` and RenderDoc respectively; if
either is unavailable it says so and carries on, because a machine that can
only contribute correctness evidence and timings should still contribute
them. The default is ten randomized process repetitions. Add `--sweeps` for
the pass-count sweeps, which take roughly another half hour. Unrecognized
arguments pass through to the timing-matrix runner. Use `--skip-validation`
only when a retained correctness collection already covers the exact sources,
driver, and OS.

The main steps can also be run on their own:

```bash
python3 paper/run-study-matrix.py --wgpu ../wgpu --validation  # correctness
python3 paper/run-study-matrix.py --wgpu ../wgpu               # timings
python3 paper/profile-hosts.py    --wgpu ../wgpu     # host CPU profile
python3 paper/capture-streams.py  --wgpu ../wgpu     # extracted barrier records
```

Add `--pass-list 1,2,4,8,16,32,64,128` to the matrix runner to sweep the pass
count instead of measuring a single point, and `--blade-device-id` /
`--wgpu-adapter-name` to pin one adapter instead of collecting all of them.

See [COLLECTING.md](COLLECTING.md) for adapter discovery, correctness runs,
separate CPU/GPU collections, Linux, Windows, and macOS commands, profiling,
and capture requirements. `run-sync-matrix.sh` remains as a Blade-only pilot
runner; it is not the final matched collector.
`tools/metal-hazard-bench.swift` is the standalone Metal tracked-versus-untracked
harness; its results live in
[`../docs/metal-hazard-tracking.md`](../docs/metal-hazard-tracking.md).

Summarize a collection with deterministic paired hierarchical intervals:

```bash
python3 paper/analyze.py paper/data/raw/<collection-id> \
  --output paper/data/derived/<analysis-id>
```

The collection workflow on each machine is:

1. Check out the frozen benchmark commit and build `sync-bench` in release
   mode.
2. Run small validation-enabled cases for all workloads and policies; retain
   their output as correctness evidence.
3. Record the selected adapter ID, stabilize clocks/power and background load,
   then run the randomized matrix with validation disabled.
4. Copy the entire raw collection directory back to the analysis machine
   without editing its CSV or metadata files.
5. Run `analyze.py`, inspect time-ordered samples for drift or throttling, and
   retain both raw and derived directories.
6. Use at least ten randomized process repetitions for the final collection.
   Capture one representative run per workload, and profile a record-only or
   batched-recording path rather than a process dominated by completion waits.

`run-study-matrix.py` automates the matched Blade and wgpu collection,
randomizes both implementations together, and rejects mismatched output hashes.

The initial benchmark isolates the cost of Blade's unconditional pass-boundary
barriers from barriers placed only at application-declared hazards. The paper
does not require a second resource tracker inside Blade. Resource-tracked
behavior comes from matched wgpu workloads and captured Vulkan barrier records,
and is explicitly reported as an end-to-end comparison rather than a minimal
oracle. The current
whole-process CPU profiles are diagnostic only; they do not isolate the
record-and-submit interval.

## Collected so far

`build-tables.py` discovers collections under `data/raw/` and classifies them
by what they contain, so a retest lands in the tables without editing any
script. The current set is the archival round of 2026-07-28: every machine on
blade `87ed067` + wgpu `7d37a77` from clean trees, ten process repetitions,
1000 warm-ups, a retained validation collection per Vulkan machine (the small
`val` entries), and captures of all six policies on four machines. The study
host's sweeps add one paper-only commit.

| Collection | Machine | Device measured by Blade | Role |
|---|---|---|---|
| `20260728T035118Z-zork` | zork | NVIDIA RTX 5070 | matrix, placement × scope (val: `035105Z`) |
| `20260728T035844Z-rubik-...rx-7900-xt...` | rubik | AMD RX 7900 XT | matrix (val: `035824Z-...`) |
| `20260728T035844Z-rubik-...ryzen-5-9600x...` | rubik | AMD Raphael iGPU | matrix (val: `035824Z-...`) |
| `20260728T035352Z-k6` | k6 | AMD Radeon 780M | pass-count sweep 1-64, GPU-timed; its 16-pass cells are the 780M matrix (val: `100356Z`) |
| `20260728T035923Z-matrix` | matrix | Intel Xe (RPL-U) | matrix (val: `035920Z`) |
| `20260728T034901Z-mac` | mac | Apple M3 | matrix, Metal case study |
| `20260728T…-zork` | zork | NVIDIA RTX 5070 | pass-count sweeps 1-64, GPU-timed and timestamp-free |

Per-host `*-profile` and `*-captures` directories accompany the round on
rubik, matrix, and k6 (zork's profile is pending a re-run).
`build-tables.py` keeps only the most recent matrix-capable collection per
machine and device --- a dedicated matrix wins over a sweep standing in for
one --- so a retest supersedes an earlier run rather than appearing beside
it. The two `rubik` directories come from one invocation: the collector runs
every enumerated adapter in turn.

Because `data/raw/` is not in git, a collection exists only where it was
copied. `build-tables.py` now fails when `main.tex` cites a device whose matrix
data are absent, since otherwise table rows can vanish while the prose keeps
referring to them. Restore the named collections before building the PDF.
Before submission, publish the immutable raw directories and checksums as a
versioned artifact; local, ignored data are not a reproducibility story.

The result gates in [experiments.md](experiments.md) are not all met. In
addition to the missing k6 artifact and only three process repetitions, the
development-time synchronization-validation logs were not retained, and the
retained benchmark revision sampled only one output from each independent
family. The retained wgpu graphics executable also emits SPIR-V that Vulkan
validation rejects under `VUID-StandaloneSpirv-None-10684`; the shared shader
has been rewritten equivalently and now passes the automated correctness
matrix. The current benchmark hashes every independent output and corrects a
mistyped FNV-1a multiplier that both retained binaries shared; old and new
digest strings are therefore not directly comparable. It also prevents the
graphics-chain readback row from saturating to all `0xff`, which previously
made that hash insensitive to missing late passes. These shader
fixes change graphics binaries: rerun the full timing collection as well as its
small correctness matrix before submission. The current flat CPU profiles
include completion waits and therefore cannot attribute the record-and-submit
gap; the paper now uses them only as a diagnostic. They were also unpinned and
selected different Blade/wgpu GPUs on the dual-GPU AMD host. The retained
capture manifests likewise identify hosts but not selected adapters and omit
three Blade controls. Recollect both artifacts with explicit selectors; the
current profile and capture runners reject a mismatched device, and the capture
runner additionally checks output hashes and one newly created capture per
configuration.

The workload set is closed; see [experiments.md](experiments.md) for what was
considered and declined, and why. What remains is filling the existing grid:

| machine | matrix | profile | captures |
|---|:--:|:--:|:--:|
| zork | yes | yes | yes |
| rubik | yes | yes | yes |
| k6 | missing locally | missing locally | missing locally |
| matrix | yes | yes | yes |
| mac | yes | n/a | n/a |

`python3 paper/collect.py --wgpu ../wgpu` fills a row; the profile step needs a
permissive `kernel.perf_event_paranoid` and says so when it skips. Apple has
neither a `perf` nor a RenderDoc path, so those two cells are not applicable;
its three-repetition, Low-Power-Mode timing collection is still exploratory.

The collection change that materially improved the study has been made: every
2026-07-27 run warms up for 1000 iterations, chosen because the earlier ten
left severe within-block drift (pinning `power_dpm` levels alone did not).
Residual drift still exists and is priced by the per-cell stability diagnostic;
the warm-up count is not proof of a fixed clock. The analysis now also treats
the three process repetitions as the experimental blocks. That correction
widens unstable Intel and RTX compute-chain intervals while leaving the large
discrete-GPU independent-pass effects intact. Current thresholds must be read
from regenerated tables, not copied into this README.

## Current scope

- Vulkan on AMD and NVIDIA is the primary controlled comparison.
- Apple/Metal is a separate investigation because Metal changes where hazard
  tracking occurs and does not expose Vulkan image layouts.
- `wgpu` is the native tracked baseline. Its benchmark deliberately uses the
  trusted-shader path to match Blade's unchecked shader-runtime policy, so it
  is not a browser-WebGPU safety baseline. CPU profiles and command captures
  help describe differences, but the total Blade-versus-wgpu delta is not
  attributed specifically to tracking.
- `wgpu-hal` is an optional diagnostic baseline, not a required second
  implementation.
- Software Vulkan implementations may be used for correctness checks but never
  as performance data.
