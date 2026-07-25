# Blade synchronization paper

This directory contains the working technical report and its experiment
protocol. The draft is intentionally explicit about which claims still require
measurements. It is not ready to cite or upload while `RESULTS PENDING` markers
remain.

## Draft

Every table in the paper is generated from the raw collections, so regenerate
them before building:

```bash
python3 build-tables.py
latexmk -pdf main.tex
```

`build-tables.py` writes `data/derived/tables/*.tex`, which `main.tex` pulls in
with `\input`. It uses only the standard library and re-uses `analyze.py` for
bootstrap intervals. It also prints a warning for any collection whose two
implementations selected different devices.

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

Raw measurements belong under `data/raw/`. Generated summaries, tables, and
figures belong under `data/derived/`. Every result used by the paper must retain
the raw rows and machine metadata from which it was derived.

With Blade and the benchmark wgpu branch checked out as sibling directories,
run the randomized, repeated matched matrix after committing both artifacts:

```bash
python3 paper/run-study-matrix.py \
  --wgpu ../wgpu \
  --backend vulkan \
  --blade-device-id <id> \
  --wgpu-adapter-name "<name>" \
  --output paper/data/raw/<collection-id>
```

Add `--pass-list 1,2,4,8,16,32,64,128` to sweep the pass count instead of
measuring a single point.

See [COLLECTING.md](COLLECTING.md) for adapter discovery, correctness runs,
separate CPU/GPU collections, Linux, Windows, and macOS commands, profiling,
and capture requirements. `run-sync-matrix.sh` remains as a Blade-only pilot
runner; it is not the final matched collector.
`tools/metal-hazard-bench.swift` is the standalone Metal tracked-versus-untracked
harness; its results live in
[`../docs/metal-hazard-tracking.md`](../docs/metal-hazard-tracking.md).

Summarize a collection with deterministic bootstrap intervals:

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
6. Repeat the same frozen workloads with the matched wgpu benchmark, then
   capture and profile one representative run per workload.

`run-study-matrix.py` automates the matched Blade and wgpu collection,
randomizes both implementations together, and rejects mismatched output hashes.

The initial benchmark isolates the cost of Blade's unconditional pass-boundary
barriers from barriers placed only at application-declared hazards. The paper
does not require a second resource tracker inside Blade. Precise tracked
behavior comes from matched wgpu workloads, CPU profiles, and captured Vulkan
commands, and is explicitly reported as an end-to-end comparison.

## Collected so far

`build-tables.py` discovers collections under `data/raw/` and classifies them
by what they contain, so a retest lands in the tables without editing any
script. Collections named `<timestamp>-<hostname>` come from the four-policy
era; later ones cross placement with scope.

| Collection | Machine | Device measured by Blade | Role |
|---|---|---|---|
| `20260725T060107Z-zork` | zork | NVIDIA RTX 5070 | 16-pass matrix |
| `20260725T155439Z-rubik` | rubik | AMD Raphael iGPU | 16-pass matrix (wgpu ran on the RX 7900 XT; cross-implementation cells are void) |
| `20260725T060322Z-k6` | k6 | AMD Radeon 780M | 16-pass matrix |
| `20260725T160725Z-matrix` | matrix | Intel Xe (RPL-U) | 16-pass matrix |
| `20260725T062529Z-mac` | mac | Apple M3 | 16-pass matrix, Metal case study |
| `20260725T185206Z-zork` | zork | NVIDIA RTX 5070 | placement crossed with scope |

Plus two pass-count sweeps on zork, one GPU-timed and one timestamp-free.

Still outstanding before the working-draft banner can come off: Vulkan
command-stream captures and CPU profiles. Also outstanding, independently of
the banner: application workloads, a discrete AMD part running Blade, and the
crossed placement/scope matrix on anything other than NVIDIA.

## Current scope

- Vulkan on AMD and NVIDIA is the primary controlled comparison.
- Apple/Metal is a separate investigation because Metal changes where hazard
  tracking occurs and does not expose Vulkan image layouts.
- `wgpu` is the realistic tracked baseline. CPU profiles and command captures
  help explain differences, but the total Blade-versus-wgpu delta is not
  attributed specifically to tracking.
- `wgpu-hal` is an optional diagnostic baseline, not a required second
  implementation.
- Software Vulkan implementations may be used for correctness checks but never
  as performance data.
