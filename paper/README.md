# Blade synchronization paper

This directory contains the working technical report and its experiment
protocol. The draft is intentionally explicit about which claims still require
measurements. It is not ready to cite or upload while `RESULTS PENDING` markers
remain.

## Draft

Build the paper from this directory with:

```bash
latexmk -pdf main.tex
```

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

See [COLLECTING.md](COLLECTING.md) for adapter discovery, correctness runs,
separate CPU/GPU collections, Linux, Windows, and macOS commands, profiling,
and capture requirements. `run-sync-matrix.sh` remains as a Blade-only pilot
runner; it is not the final matched collector.

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
